// FastEmbedProvider — in-process ONNX embeddings via fastembed-rs
// Default model: jina-embeddings-v2-base-code (768-dim, code-specialized)
//
// Alternative: CodeRankEmbed (nomic-ai/CodeRankEmbed) via the `coderankembed()` constructor.
// ONNX weights served from `sirasagi62/code-rank-embed-onnx` on HuggingFace.
// CoIR nDCG@10 ~60 vs jina-v2-base-code's ~48.

use anyhow::Context;
use async_trait::async_trait;
use fastembed::{
    EmbeddingModel, InitOptionsUserDefined, Pooling, SparseInitOptions, SparseModel,
    SparseTextEmbedding, TextEmbedding, TextInitOptions, TokenizerFiles, UserDefinedEmbeddingModel,
};
use skelesearch_core::{EmbedProvider, SparseEmbedding, SparseEmbedProvider};

/// Configuration for constructing a [`FastEmbedProvider`].
///
/// Use [`Default::default()`] for a sensible starting point, then override
/// individual fields as needed.
pub struct FastEmbedOptions {
    /// Which fastembed model to load.
    pub model: EmbeddingModel,
    /// Request INT8-quantized inference.
    ///
    /// fastembed v5 represents quantized models as distinct [`EmbeddingModel`]
    /// variants (e.g. `BGESmallENV15Q`). `JinaEmbeddingsV2BaseCode` has no
    /// quantized variant; when `quantized` is true for a model without one,
    /// a warning is emitted and the full-precision model is used instead.
    /// TODO: add a model→Q-variant mapping and return an error or switch
    /// to the Q variant automatically once we support models that have one.
    pub quantized: bool,
    /// Show download progress bar during model weight download.
    pub show_download_progress: bool,
}

impl Default for FastEmbedOptions {
    fn default() -> Self {
        Self {
            model: EmbeddingModel::JinaEmbeddingsV2BaseCode,
            quantized: false,
            show_download_progress: true,
        }
    }
}

/// An [`EmbedProvider`] backed by [`fastembed::TextEmbedding`].
///
/// `dim` is cached at construction time so callers don't pay a model-info
/// lookup on every batch.  The underlying `TextEmbedding` is `!Send`, so we
/// wrap it in a `std::sync::Mutex` and offload embedding work to a blocking
/// thread via `tokio::task::spawn_blocking`.
pub struct FastEmbedProvider {
    // Interior mutability through Mutex: TextEmbedding's embed() takes &self
    // but fastembed does not guarantee Send, so we pin it to a blocking thread.
    model: std::sync::Arc<std::sync::Mutex<TextEmbedding>>,
    dim: usize,
    /// Provider name returned by [`EmbedProvider::name()`].
    ///
    /// Stored at construction time so different constructor paths can each
    /// advertise a distinct, stable identity (e.g. `"fastembed"`,
    /// `"fastembed-q"`, `"coderankembed"`).
    name: String,
}

impl FastEmbedProvider {
    /// Construct a `FastEmbedProvider` using the default model
    /// (`jina-embeddings-v2-base-code`, 768-dim).
    ///
    /// Downloads the ONNX weights on first use (~90 MB). Subsequent calls
    /// use the local fastembed cache (`~/.cache/fastembed` by default).
    ///
    /// # Errors
    /// Returns an error if the model cannot be loaded (network unavailable,
    /// disk full, ONNX runtime failure, etc.).
    pub fn default() -> anyhow::Result<Self> {
        Self::with_model(EmbeddingModel::JinaEmbeddingsV2BaseCode)
    }

    /// Construct with an explicit fastembed [`EmbeddingModel`].
    pub fn with_model(model: EmbeddingModel) -> anyhow::Result<Self> {
        Self::with_options(FastEmbedOptions {
            model,
            ..Default::default()
        })
    }

    /// Construct with full control over embedding options.
    ///
    /// When `opts.quantized` is `true` but the chosen model has no Q-variant,
    /// this emits a tracing warning and proceeds with the full-precision model.
    pub fn with_options(opts: FastEmbedOptions) -> anyhow::Result<Self> {
        if opts.quantized {
            // fastembed v5 does not expose a runtime quantization knob on
            // TextInitOptions.  Quantized inference requires selecting the
            // appropriate Q-variant EmbeddingModel (e.g. BGESmallENV15Q).
            // JinaEmbeddingsV2BaseCode has no quantized variant, so we
            // continue with the full-precision weights and warn the caller.
            tracing::warn!(
                model = ?opts.model,
                "quantized=true requested but no Q-variant exists for this model; \
                 using full-precision weights. \
                 Select a model with a Q suffix \
                 (e.g. EmbeddingModel::BGESmallENV15Q) to get INT8 inference."
            );
        }

        let model_info = TextEmbedding::get_model_info(&opts.model)
            .context("failed to retrieve model info")?;
        let dim = model_info.dim;

        let te = TextEmbedding::try_new(
            TextInitOptions::new(opts.model)
                .with_show_download_progress(opts.show_download_progress),
        )
        .context("failed to initialize TextEmbedding model")?;

        let name = if opts.quantized { "fastembed-q" } else { "fastembed" }.to_owned();
        Ok(Self {
            model: std::sync::Arc::new(std::sync::Mutex::new(te)),
            dim,
            name,
        })
    }

    // ---------------------------------------------------------------------------
    // User-defined (BYOM) constructors
    // ---------------------------------------------------------------------------

    /// Construct a [`FastEmbedProvider`] from an arbitrary ONNX model hosted on
    /// HuggingFace.
    ///
    /// Model files (`model.onnx`, `tokenizer.json`, `config.json`,
    /// `special_tokens_map.json`, `tokenizer_config.json`) are downloaded to the
    /// fastembed cache on first call, then served from disk.
    ///
    /// The provider name will be set to `repo` (e.g.
    /// `"sirasagi62/code-rank-embed-onnx"`). Use the specialised constructors
    /// (e.g. [`Self::coderankembed()`]) for canonical short names.
    ///
    /// # Parameters
    /// - `repo`  — HuggingFace repo id, e.g. `"sirasagi62/code-rank-embed-onnx"`.
    /// - `dim`   — Output embedding dimension of the model. **Must** match the
    ///   model's actual output; a mismatch causes downstream vector-space errors
    ///   that are silent and hard to debug. There is no runtime check.
    ///
    /// # Pooling
    /// Assumes mean pooling over token embeddings (last-hidden-state output).
    /// If the ONNX graph already performs pooling internally, set `.with_pooling`
    /// to `None` by loading the model manually.
    ///
    /// # Errors
    /// Network unavailable, disk full, missing ONNX file, ONNX parse failure, etc.
    pub fn with_user_defined(repo: &str, dim: usize) -> anyhow::Result<Self> {
        Self::with_user_defined_impl(repo, dim, repo, 8192)
    }

    /// Construct a [`FastEmbedProvider`] backed by
    /// [`nomic-ai/CodeRankEmbed`](https://huggingface.co/nomic-ai/CodeRankEmbed)
    /// via its ONNX export at `sirasagi62/code-rank-embed-onnx`.
    ///
    /// **Performance**: CoIR nDCG@10 ≈ 60 vs jina-v2-base-code's ≈ 48.
    /// **Dimensions**: 768 (matches jina default; manifests from a jina index are
    /// NOT interchangeable — different embedding spaces despite equal dim).
    /// **Context**: 8 192 tokens (nomic-bert long-context backbone).
    /// **Query prefix**: prepend `"Represent this query for searching relevant code: "`
    ///   to query strings for best retrieval quality.
    ///
    /// Downloads ~548 MB of model weights on first call; uses fastembed cache
    /// (`$FASTEMBED_CACHE_DIR` / `$HF_HOME` / `.fastembed_cache`) thereafter.
    ///
    /// # Errors
    /// Network unavailable, disk full, ONNX parse failure, etc.
    pub fn coderankembed() -> anyhow::Result<Self> {
        Self::with_user_defined_impl(
            "sirasagi62/code-rank-embed-onnx",
            768,
            "coderankembed",
            8192,
        )
    }

    /// Construct a [`FastEmbedProvider`] backed by
    /// [`Alibaba-NLP/gte-modernbert-base`](https://huggingface.co/Alibaba-NLP/gte-modernbert-base).
    ///
    /// **Performance**: CoIR nDCG@10 = 79.31, MTEB Code = 71.66 — substantially
    /// better than jina-v2-base-code (~48) and CodeRankEmbed (~60).
    /// **Dimensions**: 768 (same as jina/CodeRankEmbed).
    /// **Context**: 8 192 tokens. **Pooling**: CLS (not mean).
    /// **Size**: ~280 MB ONNX weights.
    ///
    /// Downloads model from HuggingFace on first call; uses fastembed cache
    /// (`$FASTEMBED_CACHE_DIR` / `$HF_HOME` / `~/.cache/fastembed`) thereafter.
    pub fn gte_modernbert() -> anyhow::Result<Self> {
        let repo = "Alibaba-NLP/gte-modernbert-base";
        let dim = 768;
        let provider_name = "gte-modernbert";
        let max_length = 8192;

        let model_repo = pull_from_hf(repo, true)
            .with_context(|| format!("failed to access HuggingFace repo '{repo}'"))?;

        // GTE-ModernBERT stores ONNX in onnx/ subdirectory.
        let onnx_bytes = std::fs::read(
            model_repo
                .get("onnx/model.onnx")
                .with_context(|| format!("failed to download onnx/model.onnx from '{repo}'"))?,
        )
        .context("failed to read onnx/model.onnx")?;

        let tokenizer_files = TokenizerFiles {
            tokenizer_file: std::fs::read(
                model_repo.get("tokenizer.json").context("failed to download tokenizer.json")?,
            ).context("failed to read tokenizer.json")?,
            config_file: std::fs::read(
                model_repo.get("config.json").context("failed to download config.json")?,
            ).context("failed to read config.json")?,
            special_tokens_map_file: std::fs::read(
                model_repo.get("special_tokens_map.json").context("failed to download special_tokens_map.json")?,
            ).context("failed to read special_tokens_map.json")?,
            tokenizer_config_file: std::fs::read(
                model_repo.get("tokenizer_config.json").context("failed to download tokenizer_config.json")?,
            ).context("failed to read tokenizer_config.json")?,
        };

        // GTE-ModernBERT uses CLS pooling: outputs.last_hidden_state[:, 0].
        let user_model = UserDefinedEmbeddingModel::new(onnx_bytes, tokenizer_files)
            .with_pooling(Pooling::Cls);

        let mut te = TextEmbedding::try_new_from_user_defined(
            user_model,
            InitOptionsUserDefined::new().with_max_length(max_length),
        )
        .with_context(|| format!("failed to initialise TextEmbedding from '{repo}'"))?;

        // Verify dimension.
        {
            let probe = te.embed(vec!["dim probe".to_string()], None).context("dimension probe failed")?;
            if let Some(first) = probe.first() {
                anyhow::ensure!(first.len() == dim, "declared dim={dim} but model produced {}-dim vectors", first.len());
            }
        }

        Ok(Self {
            model: std::sync::Arc::new(std::sync::Mutex::new(te)),
            dim,
            name: provider_name.to_owned(),
        })
    }

    /// Internal helper shared by [`with_user_defined`] and [`coderankembed`].
    ///
    /// Downloads the model repo from HuggingFace (reusing the fastembed cache
    /// dir convention), assembles `TokenizerFiles`, and initialises a
    /// `TextEmbedding` with mean pooling.
    ///
    /// # Parameters
    /// - `repo`         — HuggingFace repo id.
    /// - `dim`          — Declared output dimension; trusted, not validated.
    /// - `provider_name`— Name stored in the provider, returned by `name()`.
    /// - `max_length`   — Max token sequence length passed to the tokenizer.
    ///   The tokenizer's own `model_max_length` may further cap this.
    fn with_user_defined_impl(
        repo: &str,
        dim: usize,
        provider_name: &str,
        max_length: usize,
    ) -> anyhow::Result<Self> {
        let model_repo = pull_from_hf(repo, true)
            .with_context(|| format!("failed to access HuggingFace repo '{repo}'"))?;

        // Download (or serve from cache) all required model files.
        let onnx_bytes = std::fs::read(
            model_repo
                .get("model.onnx")
                .with_context(|| format!("failed to download model.onnx from '{repo}'"))?,
        )
        .context("failed to read model.onnx")?;

        let tokenizer_files = TokenizerFiles {
            tokenizer_file: std::fs::read(
                model_repo
                    .get("tokenizer.json")
                    .context("failed to download tokenizer.json")?,
            )
            .context("failed to read tokenizer.json")?,
            config_file: std::fs::read(
                model_repo
                    .get("config.json")
                    .context("failed to download config.json")?,
            )
            .context("failed to read config.json")?,
            special_tokens_map_file: std::fs::read(
                model_repo
                    .get("special_tokens_map.json")
                    .context("failed to download special_tokens_map.json")?,
            )
            .context("failed to read special_tokens_map.json")?,
            tokenizer_config_file: std::fs::read(
                model_repo
                    .get("tokenizer_config.json")
                    .context("failed to download tokenizer_config.json")?,
            )
            .context("failed to read tokenizer_config.json")?,
        };

        // nomic-bert / arctic-embed architecture outputs last-hidden-state
        // token embeddings that require mean pooling to produce sentence vectors.
        // If the ONNX graph already includes pooling, double-pooling will silently
        // produce wrong embeddings — verify once with the canonical Python pipeline
        // if the output distribution looks off.
        let user_model = UserDefinedEmbeddingModel::new(onnx_bytes, tokenizer_files)
            .with_pooling(Pooling::Mean);

        let mut te = TextEmbedding::try_new_from_user_defined(
            user_model,
            InitOptionsUserDefined::new().with_max_length(max_length),
        )
        .with_context(|| format!("failed to initialise TextEmbedding from '{repo}'"))?;

        // Verify that the model actually produces vectors of the declared
        // dimension. A mismatch is silent and corrupts every search result.
        {
            let probe = te
                .embed(vec!["dim probe".to_string()], None)
                .context("dimension probe failed")?;
            if let Some(first) = probe.first() {
                anyhow::ensure!(
                    first.len() == dim,
                    "declared dim={dim} but model produced {}-dim vectors",
                    first.len()
                );
            }
        }

        Ok(Self {
            model: std::sync::Arc::new(std::sync::Mutex::new(te)),
            dim,
            name: provider_name.to_owned(),
        })
    }
}

#[async_trait]
impl EmbedProvider for FastEmbedProvider {
    fn dim(&self) -> usize {
        self.dim
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn query_prefix(&self) -> Option<&str> {
        // CodeRankEmbed uses instruction-style embeddings for retrieval.
        // Without this prefix the query embedding lands in the wrong space.
        if self.name == "coderankembed" {
            Some("Represent this query for searching relevant code: ")
        } else {
            None
        }
    }

    #[tracing::instrument(skip_all, fields(batch_size = texts.len()))]
    async fn embed_batch(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let model = std::sync::Arc::clone(&self.model);
        // Run on a blocking thread — ONNX inference is CPU-bound and
        // TextEmbedding is not designed for async contexts.
        tokio::task::spawn_blocking(move || {
            let mut locked = model
                .lock()
                .map_err(|_| anyhow::anyhow!("TextEmbedding mutex poisoned"))?;

            let expected = texts.len();
            let embeddings = locked
                .embed(texts, None)
                .context("TextEmbedding::embed failed")?;

            // Guard against the provider silently dropping or reordering outputs.
            anyhow::ensure!(
                embeddings.len() == expected,
                "expected {} embeddings, got {}",
                expected,
                embeddings.len()
            );

            Ok(embeddings)
        })
        .await
        .context("embedding task panicked")?
    }
}

// ---------------------------------------------------------------------------
// Sparse provider
// ---------------------------------------------------------------------------

/// A [`SparseEmbedProvider`] backed by [`fastembed::SparseTextEmbedding`].
///
/// Default model: BGE-M3 sparse (8192-token context, 100+ languages).
/// Downloads ONNX weights on first use (~50 MB). Subsequent calls use the
/// local fastembed cache (`$FASTEMBED_CACHE_DIR` / `~/.fastembed_cache`).
///
/// `SparseTextEmbedding::embed` takes `&mut self`, so we wrap it in a
/// `std::sync::Mutex` for interior mutability across shared references.
pub struct FastEmbedSparseProvider {
    model: std::sync::Arc<std::sync::Mutex<SparseTextEmbedding>>,
}

impl FastEmbedSparseProvider {
    /// Construct a [`FastEmbedSparseProvider`] using the BGE-M3 sparse model.
    ///
    /// # Errors
    /// Returns an error if the model cannot be loaded (network unavailable,
    /// disk full, ONNX runtime failure, etc.).
    pub fn bgem3() -> anyhow::Result<Self> {
        let model = SparseTextEmbedding::try_new(SparseInitOptions::new(SparseModel::BGEM3))
            .context("failed to initialize BGE-M3 sparse model")?;
        Ok(Self { model: std::sync::Arc::new(std::sync::Mutex::new(model)) })
    }
}

impl SparseEmbedProvider for FastEmbedSparseProvider {
    fn name(&self) -> &str {
        "bgem3-sparse"
    }

    fn embed_batch(&self, texts: &[&str]) -> anyhow::Result<Vec<SparseEmbedding>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let mut locked = self
            .model
            .lock()
            .map_err(|_| anyhow::anyhow!("SparseTextEmbedding mutex poisoned"))?;
        let results = locked
            .embed(texts.to_vec(), None)
            .context("SparseTextEmbedding::embed failed")?;
        // fastembed uses `usize` for indices; we normalise to `u32` because
        // vocabulary sizes for code models fit comfortably in 32 bits and
        // u32 halves the memory footprint of the in-memory inverted index.
        Ok(results
            .into_iter()
            .map(|r| SparseEmbedding {
                indices: r.indices.into_iter().map(|i| i as u32).collect(),
                values: r.values,
            })
            .collect())
    }
}

// ---------------------------------------------------------------------------
// HuggingFace download helper (sync)
// ---------------------------------------------------------------------------

/// Download (or serve from cache) a HuggingFace model repo.
///
/// Cache location precedence (mirrors fastembed's own convention):
/// 1. `$HF_HOME`
/// 2. `$FASTEMBED_CACHE_DIR`
/// 3. `~/.cache/fastembed` (stable cross-CWD fallback)
///
/// The HF API endpoint can be overridden via `$HF_ENDPOINT`.
fn pull_from_hf(
    repo: &str,
    show_progress: bool,
) -> anyhow::Result<hf_hub::api::sync::ApiRepo> {
    use std::path::PathBuf;

    let cache_dir = std::env::var("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            std::env::var("FASTEMBED_CACHE_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|_| {
                    // Stable cross-CWD fallback: ~/.cache/fastembed
                    std::env::var("HOME")
                        .map(std::path::PathBuf::from)
                        .unwrap_or_else(|_| PathBuf::from("."))
                        .join(".cache")
                        .join("fastembed")
                })
        });

    let endpoint = std::env::var("HF_ENDPOINT")
        .unwrap_or_else(|_| "https://huggingface.co".to_string());

    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_cache_dir(cache_dir)
        .with_endpoint(endpoint)
        .with_progress(show_progress)
        .build()
        .context("failed to build HuggingFace API client")?;

    Ok(api.model(repo.to_string()))
}


// ---------------------------------------------------------------------------
// Provider factory
// ---------------------------------------------------------------------------

/// Build an [`EmbedProvider`] by name, returning it boxed as a trait object.
///
/// Supported names:
/// - `"fastembed"` — jina-embeddings-v2-base-code (768-dim, ~90MB, fast, default)
/// - `"gte-modernbert"` — GTE-ModernBERT-base (768-dim, CoIR 79.31, ~280MB ONNX / ~8GB RAM)
/// - `"jina"` / `"fastembed-legacy"` — alias for fastembed
/// - `"fastembed-q"` / `"fastembed-int8"` — quantized jina variant
/// - `"coderankembed"` — nomic-ai/CodeRankEmbed via ONNX (768-dim, CoIR ≈ 60)
/// - `"openai"` — when the `openai` feature is enabled
/// - `"voyage"` — when the `voyage` feature is enabled
///
/// Unknown names are rejected immediately with a clear error.
///
/// # Errors
/// Returns an error if the name is unrecognized or if the underlying
/// provider fails to initialize.
pub fn provider_from_name(name: &str) -> anyhow::Result<Box<dyn skelesearch_core::EmbedProvider>> {
    match name {
        "fastembed" => {
            let p = FastEmbedProvider::default()
                .map_err(|e| e.context("failed to initialise fastembed (jina) provider"))?;
            Ok(Box::new(p))
        }
        "gte-modernbert" => {
            let p = FastEmbedProvider::gte_modernbert()
                .map_err(|e| e.context("failed to initialise gte-modernbert provider"))?;
            Ok(Box::new(p))
        }
        "jina" | "fastembed-legacy" => {
            let p = FastEmbedProvider::default()
                .map_err(|e| e.context("failed to initialise fastembed (jina) provider"))?;
            Ok(Box::new(p))
        }
        "fastembed-q" | "fastembed-int8" => {
            let p = FastEmbedProvider::with_options(FastEmbedOptions {
                quantized: true,
                ..Default::default()
            })
            .map_err(|e| e.context("failed to initialise fastembed-q provider"))?;
            Ok(Box::new(p))
        }
        "coderankembed" => {
            let p = FastEmbedProvider::coderankembed()
                .map_err(|e| e.context("failed to initialise coderankembed provider"))?;
            Ok(Box::new(p))
        }
        #[cfg(feature = "openai")]
        "openai" => Ok(Box::new(skelesearch_embed_openai::OpenAIProvider::new()?)),
        #[cfg(feature = "voyage")]
        "voyage" => Ok(Box::new(skelesearch_embed_voyage::provider_voyage()?)),
        other => {
            let mut supported = vec!["fastembed", "gte-modernbert", "fastembed-legacy", "jina", "fastembed-q", "fastembed-int8", "coderankembed"];
            #[cfg(feature = "openai")] supported.push("openai");
            #[cfg(feature = "voyage")] supported.push("voyage");
            anyhow::bail!("unknown embedding provider: '{}'. Supported: {}", other, supported.join(", "))
        }
    }
}
