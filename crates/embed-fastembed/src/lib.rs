// FastEmbedProvider — in-process ONNX embeddings via fastembed-rs
// Default model: jina-embeddings-v2-base-code (768-dim, code-specialized)

use anyhow::Context;
use async_trait::async_trait;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use skelesearch_core::EmbedProvider;

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
        let model_info = TextEmbedding::get_model_info(&model)
            .context("failed to retrieve model info")?;
        let dim = model_info.dim;

        let te = TextEmbedding::try_new(
            InitOptions::new(model).with_show_download_progress(true),
        )
        .context("failed to initialize TextEmbedding model")?;

        Ok(Self {
            model: std::sync::Arc::new(std::sync::Mutex::new(te)),
            dim,
        })
    }
}

#[async_trait]
impl EmbedProvider for FastEmbedProvider {
    fn dim(&self) -> usize {
        self.dim
    }

    fn name(&self) -> &str {
        "fastembed"
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
            let locked = model
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
// Provider factory
// ---------------------------------------------------------------------------

/// Build an [`EmbedProvider`] by name, returning it boxed as a trait object.
///
/// Supported names: `"fastembed"`, `"openai"` (when the `openai` feature is enabled).
/// Unknown names are rejected immediately with a clear error.
///
/// # Errors
/// Returns an error if the name is unrecognized or if the underlying
/// provider fails to initialize.
pub fn provider_from_name(name: &str) -> anyhow::Result<Box<dyn skelesearch_core::EmbedProvider>> {
    match name {
        "fastembed" => {
            let p = FastEmbedProvider::default()
                .map_err(|e| e.context("failed to initialise fastembed provider"))?;
            Ok(Box::new(p))
        }
        #[cfg(feature = "openai")]
        "openai" => Ok(Box::new(skelesearch_embed_openai::OpenAIProvider::new()?)),
        #[cfg(feature = "voyage")]
        "voyage" => Ok(Box::new(skelesearch_embed_voyage::provider_voyage()?)),
        other => {
            let mut supported = vec!["fastembed"];
            #[cfg(feature = "openai")] supported.push("openai");
            #[cfg(feature = "voyage")] supported.push("voyage");
            anyhow::bail!("unknown embedding provider: '{}'. Supported: {}", other, supported.join(", "))
        }
    }
}