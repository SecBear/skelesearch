//! Local ONNX cross-encoder reranker for skelesearch.
//!
//! Cross-encoder models score (query, document) pairs jointly, providing
//! substantially higher precision than bi-encoder retrieval alone at the
//! cost of O(N) inference calls — batched here for efficiency.
//!
//! # Model files
//!
//! This crate does **not** bundle model weights. Export a HuggingFace cross-encoder
//! to ONNX with [`optimum`](https://huggingface.co/docs/optimum):
//!
//! ```sh
//! pip install optimum[onnxruntime]
//! optimum-cli export onnx \
//!   --model cross-encoder/ms-marco-MiniLM-L-6-v2 \
//!   --task text-classification \
//!   ./models/ms-marco-MiniLM-L-6-v2/
//! ```
//!
//! Recommended models (passage reranking, ONNX-exportable):
//! - `cross-encoder/ms-marco-MiniLM-L-6-v2` — fast, strong MS-MARCO baseline
//! - `BAAI/bge-reranker-base` — multilingual, instruction-following
//! - `cross-encoder/ms-marco-electra-base` — highest quality, heavier

use std::{path::Path, sync::Arc};

use anyhow::Context;
use async_trait::async_trait;
use ndarray::Array2;
use ort::session::Session;
use skelesearch_core::reranker::{RerankCandidate, Reranker};
use tokenizers::{
    EncodeInput, PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationDirection,
    TruncationParams, TruncationStrategy,
};
use tracing::{debug, instrument};

/// Canonical HuggingFace repo for the default local reranker model.
///
/// ms-marco-MiniLM-L-6-v2 (cross-encoder): 22M params, 512-token context.
/// Fast on CPU (~50-100ms for 10 candidates). Not code-specific but effective
/// for reranking code search results where chunks are <400 tokens.
///
/// For GPU users wanting higher quality, see `GTE_MODEL_REPO` below.
const DEFAULT_MODEL_REPO: &str = "cross-encoder/ms-marco-MiniLM-L6-v2";

/// Cache directory name used for the default model.
const DEFAULT_MODEL_CACHE_NAME: &str = "ms-marco-MiniLM-L6-v2";

/// HuggingFace repo for the high-quality GPU reranker.
///
/// gte-reranker-modernbert-base (Alibaba-NLP): 149M params, 8192-token context.
/// CoIR avg 79.99, CodeSearchNet-Python 98.37. Apache-2.0.
/// Requires GPU (CUDA) for acceptable latency; ~4s/query on CPU.
#[allow(dead_code)]
const GTE_MODEL_REPO: &str = "Alibaba-NLP/gte-reranker-modernbert-base";

/// Default max token sequence length.
///
/// MiniLM supports 512 tokens. Our chunks are ~375 tokens after tokenization,
/// so 512 is sufficient. Configurable via `with_max_seq_len()` for models
/// with longer context (e.g. gte-modernbert-base supports 8192).
const DEFAULT_MAX_SEQ_LEN: usize = 512;

/// Default sub-batch size for ONNX forward passes.
///
/// Large batches create huge [N, seq_len] tensors that thrash CPU cache.
/// 64 keeps each forward pass fast while amortizing tokenization overhead.
const DEFAULT_SUB_BATCH: usize = 64;

/// Local ONNX cross-encoder reranker.
///
/// Loads a cross-encoder model from a local directory containing:
/// - `model.onnx` — ONNX export of the cross-encoder
/// - `tokenizer.json` — HuggingFace tokenizer configuration
///
/// The model receives `(query, document)` pairs and returns relevance scores.
/// Candidates are split into sub-batches for inference efficiency.
///
/// # Architecture notes
///
/// - BERT-based models (e.g. MiniLM) expect `input_ids`, `attention_mask`,
///   and `token_type_ids`. RoBERTa-based models omit `token_type_ids`. The
///   struct auto-detects this from the model's ONNX input metadata.
/// - Models with a single-logit output (shape `[B, 1]` or `[B]`) return the
///   raw logit as the relevance score. Binary classifiers (`[B, 2]`) return
///   the softmax probability of the positive class.
/// - Inference is synchronous (ort is CPU-bound); `rerank` dispatches to a
///   `tokio` blocking thread pool to avoid stalling the async runtime.
pub struct LocalReranker {
    /// Wrapped in Mutex because ort rc.11+ requires &mut self for session.run().
    session: Arc<std::sync::Mutex<Session>>,
    tokenizer: Arc<Tokenizer>,
    /// True when the ONNX model's input list includes `token_type_ids`.
    has_token_type_ids: bool,
    /// Max token length per (query, candidate) pair. Longer inputs are truncated.
    max_seq_len: usize,
    /// Max candidates per ONNX forward pass.
    sub_batch_size: usize,
}

impl LocalReranker {
    /// Load a cross-encoder from a local directory.
    ///
    /// The directory must contain `model.onnx` and `tokenizer.json`.
    /// Returns an error with the full path if either file is missing.
    pub fn new(model_dir: impl AsRef<Path>) -> anyhow::Result<Self> {
        let dir = model_dir.as_ref();

        let model_path = dir.join("model.onnx");
        let tokenizer_path = dir.join("tokenizer.json");

        anyhow::ensure!(
            model_path.exists(),
            "ONNX model not found: {}",
            model_path.display()
        );
        anyhow::ensure!(
            tokenizer_path.exists(),
            "tokenizer.json not found: {}",
            tokenizer_path.display()
        );

        #[allow(unused_mut)]
        let mut builder = Session::builder()?;

        // Register hardware-accelerated execution providers when feature-enabled.
        // CoreML targets Apple Neural Engine / GPU; CUDA targets NVIDIA GPUs.
        // If registration fails (e.g. no compatible hardware), fall back to CPU.
        #[cfg(feature = "coreml")]
        {
            use ort::execution_providers::CoreMLExecutionProvider;
            builder = builder.with_execution_providers([
                CoreMLExecutionProvider::default().build()
            ])?;
            tracing::info!("CoreML execution provider registered");
        }
        #[cfg(feature = "cuda")]
        {
            use ort::execution_providers::CUDAExecutionProvider;
            builder = builder.with_execution_providers([
                CUDAExecutionProvider::default().build()
            ])?;
            tracing::info!("CUDA execution provider registered");
        }

        let session = builder
            .commit_from_file(&model_path)
            .with_context(|| format!("failed to load ONNX model: {}", model_path.display()))?;

        // Detect BERT vs RoBERTa: only BERT-family models expect token_type_ids.
        let has_token_type_ids = session
            .inputs()
            .iter()
            .any(|inp| inp.name() == "token_type_ids");

        let mut tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            anyhow::anyhow!(
                "failed to load tokenizer from {}: {e}",
                tokenizer_path.display()
            )
        })?;

        // Configure truncation (once, at load time):
        //   LongestFirst truncates the document before the query, preserving
        //   query tokens as much as possible — important for relevance.
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: DEFAULT_MAX_SEQ_LEN,
                strategy: TruncationStrategy::LongestFirst,
                stride: 0,
                direction: TruncationDirection::Right,
            }))
            .map_err(|e| anyhow::anyhow!("failed to configure tokenizer truncation: {e}"))?;

        // Pad all sequences in a batch to the longest sequence in that batch.
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            direction: PaddingDirection::Right,
            pad_to_multiple_of: None,
            pad_id: 0,
            pad_type_id: 0,
            pad_token: "[PAD]".to_string(),
        }));

        Ok(Self {
            session: Arc::new(std::sync::Mutex::new(session)),
            tokenizer: Arc::new(tokenizer),
            has_token_type_ids,
            max_seq_len: DEFAULT_MAX_SEQ_LEN,
            sub_batch_size: DEFAULT_SUB_BATCH,
        })
    }

    /// Set the maximum token sequence length per (query, candidate) pair.
    /// Default: 1024. gte-modernbert-base supports up to 8192.
    pub fn with_max_seq_len(mut self, len: usize) -> Self {
        self.max_seq_len = len;
        // Re-apply truncation with the new length.
        Arc::get_mut(&mut self.tokenizer)
            .expect("tokenizer Arc not unique during configuration")
            .with_truncation(Some(TruncationParams {
                max_length: len,
                strategy: TruncationStrategy::LongestFirst,
                stride: 0,
                direction: TruncationDirection::Right,
            }))
            .expect("truncation configuration failed");
        self
    }

    /// Set the sub-batch size for ONNX inference. Default: 64.
    pub fn with_sub_batch_size(mut self, size: usize) -> Self {
        self.sub_batch_size = size;
        self
    }

    /// Load the default reranker model (ms-marco-MiniLM-L6-v2).
    ///
    /// Fast CPU cross-encoder (22M params, ~50-100ms for 10 candidates).
    /// Looks for model files in `~/.cache/skelesearch/reranker/ms-marco-MiniLM-L6-v2/`.
    ///
    /// # Download
    ///
    /// ```sh
    /// mkdir -p ~/.cache/skelesearch/reranker/ms-marco-MiniLM-L6-v2
    /// uv tool run --from huggingface_hub hf download \
    ///     cross-encoder/ms-marco-MiniLM-L6-v2 \
    ///     onnx/model.onnx tokenizer.json \
    ///     --local-dir ~/.cache/skelesearch/reranker/ms-marco-MiniLM-L6-v2
    /// mv ~/.cache/skelesearch/reranker/ms-marco-MiniLM-L6-v2/onnx/model.onnx \
    ///     ~/.cache/skelesearch/reranker/ms-marco-MiniLM-L6-v2/model.onnx
    /// ```
    ///
    /// For higher quality (requires NVIDIA GPU), use gte-reranker-modernbert-base:
    /// ```sh
    /// # Same pattern but with Alibaba-NLP/gte-reranker-modernbert-base
    /// # Then: LocalReranker::new(path).with_max_seq_len(8192)
    /// ```
    ///
    /// The ONNX file (`onnx/model.onnx` from the HF repo) must be placed as
    /// `model.onnx` in the cache directory. `tokenizer.json` is placed as-is.
    pub fn default_model() -> anyhow::Result<Self> {
        let cache = dirs::cache_dir()
            .unwrap_or_else(|| std::path::PathBuf::from(".cache"))
            .join("skelesearch")
            .join("reranker")
            .join(DEFAULT_MODEL_CACHE_NAME);

        if !cache.join("model.onnx").exists() {
            anyhow::bail!(
                concat!(
                    "Default reranker model not found at {path}.\n\n",
                    "Download with:\n",
                    "  mkdir -p {path}\n",
                    "  uv tool run --from huggingface_hub hf download \\\n",
                    "      {repo} \\\n",
                    "      onnx/model.onnx tokenizer.json \\\n",
                    "      --local-dir {path}\n\n",
                    "Then: mv {path}/onnx/model.onnx {path}/model.onnx",
                ),
                path = cache.display(),
                repo = DEFAULT_MODEL_REPO,
            );
        }

        Self::new(&cache)
    }
}

/// Synchronous batch inference. Called from a `spawn_blocking` context.
///
/// Builds padded tensors for all `(query, candidate)` pairs, runs the ONNX
/// session, and converts the raw logits to `f64` relevance scores.
fn run_inference(
    session: &mut Session,
    tokenizer: &Tokenizer,
    query: &str,
    candidates: &[RerankCandidate],
    has_token_type_ids: bool,
) -> anyhow::Result<Vec<f64>> {
    let batch_size = candidates.len();

    // Build sentence-pair inputs for the tokenizer.
    let pairs: Vec<EncodeInput> = candidates
        .iter()
        .map(|c| EncodeInput::Dual(query.into(), c.text.as_str().into()))
        .collect();

    // encode_batch pads all sequences to BatchLongest and truncates at MAX_SEQ_LEN.
    let encodings = tokenizer
        .encode_batch(pairs, /* add_special_tokens */ true)
        .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;

    // All sequences in the batch have the same length after padding.
    let seq_len = encodings
        .first()
        .map(|e| e.get_ids().len())
        .unwrap_or(0);

    debug!(batch_size, seq_len, "batch encoded");

    let mut input_ids = Array2::<i64>::zeros((batch_size, seq_len));
    let mut attention_mask = Array2::<i64>::zeros((batch_size, seq_len));
    let mut token_type_ids = Array2::<i64>::zeros((batch_size, seq_len));

    for (i, enc) in encodings.iter().enumerate() {
        for (j, &id) in enc.get_ids().iter().enumerate() {
            input_ids[[i, j]] = id as i64;
        }
        for (j, &mask) in enc.get_attention_mask().iter().enumerate() {
            attention_mask[[i, j]] = mask as i64;
        }
        for (j, &type_id) in enc.get_type_ids().iter().enumerate() {
            token_type_ids[[i, j]] = type_id as i64;
        }
    }

    // Collect output names before running inference (session.run borrows &mut self,
    // so we can't access session.outputs() inside the error handler below).
    let output_names: Vec<String> = session.outputs().iter().map(|o| o.name().to_string()).collect();

    // Run ONNX inference. Include token_type_ids only when the model expects it
    // — passing an unexpected input causes an ort runtime error.
    let outputs = if has_token_type_ids {
        session.run(ort::inputs![
            "input_ids" => ort::value::TensorRef::from_array_view(input_ids.view())?,
            "attention_mask" => ort::value::TensorRef::from_array_view(attention_mask.view())?,
            "token_type_ids" => ort::value::TensorRef::from_array_view(token_type_ids.view())?
        ])?
    } else {
        session.run(ort::inputs![
            "input_ids" => ort::value::TensorRef::from_array_view(input_ids.view())?,
            "attention_mask" => ort::value::TensorRef::from_array_view(attention_mask.view())?
        ])?
    };
    // Most HuggingFace optimum exports name the output tensor "logits".
    let logits_val = outputs
        .get("logits")
        .with_context(|| {
            format!(
                "model output 'logits' not found; available outputs: {:?}",
                output_names
            )
        })?;
    let logits = logits_val.try_extract_array::<f32>()?;
    let shape = logits.shape();

    // Determine score extraction strategy from the output tensor shape.
    //
    // [B, 2] — binary classifier (neg class, pos class); return P(relevant)
    //          via numerically stable 2-class softmax.
    // [B, 1] — single-logit reranker; use the raw logit directly.
    // [B]    — flat output (some models collapse the last dim); use as-is.
    let scores: Vec<f64> = match shape {
        [_, 2] => (0..batch_size)
            .map(|i| {
                let neg = logits[ndarray::IxDyn(&[i, 0])];
                let pos = logits[ndarray::IxDyn(&[i, 1])];
                // Numerically stable softmax: subtract max before exp.
                let m = neg.max(pos);
                let e_neg = (neg - m).exp();
                let e_pos = (pos - m).exp();
                (e_pos / (e_neg + e_pos)) as f64
            })
            .collect(),

        [_, 1] => (0..batch_size)
            .map(|i| logits[ndarray::IxDyn(&[i, 0])] as f64)
            .collect(),

        [_] => logits.iter().map(|&s| s as f64).collect(),

        _ => anyhow::bail!(
            "unexpected logits shape {:?}; expected [B], [B, 1], or [B, 2]",
            shape
        ),
    };

    debug!(batch_size, "inference complete");
    Ok(scores)
}

#[async_trait]
impl Reranker for LocalReranker {
    #[instrument(skip_all, fields(candidates = candidates.len()))]
    async fn rerank(
        &self,
        query: &str,
        candidates: Vec<RerankCandidate>,
    ) -> anyhow::Result<Vec<f64>> {
        if candidates.is_empty() {
            return Ok(vec![]);
        }

        let session = Arc::clone(&self.session);
        let tokenizer = Arc::clone(&self.tokenizer);
        let query = query.to_string();
        let has_token_type_ids = self.has_token_type_ids;
        let sub_batch_size = self.sub_batch_size;
        let total_candidates = candidates.len();

        // ort inference is synchronous and CPU-bound; move it off the async executor.
        let start = std::time::Instant::now();
        let scores = tokio::task::spawn_blocking(move || {
            let mut session_guard = session.lock()
                .map_err(|_| anyhow::anyhow!("session mutex poisoned"))?;

            if total_candidates <= sub_batch_size {
                // Small enough for a single pass.
                return run_inference(
                    &mut session_guard, &tokenizer, &query, &candidates, has_token_type_ids,
                );
            }

            // Sub-batch: split candidates into chunks to keep tensors small.
            let mut all_scores = Vec::with_capacity(total_candidates);
            for chunk in candidates.chunks(sub_batch_size) {
                let batch_scores = run_inference(
                    &mut session_guard, &tokenizer, &query, chunk, has_token_type_ids,
                )?;
                all_scores.extend(batch_scores);
            }
            Ok(all_scores)
        })
        .await
        .context("reranker inference thread panicked")??;

        let elapsed_ms = start.elapsed().as_millis() as u64;
        tracing::info!(
            candidates = total_candidates,
            sub_batches = total_candidates.div_ceil(sub_batch_size),
            elapsed_ms,
            "local rerank complete"
        );

        Ok(scores)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use skelesearch_core::reranker::RerankCandidate;

    fn make_candidates(texts: &[&str]) -> Vec<RerankCandidate> {
        texts
            .iter()
            .enumerate()
            .map(|(i, &t)| RerankCandidate {
                index: i,
                text: t.to_string(),
            })
            .collect()
    }

    #[test]
    fn new_rejects_missing_directory() {
        let result = LocalReranker::new("/nonexistent/path/to/model");
        assert!(result.is_err());
        let msg = result.err().expect("expected error").to_string();
        assert!(
            msg.contains("model.onnx not found") || msg.contains("ONNX model not found"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn new_rejects_directory_missing_tokenizer() {
        // Create a temp dir with only model.onnx (empty placeholder).
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model.onnx"), b"").unwrap();
        // tokenizer.json is absent.
        let result = LocalReranker::new(dir.path());
        assert!(result.is_err());
        let msg = result.err().expect("expected error").to_string();
        assert!(msg.contains("tokenizer.json not found"), "unexpected: {msg}");
    }

    #[tokio::test]
    async fn rerank_empty_candidates_returns_empty() {
        // We cannot construct a real LocalReranker in a unit test (no model
        // files), but we can test the Reranker trait contract via the NoopReranker
        // to ensure the calling convention is correct.
        use skelesearch_core::reranker::NoopReranker;
        let scores = NoopReranker
            .rerank("query", vec![])
            .await
            .expect("noop should not fail");
        assert!(scores.is_empty());
    }

    #[test]
    fn make_candidates_helper_is_correct() {
        let c = make_candidates(&["a", "b"]);
        assert_eq!(c.len(), 2);
        assert_eq!(c[0].index, 0);
        assert_eq!(c[1].text, "b");
    }

    #[test]
    fn default_model_missing_emits_download_instructions() {
        // Unless the user happens to have the model cached, this test confirms
        // the error path is reachable and contains actionable guidance.
        //
        // If the model IS present in ~/.cache/skelesearch/reranker/ms-marco-MiniLM-L6-v2/,
        // this test is skipped so CI machines with cached models aren't broken.
        let cache = dirs::cache_dir()
            .unwrap_or_else(|| std::path::PathBuf::from(".cache"))
            .join("skelesearch")
            .join("reranker")
            .join(DEFAULT_MODEL_CACHE_NAME);
        if cache.join("model.onnx").exists() {
            // Model is cached; skip the error-path check.
            return;
        }
        let result = LocalReranker::default_model();
        assert!(result.is_err(), "expected error when model is absent");
        let msg = result.err().expect("expected error").to_string();
        assert!(msg.contains("hf download"), "expected download hint in: {msg}");
        assert!(msg.contains(DEFAULT_MODEL_REPO), "expected repo name in: {msg}");
    }
}
