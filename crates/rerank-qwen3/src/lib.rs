//! Qwen3-Reranker-0.6B cross-encoder (seq-cls ONNX variant) for skelesearch.
//!
//! Scores (query, document) pairs via a pointwise binary classifier.
//! The model outputs a single relevance logit per pair; sigmoid converts
//! it to a probability in [0, 1] used as the ranking score.
//!
//! # Tokenization
//!
//! Qwen3 is a decoder (GPT-style) model. LEFT padding is required so
//! that the final token (the generation position the model scores) is
//! correctly aligned across all items in a padded batch.
//!
//! # Prompt format
//!
//! Each pair is formatted with the model's instruct template before
//! tokenization.  Deviating from the template degrades performance.
//!
//! # Model source
//!
//! `from_hf()` downloads from `zhiqing/Qwen3-Reranker-0.6B-seq-cls-ONNX`
//! via the HuggingFace hub (cached to `~/.cache/huggingface/`).
//! `from_path()` loads from a local directory containing `model.onnx`
//! and `tokenizer.json`.

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

/// HuggingFace repository for the Qwen3 seq-cls ONNX reranker.
const HF_REPO: &str = "zhiqing/Qwen3-Reranker-0.6B-seq-cls-ONNX";

/// Maximum token sequence length.
///
/// Qwen3 supports 32k context; 8192 covers any realistic code chunk
/// while keeping tensor allocations bounded on CPU inference.
const MAX_SEQ_LEN: usize = 8192;

/// Max candidates per ONNX forward pass.
///
/// Qwen3-0.6B is a decoder model with large hidden states — 8 is
/// conservative and avoids OOM even on systems with limited RAM.
const SUB_BATCH: usize = 8;

/// Qwen3 pad token id.  `<|endoftext|>` (token 151643) doubles as the
/// EOS/PAD token for all Qwen3 models.
const QWEN3_PAD_ID: u32 = 151643;

/// Fixed instruction used in the prompt template for code search.
const INSTRUCT: &str =
    "Given a code search query, retrieve relevant code passages that answer the query";

// ---------------------------------------------------------------------------
// Prompt formatting
// ---------------------------------------------------------------------------

/// Format a (query, document) pair into the Qwen3-Reranker instruct template.
///
/// The template is mandated by the model authors; any deviation reduces
/// relevance signal quality.
fn format_prompt(query: &str, document: &str) -> String {
    format!(
        "<|im_start|>system\n\
         Judge whether the Document meets the requirements based on the Query and the Instruct \
         provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n\
         <|im_start|>user\n\
         <Instruct>: {INSTRUCT}\n\
         <Query>: {query}\n\
         <Document>: {document}<|im_end|>\n\
         <|im_start|>assistant\n\
         <think>\n\
         \n\
         </think>\n\
         \n"
    )
}

// ---------------------------------------------------------------------------
// Struct
// ---------------------------------------------------------------------------

/// Qwen3-Reranker-0.6B cross-encoder (seq-cls ONNX).
///
/// Thread-safe: the ONNX session is protected by a `Mutex` because ort
/// rc.11 requires `&mut Session` for `run()`.  Callers hold an `Arc`
/// and invoke `rerank` from any number of async tasks; each call
/// acquires the lock for the duration of inference.
pub struct Qwen3Reranker {
    /// Mutex because ort rc.11 requires &mut self for session.run().
    session: Arc<std::sync::Mutex<Session>>,
    tokenizer: Arc<Tokenizer>,
}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

impl Qwen3Reranker {
    /// Load from a local directory containing `model.onnx` and `tokenizer.json`.
    ///
    /// Returns an error (with the full path) if either file is missing.
    pub fn from_path(model_dir: impl AsRef<Path>) -> anyhow::Result<Self> {
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

        Self::load_files(&model_path, &tokenizer_path)
    }

    /// Download the Qwen3-Reranker-0.6B model from HuggingFace and load it.
    ///
    /// Files are cached in the HuggingFace hub cache directory
    /// (`~/.cache/huggingface/hub/`).  Subsequent calls skip the download
    /// and load from cache.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The HuggingFace API cannot be initialised (bad environment)
    /// - The download fails (network error, authentication required)
    /// - The ONNX session or tokenizer fails to load
    pub fn from_hf() -> anyhow::Result<Self> {
        use hf_hub::api::sync::Api;

        let api = Api::new().context("failed to create HuggingFace API client")?;
        let repo = api.model(HF_REPO.to_string());

        tracing::info!(repo = HF_REPO, "fetching Qwen3 reranker from HuggingFace hub");

        let model_path = repo
            .get("model.onnx")
            .with_context(|| format!("failed to download model.onnx from {HF_REPO}"))?;
        let tokenizer_path = repo
            .get("tokenizer.json")
            .with_context(|| format!("failed to download tokenizer.json from {HF_REPO}"))?;

        tracing::info!(repo = HF_REPO, "Qwen3 reranker model ready");

        Self::load_files(&model_path, &tokenizer_path)
    }

    /// Shared initialisation path: build session + tokenizer from resolved paths.
    fn load_files(model_path: &Path, tokenizer_path: &Path) -> anyhow::Result<Self> {
        let session = Session::builder()
            .context("failed to create ONNX session builder")?
            .commit_from_file(model_path)
            .with_context(|| format!("failed to load ONNX model: {}", model_path.display()))?;

        let mut tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| {
            anyhow::anyhow!(
                "failed to load tokenizer from {}: {e}",
                tokenizer_path.display()
            )
        })?;

        // Qwen3 is a decoder model — LEFT padding aligns the rightmost
        // (generation-position) token consistently across batch items.
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            direction: PaddingDirection::Left,
            pad_to_multiple_of: None,
            pad_id: QWEN3_PAD_ID,
            pad_type_id: 0,
            pad_token: "<|endoftext|>".to_string(),
        }));

        // Truncate at MAX_SEQ_LEN using LongestFirst so document tokens are
        // dropped before query tokens when the pair exceeds the budget.
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: MAX_SEQ_LEN,
                strategy: TruncationStrategy::LongestFirst,
                stride: 0,
                direction: TruncationDirection::Right,
            }))
            .map_err(|e| anyhow::anyhow!("failed to configure tokenizer truncation: {e}"))?;

        Ok(Self {
            session: Arc::new(std::sync::Mutex::new(session)),
            tokenizer: Arc::new(tokenizer),
        })
    }
}

// ---------------------------------------------------------------------------
// Synchronous inference (called inside spawn_blocking)
// ---------------------------------------------------------------------------

/// Score one sub-batch of pre-formatted prompt strings.
///
/// Returns sigmoid-transformed relevance scores in the same order as
/// `prompts`.  Must be called with an already-locked session.
fn run_batch(
    session: &mut Session,
    tokenizer: &Tokenizer,
    prompts: &[String],
) -> anyhow::Result<Vec<f64>> {
    let batch_size = prompts.len();

    // Single-sequence encoding: the prompt already contains both query and
    // document — no sentence-pair encoding needed.
    let inputs: Vec<EncodeInput> = prompts
        .iter()
        .map(|s| EncodeInput::Single(s.as_str().into()))
        .collect();

    let encodings = tokenizer
        .encode_batch(inputs, /* add_special_tokens */ true)
        .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;

    let seq_len = encodings.first().map(|e| e.get_ids().len()).unwrap_or(0);
    debug!(batch_size, seq_len, "sub-batch encoded");

    let mut input_ids = Array2::<i64>::zeros((batch_size, seq_len));
    let mut attention_mask = Array2::<i64>::zeros((batch_size, seq_len));

    for (i, enc) in encodings.iter().enumerate() {
        for (j, &id) in enc.get_ids().iter().enumerate() {
            input_ids[[i, j]] = id as i64;
        }
        for (j, &mask) in enc.get_attention_mask().iter().enumerate() {
            attention_mask[[i, j]] = mask as i64;
        }
    }

    // Collect output names before run() borrows &mut session.
    let output_names: Vec<String> = session
        .outputs()
        .iter()
        .map(|o| o.name().to_string())
        .collect();

    // Qwen3 seq-cls has no token_type_ids (decoder architecture).
    let outputs = session.run(ort::inputs![
        "input_ids" => ort::value::TensorRef::from_array_view(input_ids.view())?,
        "attention_mask" => ort::value::TensorRef::from_array_view(attention_mask.view())?
    ])?;

    let logits_val = outputs.get("logits").with_context(|| {
        format!(
            "model output 'logits' not found; available: {:?}",
            output_names
        )
    })?;
    let logits = logits_val.try_extract_array::<f32>()?;
    let shape = logits.shape();

    // Apply sigmoid to the raw logit to get a probability in [0, 1].
    //
    // [B, 2] — binary classifier; take the positive-class logit (index 1)
    //          and apply sigmoid.  (Softmax and sigmoid are equivalent here
    //          since we only need one probability for ranking.)
    // [B, 1] — single-logit reranker; apply sigmoid directly.
    // [B]    — flat output; apply sigmoid to each element.
    let sigmoid = |logit: f64| 1.0 / (1.0 + (-logit).exp());

    let scores: Vec<f64> = match shape {
        [_, 2] => (0..batch_size)
            .map(|i| sigmoid(logits[ndarray::IxDyn(&[i, 1])] as f64))
            .collect(),
        [_, 1] => (0..batch_size)
            .map(|i| sigmoid(logits[ndarray::IxDyn(&[i, 0])] as f64))
            .collect(),
        [_] => logits
            .iter()
            .map(|&s| sigmoid(s as f64))
            .collect(),
        _ => anyhow::bail!(
            "unexpected logits shape {:?}; expected [B], [B, 1], or [B, 2]",
            shape
        ),
    };

    Ok(scores)
}

// ---------------------------------------------------------------------------
// Reranker trait
// ---------------------------------------------------------------------------

#[async_trait]
impl Reranker for Qwen3Reranker {
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
        let total = candidates.len();
        let start = std::time::Instant::now();

        // ort inference is synchronous and CPU-bound; run off the async executor.
        let scores = tokio::task::spawn_blocking(move || {
            // Format all pairs before acquiring the session lock to minimise
            // the time the lock is held.
            let prompts: Vec<String> = candidates
                .iter()
                .map(|c| format_prompt(&query, &c.text))
                .collect();

            let mut session_guard = session
                .lock()
                .map_err(|_| anyhow::anyhow!("session mutex poisoned"))?;

            if prompts.len() <= SUB_BATCH {
                return run_batch(&mut session_guard, &tokenizer, &prompts);
            }

            // Sub-batch to keep tensor sizes bounded.
            let mut all_scores = Vec::with_capacity(total);
            for chunk in prompts.chunks(SUB_BATCH) {
                let batch_scores = run_batch(&mut session_guard, &tokenizer, chunk)?;
                all_scores.extend(batch_scores);
            }
            Ok(all_scores)
        })
        .await
        .context("Qwen3 reranker inference thread panicked")??;

        tracing::info!(
            candidates = total,
            elapsed_ms = start.elapsed().as_millis() as u64,
            "Qwen3 rerank complete"
        );

        Ok(scores)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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
    fn from_path_rejects_missing_model() {
        let result = Qwen3Reranker::from_path("/nonexistent/qwen3/path");
        assert!(result.is_err());
        let msg = result.err().expect("expected error").to_string();
        assert!(
            msg.contains("ONNX model not found"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn from_path_rejects_missing_tokenizer() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model.onnx"), b"").unwrap();
        let result = Qwen3Reranker::from_path(dir.path());
        assert!(result.is_err());
        let msg = result.err().expect("expected error").to_string();
        assert!(msg.contains("tokenizer.json not found"), "unexpected: {msg}");
    }

    #[tokio::test]
    async fn rerank_empty_candidates_returns_empty() {
        use skelesearch_core::reranker::NoopReranker;
        let scores = NoopReranker
            .rerank("query", vec![])
            .await
            .expect("noop should not fail");
        assert!(scores.is_empty());
    }

    #[test]
    fn format_prompt_contains_required_tokens() {
        let p = format_prompt("find sort function", "fn sort(arr: &mut [i32]) {}");
        assert!(p.contains("<|im_start|>system"));
        assert!(p.contains("<|im_start|>user"));
        assert!(p.contains("<|im_start|>assistant"));
        assert!(p.contains("<|im_end|>"));
        assert!(p.contains("find sort function"));
        assert!(p.contains("fn sort(arr: &mut [i32]) {}"));
        assert!(p.contains(INSTRUCT));
        assert!(p.contains("<think>"));
    }

    #[test]
    fn make_candidates_helper() {
        let c = make_candidates(&["a", "b", "c"]);
        assert_eq!(c.len(), 3);
        assert_eq!(c[0].index, 0);
        assert_eq!(c[2].text, "c");
    }
}
