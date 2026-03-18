// OpenAIProvider — cloud embedding via OpenAI text-embedding-3-small (default)
//
// Auth resolution order:
//   1. OPENAI_API_KEY env var
//   2. ~/.pi/agent/auth.json, key "openai-codex", field "access"

use anyhow::Context;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use skelesearch_core::EmbedProvider;

// ---------------------------------------------------------------------------
// OpenAI API wire types
// ---------------------------------------------------------------------------

/// Request body for POST /v1/embeddings.
#[derive(Debug, Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    input: &'a [String],
}

/// Top-level response envelope.
#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
    #[allow(dead_code)]
    usage: EmbeddingUsage,
}

/// Per-item embedding, preserving the API-supplied `index` for reordering.
///
/// The API does not guarantee that `data` comes back in the same order as
/// the input, so callers **must** sort by `index` before returning.
#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
    index: usize,
}

/// Token usage (informational; logged at trace level).
#[derive(Debug, Deserialize)]
struct EmbeddingUsage {
    #[allow(dead_code)]
    prompt_tokens: u64,
    #[allow(dead_code)]
    total_tokens: u64,
}

// ---------------------------------------------------------------------------
// API key resolution
// ---------------------------------------------------------------------------

/// Resolve the OpenAI API key:
///   1. `OPENAI_API_KEY` env var (non-empty)
///   2. `~/.pi/agent/auth.json` → `["openai-codex"]["access"]`
fn resolve_api_key() -> anyhow::Result<String> {
    if let Ok(key) = std::env::var("OPENAI_API_KEY") {
        if !key.is_empty() {
            return Ok(key);
        }
    }

    let home = std::env::var("HOME").context("HOME not set")?;
    let path = std::path::PathBuf::from(home).join(".pi/agent/auth.json");
    let raw = std::fs::read_to_string(&path).with_context(|| {
        format!(
            "no OPENAI_API_KEY env var and cannot read {}",
            path.display()
        )
    })?;
    let creds: serde_json::Value = serde_json::from_str(&raw)
        .with_context(|| format!("failed to parse {}", path.display()))?;

    creds
        .get("openai-codex")
        .and_then(|e| e.get("access"))
        .and_then(|v| v.as_str())
        .map(String::from)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "no openai-codex credential in {}; set OPENAI_API_KEY or add the credential",
                path.display()
            )
        })
}

// ---------------------------------------------------------------------------
// Provider struct
// ---------------------------------------------------------------------------

/// Maximum number of texts per individual API request.
///
/// The OpenAI embeddings endpoint accepts up to 2048 inputs per call.
const BATCH_LIMIT: usize = 2048;

/// [`EmbedProvider`] backed by the OpenAI `/v1/embeddings` endpoint.
///
/// Default model: `text-embedding-3-small` (1536-dim).
/// Use [`OpenAIProvider::with_model`] to select a different model and dim.
pub struct OpenAIProvider {
    client: reqwest::Client,
    api_key: String,
    model: String,
    dim: usize,
    api_url: String,
}

impl OpenAIProvider {
    /// Construct using `text-embedding-3-small` (1536-dim).
    ///
    /// Resolves the API key from `OPENAI_API_KEY` or `~/.pi/agent/auth.json`.
    ///
    /// # Errors
    /// Returns an error if no API key can be found or the HTTP client cannot
    /// be built (e.g., TLS initialisation failure).
    pub fn new() -> anyhow::Result<Self> {
        Self::with_model("text-embedding-3-small", 1536)
    }

    /// Construct with an explicit model name and declared dimensionality.
    ///
    /// `dim` must match the model's actual output dimensionality; there is no
    /// runtime validation — the mismatch will surface as count errors later.
    ///
    /// # Errors
    /// Same as [`Self::new`].
    pub fn with_model(model: &str, dim: usize) -> anyhow::Result<Self> {
        let api_key = resolve_api_key().context("failed to resolve OpenAI API key")?;
        let client = reqwest::Client::builder()
            .build()
            .context("failed to build reqwest client")?;

        Ok(Self {
            client,
            api_key,
            model: model.to_string(),
            dim,
            api_url: "https://api.openai.com/v1/embeddings".to_string(),
        })
    }

    /// Override the API URL (useful for testing against a local proxy or mock).
    pub fn with_url(mut self, url: impl Into<String>) -> Self {
        self.api_url = url.into();
        self
    }

    /// POST a single sub-batch (≤ BATCH_LIMIT texts) to the API.
    ///
    /// Retries up to 3 times on 429 with exponential back-off (1 s, 2 s, 4 s).
    /// Auth failures (401/403) and other HTTP errors are returned immediately.
    async fn post_batch(&self, texts: &[String]) -> anyhow::Result<Vec<EmbeddingData>> {
        let body = EmbeddingRequest {
            model: &self.model,
            input: texts,
        };

        let mut delay_secs = 1u64;
        for attempt in 1..=3u32 {
            let response = self
                .client
                .post(&self.api_url)
                .bearer_auth(&self.api_key)
                .json(&body)
                .send()
                .await
                .context("network error sending embedding request")?;

            let status = response.status();

            if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
                if attempt < 3 {
                    tracing::warn!(
                        attempt,
                        delay_secs,
                        "rate limited by OpenAI; retrying after {}s",
                        delay_secs
                    );
                    tokio::time::sleep(std::time::Duration::from_secs(delay_secs)).await;
                    delay_secs *= 2;
                    continue;
                }
                let body_text = response.text().await.unwrap_or_default();
                anyhow::bail!(
                    "OpenAI rate limit exceeded after 3 attempts; last response: {}",
                    body_text
                );
            }

            if status == reqwest::StatusCode::UNAUTHORIZED
                || status == reqwest::StatusCode::FORBIDDEN
            {
                let body_text = response.text().await.unwrap_or_default();
                anyhow::bail!(
                    "OpenAI authentication failed (HTTP {}): {}. \
                     Check OPENAI_API_KEY env var or ~/.pi/agent/auth.json openai-codex credential.",
                    status,
                    body_text
                );
            }

            if !status.is_success() {
                let body_text = response.text().await.unwrap_or_default();
                anyhow::bail!(
                    "OpenAI embeddings API error (HTTP {}): {}",
                    status,
                    body_text
                );
            }

            let parsed: EmbeddingResponse = response
                .json()
                .await
                .context("failed to deserialize OpenAI embedding response")?;

            tracing::trace!(
                prompt_tokens = parsed.usage.prompt_tokens,
                total_tokens = parsed.usage.total_tokens,
                "OpenAI embedding usage"
            );

            return Ok(parsed.data);
        }

        // Unreachable: the loop exits on either success or an explicit bail.
        anyhow::bail!("embedding retry loop exhausted unexpectedly")
    }
}

// ---------------------------------------------------------------------------
// EmbedProvider implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl EmbedProvider for OpenAIProvider {
    fn dim(&self) -> usize {
        self.dim
    }

    fn name(&self) -> &str {
        "openai"
    }

    #[tracing::instrument(skip_all, fields(batch_size = texts.len()))]
    async fn embed_batch(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let expected = texts.len();
        // Accumulate EmbeddingData items across all sub-batches.  Each item
        // carries an `index` referring to its position in the *sub-batch*, so
        // we track the global offset and shift indices accordingly.
        let mut all_data: Vec<EmbeddingData> = Vec::with_capacity(expected);
        let mut global_offset = 0usize;

        for chunk in texts.chunks(BATCH_LIMIT) {
            let mut chunk_data = self.post_batch(chunk).await.with_context(|| {
                format!(
                    "embedding sub-batch at offset {} (size {}) failed",
                    global_offset,
                    chunk.len()
                )
            })?;

            if chunk_data.len() != chunk.len() {
                anyhow::bail!(
                    "expected {} embeddings for sub-batch at offset {}, got {}",
                    chunk.len(),
                    global_offset,
                    chunk_data.len()
                );
            }

            // Shift each item's index to the global coordinate space.
            for item in &mut chunk_data {
                item.index += global_offset;
            }
            all_data.extend(chunk_data);
            global_offset += chunk.len();
        }

        // Guard: total count must equal input length.
        anyhow::ensure!(
            all_data.len() == expected,
            "expected {} total embeddings, got {}",
            expected,
            all_data.len()
        );

        // Sort by global index — the API does not guarantee ordering.
        all_data.sort_unstable_by_key(|d| d.index);

        Ok(all_data.into_iter().map(|d| d.embedding).collect())
    }
}

// ---------------------------------------------------------------------------
// Provider factory
// ---------------------------------------------------------------------------

/// Convenience constructor returning an [`OpenAIProvider`] with default
/// settings (`text-embedding-3-small`, 1536-dim).
///
/// # Errors
/// Returns an error if API key resolution or HTTP client construction fails.
pub fn provider_openai() -> anyhow::Result<OpenAIProvider> {
    OpenAIProvider::new().context("failed to initialise OpenAI embedding provider")
}
