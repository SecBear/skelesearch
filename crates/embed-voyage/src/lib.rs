// VoyageProvider — cloud embedding via Voyage AI voyage-code-3 (default)
//
// Auth resolution:
//   VOYAGE_API_KEY env var (non-empty)

use anyhow::Context;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use skelesearch_core::EmbedProvider;

// ---------------------------------------------------------------------------
// Voyage AI API wire types
// ---------------------------------------------------------------------------

/// Request body for POST /v1/embeddings.
#[derive(Debug, Serialize)]
struct VoyageEmbeddingRequest<'a> {
    model: &'a str,
    input: &'a [String],
}

/// Top-level response envelope.
#[derive(Debug, Deserialize)]
struct VoyageEmbeddingResponse {
    data: Vec<VoyageEmbeddingData>,
    #[allow(dead_code)]
    usage: VoyageEmbeddingUsage,
}

/// Per-item embedding, preserving the API-supplied `index` for reordering.
///
/// The API does not guarantee that `data` comes back in the same order as
/// the input, so callers **must** sort by `index` before returning.
#[derive(Debug, Deserialize)]
struct VoyageEmbeddingData {
    embedding: Vec<f32>,
    index: usize,
}

/// Token usage (informational; logged at trace level).
#[derive(Debug, Deserialize)]
struct VoyageEmbeddingUsage {
    #[allow(dead_code)]
    total_tokens: u64,
}

// ---------------------------------------------------------------------------
// API key resolution
// ---------------------------------------------------------------------------

/// Resolve the Voyage AI API key from the `VOYAGE_API_KEY` env var.
///
/// # Errors
/// Returns an error if the variable is absent or empty.
fn resolve_api_key() -> anyhow::Result<String> {
    let key = std::env::var("VOYAGE_API_KEY")
        .context("VOYAGE_API_KEY env var not set")?;
    anyhow::ensure!(!key.is_empty(), "VOYAGE_API_KEY env var is empty");
    Ok(key)
}

// ---------------------------------------------------------------------------
// Provider struct
// ---------------------------------------------------------------------------

/// Maximum number of texts per individual Voyage AI API request.
///
/// Voyage AI's embeddings endpoint accepts up to 128 inputs per call.
const BATCH_LIMIT: usize = 128;

/// [`EmbedProvider`] backed by the Voyage AI `/v1/embeddings` endpoint.
///
/// Default model: `voyage-code-3` (1024-dim, 32K context).
/// Use [`VoyageProvider::with_model`] to select a different model and dim.
pub struct VoyageProvider {
    client: reqwest::Client,
    api_key: String,
    model: String,
    dim: usize,
    api_url: String,
    /// Cumulative token usage across all embed_batch calls.
    total_tokens_used: std::sync::atomic::AtomicU64,
}

impl VoyageProvider {
    /// Construct using `voyage-code-3` (1024-dim).
    ///
    /// Resolves the API key from `VOYAGE_API_KEY`.
    ///
    /// # Errors
    /// Returns an error if no API key can be found or the HTTP client cannot
    /// be built (e.g., TLS initialisation failure).
    pub fn new() -> anyhow::Result<Self> {
        Self::with_model("voyage-code-3", 1024)
    }

    /// Construct with an explicit model name and declared dimensionality.
    ///
    /// `dim` must match the model's actual output dimensionality; there is no
    /// runtime validation — the mismatch will surface as count errors later.
    ///
    /// # Errors
    /// Same as [`Self::new`].
    pub fn with_model(model: &str, dim: usize) -> anyhow::Result<Self> {
        let api_key = resolve_api_key().context("failed to resolve Voyage AI API key")?;
        let client = reqwest::Client::builder()
            .build()
            .context("failed to build reqwest client")?;

        Ok(Self {
            client,
            api_key,
            model: model.to_string(),
            dim,
            api_url: "https://api.voyageai.com/v1/embeddings".to_string(),
            total_tokens_used: std::sync::atomic::AtomicU64::new(0),
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
    async fn post_batch(&self, texts: &[String]) -> anyhow::Result<Vec<VoyageEmbeddingData>> {
        let body = VoyageEmbeddingRequest {
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
                        "rate limited by Voyage AI; retrying after {}s",
                        delay_secs
                    );
                    tokio::time::sleep(std::time::Duration::from_secs(delay_secs)).await;
                    delay_secs *= 2;
                    continue;
                }
                let body_text = response.text().await.unwrap_or_default();
                anyhow::bail!(
                    "Voyage AI rate limit exceeded after 3 attempts; last response: {}",
                    body_text
                );
            }

            if status == reqwest::StatusCode::UNAUTHORIZED
                || status == reqwest::StatusCode::FORBIDDEN
            {
                let body_text = response.text().await.unwrap_or_default();
                anyhow::bail!(
                    "Voyage AI authentication failed (HTTP {}): {}. \
                     Check VOYAGE_API_KEY env var.",
                    status,
                    body_text
                );
            }

            if !status.is_success() {
                let body_text = response.text().await.unwrap_or_default();
                anyhow::bail!(
                    "Voyage AI embeddings API error (HTTP {}): {}",
                    status,
                    body_text
                );
            }

            let parsed: VoyageEmbeddingResponse = response
                .json()
                .await
                .context("failed to deserialize Voyage AI embedding response")?;

            let tokens = parsed.usage.total_tokens;
            let cumulative = self.total_tokens_used.fetch_add(tokens, std::sync::atomic::Ordering::Relaxed) + tokens;
            tracing::info!(
                batch_tokens = tokens,
                cumulative_tokens = cumulative,
                "Voyage AI embedding usage"
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
impl EmbedProvider for VoyageProvider {
    fn dim(&self) -> usize {
        self.dim
    }

    fn name(&self) -> &str {
        "voyage"
    }

    async fn embed_batch(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let expected = texts.len();
        // Accumulate VoyageEmbeddingData items across all sub-batches.  Each item
        // carries an `index` referring to its position in the *sub-batch*, so
        // we track the global offset and shift indices accordingly.
        let mut all_data: Vec<VoyageEmbeddingData> = Vec::with_capacity(expected);
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

/// Convenience constructor returning a [`VoyageProvider`] with default
/// settings (`voyage-code-3`, 1024-dim).
///
/// # Errors
/// Returns an error if API key resolution or HTTP client construction fails.
pub fn provider_voyage() -> anyhow::Result<VoyageProvider> {
    VoyageProvider::new().context("failed to initialise Voyage AI embedding provider")
}