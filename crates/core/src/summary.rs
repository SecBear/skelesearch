//! Summary generation for bridging the vocabulary gap between natural-language
//! queries and source code at index time.
//!
//! When a `SummaryProvider` is attached to the `Indexer`, each chunk's
//! description is generated before embedding.  The HNSW index then stores
//! description-based vectors instead of raw-code vectors, so natural-language
//! queries match semantically relevant chunks even when they share no tokens
//! with the query.

use async_trait::async_trait;

/// Generates natural-language descriptions of code chunks.
///
/// Implementations are expected to be cheap to call in batch and to degrade
/// gracefully: returning an empty string is always acceptable and causes the
/// indexer to fall back to embedding raw code for that chunk.
#[async_trait]
pub trait SummaryProvider: Send + Sync {
    /// Generate descriptions for a batch of code texts.
    ///
    /// The returned `Vec` **MUST** have the same length as `texts`.
    /// An empty string at position `i` means "no summary available for texts[i]";
    /// the indexer will embed the raw code for that chunk.
    async fn summarize_batch(&self, texts: Vec<String>) -> anyhow::Result<Vec<String>>;
}

/// No-op provider that returns empty strings for all inputs.
///
/// Used as the default when no summary provider is configured, making
/// the trait optional without requiring callers to special-case `None`.
pub struct NoopSummaryProvider;

#[async_trait]
impl SummaryProvider for NoopSummaryProvider {
    async fn summarize_batch(&self, texts: Vec<String>) -> anyhow::Result<Vec<String>> {
        Ok(vec![String::new(); texts.len()])
    }
}

/// OpenAI-backed summary provider using the chat completions API.
///
/// Calls gpt-4o-mini in parallel batches to generate concise English
/// descriptions of each code chunk.  Failures are degraded gracefully:
/// a failed call returns empty strings so raw-code embedding is used.
pub struct OpenAISummaryProvider {
    client: reqwest::Client,
    api_url: String,
    api_key: String,
    model: String,
}

impl OpenAISummaryProvider {
    /// Create from an API key. Uses `gpt-4o-mini` by default.
    pub fn new(api_key: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_url: "https://api.openai.com/v1/chat/completions".into(),
            api_key,
            model: "gpt-4o-mini".into(),
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }
}

#[async_trait]
impl SummaryProvider for OpenAISummaryProvider {
    async fn summarize_batch(&self, texts: Vec<String>) -> anyhow::Result<Vec<String>> {
        // Call the API once per chunk.  For large batches this is sequential;
        // a future optimisation could fan out concurrently, but for now we
        // keep it simple and let the caller control batch size.
        let mut results = Vec::with_capacity(texts.len());
        for text in &texts {
            let prompt = format!(
                "Describe the following code in one or two concise English sentences. \
                 Focus on what it does and what problem it solves, not how it is implemented. \
                 Return ONLY the description, no other text.\n\n```\n{}\n```",
                text
            );
            let body = serde_json::json!({
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 150,
                "temperature": 0.0,
            });

            match self
                .client
                .post(&self.api_url)
                .bearer_auth(&self.api_key)
                .json(&body)
                .send()
                .await
            {
                Ok(resp) if resp.status().is_success() => {
                    let json: serde_json::Value = resp.json().await.unwrap_or_default();
                    let desc = json["choices"][0]["message"]["content"]
                        .as_str()
                        .unwrap_or_default()
                        .trim()
                        .to_string();
                    results.push(desc);
                }
                Ok(resp) => {
                    let status = resp.status();
                    let body = resp.text().await.unwrap_or_default();
                    tracing::warn!(%status, "summary API call failed: {body}");
                    results.push(String::new());
                }
                Err(e) => {
                    tracing::warn!(error = %e, "summary API request error");
                    results.push(String::new());
                }
            }
        }
        Ok(results)
    }
}
