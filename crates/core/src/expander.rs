//! Query expansion for bridging vocabulary gaps between natural language
//! queries and code identifiers.

use async_trait::async_trait;

/// Expands a natural language query with code-vocabulary synonyms.
#[async_trait]
pub trait QueryExpander: Send + Sync {
    /// Given a query, return additional keywords/identifiers that might
    /// appear in relevant code. Returns empty vec if no expansion is needed.
    async fn expand(&self, query: &str) -> anyhow::Result<Vec<String>>;
}

/// No-op expander that returns no additional keywords.
pub struct NoopExpander;

#[async_trait]
impl QueryExpander for NoopExpander {
    async fn expand(&self, _query: &str) -> anyhow::Result<Vec<String>> {
        Ok(vec![])
    }
}

/// LLM-based expander that calls an OpenAI-compatible completions API
/// to generate code-vocabulary keywords for conceptual queries.
pub struct LLMExpander {
    client: reqwest::Client,
    api_url: String,
    api_key: String,
    model: String,
}

impl LLMExpander {
    /// Create from an API key. Defaults to OpenAI gpt-4o-mini.
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

    pub fn with_url(mut self, url: impl Into<String>) -> Self {
        self.api_url = url.into();
        self
    }
}

#[async_trait]
impl QueryExpander for LLMExpander {
    #[tracing::instrument(skip_all, fields(query = %query))]
    async fn expand(&self, query: &str) -> anyhow::Result<Vec<String>> {
        let prompt = format!(
            "Given this code search query, list 3-5 code identifiers, function names, \
             struct names, or keywords that might appear in relevant source files. \
             Return ONLY the keywords separated by commas, nothing else.\n\n\
             Query: {}",
            query
        );

        let body = serde_json::json!({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100,
            "temperature": 0.0,
        });

        let resp = self
            .client
            .post(&self.api_url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            tracing::warn!(%status, "query expansion LLM call failed: {body}");
            // Graceful degradation — search proceeds without expansion.
            return Ok(vec![]);
        }

        let json: serde_json::Value = resp.json().await?;
        let content = json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or_default();

        let keywords: Vec<String> = content
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty() && s.len() > 1)
            .collect();

        tracing::debug!(expanded = ?keywords, "query expansion complete");
        Ok(keywords)
    }
}
