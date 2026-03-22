//! Cloud reranker providers for skelesearch.
//!
//! Supports Jina reranker-v3, Cohere, and Voyage reranker APIs through a unified client.
//! All three share nearly identical REST interfaces:
//! POST { model, query, documents, top_n } → [{ index, relevance_score }]

use anyhow::Context;
use async_trait::async_trait;
use skelesearch_core::reranker::{RerankCandidate, Reranker};

/// Supported reranker providers.
#[derive(Debug, Clone)]
pub enum RerankProvider {
    /// Jina AI — reranker-v3 (Oct 2025). Best code quality per dollar.
    Jina,
    /// Cohere — highest quality general reranker.
    Cohere,
    /// Voyage AI — cheapest, instruction-following.
    Voyage,
}

impl RerankProvider {
    fn endpoint(&self) -> &str {
        match self {
            Self::Jina => "https://api.jina.ai/v1/rerank",
            Self::Cohere => "https://api.cohere.com/v2/rerank",
            Self::Voyage => "https://api.voyageai.com/v1/rerank",
        }
    }

    fn default_model(&self) -> &str {
        match self {
            Self::Jina => "jina-reranker-v3",
            Self::Cohere => "rerank-v3.5",
            Self::Voyage => "rerank-2.5",
        }
    }

    /// Field name for the result count limit — Voyage diverges from the others.
    fn top_n_field(&self) -> &str {
        match self {
            Self::Voyage => "top_k",
            _ => "top_n",
        }
    }

    /// Extract ranked results from the provider-specific response envelope.
    ///
    /// Jina/Cohere wrap results in `results`; Voyage uses `data`.
    fn parse_results(&self, json: &serde_json::Value) -> Vec<RerankResult> {
        let arr = match self {
            Self::Cohere | Self::Jina => json["results"].as_array(),
            Self::Voyage => json["data"].as_array(),
        };
        arr.map(|results| {
            results
                .iter()
                .filter_map(|r| {
                    Some(RerankResult {
                        index: r["index"].as_u64()? as usize,
                        score: r["relevance_score"].as_f64()?,
                    })
                })
                .collect()
        })
        .unwrap_or_default()
    }
}

#[derive(Debug)]
struct RerankResult {
    index: usize,
    score: f64,
}

/// Cloud reranker client supporting Jina, Cohere, and Voyage APIs.
pub struct ApiReranker {
    client: reqwest::Client,
    provider: RerankProvider,
    api_key: String,
    model: String,
}

impl ApiReranker {
    pub fn new(provider: RerankProvider, api_key: String) -> Self {
        let model = provider.default_model().to_string();
        Self {
            client: reqwest::Client::new(),
            provider,
            api_key,
            model,
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }
}

#[async_trait]
impl Reranker for ApiReranker {
    #[tracing::instrument(skip_all, fields(provider = ?self.provider, candidates = candidates.len()))]
    async fn rerank(
        &self,
        query: &str,
        candidates: Vec<RerankCandidate>,
    ) -> anyhow::Result<Vec<f64>> {
        if candidates.is_empty() {
            return Ok(vec![]);
        }

        let documents: Vec<&str> = candidates.iter().map(|c| c.text.as_str()).collect();
        let top_n_field = self.provider.top_n_field();

        // Rerank all candidates; caller is responsible for truncation.
        let body = serde_json::json!({
            "model": self.model,
            "query": query,
            "documents": documents,
            top_n_field: candidates.len(),
        });

        let resp = self
            .client
            .post(self.provider.endpoint())
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .context("reranker API request failed")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("reranker API error (HTTP {status}): {body}");
        }

        let json: serde_json::Value = resp
            .json()
            .await
            .context("failed to parse reranker response")?;

        let results = self.provider.parse_results(&json);

        // Map API results back to original candidate order.
        // The API may return a subset or reordered slice; unscored candidates
        // default to 0.0 (lowest relevance).
        let mut scores = vec![0.0f64; candidates.len()];
        for r in &results {
            if r.index < scores.len() {
                scores[r.index] = r.score;
            }
        }

        Ok(scores)
    }
}

/// Build a reranker from a provider name string.
///
/// Supported names: `"jina"`, `"cohere"`, `"voyage"`.
pub fn reranker_from_name(name: &str, api_key: String) -> anyhow::Result<ApiReranker> {
    let provider = match name {
        "jina" => RerankProvider::Jina,
        "cohere" => RerankProvider::Cohere,
        "voyage" => RerankProvider::Voyage,
        other => anyhow::bail!(
            "unknown reranker provider: '{}'. Supported: jina, cohere, voyage",
            other
        ),
    };
    Ok(ApiReranker::new(provider, api_key))
}
