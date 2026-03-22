use async_trait::async_trait;

/// Generates natural-language summaries of code chunks.
/// Used at index time to bridge the vocabulary gap between
/// code and natural-language queries.
#[async_trait]
pub trait SummaryProvider: Send + Sync {
    /// Summarize a batch of code chunks. Each input is the raw code text.
    /// Each output should be a 1-2 sentence natural-language description
    /// of what the code does.
    ///
    /// Returns one description per input, in the same order.
    /// If summarization fails for a chunk, return an empty string for that entry.
    async fn summarize_batch(&self, chunks: Vec<String>) -> anyhow::Result<Vec<String>>;
}

/// No-op summary provider that returns empty strings.
/// Used when summarization is disabled.
pub struct NoopSummaryProvider;

#[async_trait]
impl SummaryProvider for NoopSummaryProvider {
    async fn summarize_batch(&self, chunks: Vec<String>) -> anyhow::Result<Vec<String>> {
        Ok(chunks.iter().map(|_| String::new()).collect())
    }
}

/// Summary provider backed by OpenAI's chat completions API.
/// Uses GPT-4o-mini for cost efficiency (~$0.15/1M input tokens).
pub struct OpenAISummaryProvider {
    client: reqwest::Client,
    api_key: String,
    model: String,
}

impl OpenAISummaryProvider {
    pub fn new(api_key: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            model: "gpt-4o-mini".to_string(),
        }
    }

    pub fn with_model(api_key: String, model: &str) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            model: model.to_string(),
        }
    }

    async fn summarize_one(&self, code: &str) -> anyhow::Result<String> {
        use anyhow::Context;

        let body = serde_json::json!({
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a code documentation assistant. Given a code snippet, write a 1-2 sentence summary of what it does. Focus on the purpose and behavior, not the syntax. Be specific about function names, types, and key operations. Do not include markdown formatting."
                },
                {
                    "role": "user",
                    "content": code
                }
            ],
            "max_tokens": 100,
            "temperature": 0.0
        });

        let resp = self
            .client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await
            .context("OpenAI summary request failed")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!(
                "OpenAI summary API error {}: {}",
                status,
                &text[..text.len().min(200)]
            );
        }

        let data: serde_json::Value = resp.json().await?;
        let summary = data["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .trim()
            .to_string();

        Ok(summary)
    }
}

#[async_trait]
impl SummaryProvider for OpenAISummaryProvider {
    async fn summarize_batch(&self, chunks: Vec<String>) -> anyhow::Result<Vec<String>> {
        // Process up to 10 requests concurrently per sub-batch, preserving order.
        // We use tokio::task::JoinSet and pair each result with its original index
        // so the final Vec matches the input order regardless of completion order.
        let mut results = vec![String::new(); chunks.len()];

        for (batch_start, batch) in chunks.chunks(10).enumerate() {
            let mut set = tokio::task::JoinSet::new();

            for (local_idx, code) in batch.iter().enumerate() {
                let global_idx = batch_start * 10 + local_idx;
                // Clone both the client handle and the string so the spawned
                // task owns everything it needs without holding &self.
                let client = self.client.clone();
                let api_key = self.api_key.clone();
                let model = self.model.clone();
                let code = code.clone();

                set.spawn(async move {
                    let provider = OpenAISummaryProvider { client, api_key, model };
                    let summary = provider.summarize_one(&code).await;
                    (global_idx, summary)
                });
            }

            while let Some(res) = set.join_next().await {
                match res {
                    Ok((idx, Ok(summary))) => results[idx] = summary,
                    Ok((idx, Err(e))) => {
                        tracing::warn!(error = %e, chunk_index = idx, "chunk summary failed, using empty");
                        // results[idx] already holds String::new()
                    }
                    Err(join_err) => {
                        tracing::warn!(error = %join_err, "summary task panicked");
                    }
                }
            }
        }

        Ok(results)
    }
}
