use async_trait::async_trait;

/// Embedding provider trait — backend-agnostic contract.
///
/// Callers set `dim` once at index creation time; the provider must return
/// vectors of exactly that dimensionality from `embed_batch`.
#[async_trait]
pub trait EmbedProvider: Send + Sync {
    /// Dimensionality of vectors produced by this provider.
    fn dim(&self) -> usize;

    /// Embed a batch of texts, returning one vector per input in order.
    ///
    /// # Errors
    /// Returns an error if the underlying model call fails or the returned
    /// vector count does not match `texts.len()`.
    async fn embed_batch(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>>;
}
