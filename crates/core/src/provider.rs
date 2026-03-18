use async_trait::async_trait;

/// Embedding provider trait — backend-agnostic contract.
///
/// Callers set `dim` once at index creation time; the provider must return
/// vectors of exactly that dimensionality from `embed_batch`.
#[async_trait]
pub trait EmbedProvider: Send + Sync {
    /// Dimensionality of vectors produced by this provider.
    fn dim(&self) -> usize;

    /// Human-readable provider name for manifest storage.
    fn name(&self) -> &str { "unknown" }

    /// Embed a batch of texts, returning one vector per input in order.
    ///
    /// # Errors
    /// Returns an error if the underlying model call fails or the returned
    /// vector count does not match `texts.len()`.
    async fn embed_batch(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>>;
}


// Blanket impl so `Box<dyn EmbedProvider>` satisfies `P: EmbedProvider` bounds.
// Without this, generic call sites (Indexer, Searcher) cannot accept a boxed provider.
#[async_trait]
impl<T: EmbedProvider + ?Sized> EmbedProvider for Box<T> {
    fn dim(&self) -> usize {
        (**self).dim()
    }

    fn name(&self) -> &str {
        (**self).name()
    }

    async fn embed_batch(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
        (**self).embed_batch(texts).await
    }
}