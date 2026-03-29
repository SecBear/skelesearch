use async_trait::async_trait;

fn env_var_present(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|value| !value.trim().is_empty())
        .unwrap_or(false)
}

/// Resolve the preferred embedding provider for new indexing work.
pub fn preferred_index_provider_name() -> &'static str {
    if env_var_present("VOYAGE_API_KEY") {
        "voyage"
    } else if env_var_present("OPENAI_API_KEY") {
        "openai"
    } else {
        "fastembed"
    }
}

/// Embedding provider trait — backend-agnostic contract.
///
/// Callers set `dim` once at index creation time; the provider must return
/// vectors of exactly that dimensionality from `embed_batch`.
#[async_trait]
pub trait EmbedProvider: Send + Sync {
    /// Dimensionality of vectors produced by this provider.
    fn dim(&self) -> usize;

    /// Human-readable provider name for manifest storage.
    fn name(&self) -> &str {
        "unknown"
    }

    /// Optional prefix to prepend to query text before embedding.
    /// Models like CodeRankEmbed require instruction prefixes for
    /// query-document alignment. Default implementation returns `None`.
    fn query_prefix(&self) -> Option<&str> {
        None
    }

    /// Embed a batch of texts, returning one vector per input in order.
    ///
    /// # Errors
    /// Returns an error if the underlying model call fails or the returned
    /// vector count does not match `texts.len()`.
    async fn embed_batch(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>>;

    /// Embed queries specifically (as opposed to documents).
    /// Providers that distinguish query vs document embeddings (e.g. Voyage AI
    /// with `input_type`) should override this. Default delegates to `embed_batch`.
    async fn embed_queries(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
        self.embed_batch(texts).await
    }
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

    fn query_prefix(&self) -> Option<&str> {
        (**self).query_prefix()
    }

    async fn embed_batch(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
        (**self).embed_batch(texts).await
    }

    async fn embed_queries(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
        (**self).embed_queries(texts).await
    }
}
