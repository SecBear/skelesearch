/// Sparse embedding: token IDs and their activation weights.
///
/// Produced by lexical-semantic models (BGE-M3 sparse, SPLADE). Each `indices[i]`
/// is a vocabulary token ID and `values[i]` is its activation weight. The two
/// slices are co-indexed and must have equal length. Zero-weight tokens are not
/// represented — the embedding is already in sparse format.
#[derive(Debug, Clone)]
pub struct SparseEmbedding {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

impl SparseEmbedding {
    /// True when the embedding has no active tokens (e.g., empty input or padding-only).
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

/// Provider for sparse embeddings (e.g. BGE-M3 sparse, SPLADE).
///
/// Implementations must be `Send + Sync` so they can be shared across
/// threads.  `embed_batch` is synchronous — callers that need to run it
/// from an async executor should use `tokio::task::spawn_blocking`.
pub trait SparseEmbedProvider: Send + Sync {
    fn name(&self) -> &str;

    /// Embed a batch of texts. Returns one [`SparseEmbedding`] per input
    /// text in the same order. Empty texts produce empty embeddings.
    fn embed_batch(&self, texts: &[&str]) -> anyhow::Result<Vec<SparseEmbedding>>;
}
