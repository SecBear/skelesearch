//! Cross-encoder reranker for second-stage precision improvement.
//!
//! After initial retrieval (BM25+vector+RRF) and MMR diversity reranking,
//! a cross-encoder reranker scores each (query, document) pair jointly,
//! improving precision by 5-15% nDCG@10 over bi-encoder retrieval alone.
//!
//! The trait is intentionally minimal: concrete implementations (ONNX,
//! HTTP API) live in separate crates so the core crate stays dependency-light.

use async_trait::async_trait;

/// A scored (query, document) pair ready for reranking.
#[derive(Debug, Clone)]
pub struct RerankCandidate {
    /// Original retrieval index (for mapping back to SearchResult).
    pub index: usize,
    /// The document text to score against the query.
    pub text: String,
}

/// Reranker trait — scores (query, document) pairs jointly.
///
/// Implementations receive all candidates and return a score per candidate
/// in the same order.  Higher scores → higher relevance.
#[async_trait]
pub trait Reranker: Send + Sync {
    /// Score each candidate against the query.
    ///
    /// Returns scores in the same order as `candidates`.
    /// The magnitude of scores is implementation-defined; only the ordering
    /// is used by the search pipeline.
    async fn rerank(
        &self,
        query: &str,
        candidates: Vec<RerankCandidate>,
    ) -> anyhow::Result<Vec<f64>>;
}

/// No-op reranker that preserves original retrieval order.
///
/// Used when no reranker is configured.  Returns strictly decreasing scores
/// so the sort in the search pipeline is stable and order-preserving.
pub struct NoopReranker;

#[async_trait]
impl Reranker for NoopReranker {
    async fn rerank(
        &self,
        _query: &str,
        candidates: Vec<RerankCandidate>,
    ) -> anyhow::Result<Vec<f64>> {
        // Descending integers: candidate 0 gets the highest score,
        // preserving the order established by hybrid_search + MMR.
        Ok((0..candidates.len())
            .map(|i| (candidates.len() - i) as f64)
            .collect())
    }
}
