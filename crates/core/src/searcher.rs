use std::sync::Arc;

use crate::{ChunkRecord, EmbedProvider, SearchResult, StorageBackend};

// ---------------------------------------------------------------------------
// Public output types
// ---------------------------------------------------------------------------

/// Per-file context: all indexed chunks plus one-hop import graph for that file.
#[derive(Debug, Clone, Default)]
pub struct FileContext {
    pub chunks: Vec<ChunkRecord>,
    /// Paths that `file_path` imports (outbound edges).
    pub imports: Vec<String>,
    /// Paths that import `file_path` (inbound edges).
    pub imported_by: Vec<String>,
}

// ---------------------------------------------------------------------------
// Searcher
// ---------------------------------------------------------------------------

/// Read-path wrapper: embeds queries, delegates to the storage backend, and
/// shapes raw results into labelled `SearchResult` rows.
///
/// `B` is the storage backend; `P` is the embedding provider.
pub struct Searcher<B, P> {
    backend: Arc<B>,
    provider: P,
}

impl<B: StorageBackend, P: EmbedProvider> Searcher<B, P> {
    pub fn new(backend: Arc<B>, provider: P) -> Self {
        Self { backend, provider }
    }

    /// Search the index for `query`.
    ///
    /// - `top_k`        — maximum number of primary results.
    /// - `include_graph` — when `true`, augment results with transitive import
    ///   neighbours up to `max_depth` hops.
    /// - `max_depth`    — BFS depth for graph augmentation; 0 disables graph
    ///   traversal even when `include_graph` is `true`.
    ///
    /// Returns an empty `Vec` when no results match; never returns an error
    /// for a zero-result query.
    pub async fn search(
        &self,
        query: &str,
        top_k: usize,
        include_graph: bool,
        max_depth: usize,
    ) -> anyhow::Result<Vec<SearchResult>> {
        // Produce the query vector.  A single-element batch keeps the
        // provider interface uniform.
        let embeddings = self.provider.embed_batch(vec![query.to_string()]).await?;
        let query_vec = embeddings
            .into_iter()
            .next()
            .unwrap_or_else(|| vec![0.0; self.provider.dim()]);

        let mut hits = self
            .backend
            .hybrid_search(&query_vec, query, top_k)
            .await?;

        if hits.is_empty() {
            return Ok(vec![]);
        }

        // Label quality using relative thresholds against the top score.
        let scores: Vec<f64> = hits.iter().map(|h| h.score).collect();
        let labels = Self::label_match_quality(&scores);
        for (hit, label) in hits.iter_mut().zip(labels) {
            hit.match_quality = label;
            // The backend used a hybrid (vector + FTS) RRF query; the dominant
            // signal for these embeddings-present results is vector similarity.
            hit.why = "vector".to_string();
        }

        if include_graph && max_depth > 0 {
            hits = self.augment_with_graph(hits, max_depth).await?;
        }

        Ok(hits)
    }

    /// Return all indexed chunks, outbound imports, and inbound importers for
    /// `file_path`.  Returns empty arrays for files not in the index — never
    /// an error.
    pub async fn file_context(&self, file_path: &str) -> anyhow::Result<FileContext> {
        let chunks = self.backend.get_chunks_for_file(file_path).await?;
        let imports = self.backend.get_imports(file_path).await?;
        let imported_by = self.backend.get_importers(file_path).await?;
        Ok(FileContext { chunks, imports, imported_by })
    }

    /// Assign quality labels to a slice of scores using relative thresholds:
    ///
    /// - `>= 0.8 × top_score` → `"high"`
    /// - `>= 0.5 × top_score` → `"moderate"`
    /// - otherwise           → `"low"`
    ///
    /// Returns an empty `Vec` when `scores` is empty.
    pub fn label_match_quality(scores: &[f64]) -> Vec<String> {
        let top = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if top == f64::NEG_INFINITY {
            return vec![];
        }
        scores
            .iter()
            .map(|&s| {
                if s >= 0.8 * top {
                    "high"
                } else if s >= 0.5 * top {
                    "moderate"
                } else {
                    "low"
                }
            })
            .map(str::to_string)
            .collect()
    }

    // -- Private helpers -----------------------------------------------------

    /// Extend `hits` with transitive import-graph neighbours up to `max_depth`
    /// hops.  For each primary result file, `traverse_imports` performs a
    /// level-batched BFS; each discovered file's chunks are added as `"graph
    /// (depth N)"` results.  The visited set inside `traverse_imports` handles
    /// cycles.  Files with no chunks are silently skipped.
    async fn augment_with_graph(
        &self,
        mut hits: Vec<SearchResult>,
        max_depth: usize,
    ) -> anyhow::Result<Vec<SearchResult>> {
        let present: std::collections::HashSet<String> =
            hits.iter().map(|h| h.file_path.clone()).collect();

        // Track chunks already represented so we never emit duplicates.
        let mut seen_chunks: std::collections::HashSet<(String, usize)> =
            hits.iter().map(|h| (h.file_path.clone(), h.chunk_idx)).collect();

        for file_path in &present {
            let reachable = self.backend.traverse_imports(file_path, max_depth).await?;
            for target in reachable {
                // `traverse_imports` already de-duplicates across BFS levels,
                // but multiple primary files may independently reach the same
                // target; guard with `seen_chunks` per-chunk.
                let chunks = self.backend.get_chunks_for_file(&target).await?;
                for chunk in chunks {
                    let key = (chunk.file_path.clone(), chunk.chunk_idx);
                    if seen_chunks.insert(key) {
                        hits.push(SearchResult {
                            file_path: chunk.file_path,
                            chunk_idx: chunk.chunk_idx,
                            content: chunk.content,
                            start_line: chunk.start_line,
                            end_line: chunk.end_line,
                            chunk_type: chunk.chunk_type,
                            score: 0.0,
                            match_quality: "low".to_string(),
                            why: format!("graph (depth {max_depth})"),
                        });
                    }
                }
            }
        }

        Ok(hits)
    }
}
