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
    /// - `top_k` — maximum number of primary results.
    /// - `include_graph` — when `true`, augment results with one-hop import
    ///   neighbours, annotating them with `why = "imports <target>"`.
    ///
    /// Returns an empty `Vec` when no results match; never returns an error
    /// for a zero-result query.
    pub async fn search(
        &self,
        query: &str,
        top_k: usize,
        include_graph: bool,
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

        if include_graph {
            hits = self.augment_with_graph(hits, top_k).await?;
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

    /// Extend `hits` with one-hop import-graph neighbours.
    ///
    /// For each unique file in the primary result set, fetch the files it
    /// imports and the files that import it.  Add synthetic results for any
    /// neighbour not already present, annotated with
    /// `why = "imports <original_file>"`.
    async fn augment_with_graph(
        &self,
        mut hits: Vec<SearchResult>,
        top_k: usize,
    ) -> anyhow::Result<Vec<SearchResult>> {
        let present: std::collections::HashSet<String> =
            hits.iter().map(|h| h.file_path.clone()).collect();

        let mut neighbours: Vec<SearchResult> = Vec::new();

        for file_path in &present {
            // Outbound: files this file imports.
            for target in self.backend.get_imports(file_path).await? {
                if !present.contains(&target) {
                    let chunks = self.backend.get_chunks_for_file(&target).await?;
                    for chunk in chunks {
                        neighbours.push(SearchResult {
                            file_path: chunk.file_path,
                            chunk_idx: chunk.chunk_idx,
                            content: chunk.content,
                            start_line: chunk.start_line,
                            end_line: chunk.end_line,
                            chunk_type: chunk.chunk_type,
                            score: 0.0,
                            match_quality: "low".to_string(),
                            why: format!("imports {file_path}"),
                        });
                    }
                }
            }
            // Inbound: files that import this file.
            for importer in self.backend.get_importers(file_path).await? {
                if !present.contains(&importer) {
                    let chunks = self.backend.get_chunks_for_file(&importer).await?;
                    for chunk in chunks {
                        neighbours.push(SearchResult {
                            file_path: chunk.file_path,
                            chunk_idx: chunk.chunk_idx,
                            content: chunk.content,
                            start_line: chunk.start_line,
                            end_line: chunk.end_line,
                            chunk_type: chunk.chunk_type,
                            score: 0.0,
                            match_quality: "low".to_string(),
                            why: format!("imports {file_path}"),
                        });
                    }
                }
            }
        }

        // Deduplicate neighbours by (file_path, chunk_idx).
        let mut seen_chunks: std::collections::HashSet<(String, usize)> =
            hits.iter().map(|h| (h.file_path.clone(), h.chunk_idx)).collect();
        for n in neighbours {
            let key = (n.file_path.clone(), n.chunk_idx);
            if seen_chunks.insert(key) {
                hits.push(n);
            }
        }

        // Respect the caller's top_k for the non-graph portion; graph hits are
        // additive.  We've already trimmed the base results to top_k in the
        // backend, so just return everything here.
        let _ = top_k; // retained for documentation; base is already bounded
        Ok(hits)
    }
}
