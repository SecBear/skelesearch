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
    #[tracing::instrument(skip_all, fields(%query, top_k, diversity))]
    pub async fn search(
        &self,
        query: &str,
        top_k: usize,
        include_graph: bool,
        max_depth: usize,
        diversity: f32,
        max_tokens: Option<usize>,
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
            // why is set by hybrid_search: "fts", "vector", or "hybrid"
        }

        // Graph augmentation disabled in v1.2: import edges store raw tree-sitter
        // capture text (e.g. "use crate::foo::bar;"), not resolved file paths.
        // traverse_imports matches against file paths and always returns empty.
        // See AD-3 in docs/superpowers/plans/2026-03-18-skelesearch-v1.2-production.md
        // for the planned identifier-based dependency graph approach.
        // The include_graph parameter is accepted but currently a no-op.
        let _ = (include_graph, max_depth);

        // MMR re-ranking: clamp diversity to [0, 1]; skip if 0 or only one result.
        let diversity = diversity.clamp(0.0, 1.0);
        if diversity > 0.0 && hits.len() > 1 {
            let keys: Vec<(String, usize)> = hits
                .iter()
                .map(|h| (h.file_path.clone(), h.chunk_idx))
                .collect();
            let result_vecs = self.backend.get_chunk_embeddings(&keys).await?;
            hits = mmr_rerank(hits, &query_vec, &result_vecs, 1.0 - diversity);
        }

        // Apply token budget if specified.  Results are already scored
        // highest-first; greedily include until the budget is exhausted.
        // The first result is always included even if it alone exceeds the
        // budget — callers must never receive an empty set when hits exist.
        let hits = if let Some(budget) = max_tokens {
            let mut total = 0usize;
            let mut count = 0usize;
            hits.into_iter()
                .take_while(move |r| {
                    total += r.content.len() / 4; // approximate: 1 token ~ 4 chars
                    count += 1;
                    count == 1 || total <= budget
                })
                .collect()
        } else {
            hits
        };

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
    /// hops.  Retained for the v2 identifier-based dependency graph approach.
    /// Currently a no-op in callers: import edges store raw tree-sitter capture
    /// text (e.g. `"use crate::foo::bar;"`), not resolved file paths, so
    /// `traverse_imports` always returns empty.
    /// See AD-3 in docs/superpowers/plans/2026-03-18-skelesearch-v1.2-production.md.
    #[allow(dead_code)]
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

// ---------------------------------------------------------------------------
// MMR re-ranking
// ---------------------------------------------------------------------------

/// Cosine similarity between two vectors.  Returns 0.0 for zero vectors.
fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Maximal Marginal Relevance re-ranking.
///
/// Re-orders `results` to balance relevance against diversity.
/// `lambda` in [0, 1]: 1.0 = pure relevance (no change), 0.0 = pure diversity.
/// `query_vec` is the embedded query; `result_vecs` are embeddings for each result
/// in the same order as `results`.
fn mmr_rerank(
    results: Vec<SearchResult>,
    query_vec: &[f32],
    result_vecs: &[Vec<f32>],
    lambda: f32,
) -> Vec<SearchResult> {
    let n = results.len();
    let mut selected: Vec<usize> = Vec::with_capacity(n);
    let mut candidates: Vec<usize> = (0..n).collect();

    while !candidates.is_empty() {
        let best = candidates
            .iter()
            .copied()
            .max_by(|&i, &j| {
                let score_i = mmr_score(i, &selected, query_vec, result_vecs, lambda);
                let score_j = mmr_score(j, &selected, query_vec, result_vecs, lambda);
                score_i.partial_cmp(&score_j).unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("candidates is non-empty");
        candidates.retain(|&c| c != best);
        selected.push(best);
    }

    selected.into_iter().map(|i| results[i].clone()).collect()
}

/// MMR score for candidate `i` given already-selected items.
fn mmr_score(
    i: usize,
    selected: &[usize],
    query_vec: &[f32],
    result_vecs: &[Vec<f32>],
    lambda: f32,
) -> f32 {
    let relevance = cosine_sim(query_vec, &result_vecs[i]);
    let redundancy = selected
        .iter()
        .map(|&s| cosine_sim(&result_vecs[i], &result_vecs[s]))
        .fold(f32::NEG_INFINITY, f32::max);
    // When no items have been selected yet, redundancy is -inf; treat as 0.
    let redundancy = if redundancy == f32::NEG_INFINITY { 0.0 } else { redundancy };
    lambda * relevance - (1.0 - lambda) * redundancy
}
