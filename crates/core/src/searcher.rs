use std::sync::Arc;

use crate::{expander::QueryExpander, reranker::{RerankCandidate, Reranker}, ChunkRecord, EmbedProvider, SearchResult, StorageBackend};

// ---------------------------------------------------------------------------
// Public output types
// ---------------------------------------------------------------------------

/// Timing breakdown of search pipeline phases.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct SearchTimings {
    /// Query embedding generation (ms).
    pub embed_ms: u64,
    /// HNSW + BM25 + RRF fusion retrieval (ms).
    pub retrieve_ms: u64,
    /// LLM query expansion (ms, 0 if skipped).
    pub expand_ms: u64,
    /// Cross-encoder reranking (ms, 0 if skipped).
    pub rerank_ms: u64,
    /// Graph augmentation (ms, 0 if disabled).
    pub graph_ms: u64,
    /// End-to-end total (ms).
    pub total_ms: u64,
}

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
// Query expansion
// ---------------------------------------------------------------------------

/// Common English stop words that don't help BM25 matching.
const STOP_WORDS: &[&str] = &[
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "because", "but", "and", "or", "if", "while", "what", "which", "who",
    "whom", "this", "that", "these", "those", "i", "me", "my", "we", "our",
    "you", "your", "he", "him", "his", "she", "her", "it", "its", "they",
    "them", "their", "about",
];

/// Extract significant keywords from a natural language query.
/// Returns the original query plus deduplicated keywords, giving BM25
/// more signal for term matching.
/// Sanitize a query string for CozoDB's FTS mini-language.
///
/// The FTS index uses `tokenizer: Simple, filters: [Lowercase, AlphaNumOnly]`,
/// so indexed tokens are lowercase alphanumeric only.  The query string must
/// match: strip non-alphanumeric characters (replace with space), collapse
/// whitespace, and remove FTS reserved keywords (`AND`, `OR`, `NOT`, `NEAR`)
/// that the LLM expander or user query might inject.
fn sanitize_fts_query(query: &str) -> String {
    // Replace non-alphanumeric, non-whitespace chars with space.
    let cleaned: String = query
        .chars()
        .map(|c| if c.is_alphanumeric() || c.is_whitespace() { c } else { ' ' })
        .collect();

    // Split into tokens, drop FTS reserved keywords and single-char noise.
    cleaned
        .split_whitespace()
        .filter(|t| t.len() > 1)
        .filter(|t| !matches!(t.to_uppercase().as_str(), "AND" | "OR" | "NOT" | "NEAR"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Split camelCase/PascalCase tokens into constituent words.
///
/// Returns the original token plus the split parts.  "superRefine" becomes
/// `["superRefine", "super", "Refine"]`.  Pure-lowercase or pure-uppercase
/// tokens pass through unchanged (returns just the original).
fn split_camel_case(token: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut start = 0;
    let chars: Vec<char> = token.chars().collect();
    for i in 1..chars.len() {
        // Split on lowercase→uppercase boundary (camelCase) or a run of
        // uppercase followed by a lowercase (e.g. "HTTPClient" → "HTTP", "Client").
        let split = chars[i].is_uppercase() && (
            chars[i - 1].is_lowercase()
            || (i + 1 < chars.len() && chars[i + 1].is_lowercase() && chars[i - 1].is_uppercase())
        );
        if split {
            let part: String = chars[start..i].iter().collect();
            if part.len() > 1 {
                parts.push(part);
            }
            start = i;
        }
    }
    let tail: String = chars[start..].iter().collect();
    if tail.len() > 1 {
        parts.push(tail);
    }
    if parts.len() <= 1 {
        // No split occurred or single part — just return the original.
        return vec![token.to_string()];
    }
    let mut result = vec![token.to_string()];
    result.extend(parts);
    result
}

fn expand_query(query: &str) -> String {
    let mut keywords: Vec<String> = Vec::new();
    for word in query.split_whitespace() {
        if word.len() <= 2 || STOP_WORDS.contains(&word.to_lowercase().as_str()) {
            continue;
        }
        // Split camelCase/PascalCase tokens so BM25 can match individual parts.
        for part in split_camel_case(word) {
            if !keywords.contains(&part) {
                keywords.push(part);
            }
        }
    }

    if keywords.is_empty() || keywords.len() == query.split_whitespace().count() {
        return query.to_string();
    }

    format!("{} {}", query, keywords.join(" "))
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
    /// Optional cross-encoder reranker applied after MMR and before the token
    /// budget filter.  `None` skips the stage (backwards-compatible default).
    reranker: Option<Box<dyn Reranker>>,
    /// Optional LLM-based query expander that enriches conceptual queries with
    /// code-vocabulary keywords before BM25 matching.  `None` skips expansion.
    expander: Option<Box<dyn QueryExpander>>,
    /// Whether to apply the log-dampened PageRank score boost.  Enabled by
    /// default; set to `false` via `with_pagerank_boost(false)` to ablate.
    pagerank_boost: bool,
    /// When true, delegate retrieval to `unified_search` which fuses FTS + HNSW +
    /// graph + PageRank in a single Datalog round-trip.  MMR and cross-encoder
    /// reranking still run in Rust after the unified query returns.
    /// Disabled by default; enable via `with_unified_search(true)`.
    use_unified_search: bool,
    /// LRU cache for query embeddings.  Bounded at 256 entries.
    ///
    /// ASSUMPTION: `provider` is fixed at construction time, so cached
    /// vectors always match the current provider's dimension.  If the
    /// provider were hot-swapped this cache would need to be flushed.
    query_embed_cache: std::sync::Mutex<lru::LruCache<String, Vec<f32>>>,
}

impl<B: StorageBackend, P: EmbedProvider> Searcher<B, P> {
    pub fn new(backend: Arc<B>, provider: P) -> Self {
        Self {
            backend,
            provider,
            reranker: None,
            expander: None,
            pagerank_boost: true,
            use_unified_search: false,
            query_embed_cache: std::sync::Mutex::new(
                lru::LruCache::new(std::num::NonZeroUsize::new(256).unwrap()),
            ),
        }
    }

    /// Set whether the log-dampened PageRank score boost is applied.
    /// Pass `false` to disable for ablation benchmarks.
    pub fn with_pagerank_boost(mut self, enabled: bool) -> Self {
        self.pagerank_boost = enabled;
        self
    }

    /// Enable single-query unified retrieval (FTS + HNSW + graph + PageRank in one
    /// Datalog round-trip).  MMR and cross-encoder reranking still run in Rust.
    /// Default: `false` (uses the multi-phase hybrid pipeline for compatibility).
    pub fn with_unified_search(mut self, enabled: bool) -> Self {
        self.use_unified_search = enabled;
        self
    }

    /// Attach an LLM-based query expander.  Called once at construction time;
    /// the expander runs before BM25 keyword extraction for semantic queries.
    pub fn with_expander(mut self, expander: Box<dyn QueryExpander>) -> Self {
        self.expander = Some(expander);
        self
    }

    /// Attach a cross-encoder reranker.  Called once at construction time;
    /// the reranker runs after MMR and before the token-budget filter.
    pub fn with_reranker(mut self, reranker: Box<dyn Reranker>) -> Self {
        self.reranker = Some(reranker);
        self
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
        let (results, _timings) = self
            .search_with_timings(query, top_k, include_graph, max_depth, diversity, max_tokens)
            .await?;
        Ok(results)
    }

    /// Like [`Self::search`] but also returns a [`SearchTimings`] breakdown.
    ///
    /// Use this from callers that need per-phase latency data (e.g. the MCP
    /// server surface).  All other callers should prefer [`Self::search`] to
    /// avoid destructuring the tuple.
    #[tracing::instrument(skip_all, fields(%query, top_k, diversity))]
    pub async fn search_with_timings(
        &self,
        query: &str,
        top_k: usize,
        include_graph: bool,
        max_depth: usize,
        diversity: f32,
        max_tokens: Option<usize>,
    ) -> anyhow::Result<(Vec<SearchResult>, SearchTimings)> {
        let total_start = std::time::Instant::now();
        let mut timings = SearchTimings::default();

        // -- Embedding (cache-aware) -------------------------------------------
        // Check the LRU cache first to avoid a redundant embed API call for
        // repeated or near-identical queries.  Empty queries are not cached.
        let embed_start = std::time::Instant::now();
        let normalized_query = query.trim().to_string();
        let query_vec: Vec<f32> = if normalized_query.is_empty() {
            // Empty query: skip cache, return zero vector.
            vec![0.0; self.provider.dim()]
        } else {
            // Probe cache with a scoped lock — guard must drop before any await.
            let cached = {
                self.query_embed_cache
                    .lock()
                    .expect("query_embed_cache mutex poisoned")
                    .get(&normalized_query)
                    .cloned()
            };
            if let Some(vec) = cached {
                tracing::debug!("query embedding cache hit");
                vec
            } else {
                // Apply model-specific query prefix (e.g. CodeRankEmbed instruction prefix).
                let query_text = match self.provider.query_prefix() {
                    Some(prefix) => format!("{prefix}{normalized_query}"),
                    None => normalized_query.clone(),
                };
                let embeddings = self.provider.embed_batch(vec![query_text]).await?;
                let vec = embeddings
                    .into_iter()
                    .next()
                    .unwrap_or_else(|| vec![0.0; self.provider.dim()]);
                // Re-acquire lock after await to store the result.
                self.query_embed_cache
                    .lock()
                    .expect("query_embed_cache mutex poisoned")
                    .put(normalized_query, vec.clone());
                vec
            }
        };
        timings.embed_ms = embed_start.elapsed().as_millis() as u64;

        // -- LLM query expansion -----------------------------------------------
        // Only runs when an expander is configured and the query looks conceptual.
        // Failures degrade gracefully: expansion is skipped, search proceeds.
        let expand_start = std::time::Instant::now();
        let expanded_keywords_raw = if let Some(ref expander) = self.expander {
            use crate::router::{classify_query, QueryStrategy};
            if classify_query(query) == QueryStrategy::Semantic {
                expander.expand(query).await.unwrap_or_default()
            } else {
                vec![]
            }
        } else {
            vec![]
        };
        // Only charge expand_ms when the expander is configured (i.e. it ran
        // or was at least considered); 0 when no expander is attached.
        if self.expander.is_some() {
            timings.expand_ms = expand_start.elapsed().as_millis() as u64;
        }

        // Filter expansion keywords that have no lexical overlap with the
        // original query — prevents semantic drift from LLM hallucination.
        let query_terms: std::collections::HashSet<String> = query
            .split_whitespace()
            .map(|w| w.to_lowercase())
            .collect();
        let raw_len = expanded_keywords_raw.len();
        let expanded_keywords: Vec<String> = expanded_keywords_raw
            .into_iter()
            .filter(|kw| {
                // Keep keyword if any of its subwords overlap with query terms,
                // or if it's a camelCase/snake_case identifier (likely a real code symbol).
                let kw_lower = kw.to_lowercase();
                let has_overlap = kw_lower.split_whitespace()
                    .any(|part| query_terms.iter().any(|qt| {
                        part.contains(qt.as_str()) || qt.contains(part)
                    }));
                let is_identifier = kw.contains('_') || kw.chars().any(|c| c.is_uppercase());
                has_overlap || is_identifier
            })
            .collect();
        if expanded_keywords.len() < raw_len {
            tracing::debug!(
                dropped = raw_len - expanded_keywords.len(),
                "filtered expansion keywords with no query overlap"
            );
        }

        // Merge LLM keywords into the BM25 query text.
        // Original query terms appear first and get natural BM25 term frequency.
        // Expansion keywords are appended once (lower weight than the original
        // query which may have terms repeated from expand_query()).
        let mut bm25_query = expand_query(query);
        if !expanded_keywords.is_empty() {
            // Only append keywords not already present in the query
            let existing: std::collections::HashSet<String> = bm25_query
                .split_whitespace()
                .map(|w| w.to_lowercase())
                .collect();
            let novel: Vec<&str> = expanded_keywords
                .iter()
                .filter(|kw| !existing.contains(&kw.to_lowercase()))
                .map(|s| s.as_str())
                .collect();
            if !novel.is_empty() {
                bm25_query = format!("{} {}", bm25_query, novel.join(" "));
            }
        }

        // Sanitize before sending to CozoDB FTS — strip dots, hyphens, and
        // other special characters that break the FTS query mini-language.
        let bm25_query = sanitize_fts_query(&bm25_query);

        // -- Hybrid retrieval --------------------------------------------------
        // When `use_unified_search` is true, a single Datalog round-trip handles
        // FTS + HNSW + PageRank boost + optional graph walk together.  The separate
        // pagerank_boost and graph augmentation steps are skipped in that case.
        let retrieve_start = std::time::Instant::now();
        let mut hits = if self.use_unified_search {
            let gd = if include_graph { max_depth } else { 0 };
            self.backend
                .unified_search(&query_vec, &bm25_query, top_k, gd)
                .await?
        } else {
            self.backend
                .hybrid_search(&query_vec, &bm25_query, top_k)
                .await?
        };
        timings.retrieve_ms = retrieve_start.elapsed().as_millis() as u64;

        // PageRank boost: structurally important files get a mild score uplift.
        // Log-dampened to prevent hub files from dominating all queries.
        // Applied before graph augmentation so expanded hits inherit consistent scaling.
        // Gated on `self.pagerank_boost` so benchmarks can ablate this signal.
        if !self.use_unified_search && self.pagerank_boost {
            let file_paths: Vec<&str> = hits.iter().map(|h| h.file_path.as_str()).collect();
            let ranks = self.backend.get_file_ranks(&file_paths).await.unwrap_or_default();
            if !ranks.is_empty() {
                let median = {
                    let mut vals: Vec<f64> =
                        ranks.values().copied().filter(|v| *v > 0.0).collect();
                    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    if vals.is_empty() { 1.0 } else { vals[vals.len() / 2] }
                };
                for hit in &mut hits {
                    if let Some(&pr) = ranks.get(&hit.file_path) {
                        if pr > 0.0 {
                            let boost = 1.0 + 0.3 * (1.0 + pr / median).ln();
                            hit.score *= boost;
                        }
                    }
                }
            }
        }

        if hits.is_empty() {
            timings.total_ms = total_start.elapsed().as_millis() as u64;
            tracing::info!(
                embed_ms = timings.embed_ms,
                retrieve_ms = timings.retrieve_ms,
                expand_ms = timings.expand_ms,
                rerank_ms = timings.rerank_ms,
                graph_ms = timings.graph_ms,
                total_ms = timings.total_ms,
                "search pipeline timings"
            );
            return Ok((vec![], timings));
        }

        // -- Graph augmentation ------------------------------------------------
        // Pull in chunks from files reachable via resolved import edges.
        // Runs after hybrid search (so we have seed hits) but before MMR +
        // reranker (so expanded results participate in filtering).
        if !self.use_unified_search && include_graph && max_depth > 0 {
            let graph_start = std::time::Instant::now();
            let best_score = hits.iter().map(|h| h.score).fold(0.0_f64, f64::max);
            hits = self.augment_with_graph(hits, max_depth, best_score, &query_vec).await?;
            timings.graph_ms = graph_start.elapsed().as_millis() as u64;
        }

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

        // Strong signal detection: when the top result is already clearly
        // dominant (high absolute score with a large gap to the runner-up),
        // the cross-encoder reranker will not change the outcome meaningfully
        // and adds avoidable latency.  Skip both reranking and blending.
        let strong_signal = !hits.is_empty()
            && hits[0].score >= 0.016
            && (hits.len() < 2 || hits[0].score - hits[1].score >= 0.003);

        if strong_signal {
            tracing::debug!(
                top_score = hits[0].score,
                "strong signal detected, skipping reranker"
            );
        }

        // -- Reranking ---------------------------------------------------------
        // Optional cross-encoder reranking for precision improvement.
        // Reranker sees all post-MMR results; it reorders them before the
        // token-budget filter selects the top slice.
        //
        // Fusion score and reranker score are blended by rank position:
        //   positions 0-2  → fusion 25 %, reranker 75 %
        //   positions 3-9  → fusion 40 %, reranker 60 %
        //   positions 10+  → fusion 60 %, reranker 40 %
        // Top results already earned their position via hybrid search, so
        // we lean on fusion; the tail benefits more from reranker signal.
        let hits = if !strong_signal {
            if let Some(ref reranker) = self.reranker {
                let rerank_start = std::time::Instant::now();
                let candidates: Vec<RerankCandidate> = hits
                    .iter()
                    .enumerate()
                    .map(|(i, h)| RerankCandidate { index: i, text: h.content.clone() })
                    .collect();
                let scores = reranker.rerank(query, candidates).await?;
                timings.rerank_ms = rerank_start.elapsed().as_millis() as u64;
                let mut scored: Vec<_> = hits.into_iter().zip(scores).collect();
                // Sort by reranker score descending to establish position weights.
                scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let mut blended: Vec<SearchResult> = scored
                    .into_iter()
                    .enumerate()
                    .map(|(i, (mut hit, rerank_score))| {
                        let fusion_weight =
                            if i < 3 { 0.25_f64 } else if i < 10 { 0.40 } else { 0.60 };
                        let rerank_weight = 1.0 - fusion_weight;
                        hit.score = hit.score * fusion_weight + rerank_score * rerank_weight;
                        hit
                    })
                    .collect();
                // Re-sort by the blended score so final order reflects both signals.
                blended.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
                blended
            } else {
                hits
            }
        } else {
            hits
        };

        // Source-type penalties applied AFTER reranker blending so that the
        // reranker's relevance signal isn't overriding these structural priors.
        // Test files rarely answer non-test queries; doc/prose files match NL
        // queries deceptively well but aren't "the implementation".
        let mut hits = hits;
        for hit in &mut hits {
            if is_test_file(&hit.file_path) {
                hit.score *= 0.3;
            } else if is_doc_file(&hit.file_path) {
                hit.score *= 0.5;
            } else if is_barrel_file(&hit.file_path) {
                hit.score *= 0.4;
            }
        }
        hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        // Label quality using relative thresholds against the post-penalty
        // top score so labels reflect final ranking.
        let scores: Vec<f64> = hits.iter().map(|h| h.score).collect();
        let labels = Self::label_match_quality(&scores);
        for (hit, label) in hits.iter_mut().zip(labels) {
            hit.match_quality = label;
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

        timings.total_ms = total_start.elapsed().as_millis() as u64;
        tracing::info!(
            embed_ms = timings.embed_ms,
            retrieve_ms = timings.retrieve_ms,
            expand_ms = timings.expand_ms,
            rerank_ms = timings.rerank_ms,
            graph_ms = timings.graph_ms,
            total_ms = timings.total_ms,
            "search pipeline timings"
        );
        Ok((hits, timings))
    }

    /// Return all indexed chunks, outbound imports, and inbound importers for
    /// `file_path`.  Returns empty arrays for files not in the index — never
    /// an error.
    pub async fn file_context(&self, file_path: &str) -> anyhow::Result<FileContext> {
        let (chunks_res, imports_res, importers_res) = tokio::join!(
            self.backend.get_chunks_for_file(file_path),
            self.backend.get_imports(file_path),
            self.backend.get_importers(file_path),
        );
        let chunks = chunks_res?;
        let imports = imports_res?;
        let imported_by = importers_res?;
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

    /// Extend `hits` with chunks from files reachable via resolved import edges.
    ///
    /// Graph-expanded results are scored using `cosine_sim × 0.7^depth × best_score`.
    /// Chunks with cosine similarity < 0.25 to the query are dropped to prevent result
    /// flooding from tangentially related imported files.
    async fn augment_with_graph(
        &self,
        mut hits: Vec<SearchResult>,
        max_depth: usize,
        best_score: f64,
        query_vec: &[f32],
    ) -> anyhow::Result<Vec<SearchResult>> {
        let present: std::collections::HashSet<String> =
            hits.iter().map(|h| h.file_path.clone()).collect();

        let mut seen_chunks: std::collections::HashSet<(String, usize)> =
            hits.iter().map(|h| (h.file_path.clone(), h.chunk_idx)).collect();

        // Capture seed chunks from the original hits before any augmentation.
        // Cap at 10 to limit HNSW graph traversal cost.
        let seed_chunks: Vec<(String, usize)> = hits
            .iter()
            .take(10)
            .map(|h| (h.file_path.clone(), h.chunk_idx))
            .collect();

        // Phase 1: Collect all graph-reachable files with their minimum depth.
        // Multiple seed files may reach the same target; keep the shortest path.
        let mut depth_map: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for file_path in &present {
            let reachable = self.backend.traverse_imports(file_path, max_depth, None).await?;
            for (target, depth) in reachable {
                depth_map.entry(target).and_modify(|d| { if depth < *d { *d = depth; } }).or_insert(depth);
            }
        }

        // Phase 2: HNSW proximity expansion — collect specific (fp, ci) targets.
        // Walks layer 0 of the HNSW graph using seed chunks from the original hits.
        // Cosine distance threshold 0.3: captures semantically close neighbors
        // while excluding weakly related chunks.
        let mut hnsw_map: std::collections::HashMap<(String, usize), f64> = std::collections::HashMap::new();
        if !seed_chunks.is_empty() {
            let neighbors = self.backend.hnsw_neighbors(&seed_chunks, 0.3, 50).await?;
            for (fp, ci, dist) in neighbors {
                // Keep closest distance (smallest dist = most similar) for each chunk.
                hnsw_map.entry((fp, ci)).and_modify(|d| { if dist < *d { *d = dist; } }).or_insert(dist);
            }
        }

        // Phase 3: Batch fetch ALL chunks for all graph-reachable files in one query.
        // HNSW files that are also graph-reachable get all chunks fetched here too.
        // Pure HNSW files (not in depth_map) are omitted intentionally — we only
        // need the specific (fp, ci) chunk, so we avoid pulling every chunk for those files.
        let graph_file_list: Vec<&str> = depth_map.keys().map(|s| s.as_str()).collect();
        let all_graph_chunks = self.backend.get_chunks_for_files(&graph_file_list).await?;

        // Phase 4a: Distribute graph chunks filtered by query relevance.
        // Chunks with cosine similarity < 0.25 are discarded to avoid flooding results
        // with tangentially related code from imported files.
        // Score = best_score × sim × 0.7^depth (closer import + more relevant = higher score).
        let pre_graph_count = hits.len();
        const GRAPH_SIM_THRESHOLD: f64 = 0.25;
        for chunk in all_graph_chunks {
            let key = (chunk.file_path.clone(), chunk.chunk_idx);
            if !seen_chunks.insert(key) {
                continue;
            }
            // Skip chunks without embeddings — cannot assess relevance.
            let emb = match &chunk.embedding {
                Some(e) if !e.is_empty() => e,
                _ => continue,
            };
            let sim = cosine_sim(query_vec, emb) as f64;
            if sim < GRAPH_SIM_THRESHOLD {
                continue;
            }
            let depth = *depth_map.get(&chunk.file_path).unwrap_or(&1);
            let score = (best_score * sim * 0.7_f64.powi(depth as i32)).min(best_score * 0.5);
            hits.push(SearchResult {
                file_path: chunk.file_path,
                chunk_idx: chunk.chunk_idx,
                content: chunk.content,
                start_line: chunk.start_line,
                end_line: chunk.end_line,
                chunk_type: chunk.chunk_type,
                score,
                match_quality: "graph".to_string(),
                why: "graph".to_string(),
            });
        }

        // Phase 4b: Fetch and score HNSW-only chunks (specific (fp, ci) pairs not already seen).
        // These are chunks in files that are NOT graph-reachable — pure proximity neighbors.
        // We only need one specific chunk per HNSW hit, so fetch file-by-file here.
        // In practice hnsw_map is small (≤50 entries) so this is cheap.
        let hnsw_only: Vec<(&String, usize, f64)> = hnsw_map
            .iter()
            .filter(|((fp, ci), _)| {
                // Skip if already added (either as a seed or via graph augmentation).
                !seen_chunks.contains(&(fp.clone(), *ci))
                    // Also skip if the file is graph-reachable — that fetch is already done.
                    && !depth_map.contains_key(fp)
            })
            .map(|((fp, ci), dist)| (fp, *ci, *dist))
            .collect();

        // Group HNSW-only targets by file to minimise repeated backend calls.
        let mut hnsw_by_file: std::collections::HashMap<&String, Vec<(usize, f64)>> = std::collections::HashMap::new();
        for (fp, ci, dist) in hnsw_only {
            hnsw_by_file.entry(fp).or_default().push((ci, dist));
        }
        for (fp, targets) in hnsw_by_file {
            if let Ok(chunks) = self.backend.get_chunks_for_file(fp).await {
                for chunk in chunks {
                    if let Some((_, dist)) = targets.iter().find(|(ci, _)| *ci == chunk.chunk_idx) {
                        let key = (chunk.file_path.clone(), chunk.chunk_idx);
                        if seen_chunks.insert(key) {
                            // Apply same relevance gate as Phase 4a.
                            if let Some(ref emb) = chunk.embedding {
                                if !emb.is_empty() {
                                    let sim = cosine_sim(query_vec, emb) as f64;
                                    if sim < GRAPH_SIM_THRESHOLD {
                                        continue;
                                    }
                                }
                            }
                            // Score proportional to proximity: closer neighbor → higher score.
                            // Capped at 0.4× best_score to rank below import-graph results.
                            let prox_score = best_score * 0.4 * (1.0_f64 - dist).max(0.0);
                            hits.push(SearchResult {
                                file_path: chunk.file_path,
                                chunk_idx: chunk.chunk_idx,
                                content: chunk.content,
                                start_line: chunk.start_line,
                                end_line: chunk.end_line,
                                chunk_type: chunk.chunk_type,
                                score: prox_score,
                                match_quality: "hnsw_proximity".to_string(),
                                why: "hnsw_proximity".to_string(),
                            });
                        }
                    }
                }
            }
        }

        // Cap graph-added results to top 20 to prevent result flooding.
        // Sort the graph portion by score descending before truncating so the
        // highest-relevance graph chunks survive.
        if hits.len() > pre_graph_count + 20 {
            let graph_part = &mut hits[pre_graph_count..];
            graph_part.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
            hits.truncate(pre_graph_count + 20);
        }

        tracing::debug!(
            seed_files = present.len(),
            hnsw_seeds = seed_chunks.len(),
            graph_files = depth_map.len(),
            total_hits = hits.len(),
            "graph augmentation complete"
        );

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


// ---------------------------------------------------------------------------
// File-type detection (test, doc/prose)
// ---------------------------------------------------------------------------

/// Returns `true` when `path` looks like a test or spec file.
///
/// Matches on (case-insensitive):
/// - Directory components: `/tests/`, `/test/`, `/__tests__/`, `/spec/`,
///   `/specs/`, `/testing/`, `/testutil/`, `/test_utils/`, `/testdata/`
/// - File stem ending in `.test`, `.spec`, `_test`, or `_spec` before
///   the final extension (e.g. `foo.test.ts`, `bar_spec.rb`).
/// - Exact file names: `test.rs`, `test.py`, `test.go`, `test.ts`, `test.js`.
fn is_test_file(path: &str) -> bool {
    let lower = path.to_lowercase();
    // Normalise separators so Windows paths work too.
    let norm = lower.replace('\\', "/");

    // Directory-based patterns.
    const TEST_DIRS: &[&str] = &[
        "/tests/", "/test/", "/__tests__/", "/spec/", "/specs/",
        "/testing/", "/testutil/", "/test_utils/", "/testdata/",
    ];
    if TEST_DIRS.iter().any(|d| norm.contains(d)) {
        return true;
    }
    // Root-level test directories: strip leading '/' and check starts_with.
    if TEST_DIRS.iter().any(|d| norm.starts_with(d.trim_start_matches('/'))) {
        return true;
    }

    // Isolate the file name (everything after the last '/').
    let file_name = norm.rsplit('/').next().unwrap_or(&norm);

    // Exact file names that are test entry-points by convention.
    const EXACT_NAMES: &[&str] = &[
        "test.rs", "test.py", "test.go", "test.ts", "test.js",
    ];
    if EXACT_NAMES.contains(&file_name) {
        return true;
    }

    // Stem-based patterns: strip the final extension (after last '.')
    // and check if the remainder ends with a test/spec suffix.
    // E.g. "foo.test.ts" → stem = "foo.test"; "bar_spec.rb" → "bar_spec".
    let stem = match file_name.rfind('.') {
        Some(dot) => &file_name[..dot],
        None => file_name,
    };
    stem.ends_with(".test")
        || stem.ends_with(".spec")
        || stem.ends_with("_test")
        || stem.ends_with("_spec")
}

/// Returns `true` when `path` looks like a documentation or prose file.
///
/// Matches on (case-insensitive):
/// - Extensions: `.md`, `.mdx`, `.rst`, `.adoc`, `.txt`, `.org`
/// - Exact file names (stem): `README`, `CHANGELOG`, `CHANGES`,
///   `HISTORY`, `NEWS`, `AUTHORS`, `CONTRIBUTORS`, `LICENSE`, `LICENCE`
fn is_doc_file(path: &str) -> bool {
    let lower = path.to_lowercase();
    let norm = lower.replace('\\', "/");

    // Extension-based patterns.
    const DOC_EXTS: &[&str] = &[".md", ".mdx", ".rst", ".adoc", ".txt", ".org"];
    if DOC_EXTS.iter().any(|e| norm.ends_with(e)) {
        return true;
    }

    // Well-known prose filenames (with or without extension).
    let file_name = norm.rsplit('/').next().unwrap_or(&norm);
    let stem = match file_name.rfind('.') {
        Some(dot) => &file_name[..dot],
        None => file_name,
    };
    matches!(
        stem,
        "readme" | "changelog" | "changes" | "history" | "news"
            | "authors" | "contributors" | "license" | "licence"
    )
}

/// Returns `true` when `path` looks like a barrel/re-export file.
///
/// Barrel files primarily re-export symbols from other modules without
/// adding implementation. They contain every public symbol name, giving
/// them artificially high BM25 keyword density.
///
/// Matches on (case-insensitive):
/// - File names: `index.ts`, `index.js`, `index.mjs`, `mod.rs`, `__init__.py`
/// - File stems: `barrel`, `exports`
fn is_barrel_file(path: &str) -> bool {
    let lower = path.to_lowercase();
    let norm = lower.replace('\\', "/");
    let file_name = norm.rsplit('/').next().unwrap_or(&norm);
    matches!(
        file_name,
        "index.ts" | "index.js" | "index.jsx" | "index.tsx" | "index.mjs" | "index.cjs"
            | "mod.rs" | "__init__.py"
            | "barrel.ts" | "barrel.js" | "exports.ts" | "exports.js"
    )
}

#[cfg(test)]
mod tests {
    use super::{is_test_file, is_doc_file, is_barrel_file, sanitize_fts_query, expand_query, split_camel_case};

    #[test]
    fn test_is_test_file_positive_dir_patterns() {
        // Directory segment matches.
        assert!(is_test_file("src/tests/foo.rs"));
        assert!(is_test_file("src/test/bar.py"));
        assert!(is_test_file("src/__tests__/baz.ts"));
        assert!(is_test_file("src/spec/qux.rb"));
        assert!(is_test_file("src/specs/qux.rb"));
        assert!(is_test_file("src/testing/helpers.go"));
        assert!(is_test_file("src/testutil/mock.go"));
        assert!(is_test_file("src/test_utils/fixtures.py"));
        assert!(is_test_file("src/testdata/sample.json"));
    }

    #[test]
    fn test_is_test_file_positive_exact_names() {
        // Exact file names.
        assert!(is_test_file("src/test.rs"));
        assert!(is_test_file("src/test.py"));
        assert!(is_test_file("src/test.go"));
        assert!(is_test_file("src/test.ts"));
        assert!(is_test_file("src/test.js"));
        // Case-insensitive.
        assert!(is_test_file("src/TEST.RS"));
    }

    #[test]
    fn test_is_test_file_positive_stem_suffixes() {
        // Stem-based patterns.
        assert!(is_test_file("src/foo.test.ts"));
        assert!(is_test_file("src/foo.spec.ts"));
        assert!(is_test_file("src/bar_test.go"));
        assert!(is_test_file("src/bar_spec.rb"));
        assert!(is_test_file("src/baz.test.js"));
        // Case-insensitive stem.
        assert!(is_test_file("src/Foo.Test.Ts"));
    }

    #[test]
    fn test_is_test_file_negative() {
        // Production files that merely contain the word "test" in the path
        // but not as a directory component or stem suffix.
        assert!(!is_test_file("src/contest/rules.rs"));
        assert!(!is_test_file("src/attestation.rs"));
        assert!(!is_test_file("src/latest/version.ts"));
        assert!(!is_test_file("src/searcher.rs"));
        assert!(!is_test_file("src/router.rs"));
        assert!(!is_test_file("README.md"));
        // File named "tester.rs" — stem is "tester", not a match.
        assert!(!is_test_file("src/tester.rs"));
    }

    // --- sanitize_fts_query tests ---

    #[test]
    fn sanitize_strips_dots_and_hyphens() {
        // LLM expansion produces "JSON.stringify" and "key-value" which
        // break CozoDB FTS query parsing.
        assert_eq!(
            sanitize_fts_query("JSON.stringify key-value"),
            "JSON stringify key value"
        );
    }

    #[test]
    fn sanitize_removes_fts_reserved_keywords() {
        // "AND", "OR", "NOT" are CozoDB FTS operators.
        assert_eq!(
            sanitize_fts_query("hello AND world OR bye NOT gone"),
            "hello world bye gone"
        );
        // Case-insensitive matching of reserved words.
        assert_eq!(
            sanitize_fts_query("find and connect or disconnect"),
            "find connect disconnect"
        );
    }

    #[test]
    fn sanitize_preserves_normal_queries() {
        assert_eq!(
            sanitize_fts_query("error handling middleware"),
            "error handling middleware"
        );
    }

    #[test]
    fn sanitize_handles_special_chars_from_code() {
        // Parens, brackets, asterisks, carets, slashes — all FTS syntax.
        assert_eq!(
            sanitize_fts_query("foo() => bar[0] + baz*"),
            "foo bar baz"
        );
    }

    #[test]
    fn sanitize_collapses_whitespace() {
        assert_eq!(
            sanitize_fts_query("  hello   world  "),
            "hello world"
        );
    }

    #[test]
    fn sanitize_drops_single_char_tokens() {
        // After stripping punctuation, leftover single chars are noise.
        assert_eq!(
            sanitize_fts_query("a + b = c"),
            ""
        );
    }

    #[test]
    fn sanitize_near_with_slash() {
        // NEAR/3(...) is FTS syntax — slashes and parens stripped,
        // NEAR removed as reserved.
        assert_eq!(
            sanitize_fts_query("NEAR/3(hello world)"),
            "hello world"
        );
    }

    // --- expand_query tests ---

    #[test]
    fn expand_query_deduplicates_keywords() {
        let result = expand_query("how does error handling work");
        // "how", "does", "work" are stop words; only "error" and "handling" extracted.
        assert!(result.contains("error"));
        assert!(result.contains("handling"));
    }

    #[test]
    fn expand_query_noop_for_all_keywords() {
        // When every word is already a keyword (no stop words removed),
        // the query passes through unchanged.
        let result = expand_query("error handling middleware");
        assert_eq!(result, "error handling middleware");
    }

    #[test]
    fn expand_query_splits_camel_case() {
        let result = expand_query("find superRefine usage");
        assert!(result.contains("superRefine"), "should keep original token");
        assert!(result.contains("super"), "should split camelCase");
        assert!(result.contains("Refine"), "should split camelCase");
    }

    #[test]
    fn expand_query_splits_pascal_case() {
        let result = expand_query("what is AsyncClient");
        assert!(result.contains("AsyncClient"));
        assert!(result.contains("Async"));
        assert!(result.contains("Client"));
    }

    // --- split_camel_case tests ---

    #[test]
    fn split_camel_case_simple() {
        assert_eq!(split_camel_case("superRefine"), vec!["superRefine", "super", "Refine"]);
    }

    #[test]
    fn split_camel_case_pascal() {
        assert_eq!(split_camel_case("AsyncClient"), vec!["AsyncClient", "Async", "Client"]);
    }

    #[test]
    fn split_camel_case_acronym() {
        assert_eq!(split_camel_case("HTTPClient"), vec!["HTTPClient", "HTTP", "Client"]);
    }

    #[test]
    fn split_camel_case_all_lower() {
        // No split — returns just the original.
        assert_eq!(split_camel_case("hello"), vec!["hello"]);
    }

    #[test]
    fn split_camel_case_all_upper() {
        assert_eq!(split_camel_case("HTTP"), vec!["HTTP"]);
    }

    #[test]
    fn split_camel_case_three_parts() {
        assert_eq!(
            split_camel_case("getFileContext"),
            vec!["getFileContext", "get", "File", "Context"]
        );
    }

    // --- is_doc_file tests ---

    #[test]
    fn doc_file_markdown() {
        assert!(is_doc_file("README.md"));
        assert!(is_doc_file("docs/api.md"));
        assert!(is_doc_file("CHANGELOG.md"));
        assert!(is_doc_file("src/docs/guide.mdx"));
    }

    #[test]
    fn doc_file_in_docs_dir_still_needs_doc_extension() {
        // A .py file in docs/ is code, not prose — should NOT be penalized.
        assert!(!is_doc_file("docs/tutorial.py"));
        assert!(!is_doc_file("doc/reference.rs"));
        // But markdown in docs/ IS a doc file.
        assert!(is_doc_file("docs/guide.md"));
        assert!(is_doc_file("doc/api.rst"));
    }

    #[test]
    fn doc_file_well_known_stems() {
        assert!(is_doc_file("README"));
        assert!(is_doc_file("CHANGELOG"));
        assert!(is_doc_file("LICENSE"));
        assert!(is_doc_file("AUTHORS.txt"));
        assert!(is_doc_file("CONTRIBUTORS.md"));
    }

    #[test]
    fn doc_file_negative() {
        // Source code is not a doc file.
        assert!(!is_doc_file("src/parser.rs"));
        assert!(!is_doc_file("src/index.ts"));
        assert!(!is_doc_file("httpx/_client.py"));
        // Test files are not doc files.
        assert!(!is_doc_file("tests/test_auth.py"));
        // File with 'doc' in the name but not in a docs/ directory.
        assert!(!is_doc_file("src/docstring.rs"));
    }

    // --- is_barrel_file tests ---

    #[test]
    fn barrel_file_positive() {
        assert!(is_barrel_file("src/index.ts"));
        assert!(is_barrel_file("src/index.js"));
        assert!(is_barrel_file("src/mod.rs"));
        assert!(is_barrel_file("httpx/__init__.py"));
        assert!(is_barrel_file("src/barrel.ts"));
        assert!(is_barrel_file("src/exports.js"));
    }

    #[test]
    fn barrel_file_negative() {
        assert!(!is_barrel_file("src/index.html"));
        assert!(!is_barrel_file("src/main.rs"));
        assert!(!is_barrel_file("src/client.ts"));
        assert!(!is_barrel_file("src/models.py"));
    }
}