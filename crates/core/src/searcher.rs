use std::sync::Arc;

use crate::{expander::QueryExpander, reranker::{RerankCandidate, Reranker}, ChunkRecord, EmbedProvider, SearchResult, StorageBackend};

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
}

impl<B: StorageBackend, P: EmbedProvider> Searcher<B, P> {
    pub fn new(backend: Arc<B>, provider: P) -> Self {
        Self { backend, provider, reranker: None, expander: None }
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
        // Produce the query vector.  A single-element batch keeps the
        // provider interface uniform.
        let embeddings = self.provider.embed_batch(vec![query.to_string()]).await?;
        let query_vec = embeddings
            .into_iter()
            .next()
            .unwrap_or_else(|| vec![0.0; self.provider.dim()]);

        // LLM-based query expansion for conceptual queries.
        // Only runs when an expander is configured and the query looks conceptual.
        // Failures degrade gracefully: expansion is skipped, search proceeds.
        let expanded_keywords = if let Some(ref expander) = self.expander {
            use crate::router::{classify_query, QueryStrategy};
            if classify_query(query) == QueryStrategy::Semantic {
                expander.expand(query).await.unwrap_or_default()
            } else {
                vec![]
            }
        } else {
            vec![]
        };

        // Merge LLM keywords into the BM25 query text.
        let mut bm25_query = expand_query(query);
        if !expanded_keywords.is_empty() {
            bm25_query = format!("{} {}", bm25_query, expanded_keywords.join(" "));
        }

        // Sanitize before sending to CozoDB FTS — strip dots, hyphens, and
        // other special characters that break the FTS query mini-language.
        let bm25_query = sanitize_fts_query(&bm25_query);

        let mut hits = self
            .backend
            .hybrid_search(&query_vec, &bm25_query, top_k)
            .await?;

        if hits.is_empty() {
            return Ok(vec![]);
        }

        // Graph augmentation: pull in chunks from files reachable via resolved
        // import edges.  Runs after hybrid search (so we have seed hits) but
        // before MMR + reranker (so expanded results participate in filtering).
        if include_graph && max_depth > 0 {
            let best_score = hits.iter().map(|h| h.score).fold(0.0_f64, f64::max);
            hits = self.augment_with_graph(hits, max_depth, best_score).await?;
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
            && hits[0].score >= 0.8
            && (hits.len() < 2 || hits[0].score - hits[1].score >= 0.15);

        if strong_signal {
            tracing::debug!(
                top_score = hits[0].score,
                "strong signal detected, skipping reranker"
            );
        }

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
                let candidates: Vec<RerankCandidate> = hits
                    .iter()
                    .enumerate()
                    .map(|(i, h)| RerankCandidate { index: i, text: h.content.clone() })
                    .collect();
                let scores = reranker.rerank(query, candidates).await?;
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

    /// Extend `hits` with chunks from files reachable via resolved import edges.
    ///
    /// Graph-expanded results receive a depth-decayed score: `best_score * 0.5^depth`.
    /// This ensures they rank below direct hits but above noise, and participate
    /// meaningfully in downstream MMR and reranker stages.
    async fn augment_with_graph(
        &self,
        mut hits: Vec<SearchResult>,
        max_depth: usize,
        best_score: f64,
    ) -> anyhow::Result<Vec<SearchResult>> {
        let present: std::collections::HashSet<String> =
            hits.iter().map(|h| h.file_path.clone()).collect();

        let mut seen_chunks: std::collections::HashSet<(String, usize)> =
            hits.iter().map(|h| (h.file_path.clone(), h.chunk_idx)).collect();

        // BFS traversal depth is handled by traverse_imports.  We assign scores
        // as if all reachable files are at depth 1 for simplicity — the traversal
        // already bounds total hops via max_depth.
        let graph_score = best_score * 0.5;

        for file_path in &present {
            let reachable = self.backend.traverse_imports(file_path, max_depth).await?;
            for target in reachable {
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
                            score: graph_score,
                            match_quality: "graph".to_string(),
                            why: "graph".to_string(),
                        });
                    }
                }
            }
        }

        tracing::debug!(
            seed_files = present.len(),
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

#[cfg(test)]
mod tests {
    use super::{is_test_file, is_doc_file, sanitize_fts_query, expand_query, split_camel_case};

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
}