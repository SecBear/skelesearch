// Tool input and output types for the skelesearch MCP server.
//
// Input types must be `Deserialize + JsonSchema` so rmcp can generate tool
// schemas and deserialise caller-supplied arguments.
// Output types must be `Serialize` so they can be serialised to JSON strings
// returned to the client.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

/// Input for the `search_code` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchCodeInput {
    /// Natural-language or keyword query to search for.
    pub query: String,
    /// Maximum number of results to return (default: 5).
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// When true, augment results with transitive import-graph neighbours.
    #[serde(default)]
    pub include_graph: bool,
    /// Maximum graph traversal depth (default: 2 when include_graph is true).
    #[serde(default)]
    pub max_depth: Option<usize>,
    /// Diversity factor for MMR re-ranking (0.0–1.0). 0.0 = pure relevance,
    /// higher values reduce redundancy. Default: 0.3.
    #[serde(default = "default_diversity")]
    pub diversity: f32,
    /// Maximum token budget for results. Omit for unlimited.
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// When true, scope results to files changed on the current git branch.
    #[serde(default)]
    pub branch_scope: bool,
    /// Optional session ID for result deduplication across searches.
    /// When set, results seen in previous searches with the same session ID
    /// are deprioritized (moved to bottom of results).
    #[serde(default)]
    pub session_id: Option<String>,
}

fn default_top_k() -> usize {
    5
}

fn default_diversity() -> f32 {
    0.3
}

/// Input for the `index_codebase` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct IndexCodebaseInput {
    /// Absolute or relative path to the root directory to index.
    pub path: String,
    /// Embedding provider to use. Supported: `"fastembed"`, `"openai"`, `"voyage"`.
    /// When omitted, runtime defaults prefer `voyage`/`openai` if their API keys
    /// are present, otherwise local `fastembed`.
    pub provider: Option<String>,
}

/// Input for the `index_status` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct IndexStatusInput {
    /// Optional path filter (reserved for future use; ignored in v1).
    pub path: Option<String>,
}

/// Input for the `get_file_context` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetFileContextInput {
    /// File path to retrieve context for.
    pub file_path: String,
}

// ---------------------------------------------------------------------------
// Outputs
// ---------------------------------------------------------------------------

/// A single search result row.
#[derive(Debug, Serialize, JsonSchema)]
pub struct SearchCodeRow {
    pub file_path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub content: String,
    pub score: f64,
    /// Relative quality label: `"high"`, `"moderate"`, or `"low"`.
    pub match_quality: String,
    /// Retrieval provenance: `"vector"`, `"fts"`, `"hybrid"`, `"graph"`, or `"hnsw_proximity"`.
    pub why: String,
}

/// Response envelope for the `search_code` tool, including per-phase timings.
#[derive(Debug, Serialize, JsonSchema)]
pub struct SearchCodeResponse {
    pub results: Vec<SearchCodeRow>,
    /// Per-phase latency breakdown of the search pipeline.
    /// Hidden from both JSON schema and wire output — exposed only via tracing.
    #[serde(skip)]
    #[schemars(skip)]
    pub _timings: skelesearch_core::SearchTimings,
}

/// Output of the `index_codebase` tool.
///
/// The handler returns immediately; indexing runs in the background.
/// Poll `index_status` to observe progress and detect completion.
#[derive(Debug, Serialize, JsonSchema)]
pub struct IndexCodebaseOutput {
    /// `"indexing_started"` or `"already_indexing"`.
    pub status: String,
    /// The path that is being (or was already being) indexed.
    pub path: String,
    /// Best-effort count of indexable files discovered before spawning.
    /// Zero when the quick-count timed out or when already indexing.
    pub files_queued: usize,
    /// Human-readable description of what happened.
    pub message: String,
}

/// Output of the `index_status` tool.
#[derive(Debug, Serialize, JsonSchema)]
pub struct IndexStatusOutput {
    pub indexed_files: usize,
    pub total_chunks: usize,
    /// RFC 3339 timestamp of the most recent indexing run, if any.
    pub last_indexed: Option<String>,
    /// Best-effort manifest-based count of files that may be out of date.
    /// This is not forced to zero just because indexing is currently idle.
    pub estimated_stale: usize,
    /// Freshness state derived from manifest truth plus the live refresh overlay.
    /// `refreshing` means a refresh is in flight, `unknown` means the check failed,
    /// and watcher state is reported separately via `watching`.
    pub freshness_state: IndexFreshnessState,
    /// RFC 3339 timestamp when freshness was checked, when available.
    pub freshness_checked_at: Option<String>,
    /// Error string when freshness could not be determined.
    pub freshness_error: Option<String>,
    /// Whether a background watcher is active for this server instance.
    /// This does not imply the index is fresh.
    pub watching: bool,
    /// Live progress for an active or recently-completed background index.
    /// `null` when `index_codebase` has never been called on this server instance.
    pub indexing: Option<IndexingProgress>,
}

#[derive(Debug, Clone, Copy, Serialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum IndexFreshnessState {
    Fresh,
    Stale,
    Refreshing,
    Unknown,
}

/// Progress snapshot for a background `index_codebase` operation.
#[derive(Debug, Serialize, JsonSchema)]
pub struct IndexingProgress {
    /// `"running"`, `"done"`, or `"failed"`.
    pub status: String,
    /// The path being indexed.
    pub path: String,
    /// Files indexed so far (set after completion, 0 while running).
    pub files_done: usize,
    /// Total files discovered before indexing started (0 if quick-count timed out).
    pub files_total: usize,
    /// Total chunks written to the backend.
    pub chunks_done: usize,
    /// Embedding cache hits during this run.
    pub cache_hits: usize,
    /// Seconds elapsed since indexing started.
    pub elapsed_seconds: f64,
    /// Error message when `status` is `"failed"`; null otherwise.
    pub error: Option<String>,
}

/// A single indexed chunk record.
#[derive(Debug, Serialize, JsonSchema)]
pub struct ChunkInfo {
    pub file_path: String,
    pub chunk_idx: usize,
    pub content: String,
    pub chunk_type: String,
    pub start_line: usize,
    pub end_line: usize,
}

/// Output of the `get_file_context` tool.
#[derive(Debug, Serialize, JsonSchema)]
pub struct FileContextOutput {
    pub chunks: Vec<ChunkInfo>,
    pub imports: Vec<String>,
    pub imported_by: Vec<String>,
}

/// Input for the `smart_search` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SmartSearchInput {
    /// Natural-language or keyword query; automatically routed to grep or semantic search.
    pub query: String,
    /// Maximum number of results to return (default: 5).
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// When true (semantic path only), augment results with one-hop import-graph neighbours.
    #[serde(default)]
    pub include_graph: bool,
    /// Diversity factor for MMR re-ranking (0.0–1.0). 0.0 = pure relevance,
    /// higher values reduce redundancy. Default: 0.3.
    #[serde(default = "default_diversity")]
    pub diversity: f32,
    /// Maximum token budget for results. Omit for unlimited.
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// When true, scope results to files changed on the current git branch.
    #[serde(default)]
    pub branch_scope: bool,
    /// Optional session ID for result deduplication across searches.
    /// When set, results seen in previous searches with the same session ID
    /// are deprioritized (moved to bottom of results).
    #[serde(default)]
    pub session_id: Option<String>,
    /// Search intent that controls retrieval strategy.
    /// - "find": pure vector + BM25 search (default)
    /// - "understand": vector search + deep graph expansion (2-3 hops)
    /// - "impact": reverse graph traversal — what depends on the query target
    /// - "trace": find connection path between two symbols
    /// When omitted, auto-detected from query content.
    #[serde(default)]
    pub intent: Option<String>,
    /// Known symbol names or file paths to anchor the search.
    /// For "impact" intent: the file path of the target to analyze (first entry used).
    /// For "trace" intent: [start_symbol, end_symbol].
    /// For other intents: prepended to the query as BM25 boost terms.
    #[serde(default)]
    pub symbols: Vec<String>,
    /// Scope search to a specific directory or module path.
    /// Example: "src/auth" limits results to files under that path.
    #[serde(default)]
    pub scope: Option<String>,
    /// Project path to search. When set, searches that project's index instead
    /// of the default (server cwd). The project must have been indexed.
    #[serde(default)]
    pub project: Option<String>,
}

/// A single grep result row returned by `smart_search` on the grep path.
#[derive(Debug, Serialize, JsonSchema)]
pub struct GrepSearchRow {
    pub file_path: String,
    pub line_number: usize,
    pub line_content: String,
}

/// Typed result set for `smart_search`; variant chosen by the query classifier.
#[derive(Debug, Serialize, JsonSchema)]
#[serde(tag = "kind", content = "items", rename_all = "lowercase")]
pub enum SmartSearchResults {
    Grep(Vec<GrepSearchRow>),
    Semantic(Vec<SearchCodeRow>),
    /// Returned when intent = "impact"; wraps the full ImpactSetOutput.
    Impact(ImpactSetOutput),
}

/// Output of the `smart_search` tool.
#[derive(Debug, Serialize, JsonSchema)]
pub struct SmartSearchOutput {
    /// Which strategy was chosen: `"grep"` or `"semantic"`.
    pub strategy: String,
    /// Typed result set; inspect `kind` to determine the row schema.
    pub results: SmartSearchResults,
}

/// Input for the `find_symbol` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct FindSymbolInput {
    /// Symbol name to search for.
    pub name: String,
    /// Optional kind filter (e.g., "function", "struct", "class").
    pub kind: Option<String>,
    /// Project path to search.
    #[serde(default)]
    pub project: Option<String>,
}

/// A single symbol definition.
#[derive(Debug, Serialize, JsonSchema)]
pub struct SymbolRow {
    pub file_path: String,
    pub name: String,
    pub kind: String,
    pub start_line: usize,
    pub end_line: usize,
}

/// A caller or callee edge from the function-level call graph.
#[derive(Debug, Serialize, JsonSchema)]
pub struct CallEdgeInfo {
    /// File containing the caller/callee.
    pub file_path: String,
    /// Function/method name.
    pub symbol: String,
    /// Resolution confidence (1.0 = import-resolved, 0.3 = name-only).
    pub confidence: f64,
}

/// Input for the `find_impact_set` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct FindImpactSetInput {
    /// File path to analyze impact for.
    pub file_path: String,
    /// Maximum traversal depth (default: 3, capped at 5).
    pub max_depth: Option<usize>,
    /// Project path to search.
    #[serde(default)]
    pub project: Option<String>,
}

/// A single entry in the transitive importer list.
#[derive(Debug, Serialize, JsonSchema)]
pub struct ImpactEntry {
    pub file_path: String,
    pub depth: usize,
}

/// Output of the `find_impact_set` tool.
#[derive(Debug, Serialize, JsonSchema)]
pub struct ImpactSetOutput {
    /// The file being analyzed.
    pub file_path: String,
    /// Files that directly import this file (depth == 1).
    pub direct_importers: Vec<String>,
    /// Files that transitively import this file, grouped by depth.
    pub transitive_importers: Vec<ImpactEntry>,
    /// Test files that (transitively) import this file.
    pub affected_tests: Vec<String>,
    /// Function-level callers from the call graph (higher precision than file-level importers).
    /// Currently always empty; populated per-symbol via get_symbol_context.
    // TODO(PER-144): wire get_callers for a specific file's exported symbols once
    // a "get all callers for file" query exists in StorageBackend.
    pub function_callers: Vec<CallEdgeInfo>,
}
/// Input for the `find_test_context` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct FindTestContextInput {
    /// File path to find tests for.
    pub file_path: String,
    /// Project path to search.
    #[serde(default)]
    pub project: Option<String>,
}

/// Output of the `find_test_context` tool.
#[derive(Debug, Serialize, JsonSchema)]
pub struct TestContextOutput {
    /// The source file.
    pub file_path: String,
    /// Test files that directly import this file.
    pub test_files: Vec<String>,
    /// Test files in the same directory or a sibling test directory.
    pub colocated_tests: Vec<String>,
}

fn default_true() -> bool {
    true
}

/// Input for the `get_symbol_context` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetSymbolContextInput {
    /// Symbol name to look up.
    pub name: String,
    /// Optional kind filter (e.g., "function", "struct", "class").
    pub kind: Option<String>,
    /// Include test files that reference this symbol (default: true).
    #[serde(default = "default_true")]
    pub include_tests: bool,
    /// Project path to search.
    #[serde(default)]
    pub project: Option<String>,
}

/// Output of the `get_symbol_context` tool.
#[derive(Debug, Serialize, JsonSchema)]
pub struct SymbolContextOutput {
    /// The first matching symbol definition, if found.
    pub symbol: Option<SymbolRow>,
    /// Number of matches found for the requested symbol name/kind.
    pub match_count: usize,
    /// True when multiple definitions matched and only the first was returned.
    pub ambiguous: bool,
    /// Source code of the chunk containing this symbol.
    pub source: Option<String>,
    /// Files that import the symbol's file (callers at file level).
    pub imported_by: Vec<String>,
    /// True when imported_by was truncated for token efficiency.
    pub imported_by_truncated: bool,
    /// Files that the symbol's file imports (dependencies).
    pub imports: Vec<String>,
    /// True when imports was truncated for token efficiency.
    pub imports_truncated: bool,
    /// Test files that import the symbol's file.
    pub test_files: Vec<String>,
    /// Symbol role classification.
    pub role: Option<String>,
    /// Functions that call this symbol (from function-level call graph, capped at 20).
    pub callers: Vec<CallEdgeInfo>,
    /// Functions this symbol calls (capped at 20, resolved callees only).
    pub callees: Vec<CallEdgeInfo>,
}

// ---------------------------------------------------------------------------
// Repo Map
// ---------------------------------------------------------------------------

/// Input for the `get_repo_map` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetRepoMapInput {
    /// Maximum token budget for the response (default: 8192).
    /// Larger budgets include more symbols and edges.
    #[serde(default = "default_map_tokens")]
    pub max_tokens: usize,
    /// If true, include per-file symbol lists (default: true).
    #[serde(default = "default_true")]
    pub include_symbols: bool,
    /// If true, include file-level import edges (default: true).
    #[serde(default = "default_true")]
    pub include_edges: bool,
    /// Project path to query.
    #[serde(default)]
    pub project: Option<String>,
}

fn default_map_tokens() -> usize {
    8192
}
