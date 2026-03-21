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
    /// Embedding provider to use. Supported: `"fastembed"` (default).
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
    /// Retrieval provenance: `"vector"`, `"fts"`, or `"hybrid"`.
    pub why: String,
}

/// Response envelope for the `search_code` tool, including per-phase timings.
#[derive(Debug, Serialize, JsonSchema)]
pub struct SearchCodeResponse {
    pub results: Vec<SearchCodeRow>,
    /// Per-phase latency breakdown of the search pipeline.
    #[schemars(skip)]
    pub _timings: skelesearch_core::SearchTimings,
}

/// Output of the `index_codebase` tool.
#[derive(Debug, Serialize, JsonSchema)]
pub struct IndexCodebaseOutput {
    pub status: String,
    pub indexed: usize,
    pub chunks: usize,
    /// Embedding cache hits during this indexing run.
    pub cache_hits: usize,
}

/// Output of the `index_status` tool.
#[derive(Debug, Serialize, JsonSchema)]
pub struct IndexStatusOutput {
    pub indexed_files: usize,
    pub total_chunks: usize,
    /// RFC 3339 timestamp of the most recent indexing run, if any.
    pub last_indexed: Option<String>,
    /// Number of files that appear to have changed since last indexing (v1: always 0).
    pub estimated_stale: usize,
    /// Whether a watch process is running (v1: always false).
    pub watching: bool,
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

/// Input for the `find_impact_set` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct FindImpactSetInput {
    /// File path to analyze impact for.
    pub file_path: String,
    /// Maximum traversal depth (default: 3, capped at 5).
    pub max_depth: Option<usize>,
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
}

/// Input for the `find_test_context` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct FindTestContextInput {
    /// File path to find tests for.
    pub file_path: String,
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