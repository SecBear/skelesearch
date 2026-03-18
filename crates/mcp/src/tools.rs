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
}

fn default_top_k() -> usize {
    5
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
    /// Retrieval provenance: `"vector"`, `"fts"`, `"both"`, or `"imports <file>"`.
    pub why: String,
}

/// Output of the `index_codebase` tool.
#[derive(Debug, Serialize, JsonSchema)]
pub struct IndexCodebaseOutput {
    pub status: String,
    pub indexed: usize,
    pub chunks: usize,
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
}

/// Output of the `smart_search` tool.
#[derive(Debug, Serialize, JsonSchema)]
pub struct SmartSearchOutput {
    /// Which strategy was chosen: `"grep"` or `"semantic"`.
    pub strategy: String,
    /// Results; shape depends on strategy.
    pub results: serde_json::Value,
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