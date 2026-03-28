use crate::cochange::CoChangePair;
use crate::sparse::SparseEmbedding;
use crate::symbols::SymbolDef;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use std::path::{Path, PathBuf};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Public record types
// ---------------------------------------------------------------------------

pub const INDEX_DB_FILE: &str = "index.db";
pub const MANIFEST_DB_FILE: &str = "manifest.db";

pub fn generation_db_paths(generation_dir: &Path) -> (PathBuf, PathBuf) {
    (
        generation_dir.join(INDEX_DB_FILE),
        generation_dir.join(MANIFEST_DB_FILE),
    )
}

#[derive(Debug, Clone, PartialEq)]
pub struct FileRecord {
    pub file_path: String,
    pub language: String,
    /// Unix timestamp (seconds) of the file's last modification.
    pub last_modified: i64,
    /// Unix timestamp (seconds) when indexing last wrote this file.
    pub last_indexed: i64,
    pub chunk_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChunkRecord {
    pub file_path: String,
    pub chunk_idx: usize,
    pub content: String,
    /// Whitespace-normalised version of `content` used for FTS.
    pub normalized: String,
    /// LLM-generated natural-language summary. Empty when no summary provider was
    /// attached at index time. The embedding is computed from this field when non-empty,
    /// from `content` otherwise.
    pub description: String,
    pub chunk_type: String,
    pub start_line: usize,
    pub end_line: usize,
    /// `None` until the chunk has been embedded.
    pub embedding: Option<Vec<f32>>,
    /// Docstring/doc-comment embedding. `None` when dual-embedding was not enabled at
    /// index time, or when no doc comment was found for this chunk.
    pub doc_embedding: Option<Vec<f32>>,
    /// Progressive materialization tier (1 = fast token-window, 2 = AST-aware).
    /// Defaults to `2` for all existing chunks and newly indexed chunks.
    pub materialization_tier: u8,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EdgeRecord {
    pub from_file: String,
    pub from_chunk: usize,
    pub to_file: String,
    pub edge_type: String,
}


#[derive(Debug, Clone, PartialEq)]
pub struct CallEdge {
    pub caller_file: String,
    pub caller_symbol: String,
    pub callee_name: String,
    pub start_line: usize,
    /// `None` when the callee could not be resolved to a specific file.
    pub callee_file: Option<String>,
    /// `None` when the callee could not be resolved to a specific symbol.
    pub callee_symbol: Option<String>,
    /// Extraction confidence in [0.0, 1.0].
    pub confidence: f64,
    /// `true` for computed/dynamic dispatch (virtual, trait-object, etc.).
    pub dynamic: bool,
}
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub file_path: String,
    pub chunk_idx: usize,
    pub content: String,
    pub start_line: usize,
    pub end_line: usize,
    pub chunk_type: String,
    pub score: f64,
    /// Relative quality label: `"high"`, `"moderate"`, or `"low"`.
    /// Set by `Searcher`; empty string until shaped.
    pub match_quality: String,
    /// Retrieval provenance: `"vector"`, `"fts"`, `"hybrid"`, `"graph"`, or `"hnsw_proximity"`.
    /// Set by `Searcher`; empty string until shaped.
    pub why: String,
    /// Progressive materialization tier of the retrieved chunk.
    /// `1` = token-window (fast), `2` = AST-aware (full quality).
    pub materialization_tier: u8,
}

#[derive(Debug, Clone)]
pub struct IndexStats {
    pub indexed_files: usize,
    pub total_chunks: usize,
    /// `None` when the index contains no files.
    pub last_indexed: Option<DateTime<Utc>>,
    /// Always `false` in v1 (watching is a v2 feature).
    pub watching: bool,
    /// Estimated number of stale entries (files changed since last index).
    pub estimated_stale: usize,
}

/// Compact codebase overview assembled from indexed data.
#[derive(Debug, Clone)]
pub struct RepoMapData {
    pub files: Vec<RepoMapFile>,
    pub import_edges: Vec<(String, String)>,  // (from_file, to_file)
}

#[derive(Debug, Clone)]
pub struct RepoMapFile {
    pub path: String,
    pub language: String,
    pub chunk_count: usize,
    pub role: String,           // entry/core/utility/leaf/unknown
    pub symbols: Vec<RepoMapSymbol>,
}

#[derive(Debug, Clone)]
pub struct RepoMapSymbol {
    pub name: String,
    pub kind: String,
    pub start_line: usize,
}

// ---------------------------------------------------------------------------
// StorageBackend trait
// ---------------------------------------------------------------------------

#[async_trait]
pub trait StorageBackend: Send + Sync {
    async fn initialize(&self, dim: usize) -> anyhow::Result<()>;

    async fn upsert_file(&self, record: &FileRecord) -> anyhow::Result<()>;
    async fn delete_file(&self, file_path: &str) -> anyhow::Result<()>;
    async fn list_indexed_paths(&self) -> anyhow::Result<Vec<String>>;

    async fn upsert_chunks(&self, chunks: &[ChunkRecord]) -> anyhow::Result<()>;
    async fn delete_chunks_for_file(&self, file_path: &str) -> anyhow::Result<()>;
    /// Delete only Tier 1 (token-window) chunks for the given file, leaving
    /// Tier 2 (AST-aware) chunks intact.  Used by the progressive background
    /// upgrade after Tier 2 chunks have been written.
    /// Default no-op for backends that don't track materialization_tier.
    async fn delete_tier1_chunks_for_file(&self, file_path: &str) -> anyhow::Result<()> {
        let _ = file_path;
        Ok(())
    }
    async fn get_chunks_for_file(&self, file_path: &str) -> anyhow::Result<Vec<ChunkRecord>>;
    /// Batch fetch chunks for multiple files in a single query.
    async fn get_chunks_for_files(&self, file_paths: &[&str]) -> anyhow::Result<Vec<ChunkRecord>>;

    async fn upsert_edges(&self, edges: &[EdgeRecord]) -> anyhow::Result<()>;
    async fn delete_edges_for_file(&self, file_path: &str) -> anyhow::Result<()>;
    async fn get_importers(&self, file_path: &str) -> anyhow::Result<Vec<String>>;
    async fn get_imports(&self, file_path: &str) -> anyhow::Result<Vec<String>>;

    /// BFS traversal of the import graph starting from `file_path`, up to
    /// `max_depth` hops.  Returns `(file_path, depth)` for each reachable file
    /// (excluding the start node).  Cycles are handled by the visited set.
    /// Pass `edge_types = None` to traverse all edge types (current behavior).
    async fn traverse_imports(&self, file_path: &str, max_depth: usize, edge_types: Option<&[&str]>) -> anyhow::Result<Vec<(String, usize)>>;

    /// Reverse BFS: find all files that (transitively) import `file_path`,
    /// up to `max_depth` hops. Returns files grouped by distance.
    /// Cycles are handled by the visited set.
    async fn traverse_importers(
        &self,
        file_path: &str,
        max_depth: usize,
        edge_types: Option<&[&str]>,
    ) -> anyhow::Result<Vec<(String, usize)>>; // (file_path, depth)

    async fn hybrid_search(
        &self,
        query_vec: &[f32],
        query_str: &str,
        top_k: usize,
    ) -> anyhow::Result<Vec<SearchResult>>;

    async fn stats(&self) -> anyhow::Result<IndexStats>;

    async fn upsert_symbols(&self, symbols: &[SymbolDef]) -> anyhow::Result<()>;
    async fn delete_symbols_for_file(&self, file_path: &str) -> anyhow::Result<()>;
    async fn find_symbols(&self, name: &str, kind: Option<&str>) -> anyhow::Result<Vec<SymbolDef>>;

    /// Fetch stored embedding vectors for specific chunks.
    /// Returns embeddings in the same order as `keys`. Missing chunks yield a zero vector.
    async fn get_chunk_embeddings(&self, keys: &[(String, usize)]) -> anyhow::Result<Vec<Vec<f32>>>;

    /// Compute PageRank over the file-level import graph and store results.
    async fn compute_pagerank(&self, edge_types: Option<&[&str]>) -> anyhow::Result<()>;

    /// Retrieve PageRank scores for the given file paths.
    /// Returns a HashMap; files not in the graph get score 0.0.
    async fn get_file_ranks(&self, file_paths: &[&str]) -> anyhow::Result<std::collections::HashMap<String, f64>>;

    /// Compute file-level import-degree roles for all indexed symbols and store
    /// results in `symbol_roles`.  Files with no edges get conservative defaults.
    /// No-op when no symbols have been indexed yet.
    async fn compute_symbol_roles(&self) -> anyhow::Result<()>;

    /// Retrieve the role for each requested file path.
    /// Returns a map of `file_path → role`; files with no computed role are absent.
    /// Gracefully returns an empty map when `symbol_roles` does not exist yet.
    async fn get_symbol_roles(&self, file_paths: &[&str]) -> anyhow::Result<std::collections::HashMap<String, String>>;
    /// Upsert co-change pairs derived from git history.
    async fn upsert_cochange_edges(&self, pairs: &[CoChangePair]) -> anyhow::Result<()>;
    /// Return co-change neighbors of `file_path` with Jaccard similarity >= `min_score`.
    /// Returns `(neighbor_file_path, jaccard)` pairs sorted by jaccard descending.
    /// No-op on backends that have not indexed co-change data.
    async fn get_cochange_neighbors(&self, _file_path: &str, _min_score: f64) -> anyhow::Result<Vec<(String, f64)>> {
        Ok(vec![])
    }

    /// Walk the HNSW proximity graph at layer 0 to find vector-similar chunks
    /// without re-embedding. Returns `(file_path, chunk_idx, distance)` tuples
    /// for neighbors of any seed chunk within `max_dist` (cosine distance;
    /// 0 = identical, 1 = orthogonal). Returns an empty vec if the index does
    /// not exist yet or no seeds are provided.
    async fn hnsw_neighbors(
        &self,
        seeds: &[(String, usize)], // (file_path, chunk_idx)
        max_dist: f64,
        limit: usize,
    ) -> anyhow::Result<Vec<(String, usize, f64)>>;

    /// Store sparse embedding entries for a chunk.
    /// Called at index time for each chunk when a sparse provider is active.
    async fn store_sparse_vectors(
        &self,
        _file_path: &str,
        _chunk_idx: usize,
        _sparse: &SparseEmbedding,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// Remove all sparse index entries for the given file.
    async fn delete_sparse_for_file(&self, _file_path: &str) -> anyhow::Result<()> {
        Ok(())
    }

    /// Dot-product search over the sparse index.
    /// Returns `(file_path, chunk_idx, score)` tuples sorted by score descending,
    /// truncated to `top_k`. Empty when no sparse index exists.
    async fn sparse_search(
        &self,
        _query_sparse: &SparseEmbedding,
        _top_k: usize,
    ) -> anyhow::Result<Vec<(String, usize, f64)>> {
        Ok(vec![])
    }

    /// Search the doc-embedding HNSW index (`chunks:doc_index`).
    /// Returns `(file_path, chunk_idx, cosine_distance)` tuples sorted by distance
    /// ascending (lower = more similar). Returns an empty vec when no doc index
    /// exists or dual-embedding was not enabled at index time.
    async fn doc_vector_search(
        &self,
        _query_vec: &[f32],
        _limit: usize,
    ) -> anyhow::Result<Vec<(String, usize, f64)>> {
        Ok(vec![])
    }


    /// Single-query retrieval combining FTS + HNSW + graph walk + PageRank boost.
    /// Returns results with provenance (\"hybrid\", \"graph\").
    /// `graph_depth > 0` enables a single-hop graph walk in the Datalog query;
    /// 0 skips the graph rule entirely.
    ///
    /// Non-Cozo backends delegate to `hybrid_search` (no graph walk).
    async fn unified_search(
        &self,
        query_vec: &[f32],
        query_str: &str,
        top_k: usize,
        graph_depth: usize,
        fts_weight: f64,
        graph_score_factor: f64,
        graph_min_score: f64,
        pagerank_factor: f64,
    ) -> anyhow::Result<Vec<SearchResult>> {
        let _ = (graph_depth, fts_weight, graph_score_factor, graph_min_score, pagerank_factor);
        self.hybrid_search(query_vec, query_str, top_k).await
    }

    /// Remove near-duplicate chunks across different files using the LSH index.
    /// Chunks sharing an LSH hash bucket but from different files are collapsed:
    /// one representative is kept (lowest file_path, then lowest chunk_idx),
    /// the rest are deleted.  Returns the number of chunks removed.
    /// No-op if the LSH index does not exist or contains no cross-file duplicates.
    async fn deduplicate_chunks(&self) -> anyhow::Result<usize>;

    /// Fetch all data needed for a compact repo map in minimal round-trips.
    /// Returns files with their symbols and roles, plus file-level import edges.
    async fn get_repo_map_data(&self) -> anyhow::Result<RepoMapData>;

    async fn upsert_call_edges(&self, edges: &[CallEdge]) -> anyhow::Result<()>;
    async fn delete_call_edges_for_file(&self, file_path: &str) -> anyhow::Result<()>;
    async fn get_callers(&self, file_path: &str, symbol_name: &str) -> anyhow::Result<Vec<CallEdge>>;
    async fn get_callees(&self, file_path: &str, symbol_name: &str) -> anyhow::Result<Vec<CallEdge>>;
}

// ---------------------------------------------------------------------------
// Role classification
// ---------------------------------------------------------------------------

/// Classify the structural role of a symbol based on its file's import-graph degree.
///
/// Dead code detection is deferred until function-level call graph is available.
/// File-level import degree is too coarse — main(), test fns, and CLI handlers
/// all have in_degree=0 at the file level but are not dead (PER-133 #2).
///
/// Roles are mutually exclusive and ordered by priority:
/// 1. `entry`   — heavily imported, few or no outbound deps (public API surface)
/// 2. `core`    — both heavily imported and imports many others (central module)
/// 3. `utility` — not widely imported but imports many (shared helpers)
/// 4. `leaf`    — no outbound imports (self-contained implementation)
/// 5. `internal`— default when no structural pattern matches
pub(crate) fn classify_symbol_role(in_degree: usize, out_degree: usize) -> &'static str {
    // File-level structural patterns.
    if in_degree >= 3 && out_degree <= 1 {
        "entry"
    } else if in_degree >= 2 && out_degree >= 2 {
        "core"
    } else if in_degree <= 1 && out_degree >= 2 {
        "utility"
    } else if out_degree == 0 {
        "leaf"
    } else {
        "internal"
    }
}

