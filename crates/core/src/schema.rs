use crate::symbols::SymbolDef;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use cozo::{DataValue, DbInstance, NamedRows};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Public record types
// ---------------------------------------------------------------------------

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
    pub chunk_type: String,
    pub start_line: usize,
    pub end_line: usize,
    /// `None` until the chunk has been embedded.
    pub embedding: Option<Vec<f32>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EdgeRecord {
    pub from_file: String,
    pub from_chunk: usize,
    pub to_file: String,
    pub edge_type: String,
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
}

// ---------------------------------------------------------------------------
// CozoBackend — the only place in the codebase that touches Cozo directly.
// ---------------------------------------------------------------------------

pub struct CozoBackend {
    db: DbInstance,
    /// Embedding dimension set during `initialize`; 0 until initialized.
    dim: Arc<AtomicUsize>,
}

impl CozoBackend {
    pub fn open(path: impl AsRef<std::path::Path>) -> anyhow::Result<Self> {
        let path_str = path.as_ref().to_string_lossy();
        let db = DbInstance::new("sqlite", path_str.as_ref(), Default::default())
            .map_err(|e| anyhow::anyhow!("cozo open: {}", e))?;
        Ok(Self { db, dim: Arc::new(AtomicUsize::new(0)) })
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    fn run_mut(&self, script: &str, params: BTreeMap<String, DataValue>) -> anyhow::Result<NamedRows> {
        self.db
            .run_script(script, params, cozo::ScriptMutability::Mutable)
            .map_err(|e| {
                tracing::debug!(script = %script, error = %e, "CozoDB write query failed");
                anyhow::anyhow!("{}", e)
            })
    }

    fn run_imm(&self, script: &str, params: BTreeMap<String, DataValue>) -> anyhow::Result<NamedRows> {
        self.db
            .run_script(script, params, cozo::ScriptMutability::Immutable)
            .map_err(|e| {
                tracing::debug!(script = %script, error = %e, "CozoDB read query failed");
                anyhow::anyhow!("{}", e)
            })
    }

    /// Run a script, ignoring errors that indicate idempotent creation (e.g. schema
    /// already set up). Intentionally narrow: only swallows messages that contain
    /// "already exists" so we don't accidentally hide real conflicts such as
    /// dimension mismatches or concurrent-write failures.
    fn run_mut_ignore(&self, script: &str) -> anyhow::Result<()> {
        match self.run_mut(script, BTreeMap::new()) {
            Ok(_) => Ok(()),
            Err(e) => {
                let msg = e.to_string().to_lowercase();
                // Suppress errors that are unambiguously 'already created':
                // - CozoDB returns "already exists" in some versions
                // - CozoDB's :create returns "conflicts with an existing one"
                //   when the relation already exists (e.g. double-initialize).
                // Do NOT broaden this to a generic 'conflict' — that would hide
                // real dimension-mismatch or concurrent-write failures.
                if msg.contains("already exists") || msg.contains("conflicts with an existing one") {
                    Ok(())
                } else {
                    Err(e)
                }
            }
        }
    }

    fn dv_str(s: &str) -> DataValue {
        DataValue::Str(s.into())
    }

    fn dv_int(n: i64) -> DataValue {
        DataValue::Num(cozo::Num::Int(n))
    }

    fn dv_float(f: f64) -> DataValue {
        DataValue::Num(cozo::Num::Float(f))
    }

    fn embedding_to_dv(emb: &Option<Vec<f32>>, dim: usize) -> DataValue {
        match emb {
            Some(vec) => DataValue::List(
                vec.iter().map(|&f| Self::dv_float(f as f64)).collect(),
            ),
            // The <F32; dim> schema type requires exactly `dim` floats.
            // Use a zero vector as the "no embedding" sentinel.
            None => DataValue::List(
                (0..dim).map(|_| Self::dv_float(0.0)).collect(),
            ),
        }
    }

    fn row_to_chunk(row: &[DataValue]) -> anyhow::Result<ChunkRecord> {
        // Column order: file_path, chunk_idx, content, normalized, chunk_type,
        //               start_line, end_line, embedding
        let file_path = Self::str_col(&row[0])?;
        let chunk_idx = Self::int_col(&row[1])? as usize;
        let content = Self::str_col(&row[2])?;
        let normalized = Self::str_col(&row[3])?;
        let chunk_type = Self::str_col(&row[4])?;
        let start_line = Self::int_col(&row[5])? as usize;
        let end_line = Self::int_col(&row[6])? as usize;
        let embedding = match &row[7] {
            DataValue::List(items) if items.is_empty() => None,
            DataValue::List(items) => Some(
                items
                    .iter()
                    .map(|d| match d {
                        DataValue::Num(cozo::Num::Float(f)) => *f as f32,
                        DataValue::Num(cozo::Num::Int(i)) => *i as f32,
                        _ => 0.0,
                    })
                    .collect(),
            ),
            _ => None,
        };
        Ok(ChunkRecord { file_path, chunk_idx, content, normalized, chunk_type, start_line, end_line, embedding })
    }

    fn str_col(dv: &DataValue) -> anyhow::Result<String> {
        match dv {
            DataValue::Str(s) => Ok(s.to_string()),
            other => anyhow::bail!("expected Str, got {:?}", other),
        }
    }

    fn int_col(dv: &DataValue) -> anyhow::Result<i64> {
        match dv {
            DataValue::Num(cozo::Num::Int(n)) => Ok(*n),
            DataValue::Num(cozo::Num::Float(f)) => Ok(*f as i64),
            other => anyhow::bail!("expected Int, got {:?}", other),
        }
    }
    /// Run FTS only and return raw tuples:
    /// `(file_path, chunk_idx, bm25_score, content, chunk_type, start_line, end_line)`.
    /// Results are ordered by bm25 score descending.
    #[tracing::instrument(skip_all, fields(limit))]
    fn fts_search(
        &self,
        query_text: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<(String, usize, f64, String, String, usize, usize)>> {
        let script = format!(
            r#"?[file_path, chunk_idx, bm25, content, chunk_type, start_line, end_line] :=
    ~chunks:text{{ file_path, chunk_idx | query: $qs, k: {limit}, score_kind: 'tf_idf', bind_score: bm25 }},
    *chunks[file_path, chunk_idx, content, _, chunk_type, start_line, end_line, _]
:order -bm25
:limit {limit}"#
        );
        let mut p = BTreeMap::new();
        p.insert("qs".into(), Self::dv_str(query_text));
        let rows = match self.run_imm(&script, p) {
            Ok(r) => r,
            Err(e) => {
                let msg = e.to_string();
                if msg.contains("no results") || msg.contains("empty") {
                    return Ok(vec![]);
                }
                return Err(e);
            }
        };
        rows.rows
            .iter()
            .map(|r| {
                Ok((
                    Self::str_col(&r[0])?,
                    Self::int_col(&r[1])? as usize,
                    match &r[2] {
                        DataValue::Num(cozo::Num::Float(f)) => *f,
                        DataValue::Num(cozo::Num::Int(i)) => *i as f64,
                        _ => 0.0,
                    },
                    Self::str_col(&r[3])?,
                    Self::str_col(&r[4])?,
                    Self::int_col(&r[5])? as usize,
                    Self::int_col(&r[6])? as usize,
                ))
            })
            .collect()
    }

    /// Run HNSW vector search and return raw tuples:
    /// `(file_path, chunk_idx, cosine_distance, content, chunk_type, start_line, end_line)`.
    /// Results are ordered by cosine distance ascending (lower = more similar).
    #[tracing::instrument(skip_all, fields(limit))]
    fn vector_search(
        &self,
        query_vec: &[f32],
        limit: usize,
    ) -> anyhow::Result<Vec<(String, usize, f64, String, String, usize, usize)>> {
        let query_vec_dv: DataValue = {
            let arr = ndarray::Array1::from(query_vec.to_vec());
            DataValue::Vec(cozo::Vector::F32(arr))
        };
        let script = format!(
            r#"?[file_path, chunk_idx, dist, content, chunk_type, start_line, end_line] :=
    ~chunks:semantic{{ file_path, chunk_idx | query: $qv, k: {limit}, ef: 64, bind_distance: dist }},
    *chunks[file_path, chunk_idx, content, _, chunk_type, start_line, end_line, _]
:order dist
:limit {limit}"#
        );
        let mut p = BTreeMap::new();
        p.insert("qv".into(), query_vec_dv);
        let rows = match self.run_imm(&script, p) {
            Ok(r) => r,
            Err(e) => {
                let msg = e.to_string();
                if msg.contains("no results") || msg.contains("empty") {
                    return Ok(vec![]);
                }
                return Err(e);
            }
        };
        rows.rows
            .iter()
            .map(|r| {
                Ok((
                    Self::str_col(&r[0])?,
                    Self::int_col(&r[1])? as usize,
                    match &r[2] {
                        DataValue::Num(cozo::Num::Float(f)) => *f,
                        DataValue::Num(cozo::Num::Int(i)) => *i as f64,
                        _ => 0.0,
                    },
                    Self::str_col(&r[3])?,
                    Self::str_col(&r[4])?,
                    Self::int_col(&r[5])? as usize,
                    Self::int_col(&r[6])? as usize,
                ))
            })
            .collect()
    }

    /// FTS-only search fallback for when no embeddings exist.
    /// Returns `SearchResult` rows with `why = "fts"`.
    fn fts_only_search(
        &self,
        query_text: &str,
        top_k: usize,
    ) -> anyhow::Result<Vec<SearchResult>> {
        self.fts_search(query_text, top_k)?
            .into_iter()
            .map(|(file_path, chunk_idx, score, content, chunk_type, start_line, end_line)| {
                Ok(SearchResult {
                    file_path,
                    chunk_idx,
                    content,
                    start_line,
                    end_line,
                    chunk_type,
                    score,
                    match_quality: String::new(),
                    why: "fts".to_string(),
                })
            })
            .collect()
    }

}

#[async_trait]
impl StorageBackend for CozoBackend {
    async fn initialize(&self, dim: usize) -> anyhow::Result<()> {
        self.dim.store(dim, Ordering::Relaxed);
        // Create the three base relations — idempotent via error message check.
        self.run_mut_ignore(
            ":create files { file_path: String => language: String, last_modified: Int, last_indexed: Int, chunk_count: Int }",
        )?;

        // The embedding field uses CozoDB's fixed-dimension vector type <F32; dim>.
        // The relation is created with the specific dim supplied at initialization time.
        let chunks_schema = format!(
            ":create chunks {{ file_path: String, chunk_idx: Int => content: String, normalized: String, chunk_type: String, start_line: Int, end_line: Int, embedding: <F32; {dim}> }}"
        );
        self.run_mut_ignore(&chunks_schema)?;

        self.run_mut_ignore(
            ":create code_edges { from_file: String, from_chunk: Int, to_file: String => edge_type: String, created_at: Int }",
        )?;

        // Create HNSW vector index — idempotent.
        let hnsw = format!(
            "::hnsw create chunks:semantic {{ dim: {dim}, dtype: F32, fields: [embedding], distance: Cosine, m: 32, ef_construction: 128 }}"
        );
        self.run_mut_ignore(&hnsw)?;

        // Create FTS index — idempotent.
        self.run_mut_ignore(
            "::fts create chunks:text { extractor: normalized, tokenizer: Simple, filters: [Lowercase, AlphaNumOnly] }",
        )?;

        // Create LSH index for near-duplicate chunk detection — idempotent.
        match self.run_mut(
            "::lsh create chunks:dedup { extractor: normalized, tokenizer: Simple, n_gram: 5, n_perm: 128, target_threshold: 0.85 }",
            BTreeMap::new(),
        ) {
            Ok(_) => tracing::debug!("LSH dedup index created"),
            Err(e) => {
                let msg = e.to_string().to_lowercase();
                if msg.contains("already exists") || msg.contains("conflicts") {
                    tracing::debug!("LSH dedup index already exists");
                } else {
                    tracing::warn!(error = %e, "LSH dedup index creation failed — dedup disabled");
                }
            }
        }

        // Create symbols relation — idempotent.
        self.run_mut_ignore(
            ":create symbols { file_path: String, name: String, start_line: Int => kind: String, end_line: Int }",
        )?;

        // Create file_ranks relation for PageRank scores — idempotent.
        self.run_mut_ignore(
            ":create file_ranks { file_path: String => pagerank: Float }",
        )?;

        Ok(())
    }

    async fn upsert_file(&self, record: &FileRecord) -> anyhow::Result<()> {
        let mut p = BTreeMap::new();
        p.insert("fp".into(), Self::dv_str(&record.file_path));
        p.insert("lang".into(), Self::dv_str(&record.language));
        p.insert("lm".into(), Self::dv_int(record.last_modified));
        p.insert("li".into(), Self::dv_int(record.last_indexed));
        p.insert("cc".into(), Self::dv_int(record.chunk_count as i64));
        self.run_mut(
            "?[file_path, language, last_modified, last_indexed, chunk_count] <- [[$fp, $lang, $lm, $li, $cc]] \
             :put files { file_path => language, last_modified, last_indexed, chunk_count }",
            p,
        )?;
        Ok(())
    }

    async fn delete_file(&self, file_path: &str) -> anyhow::Result<()> {
        let mut p = BTreeMap::new();
        p.insert("fp".into(), Self::dv_str(file_path));
        self.run_mut(
            "?[file_path] <- [[$fp]] :rm files { file_path }",
            p,
        )?;
        Ok(())
    }

    async fn list_indexed_paths(&self) -> anyhow::Result<Vec<String>> {
        let rows = self.run_imm(
            "?[file_path] := *files[file_path, _, _, _, _]",
            BTreeMap::new(),
        )?;
        rows.rows
            .iter()
            .map(|r| Self::str_col(&r[0]))
            .collect()
    }

    async fn upsert_chunks(&self, chunks: &[ChunkRecord]) -> anyhow::Result<()> {
        if chunks.is_empty() {
            return Ok(());
        }
        let dim = self.dim.load(Ordering::Relaxed);
        const BATCH_SIZE: usize = 500;

        for batch in chunks.chunks(BATCH_SIZE) {

            let rows: Vec<Vec<DataValue>> = batch
                .into_iter()
                .map(|c| {
                    vec![
                        Self::dv_str(&c.file_path),
                        Self::dv_int(c.chunk_idx as i64),
                        Self::dv_str(&c.content),
                        Self::dv_str(&c.normalized),
                        Self::dv_str(&c.chunk_type),
                        Self::dv_int(c.start_line as i64),
                        Self::dv_int(c.end_line as i64),
                        Self::embedding_to_dv(&c.embedding, dim),
                    ]
                })
                .collect();

            let data = DataValue::List(
                rows.into_iter()
                    .map(|r| DataValue::List(r))
                    .collect(),
            );

            let mut p = BTreeMap::new();
            p.insert("rows".into(), data);

            self.run_mut(
                "?[file_path, chunk_idx, content, normalized, chunk_type, start_line, end_line, embedding] <- $rows \
                 :put chunks { file_path, chunk_idx => content, normalized, chunk_type, start_line, end_line, embedding }",
                p,
            )?;
        }
        Ok(())
    }

    async fn delete_chunks_for_file(&self, file_path: &str) -> anyhow::Result<()> {
        let mut p = BTreeMap::new();
        p.insert("fp".into(), Self::dv_str(file_path));
        self.run_mut(
            "?[file_path, chunk_idx] := *chunks[file_path, chunk_idx, _, _, _, _, _, _], file_path = $fp \
             :rm chunks { file_path, chunk_idx }",
            p,
        )?;
        Ok(())
    }

    async fn get_chunks_for_file(&self, file_path: &str) -> anyhow::Result<Vec<ChunkRecord>> {
        let mut p = BTreeMap::new();
        p.insert("fp".into(), Self::dv_str(file_path));
        let rows = self.run_imm(
            "?[file_path, chunk_idx, content, normalized, chunk_type, start_line, end_line, embedding] \
             := *chunks[$fp, chunk_idx, content, normalized, chunk_type, start_line, end_line, embedding], \
                file_path = $fp",
            p,
        )?;
        rows.rows.iter().map(|r| Self::row_to_chunk(r)).collect()
    }

    async fn get_chunks_for_files(&self, file_paths: &[&str]) -> anyhow::Result<Vec<ChunkRecord>> {
        if file_paths.is_empty() {
            return Ok(vec![]);
        }
        let fps = DataValue::List(file_paths.iter().map(|fp| Self::dv_str(fp)).collect());
        let mut p = BTreeMap::new();
        p.insert("fps".into(), fps);
        let rows = self.run_imm(
            "?[file_path, chunk_idx, content, normalized, chunk_type, start_line, end_line, embedding] \
             := *chunks[file_path, chunk_idx, content, normalized, chunk_type, start_line, end_line, embedding], \
             is_in(file_path, $fps)",
            p,
        )?;
        rows.rows.iter().map(|r| Self::row_to_chunk(r)).collect()
    }

    async fn upsert_edges(&self, edges: &[EdgeRecord]) -> anyhow::Result<()> {
        if edges.is_empty() {
            return Ok(());
        }
        let now = chrono::Utc::now().timestamp();
        const BATCH_SIZE: usize = 500;

        for batch in edges.chunks(BATCH_SIZE) {
            let rows: Vec<Vec<DataValue>> = batch
                .iter()
                .map(|e| {
                    vec![
                        Self::dv_str(&e.from_file),
                        Self::dv_int(e.from_chunk as i64),
                        Self::dv_str(&e.to_file),
                        Self::dv_str(&e.edge_type),
                        Self::dv_int(now),
                    ]
                })
                .collect();

            let data = DataValue::List(
                rows.into_iter()
                    .map(|r| DataValue::List(r))
                    .collect(),
            );

            let mut p = BTreeMap::new();
            p.insert("rows".into(), data);

            self.run_mut(
                "?[from_file, from_chunk, to_file, edge_type, created_at] <- $rows \
                 :put code_edges { from_file, from_chunk, to_file => edge_type, created_at }",
                p,
            )?;
        }
        Ok(())
    }

    async fn delete_edges_for_file(&self, file_path: &str) -> anyhow::Result<()> {
        let mut p = BTreeMap::new();
        p.insert("fp".into(), Self::dv_str(file_path));
        self.run_mut(
            "?[from_file, from_chunk, to_file] := *code_edges[from_file, from_chunk, to_file, _, _], from_file = $fp \
             :rm code_edges { from_file, from_chunk, to_file }",
            p,
        )?;
        Ok(())
    }

    async fn get_importers(&self, file_path: &str) -> anyhow::Result<Vec<String>> {
        let mut p = BTreeMap::new();
        p.insert("tf".into(), Self::dv_str(file_path));
        let rows = self.run_imm(
            "?[from_file] := *code_edges[from_file, _, $tf, _, _]",
            p,
        )?;
        rows.rows.iter().map(|r| Self::str_col(&r[0])).collect()
    }

    async fn get_imports(&self, file_path: &str) -> anyhow::Result<Vec<String>> {
        let mut p = BTreeMap::new();
        p.insert("fp".into(), Self::dv_str(file_path));
        let rows = self.run_imm(
            "?[to_file] := *code_edges[$fp, _, to_file, _, _]",
            p,
        )?;
        rows.rows.iter().map(|r| Self::str_col(&r[0])).collect()
    }

    async fn traverse_imports(&self, file_path: &str, max_depth: usize, edge_types: Option<&[&str]>) -> anyhow::Result<Vec<(String, usize)>> {
        // edge_types=Some(&[]) means caller wants zero edge types: no results.
        if let Some(types) = edge_types {
            if types.is_empty() {
                return Ok(vec![]);
            }
        }
        if max_depth == 0 {
            return Ok(vec![]);
        }

        let mut p = BTreeMap::new();
        p.insert("start".into(), Self::dv_str(file_path));
        p.insert("max_depth".into(), DataValue::Num(cozo::Num::Int(max_depth as i64)));

        // Single recursive Datalog query replaces the Rust BFS loop.
        // CozoDB handles cycles via stratification; min(depth) keeps shortest path.
        let script = if let Some(types) = edge_types {
            let types_dv = DataValue::List(types.iter().map(|t| Self::dv_str(t)).collect());
            p.insert("edge_types".into(), types_dv);
            "reach[to_file, d] := *code_edges[$start, _, to_file, edge_type, _], \
                 is_in(edge_type, $edge_types), to_file != $start, d = 1\n\
             reach[to_file, d] := reach[mid, prev], d = prev + 1, d <= $max_depth, \
                 *code_edges[mid, _, to_file, edge_type, _], \
                 is_in(edge_type, $edge_types), to_file != $start\n\
             ?[to_file, min(depth)] := reach[to_file, depth]"
        } else {
            "reach[to_file, d] := *code_edges[$start, _, to_file, _, _], to_file != $start, d = 1\n\
             reach[to_file, d] := reach[mid, prev], d = prev + 1, d <= $max_depth, \
                 *code_edges[mid, _, to_file, _, _], to_file != $start\n\
             ?[to_file, min(depth)] := reach[to_file, depth]"
        };

        let rows = self.run_imm(script, p)?;
        rows.rows
            .iter()
            .map(|r| Ok((Self::str_col(&r[0])?, Self::int_col(&r[1])? as usize)))
            .collect()
    }

    async fn traverse_importers(&self, file_path: &str, max_depth: usize, edge_types: Option<&[&str]>) -> anyhow::Result<Vec<(String, usize)>> {
        // edge_types=Some(&[]) means caller wants zero edge types: no results.
        if let Some(types) = edge_types {
            if types.is_empty() {
                return Ok(vec![]);
            }
        }
        if max_depth == 0 {
            return Ok(vec![]);
        }

        let mut p = BTreeMap::new();
        p.insert("start".into(), Self::dv_str(file_path));
        p.insert("max_depth".into(), DataValue::Num(cozo::Num::Int(max_depth as i64)));

        // Single recursive Datalog query — reverse direction (to_file → from_file).
        // CozoDB handles cycles; min(depth) keeps shortest path to each importer.
        let script = if let Some(types) = edge_types {
            let types_dv = DataValue::List(types.iter().map(|t| Self::dv_str(t)).collect());
            p.insert("edge_types".into(), types_dv);
            "reach[from_file, d] := *code_edges[from_file, _, $start, edge_type, _], \
                 is_in(edge_type, $edge_types), from_file != $start, d = 1\n\
             reach[from_file, d] := reach[mid, prev], d = prev + 1, d <= $max_depth, \
                 *code_edges[from_file, _, mid, edge_type, _], \
                 is_in(edge_type, $edge_types), from_file != $start\n\
             ?[from_file, min(depth)] := reach[from_file, depth]"
        } else {
            "reach[from_file, d] := *code_edges[from_file, _, $start, _, _], from_file != $start, d = 1\n\
             reach[from_file, d] := reach[mid, prev], d = prev + 1, d <= $max_depth, \
                 *code_edges[from_file, _, mid, _, _], from_file != $start\n\
             ?[from_file, min(depth)] := reach[from_file, depth]"
        };

        let rows = self.run_imm(script, p)?;
        rows.rows
            .iter()
            .map(|r| Ok((Self::str_col(&r[0])?, Self::int_col(&r[1])? as usize)))
            .collect()
    }

    #[tracing::instrument(skip_all, fields(top_k))]
    async fn hybrid_search(
        &self,
        query_vec: &[f32],
        query_str: &str,
        top_k: usize,
    ) -> anyhow::Result<Vec<SearchResult>> {
        // Single round-trip: count total chunks and chunks-with-embeddings together.
        let guard_rows = self.run_imm(
            "total[count(fp)] := *chunks[fp, _, _, _, _, _, _, _]\n\
             with_emb[count(fp)] := *chunks[fp, _, _, _, _, _, _, emb], !is_null(emb)\n\
             ?[t, e] := total[t], with_emb[e]",
            BTreeMap::new(),
        )?;
        let (total, emb_count) = guard_rows
            .rows
            .first()
            .map(|r| {
                let t = match &r[0] { DataValue::Num(cozo::Num::Int(n)) => *n, _ => 0 };
                let e = match &r[1] { DataValue::Num(cozo::Num::Int(n)) => *n, _ => 0 };
                (t, e)
            })
            .unwrap_or((0, 0));
        if total == 0 {
            return Ok(vec![]);
        }
        if emb_count == 0 {
            // No embeddings yet — fall back to FTS only.
            return self.fts_only_search(query_str, top_k);
        }

        // Fetch extra candidates from each leg so the fusion has material to rank.
        let fetch_k = (top_k * 2).max(50);

        // Run FTS and HNSW concurrently. CozoBackend wraps a DbInstance which is
        // Send + Sync, so scoped threads give us true parallelism on the RocksDB
        // backend. On SQLite the gain is smaller (single shared connection) but
        // the code is still correct either way.
        let (fts_results, vec_results) = std::thread::scope(|s| {
            let fts_handle = s.spawn(|| self.fts_search(query_str, fetch_k));
            let vec_handle = s.spawn(|| self.vector_search(query_vec, fetch_k));
            (fts_handle.join().unwrap(), vec_handle.join().unwrap())
        });
        let fts_results = fts_results?;
        let vec_results = vec_results?;

        // Rank maps: (file_path, chunk_idx) -> 1-indexed rank.
        // FTS: rank 1 = highest bm25 (results arrive score-descending).
        // Vector: rank 1 = smallest distance (results arrive distance-ascending).
        use std::collections::HashMap;
        type ChunkKey = (String, usize);
        type ChunkMeta = (String, String, usize, usize); // (content, chunk_type, start, end)

        let mut chunk_meta: HashMap<ChunkKey, ChunkMeta> = HashMap::new();
        let mut fts_rank: HashMap<ChunkKey, usize> = HashMap::new();
        let mut vec_rank: HashMap<ChunkKey, usize> = HashMap::new();

        for (rank, (fp, ci, _score, content, chunk_type, start_line, end_line)) in
            fts_results.into_iter().enumerate()
        {
            let key = (fp, ci);
            fts_rank.insert(key.clone(), rank + 1);
            chunk_meta.entry(key).or_insert((content, chunk_type, start_line, end_line));
        }
        for (rank, (fp, ci, _dist, content, chunk_type, start_line, end_line)) in
            vec_results.into_iter().enumerate()
        {
            let key = (fp, ci);
            vec_rank.insert(key.clone(), rank + 1);
            chunk_meta.entry(key).or_insert((content, chunk_type, start_line, end_line));
        }

        // RRF fusion: score = 0.55 / (60 + fts_rank) + 0.45 / (60 + vec_rank).
        // Chunks absent from one list use sentinel rank 1000.
        const SENTINEL: usize = 1000;

        let mut scored: Vec<(f64, ChunkKey, ChunkMeta, &'static str)> = chunk_meta
            .into_iter()
            .map(|(key, meta)| {
                let fr = fts_rank.get(&key).copied().unwrap_or(SENTINEL);
                let vr = vec_rank.get(&key).copied().unwrap_or(SENTINEL);
                let rrf = 0.55 / (60.0 + fr as f64) + 0.45 / (60.0 + vr as f64);
                let why = match (fts_rank.contains_key(&key), vec_rank.contains_key(&key)) {
                    (true, true) => "hybrid",
                    (true, false) => "fts",
                    (false, _) => "vector",
                };
                (rrf, key, meta, why)
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        scored
            .into_iter()
            .map(|(rrf, (file_path, chunk_idx), (content, chunk_type, start_line, end_line), why)| {
                Ok(SearchResult {
                    file_path,
                    chunk_idx,
                    content,
                    start_line,
                    end_line,
                    chunk_type,
                    score: rrf,
                    match_quality: String::new(),
                    why: why.to_string(),
                })
            })
            .collect()
    }

    /// Unified single-RTT retrieval: FTS + HNSW fused with score-based weighting,
    /// optional single-hop graph expansion, and PageRank boost - all in one Datalog
    /// round-trip.  Post-processing in Rust: dedup by (file_path, chunk_idx) keeping
    /// the highest score, then truncate to `top_k`.
    ///
    /// Fallbacks:
    /// - `file_ranks` always exists after `initialize()` so the boost join is safe.
    /// - If the graph produces no edges, the graph rule contributes zero rows;
    ///   the base union still returns FTS+HNSW results.
    ///
    /// CozoDB constraints applied:
    /// - Proximity search patterns (`~chunks:text`, `~chunks:semantic`) require
    ///   the actual column names (`file_path`, `chunk_idx`), not arbitrary vars.
    /// - Expressions are computed in rule bodies (no expressions in rule heads).
    #[tracing::instrument(skip_all, fields(top_k, graph_depth))]
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
        // Guard: empty index - nothing to search.
        let guard_rows = self.run_imm(
            "total[count(fp)] := *chunks[fp, _, _, _, _, _, _, _]
\
             with_emb[count(fp)] := *chunks[fp, _, _, _, _, _, _, emb], !is_null(emb)
\
             ?[t, e] := total[t], with_emb[e]",
            BTreeMap::new(),
        )?;
        let (total, emb_count) = guard_rows
            .rows
            .first()
            .map(|r| {
                let t = match &r[0] { DataValue::Num(cozo::Num::Int(n)) => *n, _ => 0 };
                let e = match &r[1] { DataValue::Num(cozo::Num::Int(n)) => *n, _ => 0 };
                (t, e)
            })
            .unwrap_or((0, 0));
        if total == 0 {
            return Ok(vec![]);
        }
        if emb_count == 0 {
            return self.fts_only_search(query_str, top_k);
        }

        let fetch_k = (top_k * 3).max(100);
        let result_limit = fetch_k * 2;

        let query_vec_dv: DataValue = {
            let arr = ndarray::Array1::from(query_vec.to_vec());
            DataValue::Vec(cozo::Vector::F32(arr))
        };

        // Build the combined Datalog script. Rules are newline-separated.
        //
        // Key constraints satisfied:
        // 1. Proximity searches use actual column names (file_path, chunk_idx).
        // 2. All expressions in rule BODIES (CozoDB forbids head expressions).
        // 3. Aggregations (sum, max) are valid in rule heads.
        // 4. NAF `not *file_ranks[file_path, _]` is valid because file_ranks
        //    always exists after initialize() and file_path is grounded.
        //
        // Scoring:
        //   FTS BM25: norm = bm25 / (bm25 + 1.0) -> [0,1)
        //   HNSW cosine dist [0,2]: sim = 1.0 - dist -> (-1,1]
        //   base fuses: 0.55 * fts + 0.45 * vec (mirrors hybrid_search RRF weights).
        //   PageRank boost: 1.0 + 0.1 * pr (linear; avoids ln() availability).
        //   Graph: 0.3 * parent_score for depth-1 import neighbors.
        let with_graph = graph_depth > 0;
        let script = if with_graph {
            format!(
                concat!(
                    "fts[file_path, chunk_idx, norm] :=
",
                    "    ~chunks:text{{ file_path, chunk_idx | query: $qs, k: {fk}, score_kind: 'tf_idf', bind_score: bm25 }},
",
                    "    norm = bm25 / (bm25 + 1.0)
",
                    "
",
                    "vec[file_path, chunk_idx, sim] :=
",
                    "    ~chunks:semantic{{ file_path, chunk_idx | query: $qv, k: {fk}, ef: 64, bind_distance: dist }},
",
                    "    sim = 1.0 - dist
",
                    "
",
                    "base[file_path, chunk_idx, sum(s)] :=
",
                    "    fts[file_path, chunk_idx, raw], s = {fw} * raw
",
                    "base[file_path, chunk_idx, sum(s)] :=
",
                    "    vec[file_path, chunk_idx, raw], s = {vw} * raw
",
                    "
",
                    "graph[file_path, chunk_idx, max(s)] :=
",
                    "    base[target_fp, _, parent_score], parent_score > {gms},
",
                    "    *code_edges[file_path, _, target_fp, _, _],
",
                    "    *chunks[file_path, chunk_idx, _, _, _, _, _, emb], !is_null(emb),
",
                    "    s = parent_score * {gsf}
",
                    "
",
                    "boosted[file_path, chunk_idx, bscore] :=
",
                    "    base[file_path, chunk_idx, score], *file_ranks[file_path, pr],
",
                    "    boost = 1.0 + {prf} * pr, bscore = score * boost
",
                    "boosted[file_path, chunk_idx, score] :=
",
                    "    base[file_path, chunk_idx, score], not *file_ranks[file_path, _]
",
                    "
",
                    "?[file_path, chunk_idx, content, chunk_type, start_line, end_line, score, why] :=
",
                    "    boosted[file_path, chunk_idx, score],
",
                    "    *chunks[file_path, chunk_idx, content, _, chunk_type, start_line, end_line, _],
",
                    "    why = 'hybrid'
",
                    "?[file_path, chunk_idx, content, chunk_type, start_line, end_line, score, why] :=
",
                    "    graph[file_path, chunk_idx, score],
",
                    "    *chunks[file_path, chunk_idx, content, _, chunk_type, start_line, end_line, _],
",
                    "    why = 'graph'
",
                    ":order -score
",
                    ":limit {rl}"
                ),
                fk = fetch_k,
                rl = result_limit,
                fw = fts_weight,
                vw = 1.0 - fts_weight,
                gsf = graph_score_factor,
                gms = graph_min_score,
                prf = pagerank_factor,
            )
        } else {
            format!(
                concat!(
                    "fts[file_path, chunk_idx, norm] :=
",
                    "    ~chunks:text{{ file_path, chunk_idx | query: $qs, k: {fk}, score_kind: 'tf_idf', bind_score: bm25 }},
",
                    "    norm = bm25 / (bm25 + 1.0)
",
                    "
",
                    "vec[file_path, chunk_idx, sim] :=
",
                    "    ~chunks:semantic{{ file_path, chunk_idx | query: $qv, k: {fk}, ef: 64, bind_distance: dist }},
",
                    "    sim = 1.0 - dist
",
                    "
",
                    "base[file_path, chunk_idx, sum(s)] :=
",
                    "    fts[file_path, chunk_idx, raw], s = {fw} * raw
",
                    "base[file_path, chunk_idx, sum(s)] :=
",
                    "    vec[file_path, chunk_idx, raw], s = {vw} * raw
",
                    "
",
                    "boosted[file_path, chunk_idx, bscore] :=
",
                    "    base[file_path, chunk_idx, score], *file_ranks[file_path, pr],
",
                    "    boost = 1.0 + {prf} * pr, bscore = score * boost
",
                    "boosted[file_path, chunk_idx, score] :=
",
                    "    base[file_path, chunk_idx, score], not *file_ranks[file_path, _]
",
                    "
",
                    "?[file_path, chunk_idx, content, chunk_type, start_line, end_line, score] :=
",
                    "    boosted[file_path, chunk_idx, score],
",
                    "    *chunks[file_path, chunk_idx, content, _, chunk_type, start_line, end_line, _]
",
                    ":order -score
",
                    ":limit {rl}"
                ),
                fk = fetch_k,
                rl = result_limit,
                fw = fts_weight,
                vw = 1.0 - fts_weight,
                prf = pagerank_factor,
            )
        };

        let mut p = BTreeMap::new();
        p.insert("qs".into(), Self::dv_str(query_str));
        p.insert("qv".into(), query_vec_dv);

        let rows = match self.run_imm(&script, p) {
            Ok(r) => r,
            Err(e) => {
                let msg = e.to_string();
                // Some CozoDB errors surface when a query leg returns zero results.
                if msg.contains("no results") || msg.contains("empty") {
                    return Ok(vec![]);
                }
                return Err(e);
            }
        };

        // Parse raw rows. Column layout:
        //   with_graph:    [file_path, chunk_idx, content, chunk_type, start_line, end_line, score, why]
        //   without_graph: [file_path, chunk_idx, content, chunk_type, start_line, end_line, score]
        // Dedup by (file_path, chunk_idx) keeping highest score (handles base/graph overlap).
        use std::collections::HashMap;
        let mut seen: HashMap<(String, usize), SearchResult> = HashMap::new();
        for row in &rows.rows {
            let file_path = match Self::str_col(&row[0]) { Ok(v) => v, Err(_) => continue };
            let chunk_idx = match Self::int_col(&row[1]) { Ok(v) => v as usize, Err(_) => continue };
            let content = match Self::str_col(&row[2]) { Ok(v) => v, Err(_) => continue };
            let chunk_type = match Self::str_col(&row[3]) { Ok(v) => v, Err(_) => continue };
            let start_line = match Self::int_col(&row[4]) { Ok(v) => v as usize, Err(_) => continue };
            let end_line = match Self::int_col(&row[5]) { Ok(v) => v as usize, Err(_) => continue };
            let score = match &row[6] {
                DataValue::Num(cozo::Num::Float(f)) => *f,
                DataValue::Num(cozo::Num::Int(i)) => *i as f64,
                _ => continue,
            };
            let why = if with_graph {
                match Self::str_col(&row[7]) {
                    Ok(v) => v,
                    Err(_) => "hybrid".to_string(),
                }
            } else {
                "hybrid".to_string()
            };
            let key = (file_path.clone(), chunk_idx);
            let entry = seen.entry(key).or_insert_with(|| SearchResult {
                file_path,
                chunk_idx,
                content,
                start_line,
                end_line,
                chunk_type,
                score,
                match_quality: String::new(),
                why,
            });
            // Keep the highest-scored representation if duplicated across rules.
            if score > entry.score {
                entry.score = score;
            }
        }

        let mut results: Vec<SearchResult> = seen.into_values().collect();
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        Ok(results)
    }


    async fn stats(&self) -> anyhow::Result<IndexStats> {
        // Single query: fc and cc use count (returns 0 on empty), ml uses max
        // (returns nothing on empty). When files is empty the join yields no rows;
        // unwrap_or covers that case.
        let rows = self.run_imm(
            "fc[count(fp)] := *files[fp, _, _, _, _]\n\
             cc[count(fp)] := *chunks[fp, _, _, _, _, _, _, _]\n\
             ml[max(li)] := *files[_, _, _, li, _]\n\
             ?[f, c, m] := fc[f], cc[c], ml[m]",
            BTreeMap::new(),
        )?;

        let (indexed_files, total_chunks, max_li) = rows
            .rows
            .first()
            .map(|r| {
                let f = match &r[0] {
                    DataValue::Num(cozo::Num::Int(n)) => *n as usize,
                    _ => 0,
                };
                let c = match &r[1] {
                    DataValue::Num(cozo::Num::Int(n)) => *n as usize,
                    _ => 0,
                };
                let m = match &r[2] {
                    DataValue::Num(cozo::Num::Int(n)) => *n,
                    _ => 0,
                };
                (f, c, m)
            })
            .unwrap_or((0, 0, 0));

        let last_indexed = if indexed_files == 0 {
            None
        } else {
            DateTime::from_timestamp(max_li, 0)
        };

        Ok(IndexStats {
            indexed_files,
            total_chunks,
            last_indexed,
            watching: false,
            estimated_stale: 0,
        })
    }

    async fn upsert_symbols(&self, symbols: &[SymbolDef]) -> anyhow::Result<()> {
        if symbols.is_empty() {
            return Ok(());
        }
        const BATCH_SIZE: usize = 500;
        for batch in symbols.chunks(BATCH_SIZE) {
            let rows: Vec<Vec<DataValue>> = batch
                .iter()
                .map(|s| {
                    vec![
                        Self::dv_str(&s.file_path),
                        Self::dv_str(&s.name),
                        Self::dv_int(s.start_line as i64),
                        Self::dv_str(&s.kind),
                        Self::dv_int(s.end_line as i64),
                    ]
                })
                .collect();
            let data = DataValue::List(
                rows.into_iter().map(|r| DataValue::List(r)).collect(),
            );
            let mut p = BTreeMap::new();
            p.insert("rows".into(), data);
            self.run_mut(
                "?[file_path, name, start_line, kind, end_line] <- $rows \
                 :put symbols { file_path, name, start_line => kind, end_line }",
                p,
            )?;
        }
        Ok(())
    }

    async fn delete_symbols_for_file(&self, file_path: &str) -> anyhow::Result<()> {
        let mut p = BTreeMap::new();
        p.insert("fp".into(), Self::dv_str(file_path));
        self.run_mut(
            "?[file_path, name, start_line] := *symbols[file_path, name, start_line, _, _], file_path = $fp \
             :rm symbols { file_path, name, start_line }",
            p,
        )?;
        Ok(())
    }

    async fn find_symbols(&self, name: &str, kind: Option<&str>) -> anyhow::Result<Vec<SymbolDef>> {
        let mut p = BTreeMap::new();
        p.insert("name".into(), Self::dv_str(name));
        let rows = if let Some(k) = kind {
            p.insert("kind".into(), Self::dv_str(k));
            self.run_imm(
                "?[file_path, name, kind, start_line, end_line] := \
                   *symbols[file_path, name, start_line, kind, end_line], \
                   name = $name, kind = $kind",
                p,
            )?
        } else {
            self.run_imm(
                "?[file_path, name, kind, start_line, end_line] := \
                   *symbols[file_path, name, start_line, kind, end_line], \
                   name = $name",
                p,
            )?
        };
        rows.rows
            .iter()
            .map(|r| {
                Ok(SymbolDef {
                    file_path: Self::str_col(&r[0])?,
                    name: Self::str_col(&r[1])?,
                    kind: Self::str_col(&r[2])?,
                    start_line: Self::int_col(&r[3])? as usize,
                    end_line: Self::int_col(&r[4])? as usize,
                })
            })
            .collect()
    }

    async fn get_chunk_embeddings(&self, keys: &[(String, usize)]) -> anyhow::Result<Vec<Vec<f32>>> {
        let dim = self.dim.load(Ordering::Relaxed);
        let zero = vec![0.0f32; dim];

        if keys.is_empty() {
            return Ok(vec![]);
        }

        // Collect unique file paths for the batched is_in() query.
        let unique_fps: Vec<DataValue> = keys
            .iter()
            .map(|(fp, _)| fp.as_str())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .map(|fp| Self::dv_str(fp))
            .collect();

        let mut p = BTreeMap::new();
        p.insert("fps".into(), DataValue::List(unique_fps));

        let rows = self.run_imm(
            "?[fp, ci, emb] := *chunks[fp, ci, _, _, _, _, _, emb], is_in(fp, $fps)",
            p,
        )?;

        // Build a HashMap keyed by (file_path, chunk_idx) for O(1) lookup.
        let mut emb_map: std::collections::HashMap<(String, usize), Vec<f32>> =
            std::collections::HashMap::new();
        for row in rows.rows {
            let fp = match &row[0] {
                DataValue::Str(s) => s.to_string(),
                _ => continue,
            };
            let ci = match &row[1] {
                DataValue::Num(cozo::Num::Int(i)) => *i as usize,
                _ => continue,
            };
            let emb = match &row[2] {
                DataValue::List(items) if !items.is_empty() => items
                    .iter()
                    .map(|d| match d {
                        DataValue::Num(cozo::Num::Float(f)) => *f as f32,
                        DataValue::Num(cozo::Num::Int(i)) => *i as f32,
                        _ => 0.0,
                    })
                    .collect(),
                _ => continue,
            };
            emb_map.insert((fp, ci), emb);
        }

        // Reconstruct results in original input order, falling back to zero vector.
        let result = keys
            .iter()
            .map(|(fp, ci)| {
                emb_map
                    .remove(&(fp.clone(), *ci))
                    .unwrap_or_else(|| zero.clone())
            })
            .collect();

        Ok(result)
    }

    async fn compute_pagerank(&self, edge_types: Option<&[&str]>) -> anyhow::Result<()> {
        // Extract file-level edges from CozoDB, compute PageRank in Rust,
        // and store results back.  CozoDB's built-in PageRank requires the
        // `graph-algo` feature which has a broken dependency (graph_builder
        // vs rayon incompatibility).  This pure-Rust implementation avoids
        // that while CozoDB development is stalled (see ADR-002).

        // 1. Extract unique directed edges: from_file -> to_file
        let edge_rows = if let Some(types) = edge_types {
            if types.is_empty() {
                // No edge types requested: treat as empty graph, reset ranks.
                let _ = self.run_mut(
                    ":replace file_ranks { file_path => pagerank }",
                    BTreeMap::new(),
                );
                return Ok(());
            }
            let types_dv = DataValue::List(types.iter().map(|t| Self::dv_str(t)).collect());
            let mut p = BTreeMap::new();
            p.insert("edge_types".into(), types_dv);
            self.run_imm(
                "?[f, t] := *code_edges[f, _, t, edge_type, _], is_in(edge_type, $edge_types)",
                p,
            )?
        } else {
            self.run_imm("?[f, t] := *code_edges[f, _, t, _, _]", BTreeMap::new())?
        };

        let mut node_to_id: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        let mut id_to_node: Vec<String> = Vec::new();
        let mut edges: Vec<(usize, usize)> = Vec::new();

        for row in &edge_rows.rows {
            if let (DataValue::Str(from), DataValue::Str(to)) = (&row[0], &row[1]) {
                let from_s = from.to_string();
                let to_s = to.to_string();
                let from_id = *node_to_id.entry(from_s.clone()).or_insert_with(|| {
                    let id = id_to_node.len();
                    id_to_node.push(from_s);
                    id
                });
                let to_id = *node_to_id.entry(to_s.clone()).or_insert_with(|| {
                    let id = id_to_node.len();
                    id_to_node.push(to_s);
                    id
                });
                edges.push((from_id, to_id));
            }
        }

        let n = id_to_node.len();
        if n == 0 {
            // Empty graph — clear file_ranks and return.
            let _ = self.run_mut(
                ":replace file_ranks { file_path => pagerank }",
                BTreeMap::new(),
            );
            return Ok(());
        }

        // 2. Build adjacency: out_degree[from] and inbound[to] = vec of from nodes.
        let mut out_degree = vec![0usize; n];
        let mut inbound: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &(from, to) in &edges {
            out_degree[from] += 1;
            inbound[to].push(from);
        }

        // 3. Power iteration.
        let damping = 0.85_f64;
        let epsilon = 0.0001_f64;
        let max_iter = 20;
        let n_f = n as f64;
        let mut rank = vec![1.0 / n_f; n];
        let mut new_rank = vec![0.0_f64; n];

        for _ in 0..max_iter {
            let sink_rank: f64 = rank.iter().enumerate()
                .filter(|(i, _)| out_degree[*i] == 0)
                .map(|(_, r)| r)
                .sum();

            for i in 0..n {
                let mut incoming_sum = 0.0_f64;
                for &src in &inbound[i] {
                    incoming_sum += rank[src] / out_degree[src] as f64;
                }
                new_rank[i] = (1.0 - damping) / n_f
                    + damping * (incoming_sum + sink_rank / n_f);
            }

            let delta: f64 = rank.iter().zip(new_rank.iter())
                .map(|(old, new)| (old - new).abs())
                .sum();

            std::mem::swap(&mut rank, &mut new_rank);

            if delta < epsilon {
                break;
            }
        }

        // 4. Store results back into CozoDB via parameterised upsert to avoid
        //    injection from file paths containing `"`, `\n`, or `]`.
        let rows_dv: Vec<DataValue> = id_to_node
            .iter()
            .enumerate()
            .map(|(i, path)| DataValue::List(vec![Self::dv_str(path), Self::dv_float(rank[i])]))
            .collect();
        let data = DataValue::List(rows_dv);
        let mut p = BTreeMap::new();
        p.insert("rows".into(), data);
        self.run_mut(
            "?[file_path, pagerank] <- $rows\n:replace file_ranks { file_path => pagerank }",
            p,
        )?;
        // Phase B1 stub: when `edge_types` is `Some` with a single type (e.g. "calls"),
        // results should be stored under a separate `rank_type`-prefixed relation so
        // callers can retrieve per-edge-type PageRank independently from import-graph rank.
        // The relation does not exist yet; this is the insertion point once Phase B1
        // adds call-edge support.
        Ok(())
    }

    async fn get_file_ranks(
        &self,
        file_paths: &[&str],
    ) -> anyhow::Result<std::collections::HashMap<String, f64>> {
        // When file_paths is empty there are no result files to boost — return
        // an empty map rather than pulling the entire table for nothing.
        if file_paths.is_empty() {
            return Ok(std::collections::HashMap::new());
        }
        // Filter to only the requested paths so callers get per-query data,
        // not the global rank distribution.
        let fps_dv = DataValue::List(
            file_paths.iter().map(|fp| Self::dv_str(fp)).collect(),
        );
        let mut p = BTreeMap::new();
        p.insert("fps".into(), fps_dv);
        let rows = self.run_imm(
            "?[file_path, pagerank] := *file_ranks[file_path, pagerank], is_in(file_path, $fps)",
            p,
        )?;
        let mut ranks = std::collections::HashMap::new();
        for row in &rows.rows {
            if let (DataValue::Str(fp), DataValue::Num(cozo::Num::Float(pr))) =
                (&row[0], &row[1])
            {
                ranks.insert(fp.to_string(), *pr);
            }
        }
        Ok(ranks)
    }

    async fn hnsw_neighbors(
        &self,
        seeds: &[(String, usize)],
        max_dist: f64,
        limit: usize,
    ) -> anyhow::Result<Vec<(String, usize, f64)>> {
        if seeds.is_empty() {
            return Ok(vec![]);
        }

        // Build a CozoDB list of [file_path, chunk_idx] pairs for the seeds.
        let seeds_dv = DataValue::List(
            seeds
                .iter()
                .map(|(fp, ci)| {
                    DataValue::List(vec![
                        Self::dv_str(fp),
                        DataValue::Num(cozo::Num::Int(*ci as i64)),
                    ])
                })
                .collect(),
        );
        let mut p = BTreeMap::new();
        p.insert("seeds".into(), seeds_dv);
        p.insert("max_dist".into(), DataValue::Num(cozo::Num::Float(max_dist)));

        // Query layer 0 of the HNSW proximity graph.
        // CozoDB names HNSW adjacency columns as fr_{key_col_name} / to_{key_col_name}.
        // For chunks{file_path: String, chunk_idx: Int}, this yields:
        //   fr_file_path, fr_chunk_idx, to_file_path, to_chunk_idx
        let script = format!(
            r#"?[to_fp, to_ci, dist] :=
                seed <- $seeds,
                seed = [fp, ci],
                *chunks:semantic{{layer: 0, fr_file_path: fp, fr_chunk_idx: ci, to_file_path: to_fp, to_chunk_idx: to_ci, dist, ignore_link: false}},
                dist < $max_dist
            :limit {limit}
            :order dist"#
        );

        // Gracefully return empty on any error — the HNSW index may not exist
        // yet (no embeddings indexed) or the graph may be empty.
        let rows = match self.run_imm(&script, p) {
            Ok(r) => r,
            Err(e) => {
                tracing::debug!(error = %e, "hnsw_neighbors: index query failed (index may not exist yet)");
                return Ok(vec![]);
            }
        };

        let mut results: Vec<(String, usize, f64)> = Vec::new();
        // Deduplicate: multiple seeds may share the same neighbor.
        let mut seen: std::collections::HashSet<(String, usize)> =
            seeds.iter().map(|(fp, ci)| (fp.clone(), *ci)).collect();
        for row in &rows.rows {
            let to_fp = match &row[0] {
                DataValue::Str(s) => s.to_string(),
                _ => continue,
            };
            let to_ci = match &row[1] {
                DataValue::Num(cozo::Num::Int(i)) => *i as usize,
                _ => continue,
            };
            let dist = match &row[2] {
                DataValue::Num(cozo::Num::Float(f)) => *f,
                DataValue::Num(cozo::Num::Int(i)) => *i as f64,
                _ => continue,
            };
            if seen.insert((to_fp.clone(), to_ci)) {
                results.push((to_fp, to_ci, dist));
            }
        }
        Ok(results)
    }

    async fn deduplicate_chunks(&self) -> anyhow::Result<usize> {
        // The LSH index relation `chunks:dedup` groups chunks by MinHash bucket.
        // Schema: { hash: Bytes, src_file_path: String, src_chunk_idx: Int }.
        // Chunks sharing a hash bucket from *different* files are near-duplicates.
        //
        // Strategy: for each hash bucket with entries from >1 file, keep the
        // representative with the lowest (file_path, chunk_idx) and delete the rest.

        // Step 1: Find all (hash, file_path, chunk_idx) triples where the hash
        // bucket contains entries from more than one file.
        let find_dups = "?[hash, fp, ci] := *chunks:dedup{hash, src_file_path: fp, src_chunk_idx: ci} :order hash, fp, ci";

        let rows = match self.run_imm(find_dups, BTreeMap::new()) {
            Ok(r) => r,
            Err(e) => {
                // LSH index may not exist (old database). Not an error.
                tracing::debug!(error = %e, "deduplicate_chunks: LSH index query failed");
                return Ok(0);
            }
        };

        if rows.rows.is_empty() {
            return Ok(0);
        }

        // Step 2: Group by hash bucket. For each bucket with entries from >1 file,
        // keep the first entry (lowest fp/ci) and mark the rest for deletion.
        let mut to_delete: Vec<(String, usize)> = Vec::new();
        let mut current_hash: Option<DataValue> = None;
        let mut bucket_representative: Option<String> = None;
        let mut bucket_has_multi_files = false;
        let mut bucket_extras: Vec<(String, usize)> = Vec::new();

        let flush_bucket = |has_multi: bool, extras: &mut Vec<(String, usize)>, deletions: &mut Vec<(String, usize)>| {
            if has_multi {
                deletions.append(extras);
            }
            extras.clear();
        };

        for row in &rows.rows {
            let hash = &row[0];
            let fp = match Self::str_col(&row[1]) {
                Ok(s) => s,
                Err(_) => continue,
            };
            let ci = match Self::int_col(&row[2]) {
                Ok(i) => i as usize,
                Err(_) => continue,
            };

            if current_hash.as_ref() == Some(hash) {
                // Same bucket — check if from a different file.
                if bucket_representative.as_deref() != Some(&fp) {
                    bucket_has_multi_files = true;
                }
                bucket_extras.push((fp, ci));
            } else {
                // New bucket — flush previous.
                flush_bucket(bucket_has_multi_files, &mut bucket_extras, &mut to_delete);
                current_hash = Some(hash.clone());
                bucket_representative = Some(fp);
                bucket_has_multi_files = false;
            }
        }

        // Flush last bucket.
        flush_bucket(bucket_has_multi_files, &mut bucket_extras, &mut to_delete);

        if to_delete.is_empty() {
            return Ok(0);
        }

        let count = to_delete.len();

        // Step 3: Delete duplicate chunks in batches.
        for batch in to_delete.chunks(200) {
            let keys = DataValue::List(
                batch
                    .iter()
                    .map(|(fp, ci)| {
                        DataValue::List(vec![
                            Self::dv_str(fp),
                            DataValue::Num(cozo::Num::Int(*ci as i64)),
                        ])
                    })
                    .collect(),
            );
            let mut p = BTreeMap::new();
            p.insert("keys".into(), keys);
            self.run_mut(
                "?[file_path, chunk_idx] <- $keys :rm chunks",
                p,
            )?;
        }

        tracing::info!(removed = count, "deduplicated near-duplicate chunks across files");
        Ok(count)
    }
}

// ---------------------------------------------------------------------------
// Blanket impl: Arc<B> delegates to B so Indexer/Searcher can hold Arc<B>
// ---------------------------------------------------------------------------

#[async_trait]
impl<B: StorageBackend> StorageBackend for Arc<B> {
    async fn initialize(&self, dim: usize) -> anyhow::Result<()> {
        (**self).initialize(dim).await
    }

    async fn upsert_file(&self, record: &FileRecord) -> anyhow::Result<()> {
        (**self).upsert_file(record).await
    }

    async fn delete_file(&self, file_path: &str) -> anyhow::Result<()> {
        (**self).delete_file(file_path).await
    }

    async fn list_indexed_paths(&self) -> anyhow::Result<Vec<String>> {
        (**self).list_indexed_paths().await
    }

    async fn upsert_chunks(&self, chunks: &[ChunkRecord]) -> anyhow::Result<()> {
        (**self).upsert_chunks(chunks).await
    }

    async fn delete_chunks_for_file(&self, file_path: &str) -> anyhow::Result<()> {
        (**self).delete_chunks_for_file(file_path).await
    }

    async fn get_chunks_for_file(&self, file_path: &str) -> anyhow::Result<Vec<ChunkRecord>> {
        (**self).get_chunks_for_file(file_path).await
    }

    async fn get_chunks_for_files(&self, file_paths: &[&str]) -> anyhow::Result<Vec<ChunkRecord>> {
        (**self).get_chunks_for_files(file_paths).await
    }

    async fn upsert_edges(&self, edges: &[EdgeRecord]) -> anyhow::Result<()> {
        (**self).upsert_edges(edges).await
    }

    async fn delete_edges_for_file(&self, file_path: &str) -> anyhow::Result<()> {
        (**self).delete_edges_for_file(file_path).await
    }

    async fn get_importers(&self, file_path: &str) -> anyhow::Result<Vec<String>> {
        (**self).get_importers(file_path).await
    }

    async fn get_imports(&self, file_path: &str) -> anyhow::Result<Vec<String>> {
        (**self).get_imports(file_path).await
    }

    async fn traverse_imports(&self, file_path: &str, max_depth: usize, edge_types: Option<&[&str]>) -> anyhow::Result<Vec<(String, usize)>> {
        (**self).traverse_imports(file_path, max_depth, edge_types).await
    }

    async fn traverse_importers(&self, file_path: &str, max_depth: usize, edge_types: Option<&[&str]>) -> anyhow::Result<Vec<(String, usize)>> {
        (**self).traverse_importers(file_path, max_depth, edge_types).await
    }

    async fn hybrid_search(
        &self,
        query_vec: &[f32],
        query_str: &str,
        top_k: usize,
    ) -> anyhow::Result<Vec<SearchResult>> {
        (**self).hybrid_search(query_vec, query_str, top_k).await
    }

    async fn stats(&self) -> anyhow::Result<IndexStats> {
        (**self).stats().await
    }

    async fn upsert_symbols(&self, symbols: &[SymbolDef]) -> anyhow::Result<()> {
        (**self).upsert_symbols(symbols).await
    }

    async fn delete_symbols_for_file(&self, file_path: &str) -> anyhow::Result<()> {
        (**self).delete_symbols_for_file(file_path).await
    }

    async fn find_symbols(&self, name: &str, kind: Option<&str>) -> anyhow::Result<Vec<SymbolDef>> {
        (**self).find_symbols(name, kind).await
    }
    async fn get_chunk_embeddings(&self, keys: &[(String, usize)]) -> anyhow::Result<Vec<Vec<f32>>> {
        (**self).get_chunk_embeddings(keys).await
    }

    async fn compute_pagerank(&self, edge_types: Option<&[&str]>) -> anyhow::Result<()> {
        (**self).compute_pagerank(edge_types).await
    }

    async fn get_file_ranks(
        &self,
        file_paths: &[&str],
    ) -> anyhow::Result<std::collections::HashMap<String, f64>> {
        (**self).get_file_ranks(file_paths).await
    }

    async fn hnsw_neighbors(
        &self,
        seeds: &[(String, usize)],
        max_dist: f64,
        limit: usize,
    ) -> anyhow::Result<Vec<(String, usize, f64)>> {
        (**self).hnsw_neighbors(seeds, max_dist, limit).await
    }

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
        (**self).unified_search(query_vec, query_str, top_k, graph_depth, fts_weight, graph_score_factor, graph_min_score, pagerank_factor).await
    }

    async fn deduplicate_chunks(&self) -> anyhow::Result<usize> {
        (**self).deduplicate_chunks().await
    }
}