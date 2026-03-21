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
    /// Retrieval provenance: `"vector"`, `"fts"`, or `"hybrid"`.
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
            .map_err(|e| anyhow::anyhow!("{}", e))
    }

    fn run_imm(&self, script: &str, params: BTreeMap<String, DataValue>) -> anyhow::Result<NamedRows> {
        self.db
            .run_script(script, params, cozo::ScriptMutability::Immutable)
            .map_err(|e| anyhow::anyhow!("{}", e))
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
    ~chunks:text{{ file_path, chunk_idx | query: $qs, k: {limit}, bind_score: bm25 }},
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
                .iter()
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
        use std::collections::HashSet;
        // edge_types=Some(&[]) means caller wants zero edge types: no results.
        if let Some(types) = edge_types {
            if types.is_empty() {
                return Ok(vec![]);
            }
        }
        if max_depth == 0 {
            return Ok(vec![]);
        }
        let mut visited: HashSet<String> = HashSet::new();
        visited.insert(file_path.to_string());
        let mut frontier = vec![file_path.to_string()];
        let mut result: Vec<(String, usize)> = Vec::new();

        for depth in 1..=max_depth {
            if frontier.is_empty() {
                break;
            }
            // Build frontier as DataValue::List for CozoDB's is_in() predicate.
            let frontier_dv = DataValue::List(
                frontier.iter().map(|k| Self::dv_str(k)).collect(),
            );
            let mut p = BTreeMap::new();
            p.insert("frontier".into(), frontier_dv);

            let query = if let Some(types) = edge_types {
                let types_dv = DataValue::List(types.iter().map(|t| Self::dv_str(t)).collect());
                p.insert("edge_types".into(), types_dv);
                "?[to_file] := *code_edges[from_file, _, to_file, edge_type, _], is_in(from_file, $frontier), is_in(edge_type, $edge_types)"
            } else {
                "?[to_file] := *code_edges[from_file, _, to_file, _, _], is_in(from_file, $frontier)"
            };

            let rows = self.run_imm(query, p)?;

            frontier.clear();
            for row in &rows.rows {
                if let Ok(to_file) = Self::str_col(&row[0]) {
                    if !visited.contains(&to_file) {
                        visited.insert(to_file.clone());
                        result.push((to_file.clone(), depth));
                        frontier.push(to_file);
                    }
                }
            }
        }
        Ok(result)
    }

    async fn traverse_importers(&self, file_path: &str, max_depth: usize, edge_types: Option<&[&str]>) -> anyhow::Result<Vec<(String, usize)>> {
        use std::collections::HashSet;
        // edge_types=Some(&[]) means caller wants zero edge types: no results.
        if let Some(types) = edge_types {
            if types.is_empty() {
                return Ok(vec![]);
            }
        }
        if max_depth == 0 {
            return Ok(vec![]);
        }
        let mut result = Vec::new();
        let mut visited: HashSet<String> = HashSet::new();
        visited.insert(file_path.to_string());
        let mut frontier = vec![file_path.to_string()];

        for depth in 1..=max_depth {
            if frontier.is_empty() {
                break;
            }
            // Build frontier as DataValue::List for CozoDB's is_in() predicate —
            // single batched query instead of N+1 per-node get_importers() calls.
            let frontier_dv = DataValue::List(
                frontier.iter().map(|k| Self::dv_str(k)).collect(),
            );
            let mut p = BTreeMap::new();
            p.insert("frontier".into(), frontier_dv);

            let query = if let Some(types) = edge_types {
                let types_dv = DataValue::List(types.iter().map(|t| Self::dv_str(t)).collect());
                p.insert("edge_types".into(), types_dv);
                "?[from_file] := *code_edges[from_file, _, to_file, edge_type, _], is_in(to_file, $frontier), is_in(edge_type, $edge_types)"
            } else {
                "?[from_file] := *code_edges[from_file, _, to_file, _, _], is_in(to_file, $frontier)"
            };

            let rows = self.run_imm(query, p)?;

            frontier.clear();
            for row in &rows.rows {
                if let Ok(from_file) = Self::str_col(&row[0]) {
                    if visited.insert(from_file.clone()) {
                        result.push((from_file.clone(), depth));
                        frontier.push(from_file);
                    }
                }
            }
        }
        Ok(result)
    }

    #[tracing::instrument(skip_all, fields(top_k))]
    async fn hybrid_search(
        &self,
        query_vec: &[f32],
        query_str: &str,
        top_k: usize,
    ) -> anyhow::Result<Vec<SearchResult>> {
        // Guard: if no chunks exist, searches will fail on empty index.
        let count_rows = self.run_imm(
            "?[count(fp)] := *chunks[fp, _, _, _, _, _, _, _]",
            BTreeMap::new(),
        )?;
        let count = count_rows
            .rows
            .first()
            .and_then(|r| match &r[0] {
                DataValue::Num(cozo::Num::Int(n)) => Some(*n),
                _ => None,
            })
            .unwrap_or(0);
        if count == 0 {
            return Ok(vec![]);
        }

        // Check whether any chunk has an embedding (HNSW requires at least one).
        let emb_count_rows = self.run_imm(
            "?[count(fp)] := *chunks[fp, _, _, _, _, _, _, emb], !is_null(emb)",
            BTreeMap::new(),
        )?;
        let emb_count = emb_count_rows
            .rows
            .first()
            .and_then(|r| match &r[0] {
                DataValue::Num(cozo::Num::Int(n)) => Some(*n),
                _ => None,
            })
            .unwrap_or(0);

        if emb_count == 0 {
            // No embeddings yet — fall back to FTS only.
            return self.fts_only_search(query_str, top_k);
        }

        // Fetch extra candidates from each leg so the fusion has material to rank.
        let fetch_k = (top_k * 2).max(50);

        let fts_results = self.fts_search(query_str, fetch_k)?;
        let vec_results = self.vector_search(query_vec, fetch_k)?;

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

    async fn stats(&self) -> anyhow::Result<IndexStats> {
        let file_rows = self.run_imm(
            "?[count(fp)] := *files[fp, _, _, _, _]",
            BTreeMap::new(),
        )?;
        let indexed_files = file_rows
            .rows
            .first()
            .and_then(|r| match &r[0] {
                DataValue::Num(cozo::Num::Int(n)) => Some(*n as usize),
                _ => None,
            })
            .unwrap_or(0);

        let chunk_rows = self.run_imm(
            "?[count(fp)] := *chunks[fp, _, _, _, _, _, _, _]",
            BTreeMap::new(),
        )?;
        let total_chunks = chunk_rows
            .rows
            .first()
            .and_then(|r| match &r[0] {
                DataValue::Num(cozo::Num::Int(n)) => Some(*n as usize),
                _ => None,
            })
            .unwrap_or(0);

        let last_indexed = if indexed_files == 0 {
            None
        } else {
            let li_rows = self.run_imm(
                "?[max(li)] := *files[_, _, _, li, _]",
                BTreeMap::new(),
            )?;
            li_rows
                .rows
                .first()
                .and_then(|r| match &r[0] {
                    DataValue::Num(cozo::Num::Int(ts)) => {
                        DateTime::from_timestamp(*ts, 0)
                    }
                    _ => None,
                })
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
                    .get(&(fp.clone(), *ci))
                    .cloned()
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
}