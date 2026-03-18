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
}

#[derive(Debug, Clone)]
pub struct IndexStats {
    pub indexed_files: usize,
    pub total_chunks: usize,
    /// `None` when the index contains no files.
    pub last_indexed: Option<DateTime<Utc>>,
    /// Always `false` in v1 (watching is a v2 feature).
    pub watching: bool,
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

    async fn hybrid_search(
        &self,
        query_vec: &[f32],
        query_str: &str,
        top_k: usize,
    ) -> anyhow::Result<Vec<SearchResult>>;

    async fn stats(&self) -> anyhow::Result<IndexStats>;
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

    /// Run a script, ignoring "relation already exists" style errors.
    /// Used for idempotent `:create` and index-creation system commands.
    fn run_mut_ignore(&self, script: &str, _skip_fragment: &str) -> anyhow::Result<()> {
        match self.run_mut(script, BTreeMap::new()) {
            Ok(_) => Ok(()),
            Err(e) => {
                let msg = e.to_string().to_lowercase();
                // CozoDB may say "already exists" or "conflicts with an existing one"
                // depending on the operation. Both indicate the relation/index is already
                // created — exactly what we want for idempotent initialization.
                if msg.contains("already") || msg.contains("conflict") {
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
}

#[async_trait]
impl StorageBackend for CozoBackend {
    async fn initialize(&self, dim: usize) -> anyhow::Result<()> {
        self.dim.store(dim, Ordering::Relaxed);
        // Create the three base relations — idempotent via error message check.
        self.run_mut_ignore(
            ":create files { file_path: String => language: String, last_modified: Int, last_indexed: Int, chunk_count: Int }",
            "already exists",
        )?;

        // The embedding field uses CozoDB's fixed-dimension vector type <F32; dim>.
        // The relation is created with the specific dim supplied at initialization time.
        let chunks_schema = format!(
            ":create chunks {{ file_path: String, chunk_idx: Int => content: String, normalized: String, chunk_type: String, start_line: Int, end_line: Int, embedding: <F32; {dim}> }}"
        );
        self.run_mut_ignore(&chunks_schema, "already exists")?;

        self.run_mut_ignore(
            ":create code_edges { from_file: String, from_chunk: Int, to_file: String => edge_type: String, created_at: Int }",
            "already exists",
        )?;

        // Create HNSW vector index — idempotent.
        let hnsw = format!(
            "::hnsw create chunks:semantic {{ dim: {dim}, dtype: F32, fields: [embedding], distance: Cosine, m: 50, ef_construction: 20 }}"
        );
        self.run_mut_ignore(&hnsw, "already exists")?;

        // Create FTS index — idempotent.
        self.run_mut_ignore(
            "::fts create chunks:text { extractor: normalized, tokenizer: Simple, filters: [Lowercase, AlphaNumOnly] }",
            "already exists",
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
        for c in chunks {
            let mut p = BTreeMap::new();
            p.insert("fp".into(), Self::dv_str(&c.file_path));
            p.insert("ci".into(), Self::dv_int(c.chunk_idx as i64));
            p.insert("cont".into(), Self::dv_str(&c.content));
            p.insert("norm".into(), Self::dv_str(&c.normalized));
            p.insert("ct".into(), Self::dv_str(&c.chunk_type));
            p.insert("sl".into(), Self::dv_int(c.start_line as i64));
            p.insert("el".into(), Self::dv_int(c.end_line as i64));
            let dim = self.dim.load(Ordering::Relaxed);
            p.insert("emb".into(), Self::embedding_to_dv(&c.embedding, dim));
            self.run_mut(
                "?[file_path, chunk_idx, content, normalized, chunk_type, start_line, end_line, embedding] \
                 <- [[$fp, $ci, $cont, $norm, $ct, $sl, $el, $emb]] \
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
        let now = chrono::Utc::now().timestamp();
        for e in edges {
            let mut p = BTreeMap::new();
            p.insert("ff".into(), Self::dv_str(&e.from_file));
            p.insert("fc".into(), Self::dv_int(e.from_chunk as i64));
            p.insert("tf".into(), Self::dv_str(&e.to_file));
            p.insert("et".into(), Self::dv_str(&e.edge_type));
            p.insert("ca".into(), Self::dv_int(now));
            self.run_mut(
                "?[from_file, from_chunk, to_file, edge_type, created_at] <- [[$ff, $fc, $tf, $et, $ca]] \
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

    async fn hybrid_search(
        &self,
        query_vec: &[f32],
        query_str: &str,
        top_k: usize,
    ) -> anyhow::Result<Vec<SearchResult>> {
        // Guard: if no chunks exist, HNSW search will fail on empty index.
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

        // Build the RRF query.  We use a union of two legs: vector (if embeddings
        // exist) and FTS.  Results that appear in both legs receive a higher score.
        let query_vec_dv: DataValue =
            DataValue::List(query_vec.iter().map(|&f| Self::dv_float(f as f64)).collect());

        let script = if emb_count > 0 {
            format!(
                r#"
vec_scored[fp, ci, score] :=
    ~chunks:semantic{{ fp, ci | query: $qv, k: 50, ef: 50, bind_distance: dist }},
    score = 1.0 / (60.0 + dist * 50.0)
fts_scored[fp, ci, score] :=
    ~chunks:text{{ fp, ci | query: $qs, k: 50, bind_score: bm25 }},
    score = 1.0 / (60.0 + 1.0 / (bm25 + 0.001))
rrf[fp, ci, sum(score)] := vec_scored[fp, ci, score]
rrf[fp, ci, sum(score)] := fts_scored[fp, ci, score]
?[rrf_score, file_path, chunk_idx, content, start_line, end_line, chunk_type] :=
    rrf[fp, ci, rrf_score],
    *chunks[fp, ci, content, _, chunk_type, start_line, end_line, _],
    file_path = fp, chunk_idx = ci
    :order -rrf_score
    :limit {top_k}
"#
            )
        } else {
            // No embeddings yet — fall back to FTS only.
            format!(
                r#"
fts_scored[fp, ci, score] :=
    ~chunks:text{{ fp, ci | query: $qs, k: 50, bind_score: bm25 }},
    score = 1.0 / (60.0 + 1.0 / (bm25 + 0.001))
?[rrf_score, file_path, chunk_idx, content, start_line, end_line, chunk_type] :=
    fts_scored[fp, ci, rrf_score],
    *chunks[fp, ci, content, _, chunk_type, start_line, end_line, _],
    file_path = fp, chunk_idx = ci
    :order -rrf_score
    :limit {top_k}
"#
            )
        };

        let mut p = BTreeMap::new();
        p.insert("qv".into(), query_vec_dv);
        p.insert("qs".into(), Self::dv_str(query_str));

        let rows = match self.run_imm(&script, p) {
            Ok(r) => r,
            Err(e) => {
                // FTS returns no results (not an error in all cozo versions) or
                // the index is actually empty despite the count above — treat as
                // empty rather than propagating a confusing error.
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
                Ok(SearchResult {
                    score: match &r[0] {
                        DataValue::Num(cozo::Num::Float(f)) => *f,
                        DataValue::Num(cozo::Num::Int(i)) => *i as f64,
                        _ => 0.0,
                    },
                    file_path: Self::str_col(&r[1])?,
                    chunk_idx: Self::int_col(&r[2])? as usize,
                    content: Self::str_col(&r[3])?,
                    start_line: Self::int_col(&r[4])? as usize,
                    end_line: Self::int_col(&r[5])? as usize,
                    chunk_type: Self::str_col(&r[6])?,
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
        })
    }
}
