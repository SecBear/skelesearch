/// CompositeBackend — replacement `StorageBackend` implementation.
///
/// Orchestrates three specialized engines:
/// - LanceDB (vector storage + relational tables via Apache Arrow)
/// - Tantivy (BM25 full-text search with a code-aware tokenizer)
/// - petgraph (in-memory import graph for BFS traversal and PageRank)
///
/// # On-disk layout
/// ```text
/// <index_dir>/
///   lance/     — LanceDB dataset directory
///   tantivy/   — Tantivy index directory
/// ```
///
/// # Breaking change
/// `CompositeBackend::open(dir)` expects a **directory** (not a `.db` file).
/// Existing CozoDB `index.db` files are not migrated — callers must re-index.
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use arrow_array::cast::{as_boolean_array, as_primitive_array, as_string_array};
use arrow_array::types::{Float64Type, UInt32Type, UInt8Type};
use arrow_array::{Array, RecordBatch, RecordBatchIterator};
use async_trait::async_trait;
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use petgraph::visit::EdgeRef as PgEdgeRef;
use tokio::sync::RwLock;

use super::graph::ImportGraph;
use super::tantivy_idx::TantivyIndex;
use crate::cochange::CoChangePair;
use crate::schema::{
    classify_symbol_role, CallEdge, ChunkRecord, EdgeRecord, FileRecord, IndexStats, RepoMapData,
    RepoMapFile, RepoMapSymbol, SearchResult, StorageBackend,
};
use crate::sparse::SparseEmbedding;
use crate::symbols::SymbolDef;

// ---------------------------------------------------------------------------
// In-memory sparse inverted index
// ---------------------------------------------------------------------------

struct SparseIndexState {
    /// token_id → [(file_path, chunk_idx, weight)]
    postings: HashMap<u32, Vec<(String, usize, f32)>>,
    loaded: bool,
}

impl SparseIndexState {
    fn new() -> Self {
        Self {
            postings: HashMap::new(),
            loaded: false,
        }
    }
}

// ---------------------------------------------------------------------------
// CompositeBackend struct
// ---------------------------------------------------------------------------

pub struct CompositeBackend {
    #[allow(dead_code)]
    root: PathBuf,
    lance: Arc<lancedb::Connection>,
    tantivy: TantivyIndex,
    edge_graph: Arc<RwLock<ImportGraph>>,
    dim: Arc<AtomicUsize>,
    sparse_idx: Arc<RwLock<SparseIndexState>>,
}

impl CompositeBackend {
    pub async fn open(dir: impl AsRef<Path>) -> anyhow::Result<Self> {
        let root = dir.as_ref().to_path_buf();

        if root.join("index.db").exists() && !root.join("lance").exists() {
            anyhow::bail!(
                "Found legacy CozoDB index at {:?}. \
                 CompositeBackend uses a different on-disk format. \
                 Re-index with `skelesearch index`.",
                root.join("index.db")
            );
        }

        std::fs::create_dir_all(root.join("lance"))?;
        let lance = Arc::new(
            lancedb::connect(root.join("lance").to_string_lossy().as_ref())
                .execute()
                .await?,
        );
        let tantivy = TantivyIndex::open_or_create(&root.join("tantivy"))?;

        Ok(Self {
            root,
            lance,
            tantivy,
            edge_graph: Arc::new(RwLock::new(ImportGraph::new())),
            dim: Arc::new(AtomicUsize::new(0)),
            sparse_idx: Arc::new(RwLock::new(SparseIndexState::new())),
        })
    }

    fn dim(&self) -> usize {
        self.dim.load(Ordering::Relaxed)
    }

    async fn reload_edge_graph(&self) -> anyhow::Result<()> {
        let tbl = match self.lance.open_table("code_edges").execute().await {
            Ok(t) => t,
            Err(_) => return Ok(()),
        };
        let batches: Vec<RecordBatch> = tbl.query().execute().await?.try_collect().await?;
        let mut graph = self.edge_graph.write().await;
        *graph = ImportGraph::new();
        for batch in &batches {
            let from_col = as_string_array(batch.column_by_name("from_file").unwrap());
            let to_col = as_string_array(batch.column_by_name("to_file").unwrap());
            let type_col = as_string_array(batch.column_by_name("edge_type").unwrap());
            for i in 0..from_col.len() {
                graph.add_edge(from_col.value(i), to_col.value(i), type_col.value(i));
            }
        }
        Ok(())
    }

    fn tantivy_search(
        &self,
        query_str: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<(String, usize, f64)>> {
        use tantivy::collector::TopDocs;
        use tantivy::query::QueryParser;
        use tantivy::schema::Value as TantivyValue;

        let searcher = self.tantivy.reader.searcher();
        let qp = QueryParser::for_index(
            &self.tantivy.index,
            vec![self.tantivy.f_normalized, self.tantivy.f_description],
        );
        let query = match qp.parse_query(query_str) {
            Ok(q) => q,
            Err(_) => qp.parse_query(&format!("\"{}\"", query_str.replace('"', "")))?,
        };
        let top_docs = searcher.search(&query, &TopDocs::with_limit(limit))?;
        let mut results = Vec::new();
        for (score, addr) in top_docs {
            let doc: tantivy::TantivyDocument = searcher.doc(addr)?;
            let fp = doc
                .get_first(self.tantivy.f_file_path)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let ci = doc
                .get_first(self.tantivy.f_chunk_idx)
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize;
            results.push((fp, ci, score as f64));
        }
        Ok(results)
    }

    async fn lance_vector_search(
        &self,
        query_vec: &[f32],
        limit: usize,
    ) -> anyhow::Result<Vec<(String, usize, f64)>> {
        let tbl = self.lance.open_table("chunks").execute().await?;
        let batches: Vec<RecordBatch> = tbl
            .query()
            .limit(limit)
            .nearest_to(query_vec)?
            .column("embedding")
            .execute()
            .await?
            .try_collect()
            .await?;
        arrow_batches_to_fp_ci_dist(&batches)
    }

    async fn materialize_results(
        &self,
        scored: &[(String, usize, f64, &'static str)],
    ) -> anyhow::Result<Vec<SearchResult>> {
        if scored.is_empty() {
            return Ok(vec![]);
        }
        let conditions: Vec<String> = scored
            .iter()
            .map(|(fp, ci, _, _)| format!("(file_path = '{}' AND chunk_idx = {ci})", esc(fp)))
            .collect();
        let tbl = self.lance.open_table("chunks").execute().await?;
        let batches: Vec<RecordBatch> = tbl
            .query()
            .only_if(conditions.join(" OR "))
            .execute()
            .await?
            .try_collect()
            .await?;
        let chunks = arrow_batches_to_chunk_records(&batches)?;
        let by_key: HashMap<(String, usize), ChunkRecord> = chunks
            .into_iter()
            .map(|c| ((c.file_path.clone(), c.chunk_idx), c))
            .collect();

        Ok(scored
            .iter()
            .filter_map(|(fp, ci, score, why)| {
                by_key.get(&(fp.clone(), *ci)).map(|c| SearchResult {
                    file_path: c.file_path.clone(),
                    chunk_idx: c.chunk_idx,
                    content: c.content.clone(),
                    start_line: c.start_line,
                    end_line: c.end_line,
                    chunk_type: c.chunk_type.clone(),
                    score: *score,
                    match_quality: String::new(),
                    why: why.to_string(),
                    materialization_tier: c.materialization_tier,
                })
            })
            .collect())
    }
}

// ---------------------------------------------------------------------------
// Upsert helper: wraps a single RecordBatch in the form lancedb expects
// ---------------------------------------------------------------------------

fn as_reader(batch: RecordBatch) -> Box<dyn arrow_array::RecordBatchReader + Send> {
    let schema = batch.schema();
    Box::new(RecordBatchIterator::new(vec![Ok(batch)], schema))
}

// ---------------------------------------------------------------------------
// StorageBackend impl
// ---------------------------------------------------------------------------

#[async_trait]
impl StorageBackend for CompositeBackend {
    async fn initialize(&self, dim: usize) -> anyhow::Result<()> {
        use super::schemas::*;
        use lancedb::index::{scalar::BTreeIndexBuilder, Index};
        use lancedb::Error as LanceError;

        self.dim.store(dim, Ordering::Relaxed);

        let fixed: &[(&str, arrow_schema::Schema)] = &[
            ("files", files_schema()),
            ("code_edges", code_edges_schema()),
            ("call_edges", call_edges_schema()),
            ("symbols", symbols_schema()),
            ("cochange_edges", cochange_edges_schema()),
            ("sparse_index", sparse_index_schema()),
            ("pagerank_scores", pagerank_scores_schema()),
            ("symbol_roles", symbol_roles_schema()),
        ];
        for (name, schema) in fixed {
            match self
                .lance
                .create_empty_table(*name, Arc::new(schema.clone()))
                .execute()
                .await
            {
                Ok(_) | Err(LanceError::TableAlreadyExists { .. }) => {}
                Err(e) => return Err(e.into()),
            }
        }
        // Handle dimension change: if the chunks table exists with a different
        // embedding dimension, drop and recreate it (forces re-index).
        let existing_dim = if let Ok(tbl) = self.lance.open_table("chunks").execute().await {
            // Probe the schema to read the embedding column size.
            tbl.schema().await.ok().and_then(|s| {
                s.field_with_name("embedding").ok().and_then(|f| {
                    if let arrow_schema::DataType::FixedSizeList(_, d) = f.data_type() {
                        Some(*d as usize)
                    } else {
                        None
                    }
                })
            })
        } else {
            None
        };
        if existing_dim.map_or(false, |d| d != dim) {
            // Dimension changed — drop the chunks table so it's recreated below.
            let _ = self.lance.drop_table("chunks", &[]).await;
        }
        match self
            .lance
            .create_empty_table("chunks", Arc::new(chunks_schema(dim)))
            .execute()
            .await
        {
            Ok(_) | Err(LanceError::TableAlreadyExists { .. }) => {}
            Err(e) => return Err(e.into()),
        }

        let idx_targets = [
            ("files", "file_path"),
            ("chunks", "file_path"),
            ("code_edges", "from_file"),
            ("code_edges", "to_file"),
            ("call_edges", "caller_file"),
            ("symbols", "file_path"),
            ("cochange_edges", "file_a"),
            ("cochange_edges", "file_b"),
            ("sparse_index", "file_path"),
            ("pagerank_scores", "file_path"),
            ("symbol_roles", "file_path"),
        ];
        for (table, col) in &idx_targets {
            if let Ok(tbl) = self.lance.open_table(*table).execute().await {
                let _ = tbl
                    .create_index(&[*col], Index::BTree(BTreeIndexBuilder::default()))
                    .execute()
                    .await;
            }
        }
        self.reload_edge_graph().await
    }

    // --- File CRUD -----------------------------------------------------------

    async fn upsert_file(&self, record: &FileRecord) -> anyhow::Result<()> {
        use arrow_array::{Int64Array, StringArray, UInt64Array};
        let schema = Arc::new(super::schemas::files_schema());
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec![record.file_path.as_str()])),
                Arc::new(StringArray::from(vec![record.language.as_str()])),
                Arc::new(Int64Array::from(vec![record.last_modified])),
                Arc::new(Int64Array::from(vec![record.last_indexed])),
                Arc::new(UInt64Array::from(vec![record.chunk_count as u64])),
            ],
        )?;
        let tbl = self.lance.open_table("files").execute().await?;
        {
            let mut mi = tbl.merge_insert(&["file_path"]);
            mi.when_matched_update_all(None)
                .when_not_matched_insert_all();
            mi.execute(as_reader(batch)).await?;
        }
        Ok(())
    }

    async fn delete_file(&self, file_path: &str) -> anyhow::Result<()> {
        let tbl = self.lance.open_table("files").execute().await?;
        tbl.delete(&format!("file_path = '{}'", esc(file_path)))
            .await?;
        Ok(())
    }

    async fn list_indexed_paths(&self) -> anyhow::Result<Vec<String>> {
        let tbl = self.lance.open_table("files").execute().await?;
        let batches: Vec<RecordBatch> = tbl
            .query()
            .select(Select::columns(&["file_path"]))
            .execute()
            .await?
            .try_collect()
            .await?;
        let mut paths = Vec::new();
        for batch in &batches {
            let col = as_string_array(batch.column_by_name("file_path").unwrap());
            for i in 0..col.len() {
                paths.push(col.value(i).to_string());
            }
        }
        Ok(paths)
    }

    // --- Chunk CRUD ----------------------------------------------------------

    async fn upsert_chunks(&self, chunks: &[ChunkRecord]) -> anyhow::Result<()> {
        if chunks.is_empty() {
            return Ok(());
        }
        let dim = self.dim();
        let batch = chunks_to_arrow(chunks, dim)?;
        let tbl = self.lance.open_table("chunks").execute().await?;
        {
            let mut mi = tbl.merge_insert(&["file_path", "chunk_idx"]);
            mi.when_matched_update_all(None)
                .when_not_matched_insert_all();
            mi.execute(as_reader(batch)).await?;
        }

        {
            let mut writer = self.tantivy.writer.lock().unwrap();
            let touched_files: std::collections::HashSet<&str> =
                chunks.iter().map(|c| c.file_path.as_str()).collect();
            for fp in &touched_files {
                writer.delete_term(tantivy::Term::from_field_text(self.tantivy.f_file_path, fp));
            }
            for chunk in chunks {
                if chunk.normalized.is_empty() {
                    continue;
                }
                let mut doc = tantivy::TantivyDocument::default();
                doc.add_text(self.tantivy.f_file_path, &chunk.file_path);
                doc.add_u64(self.tantivy.f_chunk_idx, chunk.chunk_idx as u64);
                doc.add_text(self.tantivy.f_normalized, &chunk.normalized);
                doc.add_text(self.tantivy.f_description, &chunk.description);
                doc.add_text(self.tantivy.f_chunk_type, &chunk.chunk_type);
                doc.add_u64(self.tantivy.f_tier, chunk.materialization_tier as u64);
                writer.add_document(doc)?;
            }
            writer.commit()?;
        }
        Ok(())
    }

    async fn delete_chunks_for_file(&self, file_path: &str) -> anyhow::Result<()> {
        let tbl = self.lance.open_table("chunks").execute().await?;
        tbl.delete(&format!("file_path = '{}'", esc(file_path)))
            .await?;
        let mut writer = self.tantivy.writer.lock().unwrap();
        writer.delete_term(tantivy::Term::from_field_text(
            self.tantivy.f_file_path,
            file_path,
        ));
        writer.commit()?;
        Ok(())
    }

    async fn delete_tier1_chunks_for_file(&self, file_path: &str) -> anyhow::Result<()> {
        let tbl = self.lance.open_table("chunks").execute().await?;
        tbl.delete(&format!(
            "file_path = '{}' AND materialization_tier = 1",
            esc(file_path)
        ))
        .await?;
        let mut writer = self.tantivy.writer.lock().unwrap();
        writer.delete_term(tantivy::Term::from_field_text(
            self.tantivy.f_file_path,
            file_path,
        ));
        writer.commit()?;
        Ok(())
    }

    async fn get_chunks_for_file(&self, file_path: &str) -> anyhow::Result<Vec<ChunkRecord>> {
        let tbl = self.lance.open_table("chunks").execute().await?;
        let batches: Vec<RecordBatch> = tbl
            .query()
            .only_if(format!("file_path = '{}'", esc(file_path)))
            .execute()
            .await?
            .try_collect()
            .await?;
        arrow_batches_to_chunk_records(&batches)
    }

    async fn get_chunks_for_files(&self, file_paths: &[&str]) -> anyhow::Result<Vec<ChunkRecord>> {
        if file_paths.is_empty() {
            return Ok(vec![]);
        }
        let in_list = file_paths
            .iter()
            .map(|p| format!("'{}'", esc(p)))
            .collect::<Vec<_>>()
            .join(", ");
        let tbl = self.lance.open_table("chunks").execute().await?;
        let batches: Vec<RecordBatch> = tbl
            .query()
            .only_if(format!("file_path IN ({in_list})"))
            .execute()
            .await?
            .try_collect()
            .await?;
        arrow_batches_to_chunk_records(&batches)
    }

    // --- Edge CRUD -----------------------------------------------------------

    async fn upsert_edges(&self, edges: &[EdgeRecord]) -> anyhow::Result<()> {
        if edges.is_empty() {
            return Ok(());
        }
        use arrow_array::{StringArray, UInt32Array};
        let schema = Arc::new(super::schemas::code_edges_schema());
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from_iter_values(
                    edges.iter().map(|e| e.from_file.as_str()),
                )),
                Arc::new(UInt32Array::from_iter_values(
                    edges.iter().map(|e| e.from_chunk as u32),
                )),
                Arc::new(StringArray::from_iter_values(
                    edges.iter().map(|e| e.to_file.as_str()),
                )),
                Arc::new(StringArray::from_iter_values(
                    edges.iter().map(|e| e.edge_type.as_str()),
                )),
            ],
        )?;
        let tbl = self.lance.open_table("code_edges").execute().await?;
        {
            let mut mi = tbl.merge_insert(&["from_file", "from_chunk", "to_file"]);
            mi.when_matched_update_all(None)
                .when_not_matched_insert_all();
            mi.execute(as_reader(batch)).await?;
        }
        let mut graph = self.edge_graph.write().await;
        for edge in edges {
            graph.add_edge(&edge.from_file, &edge.to_file, &edge.edge_type);
        }
        Ok(())
    }

    async fn delete_edges_for_file(&self, file_path: &str) -> anyhow::Result<()> {
        let escaped = esc(file_path);
        let tbl = self.lance.open_table("code_edges").execute().await?;
        tbl.delete(&format!("from_file = '{escaped}' OR to_file = '{escaped}'"))
            .await?;
        self.edge_graph
            .write()
            .await
            .remove_edges_for_file(file_path);
        Ok(())
    }

    async fn get_imports(&self, file_path: &str) -> anyhow::Result<Vec<String>> {
        let graph = self.edge_graph.read().await;
        Ok(if let Some(&node) = graph.node_index.get(file_path) {
            graph
                .graph
                .edges(node)
                .map(|e| graph.graph[e.target()].clone())
                .collect()
        } else {
            vec![]
        })
    }

    async fn get_importers(&self, file_path: &str) -> anyhow::Result<Vec<String>> {
        let graph = self.edge_graph.read().await;
        Ok(if let Some(&node) = graph.node_index.get(file_path) {
            graph
                .graph
                .edges_directed(node, petgraph::Direction::Incoming)
                .map(|e| graph.graph[e.source()].clone())
                .collect()
        } else {
            vec![]
        })
    }

    async fn traverse_imports(
        &self,
        file_path: &str,
        max_depth: usize,
        edge_types: Option<&[&str]>,
    ) -> anyhow::Result<Vec<(String, usize)>> {
        Ok(self
            .edge_graph
            .read()
            .await
            .bfs_forward(file_path, max_depth, edge_types))
    }

    async fn traverse_importers(
        &self,
        file_path: &str,
        max_depth: usize,
        edge_types: Option<&[&str]>,
    ) -> anyhow::Result<Vec<(String, usize)>> {
        Ok(self
            .edge_graph
            .read()
            .await
            .bfs_reverse(file_path, max_depth, edge_types))
    }

    // --- Symbol CRUD ---------------------------------------------------------

    async fn upsert_symbols(&self, symbols: &[SymbolDef]) -> anyhow::Result<()> {
        if symbols.is_empty() {
            return Ok(());
        }
        use arrow_array::{StringArray, UInt32Array};
        let schema = Arc::new(super::schemas::symbols_schema());
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from_iter_values(
                    symbols.iter().map(|s| s.file_path.as_str()),
                )),
                Arc::new(StringArray::from_iter_values(
                    symbols.iter().map(|s| s.name.as_str()),
                )),
                Arc::new(StringArray::from_iter_values(
                    symbols.iter().map(|s| s.kind.as_str()),
                )),
                Arc::new(UInt32Array::from_iter_values(
                    symbols.iter().map(|s| s.start_line as u32),
                )),
                Arc::new(UInt32Array::from_iter_values(
                    symbols.iter().map(|s| s.end_line as u32),
                )),
            ],
        )?;
        let tbl = self.lance.open_table("symbols").execute().await?;
        {
            let mut mi = tbl.merge_insert(&["file_path", "name", "kind"]);
            mi.when_matched_update_all(None)
                .when_not_matched_insert_all();
            mi.execute(as_reader(batch)).await?;
        }
        Ok(())
    }

    async fn delete_symbols_for_file(&self, file_path: &str) -> anyhow::Result<()> {
        let tbl = self.lance.open_table("symbols").execute().await?;
        tbl.delete(&format!("file_path = '{}'", esc(file_path)))
            .await?;
        Ok(())
    }

    async fn find_symbols(&self, name: &str, kind: Option<&str>) -> anyhow::Result<Vec<SymbolDef>> {
        let filter = if let Some(k) = kind {
            format!("name = '{}' AND kind = '{}'", esc(name), esc(k))
        } else {
            format!("name = '{}'", esc(name))
        };
        let tbl = self.lance.open_table("symbols").execute().await?;
        let batches: Vec<RecordBatch> = tbl
            .query()
            .only_if(filter)
            .execute()
            .await?
            .try_collect()
            .await?;
        arrow_batches_to_symbols(&batches)
    }

    // --- Search --------------------------------------------------------------

    async fn hybrid_search(
        &self,
        query_vec: &[f32],
        query_str: &str,
        top_k: usize,
    ) -> anyhow::Result<Vec<SearchResult>> {
        let overfetch = top_k * 3;
        let fts = self.tantivy_search(query_str, overfetch)?;
        let vec_hits = self.lance_vector_search(query_vec, overfetch).await?;
        let fused = rrf_fuse(fts, vec_hits, top_k);
        self.materialize_results(&fused).await
    }

    async fn unified_search(
        &self,
        query_vec: &[f32],
        query_str: &str,
        top_k: usize,
        graph_depth: usize,
        _fts_weight: f64,
        graph_score_factor: f64,
        _graph_min_score: f64,
        pagerank_factor: f64,
    ) -> anyhow::Result<Vec<SearchResult>> {
        let mut results = self.hybrid_search(query_vec, query_str, top_k).await?;
        // unified_search contract: all non-graph results use "hybrid" provenance,
        // regardless of whether they were in FTS-only, vector-only, or both legs.
        for r in &mut results {
            if r.why != "graph" {
                r.why = "hybrid".to_string();
            }
        }

        if graph_depth > 0 && !results.is_empty() {
            let seed_files: Vec<&str> = results.iter().map(|r| r.file_path.as_str()).collect();
            let importer_files = {
                let graph = self.edge_graph.read().await;
                let mut importers = Vec::new();
                for &seed in &seed_files {
                    if let Some(&node) = graph.node_index.get(seed) {
                        for e in graph
                            .graph
                            .edges_directed(node, petgraph::Direction::Incoming)
                        {
                            let src = graph.graph[e.source()].clone();
                            if !seed_files.contains(&src.as_str()) {
                                importers.push(src);
                            }
                        }
                    }
                }
                importers
            };

            if !importer_files.is_empty() {
                let refs: Vec<&str> = importer_files.iter().map(|s| s.as_str()).collect();
                let graph_chunks = self.get_chunks_for_files(&refs).await?;
                let base_score = results.first().map(|r| r.score).unwrap_or(0.0);
                let seen: std::collections::HashSet<_> = results
                    .iter()
                    .map(|r| (r.file_path.clone(), r.chunk_idx))
                    .collect();
                for chunk in graph_chunks {
                    if !seen.contains(&(chunk.file_path.clone(), chunk.chunk_idx)) {
                        results.push(SearchResult {
                            file_path: chunk.file_path,
                            chunk_idx: chunk.chunk_idx,
                            content: chunk.content,
                            start_line: chunk.start_line,
                            end_line: chunk.end_line,
                            chunk_type: chunk.chunk_type,
                            score: base_score * graph_score_factor,
                            match_quality: String::new(),
                            why: "graph".to_string(),
                            materialization_tier: chunk.materialization_tier,
                        });
                    }
                }
            }
        }

        if pagerank_factor > 0.0 {
            let all_paths: Vec<&str> = results.iter().map(|r| r.file_path.as_str()).collect();
            let ranks = self.get_file_ranks(&all_paths).await?;
            for r in &mut results {
                if let Some(&rank) = ranks.get(&r.file_path) {
                    r.score *= 1.0 + pagerank_factor * rank;
                }
            }
            results.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        Ok(results.into_iter().take(top_k).collect())
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
        let embeddings = self.get_chunk_embeddings(seeds).await?;
        let tbl = self.lance.open_table("chunks").execute().await?;
        let mut results: Vec<(String, usize, f64)> = Vec::new();
        let mut seen = std::collections::HashSet::new();

        for ((seed_file, seed_idx), emb) in seeds.iter().zip(embeddings.iter()) {
            if emb.is_empty() || emb.iter().all(|&v| v == 0.0) {
                continue;
            }
            let batches: Vec<RecordBatch> = tbl
                .query()
                .limit(limit * 2)
                .nearest_to(emb.as_slice())?
                .column("embedding")
                .only_if(format!(
                    "NOT (file_path = '{}' AND chunk_idx = {seed_idx})",
                    esc(seed_file)
                ))
                .execute()
                .await?
                .try_collect()
                .await?;
            for (fp, ci, dist) in arrow_batches_to_fp_ci_dist(&batches)? {
                if dist <= max_dist && seen.insert((fp.clone(), ci)) {
                    results.push((fp, ci, dist));
                }
            }
        }
        results.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        Ok(results)
    }

    async fn doc_vector_search(
        &self,
        query_vec: &[f32],
        limit: usize,
    ) -> anyhow::Result<Vec<(String, usize, f64)>> {
        let tbl = match self.lance.open_table("doc_chunks").execute().await {
            Err(_) => return Ok(vec![]),
            Ok(t) => t,
        };
        let batches: Vec<RecordBatch> = tbl
            .query()
            .limit(limit)
            .nearest_to(query_vec)?
            .column("embedding")
            .execute()
            .await?
            .try_collect()
            .await?;
        arrow_batches_to_fp_ci_dist(&batches)
    }

    // --- Embeddings ----------------------------------------------------------

    async fn get_chunk_embeddings(
        &self,
        keys: &[(String, usize)],
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        if keys.is_empty() {
            return Ok(vec![]);
        }
        let conditions: Vec<String> = keys
            .iter()
            .map(|(fp, ci)| format!("(file_path = '{}' AND chunk_idx = {ci})", esc(fp)))
            .collect();
        let tbl = self.lance.open_table("chunks").execute().await?;
        let batches: Vec<RecordBatch> = tbl
            .query()
            .only_if(conditions.join(" OR "))
            .execute()
            .await?
            .try_collect()
            .await?;

        let dim = self.dim();
        let mut lookup: HashMap<(String, usize), Vec<f32>> = HashMap::new();
        for batch in &batches {
            let fp_col = as_string_array(batch.column_by_name("file_path").unwrap());
            let ci_col =
                as_primitive_array::<UInt32Type>(batch.column_by_name("chunk_idx").unwrap());
            let emb_col = batch
                .column_by_name("embedding")
                .and_then(|c| c.as_any().downcast_ref::<arrow_array::FixedSizeListArray>());
            for i in 0..fp_col.len() {
                let emb = emb_col
                    .and_then(|ec| {
                        if ec.is_null(i) {
                            None
                        } else {
                            ec.value(i)
                                .as_any()
                                .downcast_ref::<arrow_array::Float32Array>()
                                .map(|a| (0..a.len()).map(|j| a.value(j)).collect())
                        }
                    })
                    .unwrap_or_else(|| vec![0f32; dim]);
                lookup.insert((fp_col.value(i).to_string(), ci_col.value(i) as usize), emb);
            }
        }
        Ok(keys
            .iter()
            .map(|k| lookup.get(k).cloned().unwrap_or_else(|| vec![0f32; dim]))
            .collect())
    }

    // --- Stats ---------------------------------------------------------------

    async fn stats(&self) -> anyhow::Result<IndexStats> {
        use arrow_array::types::Int64Type;
        use chrono::{TimeZone, Utc};
        let lance = self.lance.clone();

        // Run file count, chunk count, and max(last_indexed) in parallel.
        let (file_res, chunk_res, ts_res) = tokio::join!(
            async {
                lance
                    .open_table("files")
                    .execute()
                    .await?
                    .count_rows(None)
                    .await
            },
            async {
                lance
                    .open_table("chunks")
                    .execute()
                    .await?
                    .count_rows(None)
                    .await
            },
            async {
                // Fetch last_indexed column and compute MAX in Rust.
                // (DataFusion SQL aggregates are not exposed directly in lancedb 0.27.)
                let tbl = lance.open_table("files").execute().await?;
                let batches: Vec<RecordBatch> = tbl
                    .query()
                    .select(Select::columns(&["last_indexed"]))
                    .execute()
                    .await?
                    .try_collect()
                    .await?;
                let mut max_ts: Option<i64> = None;
                for batch in &batches {
                    let col = as_primitive_array::<Int64Type>(
                        batch.column_by_name("last_indexed").unwrap(),
                    );
                    for i in 0..col.len() {
                        let v = col.value(i);
                        max_ts = Some(max_ts.map_or(v, |m| m.max(v)));
                    }
                }
                anyhow::Ok(max_ts)
            }
        );

        let last_indexed = ts_res?.and_then(|ts| Utc.timestamp_opt(ts, 0).single());

        Ok(IndexStats {
            indexed_files: file_res.unwrap_or(0),
            total_chunks: chunk_res.unwrap_or(0),
            last_indexed,
            watching: false,
            estimated_stale: 0,
        })
    }

    // --- PageRank + Symbol Roles ---------------------------------------------

    async fn compute_pagerank(&self, edge_types: Option<&[&str]>) -> anyhow::Result<()> {
        use arrow_array::{Float64Array, StringArray};

        let (filtered, file_paths) = {
            let graph = self.edge_graph.read().await;
            if graph.graph.node_count() == 0 {
                return Ok(());
            }

            let mut g = petgraph::Graph::<String, (), petgraph::Directed>::new();
            let mut idx_map: HashMap<petgraph::graph::NodeIndex, petgraph::graph::NodeIndex> =
                HashMap::new();
            for n in graph.graph.node_indices() {
                let new_n = g.add_node(graph.graph[n].clone());
                idx_map.insert(n, new_n);
            }
            for e in graph.graph.edge_indices() {
                let (s, t) = graph.graph.edge_endpoints(e).unwrap();
                let weight = &graph.graph[e];
                if edge_types.map_or(true, |types| types.contains(&weight.as_str())) {
                    g.add_edge(idx_map[&s], idx_map[&t], ());
                }
            }
            let paths: Vec<String> = g.node_indices().map(|n| g[n].clone()).collect();
            (g, paths)
        };

        let scores = petgraph::algo::page_rank(&filtered, 0.85, 100);
        let total: f64 = scores.iter().map(|&s| s as f64).sum();
        let normalized: Vec<f64> = scores
            .iter()
            .map(|&s| if total > 0.0 { s as f64 / total } else { 0.0 })
            .collect();

        let schema = Arc::new(super::schemas::pagerank_scores_schema());
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(file_paths)),
                Arc::new(Float64Array::from(normalized)),
            ],
        )?;

        let tbl = self.lance.open_table("pagerank_scores").execute().await?;
        tbl.delete("true").await?;
        tbl.add(as_reader(batch)).execute().await?;
        Ok(())
    }

    async fn get_file_ranks(&self, file_paths: &[&str]) -> anyhow::Result<HashMap<String, f64>> {
        if file_paths.is_empty() {
            return Ok(HashMap::new());
        }
        let in_list = file_paths
            .iter()
            .map(|p| format!("'{}'", esc(p)))
            .collect::<Vec<_>>()
            .join(", ");
        let tbl = match self.lance.open_table("pagerank_scores").execute().await {
            Ok(t) => t,
            Err(_) => return Ok(HashMap::new()),
        };
        let batches: Vec<RecordBatch> = tbl
            .query()
            .only_if(format!("file_path IN ({in_list})"))
            .execute()
            .await?
            .try_collect()
            .await?;
        let mut result = HashMap::new();
        for batch in &batches {
            let fp_col = as_string_array(batch.column_by_name("file_path").unwrap());
            let sc_col = batch
                .column_by_name("score")
                .and_then(|c| c.as_any().downcast_ref::<arrow_array::Float64Array>());
            for i in 0..fp_col.len() {
                result.insert(
                    fp_col.value(i).to_string(),
                    sc_col.map(|s| s.value(i)).unwrap_or(0.0),
                );
            }
        }
        Ok(result)
    }

    async fn compute_symbol_roles(&self) -> anyhow::Result<()> {
        use arrow_array::StringArray;
        let (file_paths, roles): (Vec<String>, Vec<String>) = {
            let graph = self.edge_graph.read().await;
            if graph.graph.node_count() == 0 {
                return Ok(());
            }
            graph
                .graph
                .node_indices()
                .map(|n| {
                    let fp = graph.graph[n].clone();
                    let out = graph
                        .graph
                        .edges_directed(n, petgraph::Direction::Outgoing)
                        .count();
                    let inn = graph
                        .graph
                        .edges_directed(n, petgraph::Direction::Incoming)
                        .count();
                    (fp, classify_symbol_role(inn, out).to_string())
                })
                .unzip()
        };
        let schema = Arc::new(super::schemas::symbol_roles_schema());
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(file_paths)),
                Arc::new(StringArray::from(roles)),
            ],
        )?;
        let tbl = self.lance.open_table("symbol_roles").execute().await?;
        tbl.delete("true").await?;
        tbl.add(as_reader(batch)).execute().await?;
        Ok(())
    }

    async fn get_symbol_roles(
        &self,
        file_paths: &[&str],
    ) -> anyhow::Result<HashMap<String, String>> {
        if file_paths.is_empty() {
            return Ok(HashMap::new());
        }
        let in_list = file_paths
            .iter()
            .map(|p| format!("'{}'", esc(p)))
            .collect::<Vec<_>>()
            .join(", ");
        let tbl = match self.lance.open_table("symbol_roles").execute().await {
            Ok(t) => t,
            Err(_) => return Ok(HashMap::new()),
        };
        let batches: Vec<RecordBatch> = tbl
            .query()
            .only_if(format!("file_path IN ({in_list})"))
            .execute()
            .await?
            .try_collect()
            .await?;
        let mut result = HashMap::new();
        for batch in &batches {
            let fp_col = as_string_array(batch.column_by_name("file_path").unwrap());
            let role_col = batch
                .column_by_name("role")
                .map(|c| as_string_array(c.as_ref()));
            for i in 0..fp_col.len() {
                result.insert(
                    fp_col.value(i).to_string(),
                    role_col.map(|c| c.value(i).to_string()).unwrap_or_default(),
                );
            }
        }
        Ok(result)
    }

    // --- Co-change edges -----------------------------------------------------

    async fn upsert_cochange_edges(&self, pairs: &[CoChangePair]) -> anyhow::Result<()> {
        if pairs.is_empty() {
            return Ok(());
        }
        use arrow_array::{Float64Array, StringArray, UInt64Array};
        let schema = Arc::new(super::schemas::cochange_edges_schema());
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from_iter_values(
                    pairs.iter().map(|p| p.file_a.as_str()),
                )),
                Arc::new(StringArray::from_iter_values(
                    pairs.iter().map(|p| p.file_b.as_str()),
                )),
                Arc::new(UInt64Array::from_iter_values(
                    pairs.iter().map(|p| p.cochange_count as u64),
                )),
                Arc::new(Float64Array::from_iter_values(
                    pairs.iter().map(|p| p.jaccard),
                )),
            ],
        )?;
        let tbl = self.lance.open_table("cochange_edges").execute().await?;
        {
            let mut mi = tbl.merge_insert(&["file_a", "file_b"]);
            mi.when_matched_update_all(None)
                .when_not_matched_insert_all();
            mi.execute(as_reader(batch)).await?;
        }
        Ok(())
    }

    async fn get_cochange_neighbors(
        &self,
        file_path: &str,
        min_score: f64,
    ) -> anyhow::Result<Vec<(String, f64)>> {
        let escaped = esc(file_path);
        let filter =
            format!("(file_a = '{escaped}' OR file_b = '{escaped}') AND jaccard >= {min_score}");
        let tbl = match self.lance.open_table("cochange_edges").execute().await {
            Ok(t) => t,
            Err(_) => return Ok(vec![]),
        };
        let batches: Vec<RecordBatch> = tbl
            .query()
            .only_if(filter)
            .execute()
            .await?
            .try_collect()
            .await?;
        let mut results = Vec::new();
        for batch in &batches {
            let fa_col = as_string_array(batch.column_by_name("file_a").unwrap());
            let fb_col = as_string_array(batch.column_by_name("file_b").unwrap());
            let jac_col = batch
                .column_by_name("jaccard")
                .and_then(|c| c.as_any().downcast_ref::<arrow_array::Float64Array>());
            for i in 0..fa_col.len() {
                let fa = fa_col.value(i);
                let fb = fb_col.value(i);
                let neighbor = if fa == file_path { fb } else { fa };
                results.push((
                    neighbor.to_string(),
                    jac_col.map(|j| j.value(i)).unwrap_or(0.0),
                ));
            }
        }
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(results)
    }

    // --- Sparse index --------------------------------------------------------

    async fn store_sparse_vectors(
        &self,
        file_path: &str,
        chunk_idx: usize,
        sparse: &SparseEmbedding,
    ) -> anyhow::Result<()> {
        use arrow_array::{Float32Array, StringArray, UInt32Array};
        let n = sparse.indices.len();
        if n == 0 {
            return Ok(());
        }
        let schema = Arc::new(super::schemas::sparse_index_schema());
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec![file_path; n])),
                Arc::new(UInt32Array::from(vec![chunk_idx as u32; n])),
                Arc::new(UInt32Array::from(sparse.indices.clone())),
                Arc::new(Float32Array::from(sparse.values.clone())),
            ],
        )?;
        let tbl = self.lance.open_table("sparse_index").execute().await?;
        {
            let mut mi = tbl.merge_insert(&["file_path", "chunk_idx", "token_id"]);
            mi.when_matched_update_all(None)
                .when_not_matched_insert_all();
            mi.execute(as_reader(batch)).await?;
        }
        let mut idx = self.sparse_idx.write().await;
        for (token_id, weight) in sparse.indices.iter().zip(sparse.values.iter()) {
            idx.postings.entry(*token_id).or_default().push((
                file_path.to_string(),
                chunk_idx,
                *weight,
            ));
        }
        Ok(())
    }

    async fn delete_sparse_for_file(&self, file_path: &str) -> anyhow::Result<()> {
        if let Ok(tbl) = self.lance.open_table("sparse_index").execute().await {
            tbl.delete(&format!("file_path = '{}'", esc(file_path)))
                .await?;
        }
        let mut idx = self.sparse_idx.write().await;
        for postings in idx.postings.values_mut() {
            postings.retain(|(fp, _, _)| fp != file_path);
        }
        Ok(())
    }

    async fn sparse_search(
        &self,
        query_sparse: &SparseEmbedding,
        top_k: usize,
    ) -> anyhow::Result<Vec<(String, usize, f64)>> {
        let mut idx = self.sparse_idx.write().await;
        if !idx.loaded {
            if let Ok(tbl) = self.lance.open_table("sparse_index").execute().await {
                let batches: Vec<RecordBatch> = tbl.query().execute().await?.try_collect().await?;
                for batch in &batches {
                    use arrow_array::types::Float32Type;
                    let fp_col = as_string_array(batch.column_by_name("file_path").unwrap());
                    let ci_col = as_primitive_array::<UInt32Type>(
                        batch.column_by_name("chunk_idx").unwrap(),
                    );
                    let tid_col =
                        as_primitive_array::<UInt32Type>(batch.column_by_name("token_id").unwrap());
                    let w_col =
                        as_primitive_array::<Float32Type>(batch.column_by_name("weight").unwrap());
                    for i in 0..fp_col.len() {
                        idx.postings.entry(tid_col.value(i)).or_default().push((
                            fp_col.value(i).to_string(),
                            ci_col.value(i) as usize,
                            w_col.value(i),
                        ));
                    }
                }
            }
            idx.loaded = true;
        }

        let mut scores: HashMap<(String, usize), f64> = HashMap::new();
        for (token_id, q_weight) in query_sparse.indices.iter().zip(query_sparse.values.iter()) {
            if let Some(postings) = idx.postings.get(token_id) {
                for (fp, ci, weight) in postings {
                    *scores.entry((fp.clone(), *ci)).or_insert(0.0) +=
                        (*q_weight as f64) * (*weight as f64);
                }
            }
        }
        let mut results: Vec<(String, usize, f64)> = scores
            .into_iter()
            .map(|((fp, ci), s)| (fp, ci, s))
            .collect();
        results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        Ok(results)
    }

    // --- Complex aggregations ------------------------------------------------

    async fn deduplicate_chunks(&self) -> anyhow::Result<usize> {
        use twox_hash::XxHash64;
        let tbl = self.lance.open_table("chunks").execute().await?;
        let batches: Vec<RecordBatch> = tbl
            .query()
            .select(Select::columns(&["file_path", "chunk_idx", "normalized"]))
            .execute()
            .await?
            .try_collect()
            .await?;

        let mut buckets: HashMap<u64, Vec<(String, usize)>> = HashMap::new();
        for batch in &batches {
            let fp_col = as_string_array(batch.column_by_name("file_path").unwrap());
            let ci_col =
                as_primitive_array::<UInt32Type>(batch.column_by_name("chunk_idx").unwrap());
            let norm_col = as_string_array(batch.column_by_name("normalized").unwrap());
            for i in 0..fp_col.len() {
                let hash = XxHash64::oneshot(0, norm_col.value(i).as_bytes());
                buckets
                    .entry(hash)
                    .or_default()
                    .push((fp_col.value(i).to_string(), ci_col.value(i) as usize));
            }
        }

        let mut to_delete: Vec<(String, usize)> = Vec::new();
        for (_hash, mut entries) in buckets {
            if entries.len() < 2 {
                continue;
            }
            let files: std::collections::HashSet<_> =
                entries.iter().map(|(fp, _)| fp.as_str()).collect();
            if files.len() < 2 {
                continue;
            }
            entries.sort();
            to_delete.extend(entries.into_iter().skip(1));
        }

        let count = to_delete.len();
        if count > 0 {
            let tbl = self.lance.open_table("chunks").execute().await?;
            let conditions: Vec<String> = to_delete
                .iter()
                .map(|(fp, ci)| format!("(file_path = '{}' AND chunk_idx = {ci})", esc(fp)))
                .collect();
            tbl.delete(&conditions.join(" OR ")).await?;
        }
        Ok(count)
    }

    async fn get_repo_map_data(&self) -> anyhow::Result<RepoMapData> {
        let lance = self.lance.clone();
        let (files_res, symbols_res, roles_res) = tokio::join!(
            async {
                lance
                    .open_table("files")
                    .execute()
                    .await?
                    .query()
                    .execute()
                    .await?
                    .try_collect::<Vec<RecordBatch>>()
                    .await
            },
            async {
                lance
                    .open_table("symbols")
                    .execute()
                    .await?
                    .query()
                    .execute()
                    .await?
                    .try_collect::<Vec<RecordBatch>>()
                    .await
            },
            async {
                lance
                    .open_table("symbol_roles")
                    .execute()
                    .await?
                    .query()
                    .execute()
                    .await?
                    .try_collect::<Vec<RecordBatch>>()
                    .await
            }
        );

        let mut files_map: HashMap<String, RepoMapFile> = HashMap::new();
        for batch in &files_res? {
            use arrow_array::types::UInt64Type;
            let fp_col = as_string_array(batch.column_by_name("file_path").unwrap());
            let lang_col = as_string_array(batch.column_by_name("language").unwrap());
            let cc_col =
                as_primitive_array::<UInt64Type>(batch.column_by_name("chunk_count").unwrap());
            for i in 0..fp_col.len() {
                files_map.insert(
                    fp_col.value(i).to_string(),
                    RepoMapFile {
                        path: fp_col.value(i).to_string(),
                        language: lang_col.value(i).to_string(),
                        chunk_count: cc_col.value(i) as usize,
                        role: "internal".to_string(),
                        symbols: Vec::new(),
                    },
                );
            }
        }
        for batch in &roles_res? {
            let fp_col = as_string_array(batch.column_by_name("file_path").unwrap());
            let role_col = as_string_array(batch.column_by_name("role").unwrap());
            for i in 0..fp_col.len() {
                if let Some(f) = files_map.get_mut(fp_col.value(i)) {
                    f.role = role_col.value(i).to_string();
                }
            }
        }
        for batch in &symbols_res? {
            let fp_col = as_string_array(batch.column_by_name("file_path").unwrap());
            let name_col = as_string_array(batch.column_by_name("name").unwrap());
            let kind_col = as_string_array(batch.column_by_name("kind").unwrap());
            let sl_col =
                as_primitive_array::<UInt32Type>(batch.column_by_name("start_line").unwrap());
            for i in 0..fp_col.len() {
                if let Some(f) = files_map.get_mut(fp_col.value(i)) {
                    f.symbols.push(RepoMapSymbol {
                        name: name_col.value(i).to_string(),
                        kind: kind_col.value(i).to_string(),
                        start_line: sl_col.value(i) as usize,
                    });
                }
            }
        }

        let import_edges: Vec<(String, String)> = {
            let graph = self.edge_graph.read().await;
            graph
                .graph
                .edge_indices()
                .map(|eid| {
                    let (s, t) = graph.graph.edge_endpoints(eid).unwrap();
                    (graph.graph[s].clone(), graph.graph[t].clone())
                })
                .collect()
        };

        Ok(RepoMapData {
            files: files_map.into_values().collect(),
            import_edges,
        })
    }

    // --- Call edges ----------------------------------------------------------

    async fn upsert_call_edges(&self, edges: &[CallEdge]) -> anyhow::Result<()> {
        if edges.is_empty() {
            return Ok(());
        }
        use arrow_array::{BooleanArray, Float64Array, StringArray, UInt32Array};
        let schema = Arc::new(super::schemas::call_edges_schema());
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from_iter_values(
                    edges.iter().map(|e| e.caller_file.as_str()),
                )),
                Arc::new(StringArray::from_iter_values(
                    edges.iter().map(|e| e.caller_symbol.as_str()),
                )),
                Arc::new(StringArray::from_iter_values(
                    edges.iter().map(|e| e.callee_name.as_str()),
                )),
                Arc::new(UInt32Array::from_iter_values(
                    edges.iter().map(|e| e.start_line as u32),
                )),
                Arc::new(StringArray::from(
                    edges
                        .iter()
                        .map(|e| e.callee_file.as_deref())
                        .collect::<Vec<_>>(),
                )),
                Arc::new(StringArray::from(
                    edges
                        .iter()
                        .map(|e| e.callee_symbol.as_deref())
                        .collect::<Vec<_>>(),
                )),
                Arc::new(Float64Array::from_iter_values(
                    edges.iter().map(|e| e.confidence),
                )),
                Arc::new(BooleanArray::from(
                    edges.iter().map(|e| e.dynamic).collect::<Vec<bool>>(),
                )),
            ],
        )?;
        let tbl = self.lance.open_table("call_edges").execute().await?;
        {
            let mut mi =
                tbl.merge_insert(&["caller_file", "caller_symbol", "callee_name", "start_line"]);
            mi.when_matched_update_all(None)
                .when_not_matched_insert_all();
            mi.execute(as_reader(batch)).await?;
        }
        Ok(())
    }

    async fn delete_call_edges_for_file(&self, file_path: &str) -> anyhow::Result<()> {
        let tbl = self.lance.open_table("call_edges").execute().await?;
        tbl.delete(&format!("caller_file = '{}'", esc(file_path)))
            .await?;
        Ok(())
    }

    async fn get_callers(
        &self,
        file_path: &str,
        symbol_name: &str,
    ) -> anyhow::Result<Vec<CallEdge>> {
        let filter = format!(
            "callee_file = '{}' AND callee_symbol = '{}'",
            esc(file_path),
            esc(symbol_name)
        );
        let tbl = self.lance.open_table("call_edges").execute().await?;
        let batches: Vec<RecordBatch> = tbl
            .query()
            .only_if(filter)
            .execute()
            .await?
            .try_collect()
            .await?;
        arrow_batches_to_call_edges(&batches)
    }

    async fn get_callees(
        &self,
        file_path: &str,
        symbol_name: &str,
    ) -> anyhow::Result<Vec<CallEdge>> {
        let filter = format!(
            "caller_file = '{}' AND caller_symbol = '{}'",
            esc(file_path),
            esc(symbol_name)
        );
        let tbl = self.lance.open_table("call_edges").execute().await?;
        let batches: Vec<RecordBatch> = tbl
            .query()
            .only_if(filter)
            .execute()
            .await?
            .try_collect()
            .await?;
        arrow_batches_to_call_edges(&batches)
    }
}

// ---------------------------------------------------------------------------
// Arrow ↔ Rust conversions
// ---------------------------------------------------------------------------

pub(crate) fn chunks_to_arrow(chunks: &[ChunkRecord], dim: usize) -> anyhow::Result<RecordBatch> {
    use arrow_array::{
        builder::{FixedSizeListBuilder, Float32Builder},
        StringArray, UInt32Array, UInt8Array,
    };

    let schema = Arc::new(super::schemas::chunks_schema(dim));
    let mut emb_builder = FixedSizeListBuilder::new(Float32Builder::new(), dim as i32);
    let mut doc_emb_builder = FixedSizeListBuilder::new(Float32Builder::new(), dim as i32);

    for chunk in chunks {
        match &chunk.embedding {
            Some(v) if v.len() == dim => {
                for &f in v {
                    emb_builder.values().append_value(f);
                }
                emb_builder.append(true);
            }
            _ => {
                for _ in 0..dim {
                    emb_builder.values().append_value(0.0);
                }
                emb_builder.append(false);
            }
        }
        match &chunk.doc_embedding {
            Some(v) if v.len() == dim => {
                for &f in v {
                    doc_emb_builder.values().append_value(f);
                }
                doc_emb_builder.append(true);
            }
            _ => {
                for _ in 0..dim {
                    doc_emb_builder.values().append_value(0.0);
                }
                doc_emb_builder.append(false);
            }
        }
    }

    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from_iter_values(
                chunks.iter().map(|c| c.file_path.as_str()),
            )),
            Arc::new(UInt32Array::from_iter_values(
                chunks.iter().map(|c| c.chunk_idx as u32),
            )),
            Arc::new(StringArray::from_iter_values(
                chunks.iter().map(|c| c.content.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                chunks.iter().map(|c| c.normalized.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                chunks.iter().map(|c| c.description.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                chunks.iter().map(|c| c.chunk_type.as_str()),
            )),
            Arc::new(UInt32Array::from_iter_values(
                chunks.iter().map(|c| c.start_line as u32),
            )),
            Arc::new(UInt32Array::from_iter_values(
                chunks.iter().map(|c| c.end_line as u32),
            )),
            Arc::new(emb_builder.finish()),
            Arc::new(doc_emb_builder.finish()),
            Arc::new(UInt8Array::from_iter_values(
                chunks.iter().map(|c| c.materialization_tier),
            )),
        ],
    )?)
}

fn arrow_batches_to_chunk_records(batches: &[RecordBatch]) -> anyhow::Result<Vec<ChunkRecord>> {
    let mut records = Vec::new();
    for batch in batches {
        let fp_col = as_string_array(batch.column_by_name("file_path").unwrap());
        let ci_col = as_primitive_array::<UInt32Type>(batch.column_by_name("chunk_idx").unwrap());
        let content_col = as_string_array(batch.column_by_name("content").unwrap());
        let norm_col = as_string_array(batch.column_by_name("normalized").unwrap());
        let desc_col = as_string_array(batch.column_by_name("description").unwrap());
        let ct_col = as_string_array(batch.column_by_name("chunk_type").unwrap());
        let sl_col = as_primitive_array::<UInt32Type>(batch.column_by_name("start_line").unwrap());
        let el_col = as_primitive_array::<UInt32Type>(batch.column_by_name("end_line").unwrap());
        let tier_col =
            as_primitive_array::<UInt8Type>(batch.column_by_name("materialization_tier").unwrap());
        let emb_col = batch
            .column_by_name("embedding")
            .and_then(|c| c.as_any().downcast_ref::<arrow_array::FixedSizeListArray>());

        for i in 0..fp_col.len() {
            let embedding = emb_col.and_then(|ec| {
                if ec.is_null(i) {
                    None
                } else {
                    ec.value(i)
                        .as_any()
                        .downcast_ref::<arrow_array::Float32Array>()
                        .map(|a| (0..a.len()).map(|j| a.value(j)).collect())
                }
            });
            records.push(ChunkRecord {
                file_path: fp_col.value(i).to_string(),
                chunk_idx: ci_col.value(i) as usize,
                content: content_col.value(i).to_string(),
                normalized: norm_col.value(i).to_string(),
                description: desc_col.value(i).to_string(),
                chunk_type: ct_col.value(i).to_string(),
                start_line: sl_col.value(i) as usize,
                end_line: el_col.value(i) as usize,
                embedding,
                doc_embedding: None,
                materialization_tier: tier_col.value(i),
            });
        }
    }
    Ok(records)
}

fn arrow_batches_to_symbols(batches: &[RecordBatch]) -> anyhow::Result<Vec<SymbolDef>> {
    let mut results = Vec::new();
    for batch in batches {
        let fp_col = as_string_array(batch.column_by_name("file_path").unwrap());
        let name_col = as_string_array(batch.column_by_name("name").unwrap());
        let kind_col = as_string_array(batch.column_by_name("kind").unwrap());
        let sl_col = as_primitive_array::<UInt32Type>(batch.column_by_name("start_line").unwrap());
        let el_col = as_primitive_array::<UInt32Type>(batch.column_by_name("end_line").unwrap());
        for i in 0..fp_col.len() {
            results.push(SymbolDef {
                file_path: fp_col.value(i).to_string(),
                name: name_col.value(i).to_string(),
                kind: kind_col.value(i).to_string(),
                start_line: sl_col.value(i) as usize,
                end_line: el_col.value(i) as usize,
            });
        }
    }
    Ok(results)
}

fn arrow_batches_to_call_edges(batches: &[RecordBatch]) -> anyhow::Result<Vec<CallEdge>> {
    let mut results = Vec::new();
    for batch in batches {
        let cf_col = as_string_array(batch.column_by_name("caller_file").unwrap());
        let cs_col = as_string_array(batch.column_by_name("caller_symbol").unwrap());
        let cn_col = as_string_array(batch.column_by_name("callee_name").unwrap());
        let sl_col = as_primitive_array::<UInt32Type>(batch.column_by_name("start_line").unwrap());
        let cef_col = batch
            .column_by_name("callee_file")
            .map(|c| as_string_array(c.as_ref()));
        let ces_col = batch
            .column_by_name("callee_symbol")
            .map(|c| as_string_array(c.as_ref()));
        let conf_col =
            as_primitive_array::<Float64Type>(batch.column_by_name("confidence").unwrap());
        let dyn_col = as_boolean_array(batch.column_by_name("dynamic").unwrap());
        for i in 0..cf_col.len() {
            results.push(CallEdge {
                caller_file: cf_col.value(i).to_string(),
                caller_symbol: cs_col.value(i).to_string(),
                callee_name: cn_col.value(i).to_string(),
                start_line: sl_col.value(i) as usize,
                callee_file: cef_col.and_then(|c| {
                    if c.is_null(i) {
                        None
                    } else {
                        Some(c.value(i).to_string())
                    }
                }),
                callee_symbol: ces_col.and_then(|c| {
                    if c.is_null(i) {
                        None
                    } else {
                        Some(c.value(i).to_string())
                    }
                }),
                confidence: conf_col.value(i),
                dynamic: dyn_col.value(i),
            });
        }
    }
    Ok(results)
}

fn arrow_batches_to_fp_ci_dist(
    batches: &[RecordBatch],
) -> anyhow::Result<Vec<(String, usize, f64)>> {
    let mut results = Vec::new();
    for batch in batches {
        let fp_col = as_string_array(batch.column_by_name("file_path").unwrap());
        let ci_col = as_primitive_array::<UInt32Type>(batch.column_by_name("chunk_idx").unwrap());
        let dist_col = batch
            .column_by_name("_distance")
            .and_then(|c| c.as_any().downcast_ref::<arrow_array::Float32Array>());
        for i in 0..fp_col.len() {
            results.push((
                fp_col.value(i).to_string(),
                ci_col.value(i) as usize,
                dist_col.map(|d| d.value(i) as f64).unwrap_or(1.0),
            ));
        }
    }
    Ok(results)
}

/// Reciprocal Rank Fusion over FTS and vector hit lists.
fn rrf_fuse(
    fts: Vec<(String, usize, f64)>,
    vec_hits: Vec<(String, usize, f64)>,
    top_k: usize,
) -> Vec<(String, usize, f64, &'static str)> {
    const K: f64 = 60.0;
    let mut scores: HashMap<(String, usize), (f64, bool, bool)> = HashMap::new();
    for (rank, (fp, ci, _)) in fts.iter().enumerate() {
        let e = scores
            .entry((fp.clone(), *ci))
            .or_insert((0.0, false, false));
        e.0 += 1.0 / (K + (rank + 1) as f64);
        e.1 = true;
    }
    for (rank, (fp, ci, _)) in vec_hits.iter().enumerate() {
        let e = scores
            .entry((fp.clone(), *ci))
            .or_insert((0.0, false, false));
        e.0 += 1.0 / (K + (rank + 1) as f64);
        e.2 = true;
    }
    let mut ranked: Vec<(String, usize, f64, &'static str)> = scores
        .into_iter()
        .map(|((fp, ci), (score, in_fts, in_vec))| {
            let why = match (in_fts, in_vec) {
                (true, true) => "hybrid",
                (true, false) => "fts",
                _ => "vector",
            };
            (fp, ci, score, why)
        })
        .collect();
    ranked.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    ranked.truncate(top_k);
    ranked
}

/// Escape single quotes in SQL string literals (DataFusion SQL safe).
fn esc(s: &str) -> std::borrow::Cow<'_, str> {
    if s.contains('\'') {
        std::borrow::Cow::Owned(s.replace('\'', "''"))
    } else {
        std::borrow::Cow::Borrowed(s)
    }
}
