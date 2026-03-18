use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;
use std::time::UNIX_EPOCH;

use anyhow::Context as _;

use crate::{
    ChunkRecord, Chunker, EdgeRecord, EmbedProvider, FileRecord, ManifestStore, StorageBackend,
};

// ---------------------------------------------------------------------------
// Public output type
// ---------------------------------------------------------------------------

/// Summary produced by a single `Indexer::index_path` run.
#[derive(Debug, Clone, Default)]
pub struct IndexResult {
    /// Number of files that were newly indexed or re-indexed during this run.
    pub indexed_files: usize,
    /// Number of files removed from the index (renames, deletes).
    pub deleted_files: usize,
    /// Total chunks written to the backend during this run.
    pub total_chunks: usize,
}

// ---------------------------------------------------------------------------
// Indexer
// ---------------------------------------------------------------------------

/// Maximum number of files processed in one pipeline batch.
///
/// Bounds peak memory to roughly `FILE_BATCH_SIZE` files worth of chunk
/// content + embeddings.  Larger values reduce backend round-trips at the
/// cost of more in-flight memory.
const FILE_BATCH_SIZE: usize = 50;

/// Orchestrates file walking, change detection, chunking, embedding, and
/// backend upsert.
///
/// `B` is the storage backend; `P` is the embedding provider.
/// Both are held behind `Arc` so callers can retain a handle for inspection
/// after handing the indexer ownership.
pub struct Indexer<B, P> {
    backend: Arc<B>,
    manifest: Arc<ManifestStore>,
    provider: P,
    /// Maximum number of chunk texts sent to the provider in one `embed_batch` call.
    batch_size: usize,
}

impl<B: StorageBackend, P: EmbedProvider> Indexer<B, P> {
    pub fn new(backend: Arc<B>, manifest: Arc<ManifestStore>, provider: P) -> Self {
        Self {
            backend,
            manifest,
            provider,
            batch_size: 64,
        }
    }

    /// Expose the embedding provider for test observability (e.g. call-count
    /// assertions).
    pub fn provider(&self) -> &P {
        &self.provider
    }

    /// Walk `root`, detect changed files via the manifest, chunk and embed
    /// them, upsert to the backend, then reconcile deletions/renames.
    ///
    /// ## Memory contract
    ///
    /// File content, chunk texts, and embeddings are held in memory only for
    /// the current `FILE_BATCH_SIZE`-file batch.  Phase 1 collects metadata
    /// only — no file content is loaded until Phase 2.
    pub async fn index_path(&self, root: &Path) -> anyhow::Result<IndexResult> {
        let dim = self.provider.dim();
        self.backend.initialize(dim).await?;

        let chunker = Chunker::default();
        let mut visited: HashSet<String> = HashSet::new();
        let mut result = IndexResult::default();


        // -- Crash recovery: re-index files from incomplete prior batches -----
        //    A batch is "pending" iff begin_batch was written before a crash
        //    prevented complete_batch from running.  Force those files through
        //    even if mtime_size_unchanged would otherwise skip them.

        let incomplete = self.manifest.find_incomplete_batches()?;
        let force_reindex: HashSet<String> = incomplete
            .iter()
            .flat_map(|b| b.files.iter().cloned())
            .collect();
        // Retire the stale pending rows so they don't accumulate.
        for batch in &incomplete {
            self.manifest.complete_batch(&batch.run_id, batch.batch_idx as usize)?;
        }
        // -- Phase 1: Walk ALL files, collect metadata only -------------------
        //    Use mtime+size as a fast skip — no file content is read here.
        //    Files whose mtime+size match the manifest are assumed unchanged
        //    and skipped immediately.  The full hash check happens in Phase 2
        //    when we read the content anyway.

        struct FileCandidate {
            rel_path: String,
            mtime: i64,
            size: i64,
            lang: String,
        }

        let mut candidates: Vec<FileCandidate> = Vec::new();

        let walker = ignore::WalkBuilder::new(root).build();
        for entry in walker {
            let entry = entry.context("directory walk entry error")?;
            if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
                continue;
            }
            let abs_path = entry.path().to_path_buf();
            let rel_path = abs_path
                .strip_prefix(root)
                .unwrap_or(&abs_path)
                .to_string_lossy()
                .to_string();

            visited.insert(rel_path.clone());

            let meta = std::fs::metadata(&abs_path)
                .with_context(|| format!("metadata for {rel_path}"))?;
            let mtime = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0);
            let size = meta.len() as i64;

            // Fast skip: mtime+size unchanged — unless forced by crash recovery.
            if !force_reindex.contains(&rel_path)
                && self.manifest.mtime_size_unchanged(&rel_path, mtime, size)?
            {
                continue;
            }

            let lang = language_for(&rel_path);
            candidates.push(FileCandidate { rel_path, mtime, size, lang });
        }

        // -- Phase 2: Process candidates in bounded batches -------------------
        //    For each batch: read content, full hash check, chunk, delete
        //    stale backend data, embed, upsert, update manifest.
        //    All content/embedding memory is dropped at end of each iteration.

        let run_id = format!("run_{}", chrono::Utc::now().timestamp_millis());
        let now = chrono::Utc::now().timestamp();
        let mut batch_idx = 0usize;

        for batch in candidates.chunks(FILE_BATCH_SIZE) {
            // Write checkpoint before processing.
            let file_paths: Vec<&str> = batch.iter().map(|f| f.rel_path.as_str()).collect();
            self.manifest.begin_batch(&run_id, batch_idx, &file_paths)?;
            // 2a. Read content, hash, full unchanged check, chunk.
            struct BatchFile<'a> {
                candidate: &'a FileCandidate,
                hash: String,
                chunks: Vec<crate::ParsedChunk>,
                edges: Vec<crate::ImportEdge>,
            }

            let mut batch_files: Vec<BatchFile<'_>> = Vec::with_capacity(batch.len());

            for fc in batch {
                let abs_path = root.join(&fc.rel_path);
                let content = std::fs::read(&abs_path)
                    .with_context(|| format!("reading {}", fc.rel_path))?;
                let hash = file_hash(&content);

                // Rare case: mtime/size changed but content is identical.
                if self.manifest.is_unchanged(&fc.rel_path, fc.mtime, fc.size, &hash)? {
                    continue;
                }

                let source = String::from_utf8_lossy(&content).to_string();
                let chunks = chunker.chunk_file(&fc.rel_path, &source).unwrap_or_default();
                let edges = chunker.extract_edges(&fc.rel_path, &source).unwrap_or_default();

                batch_files.push(BatchFile { candidate: fc, hash, chunks, edges });
            }

            if batch_files.is_empty() {
                continue;
            }

            // 2b. Delete stale backend data for this batch.
            for bf in &batch_files {
                self.backend
                    .delete_chunks_for_file(&bf.candidate.rel_path)
                    .await?;
                self.backend
                    .delete_edges_for_file(&bf.candidate.rel_path)
                    .await?;
            }

            // 2c. Collect chunk texts for this batch, embed in sub-batches.
            let batch_texts: Vec<String> = batch_files
                .iter()
                .flat_map(|bf| bf.chunks.iter().map(|c| c.content.clone()))
                .collect();

            let embeddings: Vec<Vec<f32>> = if batch_texts.is_empty() {
                vec![]
            } else {
                let mut out = Vec::with_capacity(batch_texts.len());
                for sub in batch_texts.chunks(self.batch_size) {
                    let mut embs = self.provider.embed_batch(sub.to_vec()).await?;
                    out.append(&mut embs);
                }
                out
            };

            // 2d. Upsert files, chunks, and edges; update manifest.
            let mut emb_iter = embeddings.into_iter();

            for bf in &batch_files {
                let fc = bf.candidate;

                let chunk_records: Vec<ChunkRecord> = bf
                    .chunks
                    .iter()
                    .map(|c| ChunkRecord {
                        file_path: fc.rel_path.clone(),
                        chunk_idx: c.chunk_idx,
                        content: c.content.clone(),
                        normalized: c.normalized.clone(),
                        chunk_type: c.chunk_type.clone(),
                        start_line: c.start_line,
                        end_line: c.end_line,
                        embedding: emb_iter.next(),
                    })
                    .collect();

                let chunk_count = chunk_records.len();
                result.total_chunks += chunk_count;

                self.backend
                    .upsert_file(&FileRecord {
                        file_path: fc.rel_path.clone(),
                        language: fc.lang.clone(),
                        last_modified: fc.mtime,
                        last_indexed: now,
                        chunk_count,
                    })
                    .await?;

                if !chunk_records.is_empty() {
                    self.backend.upsert_chunks(&chunk_records).await?;
                }

                let edge_records: Vec<EdgeRecord> = bf
                    .edges
                    .iter()
                    .map(|e| EdgeRecord {
                        from_file: e.from_file.clone(),
                        // ImportEdge doesn't carry a chunk index; use 0 as a
                        // sentinel — sufficient for v1 graph traversal.
                        from_chunk: 0,
                        to_file: e.to_file.clone(),
                        edge_type: "imports".into(),
                    })
                    .collect();

                if !edge_records.is_empty() {
                    self.backend.upsert_edges(&edge_records).await?;
                }

                self.manifest.upsert(&fc.rel_path, fc.mtime, fc.size, &bf.hash)?;
                result.indexed_files += 1;
            }
            // batch_files (and all content/embeddings within) drop here.

            self.manifest.complete_batch(&run_id, batch_idx)?;
            batch_idx += 1;
        }

        // Clean up completed batch records for this run.
        self.manifest.clear_completed_batches(&run_id)?;

        // -- Phase 3: Reconcile deletions and renames ------------------------
        //    Any manifest path not visited this run is stale (file gone or
        //    moved).  Remove it from both the backend and the manifest.

        let stale = self.manifest.stale_paths_against(&visited)?;
        result.deleted_files = stale.len();

        for path in &stale {
            self.backend.delete_chunks_for_file(path).await?;
            self.backend.delete_edges_for_file(path).await?;
            self.backend.delete_file(path).await?;
            self.manifest.remove(path)?;
        }

        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute an xxHash3-64 hex digest of `data`.
fn file_hash(data: &[u8]) -> String {
    use std::hash::Hasher as _;
    let mut h = twox_hash::XxHash3_64::default();
    h.write(data);
    format!("{:016x}", h.finish())
}

/// Map a relative file path to a language label.
fn language_for(rel_path: &str) -> String {
    match Path::new(rel_path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
    {
        "rs" => "rust",
        "py" => "python",
        "ts" | "tsx" => "typescript",
        "js" | "jsx" => "javascript",
        "go" => "go",
        "nix" => "nix",
        "java" => "java",
        "c" | "h" => "c",
        "cpp" | "cc" | "cxx" | "hpp" => "cpp",
        "rb" => "ruby",
        "php" => "php",
        "cs" => "csharp",
        "kt" | "kts" => "kotlin",
        "swift" => "swift",
        "scala" | "sc" => "scala",
        ext if !ext.is_empty() => ext,
        _ => "unknown",
    }
    .to_string()
}
