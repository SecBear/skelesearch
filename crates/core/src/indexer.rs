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

    /// Walk `root`, detect changed files via the manifest, chunk and embed
    /// them, upsert to the backend, then reconcile deletions/renames.
    pub async fn index_path(&self, root: &Path) -> anyhow::Result<IndexResult> {
        let dim = self.provider.dim();
        self.backend.initialize(dim).await?;

        let chunker = Chunker::default();
        let mut visited: HashSet<String> = HashSet::new();
        let mut result = IndexResult::default();

        // -- Phase 1: Walk and collect files that need re-indexing ----------

        struct FileWork {
            rel_path: String,
            mtime: i64,
            size: i64,
            hash: String,
            lang: String,
            chunks: Vec<crate::ParsedChunk>,
            edges: Vec<crate::ImportEdge>,
        }

        let mut work: Vec<FileWork> = Vec::new();

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

            // Read content once — needed for both hashing and chunking.
            let content = std::fs::read(&abs_path)
                .with_context(|| format!("reading {rel_path}"))?;
            let hash = file_hash(&content);

            // Skip if nothing changed (mtime, size, and content hash all match).
            if self.manifest.is_unchanged(&rel_path, mtime, size, &hash)? {
                continue;
            }

            let source = String::from_utf8_lossy(&content).to_string();
            let lang = language_for(&rel_path);
            let chunks = chunker.chunk_file(&rel_path, &source).unwrap_or_default();
            let edges = chunker.extract_edges(&rel_path, &source).unwrap_or_default();

            work.push(FileWork { rel_path, mtime, size, hash, lang, chunks, edges });
        }

        // -- Phase 2: Delete old data for every file that will be re-indexed -
        //    CozoDB :rm is idempotent, so this is safe for new files too.

        for fw in &work {
            self.backend.delete_chunks_for_file(&fw.rel_path).await?;
            self.backend.delete_edges_for_file(&fw.rel_path).await?;
        }

        // -- Phase 3: Batch-embed all chunks ---------------------------------
        //    Gather all texts, emit in batches of `batch_size` to satisfy the
        //    "fewer calls than chunks" invariant under the counting test.

        let all_texts: Vec<String> = work
            .iter()
            .flat_map(|fw| fw.chunks.iter().map(|c| c.content.clone()))
            .collect();

        let embeddings: Vec<Vec<f32>> = if all_texts.is_empty() {
            vec![]
        } else {
            let mut out = Vec::with_capacity(all_texts.len());
            for batch in all_texts.chunks(self.batch_size) {
                let mut batch_embs = self.provider.embed_batch(batch.to_vec()).await?;
                out.append(&mut batch_embs);
            }
            out
        };

        // -- Phase 4: Upsert files, chunks, and edges ------------------------

        let now = chrono::Utc::now().timestamp();
        let mut emb_iter = embeddings.into_iter();

        for fw in &work {
            let chunk_records: Vec<ChunkRecord> = fw
                .chunks
                .iter()
                .map(|c| ChunkRecord {
                    file_path: fw.rel_path.clone(),
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
                    file_path: fw.rel_path.clone(),
                    language: fw.lang.clone(),
                    last_modified: fw.mtime,
                    last_indexed: now,
                    chunk_count,
                })
                .await?;

            if !chunk_records.is_empty() {
                self.backend.upsert_chunks(&chunk_records).await?;
            }

            let edge_records: Vec<EdgeRecord> = fw
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

            self.manifest.upsert(&fw.rel_path, fw.mtime, fw.size, &fw.hash)?;
            result.indexed_files += 1;
        }

        // -- Phase 5: Reconcile deletions and renames ------------------------
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
        ext if !ext.is_empty() => ext,
        _ => "unknown",
    }
    .to_string()
}
