use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;
use std::time::UNIX_EPOCH;

use anyhow::Context as _;

use crate::{
    content_hash, ChunkRecord, Chunker, EdgeRecord, EmbedProvider, FileRecord, ManifestStore,
    StorageBackend,
};
use crate::symbols::extract_symbols;

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
    /// Number of files that failed to parse during chunking.
    pub parse_errors: usize,
    /// Number of chunk embeddings served from the persistent cache.
    pub cache_hits: usize,
    /// Number of chunk embeddings computed fresh by the provider.
    pub cache_misses: usize,
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
    /// Gitignore-style patterns for files/directories to skip during indexing.
    exclude: Vec<String>,
}

impl<B: StorageBackend, P: EmbedProvider> Indexer<B, P> {
    pub fn new(backend: Arc<B>, manifest: Arc<ManifestStore>, provider: P) -> Self {
        Self {
            backend,
            manifest,
            provider,
            batch_size: 64,
            exclude: vec![],
        }
    }

    /// Expose the embedding provider for test observability (e.g. call-count
    /// assertions).
    pub fn provider(&self) -> &P {
        &self.provider
    }


    /// Set gitignore-style path patterns that the indexer will skip.
    ///
    /// Supported cases: exact relative paths (`vendor/big.rs`), directory
    /// prefixes with a trailing slash (`target/`, `node_modules/`), and
    /// glob wildcards (`*.lock`, `**/*.min.js`).
    pub fn with_excludes(mut self, exclude: Vec<String>) -> Self {
        self.exclude = exclude;
        self
    }
    /// Walk `root`, detect changed files via the manifest, chunk and embed
    /// them, upsert to the backend, then reconcile deletions/renames.
    ///
    /// ## Memory contract
    ///
    /// File content, chunk texts, and embeddings are held in memory only for
    /// the current `FILE_BATCH_SIZE`-file batch.  Phase 1 collects metadata
    /// only — no file content is loaded until Phase 2.
    #[tracing::instrument(skip_all, fields(root = %root.display()))]
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
        //
        //    We do NOT mark stale pending rows complete here — that would lie
        //    about the data.  Instead, we reindex their files in this run and
        //    delete the stale rows only after all new batches succeed.

        let incomplete = self.manifest.find_incomplete_batches()?;
        let force_reindex: HashSet<String> = incomplete
            .iter()
            .flat_map(|b| b.files.iter().cloned())
            .collect();
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

        // Build an exclude matcher from config patterns using gitignore semantics.
        // This covers directory prefixes (target/), globs (*.lock, **/*.min.js),
        // and exact paths (vendor/big.rs) without adding new dependencies.
        let exclude_matcher: Option<ignore::gitignore::Gitignore> = if self.exclude.is_empty() {
            None
        } else {
            let mut builder = ignore::gitignore::GitignoreBuilder::new(root);
            for pat in &self.exclude {
                if let Err(e) = builder.add_line(None, pat) {
                    tracing::warn!(pattern = %pat, error = %e, "invalid exclude pattern, skipping");
                }
            }
            match builder.build() {
                Ok(gi) => Some(gi),
                Err(e) => {
                    tracing::warn!(error = %e, "failed to build exclude matcher, excludes disabled");
                    None
                }
            }
        };

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

            // Skip files matching any exclude pattern before adding to `visited` so
            // previously-indexed excluded files are reconciled out on the next run.
            if let Some(ref exc) = exclude_matcher {
                if exc.matched_path_or_any_parents(Path::new(&rel_path), false).is_ignore() {
                    continue;
                }
            }

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
                symbols: Vec<crate::symbols::SymbolDef>,
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

                // Skip binary files: check for null bytes in the first 8KB.
                let check_len = content.len().min(8192);
                if content[..check_len].contains(&0u8) {
                    tracing::debug!(file = %fc.rel_path, "skipping binary file");
                    continue;
                }

                let source = String::from_utf8_lossy(&content).to_string();
                let chunks = match chunker.chunk_file(&fc.rel_path, &source) {
                    Ok(c) => c,
                    Err(e) => {
                        tracing::warn!(file = %fc.rel_path, error = %e, "chunk parse failed, skipping");
                        result.parse_errors += 1;
                        continue;
                    }
                };
                let edges = chunker.extract_edges(&fc.rel_path, &source).unwrap_or_default();
                let symbols = extract_symbols(&fc.rel_path, &source).unwrap_or_default();

                batch_files.push(BatchFile { candidate: fc, hash, chunks, edges, symbols });
            }

            if batch_files.is_empty() {
                // All files in this slice were filtered out (unchanged hash, binary,
                // or parse fail). Complete the batch so no phantom pending row is left.
                self.manifest.complete_batch(&run_id, batch_idx)?;
                batch_idx += 1;
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
                self.backend
                    .delete_symbols_for_file(&bf.candidate.rel_path)
                    .await?;
            }

            // 2c. Build a cross-file pending list of (file_idx, chunk) pairs —
            //     *without* copying texts — then embed in sub-batches of
            //     self.batch_size.  This means we never hold a Vec<String> for all
            //     chunk texts across the whole file batch at once; the text clone
            //     for each sub-batch is bounded to batch_size entries.
            //
            //     Results are distributed into per-file ChunkRecord accumulators.
            let pending: Vec<(usize, &crate::ParsedChunk)> = batch_files
                .iter()
                .enumerate()
                .flat_map(|(fi, bf)| bf.chunks.iter().map(move |c| (fi, c)))
                .collect();

            let mut chunk_records_per_file: Vec<Vec<ChunkRecord>> =
                vec![Vec::new(); batch_files.len()];

            for sub in pending.chunks(self.batch_size) {
                // --- embedding cache lookup ---
                let hashes: Vec<String> =
                    sub.iter().map(|(_, c)| content_hash(&c.content)).collect();
                let cached = self.manifest.get_cached_embeddings(&hashes)?;

                // Partition into hits and misses; track original sub-indices.
                let mut miss_indices: Vec<usize> = Vec::new();
                let mut miss_texts: Vec<String> = Vec::new();
                for (i, hit) in cached.iter().enumerate() {
                    if hit.is_none() {
                        miss_indices.push(i);
                        miss_texts.push(sub[i].1.content.clone());
                    }
                }

                let cached_count = sub.len() - miss_indices.len();
                let miss_count = miss_indices.len();
                tracing::debug!(hits = cached_count, misses = miss_count, "embedding cache");
                result.cache_hits += cached_count;
                result.cache_misses += miss_count;

                // Embed only the cache misses (skip provider call if none).
                let fresh_embs: Vec<Vec<f32>> = if miss_texts.is_empty() {
                    vec![]
                } else {
                    let result = self.provider.embed_batch(miss_texts).await?;
                    anyhow::ensure!(
                        result.len() == miss_count,
                        "embedding count mismatch: provider returned {} vectors for {} texts",
                        result.len(),
                        miss_count
                    );
                    result
                };

                // Persist fresh embeddings to cache.
                let to_cache: Vec<(String, Vec<f32>)> = miss_indices
                    .iter()
                    .zip(fresh_embs.iter())
                    .map(|(&i, emb)| (hashes[i].clone(), emb.clone()))
                    .collect();
                self.manifest.cache_embeddings(&to_cache)?;

                // Merge cached + fresh into per-original-index embeddings.
                // cached[i] is Some(hit) or None (miss); fill misses from fresh_embs in order.
                let mut fresh_iter = fresh_embs.into_iter();
                let embs: Vec<Vec<f32>> = cached
                    .into_iter()
                    .map(|hit| match hit {
                        Some(v) => v,
                        None => fresh_iter.next().expect("miss count matches fresh_embs"),
                    })
                    .collect();

                for ((fi, chunk), emb) in sub.iter().zip(embs) {
                    let fc = batch_files[*fi].candidate;
                    chunk_records_per_file[*fi].push(ChunkRecord {
                        file_path: fc.rel_path.clone(),
                        chunk_idx: chunk.chunk_idx,
                        content: chunk.content.clone(),
                        normalized: chunk.normalized.clone(),
                        chunk_type: chunk.chunk_type.clone(),
                        start_line: chunk.start_line,
                        end_line: chunk.end_line,
                        embedding: Some(emb),
                    });
                }
            }

            // 2d. Upsert file, chunks, edges, symbols, and manifest for each file.
            for (fi, bf) in batch_files.iter().enumerate() {
                let fc = bf.candidate;
                let chunk_records = &chunk_records_per_file[fi];
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
                    self.backend.upsert_chunks(chunk_records).await?;
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

                if !bf.symbols.is_empty() {
                    self.backend.upsert_symbols(&bf.symbols).await?;
                }

                self.manifest.upsert(&fc.rel_path, fc.mtime, fc.size, &bf.hash)?;
                result.indexed_files += 1;
            }
            // batch_files and chunk_records_per_file drop here.

            self.manifest.complete_batch(&run_id, batch_idx)?;
            batch_idx += 1;
        }

        // Clean up stale pending rows from crashed prior runs.  Only reached
        // after all current-run batches have successfully completed, so the
        // reindexed files are committed before we retire the crash evidence.
        for batch in &incomplete {
            self.manifest.delete_batch(&batch.run_id, batch.batch_idx)?;
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
            self.backend.delete_symbols_for_file(path).await?;
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
