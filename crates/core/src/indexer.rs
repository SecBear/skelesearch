use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;
use std::time::UNIX_EPOCH;

use anyhow::Context as _;

use crate::{
    content_hash, CallEdge, ChunkRecord, Chunker, EdgeRecord, EmbedProvider, FileRecord, ManifestStore,
    SparseEmbedProvider, StorageBackend,
};
use crate::symbols::{extract_import_aliases, extract_references, extract_symbols};
use crate::cochange;


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
    /// When `false`, skip appending extracted symbol names to chunk normalised text.
    /// Allows benchmark profiles to isolate the contribution of symbol enrichment.
    symbol_enrichment: bool,
    /// Extension allowlist. `None` → use `default_include_extensions()`.
    /// `Some(set)` → only index files whose lowercased extension is in the set.
    include_extensions: Option<HashSet<String>>,
    /// Optional provider that generates natural-language descriptions of code chunks.
    /// When set, descriptions are embedded instead of raw code, bridging the
    /// vocabulary gap between natural-language queries and source code.
    summary_provider: Option<Box<dyn crate::summary::SummaryProvider>>,
    /// Prepend AST scope chain to embedding text for structural context.
    scope_prefix: bool,
    /// Optional sparse embedding provider (e.g. BGE-M3 sparse).
    /// When set, sparse vectors are computed and stored for each chunk at index time.
    sparse_provider: Option<Arc<dyn SparseEmbedProvider>>,
    /// Enable dual HNSW indexing: extract doc comments from each chunk and embed
    /// them separately as `doc_embedding`.
    dual_embedding: bool,
}

impl<B: StorageBackend + 'static, P: EmbedProvider> Indexer<B, P> {
    pub fn new(backend: Arc<B>, manifest: Arc<ManifestStore>, provider: P) -> Self {
        Self {
            backend,
            manifest,
            provider,
            batch_size: 64,
            exclude: vec![],
            symbol_enrichment: true,
            include_extensions: None,
            summary_provider: None,
            scope_prefix: false,
            sparse_provider: None,
            dual_embedding: false,
        }
    }

    /// Expose the embedding provider for test observability (e.g. call-count
    /// assertions).
    pub fn provider(&self) -> &P {
        &self.provider
    }


    /// Control whether extracted symbol names are appended to each chunk's
    /// normalised text.  Enabled by default; disable to benchmark without.
    pub fn with_symbol_enrichment(mut self, enabled: bool) -> Self {
        self.symbol_enrichment = enabled;
        self
    }

    /// Enable AST scope-chain prefixes in embedding text.
    /// Prepends `File: X\nScope: impl Foo > fn bar\nType: code` to each chunk.
    pub fn with_scope_prefix(mut self, enabled: bool) -> Self {
        self.scope_prefix = enabled;
        self
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

    /// Override the default extension allowlist.  Pass `None` to use the
    /// built-in list; pass `Some(extensions)` to replace it entirely.
    pub fn with_include_extensions(mut self, extensions: Option<Vec<String>>) -> Self {
        self.include_extensions = extensions.map(|exts| {
            exts.into_iter().map(|e| e.to_lowercase()).collect()
        });
        self
    }
    /// Attach a sparse embedding provider. When set, sparse vectors are computed
    /// and stored alongside dense embeddings for each chunk during indexing.
    pub fn with_sparse_provider(mut self, provider: Arc<dyn SparseEmbedProvider>) -> Self {
        self.sparse_provider = Some(provider);
        self
    }

    /// Enable dual HNSW indexing: extract doc comments from each chunk and embed
    /// them separately as `doc_embedding`. Requires the schema to have the
    /// `chunks:doc_index` HNSW index (initialized via `IndexConfig.dual_embedding = true`).
    pub fn with_dual_embedding(mut self, enabled: bool) -> Self {
        self.dual_embedding = enabled;
        self
    }

    /// Attach an LLM summary provider.  When set, each chunk's description is
    /// generated before embedding; the HNSW index stores description-based
    /// vectors instead of raw-code vectors.
    pub fn with_summary_provider(mut self, provider: Box<dyn crate::summary::SummaryProvider>) -> Self {
        self.summary_provider = Some(provider);
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
        let index_start = std::time::Instant::now();
        let dim = self.provider.dim();
        self.backend.initialize(dim).await?;

        // -- Detect provider/dimension changes ----------------------------------
        //    If the manifest records a different embedding provider or dimension,
        //    stored vectors are incompatible with the current provider's output.
        //    * Dim changed  → hard error: CozoDB's typed vector column cannot
        //      store vectors of a different length without schema recreation.
        //      The user must delete .skelesearch/ to switch dimensions.
        //    * Provider changed, same dim → clear file_hashes so every file
        //      becomes a candidate and gets re-embedded with the new provider.
        //      Phase 2b deletes stale backend data before upserting fresh vectors.
        let stored_provider = self.manifest.get_meta("provider")?;
        let stored_dim = self.manifest.get_meta("dim")?
            .and_then(|s| s.parse::<usize>().ok());
        if let (Some(prev_provider), Some(prev_dim)) = (&stored_provider, stored_dim) {
            let cur_provider = self.provider.name();
            if prev_dim != dim {
                anyhow::bail!(
                    "embedding dimension mismatch: stored index uses provider '{}' (dim={}) \
                     but this run requests provider '{}' (dim={}). \
                     Delete the .skelesearch/ directory to re-index with the new provider.",
                    prev_provider, prev_dim, cur_provider, dim
                );
            } else if prev_provider.as_str() != cur_provider {
                tracing::info!(
                    prev_provider = %prev_provider,
                    cur_provider  = %cur_provider,
                    "embedding provider changed — forcing full re-index"
                );
                self.manifest.clear_file_hashes()?;
                // Also clear the embedding cache: vectors are provider-specific.
                // Same-dim providers produce incompatible vectors; keeping stale
                // entries would silently serve the old provider's embeddings.
                self.manifest.clear_embedding_cache()?;
            }
        }
        // Write provider/dim metadata now (before any indexing work) so that
        // a crash mid-index still leaves the metadata in a consistent state.
        // On the next run the mismatch check above can make correct decisions
        // rather than operating on stale values written only at the very end.
        self.manifest.set_meta("provider", self.provider.name())?;
        self.manifest.set_meta("dim", &self.provider.dim().to_string())?;
        if let Ok(index_root) = root.canonicalize() {
            self.manifest.set_meta("index_root", &index_root.to_string_lossy())?;
        } else {
            self.manifest.set_meta("index_root", &root.to_string_lossy())?;
        }

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

        // Build the effective extension allowlist.  The closure captures a
        // reference so we pay the set-construction cost once, not per file.
        let effective_exts: HashSet<String> = match &self.include_extensions {
            Some(exts) => exts.clone(),
            None => default_include_extensions(),
        };
        let extension_allowed = |path: &Path| -> bool {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            // Permit known extensionless files regardless of the allowlist.
            if is_known_extensionless(name) {
                return true;
            }
            match path.extension().and_then(|e| e.to_str()) {
                Some(ext) => effective_exts.contains(&ext.to_lowercase()),
                // No extension: skip by default.
                None => false,
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

            // Extension allowlist: skip files whose extension is not permitted.
            // Checked before the exclude matcher so binary/media files never
            // reach the gitignore logic — a small but measurable win on large repos.
            if !extension_allowed(&abs_path) {
                tracing::trace!(file = %rel_path, "skipping: extension not in allowlist");
                continue;
            }

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

        // Build the full set of known project-relative paths — current walk
        // candidates plus files already in the index — so import resolvers can
        // check membership without performing any filesystem I/O.
        let indexed_paths = self.backend.list_indexed_paths().await?;
        let all_files: HashSet<String> = candidates
            .iter()
            .map(|c| c.rel_path.clone())
            .chain(indexed_paths.into_iter())
            .collect();

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
                /// Call-site references extracted from source during Phase 2a.
                /// Used in Phase 2c to emit "calls" edges.
                references: Vec<crate::symbols::ReferenceCapture>,
                /// Import aliases extracted during Phase 2a (e.g. `use foo as bar`).
                /// Merged into `resolved_import_targets` during Phase 2d.
                aliases: Vec<crate::symbols::ImportAlias>,
                /// Import-first resolution map built in Phase 2d.
                /// Maps callee name / file stem → resolved project-relative path.
                resolved_import_targets: HashMap<String, String>,
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
                let references = extract_references(&fc.rel_path, &source).unwrap_or_default();
                let aliases = extract_import_aliases(&fc.rel_path, &source);
                // `source` is dropped here — only structured data retained for the batch.

                batch_files.push(BatchFile { candidate: fc, hash, chunks, edges, symbols, references, aliases, resolved_import_targets: HashMap::new() });
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
                self.backend
                    .delete_sparse_for_file(&bf.candidate.rel_path)
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
                // --- 1. Generate descriptions (if summary provider is attached) ---
                //     Descriptions replace raw code as the embedding target, bridging
                //     the vocabulary gap between natural-language queries and source code.
                let descriptions: Vec<String> = if let Some(ref sp) = self.summary_provider {
                    let code_texts: Vec<String> = sub.iter()
                        .map(|(_, c)| c.content.clone())
                        .collect();
                    sp.summarize_batch(code_texts).await?
                } else {
                    vec![String::new(); sub.len()]
                };

                // --- 2. Build the text to embed for each chunk ---
                //     When a description is available, embed it (natural language → HNSW).
                //     When not, use the Anthropic Contextual Retrieval format:
                //     prepend path + type so the model sees where the chunk lives.
                let embed_texts: Vec<String> = sub.iter()
                    .zip(descriptions.iter())
                    .map(|((fi, chunk), desc)| {
                        if !desc.is_empty() {
                            desc.clone()
                        } else {
                            let rel_path = &batch_files[*fi].candidate.rel_path;
                            if self.scope_prefix {
                                let scope = build_scope_chain(
                                    &batch_files[*fi].symbols,
                                    chunk.start_line,
                                    chunk.end_line,
                                );
                                if scope.is_empty() {
                                    format!(
                                        "File: {}\nType: {}\n\n{}",
                                        rel_path, chunk.chunk_type, chunk.content
                                    )
                                } else {
                                    format!(
                                        "File: {}\nScope: {}\nType: {}\n\n{}",
                                        rel_path, scope, chunk.chunk_type, chunk.content
                                    )
                                }
                            } else {
                                format!("{} {}\n{}", rel_path, chunk.chunk_type, chunk.content)
                            }
                        }
                    })
                    .collect();

                // --- 3. Embedding cache lookup keyed on the actual embed text ---
                //     Keying on embed_text (not raw content) means description-based
                //     and raw-code embeddings never collide in the cache.
                let hashes: Vec<String> =
                    embed_texts.iter().map(|t| content_hash(t)).collect();
                let cached = self.manifest.get_cached_embeddings(&hashes, dim)?;

                // Partition into hits and misses.
                let mut miss_indices: Vec<usize> = Vec::new();
                let mut miss_texts: Vec<String> = Vec::new();
                for (i, hit) in cached.iter().enumerate() {
                    if hit.is_none() {
                        miss_indices.push(i);
                        miss_texts.push(embed_texts[i].clone());
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
                let mut fresh_iter = fresh_embs.into_iter();
                let embs: Vec<Vec<f32>> = cached
                    .into_iter()
                    .map(|hit| match hit {
                        Some(v) => v,
                        None => fresh_iter.next().expect("miss count matches fresh_embs"),
                    })
                    .collect();

                for (idx, ((fi, chunk), emb)) in sub.iter().zip(embs).enumerate() {
                    let fc = batch_files[*fi].candidate;
                    chunk_records_per_file[*fi].push(ChunkRecord {
                        file_path: fc.rel_path.clone(),
                        chunk_idx: chunk.chunk_idx,
                        content: chunk.content.clone(),
                        normalized: chunk.normalized.clone(),
                        description: descriptions[idx].clone(),
                        chunk_type: chunk.chunk_type.clone(),
                        start_line: chunk.start_line,
                        end_line: chunk.end_line,
                        embedding: Some(emb),
                        doc_embedding: None,
                    });
                }

                // --- Doc embeddings (optional dual-HNSW) ---
                // Extract doc comments and embed them when dual_embedding is enabled.
                // A missing doc comment results in doc_embedding=None (zero-vector sentinel
                // stored in the schema). Embedding failures are non-fatal.
                if self.dual_embedding {
                    let doc_texts: Vec<(usize, String)> = sub
                        .iter()
                        .enumerate()
                        .filter_map(|(sub_idx, (fi, chunk))| {
                            let lang = &batch_files[*fi].candidate.lang;
                            extract_doc_comment(&chunk.content, lang)
                                .map(|doc| (sub_idx, doc))
                        })
                        .collect();

                    if !doc_texts.is_empty() {
                        let doc_embed_texts: Vec<String> =
                            doc_texts.iter().map(|(_, t)| t.clone()).collect();
                        match self.provider.embed_batch(doc_embed_texts).await {
                            Ok(doc_embs) => {
                                for ((sub_idx, _), doc_emb) in
                                    doc_texts.iter().zip(doc_embs.iter())
                                {
                                    let (fi, chunk) = sub[*sub_idx];
                                    if let Some(cr) = chunk_records_per_file[fi]
                                        .iter_mut()
                                        .find(|r| r.chunk_idx == chunk.chunk_idx)
                                    {
                                        cr.doc_embedding = Some(doc_emb.clone());
                                    }
                                }
                            }
                            Err(e) => {
                                tracing::warn!(
                                    error = %e,
                                    "doc embed_batch failed, skipping doc embeddings for sub-batch"
                                );
                            }
                        }
                    }
                }
            }

            // 2d. Upsert file, chunks, edges, symbols, and manifest for each file.
            for (fi, bf) in batch_files.iter_mut().enumerate() {
                let fc = bf.candidate;
                // Enrich each chunk's normalized text with the names of symbols
                // that overlap its line range, so BM25 queries using symbol names
                // match the relevant chunk even when the name doesn't appear verbatim.
                // Enrich each chunk's normalized text with the names of symbols
                // that overlap its line range only when symbol enrichment is enabled.
                if self.symbol_enrichment {
                    for cr in chunk_records_per_file[fi].iter_mut() {
                        let overlapping: Vec<&crate::symbols::SymbolDef> = bf.symbols.iter()
                            .filter(|s| !s.name.is_empty()
                                && s.start_line <= cr.end_line
                                && s.end_line >= cr.start_line)
                            .collect();
                        if !overlapping.is_empty() {
                            let extra: String = overlapping.iter()
                                .map(|s| format!("{} {}",
                                    crate::normalize_for_fts(&s.name), s.kind))
                                .collect::<Vec<_>>()
                                .join(" ");
                            if !extra.is_empty() {
                                cr.normalized.push(' ');
                                cr.normalized.push_str(&extra);
                            }
                        }
                    }
                }
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

                // -- Sparse embeddings ------------------------------------------------
                // When a sparse provider is configured, compute sparse representations
                // for this file's chunks and store them. Failures are logged and skipped
                // rather than aborting the index run — sparse is an optional enhancement.
                if let Some(ref sp) = self.sparse_provider {
                    let texts: Vec<&str> =
                        chunk_records.iter().map(|cr| cr.content.as_str()).collect();
                    let sp_arc = Arc::clone(sp);
                    let texts_owned: Vec<String> =
                        texts.iter().map(|t| t.to_string()).collect();
                    match tokio::task::spawn_blocking(move || {
                        let refs: Vec<&str> =
                            texts_owned.iter().map(|s| s.as_str()).collect();
                        sp_arc.embed_batch(&refs)
                    })
                    .await
                    {
                        Ok(Ok(sparse_embs)) => {
                            for (cr, sparse) in chunk_records.iter().zip(sparse_embs.iter()) {
                                if !sparse.is_empty() {
                                    if let Err(e) = self
                                        .backend
                                        .store_sparse_vectors(
                                            &cr.file_path,
                                            cr.chunk_idx,
                                            sparse,
                                        )
                                        .await
                                    {
                                        tracing::warn!(
                                            file_path = %cr.file_path,
                                            error = %e,
                                            "store_sparse_vectors failed, skipping"
                                        );
                                    }
                                }
                            }
                        }
                        Ok(Err(e)) => {
                            tracing::warn!(error = %e, "sparse embed_batch failed, skipping chunk batch");
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, "sparse embed_batch task panicked, skipping");
                        }
                    }
                }

                // Resolve raw import captures to canonical project-relative paths.
                // Unresolvable edges (external deps, unknown languages, parse
                // failures) are silently dropped — only intra-project edges are stored.
                let ext = std::path::Path::new(&fc.rel_path)
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("");
                let edge_records: Vec<EdgeRecord> = bf.edges.iter().filter_map(|e| {
                    // Strip language-specific syntax from the raw capture text.
                    let clean = crate::resolve::extract_import_path(&e.to_file, ext)?;
                    // Map the bare token to a project-relative path via the
                    // language-specific resolver; returns None for external deps.
                    let resolved = match crate::resolve::resolver_for_extension(ext) {
                        Some(resolver) => {
                            resolver.resolve(&clean, Path::new(&fc.rel_path), root, &all_files)?
                        }
                        None => return None, // no resolver registered for this language
                    };
                    Some(EdgeRecord {
                        from_file: e.from_file.clone(),
                        // ImportEdge carries no chunk index; 0 is a sentinel
                        // sufficient for v1 graph traversal.
                        from_chunk: 0,
                        to_file: resolved,
                        edge_type: "imports".into(),
                    })
                }).collect();

                if !edge_records.is_empty() {
                    self.backend.upsert_edges(&edge_records).await?;
                }

                // Build import targets map for Phase 2e call-edge resolution.
                // Maps callee name or file stem → resolved project-relative path.
                // We use two keys per resolved edge: the file stem (e.g. "schema"
                // from "crates/core/src/schema.rs") and the full file name with
                // extension (e.g. "schema.rs"), so both namespace-style and
                // direct-import patterns resolve correctly.
                {
                    let mut import_targets: HashMap<String, String> = HashMap::new();
                    for e in &edge_records {
                        if let Some(stem) = Path::new(&e.to_file).file_stem().and_then(|s| s.to_str()) {
                            import_targets.entry(stem.to_string()).or_insert_with(|| e.to_file.clone());
                        }
                        if let Some(name) = Path::new(&e.to_file).file_name().and_then(|s| s.to_str()) {
                            import_targets.entry(name.to_string()).or_insert_with(|| e.to_file.clone());
                        }
                    }
                    // Merge import aliases into the resolution map.
                    // If `bar` is an alias for `foo`, and `foo` maps to a resolved
                    // file path, then `bar` should also map to that same file.
                    for alias in &bf.aliases {
                        if let Some(resolved) = import_targets.get(&alias.original).cloned() {
                            import_targets.entry(alias.alias.clone()).or_insert(resolved);
                        }
                    }
                    bf.resolved_import_targets = import_targets;
                }
                tracing::info!(
                    file = %fc.rel_path,
                    total = bf.edges.len(),
                    resolved = edge_records.len(),
                    "import edges"
                );

                if !bf.symbols.is_empty() {
                    self.backend.upsert_symbols(&bf.symbols).await?;
                }

                self.manifest.upsert(&fc.rel_path, fc.mtime, fc.size, &bf.hash)?;
                result.indexed_files += 1;
            }
            // Phase 2e: Resolve call edges now that all symbols for this batch
            // are committed.  By deferring to here, find_symbols() sees the
            // complete symbol table for all files indexed so far.
            {
                let mut all_call_edges: Vec<CallEdge> = Vec::new();
                for bf in &batch_files {
                    let fc = bf.candidate;
                    self.backend.delete_call_edges_for_file(&fc.rel_path).await?;
                    if bf.references.is_empty() { continue; }

                    // Collect unique callee names for bulk lookup — one DB call
                    // per name, not per reference (avoids N+1).
                    let unique_names: HashSet<&str> =
                        bf.references.iter().map(|r| r.name.as_str()).collect();

                    let mut symbol_map: HashMap<String, Vec<crate::symbols::SymbolDef>> =
                        HashMap::new();
                    for name in &unique_names {
                        match self.backend.find_symbols(name, None).await {
                            Ok(syms) if !syms.is_empty() => {
                                symbol_map.insert(name.to_string(), syms);
                            }
                            _ => {}
                        }
                    }

                    for reference in &bf.references {
                        let caller_symbol =
                            find_enclosing_symbol(&bf.symbols, reference.start_line)
                            .unwrap_or_else(|| "<module>".to_string());
                        let (callee_file, callee_symbol, confidence) = resolve_call(
                            &reference.name,
                            &fc.rel_path,
                            &bf.symbols,
                            &bf.resolved_import_targets,
                            &symbol_map,
                        );
                        all_call_edges.push(CallEdge {
                            caller_file: fc.rel_path.clone(),
                            caller_symbol,
                            callee_name: reference.name.clone(),
                            start_line: reference.start_line,
                            callee_file,
                            callee_symbol,
                            confidence,
                            dynamic: false,
                        });
                    }
                }
                if !all_call_edges.is_empty() {
                    self.backend.upsert_call_edges(&all_call_edges).await?;
                    tracing::info!(count = all_call_edges.len(), "call edges resolved");
                }
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
            self.backend.delete_call_edges_for_file(path).await?;
            self.backend.delete_sparse_for_file(path).await?;
            self.backend.delete_file(path).await?;
            self.manifest.remove(path)?;
        }

        // Spawn PageRank computation as a background task now that all edges are
        // settled.  PageRank is a score boost, not a filter — stale ranks degrade
        // gracefully (new files simply get no boost until recomputation completes).
        //
        // NOTE: compute_pagerank(None) includes ALL edge types (imports + calls).
        // Per-edge-type PageRank (separate import vs call centrality) is deferred
        // until file_ranks supports a rank_type column — two concurrent spawns
        // writing to the same :replace relation would race and the loser's ranks
        // get silently wiped.
        let backend = Arc::clone(&self.backend);
        tokio::spawn(async move {
            if let Err(e) = backend.compute_pagerank(None).await {
                tracing::warn!(error = %e, "background PageRank computation failed");
            } else {
                tracing::info!("PageRank computation completed");
                // Chain symbol role classification immediately after PageRank so
                // roles are available for the next search without a separate trigger.
                if let Err(e) = backend.compute_symbol_roles().await {
                    tracing::warn!(error = %e, "background symbol role classification failed");
                } else {
                    tracing::info!("symbol role classification completed");
                }
            }
        });

        // Co-change analysis: mine git history for files that frequently change
        // together.  This is optional — if git isn't available, the repo has no
        // commits, or the working directory is not a git repo, skip silently.
        {
            let cochange_backend = Arc::clone(&self.backend);
            let cochange_root = root.to_path_buf();
            tokio::spawn(async move {
                // compute_cochange_pairs calls std::process::Command (git log), which is
                // a blocking subprocess.  Offload it to the blocking thread pool so the
                // tokio worker thread is not stalled while git runs.
                let pairs = match tokio::task::spawn_blocking(move || {
                    cochange::compute_cochange_pairs(&cochange_root, 3, 500)
                })
                .await
                {
                    Ok(Ok(p)) if p.is_empty() => return,
                    Ok(Ok(p)) => p,
                    Ok(Err(e)) => {
                        tracing::debug!(error = %e, "co-change analysis skipped");
                        return;
                    }
                    Err(e) => {
                        tracing::debug!(error = %e, "co-change spawn_blocking panicked");
                        return;
                    }
                };
                let n = pairs.len();
                if let Err(e) = cochange_backend.upsert_cochange_edges(&pairs).await {
                    tracing::warn!(error = %e, "co-change edge upsert failed");
                } else {
                    tracing::info!(pairs = n, "co-change analysis complete");
                }
            });
        }

        // Spawn LSH deduplication as a background task.  Near-duplicate chunks
        // across different files waste HNSW graph capacity; removing them improves
        // retrieval diversity.  Like PageRank, this is a quality boost — stale
        // duplicates degrade gracefully (slightly lower recall) until the next
        // full re-index.
        let backend2 = Arc::clone(&self.backend);
        tokio::spawn(async move {
            match backend2.deduplicate_chunks().await {
                Ok(0) => {}
                Ok(n) => tracing::info!(removed = n, "background LSH deduplication completed"),
                Err(e) => tracing::warn!(error = %e, "background LSH deduplication failed"),
            }
        });

        tracing::info!(
            files_indexed = result.indexed_files,
            files_deleted = result.deleted_files,
            chunks_embedded = result.total_chunks,
            parse_errors = result.parse_errors,
            elapsed_ms = index_start.elapsed().as_millis() as u64,
            "index complete"
        );

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

/// The default set of file extensions that the indexer will process.
///
/// Any extension not in this set is silently skipped unless the caller
/// provides an override via `Indexer::with_include_extensions`.
/// The list is lower-cased; comparisons use `ext.to_lowercase()`.
///
/// **To add a new extension**, insert it here and into the relevant
/// category comment.  Do NOT add binary/asset formats here.
pub(crate) fn default_include_extensions() -> HashSet<String> {
    // Keep sorted within each category for ease of review.
    let exts: &[&str] = &[
        // --- Programming languages ---
        "astro", "bash", "bat", "c", "cc", "clj", "cljs", "cljc",
        "cmd", "cpp", "cs", "cxx", "dart", "dhall", "el", "erl",
        "ex", "exs", "fish", "fs", "fsi", "fsx", "go", "graphql",
        "gql", "h", "hcl", "hpp", "hrl", "hs", "java", "jl",
        "js", "jsx", "kt", "kts", "lua", "mjs", "cjs", "ml", "mli",
        "nim", "nix", "php", "proto", "ps1", "py", "r", "rb",
        "rs", "scala", "sh", "sql", "svelte", "swift", "tf", "thrift",
        "ts", "tsx", "v", "vim", "vue", "zig", "zsh",
        // --- Markup / documentation ---
        "adoc", "html", "md", "mdx", "org", "rst", "txt",
        // --- Styles ---
        "css", "less", "sass", "scss",
        // --- Data / config ---
        "cfg", "cmake", "conf", "env", "gradle", "hcl", "ini",
        "json", "makefile", "properties", "toml", "xml", "yaml", "yml",
    ];
    exts.iter().map(|s| s.to_string()).collect()
}

/// Files with no extension that should still be indexed.
///
/// `.gitignore`, `.env`, and other dot-files are intentionally excluded —
/// they contain no indexable code and would bloat the search index.
fn is_known_extensionless(filename: &str) -> bool {
    matches!(
        filename,
        "Makefile" | "Dockerfile" | "Rakefile" | "Gemfile"
            | "Justfile" | "Taskfile" | "Containerfile"
            | "Vagrantfile" | "Brewfile" | "Procfile"
    )
}

/// Find the innermost function/method/class symbol whose line range contains `line`.
/// Returns the symbol name, or `None` if no enclosing scope exists.
/// "Innermost" is the symbol with the smallest enclosing span.
fn find_enclosing_symbol(symbols: &[crate::symbols::SymbolDef], line: usize) -> Option<String> {
    // tree-sitter tags often capture only the declaration line (start == end),
    // not the full function body. Two strategies:
    // 1. If any symbol's range actually spans the line, use smallest enclosing.
    // 2. Otherwise, find the nearest preceding function/method definition —
    //    the last one whose start_line <= call_line and before the next symbol starts.
    let fn_symbols: Vec<&crate::symbols::SymbolDef> = symbols.iter()
        .filter(|s| matches!(s.kind.as_str(), "function" | "method" | "class"))
        .collect();

    // Strategy 1: true enclosing range (when end_line > start_line)
    if let Some(enclosing) = fn_symbols.iter()
        .filter(|s| s.end_line > s.start_line && s.start_line <= line && line <= s.end_line)
        .min_by_key(|s| s.end_line - s.start_line)
    {
        return Some(enclosing.name.clone());
    }

    // Strategy 2: nearest preceding definition (for single-line captures)
    // Sort by start_line, find last one before the call site.
    fn_symbols.iter()
        .filter(|s| s.start_line <= line)
        .max_by_key(|s| s.start_line)
        .map(|s| s.name.clone())
}

/// Resolve a call reference to a (callee_file, callee_symbol, confidence) triple.
///
/// Resolution priority:
/// 1. Same-file symbol match — confidence 1.0.
/// 2. Import-resolved match — confidence 1.0 if symbol found in target, 0.8 if
///    only the target file is known.
/// 3. Cross-file name match scored by path proximity — 0.7 / 0.5 / 0.3.
/// 4. Unresolved — confidence 0.0.
fn resolve_call(
    callee_name: &str,
    caller_file: &str,
    local_symbols: &[crate::symbols::SymbolDef],
    import_targets: &HashMap<String, String>,
    global_symbols: &HashMap<String, Vec<crate::symbols::SymbolDef>>,
) -> (Option<String>, Option<String>, f64) {
    // 1. Same-file match: the callee is defined in the same file.
    if let Some(sym) = local_symbols.iter().find(|s| s.name == callee_name) {
        return (Some(sym.file_path.clone()), Some(sym.name.clone()), 1.0);
    }

    // 2. Import-resolved match: the callee name maps to a known imported file.
    if let Some(target_file) = import_targets.get(callee_name) {
        if let Some(syms) = global_symbols.get(callee_name) {
            if let Some(sym) = syms.iter().find(|s| &s.file_path == target_file) {
                return (Some(sym.file_path.clone()), Some(sym.name.clone()), 1.0);
            }
        }
        // Target file known via import, but no matching symbol name in the DB.
        return (Some(target_file.clone()), None, 0.8);
    }

    // 3. Cross-file match scored by path proximity.
    if let Some(syms) = global_symbols.get(callee_name) {
        if let Some(best) = syms.first() {
            let confidence = path_proximity(caller_file, &best.file_path);
            return (Some(best.file_path.clone()), Some(best.name.clone()), confidence);
        }
    }

    // 4. Unresolved.
    (None, None, 0.0)
}

/// Score path proximity between two project-relative file paths.
/// Same directory → 0.7, same parent directory → 0.5, otherwise → 0.3.
fn path_proximity(a: &str, b: &str) -> f64 {
    let a_dir = Path::new(a).parent().unwrap_or(Path::new(""));
    let b_dir = Path::new(b).parent().unwrap_or(Path::new(""));
    if a_dir == b_dir {
        return 0.7;
    }
    let a_parent = a_dir.parent().unwrap_or(Path::new(""));
    let b_parent = b_dir.parent().unwrap_or(Path::new(""));
    if a_parent == b_parent {
        return 0.5;
    }
    0.3
}

/// Build a human-readable scope chain from symbols that enclose a chunk's line range.
///
/// Returns e.g. `"impl Searcher > fn search"` for a chunk inside a method,
/// or an empty string for top-level code with no enclosing symbol.
///
/// Algorithm:
/// 1. Find all symbols whose line range fully contains the chunk.
/// 2. Sort widest (outermost) first.
/// 3. Join as `"{kind} {name} > ..."` from outer to inner.
pub(crate) fn build_scope_chain(
    symbols: &[crate::symbols::SymbolDef],
    chunk_start: usize,
    chunk_end: usize,
) -> String {
    let mut enclosing: Vec<&crate::symbols::SymbolDef> = symbols
        .iter()
        .filter(|s| s.start_line <= chunk_start && s.end_line >= chunk_end)
        .collect();

    if enclosing.is_empty() {
        return String::new();
    }

    // Sort by span width descending (widest/outermost first).
    enclosing.sort_by_key(|s| std::cmp::Reverse(s.end_line.saturating_sub(s.start_line)));

    enclosing
        .iter()
        .map(|s| format!("{} {}", short_kind(&s.kind), &s.name))
        .collect::<Vec<_>>()
        .join(" > ")
}

/// Map tree-sitter tag kinds to short display forms.
fn short_kind(kind: &str) -> &str {
    match kind {
        "function" => "fn",
        "method" => "method",
        "struct" => "struct",
        "class" => "class",
        "impl" => "impl",
        "trait" => "trait",
        "enum" => "enum",
        "type" => "type",
        "interface" => "interface",
        "module" => "mod",
        other => other,
    }
}

/// Extract a leading doc comment / docstring from a chunk's source text.
///
/// Strategy by language pattern:
/// - Rust/Go/TypeScript/JS: consecutive `///`, `//!`, or `//` lines at the
///   start of the chunk (up to 30 lines).
/// - Python: a triple-quoted string `\"\"\"...\"\"\"` or `'''...'''` at the chunk start.
/// - Block comments: `/** ... */` or `/* ... */` at the chunk start.
///
/// Returns `None` when no doc comment is found or when the extracted text is
/// shorter than 10 characters (too short to produce a useful embedding).
pub(crate) fn extract_doc_comment(content: &str, _lang: &str) -> Option<String> {
    let trimmed = content.trim_start();

    // Python triple-quoted docstring (double or single quotes).
    for delim in ["\"\"\"", "'''"] {
        if trimmed.starts_with(delim) {
            let rest = &trimmed[delim.len()..];
            if let Some(end) = rest.find(delim) {
                let doc = rest[..end].trim().to_string();
                if doc.len() >= 10 {
                    return Some(doc);
                }
            }
        }
    }

    // Block comment: /** ... */ or /* ... */
    if trimmed.starts_with("/**") || trimmed.starts_with("/*") {
        if let Some(end) = trimmed.find("*/") {
            let doc: String = trimmed[..end]
                .lines()
                .map(|l| l.trim_start_matches(|c: char| c == '/' || c == '*' || c == ' '))
                .collect::<Vec<_>>()
                .join(" ")
                .trim()
                .to_string();
            if doc.len() >= 10 {
                return Some(doc);
            }
        }
    }

    // Line comments: ///, //!, or //.
    let mut doc_lines: Vec<&str> = Vec::new();
    for line in content.lines().take(30) {
        let t = line.trim();
        if t.starts_with("///") {
            doc_lines.push(t.trim_start_matches('/').trim());
        } else if t.starts_with("//!") {
            doc_lines.push(t.trim_start_matches(|c: char| c == '/' || c == '!').trim());
        } else if t.starts_with("//") && !doc_lines.is_empty() {
            // Continue a // comment block only if already started.
            doc_lines.push(t.trim_start_matches('/').trim());
        } else if !t.is_empty() {
            // First non-comment, non-blank line stops the scan.
            break;
        }
    }

    // Second pass: allow a // block to start a doc comment.
    if doc_lines.is_empty() {
        for line in content.lines().take(30) {
            let t = line.trim();
            if t.starts_with("//") {
                doc_lines.push(t.trim_start_matches('/').trim());
            } else if !t.is_empty() {
                break;
            }
        }
    }

    let doc = doc_lines.join(" ").trim().to_string();
    if doc.len() >= 10 { Some(doc) } else { None }
}

#[cfg(test)]
mod scope_chain_tests {
    use super::build_scope_chain;
    use crate::symbols::SymbolDef;

    fn sym(name: &str, kind: &str, start: usize, end: usize) -> SymbolDef {
        SymbolDef {
            file_path: "test.rs".to_string(),
            name: name.to_string(),
            kind: kind.to_string(),
            start_line: start,
            end_line: end,
        }
    }

    #[test]
    fn empty_symbols() {
        assert_eq!(build_scope_chain(&[], 5, 10), "");
    }

    #[test]
    fn no_enclosing_symbol() {
        let symbols = vec![sym("search", "function", 20, 50)];
        assert_eq!(build_scope_chain(&symbols, 5, 10), "");
    }

    #[test]
    fn single_enclosing_function() {
        let symbols = vec![sym("search", "function", 3, 15)];
        assert_eq!(build_scope_chain(&symbols, 5, 10), "fn search");
    }

    #[test]
    fn nested_impl_and_method() {
        let symbols = vec![
            sym("Searcher", "impl", 1, 50),
            sym("do_search", "function", 8, 20),
        ];
        assert_eq!(
            build_scope_chain(&symbols, 10, 15),
            "impl Searcher > fn do_search"
        );
    }

    #[test]
    fn three_levels_deep() {
        let symbols = vec![
            sym("Router", "class", 1, 100),
            sym("get", "method", 10, 50),
            sym("helper", "function", 12, 25),
        ];
        assert_eq!(
            build_scope_chain(&symbols, 15, 20),
            "class Router > method get > fn helper"
        );
    }

    #[test]
    fn partial_overlap_not_enclosing() {
        // Symbol only partially contains the chunk — should not match.
        let symbols = vec![sym("search", "function", 10, 20)];
        assert_eq!(build_scope_chain(&symbols, 5, 15), "");
    }

    #[test]
    fn sibling_symbols_only_enclosing_matches() {
        let symbols = vec![
            sym("a", "function", 1, 20),
            sym("b", "function", 25, 40),
        ];
        assert_eq!(build_scope_chain(&symbols, 5, 10), "fn a");
    }
}