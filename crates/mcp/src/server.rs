// SkeleSearchServer — rmcp ServerHandler implementation.
//
// Design notes:
//   - SkeleSearchServer holds shared state behind Arc so it is cheaply Clone.
//   - ManifestStore wraps a raw SQLite pointer and is therefore !Sync.  Holding
//     Arc<ManifestStore> in the server would make it !Sync, which is incompatible
//     with rmcp's ServerHandler bound.  We store the manifest DB path instead and
//     open a fresh ManifestStore per index_codebase call (safe because SQLite
//     handles concurrent access via file locking).
//   - ArcProvider wraps Arc<dyn EmbedProvider> so the server type is concrete
//     and Clone without a generic type parameter.
//   - The #[tool_router] block declares the four MCP tools; each delegates to a
//     public method that tests can call directly.
//   - Provider selection is explicit: unknown names return a clear error.

use std::{collections::{HashMap, HashSet}, path::PathBuf, sync::{Arc, Mutex, RwLock}};

use anyhow::Context as _;
use async_trait::async_trait;
use rmcp::{
    ServerHandler, ServiceExt,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{Implementation, ServerCapabilities, ServerInfo},
    tool, tool_handler, tool_router,
};
use skelesearch_core::{classify_query, grep_codebase, CozoBackend, Config, EmbedProvider, GrepOptions, Indexer, LLMExpander, ManifestStore, QueryExpander, QueryStrategy, Reranker, Searcher, StorageBackend};
use skelesearch_embed_fastembed::provider_from_name;

use crate::tools::{
    ChunkInfo, FileContextOutput, FindImpactSetInput, FindSymbolInput, FindTestContextInput,
    GetFileContextInput, GrepSearchRow, ImpactEntry, ImpactSetOutput, IndexCodebaseInput,
    IndexCodebaseOutput, IndexStatusInput, IndexStatusOutput, SearchCodeInput, SearchCodeResponse, SearchCodeRow,
    SmartSearchInput, SmartSearchOutput, SmartSearchResults, SymbolRow, TestContextOutput,
};

// ---------------------------------------------------------------------------
// ArcProvider — Clone wrapper around Arc<dyn EmbedProvider>
// ---------------------------------------------------------------------------

/// Wraps a type-erased provider in an Arc so `SkeleSearchServer` can be Clone.
#[derive(Clone)]
pub struct ArcProvider(pub Arc<dyn EmbedProvider + Send + Sync>);

impl ArcProvider {
    pub fn new(p: impl EmbedProvider + Send + Sync + 'static) -> Self {
        Self(Arc::new(p))
    }
}

#[async_trait]
impl EmbedProvider for ArcProvider {
    fn dim(&self) -> usize {
        self.0.dim()
    }

    fn name(&self) -> &str {
        self.0.name()
    }

    async fn embed_batch(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
        self.0.embed_batch(texts).await
    }

    async fn embed_queries(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
        self.0.embed_queries(texts).await
    }

    fn query_prefix(&self) -> Option<&str> {
        self.0.query_prefix()
    }
}

// ---------------------------------------------------------------------------
// SkeleSearchServer
// ---------------------------------------------------------------------------

/// MCP server that exposes four code-search tools.
///
/// # Thread safety
/// All fields must be `Send + Sync` because rmcp's `ServerHandler` requires it.
/// The only tricky one is the manifest: `ManifestStore` wraps a raw SQLite pointer
/// and is therefore `!Sync`.  We avoid this by storing the manifest's file path
/// and opening a fresh connection per index operation rather than sharing one.
/// Type alias for the concrete searcher used by the MCP server.
type CachedSearcher = Searcher<CozoBackend, ArcProvider>;

#[derive(Clone)]
pub struct SkeleSearchServer {
    backend: Arc<CozoBackend>,
    /// Path to the manifest SQLite database; opened fresh per index_codebase call.
    manifest_path: Arc<PathBuf>,
    /// Provider used for query embedding in `search_code`.  Wrapped in an
    /// RwLock so `run_index` can promote it to the real provider after a
    /// successful indexing run without requiring a full server restart.
    provider: Arc<RwLock<ArcProvider>>,
    tool_router: ToolRouter<Self>,
    /// Tracks content hashes seen per session for dedup.
    /// TODO: add periodic cleanup for long-running servers (sessions accumulate in memory).
    sessions: Arc<Mutex<HashMap<String, HashSet<u64>>>>,
    /// Cached searcher — built once on first search, invalidated after indexing.
    /// Keeps the LRU query-embedding cache and TCP connection pool alive across
    /// MCP calls, eliminating cold TLS handshakes and redundant embed API calls.
    cached_searcher: Arc<tokio::sync::RwLock<Option<Arc<CachedSearcher>>>>,
}

// ---------------------------------------------------------------------------
// Error classification
// ---------------------------------------------------------------------------

/// Returns true when `err` is a CozoDB "Stored relation … not found" error,
/// which occurs when the DB has never been initialized (no tables exist yet).
/// We treat this as an empty index, not a hard failure.
fn is_uninitialized_index_error(err: &anyhow::Error) -> bool {
    let msg = err.to_string().to_lowercase();
    msg.contains("stored relation") && msg.contains("not found")
}


impl SkeleSearchServer {
    /// Construct the server.
    ///
    /// `manifest_path` is the filesystem path to the manifest database (will be
    /// created if absent).  `provider` is used for search-time query embedding.
    pub fn new(
        backend: Arc<CozoBackend>,
        manifest_path: impl Into<PathBuf>,
        provider: impl EmbedProvider + Send + Sync + 'static,
    ) -> Self {
        Self {
            backend,
            manifest_path: Arc::new(manifest_path.into()),
            provider: Arc::new(RwLock::new(ArcProvider::new(provider))),
            tool_router: Self::tool_router(),
            sessions: Arc::new(Mutex::new(HashMap::new())),
            cached_searcher: Arc::new(tokio::sync::RwLock::new(None)),
        }
    }

    // -----------------------------------------------------------------------
    // Public API — callable directly from tests
    // -----------------------------------------------------------------------

    /// Returns the MCP tool names exposed by this server, sorted alphabetically.
    pub async fn tool_names(&self) -> anyhow::Result<Vec<String>> {
        Ok(self
            .tool_router
            .list_all()
            .iter()
            .map(|t| t.name.to_string())
            .collect())
    }

    /// Ensure a usable embedding provider is ready before a search.
    ///
    /// Returns an error when the index is empty so callers get a clear
    /// message rather than empty results.  When the server started with
    /// `NoopProvider` (dim == 1) but a persistent index exists, lazily
    /// initializes the correct provider (read from the manifest) and promotes
    /// it for future calls.
    async fn prepare_search_provider(&self) -> anyhow::Result<ArcProvider> {
        let stats = match self.backend.stats().await {
            Ok(s) => s,
            Err(ref e) if is_uninitialized_index_error(e) => {
                return Err(anyhow::anyhow!("index is empty; run index_codebase first"));
            }
            Err(e) => return Err(e),
        };
        if stats.total_chunks == 0 {
            return Err(anyhow::anyhow!("index is empty; run index_codebase first"));
        }

        // Fast path: already a real provider.
        {
            let guard = self.provider.read().map_err(|_| anyhow::anyhow!("provider lock poisoned"))?;
            if guard.dim() > 1 {
                return Ok(guard.clone());
            }
        }

        // Slow path: started with NoopProvider (dim == 1) but a persisted
        // index exists.  Lazily initialize the real provider and promote it
        // so subsequent calls skip this branch.
        // Read which provider built this index so we use the correct embedding
        // dimension (e.g. voyage=1024, openai=1536, fastembed=768).
        let provider_name = {
            let manifest = ManifestStore::open(self.manifest_path.as_path())
                .context("failed to open manifest")?;
            manifest.get_meta("provider")
                .context("failed to read provider from manifest")?
                .unwrap_or_else(|| "fastembed".to_string())
        };
        let real = provider_from_name(&provider_name)
            .with_context(|| format!("failed to initialize provider '{provider_name}'"))?;
        let arc_provider = ArcProvider::new(real);
        *self.provider.write().map_err(|_| anyhow::anyhow!("provider lock poisoned"))? = arc_provider.clone();
        Ok(arc_provider)
    }

    // -----------------------------------------------------------------------
    // Session dedup helpers
    // -----------------------------------------------------------------------

    /// Record that these content hashes were returned in this session.
    fn record_seen(&self, session_id: &str, hashes: &[u64]) {
        if let Ok(mut sessions) = self.sessions.lock() {
            let seen = sessions.entry(session_id.to_string()).or_default();
            seen.extend(hashes);
        }
    }

    /// Return the set of content hashes seen so far in this session.
    fn get_seen(&self, session_id: &str) -> HashSet<u64> {
        self.sessions.lock()
            .map(|s| s.get(session_id).cloned().unwrap_or_default())
            .unwrap_or_default()
    }

    /// Stable hash of a chunk's full content string for session dedup.
    fn content_hash(s: &str) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        s.hash(&mut hasher);
        hasher.finish()
    }

    /// Auto-detect available API keys and configure the search pipeline.
    /// Called once per search — cheap (just env var lookups), no memoization needed.
    fn auto_configure_pipeline(&self) -> (Option<Box<dyn QueryExpander>>, Option<Box<dyn Reranker>>) {
        let expander: Option<Box<dyn QueryExpander>> =
            std::env::var("OPENAI_API_KEY").ok()
                .filter(|k| !k.is_empty())
                .map(|key| -> Box<dyn QueryExpander> {
                    Box::new(LLMExpander::new(key))
                });

        // Try reranker keys in order: JINA_API_KEY, COHERE_API_KEY, VOYAGE_API_KEY.
        let reranker: Option<Box<dyn Reranker>> = None
            .or_else(|| {
                std::env::var("JINA_API_KEY").ok()
                    .filter(|k| !k.is_empty())
                    .and_then(|key| skelesearch_rerank_api::reranker_from_name("jina", key).ok())
                    .map(|r| -> Box<dyn Reranker> { Box::new(r) })
            })
            .or_else(|| {
                std::env::var("COHERE_API_KEY").ok()
                    .filter(|k| !k.is_empty())
                    .and_then(|key| skelesearch_rerank_api::reranker_from_name("cohere", key).ok())
                    .map(|r| -> Box<dyn Reranker> { Box::new(r) })
            })
            .or_else(|| {
                std::env::var("VOYAGE_API_KEY").ok()
                    .filter(|k| !k.is_empty())
                    .and_then(|key| skelesearch_rerank_api::reranker_from_name("voyage", key).ok())
                    .map(|r| -> Box<dyn Reranker> { Box::new(r) })
            })
            .or_else(|| {
                // SKELESEARCH_RERANKER=local enables the local ONNX reranker.
                // SKELESEARCH_RERANKER_MODEL_DIR overrides the default cache path.
                // Best with CoreML (--features coreml) on Apple Silicon — model stays warm.
                let local = std::env::var("SKELESEARCH_RERANKER").ok()
                    .filter(|v| v == "local");
                if local.is_none() { return None; }
                let result = if let Ok(dir) = std::env::var("SKELESEARCH_RERANKER_MODEL_DIR") {
                    let expanded = if dir.starts_with("~/") {
                        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
                        std::path::PathBuf::from(home).join(&dir[2..])
                    } else {
                        std::path::PathBuf::from(&dir)
                    };
                    skelesearch_rerank_local::LocalReranker::new(&expanded)
                } else {
                    skelesearch_rerank_local::LocalReranker::default_model()
                };
                result.ok().map(|r| -> Box<dyn Reranker> { Box::new(r) })
            });

        if expander.is_some() {
            tracing::info!("query expansion enabled (OPENAI_API_KEY detected)");
        }
        if reranker.is_some() {
            let source = if std::env::var("SKELESEARCH_RERANKER").ok().filter(|v| v == "local").is_some() {
                "local ONNX model"
            } else {
                "cloud API key"
            };
            tracing::info!(source, "reranking enabled");
        }

        (expander, reranker)
    }

    /// Return a cached Searcher or build one on the first call.
    /// The searcher is invalidated (cache cleared) after indexing so
    /// provider changes and config changes are picked up.
    async fn get_or_build_searcher(&self) -> anyhow::Result<Arc<CachedSearcher>> {
        // Fast path: cached searcher exists.
        {
            let guard = self.cached_searcher.read().await;
            if let Some(ref s) = *guard {
                return Ok(Arc::clone(s));
            }
        }

        // Slow path: build and cache.
        let mut guard = self.cached_searcher.write().await;
        // Double-check after acquiring write lock.
        if let Some(ref s) = *guard {
            return Ok(Arc::clone(s));
        }

        let provider = self.prepare_search_provider().await?;
        let searcher = Searcher::new(Arc::clone(&self.backend), provider);
        let (expander, reranker) = self.auto_configure_pipeline();
        let searcher = if let Some(e) = expander { searcher.with_expander(e) } else { searcher };
        let searcher = if let Some(r) = reranker { searcher.with_reranker(r) } else { searcher };
        // Apply pagerank_boost and tuning from project config.
        let searcher = {
            let root = self.backend.list_indexed_paths().await
                .ok()
                .and_then(|p| common_ancestor(&p))
                .unwrap_or_else(|| PathBuf::from("/"));
            let config = Config::load(&root).unwrap_or_default();
            let searcher = searcher.with_search_tuning(&config);
            if config.search.pagerank_boost == Some(false) {
                searcher.with_pagerank_boost(false)
            } else {
                searcher
            }
        };

        tracing::info!("searcher built and cached (LRU + connection pool will be reused)");
        let arc = Arc::new(searcher);
        *guard = Some(Arc::clone(&arc));
        Ok(arc)
    }

    /// Invalidate the cached searcher (call after indexing).
    async fn invalidate_searcher_cache(&self) {
        let mut guard = self.cached_searcher.write().await;
        *guard = None;
        tracing::info!("searcher cache invalidated");
    }

    /// Semantic + FTS hybrid search.
    ///
    /// Returns an error when the index is empty (via `prepare_search_provider`).
    #[tracing::instrument(skip_all, fields(query = %input.query, top_k = input.top_k))]
    pub async fn search_code(
        &self,
        input: SearchCodeInput,
    ) -> anyhow::Result<SearchCodeResponse> {
        let searcher = self.get_or_build_searcher().await?;
        let top_k = input.top_k.max(1);
        let max_tokens = input.max_tokens.or(Some(8192)); // agent-friendly default
        let max_depth = input.max_depth.unwrap_or(if input.include_graph { 2 } else { 0 });
        let (mut results, timings) = searcher
            .search_with_timings(&input.query, top_k, input.include_graph, max_depth, input.diversity, max_tokens)
            .await?;
        tracing::info!(
            embed_ms = timings.embed_ms,
            retrieve_ms = timings.retrieve_ms,
            expand_ms = timings.expand_ms,
            rerank_ms = timings.rerank_ms,
            graph_ms = timings.graph_ms,
            total_ms = timings.total_ms,
            results = results.len(),
            "search_code pipeline timings"
        );

        // Filter to branch-changed files if requested.
        if input.branch_scope {
            // Derive project root from manifest path: .skelesearch/manifest.db -> project_root
            let root = self.manifest_path
                .parent()
                .and_then(|p| p.parent())
                .unwrap_or_else(|| std::path::Path::new("."));
            let changed = skelesearch_core::git::changed_files_on_branch(root)?;
            if !changed.is_empty() {
                results.retain(|r| changed.iter().any(|c| r.file_path.ends_with(c.as_str()) || c.ends_with(&r.file_path)));
            }
        }

        let mut rows: Vec<SearchCodeRow> = results
            .into_iter()
            .map(|r| SearchCodeRow {
                file_path: r.file_path,
                start_line: r.start_line,
                end_line: r.end_line,
                content: r.content,
                score: r.score,
                match_quality: r.match_quality,
                why: r.why,
            })
            .collect();

        // Session dedup: deprioritize (not exclude) content seen in prior searches.
        if let Some(ref sid) = input.session_id {
            let seen = self.get_seen(sid);
            // Stable partition preserving score order within each group.
            let (mut unseen, mut already_seen): (Vec<_>, Vec<_>) = rows
                .into_iter()
                .partition(|r| !seen.contains(&Self::content_hash(&r.content)));
            unseen.append(&mut already_seen);
            // Record every chunk returned (seen and unseen alike) so they are
            // deprioritized on the next call in this session.
            let hashes: Vec<u64> = unseen.iter().map(|r| Self::content_hash(&r.content)).collect();
            self.record_seen(sid, &hashes);
            rows = unseen;
        }

        Ok(SearchCodeResponse { results: rows, _timings: timings })
    }

    /// Index the codebase at `input.path`.  Returns an error for unknown
    /// provider names before any I/O.
    ///
    /// # Send safety
    /// `ManifestStore` wraps a raw SQLite pointer and is `!Send`.  Creating an
    /// `Indexer` (which holds `Arc<ManifestStore>`) and awaiting it in a
    /// multi-threaded context violates the `Send` bound required by rmcp.
    /// We avoid this by moving the indexing work into `spawn_blocking`, where a
    /// dedicated single-thread runtime runs the `!Send` future without crossing
    /// thread boundaries.
    #[tracing::instrument(skip_all, fields(path = %input.path))]
    pub async fn index_codebase(
        &self,
        input: IndexCodebaseInput,
    ) -> anyhow::Result<IndexCodebaseOutput> {
        let provider_name = input.provider.as_deref().unwrap_or("fastembed");
        // Validate provider name before launching any I/O.
        let provider = provider_from_name(provider_name).map(ArcProvider::new)?;
        self.run_index(std::path::Path::new(&input.path), provider).await
    }

    /// Index a path using an already-constructed provider.
    ///
    /// Extracted from `index_codebase` so tests can inject a provider directly
    /// without going through the string-based factory (which requires network).
    /// Production callers always go through `index_codebase`.
    pub async fn run_index(
        &self,
        path: &std::path::Path,
        provider: ArcProvider,
    ) -> anyhow::Result<IndexCodebaseOutput> {
        let backend = Arc::clone(&self.backend);
        let manifest_path = Arc::clone(&self.manifest_path);
        let path = path.to_path_buf();
        // Clone the provider so the closure can take ownership of one copy
        // while we retain another for promotion after successful indexing.
        let provider_for_closure = provider.clone();

        // ManifestStore is !Send; run indexing in a dedicated single-thread runtime
        // inside spawn_blocking so the outer future remains Send.
        let result = tokio::task::spawn_blocking(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|e| anyhow::anyhow!("runtime build: {e}"))?;
            rt.block_on(async {
                let manifest = Arc::new(ManifestStore::open(manifest_path.as_path())?);
                let config = Config::load(&path).context("load .skelesearch.toml")?;
                let indexer = Indexer::new(backend, manifest, provider_for_closure)
                    .with_excludes(config.index.exclude.clone())
                    .with_include_extensions(config.index.include_extensions.clone());
                indexer.index_path(&path).await
            })
        })
        .await
        .context("indexer task panicked")?
        .context("indexer.index_path")?;

        // Promote the provider so subsequent searches use the same embedding
        // dimension as the newly-built index.  Only runs on success so the
        // server never holds a provider for a failed/partial index.
        *self.provider.write().map_err(|_| anyhow::anyhow!("provider lock poisoned"))? = provider;

        // Invalidate cached searcher so the next search picks up the new provider
        // and any config changes from the freshly indexed project.
        self.invalidate_searcher_cache().await;

        Ok(IndexCodebaseOutput {
            status: "ok".to_string(),
            indexed: result.indexed_files,
            chunks: result.total_chunks,
            cache_hits: result.cache_hits,
        })
    }

    /// Return current index statistics.
    pub async fn index_status(
        &self,
        _input: IndexStatusInput,
    ) -> anyhow::Result<IndexStatusOutput> {
        let stats = match self.backend.stats().await {
            Ok(s) => s,
            Err(ref e) if is_uninitialized_index_error(e) => {
                return Ok(IndexStatusOutput {
                    indexed_files: 0,
                    total_chunks: 0,
                    last_indexed: None,
                    estimated_stale: 0,
                    watching: false,
                });
            }
            Err(e) => return Err(e),
        };
        Ok(IndexStatusOutput {
            indexed_files: stats.indexed_files,
            total_chunks: stats.total_chunks,
            last_indexed: stats.last_indexed.map(|dt: chrono::DateTime<chrono::Utc>| dt.to_rfc3339()),
            // v1: no file-change detection; always 0 after a fresh index run.
            estimated_stale: 0,
            watching: false,
        })
    }

    /// Return all chunks, outbound imports, and inbound importers for a file.
    /// Returns empty arrays for unknown files — never an error.
    pub async fn get_file_context(
        &self,
        input: GetFileContextInput,
    ) -> anyhow::Result<FileContextOutput> {
        let searcher = self.get_or_build_searcher().await?;
        let ctx = match searcher.file_context(&input.file_path).await {
            Ok(c) => c,
            Err(ref e) if is_uninitialized_index_error(e) => {
                return Ok(FileContextOutput { chunks: vec![], imports: vec![], imported_by: vec![] });
            }
            Err(e) => return Err(e),
        };
        Ok(FileContextOutput {
            chunks: ctx
                .chunks
                .into_iter()
                .map(|c| ChunkInfo {
                    file_path: c.file_path,
                    chunk_idx: c.chunk_idx,
                    content: c.content,
                    chunk_type: c.chunk_type,
                    start_line: c.start_line,
                    end_line: c.end_line,
                })
                .collect(),
            imports: ctx.imports,
            imported_by: ctx.imported_by,
        })
    }

    /// Classify `input.query` with [`classify_query`] and dispatch to grep or
    /// semantic search.  Returns the chosen strategy name and serialised results.
    ///
    /// Grep path: derives the common ancestor directory from indexed file paths
    /// and calls [`grep_codebase`] with `top_k` as the result cap.
    /// Semantic path: delegates to [`Self::search_code`].
    #[tracing::instrument(skip_all, fields(query = %input.query))]
    pub async fn smart_search(
        &self,
        input: SmartSearchInput,
    ) -> anyhow::Result<SmartSearchOutput> {
        let max_tokens = input.max_tokens.or(Some(8192)); // agent-friendly default

        // Intent-based routing takes priority over auto-detection.
        if let Some(ref intent) = input.intent.clone() {
            return self.smart_search_by_intent(intent, input, max_tokens).await;
        }

        // No explicit intent — auto-detect from query content (backward compatible).
        // Prepend any supplied symbols as BM25 boost terms.
        let query = if input.symbols.is_empty() {
            input.query.clone()
        } else {
            format!("{} {}", input.symbols.join(" "), input.query)
        };

        let strategy = classify_query(&query);
        let results = match &strategy {
            QueryStrategy::Grep => {
                let paths = match self.backend.list_indexed_paths().await {
                    Ok(p) => p,
                    Err(ref e) if is_uninitialized_index_error(e) => vec![],
                    Err(e) => return Err(e),
                };
                if paths.is_empty() {
                    SmartSearchResults::Grep(vec![])
                } else {
                    // If scope is set, use it as the grep root; otherwise use common ancestor.
                    let root = if let Some(ref scope) = input.scope {
                        PathBuf::from(scope)
                    } else {
                        common_ancestor(&paths).unwrap_or_else(|| PathBuf::from("/"))
                    };
                    let opts = GrepOptions { max_results: input.top_k.max(1), case_insensitive: false };
                    let matches = grep_codebase(&root, &query, &opts)?;
                    let mut rows: Vec<GrepSearchRow> = matches
                        .into_iter()
                        .map(|m| GrepSearchRow {
                            file_path: m.file_path,
                            line_number: m.line_number,
                            line_content: m.line_content,
                        })
                        .collect();
                    // Filter grep results to branch-changed files if requested.
                    if input.branch_scope {
                        let proj_root = self.manifest_path
                            .parent()
                            .and_then(|p| p.parent())
                            .unwrap_or_else(|| std::path::Path::new("."));
                        let changed = skelesearch_core::git::changed_files_on_branch(proj_root)?;
                        if !changed.is_empty() {
                            rows.retain(|r| changed.iter().any(|c| r.file_path.ends_with(c.as_str()) || c.ends_with(&r.file_path)));
                        }
                    }
                    SmartSearchResults::Grep(rows)
                }
            }
            QueryStrategy::Semantic => {
                let response = self
                    .search_code(SearchCodeInput {
                        query: query.clone(),
                        top_k: input.top_k,
                        include_graph: input.include_graph,
                        max_depth: None,
                        diversity: input.diversity,
                        max_tokens,
                        branch_scope: input.branch_scope,
                        session_id: input.session_id.clone(),
                    })
                    .await?;
                // Apply scope filter before returning.
                let mut rows = response.results;
                if let Some(ref scope) = input.scope {
                    rows.retain(|r| std::path::Path::new(&r.file_path).starts_with(scope.as_str()));
                }
                SmartSearchResults::Semantic(rows)
            }
        };
        Ok(SmartSearchOutput { strategy: strategy.to_string(), results })
    }

    /// Intent-based dispatch for explicit `intent` values.
    async fn smart_search_by_intent(
        &self,
        intent: &str,
        input: SmartSearchInput,
        max_tokens: Option<usize>,
    ) -> anyhow::Result<SmartSearchOutput> {
        match intent {
            "find" | "understand" => {
                let include_graph = intent == "understand" || input.include_graph;
                // "understand" uses depth 2; "find" uses no graph expansion by default.
                let max_depth = if intent == "understand" {
                    Some(2usize)
                } else {
                    None
                };
                // Prepend symbols as BM25 boost terms when supplied.
                let query = if input.symbols.is_empty() {
                    input.query.clone()
                } else {
                    format!("{} {}", input.symbols.join(" "), input.query)
                };
                let response = self
                    .search_code(SearchCodeInput {
                        query,
                        top_k: input.top_k,
                        include_graph,
                        max_depth,
                        diversity: input.diversity,
                        max_tokens,
                        branch_scope: input.branch_scope,
                        session_id: input.session_id.clone(),
                    })
                    .await?;
                let mut rows = response.results;
                if let Some(ref scope) = input.scope {
                    rows.retain(|r| std::path::Path::new(&r.file_path).starts_with(scope.as_str()));
                }
                Ok(SmartSearchOutput {
                    strategy: intent.to_string(),
                    results: SmartSearchResults::Semantic(rows),
                })
            }
            "impact" => {
                if input.symbols.is_empty() {
                    return Err(anyhow::anyhow!(
                        "impact intent requires at least one symbol in the 'symbols' field"
                    ));
                }
                let file_path = input.symbols.first().cloned().unwrap();
                let impact = self
                    .find_impact_set(FindImpactSetInput {
                        file_path,
                        max_depth: None, // use find_impact_set default (3, capped at 5)
                    })
                    .await?;
                Ok(SmartSearchOutput {
                    strategy: "impact".to_string(),
                    results: SmartSearchResults::Impact(impact),
                })
            }
            "trace" => {
                Err(anyhow::anyhow!("trace intent not yet implemented"))
            }
            other => {
                Err(anyhow::anyhow!(
                    "unknown intent {:?}; valid values are: find, understand, impact, trace",
                    other
                ))
            }
        }
    }


    /// Find symbol definitions by name, optionally filtered by kind.
    pub async fn find_symbol(
        &self,
        input: FindSymbolInput,
    ) -> anyhow::Result<Vec<SymbolRow>> {
        let results = match self.backend.find_symbols(&input.name, input.kind.as_deref()).await {
            Ok(r) => r,
            Err(ref e) if is_uninitialized_index_error(e) => return Ok(vec![]),
            Err(e) => return Err(e),
        };
        Ok(results
            .into_iter()
            .map(|s| SymbolRow {
                file_path: s.file_path,
                name: s.name,
                kind: s.kind,
                start_line: s.start_line,
                end_line: s.end_line,
            })
            .collect())
    }

    pub async fn find_impact_set(
        &self,
        input: FindImpactSetInput,
    ) -> anyhow::Result<ImpactSetOutput> {
        let max_depth = input.max_depth.unwrap_or(3).min(5);
        let all_importers = match self.backend.traverse_importers(&input.file_path, max_depth, None).await {
            Ok(v) => v,
            Err(ref e) if is_uninitialized_index_error(e) => vec![],
            Err(e) => return Err(e),
        };

        let direct: Vec<String> = all_importers.iter()
            .filter(|(_, d)| *d == 1)
            .map(|(f, _)| f.clone())
            .collect();

        let transitive: Vec<ImpactEntry> = all_importers.iter()
            .filter(|(_, d)| *d > 1)
            .map(|(f, d)| ImpactEntry { file_path: f.clone(), depth: *d })
            .collect();

        let tests: Vec<String> = all_importers.iter()
            .filter(|(f, _)| is_test_file_path(f))
            .map(|(f, _)| f.clone())
            .collect();

        Ok(ImpactSetOutput {
            file_path: input.file_path,
            direct_importers: direct,
            transitive_importers: transitive,
            affected_tests: tests,
        })
    }

    pub async fn find_test_context(
        &self,
        input: FindTestContextInput,
    ) -> anyhow::Result<TestContextOutput> {
        let importers = match self.backend.get_importers(&input.file_path).await {
            Ok(v) => v,
            Err(ref e) if is_uninitialized_index_error(e) => vec![],
            Err(e) => return Err(e),
        };
        let test_importers: Vec<String> = importers.into_iter()
            .filter(|f| is_test_file_path(f))
            .collect();

        let dir = std::path::Path::new(&input.file_path)
            .parent()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_default();
        let stem = std::path::Path::new(&input.file_path)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();

        let all_files = match self.backend.list_indexed_paths().await {
            Ok(v) => v,
            Err(ref e) if is_uninitialized_index_error(e) => vec![],
            Err(e) => return Err(e),
        };
        let colocated: Vec<String> = all_files.into_iter()
            .filter(|f| {
                is_test_file_path(f)
                    && (std::path::Path::new(f).starts_with(&dir)
                        || f.contains(&format!("/tests/{}", stem))
                        || f.contains(&format!("/__tests__/{}", stem)))
            })
            .filter(|f| !test_importers.contains(f))
            .collect();

        Ok(TestContextOutput {
            file_path: input.file_path,
            test_files: test_importers,
            colocated_tests: colocated,
        })
    }

    // -----------------------------------------------------------------------
    // MCP transport entry points
    // -----------------------------------------------------------------------

    /// Start the server on stdin/stdout and block until the connection closes.
    pub async fn serve_stdio(self) -> anyhow::Result<()> {
        let service = self
            .serve(rmcp::transport::stdio())
            .await
            .context("MCP server initialization failed")?;
        service.waiting().await?;
        Ok(())
    }

    /// Start the server over Streamable HTTP on the given address.
    ///
    /// The MCP Streamable HTTP transport uses POST for requests and SSE
    /// for streaming responses.  This is the preferred transport for
    /// non-subprocess consumers (VS Code extensions, remote agents, HTTP
    /// clients).
    pub async fn serve_http(self, addr: std::net::SocketAddr) -> anyhow::Result<()> {
        use rmcp::transport::streamable_http_server::{
            StreamableHttpServerConfig, StreamableHttpService,
            session::local::LocalSessionManager,
        };

        let config = StreamableHttpServerConfig::default();
        // The factory is called once per session (stateless mode, the default).
        // All fields behind Arc so cloning is cheap.
        let service = StreamableHttpService::<SkeleSearchServer, LocalSessionManager>::new(
            move || Ok(self.clone()),
            Default::default(),
            config,
        );

        let router = axum::Router::new().nest_service("/mcp", service);
        let listener = tokio::net::TcpListener::bind(addr).await
            .with_context(|| format!("bind to {addr}"))?;
        tracing::info!("skelesearch-mcp HTTP listening on {addr}");

        axum::serve(listener, router).await
            .context("HTTP server error")?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// rmcp tool declarations
// ---------------------------------------------------------------------------

#[tool_router]
impl SkeleSearchServer {
    /// Hybrid semantic + keyword code search. Returns ranked code blocks.
    #[tool(name = "search_code")]
    async fn mcp_search_code(
        &self,
        Parameters(input): Parameters<SearchCodeInput>,
    ) -> Result<String, String> {
        self.search_code(input)
            .await
            .map_err(|e| e.to_string())
            .and_then(|response| serde_json::to_string(&response).map_err(|e| e.to_string()))
    }

    /// Index a directory for code search. Run once, updates incrementally.
    #[tool(name = "index_codebase")]
    async fn mcp_index_codebase(
        &self,
        Parameters(input): Parameters<IndexCodebaseInput>,
    ) -> Result<String, String> {
        self.index_codebase(input)
            .await
            .map_err(|e| e.to_string())
            .and_then(|out| serde_json::to_string(&out).map_err(|e| e.to_string()))
    }

    /// Check if the code index exists and is current.
    #[tool(name = "index_status")]
    async fn mcp_index_status(
        &self,
        Parameters(input): Parameters<IndexStatusInput>,
    ) -> Result<String, String> {
        self.index_status(input)
            .await
            .map_err(|e| e.to_string())
            .and_then(|out| serde_json::to_string(&out).map_err(|e| e.to_string()))
    }

    /// Get all indexed chunks and import graph for a specific file.
    #[tool(name = "get_file_context")]
    async fn mcp_get_file_context(
        &self,
        Parameters(input): Parameters<GetFileContextInput>,
    ) -> Result<String, String> {
        self.get_file_context(input)
            .await
            .map_err(|e| e.to_string())
            .and_then(|out| serde_json::to_string(&out).map_err(|e| e.to_string()))
    }

    /// Find code by concept or keyword. Auto-routes to best search strategy.
    #[tool(name = "smart_search")]
    async fn mcp_smart_search(
        &self,
        Parameters(input): Parameters<SmartSearchInput>,
    ) -> Result<String, String> {
        self.smart_search(input)
            .await
            .map_err(|e| e.to_string())
            .and_then(|out| serde_json::to_string(&out).map_err(|e| e.to_string()))
    }

    /// Look up a symbol definition by exact name.
    #[tool(name = "find_symbol")]
    async fn mcp_find_symbol(
        &self,
        Parameters(input): Parameters<FindSymbolInput>,
    ) -> Result<String, String> {
        self.find_symbol(input)
            .await
            .map_err(|e| e.to_string())
            .and_then(|rows| serde_json::to_string(&rows).map_err(|e| e.to_string()))
    }

    /// Find all files affected by changes to a given file. Returns direct importers,
    /// transitive importers by depth, and affected test files.
    #[tool(name = "find_impact_set")]
    async fn mcp_find_impact_set(
        &self,
        Parameters(input): Parameters<FindImpactSetInput>,
    ) -> Result<String, String> {
        self.find_impact_set(input)
            .await
            .map_err(|e| e.to_string())
            .and_then(|r| serde_json::to_string(&r).map_err(|e| e.to_string()))
    }

    /// Find test files covering a source file. Returns test files that import it
    /// and colocated test files.
    #[tool(name = "find_test_context")]
    async fn mcp_find_test_context(
        &self,
        Parameters(input): Parameters<FindTestContextInput>,
    ) -> Result<String, String> {
        self.find_test_context(input)
            .await
            .map_err(|e| e.to_string())
            .and_then(|r| serde_json::to_string(&r).map_err(|e| e.to_string()))
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for SkeleSearchServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some(
                "skelesearch -- semantic code search for agents.\n\n\
                 Tools:\n\
                 - smart_search: Find code by concept, keyword, or symbol. Auto-routes between grep and semantic search.\n\
                 - search_code: Direct hybrid semantic + keyword search with full control over parameters.\n\
                 - find_symbol: Exact symbol name lookup.\n\
                 - find_impact_set: Find all files that depend on a given file (reverse import graph).\n\
                 - find_test_context: Find test files for a source file.\n\n\
                 Query tips for best results:\n\
                 - Describe what the target code DOES, not a question: \"middleware that validates JWT tokens\" not \"how does auth work\"\n\
                 - Include known symbol names: \"AsyncClient connection pooling retry logic\"\n\
                 - Use `intent: \"understand\"` when you need a symbol plus its structural context\n\
                 - Use `intent: \"impact\"` with `symbols: [\"SymbolName\"]` to find all dependents before refactoring\n\
                 - Set `scope: \"src/auth\"` to narrow results to a directory\n\
                 - Set `max_tokens` to control output size (default: 8192)\n\
                 - Set `session_id` to deduplicate across multi-turn searches".into(),
            ),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation {
                name: "skelesearch".into(),
                version: env!("CARGO_PKG_VERSION").into(),
                ..Default::default()
            },
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Path utilities
// ---------------------------------------------------------------------------

/// Returns true when `path` looks like a test file based on common naming conventions.
/// This is a heuristic — it covers Go, Rust, JS/TS, Ruby, and directory-based conventions.
fn is_test_file_path(path: &str) -> bool {
    let lower = path.to_lowercase();
    lower.contains("/test/")
        || lower.contains("/tests/")
        || lower.contains("/__tests__/")
        || lower.contains("/spec/")
        || lower.contains("/specs/")
        || lower.ends_with("_test.go")
        || lower.ends_with(".test.ts")
        || lower.ends_with(".test.js")
        || lower.ends_with(".test.tsx")
        || lower.ends_with(".test.jsx")
        || lower.ends_with(".spec.ts")
        || lower.ends_with(".spec.js")
        || lower.ends_with(".spec.tsx")
        || lower.ends_with(".spec.jsx")
        || lower.ends_with("_test.rs")
        || lower.ends_with("_spec.rb")
}


/// Return the deepest common ancestor directory for a slice of absolute file paths.
///
/// Walks upward from the first path's parent, popping components until every
/// remaining path starts with the candidate.  Returns `None` when `paths` is
/// empty or when the candidates collapse to `/` with no common prefix.
fn common_ancestor(paths: &[String]) -> Option<PathBuf> {
    let first = PathBuf::from(paths.first()?);	
    let mut common = first.parent()?.to_path_buf();
    for p in &paths[1..] {
        let path = PathBuf::from(p);
        // Walk upward until `common` is a prefix of `path`.
        while !path.starts_with(&common) {
            if !common.pop() {
                return None;
            }
        }
    }
    Some(common)
}
