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

use std::{path::PathBuf, sync::{Arc, RwLock}};

use anyhow::Context as _;
use async_trait::async_trait;
use rmcp::{
    ServerHandler, ServiceExt,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{Implementation, ServerCapabilities, ServerInfo},
    tool, tool_handler, tool_router,
};
use skelesearch_core::{classify_query, grep_codebase, CozoBackend, Config, EmbedProvider, GrepOptions, Indexer, ManifestStore, QueryStrategy, Searcher, StorageBackend};
use skelesearch_embed_fastembed::{FastEmbedProvider, provider_from_name};

use crate::tools::{
    ChunkInfo, FileContextOutput, FindSymbolInput, GetFileContextInput, GrepSearchRow,
    IndexCodebaseInput, IndexCodebaseOutput, IndexStatusInput, IndexStatusOutput,
    SearchCodeInput, SearchCodeRow, SmartSearchInput, SmartSearchOutput, SmartSearchResults,
    SymbolRow,
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

    async fn embed_batch(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
        self.0.embed_batch(texts).await
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
#[derive(Clone)]
pub struct SkeleSearchServer {
    backend: Arc<CozoBackend>,
    /// Path to the manifest SQLite database; opened fresh per index_codebase call.
    manifest_path: Arc<PathBuf>,
    /// Provider used for query embedding in `search_code`.
    /// Provider used for query embedding in `search_code`.  Wrapped in an
    /// RwLock so `run_index` can promote it to the real provider after a
    /// successful indexing run without requiring a full server restart.
    provider: Arc<RwLock<ArcProvider>>,
    tool_router: ToolRouter<Self>,
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
    /// upgrades to `FastEmbedProvider` and promotes it for future calls.
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
        let real = FastEmbedProvider::default().context("failed to initialize fastembed provider")?;
        let arc_provider = ArcProvider::new(real);
        *self.provider.write().map_err(|_| anyhow::anyhow!("provider lock poisoned"))? = arc_provider.clone();
        Ok(arc_provider)
    }

    /// Semantic + FTS hybrid search.
    ///
    /// Returns an error when the index is empty (via `prepare_search_provider`).
    #[tracing::instrument(skip_all, fields(query = %input.query, top_k = input.top_k))]
    pub async fn search_code(
        &self,
        input: SearchCodeInput,
    ) -> anyhow::Result<Vec<SearchCodeRow>> {
        let provider = self.prepare_search_provider().await?;
        let searcher = Searcher::new(Arc::clone(&self.backend), provider);
        let top_k = input.top_k.max(1);
        let max_depth = input.max_depth.unwrap_or(if input.include_graph { 2 } else { 0 });
        let start = std::time::Instant::now();
        let results = searcher
            .search(&input.query, top_k, input.include_graph, max_depth, input.diversity)
            .await?;
        tracing::info!(elapsed_ms = start.elapsed().as_millis() as u64, results = results.len(), "search_code complete");
        Ok(results
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
            .collect())
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
                    .with_excludes(config.index.exclude.clone());
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
        let provider = self.provider.read().map_err(|_| anyhow::anyhow!("provider lock poisoned"))?.clone();
        let searcher = Searcher::new(Arc::clone(&self.backend), provider);
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
        let strategy = classify_query(&input.query);
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
                    let root = common_ancestor(&paths).unwrap_or_else(|| PathBuf::from("/"));
                    let opts = GrepOptions { max_results: input.top_k.max(1), case_insensitive: false };
                    let matches = grep_codebase(&root, &input.query, &opts)?;
                    SmartSearchResults::Grep(
                        matches
                            .into_iter()
                            .map(|m| GrepSearchRow {
                                file_path: m.file_path,
                                line_number: m.line_number,
                                line_content: m.line_content,
                            })
                            .collect(),
                    )
                }
            }
            QueryStrategy::Semantic => {
                let rows = self
                    .search_code(SearchCodeInput {
                        query: input.query.clone(),
                        top_k: input.top_k,
                        include_graph: input.include_graph,
                        max_depth: None,
                        diversity: input.diversity,
                    })
                    .await?;
                SmartSearchResults::Semantic(rows)
            }
        };
        Ok(SmartSearchOutput { strategy: strategy.to_string(), results })
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
    /// Semantic and full-text hybrid search over the indexed codebase.
    ///
    /// Requires the codebase to be indexed first via `index_codebase`.
    /// Results are candidate chunks ranked by relevance — not guaranteed to be
    /// the exact match, but the closest the index can find.
    ///
    /// `include_graph` is accepted but graph augmentation is disabled in v1.2;
    /// set it to `false` for now.
    /// Returns an error when the index is empty.
    #[tool(name = "search_code")]
    async fn mcp_search_code(
        &self,
        Parameters(input): Parameters<SearchCodeInput>,
    ) -> Result<String, String> {
        self.search_code(input)
            .await
            .map_err(|e| e.to_string())
            .and_then(|rows| serde_json::to_string(&rows).map_err(|e| e.to_string()))
    }

    /// Index a codebase directory at `path` using the chosen embedding provider.
    ///
    /// Walks all source files under `path`, splits them into chunks, embeds each
    /// chunk, and stores the results in the local index.  The only supported
    /// provider in v1 is `"fastembed"` (default; runs locally, no API key needed).
    /// Re-run after large code changes — the index is not updated automatically
    /// unless the `watch` command is running.
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

    /// Report whether an index exists and provide basic counts.
    ///
    /// Returns `indexed_files`, `total_chunks`, an RFC 3339 `last_indexed`
    /// timestamp (or `null` if never indexed), and `estimated_stale` (v1: always 0).
    /// Call this before `search_code` to confirm the index is populated.
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

    /// Return indexed chunks, import metadata, and reverse-import metadata for a specific file.
    ///
    /// Returns the raw stored chunks for `file_path` plus the list of files it
    /// imports (`imports`) and the list of files that import it (`imported_by`).
    /// Returns empty lists when the file is not in the index rather than an error.
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

    /// Auto-route the query to grep or semantic search based on its shape.
    ///
    /// Keyword-shaped or pattern queries (identifiers, regex-like strings) are
    /// dispatched to grep for exact matches.  Natural-language queries are sent
    /// to the semantic search path (equivalent to `search_code`).  The response
    /// includes a `strategy` field (`"grep"` or `"semantic"`) so callers can
    /// see which path was taken.
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

    /// Exact-name symbol lookup with optional kind filter.
    ///
    /// Searches the symbol table for definitions whose name matches `name`
    /// exactly (case-sensitive).  Supply `kind` (e.g. `"function"`, `"struct"`,
    /// `"class"`) to narrow results to a specific symbol kind.
    /// Returns file path, start/end lines, and kind for each match.
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
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for SkeleSearchServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some(
                "skelesearch — semantic code search. \
                 Use index_codebase first, then search_code to find relevant code."
                    .into(),
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
