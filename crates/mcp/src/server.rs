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

use std::{path::PathBuf, sync::Arc};

use anyhow::Context as _;
use async_trait::async_trait;
use rmcp::{
    ServerHandler, ServiceExt,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{ServerCapabilities, ServerInfo},
    tool, tool_handler, tool_router,
};
use skelesearch_core::{CozoBackend, EmbedProvider, Indexer, ManifestStore, Searcher, StorageBackend};
use skelesearch_embed_fastembed::FastEmbedProvider;

use crate::tools::{
    ChunkInfo, FileContextOutput, GetFileContextInput, IndexCodebaseInput, IndexCodebaseOutput,
    IndexStatusInput, IndexStatusOutput, SearchCodeInput, SearchCodeRow,
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
    provider: ArcProvider,
    tool_router: ToolRouter<Self>,
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
            provider: ArcProvider::new(provider),
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

    /// Semantic + FTS hybrid search.  Returns an empty vec when the index is
    /// empty — never errors on zero results.
    pub async fn search_code(
        &self,
        input: SearchCodeInput,
    ) -> anyhow::Result<Vec<SearchCodeRow>> {
        let searcher = Searcher::new(Arc::clone(&self.backend), self.provider.clone());
        let top_k = input.top_k.max(1);
        let results = searcher
            .search(&input.query, top_k, input.include_graph)
            .await?;
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
    pub async fn index_codebase(
        &self,
        input: IndexCodebaseInput,
    ) -> anyhow::Result<IndexCodebaseOutput> {
        let provider_name = input.provider.as_deref().unwrap_or("fastembed");
        // Validate provider name before launching any I/O.
        let provider = make_provider(provider_name)?;
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

        // ManifestStore is !Send; run indexing in a dedicated single-thread runtime
        // inside spawn_blocking so the outer future remains Send.
        let result = tokio::task::spawn_blocking(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|e| anyhow::anyhow!("runtime build: {e}"))?;
            rt.block_on(async {
                let manifest = Arc::new(ManifestStore::open(manifest_path.as_path())?);
                let indexer = Indexer::new(backend, manifest, provider);
                indexer.index_path(&path).await
            })
        })
        .await
        .context("indexer task panicked")?
        .context("indexer.index_path")?;

        Ok(IndexCodebaseOutput {
            status: "ok".to_string(),
            indexed: result.indexed_files,
            chunks: result.total_chunks,
        })
    }

    /// Return current index statistics.
    pub async fn index_status(
        &self,
        _input: IndexStatusInput,
    ) -> anyhow::Result<IndexStatusOutput> {
        let stats = self.backend.stats().await?;
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
        let searcher = Searcher::new(Arc::clone(&self.backend), self.provider.clone());
        let ctx = searcher.file_context(&input.file_path).await?;
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

    // -----------------------------------------------------------------------
    // MCP stdio server entry point
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
}

// ---------------------------------------------------------------------------
// rmcp tool declarations
// ---------------------------------------------------------------------------

#[tool_router]
impl SkeleSearchServer {
    /// Semantic and full-text hybrid search over the indexed codebase.
    #[tool(name = "search_code")]
    async fn mcp_search_code(
        &self,
        Parameters(input): Parameters<SearchCodeInput>,
    ) -> Result<String, String> {
        self.search_code(input)
            .await
            .map(|rows| serde_json::to_string(&rows).unwrap_or_default())
            .map_err(|e| e.to_string())
    }

    /// Index a codebase directory and make it searchable.
    #[tool(name = "index_codebase")]
    async fn mcp_index_codebase(
        &self,
        Parameters(input): Parameters<IndexCodebaseInput>,
    ) -> Result<String, String> {
        self.index_codebase(input)
            .await
            .map(|out| serde_json::to_string(&out).unwrap_or_default())
            .map_err(|e| e.to_string())
    }

    /// Return current index statistics including file count, chunk count,
    /// last-indexed timestamp, and whether a watch process is running.
    #[tool(name = "index_status")]
    async fn mcp_index_status(
        &self,
        Parameters(input): Parameters<IndexStatusInput>,
    ) -> Result<String, String> {
        self.index_status(input)
            .await
            .map(|out| serde_json::to_string(&out).unwrap_or_default())
            .map_err(|e| e.to_string())
    }

    /// Return all indexed chunks, imports, and importers for a specific file.
    #[tool(name = "get_file_context")]
    async fn mcp_get_file_context(
        &self,
        Parameters(input): Parameters<GetFileContextInput>,
    ) -> Result<String, String> {
        self.get_file_context(input)
            .await
            .map(|out| serde_json::to_string(&out).unwrap_or_default())
            .map_err(|e| e.to_string())
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
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Provider factory
// ---------------------------------------------------------------------------

/// Create an embedding provider by name.  Returns a clear error for unknown
/// names so callers learn exactly what went wrong.
fn make_provider(name: &str) -> anyhow::Result<ArcProvider> {
    match name {
        "fastembed" => {
            let p = FastEmbedProvider::default()
                .context("failed to initialise FastEmbed provider")?;
            Ok(ArcProvider::new(p))
        }
        other => anyhow::bail!(
            "unknown provider '{}'; supported providers: fastembed",
            other
        ),
    }
}
