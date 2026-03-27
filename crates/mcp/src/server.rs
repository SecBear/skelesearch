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

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, RwLock,
    },
    time::Duration,
};

#[path = "client.rs"]
mod client;

use notify::Watcher as _;

use anyhow::Context as _;
use async_trait::async_trait;
use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{Implementation, ServerCapabilities, ServerInfo},
    service::{NotificationContext, RoleServer},
    tool, tool_handler, tool_router, ServerHandler, ServiceExt,
};
use skelesearch_core::{
    classify_query, generation_db_paths, grep_codebase, is_indexing_active_elsewhere,
    read_shared_indexing_status, try_acquire_indexing_lease, Config, CozoBackend, EmbedProvider,
    FreshnessSnapshot, FreshnessState, GrepOptions, IndexResult, Indexer, LLMExpander,
    ManifestStore, QueryExpander, QueryStrategy, Reranker, Searcher, SharedIndexingStatus,
    StorageBackend, INDEX_DB_FILE, MANIFEST_DB_FILE,
};
use skelesearch_embed_fastembed::{provider_from_name, FastEmbedSparseProvider};
use skelesearch_service::{
    protocol::{
        IndexCodebaseStatus as DaemonIndexCodebaseStatus,
        IndexFreshnessState as DaemonIndexFreshnessState, IndexingState as DaemonIndexingState,
    },
    ProjectTarget as DaemonProjectTarget,
};
use sysinfo::{Pid, System};

use self::client::{DaemonClient, TokioDaemonConnector};
use crate::tools::{
    CallEdgeInfo, ChunkInfo, FileContextOutput, FindImpactSetInput, FindSymbolInput,
    FindTestContextInput, GetFileContextInput, GetRepoMapInput, GetSymbolContextInput,
    GrepSearchRow, ImpactEntry, ImpactSetOutput, IndexCodebaseInput, IndexCodebaseOutput,
    IndexFreshnessState, IndexStatusInput, IndexStatusOutput, IndexingProgress, SearchCodeInput,
    SearchCodeResponse, SearchCodeRow, SmartSearchInput, SmartSearchOutput, SmartSearchResults,
    SymbolContextOutput, SymbolRow, TestContextOutput,
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
// Background indexing state
// ---------------------------------------------------------------------------

/// Status of a background `index_codebase` operation.
#[derive(Debug, Clone, PartialEq)]
pub enum IndexingStatus {
    Idle,
    Running,
    Done,
    Failed,
}

/// Mutable state for the background indexing task, protected by an async `RwLock`.
/// All fields are set atomically under the write lock; readers snapshot under the read lock.
#[derive(Debug, Clone)]
pub struct IndexProgress {
    pub status: IndexingStatus,
    /// Absolute path being indexed.
    pub path: String,
    /// Storage dir backing this indexing run, used to scope local state to one project.
    pub storage_dir: String,
    /// Rough file count captured before spawning (0 if quick-count timed out).
    pub files_found: usize,
    /// Files indexed on completion (0 while running).
    pub files_done: usize,
    /// Chunks written on completion (0 while running).
    pub chunks_done: usize,
    /// Embedding cache hits on completion (0 while running).
    pub cache_hits: usize,
    /// Error string if `status == Failed`.
    pub error: Option<String>,
    /// Wall-clock start so callers can compute elapsed seconds.
    pub started_at: std::time::Instant,
}

impl Default for IndexProgress {
    fn default() -> Self {
        Self {
            status: IndexingStatus::Idle,
            path: String::new(),
            storage_dir: String::new(),
            files_found: 0,
            files_done: 0,
            chunks_done: 0,
            cache_hits: 0,
            error: None,
            // Instant::now() is harmless for Idle; only read when status != Idle.
            started_at: std::time::Instant::now(),
        }
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
    daemon_client: Arc<DaemonClient<TokioDaemonConnector>>,
    tool_router: ToolRouter<Self>,
    /// Cached searcher — built once on first search, invalidated after indexing.
    /// Keeps the LRU query-embedding cache and TCP connection pool alive across
    /// MCP calls, eliminating cold TLS handshakes and redundant embed API calls.
    cached_searcher: Arc<tokio::sync::RwLock<Option<Arc<CachedSearcher>>>>,
    cached_searcher_manifest_path: Arc<tokio::sync::RwLock<Option<PathBuf>>>,
    /// Shared state for background indexing.  Written by the spawned task,
    /// read by `index_status` and `index_codebase` (duplicate-check).
    index_state: Arc<tokio::sync::RwLock<IndexProgress>>,
    /// Cache of opened backends for non-cwd projects. Keyed by project root.
    /// The default backend (self.backend) handles the cwd project; this cache
    /// serves tools that specify an explicit `path` to a different project.
    backend_cache: Arc<tokio::sync::RwLock<HashMap<PathBuf, (Arc<CozoBackend>, PathBuf)>>>,
    /// Guards against starting more than one file watcher per server lifetime.
    /// `on_initialized` is called on every client reconnect; this AtomicBool
    /// ensures the background watcher task is spawned only once.
    watcher_started: Arc<AtomicBool>,
    /// Stable per-process identifier for correlating lifecycle logs.
    instance_id: Arc<str>,
    /// Default project root for pathless operations. None means inert/non-project mode.
    default_project_root: Option<PathBuf>,
    /// When true, skip auto-index, watcher startup, and health checks until an explicit project path is supplied.
    inert_mode: bool,
    /// Shared daemon-session lease for managed helper lifetime.
    daemon_session_control: Arc<DaemonSessionControl>,
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

/// Translate a backend error into a user-actionable message for MCP callers.
///
/// CozoDB schema errors ("stored relation not found", "arity mismatch") are common
/// sources of confusing output — translate them to instructions the user can act on.
fn friendly_index_error(err: &anyhow::Error) -> String {
    friendly_index_error_inner(err, false)
}

fn friendly_index_error_inner(err: &anyhow::Error, indexing_active: bool) -> String {
    let msg = err.to_string();
    if msg.contains("stored relation") && msg.contains("not found") {
        if indexing_active {
            "Index is being built. Poll index_status to check progress; search will work once indexing completes.".to_string()
        } else {
            "Index not initialized. Run index or set VOYAGE_API_KEY for auto-indexing.".to_string()
        }
    } else if msg.contains("arity mismatch") || msg.contains("Arity mismatch") {
        "Index schema is outdated. Delete .skelesearch/ directory and re-index.".to_string()
    } else {
        msg
    }
}

fn new_instance_id() -> Arc<str> {
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    format!("{pid:x}-{nanos:x}").into()
}

fn sample_process_resources() -> Option<(u64, u64, usize)> {
    let pid = Pid::from_u32(std::process::id());
    let mut sys = System::new();
    sys.refresh_processes();
    let process = sys.process(pid)?;
    let rss_bytes = process.memory().saturating_mul(1024);
    let virtual_bytes = process.virtual_memory().saturating_mul(1024);
    let task_count = process.tasks().map(|t| t.len()).unwrap_or(0);
    Some((rss_bytes, virtual_bytes, task_count))
}

#[derive(Default)]
struct DaemonSessionControl {
    started: AtomicBool,
    shutdown_tx: std::sync::Mutex<Option<tokio::sync::watch::Sender<bool>>>,
}

impl Drop for DaemonSessionControl {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.shutdown_tx.lock() {
            if let Some(tx) = guard.take() {
                let _ = tx.send(true);
            }
        }
    }
}

fn spawn_index_resource_sampler(
    instance_id: Arc<str>,
    index_state: Arc<tokio::sync::RwLock<IndexProgress>>,
    path: String,
    trigger: &'static str,
    provider: String,
) {
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_secs(10)).await;
            let snapshot = {
                let state = index_state.read().await;
                if state.status != IndexingStatus::Running {
                    break;
                }
                (
                    state.started_at.elapsed().as_secs(),
                    state.files_done,
                    state.files_found,
                    state.chunks_done,
                    state.cache_hits,
                )
            };
            if let Some((rss_bytes, virtual_bytes, task_count)) = sample_process_resources() {
                tracing::info!(
                    instance_id = %instance_id,
                    pid = std::process::id(),
                    path = %path,
                    trigger,
                    provider = %provider,
                    elapsed_s = snapshot.0,
                    files_done = snapshot.1,
                    files_found = snapshot.2,
                    chunks_done = snapshot.3,
                    cache_hits = snapshot.4,
                    rss_bytes,
                    virtual_bytes,
                    task_count,
                    "index resource sample"
                );
            }
        }
    });
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
        let manifest_path = Arc::new(manifest_path.into());
        let instance_id = new_instance_id();
        let daemon_client = Arc::new(DaemonClient::from_env());
        let default_project_root =
            Self::project_root_from_manifest_path(manifest_path.as_ref()).ok();
        tracing::info!(
            instance_id = %instance_id,
            pid = std::process::id(),
            provider = provider.name(),
            provider_dim = provider.dim(),
            manifest_path = %manifest_path.display(),
            daemon_endpoint = %daemon_client.endpoint(),
            "constructed skelesearch MCP server"
        );
        Self {
            backend,
            manifest_path,
            provider: Arc::new(RwLock::new(ArcProvider::new(provider))),
            daemon_client,
            tool_router: Self::tool_router(),
            cached_searcher: Arc::new(tokio::sync::RwLock::new(None)),
            cached_searcher_manifest_path: Arc::new(tokio::sync::RwLock::new(None)),
            index_state: Arc::new(tokio::sync::RwLock::new(IndexProgress::default())),
            backend_cache: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            watcher_started: Arc::new(AtomicBool::new(false)),
            instance_id,
            default_project_root,
            inert_mode: false,
            daemon_session_control: Arc::new(DaemonSessionControl::default()),
        }
    }

    pub fn with_default_project_root(mut self, root: Option<PathBuf>) -> Self {
        self.inert_mode = root.is_none();
        self.default_project_root = root;
        self
    }

    /// Map an error to a friendly string, noting if indexing is in progress.
    async fn friendly_err(&self, err: anyhow::Error) -> String {
        let active = self.index_state.read().await.status == IndexingStatus::Running;
        friendly_index_error_inner(&err, active)
    }

    fn daemon_proxy_error(&self, method: &str, err: anyhow::Error) -> String {
        format!(
            "daemon proxy for {method} is unavailable at {}: {err:#}. Start skelesearchd, or set SKELESEARCH_DAEMON_SOCKET to a reachable socket.",
            self.daemon_client.endpoint()
        )
    }

    async fn ensure_daemon_session(&self) {
        if self
            .daemon_session_control
            .started
            .swap(true, Ordering::SeqCst)
        {
            return;
        }

        let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
        if let Ok(mut guard) = self.daemon_session_control.shutdown_tx.lock() {
            *guard = Some(shutdown_tx);
        }

        let daemon_client = Arc::clone(&self.daemon_client);
        let instance_id = Arc::clone(&self.instance_id);
        tokio::spawn(async move {
            maintain_daemon_session(daemon_client, shutdown_rx, instance_id).await;
        });
    }

    fn daemon_target_from_path(path: &str) -> anyhow::Result<DaemonProjectTarget> {
        let target = PathBuf::from(path);
        let abs = if target.is_absolute() {
            target
        } else {
            std::env::current_dir()
                .context("resolve current working directory for daemon project target")?
                .join(target)
        };
        let original = abs.clone();
        let mut dir = if abs.is_dir() {
            abs.clone()
        } else {
            abs.parent().unwrap_or(&abs).to_path_buf()
        };
        let root = loop {
            if dir.join(".git").exists() {
                break dir;
            }
            match dir.parent() {
                Some(parent) => dir = parent.to_path_buf(),
                None => break original,
            }
        };
        Ok(DaemonProjectTarget::RootPath {
            root_path: root.to_string_lossy().into_owned(),
            logical_id: None,
        })
    }

    fn daemon_target_for_status(&self, path: Option<&str>) -> anyhow::Result<DaemonProjectTarget> {
        if let Some(path) = path {
            return Self::daemon_target_from_path(path);
        }

        if let Some(root) = &self.default_project_root {
            return Ok(DaemonProjectTarget::RootPath {
                root_path: root.to_string_lossy().into_owned(),
                logical_id: None,
            });
        }

        anyhow::bail!(
            "no default project root for this MCP instance; pass an explicit project path"
        )
    }

    fn map_daemon_indexing_progress(
        progress: skelesearch_service::IndexingProgress,
    ) -> IndexingProgress {
        IndexingProgress {
            status: match progress.status {
                DaemonIndexingState::Running => "running".to_string(),
                DaemonIndexingState::Done => "done".to_string(),
                DaemonIndexingState::Failed => "failed".to_string(),
            },
            path: progress.path,
            files_done: progress.files_done,
            files_total: progress.files_total,
            chunks_done: progress.chunks_done,
            cache_hits: progress.cache_hits,
            elapsed_seconds: progress.elapsed_seconds,
            error: progress.error,
        }
    }

    async fn proxy_index_codebase_via_daemon(
        &self,
        input: IndexCodebaseInput,
    ) -> anyhow::Result<IndexCodebaseOutput> {
        let target = Self::daemon_target_from_path(&input.path)?;
        let response = self
            .daemon_client
            .index_codebase(target, input.provider)
            .await?;
        self.invalidate_searcher_cache().await;

        Ok(IndexCodebaseOutput {
            status: match response.status {
                DaemonIndexCodebaseStatus::IndexingStarted => "indexing_started".to_string(),
                DaemonIndexCodebaseStatus::AlreadyIndexing => "already_indexing".to_string(),
            },
            path: response.project_key.canonical_root,
            files_queued: response.files_queued,
            message: response.message,
        })
    }

    async fn proxy_index_status_via_daemon(
        &self,
        input: IndexStatusInput,
    ) -> anyhow::Result<IndexStatusOutput> {
        let target = self.daemon_target_for_status(input.path.as_deref())?;
        let response = self.daemon_client.index_status(target).await?;

        Ok(IndexStatusOutput {
            indexed_files: response.indexed_files,
            total_chunks: response.total_chunks,
            last_indexed: response.last_indexed,
            estimated_stale: response.estimated_stale,
            freshness_state: Self::map_daemon_freshness_state(response.freshness_state),
            freshness_checked_at: response.freshness_checked_at,
            freshness_error: response.freshness_error,
            watching: response.watching,
            indexing: response.indexing.map(Self::map_daemon_indexing_progress),
        })
    }

    /// Resolve a backend for the given path. If `path` is None, returns the
    /// default (cwd) backend. Otherwise, finds the project root for the path,
    /// opens a CozoBackend on first use, and caches it for the session.
    async fn resolve_backend(
        &self,
        path: Option<&str>,
    ) -> anyhow::Result<(Arc<CozoBackend>, PathBuf)> {
        let target = match path {
            None => match self.default_project_root.as_ref() {
                Some(root) => root.clone(),
                None => {
                    return Ok((
                        Arc::clone(&self.backend),
                        self.manifest_path.as_ref().clone(),
                    ))
                }
            },
            Some(p) => PathBuf::from(p),
        };

        // Walk up to find .git (same logic as main.rs find_project_root)
        let project_root = {
            let abs = if target.is_absolute() {
                target.clone()
            } else {
                std::env::current_dir().unwrap_or_default().join(&target)
            };
            let mut dir = if abs.is_dir() {
                abs.clone()
            } else {
                abs.parent().unwrap_or(&abs).to_path_buf()
            };
            loop {
                if dir.join(".git").exists() {
                    break dir;
                }
                match dir.parent() {
                    Some(p) => dir = p.to_path_buf(),
                    None => break abs,
                }
            }
        };

        let (backend_path, manifest_path) =
            Self::resolve_index_paths_for_project_root(&project_root)?;

        // Check cache first
        {
            let cache = self.backend_cache.read().await;
            if let Some((backend, manifest)) = cache.get(&project_root) {
                if *manifest == manifest_path {
                    return Ok((Arc::clone(backend), manifest.clone()));
                }
            }
        }

        let backend = Arc::new(CozoBackend::open(&backend_path)?);

        tracing::info!(instance_id = %self.instance_id, project = %project_root.display(), manifest_path = %manifest_path.display(), "opened backend for new project");

        // Cache it
        let mut cache = self.backend_cache.write().await;
        cache.insert(project_root, (Arc::clone(&backend), manifest_path.clone()));

        Ok((backend, manifest_path))
    }

    fn storage_dir_from_manifest_path(manifest_path: &std::path::Path) -> anyhow::Result<PathBuf> {
        let manifest_dir = manifest_path
            .parent()
            .with_context(|| format!("manifest path has no parent: {}", manifest_path.display()))?;
        let generations_dir = manifest_dir.parent();
        if generations_dir
            .and_then(|dir| dir.file_name())
            .and_then(|name| name.to_str())
            == Some("generations")
        {
            generations_dir
                .and_then(std::path::Path::parent)
                .map(std::path::Path::to_path_buf)
                .with_context(|| {
                    format!(
                        "generation manifest path has no storage dir parent: {}",
                        manifest_path.display()
                    )
                })
        } else {
            Ok(manifest_dir.to_path_buf())
        }
    }

    fn project_root_from_manifest_path(manifest_path: &std::path::Path) -> anyhow::Result<PathBuf> {
        Self::storage_dir_from_manifest_path(manifest_path)?
            .parent()
            .map(std::path::Path::to_path_buf)
            .with_context(|| {
                format!(
                    "manifest path has no project root parent: {}",
                    manifest_path.display()
                )
            })
    }

    fn resolve_index_paths_for_project_root(
        project_root: &Path,
    ) -> anyhow::Result<(PathBuf, PathBuf)> {
        let storage_dir = project_root.join(".skelesearch");
        std::fs::create_dir_all(&storage_dir)
            .with_context(|| format!("create .skelesearch at {}", storage_dir.display()))?;

        let pointer_path = storage_dir.join("active-generation");
        if let Ok(pointer) = std::fs::read_to_string(&pointer_path) {
            let generation_id = pointer.trim();
            if !generation_id.is_empty() {
                let generation_dir = storage_dir.join("generations").join(generation_id);
                let (backend_path, manifest_path) = generation_db_paths(&generation_dir);
                if backend_path.exists() && manifest_path.exists() {
                    return Ok((backend_path, manifest_path));
                }
            }
        }

        Ok((
            storage_dir.join(INDEX_DB_FILE),
            storage_dir.join(MANIFEST_DB_FILE),
        ))
    }

    fn persisted_provider_name_from_manifest(
        manifest_path: &std::path::Path,
    ) -> anyhow::Result<Option<String>> {
        let manifest = ManifestStore::open(manifest_path)
            .with_context(|| format!("open manifest at {}", manifest_path.display()))?;
        manifest
            .get_meta("provider")
            .context("read provider metadata from manifest")
    }

    fn startup_default_provider() -> &'static str {
        if std::env::var("VOYAGE_API_KEY").map_or(false, |k| !k.is_empty()) {
            "voyage"
        } else if std::env::var("OPENAI_API_KEY").map_or(false, |k| !k.is_empty()) {
            "openai"
        } else {
            "fastembed"
        }
    }

    fn map_freshness_state(state: FreshnessState) -> IndexFreshnessState {
        match state {
            FreshnessState::Fresh => IndexFreshnessState::Fresh,
            FreshnessState::Stale => IndexFreshnessState::Stale,
            FreshnessState::Refreshing => IndexFreshnessState::Refreshing,
            FreshnessState::Unknown => IndexFreshnessState::Unknown,
        }
    }

    fn map_daemon_freshness_state(state: DaemonIndexFreshnessState) -> IndexFreshnessState {
        match state {
            DaemonIndexFreshnessState::Fresh => IndexFreshnessState::Fresh,
            DaemonIndexFreshnessState::Stale => IndexFreshnessState::Stale,
            DaemonIndexFreshnessState::Refreshing => IndexFreshnessState::Refreshing,
            DaemonIndexFreshnessState::Unknown => IndexFreshnessState::Unknown,
        }
    }

    fn compute_freshness_snapshot(
        manifest_path: &std::path::Path,
        refreshing: bool,
    ) -> FreshnessSnapshot {
        let stale_count_result = (|| -> anyhow::Result<usize> {
            let project_root = Self::project_root_from_manifest_path(manifest_path)?;
            let manifest = ManifestStore::open(manifest_path)?;
            manifest.count_stale(&project_root)
        })();

        FreshnessSnapshot::from_stale_count_result(stale_count_result).with_refreshing(refreshing)
    }

    fn repo_map_warning_prefix(freshness: &FreshnessSnapshot) -> Option<String> {
        match freshness.state {
            FreshnessState::Fresh => None,
            FreshnessState::Stale => Some(format!(
                "⚠ {} file(s) changed since last index. Run index to update.\n\n",
                freshness.estimated_stale
            )),
            FreshnessState::Refreshing => Some(
                "⚠ Index refresh is in progress. Repo map may be temporarily outdated.\n\n"
                    .to_string(),
            ),
            FreshnessState::Unknown => Some(match &freshness.freshness_error {
                Some(err) => {
                    format!("⚠ Index freshness is unknown ({err}). Repo map may be outdated.\n\n")
                }
                None => "⚠ Index freshness is unknown. Repo map may be outdated.\n\n".to_string(),
            }),
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

    /// Return the right error message for an empty index.
    ///
    /// When auto-indexing is already running, the caller gets a 'indexing in
    /// progress' message instead of the generic 'run index_codebase first'.
    async fn empty_index_error(&self) -> anyhow::Error {
        let state = self.index_state.read().await;
        if state.status == IndexingStatus::Running {
            anyhow::anyhow!(
                "indexing in progress for '{}' — use index_status to check progress",
                state.path
            )
        } else {
            anyhow::anyhow!("index is empty; run index first")
        }
    }

    /// Check if the index is empty; if so, start background indexing from cwd.
    ///
    /// Called from `on_initialized` (after MCP handshake). Logs but never propagates
    /// errors — startup must not fail just because auto-index could not start.
    ///
    /// Guards:
    /// - `SKELESEARCH_NO_AUTO_INDEX` env var disables auto-indexing entirely.
    /// - Skips if indexing is already running (prevents double-start on reconnect).
    /// - Skips if cwd does not look like a code project (no project markers found).
    async fn auto_index_if_needed(&self) {
        tracing::info!(instance_id = %self.instance_id, "auto_index_if_needed: entry");
        if self.inert_mode {
            tracing::info!(instance_id = %self.instance_id, "auto_index_if_needed: inert mode, skipping");
            return;
        }

        // Opt-out escape hatch for managed environments.
        if std::env::var("SKELESEARCH_NO_AUTO_INDEX").is_ok() {
            tracing::info!("auto_index_if_needed: SKELESEARCH_NO_AUTO_INDEX is set, skipping");
            return;
        }

        // Don't start a second run if one is already in flight.
        {
            let state = self.index_state.read().await;
            if state.status == IndexingStatus::Running {
                tracing::info!(instance_id = %self.instance_id, path = %state.path, "auto_index_if_needed: indexing already in progress, skipping");
                return;
            }
        }

        let cwd = match std::env::current_dir() {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!(error = %e, "auto_index_if_needed: failed to determine cwd, skipping");
                return;
            }
        };

        tracing::info!(cwd = %cwd.display(), "auto_index_if_needed: checking cwd");

        let is_project = looks_like_project(&cwd);
        tracing::info!(
            is_project,
            path = %cwd.display(),
            "auto_index_if_needed: project detection result"
        );

        if !is_project {
            tracing::info!(
                path = %cwd.display(),
                "auto_index_if_needed: no project markers found — skipping (run index explicitly, ",
            );
            tracing::info!(
                "auto_index_if_needed: or set SKELESEARCH_NO_AUTO_INDEX to silence this message)"
            );
            return;
        }

        let index_status = match self
            .proxy_index_status_via_daemon(IndexStatusInput {
                path: Some(cwd.to_string_lossy().to_string()),
            })
            .await
        {
            Ok(status) => status,
            Err(err) => {
                tracing::warn!(
                    error = %self.daemon_proxy_error("index_status", err),
                    path = %cwd.display(),
                    "auto_index_if_needed: failed to query daemon index status, skipping"
                );
                return;
            }
        };

        let initial_build_needed = index_status.total_chunks == 0;
        let stale_refresh_needed = index_status.total_chunks > 0
            && index_status.freshness_state == IndexFreshnessState::Stale;

        if !initial_build_needed && !stale_refresh_needed {
            tracing::info!(
                path = %cwd.display(),
                freshness_state = ?index_status.freshness_state,
                total_chunks = index_status.total_chunks,
                "auto_index_if_needed: index is current enough for startup, skipping"
            );
            return;
        }

        let (_backend_for_target, manifest_path) = match self
            .resolve_backend(Some(cwd.to_string_lossy().as_ref()))
            .await
        {
            Ok(resolved) => resolved,
            Err(err) => {
                tracing::warn!(
                    error = %err,
                    path = %cwd.display(),
                    "auto_index_if_needed: failed resolving target manifest path, skipping"
                );
                return;
            }
        };

        let provider_name = if stale_refresh_needed {
            match Self::persisted_provider_name_from_manifest(&manifest_path) {
                Ok(Some(provider)) => provider,
                Ok(None) => {
                    tracing::warn!(
                        path = %cwd.display(),
                        manifest_path = %manifest_path.display(),
                        "auto_index_if_needed: stale refresh skipped because manifest provider metadata is missing"
                    );
                    return;
                }
                Err(err) => {
                    tracing::warn!(
                        path = %cwd.display(),
                        manifest_path = %manifest_path.display(),
                        error = %err,
                        "auto_index_if_needed: stale refresh skipped because manifest provider metadata could not be read"
                    );
                    return;
                }
            }
        } else {
            match Self::persisted_provider_name_from_manifest(&manifest_path) {
                Ok(Some(provider)) => provider,
                _ => Self::startup_default_provider().to_string(),
            }
        };

        tracing::info!(
            path = %cwd.display(),
            provider = provider_name,
            trigger = if stale_refresh_needed { "startup_stale_refresh" } else { "startup_initial_build" },
            "auto_index_if_needed: triggering index_codebase"
        );

        // Surface auto-index failures so the user can act on them.
        // A failed auto-index must not crash the server, but silence is worse —
        // the user needs to know why search tools are returning errors.
        match self
            .proxy_index_codebase_via_daemon(IndexCodebaseInput {
                path: cwd.to_string_lossy().to_string(),
                provider: Some(provider_name),
            })
            .await
        {
            Ok(_) => {
                tracing::info!("auto_index_if_needed: index_codebase started successfully");
            }
            Err(e) => {
                let friendly = self.daemon_proxy_error("index_codebase", e);
                tracing::error!(
                    error = %friendly,
                    "auto_index_if_needed: index_codebase failed to start; search tools may not be available"
                );
                let mut state = self.index_state.write().await;
                state.status = IndexingStatus::Failed;
                state.error = Some(friendly);
            }
        }
    }

    /// Ensure a usable embedding provider is ready before a search.
    ///
    /// Returns an error when the index is empty so callers get a clear
    /// message rather than empty results.  When the server started with
    /// `NoopProvider` (dim == 1) but a persistent index exists, lazily
    /// initializes the correct provider (read from the manifest) and promotes
    /// it for future calls.
    async fn prepare_search_provider(
        &self,
        backend: &Arc<CozoBackend>,
        manifest_path: &Path,
    ) -> anyhow::Result<ArcProvider> {
        let stats = match backend.stats().await {
            Ok(s) => s,
            Err(ref e) if is_uninitialized_index_error(e) => {
                return Err(self.empty_index_error().await);
            }
            Err(e) => return Err(e),
        };
        if stats.total_chunks == 0 {
            return Err(self.empty_index_error().await);
        }

        let provider_name = {
            let manifest = ManifestStore::open(manifest_path).context("failed to open manifest")?;
            manifest
                .get_meta("provider")
                .context("failed to read provider from manifest")?
                .unwrap_or_else(|| "fastembed".to_string())
        };

        // Fast path: current provider already matches the persisted index provider.
        {
            let guard = self
                .provider
                .read()
                .map_err(|_| anyhow::anyhow!("provider lock poisoned"))?;
            if guard.dim() > 1 && guard.name() == provider_name {
                return Ok(guard.clone());
            }
        }

        // Slow path: started with NoopProvider or the persisted provider changed (for example
        // after daemon-side indexing with a different embedding model). Reload the correct provider
        // from the manifest and promote it for future searches.
        let real = provider_from_name(&provider_name)
            .with_context(|| format!("failed to initialize provider '{provider_name}'"))?;
        let arc_provider = ArcProvider::new(real);
        *self
            .provider
            .write()
            .map_err(|_| anyhow::anyhow!("provider lock poisoned"))? = arc_provider.clone();
        Ok(arc_provider)
    }

    // -----------------------------------------------------------------------
    // Session dedup helpers
    // -----------------------------------------------------------------------

    /// Configure the search pipeline from project config and env vars.
    /// Expansion is opt-in (SKELESEARCH_EXPANSION=1 or config); rerankers auto-detect API keys.
    fn auto_configure_pipeline(
        &self,
        config: &Config,
    ) -> (Option<Box<dyn QueryExpander>>, Option<Box<dyn Reranker>>) {
        // Expansion requires explicit opt-in: either config.search.expansion.enabled = true
        // or SKELESEARCH_EXPANSION=1|true|yes.  Having OPENAI_API_KEY alone is not enough —
        // many devs have it set for other tools and should not pay 1s+ latency per query.
        let expansion_enabled = match config.search.expansion.enabled {
            Some(true) => true,
            Some(false) => false,
            None => std::env::var("SKELESEARCH_EXPANSION")
                .ok()
                .map(|v| matches!(v.as_str(), "1" | "true" | "yes"))
                .unwrap_or(false),
        };
        let expander: Option<Box<dyn QueryExpander>> = if expansion_enabled {
            std::env::var("OPENAI_API_KEY")
                .ok()
                .filter(|k| !k.is_empty())
                .map(|key| -> Box<dyn QueryExpander> { Box::new(LLMExpander::new(key)) })
        } else {
            None
        };

        // Try reranker keys in order: JINA_API_KEY, COHERE_API_KEY, VOYAGE_API_KEY.
        let reranker: Option<Box<dyn Reranker>> = None
            .or_else(|| {
                std::env::var("JINA_API_KEY")
                    .ok()
                    .filter(|k| !k.is_empty())
                    .and_then(|key| skelesearch_rerank_api::reranker_from_name("jina", key).ok())
                    .map(|r| -> Box<dyn Reranker> { Box::new(r) })
            })
            .or_else(|| {
                std::env::var("COHERE_API_KEY")
                    .ok()
                    .filter(|k| !k.is_empty())
                    .and_then(|key| skelesearch_rerank_api::reranker_from_name("cohere", key).ok())
                    .map(|r| -> Box<dyn Reranker> { Box::new(r) })
            })
            .or_else(|| {
                std::env::var("VOYAGE_API_KEY")
                    .ok()
                    .filter(|k| !k.is_empty())
                    .and_then(|key| skelesearch_rerank_api::reranker_from_name("voyage", key).ok())
                    .map(|r| -> Box<dyn Reranker> { Box::new(r) })
            })
            .or_else(|| {
                // SKELESEARCH_RERANKER=local enables the local ONNX reranker.
                // SKELESEARCH_RERANKER_MODEL_DIR overrides the default cache path.
                // Requires the `local-reranker` cargo feature; CoreML/CUDA builds
                // continue to opt in via their feature flags.
                let local = std::env::var("SKELESEARCH_RERANKER")
                    .ok()
                    .filter(|v| v == "local");
                if local.is_none() {
                    return None;
                }
                #[cfg(feature = "local-reranker")]
                {
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
                    return result.ok().map(|r| -> Box<dyn Reranker> { Box::new(r) });
                }
                #[cfg(not(feature = "local-reranker"))]
                {
                    tracing::warn!(
                        "SKELESEARCH_RERANKER=local set but the `local-reranker` feature is not enabled; \
                         rebuild with --features local-reranker (or coreml/cuda)"
                    );
                    None
                }
            });

        // SKELESEARCH_RERANKER=qwen3 enables the Qwen3-Reranker-0.6B.
        // Requires the `qwen3` cargo feature; emits a warning when the env
        // var is set but the feature is absent so operators get clear feedback.
        let reranker = reranker.or_else(|| {
            if std::env::var("SKELESEARCH_RERANKER").ok().as_deref() != Some("qwen3") {
                return None;
            }
            #[cfg(feature = "qwen3")]
            {
                let result = if let Ok(dir) = std::env::var("SKELESEARCH_RERANKER_MODEL_DIR") {
                    skelesearch_rerank_qwen3::Qwen3Reranker::from_path(std::path::Path::new(&dir))
                } else {
                    skelesearch_rerank_qwen3::Qwen3Reranker::from_hf()
                };
                return result.ok().map(|r| -> Box<dyn Reranker> { Box::new(r) });
            }
            #[cfg(not(feature = "qwen3"))]
            {
                tracing::warn!(
                    "SKELESEARCH_RERANKER=qwen3 set but the `qwen3` feature is not enabled; \
                     rebuild with --features qwen3"
                );
                None
            }
        });

        if expander.is_some() {
            tracing::info!("query expansion enabled (OPENAI_API_KEY detected)");
        } else {
            tracing::info!("query expansion disabled (set SKELESEARCH_EXPANSION=1 to enable)");
        }
        if reranker.is_some() {
            let source = match std::env::var("SKELESEARCH_RERANKER").ok().as_deref() {
                Some("local") => "local ONNX model",
                Some("qwen3") => "Qwen3-Reranker-0.6B (ONNX)",
                _ => "cloud API key",
            };
            tracing::info!(source, "reranking enabled");
        }

        (expander, reranker)
    }

    /// Return a cached Searcher or build one on the first call.
    ///
    /// The searcher is invalidated (cache cleared) after indexing so
    /// provider changes and config changes are picked up.
    ///
    /// # Blocking note
    /// CozoDB's SQLite backend serialises all transactions through a
    /// `ShardedLock<()>`.  Write transactions (background indexer) hold
    /// this lock for the duration of each `upsert_chunks` call, which
    /// includes incremental HNSW graph updates (potentially seconds per
    /// batch on a large corpus).  Any concurrent read (stats, search)
    /// blocks until the write completes, easily exceeding the 30 s MCP
    /// timeout.  Callers MUST check `is_indexing_active` before calling
    /// this function and return a fast error when indexing is in progress.
    async fn get_or_build_searcher(&self) -> anyhow::Result<Arc<CachedSearcher>> {
        let (backend, manifest_path) = self.resolve_backend(None).await?;

        // Fast path: cached searcher exists.
        {
            let guard = self.cached_searcher.read().await;
            let path_guard = self.cached_searcher_manifest_path.read().await;
            if let (Some(s), Some(cached_manifest_path)) = (guard.as_ref(), path_guard.as_ref()) {
                if *cached_manifest_path == manifest_path {
                    return Ok(Arc::clone(s));
                }
            }
        }

        // Slow path: build and cache.
        let build_start = std::time::Instant::now();
        let mut guard = self.cached_searcher.write().await;
        let mut path_guard = self.cached_searcher_manifest_path.write().await;
        // Double-check after acquiring write lock.
        if let (Some(s), Some(cached_manifest_path)) = (guard.as_ref(), path_guard.as_ref()) {
            if *cached_manifest_path == manifest_path {
                return Ok(Arc::clone(s));
            }
        }

        tracing::info!("searcher cache miss — building searcher");
        let t0 = std::time::Instant::now();
        let provider = self
            .prepare_search_provider(&backend, &manifest_path)
            .await?;
        tracing::info!(
            elapsed_ms = t0.elapsed().as_millis() as u64,
            "prepare_search_provider done"
        );

        let searcher = Searcher::new(Arc::clone(&backend), provider);
        // Load config early so pipeline auto-configuration can read expansion/sparse settings.
        let t1 = std::time::Instant::now();
        let indexed_root = backend
            .list_indexed_paths()
            .await
            .ok()
            .and_then(|p| common_ancestor(&p));
        let root = ManifestStore::open(&manifest_path)
            .ok()
            .and_then(|m| m.get_meta("index_root").ok().flatten())
            .map(PathBuf::from)
            .or(indexed_root)
            .unwrap_or_else(|| PathBuf::from("/"));
        tracing::info!(
            elapsed_ms = t1.elapsed().as_millis() as u64,
            "list_indexed_paths done"
        );

        let config = Config::load(&root).unwrap_or_default();
        let (expander, reranker) = self.auto_configure_pipeline(&config);
        let searcher = if let Some(e) = expander {
            searcher.with_expander(e)
        } else {
            searcher
        };
        let searcher = if let Some(r) = reranker {
            searcher.with_reranker(r)
        } else {
            searcher
        };
        // Apply pagerank_boost and tuning from project config.
        let searcher = {
            let searcher = searcher.with_search_tuning(&config);
            let searcher = if config.search.pagerank_boost == Some(false) {
                searcher.with_pagerank_boost(false)
            } else {
                searcher
            };
            if config.search.sparse.enabled {
                match FastEmbedSparseProvider::bgem3() {
                    Ok(sp) => searcher.with_sparse_provider(Arc::new(sp)),
                    Err(e) => {
                        tracing::warn!("sparse provider init failed: {e}, skipping sparse search");
                        searcher
                    }
                }
            } else {
                searcher
            }
        };

        let total_build_ms = build_start.elapsed().as_millis() as u64;
        tracing::info!(
            total_build_ms,
            "searcher built and cached (LRU + connection pool will be reused)"
        );
        let arc = Arc::new(searcher);
        *guard = Some(Arc::clone(&arc));
        *path_guard = Some(manifest_path);
        Ok(arc)
    }

    /// Invalidate the cached searcher (call after indexing).
    async fn invalidate_searcher_cache(&self) {
        let mut guard = self.cached_searcher.write().await;
        *guard = None;
        let mut path_guard = self.cached_searcher_manifest_path.write().await;
        *path_guard = None;
        tracing::info!("searcher cache invalidated");
    }

    /// Semantic + FTS hybrid search.
    ///
    /// Returns an error when the index is empty (via `prepare_search_provider`).
    /// Returns a fast error when background indexing is active to avoid blocking
    /// on CozoDB's internal write lock (ShardedLock) held by the indexer.
    #[tracing::instrument(skip_all, fields(query = %input.query, top_k = input.top_k))]
    pub async fn search_code(&self, input: SearchCodeInput) -> anyhow::Result<SearchCodeResponse> {
        let target = self.daemon_target_for_status(None)?;
        let response = self
            .daemon_client
            .search_code(skelesearch_service::SearchCodeRequest {
                target,
                query: input.query,
                top_k: input.top_k,
                include_graph: input.include_graph,
                max_depth: input.max_depth,
                diversity: input.diversity,
                max_tokens: input.max_tokens,
                branch_scope: input.branch_scope,
                session_id: input.session_id,
            })
            .await
            .map_err(|err| anyhow::anyhow!(self.daemon_proxy_error("search_code", err)))?;

        Ok(SearchCodeResponse {
            results: response
                .results
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
                .collect(),
            _timings: skelesearch_core::SearchTimings::default(),
        })
    }

    /// Index the codebase at `input.path` in the background.
    ///
    /// Returns immediately after spawning the indexing task.
    /// Callers should poll `index_status` to observe completion.
    ///
    /// # Error conditions (returned synchronously)
    /// - Unknown provider name — rejected before any I/O.
    /// - Indexing already in progress — returns `"already_indexing"` status (not an error).
    ///
    /// # Send safety
    /// `ManifestStore` wraps a raw SQLite pointer and is `!Send`.  The background
    /// task runs the indexer inside `spawn_blocking` → `current_thread` runtime,
    /// which prevents the `!Send` future from crossing thread boundaries.
    #[tracing::instrument(skip_all, fields(path = %input.path))]
    pub async fn index_codebase(
        &self,
        input: IndexCodebaseInput,
    ) -> anyhow::Result<IndexCodebaseOutput> {
        // Reject concurrent indexing before any I/O.
        {
            let state = self.index_state.read().await;
            if state.status == IndexingStatus::Running {
                return Ok(IndexCodebaseOutput {
                    status: "already_indexing".to_string(),
                    path: state.path.clone(),
                    files_queued: 0,
                    message: format!(
                        "indexing already in progress for '{}'; use index_status to check progress",
                        state.path
                    ),
                });
            }
        }

        // Validate provider name without loading the model — provider_from_name
        // for fastembed loads a 450MB ONNX model synchronously and must not run
        // on the async thread (it would block on_initialized and rmcp's message loop).
        let provider_name = input.provider.as_deref().unwrap_or("fastembed");
        match provider_name {
            "fastembed" | "voyage" | "openai" => {}
            unknown => {
                return Err(anyhow::anyhow!(
                    "unknown provider: '{unknown}'. Valid: fastembed, voyage, openai"
                ))
            }
        }
        let provider_name_owned = provider_name.to_string();

        let path = std::path::PathBuf::from(&input.path);
        let index_path = input.path.clone();

        // Quick best-effort file count before spawning.
        // Capped at 1 second so large repos don't delay the response.
        let count_path = path.clone();
        let files_queued = {
            let timeout_result = tokio::time::timeout(
                std::time::Duration::from_secs(1),
                tokio::task::spawn_blocking(move || count_files_recursive(&count_path)),
            )
            .await;
            match timeout_result {
                Ok(Ok(n)) => n,
                _ => 0,
            }
        };

        // Resolve the correct backend for the target path. For cross-project
        // indexing, this opens/caches a backend in the target's .skelesearch/.
        let (backend, manifest_path) = self.resolve_backend(Some(&index_path)).await?;
        let storage_dir = Self::storage_dir_from_manifest_path(&manifest_path)?;

        // Best-effort stale-status cleanup + visibility for logs.
        let _ = read_shared_indexing_status(&storage_dir)?;

        let now = chrono::Utc::now();
        let initial_shared_status = SharedIndexingStatus {
            instance_id: self.instance_id.to_string(),
            pid: std::process::id(),
            path: index_path.clone(),
            provider: provider_name_owned.clone(),
            trigger: "manual_or_auto".to_string(),
            status: "running".to_string(),
            started_at: now,
            updated_at: now,
            files_total: files_queued,
            files_done: 0,
            chunks_done: 0,
            cache_hits: 0,
            error: None,
        };

        let lease = match try_acquire_indexing_lease(&storage_dir, &initial_shared_status)? {
            Some(lease) => {
                tracing::info!(
                    instance_id = %self.instance_id,
                    pid = std::process::id(),
                    path = %index_path,
                    storage_dir = %storage_dir.display(),
                    "indexing lease acquired"
                );
                lease
            }
            None => {
                let shared = read_shared_indexing_status(&storage_dir)?;
                tracing::info!(
                    instance_id = %self.instance_id,
                    path = %index_path,
                    storage_dir = %storage_dir.display(),
                    holder_instance_id = shared.as_ref().map(|s| s.instance_id.as_str()),
                    holder_pid = shared.as_ref().map(|s| s.pid),
                    "indexing lease denied: another process is indexing"
                );
                return Ok(IndexCodebaseOutput {
                    status: "already_indexing".to_string(),
                    path: index_path,
                    files_queued: 0,
                    message: "another skelesearch process is indexing this project; use index_status to check progress".to_string(),
                });
            }
        };

        // Mark Running before spawning to prevent TOCTOU: a second concurrent call
        // arriving before the spawned task runs would otherwise see Idle.
        {
            let mut state = self.index_state.write().await;
            state.status = IndexingStatus::Running;
            state.path = input.path.clone();
            state.storage_dir = storage_dir.display().to_string();
            state.files_found = files_queued;
            state.files_done = 0;
            state.chunks_done = 0;
            state.cache_hits = 0;
            state.error = None;
            state.started_at = std::time::Instant::now();
        }
        tracing::info!(
            instance_id = %self.instance_id,
            trigger = "manual_or_auto",
            provider = provider_name,
            path = %input.path,
            files_queued,
            "index_codebase accepted"
        );

        let manifest_path = Arc::new(manifest_path);
        let provider_arc = Arc::clone(&self.provider);
        let cached_searcher_arc = Arc::clone(&self.cached_searcher);
        let index_state = Arc::clone(&self.index_state);
        let instance_id = Arc::clone(&self.instance_id);
        spawn_index_resource_sampler(
            Arc::clone(&instance_id),
            Arc::clone(&index_state),
            input.path.clone(),
            "manual_or_auto",
            provider_name_owned.clone(),
        );

        tokio::task::spawn(async move {
            let _lease = lease;
            let backend2 = Arc::clone(&backend);
            let manifest_path2 = Arc::clone(&manifest_path);
            let path2 = path.clone();
            let provider_name_for_closure = provider_name_owned;

            tracing::info!(
                instance_id = %instance_id,
                path = %path2.display(),
                provider = %provider_name_for_closure,
                "background index task started"
            );

            let result = tokio::task::spawn_blocking(move || {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(|e| anyhow::anyhow!("runtime build: {e}"))?;
                rt.block_on(async {
                    let provider = provider_from_name(&provider_name_for_closure)
                        .map(ArcProvider::new)
                        .with_context(|| {
                            format!(
                                "failed to initialize provider '{}'",
                                provider_name_for_closure
                            )
                        })?;
                    let manifest = Arc::new(ManifestStore::open(manifest_path2.as_path())?);
                    let config = Config::load(&path2).context("load .skelesearch.toml")?;
                    let indexer = Indexer::new(backend2, manifest, provider.clone())
                        .with_excludes(config.index.exclude.clone())
                        .with_include_extensions(config.index.include_extensions.clone())
                        .with_scope_prefix(config.index.scope_prefix);
                    let indexer = if config.search.sparse.enabled {
                        match FastEmbedSparseProvider::bgem3() {
                            Ok(sp) => indexer.with_sparse_provider(Arc::new(sp)),
                            Err(e) => {
                                tracing::warn!(
                                    "sparse provider init failed: {e}, skipping sparse indexing"
                                );
                                indexer
                            }
                        }
                    } else {
                        indexer
                    };
                    let result = indexer.index_path(&path2).await;
                    result.map(|r| (r, provider))
                })
            })
            .await;

            match result {
                Ok(Ok((index_result, provider))) => {
                    // Promote provider only on success — a partial index must never
                    // change the server's active embedding dimension.
                    if let Ok(mut guard) = provider_arc.write() {
                        *guard = provider.clone();
                    }
                    // Invalidate cached searcher so the next search rebuilds with
                    // the new provider and any fresh config.
                    *cached_searcher_arc.write().await = None;
                    tracing::info!(instance_id = %instance_id, "searcher cache invalidated after background indexing");

                    let mut state = index_state.write().await;
                    state.status = IndexingStatus::Done;
                    state.files_done = index_result.indexed_files;
                    state.chunks_done = index_result.total_chunks;
                    state.cache_hits = index_result.cache_hits;
                    tracing::info!(
                        instance_id = %instance_id,
                        path = %state.path,
                        provider = provider.name(),
                        indexed = index_result.indexed_files,
                        chunks = index_result.total_chunks,
                        cache_hits = index_result.cache_hits,
                        elapsed_s = state.started_at.elapsed().as_secs(),
                        "background indexing complete"
                    );
                }
                Ok(Err(index_err)) => {
                    let err_str = index_err.to_string();
                    let err_lower = err_str.to_lowercase();
                    if err_lower.contains("dimension mismatch")
                        || err_lower.contains("arity mismatch")
                    {
                        tracing::error!(
                            error = %index_err,
                            "Auto-index failed: index was built with a different provider \
                             (dimension mismatch). Delete .skelesearch/ directory and restart, \
                             or set VOYAGE_API_KEY in the MCP server environment."
                        );
                    } else {
                        tracing::error!(instance_id = %instance_id, error = %index_err, "background indexing failed");
                    }
                    let mut state = index_state.write().await;
                    state.status = IndexingStatus::Failed;
                    state.error = Some(friendly_index_error(&index_err));
                }
                Err(join_err) => {
                    tracing::error!(instance_id = %instance_id, error = %join_err, "indexer task panicked");
                    let mut state = index_state.write().await;
                    state.status = IndexingStatus::Failed;
                    state.error = Some(format!("indexer task panicked: {join_err}"));
                }
            }
        });

        Ok(IndexCodebaseOutput {
            status: "indexing_started".to_string(),
            path: input.path,
            files_queued,
            message: "Indexing started in the background. Use index_status to check progress."
                .to_string(),
        })
    }

    /// Index a path synchronously using an already-constructed provider.
    ///
    /// Used directly by tests to exercise the indexing pipeline without going
    /// through the async background machinery or the string-based provider factory.
    /// Production code goes through `index_codebase`, which spawns this logic
    /// in a background task.
    pub async fn run_index(
        &self,
        path: &std::path::Path,
        provider: ArcProvider,
    ) -> anyhow::Result<IndexResult> {
        let backend = Arc::clone(&self.backend);
        let manifest_path = Arc::clone(&self.manifest_path);
        let path = path.to_path_buf();
        let provider_for_closure = provider.clone();

        // ManifestStore is !Send; run in a dedicated single-thread runtime.
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
                    .with_include_extensions(config.index.include_extensions.clone())
                    .with_scope_prefix(config.index.scope_prefix);
                let indexer = if config.search.sparse.enabled {
                    match FastEmbedSparseProvider::bgem3() {
                        Ok(sp) => indexer.with_sparse_provider(Arc::new(sp)),
                        Err(e) => {
                            tracing::warn!(
                                "sparse provider init failed: {e}, skipping sparse indexing"
                            );
                            indexer
                        }
                    }
                } else {
                    indexer
                };
                indexer.index_path(&path).await
            })
        })
        .await
        .context("indexer task panicked")?
        .context("indexer.index_path")?;

        *self
            .provider
            .write()
            .map_err(|_| anyhow::anyhow!("provider lock poisoned"))? = provider;
        self.invalidate_searcher_cache().await;

        Ok(result)
    }

    /// Return current index statistics, including live background-indexing progress.
    pub async fn index_status(&self, input: IndexStatusInput) -> anyhow::Result<IndexStatusOutput> {
        let (backend, manifest_path) = self.resolve_backend(input.path.as_deref()).await?;
        let storage_dir = Self::storage_dir_from_manifest_path(&manifest_path)?;
        let indexing = self.current_indexing_progress(&storage_dir).await;
        let refreshing = matches!(
            indexing.as_ref().map(|p| p.status.as_str()),
            Some("running")
        );
        let freshness = Self::compute_freshness_snapshot(&manifest_path, refreshing);
        let stats = match backend.stats().await {
            Ok(s) => Some(s),
            Err(ref e) if is_uninitialized_index_error(e) => None,
            Err(e) => return Err(e),
        };

        if refreshing {
            return Ok(IndexStatusOutput {
                indexed_files: stats
                    .as_ref()
                    .map(|s| s.indexed_files)
                    .unwrap_or_else(|| indexing.as_ref().map(|p| p.files_done).unwrap_or(0)),
                total_chunks: stats
                    .as_ref()
                    .map(|s| s.total_chunks)
                    .unwrap_or_else(|| indexing.as_ref().map(|p| p.chunks_done).unwrap_or(0)),
                last_indexed: stats
                    .as_ref()
                    .and_then(|s| s.last_indexed)
                    .map(|dt: chrono::DateTime<chrono::Utc>| dt.to_rfc3339()),
                estimated_stale: freshness.estimated_stale,
                freshness_state: Self::map_freshness_state(freshness.state),
                freshness_checked_at: freshness.freshness_checked_at.map(|dt| dt.to_rfc3339()),
                freshness_error: freshness.freshness_error,
                watching: self.watcher_started.load(Ordering::Relaxed),
                indexing,
            });
        }
        let stats = match stats {
            Some(s) => s,
            None => {
                return Ok(IndexStatusOutput {
                    indexed_files: 0,
                    total_chunks: 0,
                    last_indexed: None,
                    estimated_stale: freshness.estimated_stale,
                    freshness_state: Self::map_freshness_state(freshness.state),
                    freshness_checked_at: freshness.freshness_checked_at.map(|dt| dt.to_rfc3339()),
                    freshness_error: freshness.freshness_error,
                    watching: self.watcher_started.load(Ordering::Relaxed),
                    indexing,
                });
            }
        };
        Ok(IndexStatusOutput {
            indexed_files: stats.indexed_files,
            total_chunks: stats.total_chunks,
            last_indexed: stats
                .last_indexed
                .map(|dt: chrono::DateTime<chrono::Utc>| dt.to_rfc3339()),
            estimated_stale: freshness.estimated_stale,
            freshness_state: Self::map_freshness_state(freshness.state),
            freshness_checked_at: freshness.freshness_checked_at.map(|dt| dt.to_rfc3339()),
            freshness_error: freshness.freshness_error,
            watching: self.watcher_started.load(Ordering::Relaxed),
            indexing,
        })
    }

    /// Snapshot the current background indexing progress for inclusion in `IndexStatusOutput`.
    /// Returns `None` when neither local nor shared cross-process indexing is active.
    async fn current_indexing_progress(
        &self,
        storage_dir: &std::path::Path,
    ) -> Option<IndexingProgress> {
        let storage_dir_str = storage_dir.display().to_string();
        let local = {
            let state = self.index_state.read().await;
            match state.status {
                IndexingStatus::Idle => None,
                _ if state.storage_dir != storage_dir_str => None,
                _ => {
                    let elapsed = state.started_at.elapsed().as_secs_f64();
                    Some(IndexingProgress {
                        status: match state.status {
                            IndexingStatus::Running => "running".to_string(),
                            IndexingStatus::Done => "done".to_string(),
                            IndexingStatus::Failed => "failed".to_string(),
                            IndexingStatus::Idle => unreachable!(),
                        },
                        path: state.path.clone(),
                        files_done: state.files_done,
                        files_total: state.files_found,
                        chunks_done: state.chunks_done,
                        cache_hits: state.cache_hits,
                        elapsed_seconds: elapsed,
                        error: state.error.clone(),
                    })
                }
            }
        };

        if matches!(local.as_ref().map(|p| p.status.as_str()), Some("running")) {
            return local;
        }

        match read_shared_indexing_status(storage_dir) {
            Ok(Some(shared)) => {
                let elapsed_seconds = (chrono::Utc::now() - shared.started_at)
                    .to_std()
                    .map(|d| d.as_secs_f64())
                    .unwrap_or(0.0);
                Some(IndexingProgress {
                    status: shared.status,
                    path: shared.path,
                    files_done: shared.files_done,
                    files_total: shared.files_total,
                    chunks_done: shared.chunks_done,
                    cache_hits: shared.cache_hits,
                    elapsed_seconds,
                    error: shared.error,
                })
            }
            Ok(None) => match is_indexing_active_elsewhere(storage_dir) {
                Ok(true) => Some(IndexingProgress {
                    status: "running".to_string(),
                    path: storage_dir.display().to_string(),
                    files_done: 0,
                    files_total: 0,
                    chunks_done: 0,
                    cache_hits: 0,
                    elapsed_seconds: 0.0,
                    error: None,
                }),
                Ok(false) => local,
                Err(err) => {
                    tracing::warn!(
                        instance_id = %self.instance_id,
                        storage_dir = %storage_dir.display(),
                        error = %err,
                        "failed to probe indexing lock while reading shared indexing status"
                    );
                    local
                }
            },
            Err(err) => {
                tracing::warn!(
                    instance_id = %self.instance_id,
                    storage_dir = %storage_dir.display(),
                    error = %err,
                    "failed to read shared indexing status"
                );
                local
            }
        }
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
                return Ok(FileContextOutput {
                    chunks: vec![],
                    imports: vec![],
                    imported_by: vec![],
                });
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
    pub async fn smart_search(&self, input: SmartSearchInput) -> anyhow::Result<SmartSearchOutput> {
        // If a non-default project is specified and search_code doesn't support
        // cross-project yet, resolve the backend and build a temporary searcher.
        if let Some(ref project) = input.project {
            let (backend, manifest_path) = self.resolve_backend(Some(project.as_str())).await?;
            let storage_dir = Self::storage_dir_from_manifest_path(&manifest_path)?;
            if matches!(
                self.current_indexing_progress(&storage_dir)
                    .await
                    .as_ref()
                    .map(|p| p.status.as_str()),
                Some("running")
            ) {
                anyhow::bail!(
                    "Index is being built for '{}'. Poll index_status to check progress; semantic search will be available once indexing completes.",
                    project
                );
            }
            // Read provider from target project's manifest — dimensions must match its index.
            let provider_name = ManifestStore::open(&manifest_path)
                .ok()
                .and_then(|m| m.get_meta("provider").ok().flatten())
                .unwrap_or_else(|| "fastembed".to_string());
            let real = provider_from_name(&provider_name).with_context(|| {
                format!("init provider '{}' for project {}", provider_name, project)
            })?;
            let provider = ArcProvider::new(real);
            let searcher = Searcher::new(backend, provider);
            // Delegate to a simplified search path for cross-project queries.
            let (mut results, _timings) = searcher
                .search_with_timings(
                    &input.query,
                    input.top_k.max(1),
                    input.include_graph,
                    if input.include_graph { 2 } else { 0 },
                    input.diversity,
                    input.max_tokens.or(Some(8192)),
                )
                .await?;
            if let Some(ref scope) = input.scope {
                results.retain(|r| std::path::Path::new(&r.file_path).starts_with(scope.as_str()));
            }
            let rows = results
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
            return Ok(SmartSearchOutput {
                strategy: input.intent.as_deref().unwrap_or("semantic").to_string(),
                results: SmartSearchResults::Semantic(rows),
            });
        }
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
                let (backend, manifest_path) = self.resolve_backend(None).await?;
                let paths = match backend.list_indexed_paths().await {
                    Ok(p) => p,
                    Err(ref e) if is_uninitialized_index_error(e) => vec![],
                    Err(e) => return Err(e),
                };
                if paths.is_empty() {
                    SmartSearchResults::Grep(vec![])
                } else {
                    // Resolve grep root from explicit scope, manifest index_root, or indexed paths.
                    let root = if let Some(ref scope) = input.scope {
                        let p = PathBuf::from(scope);
                        if p.is_absolute() {
                            p
                        } else {
                            std::env::current_dir()
                                .unwrap_or_else(|_| PathBuf::from("."))
                                .join(p)
                        }
                    } else {
                        let manifest_root = ManifestStore::open(&manifest_path)
                            .ok()
                            .and_then(|m| m.get_meta("index_root").ok().flatten())
                            .map(PathBuf::from);
                        manifest_root
                            .or_else(|| {
                                common_ancestor(&paths).map(|p| {
                                    if p.is_absolute() {
                                        p
                                    } else {
                                        std::env::current_dir()
                                            .unwrap_or_else(|_| PathBuf::from("."))
                                            .join(p)
                                    }
                                })
                            })
                            .filter(|p| p != &PathBuf::from("/") && !p.as_os_str().is_empty())
                            .or_else(|| std::env::current_dir().ok())
                            .unwrap_or_else(|| PathBuf::from("."))
                    };
                    // Guard: if the resolved root doesn't exist on disk, grep will
                    // produce an IO error.  Fall back to semantic search so the caller
                    // always gets results rather than a crash.
                    if !root.exists() {
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
                        let mut rows = response.results;
                        if let Some(ref scope) = input.scope {
                            rows.retain(|r| {
                                std::path::Path::new(&r.file_path).starts_with(scope.as_str())
                            });
                        }
                        return Ok(SmartSearchOutput {
                            strategy: "semantic".to_string(),
                            results: SmartSearchResults::Semantic(rows),
                        });
                    }
                    let opts = GrepOptions {
                        max_results: input.top_k.max(1),
                        case_insensitive: false,
                    };
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
                        let proj_root = self
                            .manifest_path
                            .parent()
                            .and_then(|p| p.parent())
                            .unwrap_or_else(|| std::path::Path::new("."));
                        let changed = skelesearch_core::git::changed_files_on_branch(proj_root)?;
                        if !changed.is_empty() {
                            rows.retain(|r| {
                                changed.iter().any(|c| {
                                    r.file_path.ends_with(c.as_str()) || c.ends_with(&r.file_path)
                                })
                            });
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
        Ok(SmartSearchOutput {
            strategy: strategy.to_string(),
            results,
        })
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
                        max_depth: None,
                        project: input.project.clone(),
                    })
                    .await?;
                Ok(SmartSearchOutput {
                    strategy: "impact".to_string(),
                    results: SmartSearchResults::Impact(impact),
                })
            }
            "trace" => {
                if input.symbols.len() < 2 {
                    return Err(anyhow::anyhow!(
                        "trace intent requires exactly 2 symbols: [start, end]"
                    ));
                }
                let start_name = input.symbols[0].clone();
                let end_name = input.symbols[1].clone();

                let (trace_backend, _) = self.resolve_backend(input.project.as_deref()).await?;
                let start_syms = trace_backend
                    .find_symbols(&start_name, None)
                    .await
                    .unwrap_or_default();
                let end_syms = trace_backend
                    .find_symbols(&end_name, None)
                    .await
                    .unwrap_or_default();

                let trace_info = if let (Some(s), Some(e)) = (start_syms.first(), end_syms.first())
                {
                    let start_callees = trace_backend
                        .get_callees(&s.file_path, &s.name)
                        .await
                        .unwrap_or_default();
                    let end_callers = trace_backend
                        .get_callers(&e.file_path, &e.name)
                        .await
                        .unwrap_or_default();

                    // Direct connection: does start directly call end?
                    let direct = start_callees.iter().any(|c| {
                        c.callee_file.as_deref() == Some(e.file_path.as_str())
                            && c.callee_symbol.as_deref() == Some(e.name.as_str())
                    });
                    if direct {
                        format!("{} directly calls {}", start_name, end_name)
                    } else {
                        // One-hop: start calls X, X calls end?
                        let start_callee_keys: std::collections::HashSet<(String, String)> =
                            start_callees
                                .iter()
                                .filter_map(|c| {
                                    Some((c.callee_file.clone()?, c.callee_symbol.clone()?))
                                })
                                .collect();
                        let intermediaries: Vec<String> = end_callers
                            .iter()
                            .filter(|c| {
                                start_callee_keys
                                    .contains(&(c.caller_file.clone(), c.caller_symbol.clone()))
                            })
                            .take(5)
                            .map(|c| format!("{}::{}", c.caller_file, c.caller_symbol))
                            .collect();
                        if !intermediaries.is_empty() {
                            format!(
                                "{} -> [{}] -> {}",
                                start_name,
                                intermediaries.join(", "),
                                end_name
                            )
                        } else {
                            format!(
                                "No direct or 1-hop call path found between {} and {}",
                                start_name, end_name
                            )
                        }
                    }
                } else {
                    format!(
                        "Could not resolve symbols: '{}' and/or '{}'",
                        start_name, end_name
                    )
                };

                Ok(SmartSearchOutput {
                    strategy: "trace".to_string(),
                    results: SmartSearchResults::Semantic(vec![SearchCodeRow {
                        file_path: String::new(),
                        start_line: 0,
                        end_line: 0,
                        content: trace_info,
                        score: 1.0,
                        match_quality: "high".to_string(),
                        why: "trace".to_string(),
                    }]),
                })
            }
            other => Err(anyhow::anyhow!(
                "unknown intent {:?}; valid values are: find, understand, impact, trace",
                other
            )),
        }
    }

    /// Find symbol definitions by name, optionally filtered by kind.
    pub async fn find_symbol(&self, input: FindSymbolInput) -> anyhow::Result<Vec<SymbolRow>> {
        let (backend, _) = self.resolve_backend(input.project.as_deref()).await?;
        let results = match backend
            .find_symbols(&input.name, input.kind.as_deref())
            .await
        {
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

    /// Return a context bundle for a symbol: definition, source chunk, import graph edges,
    /// and (optionally) test files that cover the symbol's file.
    ///
    /// Returns a partial result when the symbol is found but chunk retrieval fails —
    /// callers always get what is available rather than an opaque error.
    pub async fn get_symbol_context(
        &self,
        input: GetSymbolContextInput,
    ) -> anyhow::Result<SymbolContextOutput> {
        // Resolve backend for the target project.
        let (backend, _) = self.resolve_backend(input.project.as_deref()).await?;
        // Step 1: resolve the symbol.
        let symbols = match backend
            .find_symbols(&input.name, input.kind.as_deref())
            .await
        {
            Ok(r) => r,
            Err(ref e) if is_uninitialized_index_error(e) => {
                return Ok(SymbolContextOutput {
                    symbol: None,
                    match_count: 0,
                    ambiguous: false,
                    source: None,
                    imported_by: vec![],
                    imported_by_truncated: false,
                    imports: vec![],
                    imports_truncated: false,
                    test_files: vec![],
                    role: None,
                    callers: vec![],
                    callees: vec![],
                });
            }
            Err(e) => return Err(e),
        };
        let match_count = symbols.len();

        let sym = match symbols.into_iter().next() {
            Some(s) => s,
            None => {
                return Ok(SymbolContextOutput {
                    symbol: None,
                    match_count: 0,
                    ambiguous: false,
                    source: None,
                    imported_by: vec![],
                    imported_by_truncated: false,
                    imports: vec![],
                    imports_truncated: false,
                    test_files: vec![],
                    role: None,
                    callers: vec![],
                    callees: vec![],
                });
            }
        };

        let symbol_row = SymbolRow {
            file_path: sym.file_path.clone(),
            name: sym.name.clone(),
            kind: sym.kind.clone(),
            start_line: sym.start_line,
            end_line: sym.end_line,
        };

        // Step 2: find the chunk containing the symbol's start line.
        let source = backend
            .get_chunks_for_file(&sym.file_path)
            .await
            .ok() // source is best-effort — don't fail the whole call on a missing chunk
            .and_then(|chunks| {
                chunks
                    .into_iter()
                    .find(|c| c.start_line <= sym.start_line && sym.start_line <= c.end_line)
                    .map(|c| c.content)
            });

        // Step 3: import graph edges for the symbol's file.
        let imports = backend
            .get_imports(&sym.file_path)
            .await
            .unwrap_or_default();
        let all_importers = backend
            .get_importers(&sym.file_path)
            .await
            .unwrap_or_default();

        // Step 3b: function-level call graph edges.
        let callers_raw = backend
            .get_callers(&sym.file_path, &sym.name)
            .await
            .unwrap_or_default();
        let callees_raw = backend
            .get_callees(&sym.file_path, &sym.name)
            .await
            .unwrap_or_default();
        let callers: Vec<CallEdgeInfo> = callers_raw
            .iter()
            .take(20)
            .map(|e| CallEdgeInfo {
                file_path: e.caller_file.clone(),
                symbol: e.caller_symbol.clone(),
                confidence: e.confidence,
            })
            .collect();
        let callees: Vec<CallEdgeInfo> = callees_raw
            .iter()
            .take(20)
            .filter_map(|e| {
                // Only include resolved callees (callee_file + callee_symbol must be present).
                let file_path = e.callee_file.clone()?;
                let symbol = e.callee_symbol.clone()?;
                if file_path.is_empty() {
                    return None;
                }
                Some(CallEdgeInfo {
                    file_path,
                    symbol,
                    confidence: e.confidence,
                })
            })
            .collect();

        // Step 4: filter importers to test files when requested and cap lists for token efficiency.
        let test_files_raw: Vec<String> = if input.include_tests {
            all_importers
                .iter()
                .filter(|f| is_test_file_path(f))
                .cloned()
                .collect()
        } else {
            vec![]
        };
        let imported_by_raw: Vec<String> = all_importers
            .into_iter()
            .filter(|f| !is_test_file_path(f))
            .collect();
        let (imports, imports_truncated) = truncate_vec(imports, 20);
        let (imported_by, imported_by_truncated) = truncate_vec(imported_by_raw, 20);
        let (test_files, _test_files_truncated) = truncate_vec(test_files_raw, 20);

        // Step 5: role lookup — use persisted role when available; otherwise infer
        // an approximate file-level role from import graph degrees so ordinary
        // indexed repos do not surface a null role to MCP callers.
        let role: Option<String> = match backend
            .get_symbol_roles(&[sym.file_path.as_str()])
            .await
            .ok()
            .and_then(|mut m| m.remove(sym.file_path.as_str()))
        {
            Some(r) => Some(r),
            None => Some(infer_file_role(&imports, &imported_by)),
        };

        Ok(SymbolContextOutput {
            symbol: Some(symbol_row),
            match_count,
            ambiguous: match_count > 1,
            source,
            imported_by,
            imported_by_truncated,
            imports,
            imports_truncated,
            test_files,
            role,
            callers,
            callees,
        })
    }

    pub async fn find_impact_set(
        &self,
        input: FindImpactSetInput,
    ) -> anyhow::Result<ImpactSetOutput> {
        let (backend, _) = self.resolve_backend(input.project.as_deref()).await?;
        let max_depth = input.max_depth.unwrap_or(3).min(5);
        let all_importers = match backend
            .traverse_importers(&input.file_path, max_depth, None)
            .await
        {
            Ok(v) => v,
            Err(ref e) if is_uninitialized_index_error(e) => vec![],
            Err(e) => return Err(e),
        };

        let direct: Vec<String> = all_importers
            .iter()
            .filter(|(_, d)| *d == 1)
            .map(|(f, _)| f.clone())
            .collect();

        let transitive: Vec<ImpactEntry> = all_importers
            .iter()
            .filter(|(_, d)| *d > 1)
            .map(|(f, d)| ImpactEntry {
                file_path: f.clone(),
                depth: *d,
            })
            .collect();

        let tests: Vec<String> = all_importers
            .iter()
            .filter(|(f, _)| is_test_file_path(f))
            .map(|(f, _)| f.clone())
            .collect();

        Ok(ImpactSetOutput {
            file_path: input.file_path,
            direct_importers: direct,
            transitive_importers: transitive,
            affected_tests: tests,
            function_callers: vec![],
        })
    }

    pub async fn find_test_context(
        &self,
        input: FindTestContextInput,
    ) -> anyhow::Result<TestContextOutput> {
        let (backend, _) = self.resolve_backend(input.project.as_deref()).await?;
        let importers = match backend.get_importers(&input.file_path).await {
            Ok(v) => v,
            Err(ref e) if is_uninitialized_index_error(e) => vec![],
            Err(e) => return Err(e),
        };
        let test_importers: Vec<String> = importers
            .into_iter()
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

        let all_files = match backend.list_indexed_paths().await {
            Ok(v) => v,
            Err(ref e) if is_uninitialized_index_error(e) => vec![],
            Err(e) => return Err(e),
        };
        let colocated: Vec<String> = all_files
            .into_iter()
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
            session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
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
        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .with_context(|| format!("bind to {addr}"))?;
        tracing::info!("skelesearch-mcp HTTP listening on {addr}");

        axum::serve(listener, router)
            .await
            .context("HTTP server error")?;
        Ok(())
    }

    /// Build a compact repo map from indexed data. Renders a tree with file
    /// roles, symbols, and import edges, respecting the token budget.
    pub async fn get_repo_map(&self, input: GetRepoMapInput) -> anyhow::Result<String> {
        let (backend, manifest_path) = self.resolve_backend(input.project.as_deref()).await?;
        let data = backend.get_repo_map_data().await?;
        let storage_dir = Self::storage_dir_from_manifest_path(&manifest_path)?;
        let indexing = self.current_indexing_progress(&storage_dir).await;
        let refreshing = matches!(
            indexing.as_ref().map(|p| p.status.as_str()),
            Some("running")
        );
        let freshness = Self::compute_freshness_snapshot(&manifest_path, refreshing);
        let mut out = render_repo_map(&data, &input);
        if let Some(warning) = Self::repo_map_warning_prefix(&freshness) {
            out.insert_str(0, &warning);
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Repo map rendering
// ---------------------------------------------------------------------------

use skelesearch_core::{RepoMapData, RepoMapFile};

/// Render a compact text tree from indexed repo data.
fn render_repo_map(data: &RepoMapData, input: &GetRepoMapInput) -> String {
    use std::collections::BTreeMap;

    if data.files.is_empty() {
        return "No files indexed. Run index first.".to_string();
    }

    struct DirNode {
        children: BTreeMap<String, DirNode>,
        files: Vec<usize>,
    }
    impl DirNode {
        fn new() -> Self {
            Self {
                children: BTreeMap::new(),
                files: Vec::new(),
            }
        }
    }

    let mut root = DirNode::new();
    for (i, file) in data.files.iter().enumerate() {
        let parts: Vec<&str> = file.path.split('/').collect();
        let (dir_parts, _file_name) = parts.split_at(parts.len().saturating_sub(1));
        let mut node = &mut root;
        for part in dir_parts {
            node = node
                .children
                .entry(part.to_string())
                .or_insert_with(DirNode::new);
        }
        node.files.push(i);
    }

    let mut out = String::new();
    let max_chars = input.max_tokens * 4;

    out.push_str(&format!(
        "# Repo Map ({} files, {} import edges)\n\n",
        data.files.len(),
        data.import_edges.len()
    ));

    fn render_dir(
        out: &mut String,
        node: &DirNode,
        files: &[RepoMapFile],
        prefix: &str,
        include_symbols: bool,
        max_chars: usize,
    ) {
        for &idx in &node.files {
            if out.len() > max_chars {
                return;
            }
            let f = &files[idx];
            let basename = f.path.rsplit('/').next().unwrap_or(&f.path);
            let role_tag = if f.role.is_empty() {
                String::new()
            } else {
                format!(" [{}]", f.role)
            };
            out.push_str(&format!(
                "{}{}{} ({} chunks, {})\n",
                prefix, basename, role_tag, f.chunk_count, f.language
            ));
            if include_symbols {
                let limit = 15.min(f.symbols.len());
                for sym in &f.symbols[..limit] {
                    out.push_str(&format!("{}  {} {}\n", prefix, sym.kind, sym.name));
                }
                if f.symbols.len() > limit {
                    out.push_str(&format!(
                        "{}  ... +{} more\n",
                        prefix,
                        f.symbols.len() - limit
                    ));
                }
            }
        }
        for (name, child) in &node.children {
            if out.len() > max_chars {
                return;
            }
            let file_count = count_files(child);
            out.push_str(&format!("{}{}/  ({} files)\n", prefix, name, file_count));
            render_dir(
                out,
                child,
                files,
                &format!("{}  ", prefix),
                include_symbols,
                max_chars,
            );
        }
    }

    fn count_files(node: &DirNode) -> usize {
        node.files.len() + node.children.values().map(count_files).sum::<usize>()
    }

    render_dir(
        &mut out,
        &root,
        &data.files,
        "",
        input.include_symbols,
        max_chars,
    );

    if input.include_edges && !data.import_edges.is_empty() && out.len() < max_chars {
        out.push_str(&format!(
            "\n## Import Graph ({} edges)\n\n",
            data.import_edges.len()
        ));
        for (from, to) in &data.import_edges {
            if out.len() > max_chars {
                out.push_str("... truncated\n");
                break;
            }
            out.push_str(&format!("{} -> {}\n", from, to));
        }
    }

    if out.len() > max_chars {
        out.truncate(max_chars);
        out.push_str("\n... [truncated - increase max_tokens for full map]\n");
    }

    out
}

// ---------------------------------------------------------------------------
// rmcp tool declarations
// ---------------------------------------------------------------------------

#[tool_router]
impl SkeleSearchServer {
    /// Index a directory for code search. Run once, updates incrementally.
    #[tool(name = "index")]
    async fn mcp_index_codebase(
        &self,
        Parameters(input): Parameters<IndexCodebaseInput>,
    ) -> Result<String, String> {
        match self.proxy_index_codebase_via_daemon(input).await {
            Ok(out) => serde_json::to_string(&out).map_err(|e| e.to_string()),
            Err(err) => Err(self.daemon_proxy_error("index_codebase", err)),
        }
    }

    /// Check if the code index exists and is current.
    #[tool(name = "get_index_status")]
    async fn mcp_index_status(
        &self,
        Parameters(input): Parameters<IndexStatusInput>,
    ) -> Result<String, String> {
        match self.index_status(input).await {
            Ok(out) => serde_json::to_string(&out).map_err(|e| e.to_string()),
            Err(err) => Err(err.to_string()),
        }
    }

    /// Find code by concept or keyword. Auto-routes to best search strategy.
    #[tool(name = "search_code")]
    async fn mcp_smart_search(
        &self,
        Parameters(input): Parameters<SmartSearchInput>,
    ) -> Result<String, String> {
        match self.smart_search(input).await {
            Ok(out) => serde_json::to_string(&out).map_err(|e| e.to_string()),
            Err(e) => Err(self.friendly_err(e).await),
        }
    }

    /// Look up a symbol definition by exact name. Returns file path, line range, and kind. Use for 'where is X defined' questions.
    #[tool(name = "find_symbol")]
    async fn mcp_find_symbol(
        &self,
        Parameters(input): Parameters<FindSymbolInput>,
    ) -> Result<String, String> {
        match self.find_symbol(input).await {
            Ok(rows) => serde_json::to_string(&rows).map_err(|e| e.to_string()),
            Err(e) => Err(self.friendly_err(e).await),
        }
    }

    /// Find all files affected by changes to a given file. Returns direct importers,
    /// transitive importers by depth, and affected test files.
    #[tool(name = "find_dependents")]
    async fn mcp_find_impact_set(
        &self,
        Parameters(input): Parameters<FindImpactSetInput>,
    ) -> Result<String, String> {
        match self.find_impact_set(input).await {
            Ok(r) => serde_json::to_string(&r).map_err(|e| e.to_string()),
            Err(e) => Err(self.friendly_err(e).await),
        }
    }

    /// Find test files covering a source file. Returns test files that import it
    /// and colocated test files.
    #[tool(name = "find_tests")]
    async fn mcp_find_test_context(
        &self,
        Parameters(input): Parameters<FindTestContextInput>,
    ) -> Result<String, String> {
        match self.find_test_context(input).await {
            Ok(r) => serde_json::to_string(&r).map_err(|e| e.to_string()),
            Err(e) => Err(self.friendly_err(e).await),
        }
    }

    /// Return source code, import graph edges, and test files for a named symbol.
    /// One-call context bundle for agents that need to understand a symbol.
    #[tool(name = "get_symbol_info")]
    async fn mcp_get_symbol_context(
        &self,
        Parameters(input): Parameters<GetSymbolContextInput>,
    ) -> Result<String, String> {
        match self.get_symbol_context(input).await {
            Ok(out) => serde_json::to_string(&out).map_err(|e| e.to_string()),
            Err(e) => Err(self.friendly_err(e).await),
        }
    }

    /// Get a compact structural overview of the indexed codebase.
    /// Returns directory tree, file roles, symbols, and import edges.
    #[tool(name = "get_repo_map")]
    async fn mcp_get_repo_map(
        &self,
        Parameters(input): Parameters<GetRepoMapInput>,
    ) -> Result<String, String> {
        match self.get_repo_map(input).await {
            Ok(out) => Ok(out),
            Err(e) => Err(self.friendly_err(e).await),
        }
    }

    // -----------------------------------------------------------------------
    // File watcher
    // -----------------------------------------------------------------------

    /// Check the environment and project config; if watching is requested,
    /// spawn the background file watcher task.
    ///
    /// Called from `on_initialized` on every client reconnect.  The
    /// `watcher_started` AtomicBool acts as a once-guard so the watcher is
    /// started at most once per server process lifetime.
    async fn start_file_watcher_if_enabled(&self) {
        let root = match std::env::current_dir() {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!(error = %e, "watcher: cannot determine cwd, skipping");
                return;
            }
        };

        // Check env var first, then project config.
        let enabled = std::env::var("SKELESEARCH_WATCH")
            .ok()
            .map(|v| matches!(v.as_str(), "1" | "true" | "yes"))
            .unwrap_or(false)
            || Config::load(&root).map(|c| c.watch).unwrap_or(false);

        if !enabled {
            tracing::debug!(
                "watcher: disabled (set SKELESEARCH_WATCH=1 or watch=true in .skelesearch.toml)"
            );
            return;
        }

        // CAS false → true: if another call already started the watcher, skip.
        if self
            .watcher_started
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
            .is_err()
        {
            tracing::debug!("watcher: already started, skipping");
            return;
        }

        tracing::info!(path = %root.display(), "watcher: starting background file watcher");
        let server_clone = self.clone();
        tokio::spawn(run_file_watcher(server_clone, root));
    }
}

async fn maintain_daemon_session(
    daemon_client: Arc<DaemonClient<TokioDaemonConnector>>,
    mut shutdown_rx: tokio::sync::watch::Receiver<bool>,
    instance_id: Arc<str>,
) {
    let client_name = Some("skelesearch-mcp".to_string());
    let client_version = Some(env!("CARGO_PKG_VERSION").to_string());
    let mut active_session: Option<String> = None;
    let mut heartbeat_interval = Duration::from_secs(20);
    let mut registration_backoff = Duration::from_secs(1);

    loop {
        if *shutdown_rx.borrow() {
            break;
        }

        if active_session.is_none() {
            match daemon_client
                .register_client(client_name.clone(), client_version.clone())
                .await
            {
                Ok(registered) => {
                    heartbeat_interval =
                        Duration::from_secs(registered.heartbeat_interval_seconds.max(1));
                    tracing::info!(
                    instance_id = %instance_id,
                    session_id = %registered.session_id,
                    heartbeat_interval_s = registered.heartbeat_interval_seconds,
                    lease_ttl_s = registered.lease_ttl_seconds,
                    "registered daemon client session"
                                        );
                    registration_backoff = Duration::from_secs(1);
                    active_session = Some(registered.session_id);
                }
                Err(err) => {
                    let next_backoff = (registration_backoff * 2).min(Duration::from_secs(30));
                    tracing::warn!(
                    instance_id = %instance_id,
                    error = %err,
                    retry_in_s = next_backoff.as_secs(),
                    "failed to register daemon client session; retrying"
                                        );
                    registration_backoff = next_backoff;
                    tokio::select! {
                        changed = shutdown_rx.changed() => {
                            if changed.is_err() || *shutdown_rx.borrow() {
                                break;
                            }
                        }
                        _ = tokio::time::sleep(registration_backoff) => {}
                    }
                    continue;
                }
            }
        }

        let session_id = active_session
            .clone()
            .expect("session is active before heartbeat loop");
        tokio::select! {
                    changed = shutdown_rx.changed() => {
                        if changed.is_err() || *shutdown_rx.borrow() {
                            break;
                        }
                    }
                    _ = tokio::time::sleep(heartbeat_interval) => {
                        match daemon_client.heartbeat(session_id.clone()).await {
                            Ok(response) if response.acknowledged => {}
                            Ok(_) => {
                                tracing::warn!(
        instance_id = %instance_id,
        session_id = %session_id,
        "daemon heartbeat was not acknowledged; re-registering session"
                                );
                                active_session = None;
                            }
                            Err(err) => {
                                tracing::warn!(
        instance_id = %instance_id,
        session_id = %session_id,
        error = %err,
        "daemon heartbeat failed; re-registering session"
                                );
                                active_session = None;
                            }
                        }
                    }
                }
    }

    if let Some(session_id) = active_session {
        if let Err(err) = daemon_client.unregister_client(session_id.clone()).await {
            tracing::warn!(
            instance_id = %instance_id,
            session_id = %session_id,
            error = %err,
            "failed to unregister daemon client session during shutdown"
                        );
        }
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for SkeleSearchServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("skelesearch", env!("CARGO_PKG_VERSION")))
            .with_instructions(
                "skelesearch -- semantic code search for agents.\n\n\
                 Tools:\n\
                 - search_code: Find code by concept, keyword, or symbol. Primary search tool.\n\
                 - get_repo_map: Compact structural overview of the indexed codebase.\n\
                 - get_symbol_info: Source code, imports, dependents, tests, and role for a named symbol.\n\
                 - find_symbol: Look up a symbol definition by exact name.\n\
                 - find_dependents: Find all files that depend on a given file (reverse import graph).\n\
                 - find_tests: Find test files for a source file.\n\
                 - index: Index a codebase for search. Incremental.\n\
                 - get_index_status: Check if the index exists and is current.\n\n\
                 Query tips:\n\
                 - Describe what the target code DOES: \"middleware that validates JWT tokens\"\n\
                 - Include known symbol names: \"AsyncClient retry logic\"\n\
                 - Use intent: \"understand\" for structural context with graph expansion\n\
                 - Use scope: \"src/auth\" to narrow results to a directory"
            )
    }

    /// Called by rmcp after the MCP `initialized` notification (handshake complete).
    /// Triggers background auto-indexing if no index exists for the current project.
    fn on_initialized(
        &self,
        _context: NotificationContext<RoleServer>,
    ) -> impl std::future::Future<Output = ()> + Send + '_ {
        async move {
            tracing::info!("on_initialized: called");
            if self.inert_mode {
                tracing::info!(instance_id = %self.instance_id, "on_initialized: inert mode, skipping daemon session, auto-index, watcher, and health check");
                return;
            }
            self.ensure_daemon_session().await;
            self.auto_index_if_needed().await;
            self.start_file_watcher_if_enabled().await;
            let health_result = match self.resolve_backend(None).await {
                Ok((backend, _)) => Some(backend.stats().await),
                Err(err) => {
                    tracing::warn!(error = %err, "index health check: failed to resolve active backend");
                    None
                }
            };
            match health_result {
                Some(Ok(stats)) => {
                    tracing::info!(
                        indexed_files = stats.indexed_files,
                        total_chunks = stats.total_chunks,
                        "index health check passed"
                    );
                }
                Some(Err(ref e)) if is_uninitialized_index_error(e) => {
                    tracing::info!(
                        "index health check: not yet initialized (auto-indexing will handle this)"
                    );
                }
                Some(Err(e)) => {
                    let friendly = friendly_index_error(&e);
                    tracing::error!(error = %friendly, "index health check FAILED — search tools will return errors until resolved");
                }
                None => {}
            }
        }
    }
}

// ---------------------------------------------------------------------------
// File watcher task
// ---------------------------------------------------------------------------

/// Background task that watches `root` for file changes and triggers a
/// re-index on the owning server when relevant changes are detected.
///
/// # Design
/// - `notify_debouncer_full` coalescces rapid bursts of filesystem events into
///   a single callback after a 2-second quiet window.
/// - A sync-to-async bridge thread forwards debounced events to a tokio channel
///   so the main loop can be written as a clean async select.
/// - Changes inside `.skelesearch/` and `.git/` are ignored: they are written
///   by the indexer and VCS machinery, not by the user.
/// - If changes arrive while indexing is in progress, they are coalesced into
///   one queued follow-up refresh that runs after the active pass completes.
#[derive(Debug, Default, Clone, Copy)]
struct WatchRefreshQueue {
    refresh_in_flight: bool,
    followup_queued: bool,
}

impl WatchRefreshQueue {
    fn on_change_detected(&mut self) -> bool {
        if self.refresh_in_flight {
            self.followup_queued = true;
            false
        } else {
            true
        }
    }

    fn mark_refresh_triggered(&mut self) {
        self.refresh_in_flight = true;
    }

    fn mark_refresh_finished(&mut self) -> bool {
        if self.followup_queued {
            self.followup_queued = false;
            self.refresh_in_flight = true;
            true
        } else {
            self.refresh_in_flight = false;
            false
        }
    }

    fn refresh_in_flight(&self) -> bool {
        self.refresh_in_flight
    }
}

async fn trigger_watcher_refresh(
    server: &SkeleSearchServer,
    root: &std::path::Path,
) -> anyhow::Result<IndexCodebaseOutput> {
    // Determine the provider name from the persisted manifest so the
    // re-index uses the same embedding model as the original index.
    let (_backend, manifest_path) = server
        .resolve_backend(Some(root.to_string_lossy().as_ref()))
        .await?;
    let provider_name = SkeleSearchServer::persisted_provider_name_from_manifest(&manifest_path)?
        .ok_or_else(|| {
        anyhow::anyhow!(
            "watcher: cannot determine provider from active manifest at {}",
            manifest_path.display()
        )
    })?;

    server
        .proxy_index_codebase_via_daemon(IndexCodebaseInput {
            path: root.to_string_lossy().to_string(),
            provider: Some(provider_name),
        })
        .await
}

async fn run_file_watcher(server: SkeleSearchServer, root: PathBuf) {
    let skele_dir = root.join(".skelesearch");
    let git_dir = root.join(".git");

    let (sync_tx, sync_rx) = std::sync::mpsc::channel();
    let mut debouncer = match notify_debouncer_full::new_debouncer(
        std::time::Duration::from_secs(2),
        None,
        sync_tx,
    ) {
        Ok(d) => d,
        Err(e) => {
            tracing::error!(error = %e, "watcher: failed to create debouncer — watching disabled");
            // Reset the guard so a future call can retry if the watcher setup fails.
            server.watcher_started.store(false, Ordering::Release);
            return;
        }
    };

    if let Err(e) = debouncer
        .watcher()
        .watch(&root, notify::RecursiveMode::Recursive)
    {
        tracing::error!(error = %e, path = %root.display(), "watcher: failed to set watch path — watching disabled");
        server.watcher_started.store(false, Ordering::Release);
        return;
    }

    tracing::info!(path = %root.display(), "watcher: active (2 s debounce)");

    // Bridge sync mpsc → tokio unbounded channel so we can await in the loop.
    // The bridge thread exits when the debouncer drops (sync_tx closed → sync_rx returns Err).
    let (async_tx, mut async_rx) = tokio::sync::mpsc::unbounded_channel();
    std::thread::spawn(move || {
        while let Ok(event) = sync_rx.recv() {
            if async_tx.send(event).is_err() {
                break;
            }
        }
    });

    // Keep the debouncer alive for the loop's duration.
    let _debouncer = debouncer;

    let mut refresh_queue = WatchRefreshQueue::default();
    let mut status_poll = tokio::time::interval(Duration::from_millis(400));
    status_poll.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    loop {
        tokio::select! {
            _ = status_poll.tick() => {
                if !refresh_queue.refresh_in_flight() {
                    continue;
                }

                let status = server
                    .proxy_index_status_via_daemon(IndexStatusInput {
                        path: Some(root.to_string_lossy().to_string()),
                    })
                    .await;

                match status {
                    Ok(status) => {
                        let running = matches!(
                            status.indexing.as_ref().map(|p| p.status.as_str()),
                            Some("running")
                        );

                        if running {
                            continue;
                        }

                        if refresh_queue.mark_refresh_finished() {
                            tracing::info!("watcher: running queued follow-up refresh");
                            match trigger_watcher_refresh(&server, &root).await {
                                Ok(out) => {
                                    refresh_queue.mark_refresh_triggered();
                                    tracing::info!(
                                        status = %out.status,
                                        "watcher: follow-up refresh triggered"
                                    );
                                }
                                Err(e) => {
                                    tracing::error!(
                                        error = %server.daemon_proxy_error("index_codebase", e),
                                        "watcher: failed to trigger queued follow-up refresh"
                                    );
                                }
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            error = %server.daemon_proxy_error("index_status", e),
                            "watcher: failed to poll indexing status"
                        );
                    }
                }
            }
            event = async_rx.recv() => {
                match event {
                    None => break, // sender dropped; watcher gone
                    Some(Err(errs)) => {
                        for e in errs {
                            tracing::warn!(error = %e, "watcher: filesystem event error");
                        }
                    }
                    Some(Ok(events)) => {
                        let relevant = events
                            .iter()
                            .flat_map(|e| &e.paths)
                            .filter(|p| !p.starts_with(&skele_dir) && !p.starts_with(&git_dir))
                            .count();

                        if relevant == 0 {
                            continue;
                        }

                        if !refresh_queue.on_change_detected() {
                            tracing::debug!(
                                changed_files = relevant,
                                "watcher: re-index in flight; queued one follow-up refresh"
                            );
                            continue;
                        }

                        tracing::info!(
                            changed_files = relevant,
                            "watcher: triggering incremental re-index"
                        );

                        match trigger_watcher_refresh(&server, &root).await {
                            Ok(out) => {
                                refresh_queue.mark_refresh_triggered();
                                tracing::info!(
                                    status = %out.status,
                                    "watcher: re-index triggered"
                                );
                            }
                            Err(e) => tracing::error!(
                                error = %server.daemon_proxy_error("index_codebase", e),
                                "watcher: failed to trigger re-index"
                            ),
                        }
                    }
                }
            }
        }
    }

    tracing::info!(path = %root.display(), "watcher: stopped");
}

// ---------------------------------------------------------------------------
// Path utilities
// ---------------------------------------------------------------------------

/// Returns true when `dir` looks like a code project root.
///
/// Used by `auto_index_if_needed` to guard against auto-indexing from system
/// directories (`/`, `/tmp`, etc.) where an MCP server might be accidentally
/// launched without a meaningful working directory.
///
/// Checks for common project marker files/directories.  Any one marker suffices.
/// Covers Rust, JS/TS, Go, Python, Java, C/C++, and skelesearch's own config.
fn looks_like_project(dir: &std::path::Path) -> bool {
    // Reject well-known non-project roots.
    {
        let s = dir.to_string_lossy();
        if s == "/" || s.starts_with("/tmp") || s.starts_with("/var/tmp") {
            return false;
        }
    }

    const MARKERS: &[&str] = &[
        ".git",
        ".skelesearch.toml",
        "Cargo.toml",
        "package.json",
        "go.mod",
        "pyproject.toml",
        "setup.py",
        "pom.xml",
        "build.gradle",
        "CMakeLists.txt",
        "Makefile",
        "requirements.txt",
        ".hg",
        ".svn",
    ];
    MARKERS.iter().any(|m| dir.join(m).exists())
}

/// Returns true when `path` looks like a test file based on common naming conventions.
/// This is a heuristic — it covers Go, Rust, Python, JS/TS, Ruby, and directory-based conventions.
fn is_test_file_path(path: &str) -> bool {
    let lower = path.to_lowercase();
    let filename = lower.rsplit('/').next().unwrap_or(&lower);
    lower.contains("/test/")
        || lower.contains("/tests/")
        || lower.contains("/__tests__/")
        || lower.contains("/spec/")
        || lower.contains("/specs/")
        || lower.ends_with("_test.go")
        || lower.ends_with("_test.rs")
        || lower.ends_with(".test.ts")
        || lower.ends_with(".test.js")
        || lower.ends_with(".test.tsx")
        || lower.ends_with(".test.jsx")
        || lower.ends_with(".spec.ts")
        || lower.ends_with(".spec.js")
        || lower.ends_with(".spec.tsx")
        || lower.ends_with(".spec.jsx")
        || lower.ends_with("_spec.rb")
        || (filename.starts_with("test_") && filename.ends_with(".py"))
}

fn infer_file_role(imports: &[String], imported_by: &[String]) -> String {
    let in_degree = imported_by.len();
    let out_degree = imports.len();
    if in_degree >= 3 && out_degree <= 1 {
        "entry".to_string()
    } else if in_degree >= 2 && out_degree >= 2 {
        "core".to_string()
    } else if in_degree <= 1 && out_degree >= 2 {
        "utility".to_string()
    } else if imports.is_empty() {
        "leaf".to_string()
    } else {
        "internal".to_string()
    }
}

fn truncate_vec(mut items: Vec<String>, limit: usize) -> (Vec<String>, bool) {
    let truncated = items.len() > limit;
    if truncated {
        items.truncate(limit);
    }
    (items, truncated)
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

/// Recursively count files under `path`, following symlinks.
///
/// Used for a quick pre-spawn file count in `index_codebase`.  Does not
/// apply extension filters or `.gitignore` rules — the goal is a fast upper
/// bound, not a precise match of what the indexer will process.
///
/// The caller wraps this in `tokio::time::timeout` so it never blocks long.
fn count_files_recursive(path: &std::path::Path) -> usize {
    let mut count = 0usize;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                count += count_files_recursive(&p);
            } else if p.is_file() {
                count += 1;
            }
        }
    }
    count
}

#[cfg(test)]
mod startup_tests {
    use super::*;

    use std::sync::{Mutex as StdMutex, OnceLock};

    use skelesearch_service::{
        protocol::IndexCodebaseStatus, DaemonCapabilities, DaemonErrorCode, DaemonErrorResponse,
        DaemonRequest, DaemonResponse, HandshakeResponse, IndexCodebaseResponse,
        IndexFreshnessState, IndexStatusResponse, ProjectKey, ProtocolFrame,
    };
    use tempfile::TempDir;
    use tokio::{
        io::{AsyncBufReadExt as _, AsyncWriteExt as _, BufReader},
        net::UnixListener,
        sync::Mutex,
    };

    #[derive(Clone)]
    struct TestProvider;

    #[async_trait]
    impl EmbedProvider for TestProvider {
        fn dim(&self) -> usize {
            8
        }

        fn name(&self) -> &str {
            "test-provider"
        }

        async fn embed_batch(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
            Ok(texts.into_iter().map(|_| vec![0.1_f32; 8]).collect())
        }

        async fn embed_queries(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
            self.embed_batch(texts).await
        }
    }

    fn env_lock() -> &'static StdMutex<()> {
        static LOCK: OnceLock<StdMutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| StdMutex::new(()))
    }

    async fn spawn_startup_stub_daemon(
        socket_path: &std::path::Path,
        index_status: IndexStatusResponse,
        index_requests: Arc<Mutex<Vec<Option<String>>>>,
    ) -> anyhow::Result<tokio::task::JoinHandle<anyhow::Result<()>>> {
        let listener = UnixListener::bind(socket_path)?;
        Ok(tokio::spawn(async move {
            let (stream, _) = listener.accept().await?;
            let (read_half, mut write_half) = stream.into_split();
            let mut lines = BufReader::new(read_half).lines();
            while let Some(line) = lines.next_line().await? {
                let frame: ProtocolFrame = serde_json::from_str(&line)?;
                let response = match frame {
                    ProtocolFrame::Request { id, request } => match request {
                        DaemonRequest::Handshake(_) => ProtocolFrame::Response {
                            id,
                            response: DaemonResponse::Handshake(HandshakeResponse {
                                protocol_version: skelesearch_service::DAEMON_PROTOCOL_VERSION
                                    .to_string(),
                                server_name: "startup-stub".to_string(),
                                server_version: "0.1.0".to_string(),
                                capabilities: DaemonCapabilities {
                                    info: true,
                                    index_codebase: true,
                                    index_status: true,
                                    search_code: true,
                                    smart_search: false,
                                    register_client: true,
                                    heartbeat: true,
                                    unregister_client: true,
                                },
                            }),
                        },
                        DaemonRequest::IndexStatus(_) => ProtocolFrame::Response {
                            id,
                            response: DaemonResponse::IndexStatus(index_status.clone()),
                        },
                        DaemonRequest::IndexCodebase(req) => {
                            index_requests.lock().await.push(req.provider);
                            ProtocolFrame::Response {
                                id,
                                response: DaemonResponse::IndexCodebase(IndexCodebaseResponse {
                                    status: IndexCodebaseStatus::IndexingStarted,
                                    project_key: ProjectKey {
                                        canonical_root: "/tmp/repo".to_string(),
                                        logical_id: None,
                                    },
                                    files_queued: 0,
                                    message: "started".to_string(),
                                }),
                            }
                        }
                        other => ProtocolFrame::Response {
                            id,
                            response: DaemonResponse::Error(DaemonErrorResponse {
                                code: DaemonErrorCode::BadRequest,
                                message: format!("unexpected request in startup stub: {other:?}"),
                                details: None,
                                retryable: false,
                            }),
                        },
                    },
                    _ => continue,
                };
                write_half
                    .write_all(serde_json::to_string(&response)?.as_bytes())
                    .await?;
                write_half.write_all(b"\n").await?;
                write_half.flush().await?;
            }
            Ok(())
        }))
    }

    async fn startup_test_server(
        project_root: &std::path::Path,
        manifest_path: &std::path::Path,
    ) -> anyhow::Result<SkeleSearchServer> {
        std::fs::create_dir_all(project_root.join(".git"))?;
        std::fs::create_dir_all(
            manifest_path
                .parent()
                .ok_or_else(|| anyhow::anyhow!("manifest path missing parent"))?,
        )?;
        let backend = Arc::new(CozoBackend::open(
            manifest_path
                .parent()
                .expect("manifest parent")
                .join("index.db"),
        )?);
        Ok(SkeleSearchServer::new(
            backend,
            manifest_path,
            ArcProvider::new(TestProvider),
        ))
    }

    fn write_active_generation_fixture(
        project_root: &std::path::Path,
        generation_id: &str,
        provider_name: &str,
    ) -> anyhow::Result<PathBuf> {
        let storage_dir = project_root.join(".skelesearch");
        let generation_dir = storage_dir.join("generations").join(generation_id);
        std::fs::create_dir_all(&generation_dir)?;
        let (backend_path, manifest_path) = generation_db_paths(&generation_dir);
        let _backend = CozoBackend::open(&backend_path)?;
        let manifest = ManifestStore::open(&manifest_path)?;
        manifest.set_meta("provider", provider_name)?;
        std::fs::write(storage_dir.join("active-generation"), generation_id)?;
        Ok(manifest_path)
    }

    fn stale_status() -> IndexStatusResponse {
        IndexStatusResponse {
            project_key: ProjectKey {
                canonical_root: "/tmp/repo".to_string(),
                logical_id: None,
            },
            indexed_files: 3,
            total_chunks: 8,
            last_indexed: Some(chrono::Utc::now().to_rfc3339()),
            estimated_stale: 1,
            freshness_state: IndexFreshnessState::Stale,
            freshness_checked_at: Some(chrono::Utc::now().to_rfc3339()),
            freshness_error: None,
            watching: false,
            indexing: None,
        }
    }

    fn fresh_status() -> IndexStatusResponse {
        IndexStatusResponse {
            freshness_state: IndexFreshnessState::Fresh,
            estimated_stale: 0,
            ..stale_status()
        }
    }

    #[tokio::test]
    async fn startup_stale_refresh_uses_manifest_provider_metadata() -> anyhow::Result<()> {
        let _guard = env_lock().lock().expect("env lock");
        std::env::remove_var("SKELESEARCH_NO_AUTO_INDEX");

        let temp = TempDir::new()?;
        let root = temp.path().join("repo");
        std::fs::create_dir_all(&root)?;
        let manifest_path = root.join(".skelesearch/manifest.db");

        let socket_path = temp.path().join("daemon.sock");
        let index_requests = Arc::new(Mutex::new(Vec::new()));
        let daemon =
            spawn_startup_stub_daemon(&socket_path, stale_status(), Arc::clone(&index_requests))
                .await?;

        std::env::set_var("SKELESEARCH_DAEMON_SOCKET", &socket_path);
        let server = startup_test_server(&root, &manifest_path).await?;
        ManifestStore::open(&manifest_path)?.set_meta("provider", "openai")?;

        let original_dir = std::env::current_dir()?;
        std::env::set_current_dir(&root)?;
        server.auto_index_if_needed().await;
        std::env::set_current_dir(original_dir)?;

        let requests = index_requests.lock().await;
        assert_eq!(
            requests.len(),
            1,
            "expected stale startup refresh to queue one indexing run"
        );
        assert_eq!(requests[0].as_deref(), Some("openai"));

        std::env::remove_var("SKELESEARCH_DAEMON_SOCKET");
        daemon.abort();
        Ok(())
    }

    #[tokio::test]
    async fn startup_stale_refresh_fresh_index_is_noop() -> anyhow::Result<()> {
        let _guard = env_lock().lock().expect("env lock");
        std::env::remove_var("SKELESEARCH_NO_AUTO_INDEX");

        let temp = TempDir::new()?;
        let root = temp.path().join("repo");
        std::fs::create_dir_all(&root)?;
        let manifest_path = root.join(".skelesearch/manifest.db");

        let socket_path = temp.path().join("daemon.sock");
        let index_requests = Arc::new(Mutex::new(Vec::new()));
        let daemon =
            spawn_startup_stub_daemon(&socket_path, fresh_status(), Arc::clone(&index_requests))
                .await?;

        std::env::set_var("SKELESEARCH_DAEMON_SOCKET", &socket_path);
        let server = startup_test_server(&root, &manifest_path).await?;
        ManifestStore::open(&manifest_path)?.set_meta("provider", "openai")?;

        let original_dir = std::env::current_dir()?;
        std::env::set_current_dir(&root)?;
        server.auto_index_if_needed().await;
        std::env::set_current_dir(original_dir)?;

        let requests = index_requests.lock().await;
        assert!(
            requests.is_empty(),
            "fresh startup should not queue indexing"
        );

        std::env::remove_var("SKELESEARCH_DAEMON_SOCKET");
        daemon.abort();
        Ok(())
    }

    #[tokio::test]
    async fn resolve_backend_tracks_active_generation_pointer_changes() -> anyhow::Result<()> {
        let temp = TempDir::new()?;
        let root = temp.path().join("repo");
        std::fs::create_dir_all(&root)?;
        let manifest_path = root.join(".skelesearch/manifest.db");
        let server = startup_test_server(&root, &manifest_path).await?;

        let manifest_a = write_active_generation_fixture(&root, "gen-a", "openai")?;
        let (_backend_a, resolved_a) = server
            .resolve_backend(Some(root.to_string_lossy().as_ref()))
            .await?;
        assert_eq!(resolved_a, manifest_a);

        let manifest_b = write_active_generation_fixture(&root, "gen-b", "voyage")?;
        let (_backend_b, resolved_b) = server
            .resolve_backend(Some(root.to_string_lossy().as_ref()))
            .await?;
        assert_eq!(resolved_b, manifest_b);
        assert_eq!(
            ManifestStore::open(&resolved_b)?
                .get_meta("provider")?
                .as_deref(),
            Some("voyage")
        );

        Ok(())
    }

    #[tokio::test]
    async fn watcher_refresh_uses_active_generation_manifest_provider() -> anyhow::Result<()> {
        let _guard = env_lock().lock().expect("env lock");
        let temp = TempDir::new()?;
        let root = temp.path().join("repo");
        std::fs::create_dir_all(&root)?;
        let manifest_path = root.join(".skelesearch/manifest.db");

        let socket_path = temp.path().join("daemon.sock");
        let index_requests = Arc::new(Mutex::new(Vec::new()));
        let daemon =
            spawn_startup_stub_daemon(&socket_path, stale_status(), Arc::clone(&index_requests))
                .await?;

        std::env::set_var("SKELESEARCH_DAEMON_SOCKET", &socket_path);
        let server = startup_test_server(&root, &manifest_path).await?;
        let _active_manifest = write_active_generation_fixture(&root, "gen-a", "openai")?;

        trigger_watcher_refresh(&server, &root).await?;

        let requests = index_requests.lock().await;
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].as_deref(), Some("openai"));

        std::env::remove_var("SKELESEARCH_DAEMON_SOCKET");
        daemon.abort();
        Ok(())
    }

    #[tokio::test]
    async fn index_status_refreshing_preserves_last_good_stats() -> anyhow::Result<()> {
        let temp = TempDir::new()?;
        let root = temp.path().join("repo");
        std::fs::create_dir_all(&root)?;
        let manifest_path = root.join(".skelesearch/manifest.db");
        let server = startup_test_server(&root, &manifest_path).await?;

        let provider = ArcProvider::new(TestProvider);
        server.run_index(&root, provider).await?;

        let baseline = server.index_status(IndexStatusInput { path: None }).await?;
        let storage_dir = SkeleSearchServer::storage_dir_from_manifest_path(&manifest_path)?;
        {
            let mut state = server.index_state.write().await;
            state.path = root.display().to_string();
            state.storage_dir = storage_dir.display().to_string();
            state.status = IndexingStatus::Running;
            state.files_done = 1;
            state.chunks_done = 1;
            state.started_at = std::time::Instant::now();
            state.error = None;
        }

        let refreshing = server.index_status(IndexStatusInput { path: None }).await?;
        assert_eq!(
            refreshing.freshness_state,
            SkeleSearchServer::map_freshness_state(FreshnessState::Refreshing)
        );
        assert_eq!(refreshing.indexed_files, baseline.indexed_files);
        assert_eq!(refreshing.total_chunks, baseline.total_chunks);
        assert_eq!(refreshing.last_indexed, baseline.last_indexed);
        assert!(refreshing.indexing.is_some());

        Ok(())
    }

    #[tokio::test]
    async fn mcp_index_status_reports_local_watcher_state() -> anyhow::Result<()> {
        let temp = TempDir::new()?;
        let root = temp.path().join("repo");
        std::fs::create_dir_all(&root)?;
        let manifest_path = root.join(".skelesearch/manifest.db");
        let server = startup_test_server(&root, &manifest_path).await?;
        server.backend.initialize(8).await?;
        server.watcher_started.store(true, Ordering::Release);

        let output = server
            .mcp_index_status(Parameters(IndexStatusInput { path: None }))
            .await
            .map_err(|err| anyhow::anyhow!(err))?;
        let status: serde_json::Value = serde_json::from_str(&output)?;
        assert_eq!(status["watching"], true);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::WatchRefreshQueue;

    #[test]
    fn watcher_followup_refresh_coalesces_mid_run_edits() {
        let mut queue = WatchRefreshQueue::default();
        assert!(queue.on_change_detected());
        queue.mark_refresh_triggered();
        assert!(queue.refresh_in_flight());
        assert!(!queue.on_change_detected());
        assert!(!queue.on_change_detected());
        assert!(queue.mark_refresh_finished());
        assert!(queue.refresh_in_flight());
        assert!(!queue.mark_refresh_finished());
        assert!(!queue.refresh_in_flight());
    }
}
