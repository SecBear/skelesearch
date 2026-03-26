use std::{
    collections::HashMap,
    fs::{File, OpenOptions},
    path::{Path, PathBuf},
    sync::Arc,
    time::{Instant, SystemTime},
};

use anyhow::Context as _;
use chrono::Utc;
use async_trait::async_trait;
use skelesearch_core::{
    git::changed_files_on_branch, Config, CozoBackend, EmbedProvider, Indexer, ManifestStore,
    Searcher, SharedIndexingStatus, StorageBackend, try_acquire_indexing_lease,
};
use skelesearch_embed_fastembed::{provider_from_name, FastEmbedSparseProvider};
use skelesearch_service::{
    DaemonCapabilities, DaemonErrorCode, DaemonErrorResponse, DaemonRequest, DaemonResponse,
    HandshakeRequest, HandshakeResponse, IndexCodebaseRequest, IndexCodebaseResponse,
    IndexStatusRequest, IndexStatusResponse, IndexingProgress, InfoResponse, ProjectKey,
    ProjectKeyError, ProjectTarget, SearchCodeRequest, SearchCodeResponse, SearchResultRow,
    DAEMON_PROTOCOL_VERSION,
};
use skelesearch_service::protocol::{
    DaemonEvent, IndexCodebaseStatus, IndexingState, ProtocolErrorEvent, ProtocolFrame, RequestId,
    StreamId,
};
use tokio::sync::{Mutex, RwLock};

const STORAGE_DIR_NAME: &str = ".skelesearch";
const BACKEND_DB_FILE: &str = "index.db";
const MANIFEST_DB_FILE: &str = "manifest.db";
const INDEX_LOCK_FILE: &str = ".skelesearch.lock";
const INDEX_STATUS_FILE: &str = "indexing-status.json";


type CachedSearcher = Searcher<CozoBackend, ArcProvider>;

#[derive(Clone)]
pub struct ArcProvider(pub Arc<dyn EmbedProvider + Send + Sync>);

impl ArcProvider {
    pub fn new(p: impl EmbedProvider + Send + Sync + 'static) -> Self {
        Self(Arc::new(p))
    }
}

#[async_trait]
impl EmbedProvider for ArcProvider {
    fn dim(&self) -> usize { self.0.dim() }
    fn name(&self) -> &str { self.0.name() }
    async fn embed_batch(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> { self.0.embed_batch(texts).await }
    async fn embed_queries(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> { self.0.embed_queries(texts).await }
    fn query_prefix(&self) -> Option<&str> { self.0.query_prefix() }
}
#[derive(Clone, Default)]
pub struct DaemonState {
    projects: Arc<RwLock<HashMap<ProjectKey, Arc<ProjectState>>>>,
}

impl DaemonState {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn resolve_or_open_project(
        &self,
        target: ProjectLookup,
    ) -> anyhow::Result<Arc<ProjectState>> {
        let project_key = canonical_project_key(target)?;

        {
            let projects = self.projects.read().await;
            if let Some(project) = projects.get(&project_key) {
                project.touch_last_access().await;
                return Ok(Arc::clone(project));
            }
        }

        let mut projects = self.projects.write().await;
        if let Some(project) = projects.get(&project_key) {
            project.touch_last_access().await;
            return Ok(Arc::clone(project));
        }

        let project = Arc::new(ProjectState::open(project_key.clone())?);
        projects.insert(project_key, Arc::clone(&project));
        Ok(project)
    }

    pub async fn project_count(&self) -> usize {
        self.projects.read().await.len()
    }
}

#[derive(Clone)]
pub struct DaemonService {
    state: DaemonState,
    server_name: Arc<str>,
    server_version: Arc<str>,
    instance_id: Arc<str>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ServiceFrameOutcome {
    pub response: ProtocolFrame,
    pub events: Vec<ProtocolFrame>,
}

impl ServiceFrameOutcome {
    pub fn into_frames(self) -> impl Iterator<Item = ProtocolFrame> {
        std::iter::once(self.response).chain(self.events)
    }

    fn from_response(request_id: RequestId, response: DaemonResponse) -> Self {
        Self {
            response: ProtocolFrame::Response {
                id: request_id,
                response,
            },
            events: Vec::new(),
        }
    }

    fn protocol_error(message: impl Into<String>, request_id: Option<RequestId>) -> Self {
        Self {
            response: ProtocolFrame::Event {
                stream_id: StreamId(0),
                event: DaemonEvent::ProtocolError(ProtocolErrorEvent {
                    error: DaemonErrorResponse {
                        code: DaemonErrorCode::BadRequest,
                        message: message.into(),
                        details: None,
                        retryable: false,
                    },
                    request_id,
                }),
            },
            events: Vec::new(),
        }
    }
}

impl Default for DaemonService {
    fn default() -> Self {
        Self::new(DaemonState::new())
    }
}

impl DaemonService {
    pub fn new(state: DaemonState) -> Self {
        let server_name: Arc<str> = Arc::from("skelesearchd");
        let server_version: Arc<str> = Arc::from(env!("CARGO_PKG_VERSION"));
        let instance_id: Arc<str> = Arc::from(format!(
            "{}-{}-{}",
            server_name,
            std::process::id(),
            Utc::now().timestamp_millis()
        ));

        Self {
            state,
            server_name,
            server_version,
            instance_id,
        }
    }

    pub async fn handle_request_frame(&self, frame: ProtocolFrame) -> ServiceFrameOutcome {
        match frame {
            ProtocolFrame::Request { id, request } => self.handle_protocol_request(id, request).await,
            frame => ServiceFrameOutcome::protocol_error(
                format!(
                    "expected request frame, got '{}'",
                    frame_kind(&frame)
                ),
                None,
            ),
        }
    }

    pub async fn handle_protocol_request(
        &self,
        request_id: RequestId,
        request: DaemonRequest,
    ) -> ServiceFrameOutcome {
        let response = self.handle_request(request).await;
        ServiceFrameOutcome::from_response(request_id, response)
    }

    pub async fn handle_request(&self, request: DaemonRequest) -> DaemonResponse {
        match request {
            DaemonRequest::Handshake(request) => self.handle_handshake(request),
            DaemonRequest::Info(_) => DaemonResponse::Info(self.info_response()),
            DaemonRequest::IndexStatus(request) => self.handle_index_status(request).await,
            DaemonRequest::IndexCodebase(request) => self.handle_index_codebase(request).await,
            DaemonRequest::SearchCode(request) => self.handle_search_code(request).await,
            DaemonRequest::SmartSearch(_) => Self::unsupported_method("smart_search"),
        }
    }

    fn handle_handshake(&self, request: HandshakeRequest) -> DaemonResponse {
        if request.protocol_version != DAEMON_PROTOCOL_VERSION {
            return daemon_error(
                DaemonErrorCode::UnsupportedProtocolVersion,
                format!(
                    "unsupported protocol version '{}'; expected '{}'",
                    request.protocol_version, DAEMON_PROTOCOL_VERSION
                ),
                false,
            );
        }

        DaemonResponse::Handshake(HandshakeResponse {
            protocol_version: DAEMON_PROTOCOL_VERSION.to_string(),
            server_name: self.server_name.to_string(),
            server_version: self.server_version.to_string(),
            capabilities: daemon_capabilities(),
        })
    }

    fn info_response(&self) -> InfoResponse {
        InfoResponse {
            protocol_version: DAEMON_PROTOCOL_VERSION.to_string(),
            server_name: self.server_name.to_string(),
            server_version: self.server_version.to_string(),
            capabilities: daemon_capabilities(),
        }
    }

    async fn handle_index_status(&self, request: IndexStatusRequest) -> DaemonResponse {
        let project = match self.resolve_project(request.target).await {
            Ok(project) => project,
            Err(response) => return response,
        };

        let mut runtime = project.index_runtime_snapshot().await;
        if runtime.indexing.is_none() && runtime.indexed_files == 0 && runtime.total_chunks == 0 && runtime.last_indexed.is_none() {
            if let Ok(Some((indexed_files, total_chunks, last_indexed))) = read_persisted_index_stats(&project).await {
                runtime.indexed_files = indexed_files;
                runtime.total_chunks = total_chunks;
                runtime.last_indexed = last_indexed;
            }
        }

        DaemonResponse::IndexStatus(IndexStatusResponse {
            project_key: project.project_key.clone(),
            indexed_files: runtime.indexed_files,
            total_chunks: runtime.total_chunks,
            last_indexed: runtime.last_indexed,
            estimated_stale: runtime.estimated_stale,
            watching: runtime.watching,
            indexing: runtime.indexing,
        })
    }

    async fn handle_index_codebase(&self, request: IndexCodebaseRequest) -> DaemonResponse {
        let project = match self.resolve_project(request.target).await {
            Ok(project) => project,
            Err(response) => return response,
        };

        if project.is_indexing_running().await {
            return already_indexing_response(project.project_key.clone());
        }

        let provider_name = request.provider.unwrap_or_else(|| "fastembed".to_string());
        let now = Utc::now();
        let path = project.canonical_root.to_string_lossy().into_owned();
        let mut shared_status = SharedIndexingStatus {
            instance_id: self.instance_id.to_string(),
            pid: std::process::id(),
            path: path.clone(),
            provider: provider_name.clone(),
            trigger: "daemon_rpc".to_string(),
            status: "running".to_string(),
            started_at: now,
            updated_at: now,
            files_total: 0,
            files_done: 0,
            chunks_done: 0,
            cache_hits: 0,
            error: None,
        };

        let lease = match try_acquire_indexing_lease(&project.storage_dir, &shared_status) {
            Ok(Some(lease)) => lease,
            Ok(None) => return already_indexing_response(project.project_key.clone()),
            Err(err) => {
                return daemon_error(
                    DaemonErrorCode::IndexUnavailable,
                    format!("failed to acquire indexing lease: {err:#}"),
                    true,
                )
            }
        };

        project.mark_indexing_started(path.clone(), provider_name.clone()).await;
        project.invalidate_searcher().await;

        let project_for_task = Arc::clone(&project);
        tokio::spawn(async move {
            let started = Instant::now();
            match run_real_index(Arc::clone(&project_for_task), provider_name.clone()).await {
                Ok(index_result) => {
                    let elapsed = started.elapsed().as_secs_f64();
                    project_for_task
                        .mark_indexing_done(
                            index_result.indexed_files,
                            index_result.total_chunks,
                            index_result.cache_hits,
                            elapsed,
                        )
                        .await;
                    project_for_task.invalidate_searcher().await;

                    shared_status.status = "done".to_string();
                    shared_status.updated_at = Utc::now();
                    shared_status.files_total = index_result.indexed_files;
                    shared_status.files_done = index_result.indexed_files;
                    shared_status.chunks_done = index_result.total_chunks;
                    shared_status.cache_hits = index_result.cache_hits;
                    if let Err(err) = lease.write_status(&shared_status) {
                        tracing::warn!(project = %project_for_task.project_key, error = %err, "failed to write completed indexing status");
                    }
                }
                Err(err) => {
                    let elapsed = started.elapsed().as_secs_f64();
                    let error_message = err.to_string();
                    project_for_task.mark_indexing_failed(elapsed, error_message.clone()).await;
                    project_for_task.invalidate_searcher().await;

                    shared_status.status = "failed".to_string();
                    shared_status.updated_at = Utc::now();
                    shared_status.error = Some(error_message);
                    if let Err(write_err) = lease.write_status(&shared_status) {
                        tracing::warn!(project = %project_for_task.project_key, error = %write_err, "failed to write failed indexing status");
                    }
                }
            }
        });

        DaemonResponse::IndexCodebase(IndexCodebaseResponse {
            status: IndexCodebaseStatus::IndexingStarted,
            project_key: project.project_key.clone(),
            files_queued: 0,
            message: format!("indexing started for '{}'; poll index_status for progress", project.project_key),
        })
    }

    async fn handle_search_code(&self, request: SearchCodeRequest) -> DaemonResponse {
        let project = match self.resolve_project(request.target).await {
            Ok(project) => project,
            Err(response) => return response,
        };

        if project.is_indexing_running().await {
            return daemon_error(
                DaemonErrorCode::IndexUnavailable,
                format!("index is being built for '{}'; poll index_status to check progress", project.project_key),
                true,
            );
        }

        let searcher = match get_or_build_searcher(&project).await {
            Ok(searcher) => searcher,
            Err(err) => {
                return daemon_error(
                    DaemonErrorCode::IndexUnavailable,
                    format!("failed to build searcher for '{}': {err:#}", project.project_key),
                    true,
                )
            }
        };

        let top_k = request.top_k.max(1);
        let max_tokens = request.max_tokens.or(Some(8192));
        let max_depth = request.max_depth.unwrap_or(if request.include_graph { 2 } else { 0 });
        let search_result = searcher
            .search_with_timings(
                &request.query,
                top_k,
                request.include_graph,
                max_depth,
                request.diversity,
                max_tokens,
            )
            .await;

        let (mut results, _timings) = match search_result {
            Ok(v) => v,
            Err(err) => {
                return daemon_error(
                    DaemonErrorCode::IndexUnavailable,
                    format!("search failed for '{}': {err:#}", project.project_key),
                    true,
                )
            }
        };

        if request.branch_scope {
            match changed_files_on_branch(&project.canonical_root) {
                Ok(changed) if !changed.is_empty() => {
                    results.retain(|r| changed.iter().any(|c| r.file_path.ends_with(c.as_str()) || c.ends_with(&r.file_path)));
                }
                Ok(_) => {}
                Err(err) => {
                    tracing::warn!(project = %project.project_key, error = %err, "branch-scope filtering failed; returning unfiltered results");
                }
            }
        }

        DaemonResponse::SearchCode(SearchCodeResponse {
            project_key: project.project_key.clone(),
            results: results
                .into_iter()
                .map(|r| SearchResultRow {
                    file_path: r.file_path,
                    start_line: r.start_line,
                    end_line: r.end_line,
                    content: r.content,
                    score: r.score,
                    match_quality: r.match_quality,
                    why: r.why,
                })
                .collect(),
        })
    }

    async fn resolve_project(&self, target: ProjectTarget) -> Result<Arc<ProjectState>, DaemonResponse> {
        self.state
            .resolve_or_open_project(target.into())
            .await
            .map_err(|err| daemon_error(DaemonErrorCode::BadRequest, format!("invalid project target: {err:#}"), false))
    }

    fn unsupported_method(method: &str) -> DaemonResponse {
        daemon_error(
            DaemonErrorCode::BadRequest,
            format!("method '{method}' is not implemented by skelesearchd in this phase"),
            false,
        )
    }
}

async fn read_persisted_index_stats(
    project: &ProjectState,
 ) -> anyhow::Result<Option<(usize, usize, Option<String>)>> {
    let backend = Arc::new(CozoBackend::open(&project.backend.index_db_path)?);
    match backend.stats().await {
        Ok(stats) => Ok(Some((
            stats.indexed_files,
            stats.total_chunks,
            stats.last_indexed.map(|dt: chrono::DateTime<chrono::Utc>| dt.to_rfc3339()),
        ))),
        Err(err) if err.to_string().to_lowercase().contains("stored relation") && err.to_string().to_lowercase().contains("not found") => Ok(None),
        Err(err) => Err(err),
    }
}

fn provider_name_for_project(project: &ProjectState) -> anyhow::Result<String> {
    let manifest = ManifestStore::open(project.manifest_path.as_path())
        .context("failed to open manifest")?;
    Ok(manifest
        .get_meta("provider")
        .context("failed to read provider from manifest")?
        .unwrap_or_else(|| "fastembed".to_string()))
}

fn build_arc_provider(provider_name: &str) -> anyhow::Result<ArcProvider> {
    let boxed: Box<dyn EmbedProvider + Send + Sync> = provider_from_name(provider_name)
        .with_context(|| format!("failed to initialize provider '{provider_name}'"))?;
    Ok(ArcProvider(Arc::from(boxed)))
}

async fn build_searcher_for_project(project: &ProjectState) -> anyhow::Result<Arc<CachedSearcher>> {
    let provider_name = provider_name_for_project(project)?;
    let provider = build_arc_provider(&provider_name)?;
    {
        let mut guard = project.provider_identity.write().await;
        *guard = Some(provider_name.clone());
    }
    let backend = Arc::new(CozoBackend::open(&project.backend.index_db_path)?);
    let root = project.canonical_root.clone();
    let config = Config::load(&root).unwrap_or_default();
    let searcher = Searcher::new(backend, provider);
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
                Err(err) => {
                    tracing::warn!(project = %project.project_key, error = %err, "sparse provider init failed in daemon search; skipping sparse search");
                    searcher
                }
            }
        } else {
            searcher
        }
    };
    Ok(Arc::new(searcher))
}

async fn get_or_build_searcher(project: &ProjectState) -> anyhow::Result<Arc<CachedSearcher>> {
    {
        let guard = project.cached_searcher.read().await;
        if let Some(searcher) = guard.as_ref() {
            return Ok(Arc::clone(searcher));
        }
    }

    let mut guard = project.cached_searcher.write().await;
    if let Some(searcher) = guard.as_ref() {
        return Ok(Arc::clone(searcher));
    }
    let searcher = build_searcher_for_project(project).await?;
    *guard = Some(Arc::clone(&searcher));
    Ok(searcher)
}

async fn run_real_index(project: Arc<ProjectState>, provider_name: String) -> anyhow::Result<skelesearch_core::IndexResult> {
    let backend_path = project.backend.index_db_path.clone();
    let manifest_path = project.manifest_path.clone();
    let root = project.canonical_root.clone();
    tokio::task::spawn_blocking(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| anyhow::anyhow!("runtime build: {e}"))?;
        rt.block_on(async move {
            let backend = Arc::new(CozoBackend::open(&backend_path)?);
            let manifest = Arc::new(ManifestStore::open(&manifest_path)?);
            let provider = build_arc_provider(&provider_name)?;
            let config = Config::load(&root).context("load .skelesearch.toml")?;
            let indexer = Indexer::new(backend, manifest, provider)
                .with_excludes(config.index.exclude.clone())
                .with_include_extensions(config.index.include_extensions.clone())
                .with_scope_prefix(config.index.scope_prefix);
            let indexer = if config.search.sparse.enabled {
                match FastEmbedSparseProvider::bgem3() {
                    Ok(sp) => indexer.with_sparse_provider(Arc::new(sp)),
                    Err(err) => {
                        tracing::warn!(error = %err, "sparse provider init failed in daemon index; skipping sparse indexing");
                        indexer
                    }
                }
            } else {
                indexer
            };
            indexer.index_path(&root).await
        })
    })
    .await
    .context("daemon index task panicked")?
}

#[derive(Debug, Clone)]
pub enum ProjectLookup {
    ProjectKey(ProjectKey),
    RootPath {
        root_path: PathBuf,
        logical_id: Option<String>,
    },
}

impl ProjectLookup {
    pub fn from_root_path(root_path: impl Into<PathBuf>) -> Self {
        Self::RootPath {
            root_path: root_path.into(),
            logical_id: None,
        }
    }
}

impl From<ProjectTarget> for ProjectLookup {
    fn from(target: ProjectTarget) -> Self {
        match target {
            ProjectTarget::ProjectKey { project_key } => Self::ProjectKey(project_key),
            ProjectTarget::RootPath {
                root_path,
                logical_id,
            } => Self::RootPath {
                root_path: PathBuf::from(root_path),
                logical_id,
            },
        }
    }
}

pub struct ProjectState {
    pub project_key: ProjectKey,
    pub canonical_root: PathBuf,
    pub storage_dir: PathBuf,
    pub manifest_path: PathBuf,
    pub backend: Arc<BackendHandle>,
    pub cached_searcher: Arc<RwLock<Option<Arc<CachedSearcher>>>>,
    pub provider_identity: Arc<RwLock<Option<String>>>,
    pub index_progress: Arc<RwLock<ProjectIndexRuntime>>,
    pub coordination: CoordinationStatePlaceholder,
    manifest: Arc<ManifestHandle>,
    last_access: Arc<Mutex<SystemTime>>,
}

impl ProjectState {
    fn open(project_key: ProjectKey) -> anyhow::Result<Self> {
        let canonical_root = PathBuf::from(&project_key.canonical_root);
        let storage_dir = storage_dir_for_root(&canonical_root);
        std::fs::create_dir_all(&storage_dir).with_context(|| {
            format!(
                "create daemon project storage directory '{}'",
                storage_dir.display()
            )
        })?;

        let manifest_path = storage_dir.join(MANIFEST_DB_FILE);
        let backend = Arc::new(BackendHandle::open(storage_dir.join(BACKEND_DB_FILE))?);
        let manifest = Arc::new(ManifestHandle::open(manifest_path.clone())?);

        Ok(Self {
            project_key,
            canonical_root,
            storage_dir: storage_dir.clone(),
            manifest_path,
            backend,
            cached_searcher: Arc::new(RwLock::new(None)),
            provider_identity: Arc::new(RwLock::new(None)),
            index_progress: Arc::new(RwLock::new(ProjectIndexRuntime::default())),
            coordination: CoordinationStatePlaceholder::for_storage_dir(&storage_dir),
            manifest,
            last_access: Arc::new(Mutex::new(SystemTime::now())),
        })
    }

    pub async fn touch_last_access(&self) {
        let mut guard = self.last_access.lock().await;
        *guard = SystemTime::now();
    }

    pub async fn last_accessed_at(&self) -> SystemTime {
        *self.last_access.lock().await
    }

    pub fn manifest_handle(&self) -> Arc<ManifestHandle> {
        Arc::clone(&self.manifest)
    }

    pub async fn index_runtime_snapshot(&self) -> ProjectIndexRuntime {
        self.index_progress.read().await.clone()
    }

    pub async fn is_indexing_running(&self) -> bool {
        let progress = self.index_progress.read().await;
        matches!(
            progress.indexing.as_ref().map(|item| &item.status),
            Some(IndexingState::Running)
        )
    }

    async fn mark_indexing_started(&self, path: String, provider: String) {
        {
            let mut provider_identity = self.provider_identity.write().await;
            *provider_identity = Some(provider);
        }

        let mut runtime = self.index_progress.write().await;
        runtime.indexing = Some(IndexingProgress {
            status: IndexingState::Running,
            path,
            files_done: 0,
            files_total: 0,
            chunks_done: 0,
            cache_hits: 0,
            elapsed_seconds: 0.0,
            error: None,
        });
    }


    async fn mark_indexing_done(
        &self,
        indexed_files: usize,
        total_chunks: usize,
        cache_hits: usize,
        elapsed_seconds: f64,
    ) {
        let mut runtime = self.index_progress.write().await;
        runtime.indexed_files = indexed_files;
        runtime.total_chunks = total_chunks;
        runtime.last_indexed = Some(Utc::now().to_rfc3339());
        let path = runtime
            .indexing
            .as_ref()
            .map(|progress| progress.path.clone())
            .unwrap_or_else(|| self.canonical_root.to_string_lossy().into_owned());
        runtime.indexing = Some(IndexingProgress {
            status: IndexingState::Done,
            path,
            files_done: indexed_files,
            files_total: indexed_files,
            chunks_done: total_chunks,
            cache_hits,
            elapsed_seconds,
            error: None,
        });
    }

    async fn mark_indexing_failed(&self, elapsed_seconds: f64, error: String) {
        let mut runtime = self.index_progress.write().await;
        let path = runtime
            .indexing
            .as_ref()
            .map(|progress| progress.path.clone())
            .unwrap_or_else(|| self.canonical_root.to_string_lossy().into_owned());
        runtime.indexing = Some(IndexingProgress {
            status: IndexingState::Failed,
            path,
            files_done: 0,
            files_total: 0,
            chunks_done: 0,
            cache_hits: 0,
            elapsed_seconds,
            error: Some(error),
        });
    }

    async fn invalidate_searcher(&self) {
        let mut guard = self.cached_searcher.write().await;
        *guard = None;
    }
}

#[derive(Debug)]
pub struct BackendHandle {
    pub index_db_path: PathBuf,
    _index_db_file: File,
}

impl BackendHandle {
    fn open(index_db_path: PathBuf) -> anyhow::Result<Self> {
        let file = open_or_create_rw(&index_db_path)
            .with_context(|| format!("open backend file '{}'", index_db_path.display()))?;
        Ok(Self {
            index_db_path,
            _index_db_file: file,
        })
    }
}

#[derive(Debug)]
pub struct ManifestHandle {
    pub manifest_db_path: PathBuf,
    _manifest_db_file: File,
}

impl ManifestHandle {
    fn open(manifest_db_path: PathBuf) -> anyhow::Result<Self> {
        let file = open_or_create_rw(&manifest_db_path)
            .with_context(|| format!("open manifest file '{}'", manifest_db_path.display()))?;
        Ok(Self {
            manifest_db_path,
            _manifest_db_file: file,
        })
    }
}


#[derive(Debug, Clone, PartialEq, Default)]
pub struct ProjectIndexRuntime {
    pub indexed_files: usize,
    pub total_chunks: usize,
    pub last_indexed: Option<String>,
    pub estimated_stale: usize,
    pub watching: bool,
    pub indexing: Option<IndexingProgress>,
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoordinationStatePlaceholder {
    pub lock_path: PathBuf,
    pub status_path: PathBuf,
}

impl CoordinationStatePlaceholder {
    fn for_storage_dir(storage_dir: &Path) -> Self {
        Self {
            lock_path: storage_dir.join(INDEX_LOCK_FILE),
            status_path: storage_dir.join(INDEX_STATUS_FILE),
        }
    }
}


fn canonical_project_key(target: ProjectLookup) -> Result<ProjectKey, ProjectKeyError> {
    match target {
        ProjectLookup::ProjectKey(project_key) => ProjectKey::from_root_path_with_logical_id(
            PathBuf::from(project_key.canonical_root),
            project_key.logical_id,
        ),
        ProjectLookup::RootPath {
            root_path,
            logical_id,
        } => ProjectKey::from_root_path_with_logical_id(root_path, logical_id),
    }
}

fn storage_dir_for_root(root: &Path) -> PathBuf {
    root.join(STORAGE_DIR_NAME)
}

fn open_or_create_rw(path: &Path) -> anyhow::Result<File> {
    OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(path)
        .with_context(|| format!("open project state file '{}'", path.display()))
}


fn daemon_capabilities() -> DaemonCapabilities {
    DaemonCapabilities {
        info: true,
        index_codebase: true,
        index_status: true,
        search_code: true,
        smart_search: false,
    }
}

fn already_indexing_response(project_key: ProjectKey) -> DaemonResponse {
    DaemonResponse::IndexCodebase(IndexCodebaseResponse {
        status: IndexCodebaseStatus::AlreadyIndexing,
        project_key,
        files_queued: 0,
        message: "indexing already in progress for this project; poll index_status".to_string(),
    })
}

fn daemon_error(code: DaemonErrorCode, message: String, retryable: bool) -> DaemonResponse {
    DaemonResponse::Error(DaemonErrorResponse {
        code,
        message,
        details: None,
        retryable,
    })
}

fn frame_kind(frame: &ProtocolFrame) -> &'static str {
    match frame {
        ProtocolFrame::Request { .. } => "request",
        ProtocolFrame::Response { .. } => "response",
        ProtocolFrame::Event { .. } => "event",
        ProtocolFrame::Cancel { .. } => "cancel",
        ProtocolFrame::Ping => "ping",
        ProtocolFrame::Pong => "pong",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn repeated_lookup_reuses_same_registry_entry() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = temp.path().join("repo");
        std::fs::create_dir_all(&root).expect("create root");

        let state = DaemonState::new();
        let first = state
            .resolve_or_open_project(ProjectLookup::from_root_path(&root))
            .await
            .expect("first project lookup");
        let second = state
            .resolve_or_open_project(ProjectLookup::from_root_path(root.join(".")))
            .await
            .expect("second project lookup");

        assert_eq!(first.project_key, second.project_key);
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(state.project_count().await, 1);
    }

    #[tokio::test]
    async fn distinct_roots_create_distinct_registry_entries() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root_a = temp.path().join("repo-a");
        let root_b = temp.path().join("repo-b");
        std::fs::create_dir_all(&root_a).expect("create root_a");
        std::fs::create_dir_all(&root_b).expect("create root_b");

        let state = DaemonState::new();
        let project_a = state
            .resolve_or_open_project(ProjectLookup::from_root_path(&root_a))
            .await
            .expect("resolve project_a");
        let project_b = state
            .resolve_or_open_project(ProjectLookup::from_root_path(&root_b))
            .await
            .expect("resolve project_b");

        assert_ne!(project_a.project_key, project_b.project_key);
        assert_eq!(state.project_count().await, 2);
    }

    #[tokio::test]
    async fn index_status_defaults_before_first_index() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = temp.path().join("repo");
        std::fs::create_dir_all(&root).expect("create root");

        let service = DaemonService::default();
        let response = service
            .handle_request(DaemonRequest::IndexStatus(IndexStatusRequest {
                target: ProjectTarget::RootPath {
                    root_path: root.to_string_lossy().into_owned(),
                    logical_id: None,
                },
            }))
            .await;

        let DaemonResponse::IndexStatus(status) = response else {
            panic!("expected index status response");
        };

        assert_eq!(status.indexed_files, 0);
        assert_eq!(status.total_chunks, 0);
        assert_eq!(status.last_indexed, None);
        assert_eq!(status.estimated_stale, 0);
        assert!(!status.watching);
        assert!(status.indexing.is_none());
    }

    #[tokio::test]
    async fn duplicate_index_requests_return_already_indexing() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = temp.path().join("repo");
        std::fs::create_dir_all(&root).expect("create root");

        let service = DaemonService::default();
        let request = DaemonRequest::IndexCodebase(IndexCodebaseRequest {
            target: ProjectTarget::RootPath {
                root_path: root.to_string_lossy().into_owned(),
                logical_id: None,
            },
            provider: None,
        });

        let first = service.handle_request(request.clone()).await;
        let second = service.handle_request(request).await;

        let DaemonResponse::IndexCodebase(first) = first else {
            panic!("expected first index response");
        };
        let DaemonResponse::IndexCodebase(second) = second else {
            panic!("expected second index response");
        };

        assert_eq!(first.status, IndexCodebaseStatus::IndexingStarted);
        assert_eq!(second.status, IndexCodebaseStatus::AlreadyIndexing);

    }

    #[tokio::test]
    async fn search_code_returns_index_unavailable_for_unindexed_project() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = temp.path().join("repo");
        std::fs::create_dir_all(&root).expect("create root");

        let service = DaemonService::default();
        let response = service
            .handle_request(DaemonRequest::SearchCode(SearchCodeRequest {
                target: ProjectTarget::RootPath {
                    root_path: root.to_string_lossy().into_owned(),
                    logical_id: None,
                },
                query: "find main".to_string(),
                top_k: 3,
                include_graph: false,
                max_depth: None,
                diversity: 0.3,
                max_tokens: Some(1024),
                branch_scope: false,
                session_id: None,
            }))
            .await;

        let DaemonResponse::Error(err) = response else {
            panic!("expected daemon error response");
        };
        assert_eq!(err.code, DaemonErrorCode::IndexUnavailable);
        assert!(err.retryable);
    }
}
