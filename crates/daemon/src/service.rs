use std::{
    collections::HashMap,
    fs::{File, OpenOptions},
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant, SystemTime},
};

use anyhow::Context as _;
use chrono::Utc;
use skelesearch_core::{try_acquire_indexing_lease, SharedIndexingStatus};
use skelesearch_service::{
    DaemonCapabilities, DaemonErrorCode, DaemonErrorResponse, DaemonRequest, DaemonResponse,
    HandshakeRequest, HandshakeResponse, IndexCodebaseRequest, IndexCodebaseResponse,
    IndexStatusRequest, IndexStatusResponse, IndexingProgress, InfoResponse, ProjectKey,
    ProjectKeyError, ProjectTarget, DAEMON_PROTOCOL_VERSION,
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
const PLACEHOLDER_INDEX_DELAY_MS: u64 = 250;

#[derive(Debug, Clone, Default)]
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

#[derive(Debug, Clone)]
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
            DaemonRequest::SearchCode(_) => Self::unsupported_method("search_code"),
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

        let runtime = project.index_runtime_snapshot().await;
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

        let provider_name = request.provider.unwrap_or_else(|| "placeholder".to_string());
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

        project
            .mark_indexing_started(path.clone(), provider_name)
            .await;

        let project_for_task = Arc::clone(&project);
        tokio::spawn(async move {
            let started = Instant::now();
            match count_indexable_files(&project_for_task.canonical_root) {
                Ok(files_total) => {
                    project_for_task.update_running_totals(files_total).await;
                    shared_status.files_total = files_total;
                    shared_status.updated_at = Utc::now();
                    if let Err(err) = lease.write_status(&shared_status) {
                        tracing::warn!(
                            project = %project_for_task.project_key,
                            error = %err,
                            "failed to write running indexing status"
                        );
                    }

                    tokio::time::sleep(Duration::from_millis(PLACEHOLDER_INDEX_DELAY_MS)).await;

                    let elapsed = started.elapsed().as_secs_f64();
                    project_for_task
                        .mark_indexing_done(files_total, elapsed)
                        .await;

                    shared_status.status = "done".to_string();
                    shared_status.updated_at = Utc::now();
                    shared_status.files_done = files_total;
                    shared_status.chunks_done = files_total;
                    if let Err(err) = lease.write_status(&shared_status) {
                        tracing::warn!(
                            project = %project_for_task.project_key,
                            error = %err,
                            "failed to write completed indexing status"
                        );
                    }
                }
                Err(err) => {
                    let elapsed = started.elapsed().as_secs_f64();
                    let error_message = err.to_string();
                    project_for_task
                        .mark_indexing_failed(elapsed, error_message.clone())
                        .await;

                    shared_status.status = "failed".to_string();
                    shared_status.updated_at = Utc::now();
                    shared_status.error = Some(error_message);
                    if let Err(write_err) = lease.write_status(&shared_status) {
                        tracing::warn!(
                            project = %project_for_task.project_key,
                            error = %write_err,
                            "failed to write failed indexing status"
                        );
                    }
                }
            }
        });

        DaemonResponse::IndexCodebase(IndexCodebaseResponse {
            status: IndexCodebaseStatus::IndexingStarted,
            project_key: project.project_key.clone(),
            files_queued: 0,
            message: format!(
                "indexing started for '{}'; poll index_status for progress",
                project.project_key
            ),
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

#[derive(Debug)]
pub struct ProjectState {
    pub project_key: ProjectKey,
    pub canonical_root: PathBuf,
    pub storage_dir: PathBuf,
    pub manifest_path: PathBuf,
    pub backend: Arc<BackendHandle>,
    pub cached_searcher: Arc<RwLock<Option<Arc<CachedSearcherPlaceholder>>>>,
    pub provider_identity: Arc<RwLock<Option<ProviderIdentityPlaceholder>>>,
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
            *provider_identity = Some(ProviderIdentityPlaceholder {
                provider_name: provider,
            });
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

    async fn update_running_totals(&self, files_total: usize) {
        let mut runtime = self.index_progress.write().await;
        if let Some(progress) = runtime.indexing.as_mut() {
            progress.files_total = files_total;
        }
    }

    async fn mark_indexing_done(&self, files_total: usize, elapsed_seconds: f64) {
        let mut runtime = self.index_progress.write().await;
        runtime.indexed_files = files_total;
        runtime.total_chunks = files_total;
        runtime.last_indexed = Some(Utc::now().to_rfc3339());
        let path = runtime
            .indexing
            .as_ref()
            .map(|progress| progress.path.clone())
            .unwrap_or_else(|| self.canonical_root.to_string_lossy().into_owned());
        runtime.indexing = Some(IndexingProgress {
            status: IndexingState::Done,
            path,
            files_done: files_total,
            files_total,
            chunks_done: files_total,
            cache_hits: 0,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderIdentityPlaceholder {
    pub provider_name: String,
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

#[derive(Debug)]
pub struct CachedSearcherPlaceholder;

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

fn count_indexable_files(root: &Path) -> anyhow::Result<usize> {
    if !root.exists() {
        return Ok(0);
    }

    let mut count = 0usize;
    let mut stack = vec![root.to_path_buf()];

    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir)
            .with_context(|| format!("read project directory '{}'", dir.display()))?
        {
            let entry = entry.with_context(|| format!("read entry under '{}'", dir.display()))?;
            let file_type = entry
                .file_type()
                .with_context(|| format!("read file type for '{}'", entry.path().display()))?;
            let file_name = entry.file_name();
            let file_name = file_name.to_string_lossy();

            if file_type.is_dir() {
                if file_name == ".git" || file_name == STORAGE_DIR_NAME {
                    continue;
                }
                stack.push(entry.path());
                continue;
            }

            if file_type.is_file() {
                count += 1;
            }
        }
    }

    Ok(count)
}

fn daemon_capabilities() -> DaemonCapabilities {
    DaemonCapabilities {
        info: true,
        index_codebase: true,
        index_status: true,
        search_code: false,
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

        tokio::time::sleep(Duration::from_millis(PLACEHOLDER_INDEX_DELAY_MS + 100)).await;
    }
}
