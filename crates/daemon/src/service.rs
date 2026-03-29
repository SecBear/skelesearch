use std::{
    collections::HashMap,
    fs::{File, OpenOptions},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant, SystemTime},
};

use anyhow::Context as _;
use async_trait::async_trait;
use chrono::Utc;
use skelesearch_core::{
    generation_db_paths, git::changed_files_on_branch, preferred_index_provider_name,
    try_acquire_indexing_lease, write_file_atomic, CompositeBackend, Config, EmbedProvider,
    FreshnessSnapshot as CoreFreshnessSnapshot, FreshnessState as CoreFreshnessState, Indexer,
    ManifestStore, Searcher, SharedIndexingStatus, StorageBackend,
};
use skelesearch_embed_fastembed::{provider_from_name, FastEmbedSparseProvider};
use skelesearch_service::protocol::{
    DaemonEvent, IndexCodebaseStatus, IndexingState, ProtocolErrorEvent, ProtocolFrame, RequestId,
    StreamId,
};
use skelesearch_service::{
    DaemonCapabilities, DaemonErrorCode, DaemonErrorResponse, DaemonRequest, DaemonResponse,
    HandshakeRequest, HandshakeResponse, HeartbeatRequest, HeartbeatResponse, IndexCodebaseRequest,
    IndexCodebaseResponse, IndexFreshnessState, IndexStatusRequest, IndexStatusResponse,
    IndexingProgress, InfoResponse, ProjectKey, ProjectKeyError, ProjectTarget,
    RegisterClientRequest, RegisterClientResponse, SearchCodeRequest, SearchCodeResponse,
    SearchResultRow, UnregisterClientRequest, UnregisterClientResponse, DAEMON_PROTOCOL_VERSION,
};
use tokio::sync::{Mutex, RwLock};

const STORAGE_DIR_NAME: &str = ".skelesearch";
const INDEX_LOCK_FILE: &str = ".skelesearch.lock";
const INDEX_STATUS_FILE: &str = "indexing-status.json";
const ACTIVE_GENERATION_FILE: &str = "active-generation";
const GENERATIONS_DIR: &str = "generations";

const CLIENT_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(20);
const CLIENT_LEASE_TTL: Duration = Duration::from_secs(90);

#[derive(Debug, Clone)]
struct ClientSession {
    session_id: String,
    client_name: Option<String>,
    client_version: Option<String>,
    last_seen: Instant,
}

type CachedSearcher = Searcher<CompositeBackend, ArcProvider>;

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
#[derive(Clone)]
pub struct DaemonState {
    projects: Arc<RwLock<HashMap<ProjectKey, Arc<ProjectState>>>>,
    sessions: Arc<RwLock<HashMap<String, ClientSession>>>,
    next_session_id: Arc<AtomicU64>,
    heartbeat_interval: Duration,
    lease_ttl: Duration,
}

impl Default for DaemonState {
    fn default() -> Self {
        Self::new_with_session_timing(CLIENT_HEARTBEAT_INTERVAL, CLIENT_LEASE_TTL)
    }
}

impl DaemonState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn new_with_session_timing(heartbeat_interval: Duration, lease_ttl: Duration) -> Self {
        Self {
            projects: Arc::new(RwLock::new(HashMap::new())),
            sessions: Arc::new(RwLock::new(HashMap::new())),
            next_session_id: Arc::new(AtomicU64::new(1)),
            heartbeat_interval,
            lease_ttl,
        }
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

        let project = Arc::new(ProjectState::open(project_key.clone()).await?);
        projects.insert(project_key, Arc::clone(&project));
        Ok(project)
    }

    pub async fn project_count(&self) -> usize {
        self.projects.read().await.len()
    }

    pub async fn live_session_count(&self) -> usize {
        self.sessions.read().await.len()
    }

    pub async fn any_indexing_running(&self) -> bool {
        let projects: Vec<Arc<ProjectState>> =
            { self.projects.read().await.values().cloned().collect() };
        for project in projects {
            if project.is_indexing_running().await {
                return true;
            }
        }
        false
    }

    pub async fn register_client(
        &self,
        client_name: Option<String>,
        client_version: Option<String>,
    ) -> RegisterClientResponse {
        let next_id = self.next_session_id.fetch_add(1, Ordering::Relaxed);
        let session_id = format!("session-{next_id}");
        let session = ClientSession {
            session_id: session_id.clone(),
            client_name,
            client_version,
            last_seen: Instant::now(),
        };
        self.sessions
            .write()
            .await
            .insert(session_id.clone(), session);
        RegisterClientResponse {
            session_id,
            heartbeat_interval_seconds: self.heartbeat_interval.as_secs(),
            lease_ttl_seconds: self.lease_ttl.as_secs(),
        }
    }

    pub async fn heartbeat(&self, session_id: &str) -> HeartbeatResponse {
        let mut sessions = self.sessions.write().await;
        let acknowledged = if let Some(session) = sessions.get_mut(session_id) {
            session.last_seen = Instant::now();
            true
        } else {
            false
        };
        HeartbeatResponse {
            session_id: session_id.to_string(),
            acknowledged,
        }
    }

    pub async fn unregister_client(&self, session_id: &str) -> UnregisterClientResponse {
        let removed = self.sessions.write().await.remove(session_id).is_some();
        UnregisterClientResponse {
            session_id: session_id.to_string(),
            removed,
        }
    }

    pub async fn reap_expired_sessions(&self) -> usize {
        let now = Instant::now();
        let mut sessions = self.sessions.write().await;
        let mut expired = Vec::new();
        sessions.retain(|session_id, session| {
            let alive = now.duration_since(session.last_seen) <= self.lease_ttl;
            if !alive {
                tracing::info!(
                session_id = %session.session_id,
                client_name = ?session.client_name,
                client_version = ?session.client_version,
                "expiring daemon client session"
                                );
                expired.push(session_id.clone());
            }
            alive
        });
        expired.len()
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
            ProtocolFrame::Request { id, request } => {
                self.handle_protocol_request(id, request).await
            }
            frame => ServiceFrameOutcome::protocol_error(
                format!("expected request frame, got '{}'", frame_kind(&frame)),
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
            DaemonRequest::RegisterClient(request) => self.handle_register_client(request).await,
            DaemonRequest::Heartbeat(request) => self.handle_heartbeat(request).await,
            DaemonRequest::UnregisterClient(request) => {
                self.handle_unregister_client(request).await
            }
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

    async fn handle_register_client(&self, request: RegisterClientRequest) -> DaemonResponse {
        DaemonResponse::RegisterClient(
            self.state
                .register_client(request.client_name, request.client_version)
                .await,
        )
    }

    async fn handle_heartbeat(&self, request: HeartbeatRequest) -> DaemonResponse {
        DaemonResponse::Heartbeat(self.state.heartbeat(&request.session_id).await)
    }

    async fn handle_unregister_client(&self, request: UnregisterClientRequest) -> DaemonResponse {
        DaemonResponse::UnregisterClient(self.state.unregister_client(&request.session_id).await)
    }

    async fn handle_index_status(&self, request: IndexStatusRequest) -> DaemonResponse {
        let project = match self.resolve_project(request.target).await {
            Ok(project) => project,
            Err(response) => return response,
        };

        self.ensure_index_current_on_startup(Arc::clone(&project))
            .await;

        let mut runtime = project.index_runtime_snapshot().await;
        if runtime.indexed_files == 0 && runtime.total_chunks == 0 && runtime.last_indexed.is_none()
        {
            if let Ok(Some((indexed_files, total_chunks, last_indexed))) =
                read_persisted_index_stats(&project).await
            {
                runtime.indexed_files = indexed_files;
                runtime.total_chunks = total_chunks;
                runtime.last_indexed = last_indexed;
            }
        }

        let freshness = freshness_snapshot_for_project(
            &project,
            matches!(
                runtime.indexing.as_ref().map(|progress| &progress.status),
                Some(IndexingState::Running)
            ),
        )
        .await;
        runtime.estimated_stale = freshness.estimated_stale;
        runtime.freshness_state = protocol_freshness_state(freshness.state);
        runtime.freshness_checked_at = freshness
            .freshness_checked_at
            .map(|checked_at| checked_at.to_rfc3339());
        runtime.freshness_error = freshness.freshness_error;

        DaemonResponse::IndexStatus(IndexStatusResponse {
            project_key: project.project_key.clone(),
            indexed_files: runtime.indexed_files,
            total_chunks: runtime.total_chunks,
            last_indexed: runtime.last_indexed,
            estimated_stale: runtime.estimated_stale,
            freshness_state: runtime.freshness_state,
            freshness_checked_at: runtime.freshness_checked_at,
            freshness_error: runtime.freshness_error,
            watching: runtime.watching,
            indexing: runtime.indexing,
        })
    }

    async fn handle_index_codebase(&self, request: IndexCodebaseRequest) -> DaemonResponse {
        let project = match self.resolve_project(request.target).await {
            Ok(project) => project,
            Err(response) => return response,
        };
        let provider_name = request
            .provider
            .unwrap_or_else(|| preferred_index_provider_name().to_string());
        match self
            .start_indexing_run(Arc::clone(&project), provider_name, "daemon_rpc")
            .await
        {
            Ok(IndexStartOutcome::AlreadyIndexing) => {
                return already_indexing_response(project.project_key.clone())
            }
            Ok(IndexStartOutcome::Started) => {}
            Err(err) => {
                return daemon_error(
                    DaemonErrorCode::IndexUnavailable,
                    format!("failed to start indexing run: {err:#}"),
                    true,
                )
            }
        }

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

    async fn handle_search_code(&self, request: SearchCodeRequest) -> DaemonResponse {
        let project = match self.resolve_project(request.target).await {
            Ok(project) => project,
            Err(response) => return response,
        };

        let searcher = match get_or_build_searcher(&project).await {
            Ok(searcher) => searcher,
            Err(err) => {
                return daemon_error(
                    DaemonErrorCode::IndexUnavailable,
                    format!(
                        "failed to build searcher for '{}': {err:#}",
                        project.project_key
                    ),
                    true,
                )
            }
        };

        let top_k = request.top_k.max(1);
        let max_tokens = request.max_tokens.or(Some(8192));
        let max_depth = request
            .max_depth
            .unwrap_or(if request.include_graph { 2 } else { 0 });
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
                    results.retain(|r| {
                        changed
                            .iter()
                            .any(|c| r.file_path.ends_with(c.as_str()) || c.ends_with(&r.file_path))
                    });
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

    async fn resolve_project(
        &self,
        target: ProjectTarget,
    ) -> Result<Arc<ProjectState>, DaemonResponse> {
        self.state
            .resolve_or_open_project(target.into())
            .await
            .map_err(|err| {
                daemon_error(
                    DaemonErrorCode::BadRequest,
                    format!("invalid project target: {err:#}"),
                    false,
                )
            })
    }

    fn unsupported_method(method: &str) -> DaemonResponse {
        daemon_error(
            DaemonErrorCode::BadRequest,
            format!("method '{method}' is not implemented by skelesearchd in this phase"),
            false,
        )
    }

    async fn ensure_index_current_on_startup(&self, project: Arc<ProjectState>) {
        if std::env::var("SKELESEARCH_NO_AUTO_INDEX").is_ok() {
            tracing::debug!(
                project = %project.project_key,
                "startup ensure-current skipped: SKELESEARCH_NO_AUTO_INDEX is set"
            );
            return;
        }

        if !project.mark_startup_remediation_attempted() {
            return;
        }

        let freshness = freshness_snapshot_for_project(&project, false).await;
        let stats = match read_persisted_index_stats(&project).await {
            Ok(stats) => stats,
            Err(err) => {
                tracing::warn!(
                    project = %project.project_key,
                    error = %err,
                    "startup ensure-current: failed reading persisted index stats"
                );
                return;
            }
        };

        let total_chunks = stats.map(|(_, chunks, _)| chunks).unwrap_or(0);
        let initial_build_needed = total_chunks == 0;
        let stale_refresh_needed = total_chunks > 0 && freshness.state == CoreFreshnessState::Stale;

        let startup_plan = if initial_build_needed {
            Some((
                provider_name_for_project(&project)
                    .await
                    .unwrap_or_else(|_| "fastembed".to_string()),
                "startup_initial_build",
            ))
        } else if stale_refresh_needed {
            match persisted_provider_name_for_project(&project).await {
                Ok(Some(provider_name)) => Some((provider_name, "startup_stale_refresh")),
                Ok(None) => {
                    tracing::warn!(
                        project = %project.project_key,
                        "startup stale refresh skipped: manifest provider metadata missing"
                    );
                    None
                }
                Err(err) => {
                    tracing::warn!(
                        project = %project.project_key,
                        error = %err,
                        "startup stale refresh skipped: failed reading manifest provider metadata"
                    );
                    None
                }
            }
        } else {
            None
        };
        let Some((provider_name, trigger)) = startup_plan else {
            return;
        };

        match self
            .start_indexing_run(Arc::clone(&project), provider_name, trigger)
            .await
        {
            Ok(IndexStartOutcome::Started) => {
                tracing::info!(project = %project.project_key, trigger, "startup ensure-current scheduled indexing");
            }
            Ok(IndexStartOutcome::AlreadyIndexing) => {
                tracing::debug!(project = %project.project_key, trigger, "startup ensure-current skipped: indexing already running");
            }
            Err(err) => {
                tracing::error!(
                    project = %project.project_key,
                    trigger,
                    error = %err,
                    "startup ensure-current failed to schedule indexing"
                );
            }
        }
    }

    async fn start_indexing_run(
        &self,
        project: Arc<ProjectState>,
        provider_name: String,
        trigger: &'static str,
    ) -> anyhow::Result<IndexStartOutcome> {
        if project.is_indexing_running().await {
            return Ok(IndexStartOutcome::AlreadyIndexing);
        }

        match provider_name.as_str() {
            "fastembed" | "voyage" | "openai" => {}
            unknown => {
                anyhow::bail!("unknown provider: '{unknown}'. Valid: fastembed, voyage, openai")
            }
        }

        let now = Utc::now();
        let path = project.canonical_root.to_string_lossy().into_owned();
        let mut shared_status = SharedIndexingStatus {
            instance_id: self.instance_id.to_string(),
            pid: std::process::id(),
            path: path.clone(),
            provider: provider_name.clone(),
            trigger: trigger.to_string(),
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
            Ok(None) => return Ok(IndexStartOutcome::AlreadyIndexing),
            Err(err) => anyhow::bail!("failed to acquire indexing lease: {err:#}"),
        };

        let staged = prepare_staged_generation(
            &project.storage_dir,
            &project.active_backend_path().await,
            &project.active_manifest_path().await,
        )?;

        project
            .mark_indexing_started(path.clone(), provider_name.clone())
            .await;

        let project_for_task = Arc::clone(&project);
        tokio::spawn(async move {
            let started = Instant::now();
            match run_real_index(
                Arc::clone(&project_for_task),
                provider_name.clone(),
                staged.clone(),
            )
            .await
            {
                Ok(index_result) => {
                    if let Err(err) = project_for_task.promote_staged_generation(&staged).await {
                        let elapsed = started.elapsed().as_secs_f64();
                        let error_message = format!(
                            "failed to promote staged generation '{}': {err:#}",
                            staged.generation_id
                        );
                        project_for_task
                            .mark_indexing_failed(elapsed, error_message.clone())
                            .await;
                        project_for_task.invalidate_searcher().await;
                        cleanup_generation_dir(&staged.generation_dir);

                        shared_status.status = "failed".to_string();
                        shared_status.updated_at = Utc::now();
                        shared_status.error = Some(error_message);
                        if let Err(write_err) = lease.write_status(&shared_status) {
                            tracing::warn!(project = %project_for_task.project_key, error = %write_err, "failed to write failed indexing status");
                        }
                        return;
                    }

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
                    cleanup_generation_dir(&staged.generation_dir);
                    project_for_task
                        .mark_indexing_failed(elapsed, error_message.clone())
                        .await;
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

        Ok(IndexStartOutcome::Started)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IndexStartOutcome {
    Started,
    AlreadyIndexing,
}

async fn read_persisted_index_stats(
    project: &ProjectState,
) -> anyhow::Result<Option<(usize, usize, Option<String>)>> {
    let backend = project.active_backend().await;
    match backend.stats().await {
        Ok(stats) => Ok(Some((
            stats.indexed_files,
            stats.total_chunks,
            stats
                .last_indexed
                .map(|dt: chrono::DateTime<chrono::Utc>| dt.to_rfc3339()),
        ))),
        Err(err)
            if err.to_string().to_lowercase().contains("stored relation")
                && err.to_string().to_lowercase().contains("not found") =>
        {
            Ok(None)
        }
        Err(err) => Err(err),
    }
}

async fn freshness_snapshot_for_project(
    project: &ProjectState,
    refreshing: bool,
) -> CoreFreshnessSnapshot {
    let manifest_path = project.active_manifest_path().await;
    let snapshot = match ManifestStore::open(manifest_path.as_path()) {
        Ok(manifest) => CoreFreshnessSnapshot::from_manifest(&manifest, &project.canonical_root),
        Err(err) => CoreFreshnessSnapshot::from_stale_count_result(Err(err)),
    };
    snapshot.with_refreshing(refreshing)
}

fn protocol_freshness_state(state: CoreFreshnessState) -> IndexFreshnessState {
    match state {
        CoreFreshnessState::Fresh => IndexFreshnessState::Fresh,
        CoreFreshnessState::Stale => IndexFreshnessState::Stale,
        CoreFreshnessState::Refreshing => IndexFreshnessState::Refreshing,
        CoreFreshnessState::Unknown => IndexFreshnessState::Unknown,
    }
}

async fn provider_name_for_project(project: &ProjectState) -> anyhow::Result<String> {
    let manifest_path = project.active_manifest_path().await;
    let manifest =
        ManifestStore::open(manifest_path.as_path()).context("failed to open manifest")?;
    Ok(manifest
        .get_meta("provider")
        .context("failed to read provider from manifest")?
        .unwrap_or_else(|| "fastembed".to_string()))
}

async fn persisted_provider_name_for_project(
    project: &ProjectState,
) -> anyhow::Result<Option<String>> {
    let manifest_path = project.active_manifest_path().await;
    let manifest =
        ManifestStore::open(manifest_path.as_path()).context("failed to open manifest")?;
    manifest
        .get_meta("provider")
        .context("failed to read provider from manifest")
}

fn build_arc_provider(provider_name: &str) -> anyhow::Result<ArcProvider> {
    let boxed: Box<dyn EmbedProvider + Send + Sync> = provider_from_name(provider_name)
        .with_context(|| format!("failed to initialize provider '{provider_name}'"))?;
    Ok(ArcProvider(Arc::from(boxed)))
}

async fn build_searcher_for_project(project: &ProjectState) -> anyhow::Result<Arc<CachedSearcher>> {
    let provider_name = provider_name_for_project(project).await?;
    let provider = build_arc_provider(&provider_name)?;
    {
        let mut guard = project.provider_identity.write().await;
        *guard = Some(provider_name.clone());
    }
    let backend = project.active_backend().await;
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

async fn run_real_index(
    project: Arc<ProjectState>,
    provider_name: String,
    staged: StagedGeneration,
) -> anyhow::Result<skelesearch_core::IndexResult> {
    let backend_path = staged.backend_db_path.clone();
    let manifest_path = staged.manifest_db_path.clone();
    let root = project.canonical_root.clone();
    tokio::task::spawn_blocking(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| anyhow::anyhow!("runtime build: {e}"))?;
        rt.block_on(async move {
            if let Ok(delay_ms_raw) = std::env::var("SKELESEARCH_TEST_INDEX_DELAY_MS") {
                if let Ok(delay_ms) = delay_ms_raw.parse::<u64>() {
                    if delay_ms > 0 {
                        tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                    }
                }
            }
            // Test bypass: use TestProvider when SKELESEARCH_TEST_PROVIDER_DIM is set.
            // This prevents tests from requiring the real fastembed model download.
            #[cfg(test)]
            if let Ok(dim_str) = std::env::var("SKELESEARCH_TEST_PROVIDER_DIM") {
                if let Ok(dim) = dim_str.parse::<usize>() {
                    let backend = Arc::new(CompositeBackend::open(backend_path.parent().ok_or_else(|| anyhow::anyhow!("backend path has no parent"))?).await?);
                    let manifest = Arc::new(ManifestStore::open(&manifest_path)?);
                    let config = Config::load(&root).context("load .skelesearch.toml")?;
                    let indexer = Indexer::new(backend, manifest, TestProvider { name: "test", dim })
                        .with_excludes(config.index.exclude.clone())
                        .with_include_extensions(config.index.include_extensions.clone())
                        .with_scope_prefix(config.index.scope_prefix);
                    return indexer.index_path(&root).await;
                }
            }
            let backend = Arc::new(CompositeBackend::open(backend_path.parent().ok_or_else(|| anyhow::anyhow!("backend path has no parent"))?).await?);
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
    active_generation: Arc<RwLock<ActiveGeneration>>,
    pub cached_searcher: Arc<RwLock<Option<Arc<CachedSearcher>>>>,
    pub provider_identity: Arc<RwLock<Option<String>>>,
    pub index_progress: Arc<RwLock<ProjectIndexRuntime>>,
    pub coordination: CoordinationStatePlaceholder,
    startup_remediation_attempted: AtomicBool,
    last_access: Arc<Mutex<SystemTime>>,
}

impl ProjectState {
    async fn open(project_key: ProjectKey) -> anyhow::Result<Self> {
        let canonical_root = PathBuf::from(&project_key.canonical_root);
        let storage_dir = storage_dir_for_root(&canonical_root);
        std::fs::create_dir_all(&storage_dir).with_context(|| {
            format!(
                "create daemon project storage directory '{}'",
                storage_dir.display()
            )
        })?;

        let active_generation = resolve_or_init_active_generation(&storage_dir).await?;

        Ok(Self {
            project_key,
            canonical_root,
            storage_dir: storage_dir.clone(),
            active_generation: Arc::new(RwLock::new(active_generation)),
            cached_searcher: Arc::new(RwLock::new(None)),
            provider_identity: Arc::new(RwLock::new(None)),
            index_progress: Arc::new(RwLock::new(ProjectIndexRuntime::default())),
            coordination: CoordinationStatePlaceholder::for_storage_dir(&storage_dir),
            startup_remediation_attempted: AtomicBool::new(false),
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

    pub async fn active_backend_path(&self) -> PathBuf {
        self.active_generation
            .read()
            .await
            .backend
            .index_db_path
            .clone()
    }

    pub async fn active_backend(&self) -> Arc<CompositeBackend> {
        self.active_generation
            .read()
            .await
            .composite_backend
            .clone()
    }

    pub async fn active_manifest_path(&self) -> PathBuf {
        self.active_generation
            .read()
            .await
            .manifest
            .manifest_db_path
            .clone()
    }

    #[cfg(test)]
    pub async fn active_generation_id(&self) -> String {
        self.active_generation.read().await.generation_id.clone()
    }

    async fn promote_staged_generation(&self, staged: &StagedGeneration) -> anyhow::Result<()> {
        let promoted = ActiveGeneration::open(
            staged.generation_id.clone(),
            staged.generation_dir.clone(),
            staged.backend_db_path.clone(),
            staged.manifest_db_path.clone(),
        )
        .await?;
        tracing::info!(
            project = %self.project_key,
            generation = %promoted.generation_id,
            generation_dir = %promoted.generation_dir.display(),
            "promoting staged generation"
        );
        write_file_atomic(
            &active_generation_pointer_path(&self.storage_dir),
            staged.generation_id.as_bytes(),
        )?;
        let mut guard = self.active_generation.write().await;
        *guard = promoted;
        Ok(())
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

    fn mark_startup_remediation_attempted(&self) -> bool {
        self.startup_remediation_attempted
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok()
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

#[derive(Debug, Clone)]
struct StagedGeneration {
    generation_id: String,
    generation_dir: PathBuf,
    backend_db_path: PathBuf,
    manifest_db_path: PathBuf,
}

#[derive(Clone)]
struct ActiveGeneration {
    generation_id: String,
    generation_dir: PathBuf,
    backend: Arc<BackendHandle>,
    manifest: Arc<ManifestHandle>,
    composite_backend: Arc<CompositeBackend>,
}

impl ActiveGeneration {
    async fn open(
        generation_id: String,
        generation_dir: PathBuf,
        backend_db_path: PathBuf,
        manifest_db_path: PathBuf,
    ) -> anyhow::Result<Self> {
        let backend_root = backend_db_path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("backend path has no parent"))?
            .to_path_buf();
        Ok(Self {
            generation_id,
            generation_dir,
            backend: Arc::new(BackendHandle::open(backend_db_path)?),
            manifest: Arc::new(ManifestHandle::open(manifest_db_path)?),
            composite_backend: Arc::new(CompositeBackend::open(&backend_root).await?),
        })
    }
}

fn generations_dir(storage_dir: &Path) -> PathBuf {
    storage_dir.join(GENERATIONS_DIR)
}

fn active_generation_pointer_path(storage_dir: &Path) -> PathBuf {
    storage_dir.join(ACTIVE_GENERATION_FILE)
}

fn generation_dir_for_id(storage_dir: &Path, generation_id: &str) -> PathBuf {
    generations_dir(storage_dir).join(generation_id)
}

fn new_generation_id() -> String {
    format!(
        "gen-{}-{}",
        Utc::now().timestamp_millis(),
        Utc::now().timestamp_nanos_opt().unwrap_or_default()
    )
}

async fn resolve_or_init_active_generation(storage_dir: &Path) -> anyhow::Result<ActiveGeneration> {
    std::fs::create_dir_all(generations_dir(storage_dir)).with_context(|| {
        format!(
            "create generation directory under '{}'",
            storage_dir.display()
        )
    })?;

    let pointer_path = active_generation_pointer_path(storage_dir);
    if let Ok(pointer) = std::fs::read_to_string(&pointer_path) {
        let generation_id = pointer.trim();
        if !generation_id.is_empty() {
            let generation_dir = generation_dir_for_id(storage_dir, generation_id);
            let (backend_db_path, manifest_db_path) = generation_db_paths(&generation_dir);
            if backend_db_path.exists() && manifest_db_path.exists() {
                return ActiveGeneration::open(
                    generation_id.to_string(),
                    generation_dir,
                    backend_db_path,
                    manifest_db_path,
                )
                .await;
            }
            tracing::warn!(
                pointer = generation_id,
                path = %pointer_path.display(),
                "active generation pointer is stale; reinitializing"
            );
        }
    }

    let generation_id = new_generation_id();
    let generation_dir = generation_dir_for_id(storage_dir, &generation_id);
    std::fs::create_dir_all(&generation_dir)
        .with_context(|| format!("create generation dir '{}'", generation_dir.display()))?;

    let (backend_db_path, manifest_db_path) = generation_db_paths(&generation_dir);

    write_file_atomic(&pointer_path, generation_id.as_bytes()).with_context(|| {
        format!(
            "write active generation pointer at '{}'",
            pointer_path.display()
        )
    })?;

    ActiveGeneration::open(
        generation_id,
        generation_dir,
        backend_db_path,
        manifest_db_path,
    )
    .await
}

fn prepare_staged_generation(
    storage_dir: &Path,
    source_backend_path: &Path,
    source_manifest_path: &Path,
) -> anyhow::Result<StagedGeneration> {
    let generation_id = new_generation_id();
    let generation_dir = generation_dir_for_id(storage_dir, &generation_id);
    std::fs::create_dir_all(&generation_dir).with_context(|| {
        format!(
            "create staged generation dir '{}'",
            generation_dir.display()
        )
    })?;
    let (backend_db_path, manifest_db_path) = generation_db_paths(&generation_dir);

    let source_backend_dir = source_backend_path.parent().ok_or_else(|| {
        anyhow::anyhow!(
            "backend marker path '{}' has no parent directory",
            source_backend_path.display()
        )
    })?;

    copy_backend_dir(source_backend_dir, &generation_dir)?;
    copy_manifest_family(source_manifest_path, &manifest_db_path)?;

    Ok(StagedGeneration {
        generation_id,
        generation_dir,
        backend_db_path,
        manifest_db_path,
    })
}

fn cleanup_generation_dir(path: &Path) {
    if let Err(err) = std::fs::remove_dir_all(path) {
        if err.kind() != std::io::ErrorKind::NotFound {
            tracing::warn!(path = %path.display(), error = %err, "failed to remove staged generation directory");
        }
    }
}

fn copy_backend_dir(source_dir: &Path, destination_dir: &Path) -> anyhow::Result<()> {
    for entry in std::fs::read_dir(source_dir)
        .with_context(|| format!("read backend directory '{}'", source_dir.display()))?
    {
        let entry = entry.with_context(|| format!("read entry in '{}'", source_dir.display()))?;
        let source_path = entry.path();
        let destination_path = destination_dir.join(entry.file_name());
        let file_type = entry.file_type().with_context(|| {
            format!(
                "read file type for backend entry '{}'",
                source_path.display()
            )
        })?;

        if should_skip_backend_entry(&source_path) {
            continue;
        }

        if file_type.is_dir() {
            copy_dir_recursive(&source_path, &destination_path)?;
        } else if file_type.is_file() {
            copy_file(source_path.as_path(), destination_path.as_path())?;
        }
    }
    Ok(())
}

fn copy_dir_recursive(source_dir: &Path, destination_dir: &Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(destination_dir)
        .with_context(|| format!("create directory '{}'", destination_dir.display()))?;
    for entry in std::fs::read_dir(source_dir)
        .with_context(|| format!("read directory '{}'", source_dir.display()))?
    {
        let entry = entry.with_context(|| format!("read entry in '{}'", source_dir.display()))?;
        let source_path = entry.path();
        let destination_path = destination_dir.join(entry.file_name());
        let file_type = entry.file_type().with_context(|| {
            format!(
                "read file type for directory entry '{}'",
                source_path.display()
            )
        })?;

        if should_skip_backend_entry(&source_path) {
            continue;
        }

        if file_type.is_dir() {
            copy_dir_recursive(&source_path, &destination_path)?;
        } else if file_type.is_file() {
            copy_file(source_path.as_path(), destination_path.as_path())?;
        }
    }
    Ok(())
}

fn should_skip_backend_entry(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.ends_with(".lock"))
        .unwrap_or(false)
}

fn copy_manifest_family(source: &Path, destination: &Path) -> anyhow::Result<()> {
    for suffix in ["", "-wal", "-shm"] {
        let src = PathBuf::from(format!("{}{}", source.display(), suffix));
        if !src.exists() {
            continue;
        }
        let dst = PathBuf::from(format!("{}{}", destination.display(), suffix));
        copy_file(src.as_path(), dst.as_path())?;
    }
    Ok(())
}

fn copy_file(source: &Path, destination: &Path) -> anyhow::Result<()> {
    std::fs::copy(source, destination).with_context(|| {
        format!(
            "copy file '{}' to '{}'",
            source.display(),
            destination.display()
        )
    })?;
    if let Ok(metadata) = std::fs::metadata(source) {
        let _ = std::fs::set_permissions(destination, metadata.permissions());
    }
    Ok(())
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
    pub freshness_state: IndexFreshnessState,
    pub freshness_checked_at: Option<String>,
    pub freshness_error: Option<String>,
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
        register_client: true,
        heartbeat: true,
        unregister_client: true,
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
#[derive(Clone)]
struct TestProvider {
    name: &'static str,
    dim: usize,
}

#[cfg(test)]
#[async_trait]
impl EmbedProvider for TestProvider {
    fn dim(&self) -> usize {
        self.dim
    }

    fn name(&self) -> &str {
        self.name
    }

    async fn embed_batch(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(texts
            .into_iter()
            .enumerate()
            .map(|(idx, _)| {
                let mut v = vec![0.05_f32; self.dim];
                if !v.is_empty() {
                    v[0] = (idx as f32 + 1.0) * 0.1;
                }
                v
            })
            .collect())
    }

    async fn embed_queries(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
        self.embed_batch(texts).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex as StdMutex, OnceLock};

    fn env_lock() -> &'static StdMutex<()> {
        static LOCK: OnceLock<StdMutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| StdMutex::new(()))
    }

    async fn create_populated_index(project: &ProjectState, provider_name: &'static str) {
        std::fs::create_dir_all(project.canonical_root.join("src")).expect("create src");
        std::fs::write(
            project.canonical_root.join("src/lib.rs"),
            "pub fn hello() -> u32 { 1 }",
        )
        .expect("write source file");

        let backend = project.active_backend().await;
        let manifest = Arc::new(
            ManifestStore::open(project.active_manifest_path().await.as_path())
                .expect("open manifest"),
        );
        let indexer = Indexer::new(
            backend,
            manifest,
            TestProvider {
                name: provider_name,
                dim: 8,
            },
        );
        indexer
            .index_path(&project.canonical_root)
            .await
            .expect("index project with test provider");
    }

    fn root_target(root: &Path) -> ProjectTarget {
        ProjectTarget::RootPath {
            root_path: root.to_string_lossy().into_owned(),
            logical_id: None,
        }
    }

    async fn wait_for_index_state(
        service: &DaemonService,
        root: &Path,
        expected: IndexingState,
    ) -> IndexStatusResponse {
        for _ in 0..300 {
            let response = service
                .handle_request(DaemonRequest::IndexStatus(IndexStatusRequest {
                    target: root_target(root),
                }))
                .await;
            let DaemonResponse::IndexStatus(status) = response else {
                panic!("expected index status response");
            };
            if matches!(status.indexing.as_ref().map(|item| &item.status), Some(state) if state == &expected)
            {
                return status;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        panic!("timed out waiting for indexing state");
    }

    #[tokio::test]
    async fn active_generation_initialization_ignores_root_level_legacy_files() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = temp.path().join("repo");
        let storage_dir = root.join(STORAGE_DIR_NAME);
        std::fs::create_dir_all(&storage_dir).expect("create storage dir");
        std::fs::write(storage_dir.join("index.db"), "legacy backend")
            .expect("write legacy backend");
        std::fs::write(storage_dir.join("manifest.db"), "legacy manifest")
            .expect("write legacy manifest");

        let state = DaemonState::new();
        let project = state
            .resolve_or_open_project(ProjectLookup::from_root_path(&root))
            .await
            .expect("resolve project");

        let active_generation_id = project.active_generation_id().await;
        assert!(
            storage_dir
                .join(GENERATIONS_DIR)
                .join(active_generation_id)
                .exists(),
            "daemon should initialize a fresh active generation"
        );
        assert!(
            storage_dir.join("index.db").exists(),
            "legacy root backend files should be left untouched"
        );
        assert!(
            storage_dir.join("manifest.db").exists(),
            "legacy root manifest files should be left untouched"
        );
    }

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
        assert_eq!(status.freshness_state, IndexFreshnessState::Fresh);
        assert!(status.freshness_checked_at.is_some());
        assert_eq!(status.freshness_error, None);
        assert!(!status.watching);
        assert!(status.indexing.is_none());
    }

    #[tokio::test]
    async fn index_status_reports_stale_freshness_for_manifest_entry_missing_on_disk() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = temp.path().join("repo");
        std::fs::create_dir_all(&root).expect("create root");

        let state = DaemonState::new();
        let project = state
            .resolve_or_open_project(ProjectLookup::from_root_path(&root))
            .await
            .expect("resolve project");

        let manifest = ManifestStore::open(project.active_manifest_path().await.as_path())
            .expect("open manifest");
        manifest
            .upsert("src/deleted.rs", 1, 1, "fixture-hash")
            .expect("insert stale manifest entry");

        let service = DaemonService::new(state);
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

        assert_eq!(status.estimated_stale, 1);
        assert_eq!(status.freshness_state, IndexFreshnessState::Stale);
        assert!(status.freshness_checked_at.is_some());
        assert_eq!(status.freshness_error, None);
        assert!(!status.watching);
        assert!(status.indexing.is_none());
    }

    #[tokio::test]
    async fn startup_stale_refresh_schedules_refresh_for_populated_stale_index() {
        let _guard = env_lock().lock().unwrap_or_else(|err| err.into_inner());
        std::env::remove_var("SKELESEARCH_NO_AUTO_INDEX");
        std::env::remove_var("OPENAI_API_KEY");

        let temp = tempfile::tempdir().expect("tempdir");
        let root = temp.path().join("repo");
        std::fs::create_dir_all(&root).expect("create root");

        let state = DaemonState::new();
        let project = state
            .resolve_or_open_project(ProjectLookup::from_root_path(&root))
            .await
            .expect("resolve project");
        create_populated_index(&project, "openai").await;

        let manifest = ManifestStore::open(project.active_manifest_path().await.as_path())
            .expect("open manifest");
        manifest
            .upsert("src/deleted.rs", 1, 1, "fixture-hash")
            .expect("insert stale manifest entry");

        let service = DaemonService::new(state);
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

        assert_eq!(status.freshness_state, IndexFreshnessState::Refreshing);
        assert!(
            status.indexing.is_some(),
            "expected startup refresh to schedule indexing"
        );
    }

    #[tokio::test]
    async fn startup_stale_refresh_fresh_populated_index_is_noop() {
        let _guard = env_lock().lock().unwrap_or_else(|err| err.into_inner());
        std::env::remove_var("SKELESEARCH_NO_AUTO_INDEX");

        let temp = tempfile::tempdir().expect("tempdir");
        let root = temp.path().join("repo");
        std::fs::create_dir_all(&root).expect("create root");

        let state = DaemonState::new();
        let project = state
            .resolve_or_open_project(ProjectLookup::from_root_path(&root))
            .await
            .expect("resolve project");
        create_populated_index(&project, "openai").await;

        let service = DaemonService::new(state);
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

        assert_eq!(status.freshness_state, IndexFreshnessState::Fresh);
        assert!(
            status.indexing.is_none(),
            "fresh index should not schedule startup refresh"
        );
    }

    #[tokio::test]
    async fn startup_stale_refresh_provider_failure_keeps_existing_index_and_stale_signal() {
        let _guard = env_lock().lock().unwrap_or_else(|err| err.into_inner());
        std::env::remove_var("SKELESEARCH_NO_AUTO_INDEX");
        std::env::remove_var("VOYAGE_API_KEY");

        let temp = tempfile::tempdir().expect("tempdir");
        let root = temp.path().join("repo");
        std::fs::create_dir_all(&root).expect("create root");

        let state = DaemonState::new();
        let project = state
            .resolve_or_open_project(ProjectLookup::from_root_path(&root))
            .await
            .expect("resolve project");
        create_populated_index(&project, "voyage").await;

        let baseline_stats = read_persisted_index_stats(&project)
            .await
            .expect("read baseline persisted stats")
            .expect("baseline stats should exist");

        let manifest = ManifestStore::open(project.active_manifest_path().await.as_path())
            .expect("open manifest");
        manifest
            .upsert("src/deleted.rs", 1, 1, "fixture-hash")
            .expect("insert stale manifest entry");

        let service = DaemonService::new(state.clone());
        let _ = service
            .handle_request(DaemonRequest::IndexStatus(IndexStatusRequest {
                target: ProjectTarget::RootPath {
                    root_path: root.to_string_lossy().into_owned(),
                    logical_id: None,
                },
            }))
            .await;

        let mut failed_status: Option<IndexStatusResponse> = None;
        for _ in 0..30 {
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
            if matches!(
                status.indexing.as_ref().map(|progress| &progress.status),
                Some(IndexingState::Failed)
            ) {
                failed_status = Some(status);
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        let status =
            failed_status.expect("startup refresh should fail when provider initialization fails");
        assert_eq!(status.freshness_state, IndexFreshnessState::Stale);
        assert!(status.estimated_stale > 0);
        assert!(status.indexing.is_some());
        assert_eq!(status.indexed_files, baseline_stats.0);
        assert_eq!(status.total_chunks, baseline_stats.1);

        let persisted_after = read_persisted_index_stats(&project)
            .await
            .expect("read persisted stats after provider failure")
            .expect("stats should remain available");
        assert_eq!(persisted_after.0, baseline_stats.0);
        assert_eq!(persisted_after.1, baseline_stats.1);

        for _ in 0..3 {
            let response = service
                .handle_request(DaemonRequest::IndexStatus(IndexStatusRequest {
                    target: ProjectTarget::RootPath {
                        root_path: root.to_string_lossy().into_owned(),
                        logical_id: None,
                    },
                }))
                .await;
            let DaemonResponse::IndexStatus(polled) = response else {
                panic!("expected index status response");
            };
            assert!(
                matches!(
                    polled.indexing.as_ref().map(|progress| &progress.status),
                    Some(IndexingState::Failed)
                ),
                "status poll should remain failed without retriggering startup refresh"
            );
            tokio::time::sleep(Duration::from_millis(30)).await;
        }
    }

    #[tokio::test]
    async fn generation_swap_preserves_reads() {
        let _guard = env_lock().lock().unwrap_or_else(|err| err.into_inner());
        std::env::set_var("SKELESEARCH_NO_AUTO_INDEX", "1");
        std::env::set_var("SKELESEARCH_TEST_INDEX_DELAY_MS", "300");
        std::env::set_var("SKELESEARCH_TEST_PROVIDER_DIM", "8");

        let temp = tempfile::tempdir().expect("tempdir");
        let root = temp.path().join("repo");
        std::fs::create_dir_all(&root).expect("create root");

        let state = DaemonState::new();
        let project = state
            .resolve_or_open_project(ProjectLookup::from_root_path(&root))
            .await
            .expect("resolve project");
        create_populated_index(&project, "fastembed").await;

        let manifest = ManifestStore::open(project.active_manifest_path().await.as_path())
            .expect("open manifest");
        manifest
            .upsert("src/deleted.rs", 1, 1, "fixture-hash")
            .expect("insert stale manifest entry");

        let before_generation = project.active_generation_id().await;

        let backend = project.active_backend().await;
        let seeded_searcher = Searcher::new(
            backend,
            ArcProvider::new(TestProvider {
                name: "fastembed",
                dim: 8,
            }),
        );
        {
            let mut guard = project.cached_searcher.write().await;
            *guard = Some(Arc::new(seeded_searcher));
        }

        let service = DaemonService::new(state);

        let started = service
            .start_indexing_run(
                Arc::clone(&project),
                "fastembed".to_string(),
                "test_generation_swap",
            )
            .await
            .expect("start refresh indexing");
        assert_eq!(started, IndexStartOutcome::Started);

        let _running = wait_for_index_state(&service, &root, IndexingState::Running).await;

        let search = service
            .handle_request(DaemonRequest::SearchCode(SearchCodeRequest {
                target: root_target(&root),
                query: "hello".to_string(),
                top_k: 5,
                include_graph: false,
                max_depth: None,
                diversity: 0.3,
                max_tokens: Some(1024),
                branch_scope: false,
                session_id: None,
            }))
            .await;
        let DaemonResponse::SearchCode(search) = search else {
            panic!("expected successful search while refresh is running");
        };
        assert!(
            !search.results.is_empty(),
            "expected reads from last-good generation"
        );

        let _done = wait_for_index_state(&service, &root, IndexingState::Done).await;
        let after_generation = project.active_generation_id().await;
        assert_ne!(after_generation, before_generation);

        let response = service
            .handle_request(DaemonRequest::IndexStatus(IndexStatusRequest {
                target: root_target(&root),
            }))
            .await;
        let DaemonResponse::IndexStatus(status) = response else {
            panic!("expected index status response");
        };
        assert_eq!(status.freshness_state, IndexFreshnessState::Fresh);

        std::env::remove_var("SKELESEARCH_TEST_INDEX_DELAY_MS");
        std::env::remove_var("SKELESEARCH_TEST_PROVIDER_DIM");
        std::env::remove_var("SKELESEARCH_NO_AUTO_INDEX");
    }

    #[tokio::test]
    async fn failed_refresh_keeps_last_good() {
        let _guard = env_lock().lock().unwrap_or_else(|err| err.into_inner());
        std::env::set_var("SKELESEARCH_NO_AUTO_INDEX", "1");
        std::env::remove_var("VOYAGE_API_KEY");

        let temp = tempfile::tempdir().expect("tempdir");
        let root = temp.path().join("repo");
        std::fs::create_dir_all(&root).expect("create root");

        let state = DaemonState::new();
        let project = state
            .resolve_or_open_project(ProjectLookup::from_root_path(&root))
            .await
            .expect("resolve project");
        create_populated_index(&project, "fastembed").await;

        let before_generation = project.active_generation_id().await;
        let baseline_stats = read_persisted_index_stats(&project)
            .await
            .expect("read baseline persisted stats")
            .expect("baseline stats should exist");

        let manifest = ManifestStore::open(project.active_manifest_path().await.as_path())
            .expect("open manifest");
        manifest
            .upsert("src/deleted.rs", 1, 1, "fixture-hash")
            .expect("insert stale manifest entry");

        let service = DaemonService::new(state);
        let started = service
            .start_indexing_run(
                Arc::clone(&project),
                "voyage".to_string(),
                "test_failed_refresh",
            )
            .await
            .expect("start failing refresh indexing");
        assert_eq!(started, IndexStartOutcome::Started);

        let failed = wait_for_index_state(&service, &root, IndexingState::Failed).await;
        assert!(failed.estimated_stale > 0);
        assert_eq!(failed.freshness_state, IndexFreshnessState::Stale);

        let after_generation = project.active_generation_id().await;
        assert_eq!(after_generation, before_generation);

        let persisted_after = read_persisted_index_stats(&project)
            .await
            .expect("read persisted stats after failed refresh")
            .expect("stats should remain available");
        assert_eq!(persisted_after.0, baseline_stats.0);
        assert_eq!(persisted_after.1, baseline_stats.1);

        std::env::remove_var("SKELESEARCH_NO_AUTO_INDEX");
    }

    #[tokio::test]
    async fn staged_generation_copies_incremental_state_from_active_generation() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = temp.path().join("repo");
        std::fs::create_dir_all(&root).expect("create root");

        let state = DaemonState::new();
        let project = state
            .resolve_or_open_project(ProjectLookup::from_root_path(&root))
            .await
            .expect("resolve project");
        create_populated_index(&project, "fastembed").await;

        let active_manifest_path = project.active_manifest_path().await;
        let active_backend_path = project.active_backend_path().await;
        let active_manifest =
            ManifestStore::open(active_manifest_path.as_path()).expect("open active manifest");
        active_manifest
            .upsert("src/lib.rs", 123, 26, "fixture-hash")
            .expect("seed file hash entry");
        active_manifest
            .cache_embeddings(&[("sentinel-cache-row".to_string(), vec![0.5_f32; 8])])
            .expect("seed embedding cache entry");

        let staged = prepare_staged_generation(
            &project.storage_dir,
            &active_backend_path,
            &active_manifest_path,
        )
        .expect("prepare staged generation");

        let staged_manifest =
            ManifestStore::open(staged.manifest_db_path.as_path()).expect("open staged manifest");
        assert!(
            staged_manifest
                .mtime_size_unchanged("src/lib.rs", 123, 26)
                .expect("read staged file hash"),
            "staged manifest should retain file hash entries for unchanged-file skipping"
        );
        assert!(
            staged_manifest
                .get_cached_embeddings(&["sentinel-cache-row".to_string()], 8)
                .expect("read staged embedding cache")[0]
                .is_some(),
            "staged manifest should retain embedding cache entries for stale refresh reuse"
        );

        let staged_backend = CompositeBackend::open(staged.backend_db_path.parent().unwrap())
            .await
            .expect("open staged backend");
        assert_eq!(
            staged_backend
                .list_indexed_paths()
                .await
                .expect("staged backend paths"),
            vec!["src/lib.rs".to_string()],
            "staged backend should start from the last-good indexed data"
        );
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

    #[tokio::test]
    async fn client_session_register_heartbeat_and_unregister_round_trip() {
        let service = DaemonService::default();

        let registered = service
            .handle_request(DaemonRequest::RegisterClient(RegisterClientRequest {
                client_name: Some("skelesearch-mcp".to_string()),
                client_version: Some("0.1.0".to_string()),
            }))
            .await;

        let DaemonResponse::RegisterClient(registered) = registered else {
            panic!("expected register client response");
        };
        assert!(!registered.session_id.is_empty());
        assert!(registered.heartbeat_interval_seconds > 0);
        assert!(registered.lease_ttl_seconds >= registered.heartbeat_interval_seconds);

        let heartbeat = service
            .handle_request(DaemonRequest::Heartbeat(HeartbeatRequest {
                session_id: registered.session_id.clone(),
            }))
            .await;

        let DaemonResponse::Heartbeat(heartbeat) = heartbeat else {
            panic!("expected heartbeat response");
        };
        assert!(heartbeat.acknowledged);
        assert_eq!(heartbeat.session_id, registered.session_id);

        let unregistered = service
            .handle_request(DaemonRequest::UnregisterClient(UnregisterClientRequest {
                session_id: registered.session_id.clone(),
            }))
            .await;

        let DaemonResponse::UnregisterClient(unregistered) = unregistered else {
            panic!("expected unregister client response");
        };
        assert!(unregistered.removed);
        assert_eq!(unregistered.session_id, registered.session_id);
    }

    #[tokio::test]
    async fn client_sessions_expire_without_heartbeat() {
        let state = DaemonState::new_with_session_timing(
            Duration::from_millis(5),
            Duration::from_millis(20),
        );

        let registered = state
            .register_client(
                Some("skelesearch-mcp".to_string()),
                Some("0.1.0".to_string()),
            )
            .await;
        assert_eq!(state.live_session_count().await, 1);

        tokio::time::sleep(Duration::from_millis(30)).await;
        let expired = state.reap_expired_sessions().await;

        assert_eq!(expired, 1);
        assert_eq!(state.live_session_count().await, 0);
        assert!(!state.heartbeat(&registered.session_id).await.acknowledged);
    }
}
