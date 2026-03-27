use crate::ProjectKey;
use serde::{Deserialize, Serialize};

/// Current transport-neutral daemon protocol version.
pub const DAEMON_PROTOCOL_VERSION: &str = "1";

/// Correlation identifier for request/response pairs.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[serde(transparent)]
pub struct RequestId(pub u64);

/// Logical stream identifier used by event frames.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[serde(transparent)]
pub struct StreamId(pub u64);

/// Framed daemon protocol envelope.
///
/// This stays transport-neutral and can be carried over UDS/TCP line framing,
/// HTTP/SSE adapters, or WebSocket adapters.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "frame", rename_all = "snake_case")]
pub enum ProtocolFrame {
    Request {
        id: RequestId,
        request: DaemonRequest,
    },
    Response {
        id: RequestId,
        response: DaemonResponse,
    },
    Event {
        stream_id: StreamId,
        event: DaemonEvent,
    },
    Cancel {
        id: RequestId,
    },
    Ping,
    Pong,
}

/// Daemon-originated stream event payloads.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "event", content = "payload", rename_all = "snake_case")]
pub enum DaemonEvent {
    IndexProgress(IndexProgressEvent),
    Status(DaemonStatusEvent),
    ProtocolError(ProtocolErrorEvent),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IndexProgressEvent {
    pub project_key: ProjectKey,
    pub progress: IndexingProgress,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<RequestId>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StatusLevel {
    Debug,
    Info,
    Warn,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DaemonStatusEvent {
    pub level: StatusLevel,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<RequestId>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProtocolErrorEvent {
    pub error: DaemonErrorResponse,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<RequestId>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "method", content = "params", rename_all = "snake_case")]
pub enum DaemonRequest {
    Handshake(HandshakeRequest),
    Info(InfoRequest),
    RegisterClient(RegisterClientRequest),
    Heartbeat(HeartbeatRequest),
    UnregisterClient(UnregisterClientRequest),
    IndexCodebase(IndexCodebaseRequest),
    IndexStatus(IndexStatusRequest),
    SearchCode(SearchCodeRequest),
    SmartSearch(SmartSearchRequest),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "method", content = "result", rename_all = "snake_case")]
pub enum DaemonResponse {
    Handshake(HandshakeResponse),
    Info(InfoResponse),
    RegisterClient(RegisterClientResponse),
    Heartbeat(HeartbeatResponse),
    UnregisterClient(UnregisterClientResponse),
    IndexCodebase(IndexCodebaseResponse),
    IndexStatus(IndexStatusResponse),
    SearchCode(SearchCodeResponse),
    SmartSearch(SmartSearchResponse),
    Error(DaemonErrorResponse),
}

/// Explicit project routing target.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "target", rename_all = "snake_case")]
pub enum ProjectTarget {
    ProjectKey {
        project_key: ProjectKey,
    },
    RootPath {
        /// Absolute path on the daemon host.
        root_path: String,
        /// Optional logical ID for hosted/multi-tenant routing.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        logical_id: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HandshakeRequest {
    pub protocol_version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_version: Option<String>,
}

impl HandshakeRequest {
    pub fn new(client_name: Option<String>, client_version: Option<String>) -> Self {
        Self {
            protocol_version: DAEMON_PROTOCOL_VERSION.to_string(),
            client_name,
            client_version,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HandshakeResponse {
    pub protocol_version: String,
    pub server_name: String,
    pub server_version: String,
    pub capabilities: DaemonCapabilities,
}

impl HandshakeResponse {
    pub fn new(server_name: impl Into<String>, server_version: impl Into<String>) -> Self {
        Self {
            protocol_version: DAEMON_PROTOCOL_VERSION.to_string(),
            server_name: server_name.into(),
            server_version: server_version.into(),
            capabilities: DaemonCapabilities::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct DaemonCapabilities {
    #[serde(default)]
    pub info: bool,
    #[serde(default)]
    pub index_codebase: bool,
    #[serde(default)]
    pub index_status: bool,
    #[serde(default)]
    pub search_code: bool,
    #[serde(default)]
    pub smart_search: bool,
    #[serde(default)]
    pub register_client: bool,
    #[serde(default)]
    pub heartbeat: bool,
    #[serde(default)]
    pub unregister_client: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct InfoRequest {}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InfoResponse {
    pub protocol_version: String,
    pub server_name: String,
    pub server_version: String,
    #[serde(default)]
    pub capabilities: DaemonCapabilities,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RegisterClientRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_version: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RegisterClientResponse {
    pub session_id: String,
    pub heartbeat_interval_seconds: u64,
    pub lease_ttl_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HeartbeatRequest {
    pub session_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HeartbeatResponse {
    pub session_id: String,
    pub acknowledged: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct UnregisterClientRequest {
    pub session_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct UnregisterClientResponse {
    pub session_id: String,
    pub removed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IndexCodebaseRequest {
    pub target: ProjectTarget,
    /// Embedding provider name (e.g. fastembed/openai/voyage).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum IndexCodebaseStatus {
    IndexingStarted,
    AlreadyIndexing,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IndexCodebaseResponse {
    pub status: IndexCodebaseStatus,
    pub project_key: ProjectKey,
    pub files_queued: usize,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IndexStatusRequest {
    pub target: ProjectTarget,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IndexStatusResponse {
    pub project_key: ProjectKey,
    pub indexed_files: usize,
    pub total_chunks: usize,
    pub last_indexed: Option<String>,
    pub estimated_stale: usize,
    #[serde(default)]
    pub freshness_state: IndexFreshnessState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub freshness_checked_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub freshness_error: Option<String>,
    pub watching: bool,
    pub indexing: Option<IndexingProgress>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum IndexFreshnessState {
    Fresh,
    Stale,
    Refreshing,
    #[default]
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SearchCodeRequest {
    pub target: ProjectTarget,
    pub query: String,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default)]
    pub include_graph: bool,
    #[serde(default)]
    pub max_depth: Option<usize>,
    #[serde(default = "default_diversity")]
    pub diversity: f32,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub branch_scope: bool,
    #[serde(default)]
    pub session_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SearchResultRow {
    pub file_path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub content: String,
    pub score: f64,
    pub match_quality: String,
    pub why: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SearchCodeResponse {
    pub project_key: ProjectKey,
    pub results: Vec<SearchResultRow>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SmartSearchRequest {
    pub target: ProjectTarget,
    pub query: String,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default)]
    pub include_graph: bool,
    #[serde(default = "default_diversity")]
    pub diversity: f32,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub branch_scope: bool,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub intent: Option<String>,
    #[serde(default)]
    pub symbols: Vec<String>,
    #[serde(default)]
    pub scope: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GrepSearchRow {
    pub file_path: String,
    pub line_number: usize,
    pub line_content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", content = "items", rename_all = "lowercase")]
pub enum SmartSearchResults {
    Grep(Vec<GrepSearchRow>),
    Semantic(Vec<SearchResultRow>),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SmartSearchResponse {
    pub project_key: ProjectKey,
    pub strategy: String,
    pub results: SmartSearchResults,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum IndexingState {
    Running,
    Done,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IndexingProgress {
    pub status: IndexingState,
    pub path: String,
    pub files_done: usize,
    pub files_total: usize,
    pub chunks_done: usize,
    pub cache_hits: usize,
    pub elapsed_seconds: f64,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DaemonErrorResponse {
    pub code: DaemonErrorCode,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
    #[serde(default)]
    pub retryable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DaemonErrorCode {
    BadRequest,
    UnsupportedProtocolVersion,
    NotFound,
    IndexUnavailable,
    Internal,
}

const fn default_top_k() -> usize {
    5
}

const fn default_diversity() -> f32 {
    0.3
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fixture_project_key() -> ProjectKey {
        ProjectKey {
            canonical_root: "/tmp/example".to_string(),
            logical_id: Some("tenant/example".to_string()),
        }
    }

    #[test]
    fn protocol_request_frame_round_trip() {
        let frame = ProtocolFrame::Request {
            id: RequestId(42),
            request: DaemonRequest::SmartSearch(SmartSearchRequest {
                target: ProjectTarget::ProjectKey {
                    project_key: fixture_project_key(),
                },
                query: "find protocol parser".to_string(),
                top_k: 7,
                include_graph: true,
                diversity: 0.2,
                max_tokens: Some(1200),
                branch_scope: false,
                session_id: Some("session-1".to_string()),
                intent: Some("find".to_string()),
                symbols: vec!["Protocol".to_string()],
                scope: Some("crates/service".to_string()),
            }),
        };

        let encoded = serde_json::to_string(&frame).expect("serialize");
        let decoded: ProtocolFrame = serde_json::from_str(&encoded).expect("deserialize");
        assert_eq!(decoded, frame);
    }

    #[test]
    fn protocol_response_frame_round_trip() {
        let frame = ProtocolFrame::Response {
            id: RequestId(42),
            response: DaemonResponse::SearchCode(SearchCodeResponse {
                project_key: fixture_project_key(),
                results: vec![SearchResultRow {
                    file_path: "src/protocol.rs".to_string(),
                    start_line: 12,
                    end_line: 21,
                    content: "pub enum DaemonRequest".to_string(),
                    score: 42.0,
                    match_quality: "high".to_string(),
                    why: "hybrid".to_string(),
                }],
            }),
        };

        let encoded = serde_json::to_string(&frame).expect("serialize");
        let decoded: ProtocolFrame = serde_json::from_str(&encoded).expect("deserialize");
        assert_eq!(decoded, frame);
    }

    #[test]
    fn protocol_event_frame_round_trip() {
        let frame = ProtocolFrame::Event {
            stream_id: StreamId(7),
            event: DaemonEvent::IndexProgress(IndexProgressEvent {
                project_key: fixture_project_key(),
                progress: IndexingProgress {
                    status: IndexingState::Running,
                    path: "/tmp/example".to_string(),
                    files_done: 2,
                    files_total: 10,
                    chunks_done: 6,
                    cache_hits: 1,
                    elapsed_seconds: 0.55,
                    error: None,
                },
                request_id: Some(RequestId(42)),
            }),
        };

        let encoded = serde_json::to_string(&frame).expect("serialize");
        let decoded: ProtocolFrame = serde_json::from_str(&encoded).expect("deserialize");
        assert_eq!(decoded, frame);
    }

    #[test]
    fn protocol_cancel_ping_pong_round_trip() {
        let cancel = ProtocolFrame::Cancel { id: RequestId(9) };
        let ping = ProtocolFrame::Ping;
        let pong = ProtocolFrame::Pong;

        let encoded_cancel = serde_json::to_string(&cancel).expect("serialize cancel");
        let encoded_ping = serde_json::to_string(&ping).expect("serialize ping");
        let encoded_pong = serde_json::to_string(&pong).expect("serialize pong");

        let decoded_cancel: ProtocolFrame =
            serde_json::from_str(&encoded_cancel).expect("deserialize cancel");
        let decoded_ping: ProtocolFrame =
            serde_json::from_str(&encoded_ping).expect("deserialize ping");
        let decoded_pong: ProtocolFrame =
            serde_json::from_str(&encoded_pong).expect("deserialize pong");

        assert_eq!(decoded_cancel, cancel);
        assert_eq!(decoded_ping, ping);
        assert_eq!(decoded_pong, pong);
    }

    #[test]
    fn handshake_types_include_protocol_version_constant() {
        let request = HandshakeRequest::new(Some("cli".to_string()), Some("0.1.0".to_string()));
        let response = HandshakeResponse::new("skelesearch-daemon", "0.1.0");

        assert_eq!(request.protocol_version, DAEMON_PROTOCOL_VERSION);
        assert_eq!(response.protocol_version, DAEMON_PROTOCOL_VERSION);
    }

    #[test]
    fn protocol_client_session_frames_round_trip() {
        let register = ProtocolFrame::Request {
            id: RequestId(77),
            request: DaemonRequest::RegisterClient(RegisterClientRequest {
                client_name: Some("skelesearch-mcp".to_string()),
                client_version: Some("0.1.0".to_string()),
            }),
        };
        let heartbeat = ProtocolFrame::Request {
            id: RequestId(78),
            request: DaemonRequest::Heartbeat(HeartbeatRequest {
                session_id: "session-123".to_string(),
            }),
        };
        let unregister = ProtocolFrame::Response {
            id: RequestId(79),
            response: DaemonResponse::UnregisterClient(UnregisterClientResponse {
                session_id: "session-123".to_string(),
                removed: true,
            }),
        };

        let encoded_register = serde_json::to_string(&register).expect("serialize register");
        let encoded_heartbeat = serde_json::to_string(&heartbeat).expect("serialize heartbeat");
        let encoded_unregister = serde_json::to_string(&unregister).expect("serialize unregister");

        let decoded_register: ProtocolFrame =
            serde_json::from_str(&encoded_register).expect("deserialize register");
        let decoded_heartbeat: ProtocolFrame =
            serde_json::from_str(&encoded_heartbeat).expect("deserialize heartbeat");
        let decoded_unregister: ProtocolFrame =
            serde_json::from_str(&encoded_unregister).expect("deserialize unregister");

        assert_eq!(decoded_register, register);
        assert_eq!(decoded_heartbeat, heartbeat);
        assert_eq!(decoded_unregister, unregister);
    }

    #[test]
    fn index_status_response_deserializes_legacy_payload_without_freshness_fields() {
        let payload = json!({
            "project_key": {
                "canonical_root": "/tmp/example",
                "logical_id": null
            },
            "indexed_files": 0,
            "total_chunks": 0,
            "last_indexed": null,
            "estimated_stale": 0,
            "watching": false,
            "indexing": null
        });

        let decoded: IndexStatusResponse =
            serde_json::from_value(payload).expect("legacy index status should deserialize");

        assert_eq!(decoded.freshness_state, IndexFreshnessState::Unknown);
        assert_eq!(decoded.freshness_checked_at, None);
        assert_eq!(decoded.freshness_error, None);
    }
}
