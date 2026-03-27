pub mod project;
pub mod protocol;

pub use project::{ProjectKey, ProjectKeyError};
pub use protocol::{
    DaemonCapabilities, DaemonErrorCode, DaemonErrorResponse, DaemonEvent, DaemonRequest,
    DaemonResponse, DaemonStatusEvent, HandshakeRequest, HandshakeResponse, HeartbeatRequest,
    HeartbeatResponse, IndexCodebaseRequest, IndexCodebaseResponse, IndexFreshnessState,
    IndexProgressEvent, IndexStatusRequest, IndexStatusResponse, IndexingProgress, InfoRequest,
    InfoResponse, ProjectTarget, ProtocolErrorEvent, ProtocolFrame, RegisterClientRequest,
    RegisterClientResponse, RequestId, SearchCodeRequest, SearchCodeResponse, SearchResultRow,
    SmartSearchRequest, SmartSearchResponse, SmartSearchResults, StatusLevel, StreamId,
    UnregisterClientRequest, UnregisterClientResponse, DAEMON_PROTOCOL_VERSION,
};
