pub mod project;
pub mod protocol;

pub use project::{ProjectKey, ProjectKeyError};
pub use protocol::{
    DaemonCapabilities, DaemonErrorCode, DaemonErrorResponse, DaemonEvent, DaemonRequest,
    DaemonResponse, DaemonStatusEvent, HandshakeRequest, HandshakeResponse, IndexCodebaseRequest,
    IndexCodebaseResponse, IndexProgressEvent, IndexStatusRequest, IndexStatusResponse,
    IndexingProgress, InfoRequest, InfoResponse, ProjectTarget, ProtocolErrorEvent, ProtocolFrame,
    RequestId, SearchCodeRequest, SearchCodeResponse, SearchResultRow, SmartSearchRequest,
    SmartSearchResponse, SmartSearchResults, StatusLevel, StreamId, DAEMON_PROTOCOL_VERSION,
};
