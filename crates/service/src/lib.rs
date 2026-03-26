pub mod project;
pub mod protocol;

pub use project::{ProjectKey, ProjectKeyError};
pub use protocol::{
    DaemonCapabilities, DaemonErrorCode, DaemonErrorResponse, DaemonRequest, DaemonResponse,
    HandshakeRequest, HandshakeResponse, IndexCodebaseRequest, IndexCodebaseResponse,
    IndexStatusRequest, IndexStatusResponse, IndexingProgress, InfoRequest, InfoResponse,
    ProjectTarget, SearchCodeRequest, SearchCodeResponse, SearchResultRow, SmartSearchRequest,
    SmartSearchResponse, SmartSearchResults, DAEMON_PROTOCOL_VERSION,
};
