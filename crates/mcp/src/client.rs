use std::{
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

use anyhow::Context as _;
use async_trait::async_trait;
use skelesearch_service::{
    DaemonErrorResponse, DaemonEvent, DaemonRequest, DaemonResponse, HandshakeRequest,
    HandshakeResponse, HeartbeatRequest, HeartbeatResponse, IndexCodebaseRequest,
    IndexCodebaseResponse, IndexStatusRequest, IndexStatusResponse, ProjectTarget, ProtocolFrame,
    RegisterClientRequest, RegisterClientResponse, RequestId, SearchCodeRequest,
    SearchCodeResponse, UnregisterClientRequest, UnregisterClientResponse, DAEMON_PROTOCOL_VERSION,
};
#[cfg(test)]
use skelesearch_service::{InfoRequest, InfoResponse};
use tokio::{
    io::{
        AsyncBufReadExt as _, AsyncRead, AsyncWrite, AsyncWriteExt as _, BufReader, ReadHalf,
        WriteHalf,
    },
    sync::Mutex,
};

const DAEMON_SOCKET_ENV: &str = "SKELESEARCH_DAEMON_SOCKET";
const DAEMON_PROGRAM_ENV: &str = "SKELESEARCH_DAEMON_PROGRAM";
const DEFAULT_SOCKET_SUBPATH: &str = ".cache/skelesearch/daemon.sock";

pub(super) trait AsyncIo: AsyncRead + AsyncWrite + Unpin + Send {}
impl<T> AsyncIo for T where T: AsyncRead + AsyncWrite + Unpin + Send {}

pub(super) type BoxedIo = Box<dyn AsyncIo>;

enum ClientConnection {
    Ready {
        lines: tokio::io::Lines<BufReader<ReadHalf<BoxedIo>>>,
        writer: WriteHalf<BoxedIo>,
        handshake: Option<HandshakeResponse>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DaemonEndpoint {
    UnixSocket(PathBuf),
    TcpSocket(std::net::SocketAddr),
}

impl DaemonEndpoint {
    pub fn unix(path: impl Into<PathBuf>) -> Self {
        Self::UnixSocket(path.into())
    }

    pub fn from_env() -> Self {
        if let Ok(raw) = std::env::var(DAEMON_SOCKET_ENV) {
            let trimmed = raw.trim();
            if !trimmed.is_empty() {
                if let Some(rest) = trimmed.strip_prefix("tcp://") {
                    if let Ok(addr) = rest.parse::<std::net::SocketAddr>() {
                        return Self::TcpSocket(addr);
                    }
                }
                return Self::unix(resolve_socket_path(PathBuf::from(trimmed)));
            }
        }

        let home_dir = std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."));
        Self::unix(absolutize(home_dir.join(DEFAULT_SOCKET_SUBPATH)))
    }
}

impl std::fmt::Display for DaemonEndpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnixSocket(path) => write!(f, "{}", path.display()),
            Self::TcpSocket(addr) => write!(f, "{addr}"),
        }
    }
}

#[async_trait]
pub(super) trait DaemonConnector: Send + Sync + 'static {
    async fn connect(&self, endpoint: &DaemonEndpoint) -> anyhow::Result<BoxedIo>;
    async fn start_daemon(&self, endpoint: &DaemonEndpoint) -> anyhow::Result<()> {
        let _ = endpoint;
        anyhow::bail!("daemon auto-start is not supported by this connector")
    }
}

#[derive(Debug, Clone, Default)]
pub struct TokioDaemonConnector;

#[async_trait]
impl DaemonConnector for TokioDaemonConnector {
    async fn connect(&self, endpoint: &DaemonEndpoint) -> anyhow::Result<BoxedIo> {
        match endpoint {
            DaemonEndpoint::UnixSocket(path) => {
                #[cfg(unix)]
                {
                    let stream =
                        tokio::net::UnixStream::connect(path)
                            .await
                            .with_context(|| {
                                format!("connect to daemon unix socket '{}'", path.display())
                            })?;
                    Ok(Box::new(stream))
                }

                #[cfg(not(unix))]
                {
                    let _ = path;
                    anyhow::bail!("unix sockets are not supported on this platform")
                }
            }
            DaemonEndpoint::TcpSocket(addr) => {
                let stream = tokio::net::TcpStream::connect(addr)
                    .await
                    .with_context(|| format!("connect to daemon tcp endpoint '{addr}'"))?;
                Ok(Box::new(stream))
            }
        }
    }

    async fn start_daemon(&self, endpoint: &DaemonEndpoint) -> anyhow::Result<()> {
        let program = daemon_program_for_endpoint(endpoint)?;
        let mut command = tokio::process::Command::new(&program);
        command
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .arg("--managed")
            .arg("--idle-timeout-seconds")
            .arg("300");

        if let DaemonEndpoint::UnixSocket(path) = endpoint {
            command.arg("--socket").arg(path);
        }

        command
            .spawn()
            .with_context(|| format!("spawn skelesearch daemon via '{}'", program.display()))?;

        Ok(())
    }
}

pub struct DaemonClient<C = TokioDaemonConnector> {
    endpoint: DaemonEndpoint,
    connector: C,
    connection: Mutex<Option<ClientConnection>>,
    next_request_id: AtomicU64,
    client_name: Option<String>,
    client_version: Option<String>,
}

impl DaemonClient<TokioDaemonConnector> {
    pub fn from_env() -> Self {
        Self::new(DaemonEndpoint::from_env(), TokioDaemonConnector)
    }
}

impl<C> DaemonClient<C>
where
    C: DaemonConnector,
{
    pub fn new(endpoint: DaemonEndpoint, connector: C) -> Self {
        Self {
            endpoint,
            connector,
            connection: Mutex::new(None),
            next_request_id: AtomicU64::new(1),
            client_name: Some("skelesearch-mcp".to_string()),
            client_version: Some(env!("CARGO_PKG_VERSION").to_string()),
        }
    }

    pub fn endpoint(&self) -> &DaemonEndpoint {
        &self.endpoint
    }

    pub async fn handshake(&self) -> anyhow::Result<HandshakeResponse> {
        let mut guard = self.connection.lock().await;
        match self.ensure_handshake_locked(&mut guard).await {
            Ok(response) => Ok(response),
            Err(err) => {
                *guard = None;
                Err(err)
            }
        }
    }

    #[cfg(test)]
    pub async fn info(&self) -> anyhow::Result<InfoResponse> {
        let response = self
            .request_response(DaemonRequest::Info(InfoRequest::default()))
            .await?;
        match response {
            DaemonResponse::Info(response) => Ok(response),
            DaemonResponse::Error(err) => Err(daemon_method_error("info", &err)),
            other => anyhow::bail!(
                "daemon protocol violation: expected info response, got {}",
                response_kind(&other)
            ),
        }
    }

    pub async fn register_client(
        &self,
        client_name: Option<String>,
        client_version: Option<String>,
    ) -> anyhow::Result<RegisterClientResponse> {
        let handshake = self.handshake().await?;
        if !handshake.capabilities.register_client {
            anyhow::bail!(
                "daemon at {} does not advertise register_client capability",
                self.endpoint
            );
        }

        let response = self
            .request_response(DaemonRequest::RegisterClient(RegisterClientRequest {
                client_name,
                client_version,
            }))
            .await?;

        match response {
            DaemonResponse::RegisterClient(response) => Ok(response),
            DaemonResponse::Error(err) => Err(daemon_method_error("register_client", &err)),
            other => anyhow::bail!(
                "daemon protocol violation: expected register_client response, got {}",
                response_kind(&other)
            ),
        }
    }

    pub async fn heartbeat(
        &self,
        session_id: impl Into<String>,
    ) -> anyhow::Result<HeartbeatResponse> {
        let handshake = self.handshake().await?;
        if !handshake.capabilities.heartbeat {
            anyhow::bail!(
                "daemon at {} does not advertise heartbeat capability",
                self.endpoint
            );
        }

        let response = self
            .request_response(DaemonRequest::Heartbeat(HeartbeatRequest {
                session_id: session_id.into(),
            }))
            .await?;

        match response {
            DaemonResponse::Heartbeat(response) => Ok(response),
            DaemonResponse::Error(err) => Err(daemon_method_error("heartbeat", &err)),
            other => anyhow::bail!(
                "daemon protocol violation: expected heartbeat response, got {}",
                response_kind(&other)
            ),
        }
    }

    pub async fn unregister_client(
        &self,
        session_id: impl Into<String>,
    ) -> anyhow::Result<UnregisterClientResponse> {
        let handshake = self.handshake().await?;
        if !handshake.capabilities.unregister_client {
            anyhow::bail!(
                "daemon at {} does not advertise unregister_client capability",
                self.endpoint
            );
        }

        let response = self
            .request_response(DaemonRequest::UnregisterClient(UnregisterClientRequest {
                session_id: session_id.into(),
            }))
            .await?;

        match response {
            DaemonResponse::UnregisterClient(response) => Ok(response),
            DaemonResponse::Error(err) => Err(daemon_method_error("unregister_client", &err)),
            other => anyhow::bail!(
                "daemon protocol violation: expected unregister_client response, got {}",
                response_kind(&other)
            ),
        }
    }

    pub async fn index_codebase(
        &self,
        target: ProjectTarget,
        provider: Option<String>,
    ) -> anyhow::Result<IndexCodebaseResponse> {
        let handshake = self.handshake().await?;
        if !handshake.capabilities.index_codebase {
            anyhow::bail!(
                "daemon at {} does not advertise index_codebase capability",
                self.endpoint
            );
        }

        let response = self
            .request_response(DaemonRequest::IndexCodebase(IndexCodebaseRequest {
                target,
                provider,
            }))
            .await?;

        match response {
            DaemonResponse::IndexCodebase(response) => Ok(response),
            DaemonResponse::Error(err) => Err(daemon_method_error("index_codebase", &err)),
            other => anyhow::bail!(
                "daemon protocol violation: expected index_codebase response, got {}",
                response_kind(&other)
            ),
        }
    }

    pub async fn index_status(&self, target: ProjectTarget) -> anyhow::Result<IndexStatusResponse> {
        let handshake = self.handshake().await?;
        if !handshake.capabilities.index_status {
            anyhow::bail!(
                "daemon at {} does not advertise index_status capability",
                self.endpoint
            );
        }

        let response = self
            .request_response(DaemonRequest::IndexStatus(IndexStatusRequest { target }))
            .await?;

        match response {
            DaemonResponse::IndexStatus(response) => Ok(response),
            DaemonResponse::Error(err) => Err(daemon_method_error("index_status", &err)),
            other => anyhow::bail!(
                "daemon protocol violation: expected index_status response, got {}",
                response_kind(&other)
            ),
        }
    }

    pub async fn search_code(
        &self,
        request: SearchCodeRequest,
    ) -> anyhow::Result<SearchCodeResponse> {
        let handshake = self.handshake().await?;
        if !handshake.capabilities.search_code {
            anyhow::bail!(
                "daemon at {} does not advertise search_code capability",
                self.endpoint
            );
        }

        let response = self
            .request_response(DaemonRequest::SearchCode(request))
            .await?;

        match response {
            DaemonResponse::SearchCode(response) => Ok(response),
            DaemonResponse::Error(err) => Err(daemon_method_error("search_code", &err)),
            other => anyhow::bail!(
                "daemon protocol violation: expected search_code response, got {}",
                response_kind(&other)
            ),
        }
    }

    async fn request_response(&self, request: DaemonRequest) -> anyhow::Result<DaemonResponse> {
        let mut guard = self.connection.lock().await;

        if let Err(err) = self.ensure_handshake_locked(&mut guard).await {
            *guard = None;
            return Err(err);
        }

        let result = self.send_request_locked(&mut guard, request).await;
        if result.is_err() {
            *guard = None;
        }
        result
    }

    async fn ensure_handshake_locked(
        &self,
        guard: &mut Option<ClientConnection>,
    ) -> anyhow::Result<HandshakeResponse> {
        if guard.is_none() {
            *guard = Some(self.connect().await?);
        }

        let ClientConnection::Ready {
            lines: _,
            writer: _,
            handshake,
        } = guard
            .as_mut()
            .expect("connection is initialized when checking handshake");

        if let Some(response) = handshake.as_ref() {
            return Ok(response.clone());
        }

        let response = self
            .send_request_locked(
                guard,
                DaemonRequest::Handshake(HandshakeRequest::new(
                    self.client_name.clone(),
                    self.client_version.clone(),
                )),
            )
            .await?;

        let handshake_response = match response {
            DaemonResponse::Handshake(response) => response,
            DaemonResponse::Error(err) => return Err(daemon_method_error("handshake", &err)),
            other => {
                anyhow::bail!(
                    "daemon protocol violation: expected handshake response, got {}",
                    response_kind(&other)
                )
            }
        };

        if handshake_response.protocol_version != DAEMON_PROTOCOL_VERSION {
            anyhow::bail!(
                "daemon protocol version mismatch: daemon={}, client={}",
                handshake_response.protocol_version,
                DAEMON_PROTOCOL_VERSION
            );
        }

        let ClientConnection::Ready {
            lines: _,
            writer: _,
            handshake,
        } = guard
            .as_mut()
            .expect("connection exists after handshake request");
        *handshake = Some(handshake_response.clone());

        Ok(handshake_response)
    }

    async fn connect(&self) -> anyhow::Result<ClientConnection> {
        match self.connect_once().await {
            Ok(connection) => Ok(connection),
            Err(first_err) => {
                // Derive a short jitter from the low bits of the current wall clock
                // to decohere concurrent MCP sessions that all detect a dead daemon
                // at the same instant. No `rand` dependency needed.
                let jitter_ms = u64::from(
                    (std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .subsec_nanos()
                        % 400) as u16,
                );
                tokio::time::sleep(std::time::Duration::from_millis(jitter_ms)).await;

                // Re-probe after jitter: a sibling session may have already
                // started the daemon during our sleep.
                if let Ok(connection) = self.connect_once().await {
                    return Ok(connection);
                }

                self.connector
                    .start_daemon(&self.endpoint)
                    .await
                    .with_context(|| {
                        format!(
"unable to reach skelesearch daemon at {}. attempted auto-start via `skelesearchd`"
, self.endpoint)
                    })?;

                // Poll for socket readiness; 50×100ms = 5 s.
                for _ in 0..50 {
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                    if let Ok(connection) = self.connect_once().await {
                        return Ok(connection);
                    }
                }

                Err(first_err).with_context(|| {
                    format!(
                        "unable to reach skelesearch daemon at {} after auto-start attempt",
                        self.endpoint
                    )
                })
            }
        }
    }

    async fn connect_once(&self) -> anyhow::Result<ClientConnection> {
        let stream = self
            .connector
            .connect(&self.endpoint)
            .await
            .with_context(|| {
                format!(
"unable to reach skelesearch daemon at {}. Start it with `skelesearchd` or set {}"
 , self.endpoint, DAEMON_SOCKET_ENV)
            })?;
        let (read_half, write_half) = tokio::io::split(stream);
        let lines = BufReader::new(read_half).lines();

        Ok(ClientConnection::Ready {
            lines,
            writer: write_half,
            handshake: None,
        })
    }
    async fn send_request_locked(
        &self,
        guard: &mut Option<ClientConnection>,
        request: DaemonRequest,
    ) -> anyhow::Result<DaemonResponse> {
        let request_id = RequestId(self.next_request_id.fetch_add(1, Ordering::Relaxed));
        let frame = ProtocolFrame::Request {
            id: request_id,
            request,
        };

        let ClientConnection::Ready {
            lines,
            writer,
            handshake: _,
        } = guard
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("daemon connection was not initialized"))?;

        write_frame(writer, &frame).await?;

        loop {
            let line = lines
                .next_line()
                .await
                .context("read daemon response frame")?
                .ok_or_else(|| {
                    anyhow::anyhow!("daemon connection closed while waiting for response")
                })?;

            if line.trim().is_empty() {
                continue;
            }

            let incoming: ProtocolFrame = serde_json::from_str(&line)
                .with_context(|| format!("decode daemon response frame '{line}'"))?;

            match incoming {
                ProtocolFrame::Response { id, response } if id == request_id => {
                    return Ok(response)
                }
                ProtocolFrame::Response { id, .. } => {
                    anyhow::bail!(
                        "daemon protocol violation: received response id {} while waiting for {}",
                        id.0,
                        request_id.0
                    )
                }
                ProtocolFrame::Event {
                    stream_id: _,
                    event: DaemonEvent::ProtocolError(event),
                } => {
                    if event.request_id.is_none() || event.request_id == Some(request_id) {
                        return Err(daemon_method_error("protocol", &event.error));
                    }
                }
                ProtocolFrame::Ping => {
                    write_frame(writer, &ProtocolFrame::Pong).await?;
                }
                ProtocolFrame::Pong => {}
                ProtocolFrame::Event { .. } => {}
                ProtocolFrame::Request { .. } | ProtocolFrame::Cancel { .. } => {
                    anyhow::bail!(
                        "daemon protocol violation: received unexpected '{}' frame",
                        frame_kind(&incoming)
                    )
                }
            }
        }
    }
}

fn write_frame<'a>(
    writer: &'a mut WriteHalf<BoxedIo>,
    frame: &'a ProtocolFrame,
) -> impl std::future::Future<Output = anyhow::Result<()>> + 'a {
    async move {
        let encoded = serde_json::to_string(frame).context("encode daemon request frame")?;
        writer
            .write_all(encoded.as_bytes())
            .await
            .context("write daemon request frame")?;
        writer
            .write_all(b"\n")
            .await
            .context("write daemon request frame delimiter")?;
        writer.flush().await.context("flush daemon request frame")?;
        Ok(())
    }
}

fn daemon_method_error(method: &str, err: &DaemonErrorResponse) -> anyhow::Error {
    anyhow::anyhow!(
        "daemon {} failed (code={:?}, retryable={}): {}",
        method,
        err.code,
        err.retryable,
        err.message
    )
}

fn response_kind(response: &DaemonResponse) -> &'static str {
    match response {
        DaemonResponse::Handshake(_) => "handshake",
        DaemonResponse::Info(_) => "info",
        DaemonResponse::RegisterClient(_) => "register_client",
        DaemonResponse::Heartbeat(_) => "heartbeat",
        DaemonResponse::UnregisterClient(_) => "unregister_client",
        DaemonResponse::IndexCodebase(_) => "index_codebase",
        DaemonResponse::IndexStatus(_) => "index_status",
        DaemonResponse::SearchCode(_) => "search_code",
        DaemonResponse::SmartSearch(_) => "smart_search",
        DaemonResponse::Error(_) => "error",
    }
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

fn resolve_socket_path(path: PathBuf) -> PathBuf {
    absolutize(expand_tilde(path))
}

fn expand_tilde(path: PathBuf) -> PathBuf {
    let Some(raw) = path.to_str() else {
        return path;
    };

    if raw == "~" {
        return std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."));
    }

    if let Some(rest) = raw.strip_prefix("~/") {
        return std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."))
            .join(rest);
    }

    path
}

fn absolutize(path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        return path;
    }

    std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join(path)
}

fn daemon_program_for_endpoint(endpoint: &DaemonEndpoint) -> anyhow::Result<PathBuf> {
    let _ = endpoint;
    if let Ok(raw) = std::env::var(DAEMON_PROGRAM_ENV) {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            return Ok(PathBuf::from(trimmed));
        }
    }

    let current_exe = std::env::current_exe().context("resolve current executable path")?;
    let current_dir = current_exe
        .parent()
        .map(PathBuf::from)
        .context("resolve current executable directory")?;

    let sibling_name = if cfg!(windows) {
        "skelesearchd.exe"
    } else {
        "skelesearchd"
    };
    let sibling = current_dir.join(sibling_name);
    if sibling.exists() {
        return Ok(sibling);
    }

    Ok(PathBuf::from("skelesearchd"))
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    use std::sync::{Mutex, OnceLock};

    use skelesearch_service::protocol::{IndexFreshnessState, IndexingState};
    use skelesearch_service::{
        DaemonCapabilities, DaemonRequest, DaemonResponse, HandshakeResponse, IndexStatusResponse,
    };
    use tempfile::tempdir;
    use tokio::{io::BufReader, net::UnixListener};

    use std::sync::Arc;
    use tokio::sync::Mutex as AsyncMutex;

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[derive(Clone)]
    struct AutoStartConnector {
        socket_path: PathBuf,
        state: Arc<AsyncMutex<AutoStartState>>,
    }

    struct AutoStartState {
        started: bool,
        start_calls: usize,
        server_task: Option<tokio::task::JoinHandle<()>>,
    }

    impl AutoStartConnector {
        fn new(socket_path: PathBuf) -> Self {
            Self {
                socket_path,
                state: Arc::new(AsyncMutex::new(AutoStartState {
                    started: false,
                    start_calls: 0,
                    server_task: None,
                })),
            }
        }

        async fn start_calls(&self) -> usize {
            self.state.lock().await.start_calls
        }

        async fn join_server(&self) {
            if let Some(task) = self.state.lock().await.server_task.take() {
                task.await.expect("server task join");
            }
        }
    }

    #[async_trait]
    impl DaemonConnector for AutoStartConnector {
        async fn connect(&self, endpoint: &DaemonEndpoint) -> anyhow::Result<BoxedIo> {
            assert_eq!(endpoint, &DaemonEndpoint::unix(self.socket_path.clone()));
            if !self.state.lock().await.started {
                anyhow::bail!("daemon not started")
            }
            let stream = tokio::net::UnixStream::connect(&self.socket_path)
                .await
                .expect("connect to auto-started daemon");
            Ok(Box::new(stream))
        }

        async fn start_daemon(&self, endpoint: &DaemonEndpoint) -> anyhow::Result<()> {
            assert_eq!(endpoint, &DaemonEndpoint::unix(self.socket_path.clone()));
            let mut state = self.state.lock().await;
            state.start_calls += 1;
            if state.started {
                return Ok(());
            }
            state.started = true;

            let listener = UnixListener::bind(&self.socket_path).expect("bind test listener");
            state.server_task = Some(tokio::spawn(async move {
                let (stream, _) = listener.accept().await.expect("accept client");
                let (read_half, mut write_half) = stream.into_split();
                let mut lines = BufReader::new(read_half).lines();

                let mut handled = 0usize;
                while let Some(line) = lines.next_line().await.expect("read line") {
                    let frame: ProtocolFrame = serde_json::from_str(&line).expect("decode frame");
                    let ProtocolFrame::Request { id, request } = frame else {
                        panic!("expected request frame");
                    };

                    let response = match request {
                        DaemonRequest::Handshake(_) => ProtocolFrame::Response {
                            id,
                            response: DaemonResponse::Handshake(HandshakeResponse {
                                protocol_version: DAEMON_PROTOCOL_VERSION.to_string(),
                                server_name: "auto-start-daemon".to_string(),
                                server_version: "0.1.0".to_string(),
                                capabilities: DaemonCapabilities {
                                    info: true,
                                    index_codebase: true,
                                    index_status: true,
                                    search_code: false,
                                    smart_search: false,
                                    register_client: true,
                                    heartbeat: true,
                                    unregister_client: true,
                                },
                            }),
                        },
                        DaemonRequest::Info(_) => ProtocolFrame::Response {
                            id,
                            response: DaemonResponse::Info(InfoResponse {
                                protocol_version: DAEMON_PROTOCOL_VERSION.to_string(),
                                server_name: "auto-start-daemon".to_string(),
                                server_version: "0.1.0".to_string(),
                                capabilities: DaemonCapabilities {
                                    info: true,
                                    index_codebase: true,
                                    index_status: true,
                                    search_code: false,
                                    smart_search: false,
                                    register_client: true,
                                    heartbeat: true,
                                    unregister_client: true,
                                },
                            }),
                        },
                        other => panic!("unexpected request: {other:?}"),
                    };

                    let encoded = serde_json::to_string(&response).expect("encode response");
                    write_half
                        .write_all(encoded.as_bytes())
                        .await
                        .expect("write response");
                    write_half
                        .write_all(b"\n")
                        .await
                        .expect("write response delimiter");
                    write_half.flush().await.expect("flush response");

                    handled += 1;
                    if handled >= 2 {
                        break;
                    }
                }
            }));
            Ok(())
        }
    }

    #[tokio::test]
    async fn daemon_client_auto_starts_daemon_when_socket_is_missing() {
        let temp = tempdir().expect("tempdir");
        let socket_path = temp.path().join("daemon.sock");
        let connector = AutoStartConnector::new(socket_path.clone());
        let client = DaemonClient::new(DaemonEndpoint::unix(socket_path), connector.clone());

        let info = client.info().await.expect("info response");

        assert_eq!(info.server_name, "auto-start-daemon");
        assert_eq!(connector.start_calls().await, 1);
        connector.join_server().await;
    }

    #[test]
    fn daemon_program_prefers_env_override() {
        let _guard = env_lock().lock().expect("env lock");
        let program = "/tmp/custom-skelesearchd";
        std::env::set_var(DAEMON_PROGRAM_ENV, program);

        let resolved = daemon_program_for_endpoint(&DaemonEndpoint::unix("/tmp/daemon.sock"))
            .expect("resolve daemon program");

        std::env::remove_var(DAEMON_PROGRAM_ENV);
        assert_eq!(resolved, PathBuf::from(program));
    }

    #[tokio::test]
    async fn daemon_client_reuses_connection_for_handshake_info_and_index_status() {
        let temp = tempdir().expect("tempdir");
        let socket_path = temp.path().join("daemon.sock");

        let listener = UnixListener::bind(&socket_path).expect("bind test listener");
        let server_task = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("accept client");
            let (read_half, mut write_half) = stream.into_split();
            let mut lines = BufReader::new(read_half).lines();

            let mut handled = 0usize;
            while let Some(line) = lines.next_line().await.expect("read line") {
                let frame: ProtocolFrame = serde_json::from_str(&line).expect("decode frame");
                let ProtocolFrame::Request { id, request } = frame else {
                    panic!("expected request frame");
                };

                let response = match request {
                    DaemonRequest::Handshake(_) => ProtocolFrame::Response {
                        id,
                        response: DaemonResponse::Handshake(HandshakeResponse {
                            protocol_version: DAEMON_PROTOCOL_VERSION.to_string(),
                            server_name: "test-daemon".to_string(),
                            server_version: "0.1.0".to_string(),
                            capabilities: DaemonCapabilities {
                                info: true,
                                index_codebase: true,
                                index_status: true,
                                search_code: false,
                                smart_search: false,
                                register_client: true,
                                heartbeat: true,
                                unregister_client: true,
                            },
                        }),
                    },
                    DaemonRequest::Info(_) => ProtocolFrame::Response {
                        id,
                        response: DaemonResponse::Info(InfoResponse {
                            protocol_version: DAEMON_PROTOCOL_VERSION.to_string(),
                            server_name: "test-daemon".to_string(),
                            server_version: "0.1.0".to_string(),
                            capabilities: DaemonCapabilities {
                                info: true,
                                index_codebase: true,
                                index_status: true,
                                search_code: false,
                                smart_search: false,
                                register_client: true,
                                heartbeat: true,
                                unregister_client: true,
                            },
                        }),
                    },
                    DaemonRequest::IndexStatus(request) => ProtocolFrame::Response {
                        id,
                        response: DaemonResponse::IndexStatus(IndexStatusResponse {
                            project_key: match request.target {
                                ProjectTarget::RootPath {
                                    root_path,
                                    logical_id,
                                } => skelesearch_service::ProjectKey {
                                    canonical_root: root_path,
                                    logical_id,
                                },
                                ProjectTarget::ProjectKey { project_key } => project_key,
                            },
                            indexed_files: 7,
                            total_chunks: 14,
                            last_indexed: Some("2026-01-01T00:00:00Z".to_string()),
                            estimated_stale: 0,
                            freshness_state: IndexFreshnessState::Fresh,
                            freshness_checked_at: Some("2026-01-01T00:00:00Z".to_string()),
                            freshness_error: None,
                            watching: false,
                            indexing: Some(skelesearch_service::IndexingProgress {
                                status: IndexingState::Running,
                                path: "/tmp/repo".to_string(),
                                files_done: 3,
                                files_total: 10,
                                chunks_done: 6,
                                cache_hits: 1,
                                elapsed_seconds: 1.5,
                                error: None,
                            }),
                        }),
                    },
                    other => panic!("unexpected request: {other:?}"),
                };

                let encoded = serde_json::to_string(&response).expect("encode response");
                write_half
                    .write_all(encoded.as_bytes())
                    .await
                    .expect("write response");
                write_half
                    .write_all(b"\n")
                    .await
                    .expect("write response delimiter");
                write_half.flush().await.expect("flush response");

                handled += 1;
                if handled >= 3 {
                    break;
                }
            }
        });

        let client = DaemonClient::new(DaemonEndpoint::unix(socket_path), TokioDaemonConnector);

        let info = client.info().await.expect("info response");
        assert_eq!(info.server_name, "test-daemon");

        let status = client
            .index_status(ProjectTarget::RootPath {
                root_path: "/tmp/repo".to_string(),
                logical_id: None,
            })
            .await
            .expect("index status response");

        assert_eq!(status.indexed_files, 7);
        assert_eq!(status.total_chunks, 14);
        assert_eq!(status.indexing.as_ref().map(|p| p.files_done), Some(3));

        server_task.await.expect("server task join");
    }
}
