use std::path::PathBuf;

use anyhow::Context as _;
use skelesearch_service::{
    DaemonErrorCode, DaemonErrorResponse, DaemonEvent, DaemonRequest, DaemonResponse,
    ProtocolErrorEvent, ProtocolFrame, RequestId, StreamId,
};
use tokio::sync::watch;

use crate::service::DaemonService;

#[cfg(unix)]
use tokio::{
    io::{AsyncBufReadExt as _, AsyncWrite, AsyncWriteExt as _, BufReader},
    net::{UnixListener, UnixStream},
};

pub struct UdsListener {
    socket_path: PathBuf,
    service: DaemonService,
    #[cfg(unix)]
    listener: UnixListener,
}

impl UdsListener {
    pub fn bind(socket_path: PathBuf, service: DaemonService) -> anyhow::Result<Self> {
        if let Some(parent) = socket_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create daemon socket directory '{}'", parent.display()))?;
        }

        #[cfg(unix)]
        {
            let listener = UnixListener::bind(&socket_path)
                .with_context(|| format!("bind unix socket '{}'", socket_path.display()))?;
            Ok(Self {
                socket_path,
                service,
                listener,
            })
        }

        #[cfg(not(unix))]
        {
            let _ = socket_path;
            let _ = service;
            anyhow::bail!("unix sockets are not supported on this platform")
        }
    }

    pub async fn run(self, mut shutdown: watch::Receiver<bool>) -> anyhow::Result<()> {
        #[cfg(unix)]
        {
            loop {
                tokio::select! {
                    changed = shutdown.changed() => {
                        if changed.is_err() || *shutdown.borrow() {
                            tracing::info!(socket_path = %self.socket_path.display(), "stopping daemon socket listener");
                            break;
                        }
                    }
                    accepted = self.listener.accept() => {
                        let (stream, peer) = accepted
                            .with_context(|| format!("accept daemon client on '{}'", self.socket_path.display()))?;
                        let service = self.service.clone();
                        tracing::info!(
                            socket_path = %self.socket_path.display(),
                            peer = ?peer,
                            "daemon client connected"
                        );
                        tokio::spawn(async move {
                            let result = handle_client(stream, service).await;
                            if let Err(err) = result {
                                tracing::warn!(peer = ?peer, error = %err, "daemon client connection failed");
                            } else {
                                tracing::info!(peer = ?peer, "daemon client disconnected");
                            }
                        });
                    }
                }
            }
            Ok(())
        }

        #[cfg(not(unix))]
        {
            let _ = shutdown;
            anyhow::bail!("unix sockets are not supported on this platform")
        }
    }
}

#[cfg(unix)]
async fn handle_client(stream: UnixStream, service: DaemonService) -> anyhow::Result<()> {
    let (read_half, mut write_half) = stream.into_split();
    let mut lines = BufReader::new(read_half).lines();

    while let Some(line) = lines
        .next_line()
        .await
        .context("read daemon client request frame")?
    {
        if line.trim().is_empty() {
            continue;
        }

        let outgoing = match serde_json::from_str::<ProtocolFrame>(&line) {
            Ok(frame) => handle_incoming_frame(frame, &service).await,
            Err(err) => {
                tracing::warn!(error = %err, "invalid protocol frame JSON");
                vec![protocol_error_frame(
                    format!("invalid protocol frame JSON: {err}"),
                    None,
                )]
            }
        };

        for frame in outgoing {
            write_frame(&mut write_half, &frame).await?;
        }
    }

    Ok(())
}

#[cfg(unix)]
async fn handle_incoming_frame(frame: ProtocolFrame, service: &DaemonService) -> Vec<ProtocolFrame> {
    match frame {
        request @ ProtocolFrame::Request { .. } => {
            let (request_id, method) = match &request {
                ProtocolFrame::Request { id, request } => (*id, request_method_name(request)),
                _ => unreachable!(),
            };
            tracing::info!(request_id = request_id.0, method, "daemon request received");
            service.handle_request_frame(request).await.into_frames().collect()
        }
        ProtocolFrame::Ping => vec![ProtocolFrame::Pong],
        ProtocolFrame::Cancel { id } => {
            tracing::info!(request_id = id.0, "daemon request cancel received");
            vec![ProtocolFrame::Response {
                id,
                response: DaemonResponse::Error(DaemonErrorResponse {
                    code: DaemonErrorCode::BadRequest,
                    message: "cancel is not supported by skelesearchd in this phase".to_string(),
                    details: None,
                    retryable: false,
                }),
            }]
        }
        other => vec![protocol_error_frame(
            format!(
                "client frames of kind '{}' are not accepted by skelesearchd",
                frame_kind(&other)
            ),
            None,
        )],
    }
}

#[cfg(unix)]
async fn write_frame<W>(write_half: &mut W, frame: &ProtocolFrame) -> anyhow::Result<()>
where
    W: AsyncWrite + Unpin,
{
    let encoded = serde_json::to_string(frame).context("encode daemon response frame")?;
    write_half
        .write_all(encoded.as_bytes())
        .await
        .context("write daemon response frame")?;
    write_half
        .write_all(b"\n")
        .await
        .context("write daemon response frame delimiter")?;
    write_half
        .flush()
        .await
        .context("flush daemon response frame")?;
    Ok(())
}

fn protocol_error_frame(message: impl Into<String>, request_id: Option<RequestId>) -> ProtocolFrame {
    ProtocolFrame::Event {
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

fn request_method_name(request: &DaemonRequest) -> &'static str {
    match request {
        DaemonRequest::Handshake(_) => "handshake",
        DaemonRequest::Info(_) => "info",
        DaemonRequest::IndexCodebase(_) => "index_codebase",
        DaemonRequest::IndexStatus(_) => "index_status",
        DaemonRequest::SearchCode(_) => "search_code",
        DaemonRequest::SmartSearch(_) => "smart_search",
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    use skelesearch_service::{
        DAEMON_PROTOCOL_VERSION, DaemonRequest, DaemonResponse, HandshakeRequest, IndexStatusRequest,
        InfoRequest, ProjectTarget, ProtocolFrame, RequestId,
    };
    use tempfile::tempdir;
    use tokio::net::UnixStream;

    #[tokio::test]
    async fn uds_listener_serves_framed_handshake_and_info_on_same_connection() {
        let temp = tempdir().expect("tempdir");
        let socket_path = temp.path().join("daemon.sock");

        let listener = UdsListener::bind(socket_path.clone(), DaemonService::default())
            .expect("bind listener");

        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let run_task = tokio::spawn(async move { listener.run(shutdown_rx).await });

        let stream = UnixStream::connect(&socket_path)
            .await
            .expect("connect to daemon socket");
        let (read_half, mut write_half) = stream.into_split();
        let mut lines = BufReader::new(read_half).lines();

        let handshake_frame = ProtocolFrame::Request {
            id: RequestId(1),
            request: DaemonRequest::Handshake(HandshakeRequest {
                protocol_version: DAEMON_PROTOCOL_VERSION.to_string(),
                client_name: Some("test-client".to_string()),
                client_version: Some("0.1.0".to_string()),
            }),
        };
        send_frame(&mut write_half, &handshake_frame).await;

        let handshake_response = read_frame(&mut lines).await;
        match handshake_response {
            ProtocolFrame::Response {
                id,
                response: DaemonResponse::Handshake(response),
            } => {
                assert_eq!(id, RequestId(1));
                assert_eq!(response.protocol_version, DAEMON_PROTOCOL_VERSION);
                assert!(response.capabilities.info);
                assert!(response.capabilities.index_codebase);
                assert!(response.capabilities.index_status);
                assert!(response.capabilities.search_code);
                assert!(!response.capabilities.smart_search);
            }
            other => panic!("expected handshake response frame, got {other:?}"),
        }

        let info_frame = ProtocolFrame::Request {
            id: RequestId(2),
            request: DaemonRequest::Info(InfoRequest {}),
        };
        send_frame(&mut write_half, &info_frame).await;

        let info_response = read_frame(&mut lines).await;
        match info_response {
            ProtocolFrame::Response {
                id,
                response: DaemonResponse::Info(_),
            } => {
                assert_eq!(id, RequestId(2));
            }
            other => panic!("expected info response frame, got {other:?}"),
        }

        drop(write_half);
        drop(lines);

        let _ = shutdown_tx.send(true);
        run_task
            .await
            .expect("listener join")
            .expect("listener run");
    }

    #[tokio::test]
    async fn uds_listener_returns_index_status_defaults_for_unindexed_project() {
        let temp = tempdir().expect("tempdir");
        let socket_path = temp.path().join("daemon.sock");
        let repo_root = temp.path().join("repo");
        std::fs::create_dir_all(&repo_root).expect("create repo root");

        let listener = UdsListener::bind(socket_path.clone(), DaemonService::default())
            .expect("bind listener");

        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let run_task = tokio::spawn(async move { listener.run(shutdown_rx).await });

        let stream = UnixStream::connect(&socket_path)
            .await
            .expect("connect to daemon socket");
        let (read_half, mut write_half) = stream.into_split();
        let mut lines = BufReader::new(read_half).lines();

        let status_frame = ProtocolFrame::Request {
            id: RequestId(11),
            request: DaemonRequest::IndexStatus(IndexStatusRequest {
                target: ProjectTarget::RootPath {
                    root_path: repo_root.to_string_lossy().into_owned(),
                    logical_id: None,
                },
            }),
        };
        send_frame(&mut write_half, &status_frame).await;

        let status_response = read_frame(&mut lines).await;
        match status_response {
            ProtocolFrame::Response {
                id,
                response: DaemonResponse::IndexStatus(status),
            } => {
                assert_eq!(id, RequestId(11));
                assert_eq!(status.indexed_files, 0);
                assert_eq!(status.total_chunks, 0);
                assert_eq!(status.last_indexed, None);
                assert_eq!(status.estimated_stale, 0);
                assert!(!status.watching);
                assert!(status.indexing.is_none());
            }
            other => panic!("expected index status response frame, got {other:?}"),
        }

        drop(write_half);
        drop(lines);

        let _ = shutdown_tx.send(true);
        run_task
            .await
            .expect("listener join")
            .expect("listener run");
    }

    async fn send_frame<W>(write: &mut W, frame: &ProtocolFrame)
    where
        W: AsyncWrite + Unpin,
    {
        let encoded = serde_json::to_string(frame).expect("serialize frame");
        write
            .write_all(encoded.as_bytes())
            .await
            .expect("write frame");
        write
            .write_all(b"\n")
            .await
            .expect("write frame delimiter");
        write.flush().await.expect("flush frame");
    }

    async fn read_frame(
        lines: &mut tokio::io::Lines<BufReader<tokio::net::unix::OwnedReadHalf>>,
    ) -> ProtocolFrame {
        let line = lines
            .next_line()
            .await
            .expect("read response line")
            .expect("response line");
        serde_json::from_str(&line).expect("decode response frame")
    }
}
