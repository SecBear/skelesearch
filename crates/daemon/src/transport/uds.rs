use std::path::PathBuf;

use anyhow::Context as _;
use tokio::sync::watch;

#[cfg(unix)]
use tokio::net::{UnixListener, UnixStream};

pub struct UdsListener {
    socket_path: PathBuf,
    #[cfg(unix)]
    listener: UnixListener,
}

impl UdsListener {
    pub fn bind(socket_path: PathBuf) -> anyhow::Result<Self> {
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
                listener,
            })
        }

        #[cfg(not(unix))]
        {
            let _ = socket_path;
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
                        tracing::info!(
                            socket_path = %self.socket_path.display(),
                            peer = ?peer,
                            "daemon client connected"
                        );
                        tokio::spawn(async move {
                            if let Err(err) = handle_client_stub(stream).await {
                                tracing::debug!(error = %err, "daemon client stub handler failed");
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
async fn handle_client_stub(mut stream: UnixStream) -> anyhow::Result<()> {
    use tokio::io::AsyncWriteExt as _;

    stream
        .shutdown()
        .await
        .context("shutdown daemon client stream")?;
    Ok(())
}
