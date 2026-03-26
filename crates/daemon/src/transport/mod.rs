use std::path::{Path, PathBuf};

use anyhow::Result;
use tokio::sync::watch;

use crate::service::DaemonService;

pub mod uds;

#[derive(Debug, Clone)]
pub enum ListenerEndpoint {
    UnixSocket(PathBuf),
}

impl ListenerEndpoint {
    pub fn unix(path: impl Into<PathBuf>) -> Self {
        Self::UnixSocket(path.into())
    }

    pub fn path(&self) -> &Path {
        match self {
            Self::UnixSocket(path) => path,
        }
    }
}

pub enum BoundTransport {
    Unix(uds::UdsListener),
}

pub async fn bind(endpoint: &ListenerEndpoint, service: DaemonService) -> Result<BoundTransport> {
    match endpoint {
        ListenerEndpoint::UnixSocket(path) => Ok(BoundTransport::Unix(uds::UdsListener::bind(
            path.clone(),
            service,
        )?)),
    }
}

impl BoundTransport {
    pub async fn run(self, shutdown: watch::Receiver<bool>) -> Result<()> {
        match self {
            Self::Unix(listener) => listener.run(shutdown).await,
        }
    }
}
