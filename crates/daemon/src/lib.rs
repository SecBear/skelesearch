pub mod service;
pub mod transport;

pub use service::{DaemonService, DaemonState, ProjectLookup, ProjectState, ServiceFrameOutcome};

use std::{
    fs::{File, OpenOptions},
    io::{ErrorKind, Write},
    path::{Path, PathBuf},
};

use anyhow::Context as _;
use fs2::FileExt;
use thiserror::Error;

const DEFAULT_SOCKET_FILE: &str = "daemon.sock";
const DEFAULT_LOG_FILE: &str = "skelesearchd.log";
const LOCK_FILE: &str = "daemon.lock";
const PID_FILE: &str = "daemon.pid";

#[derive(Debug, Clone)]
pub struct DaemonPaths {
    pub cache_dir: PathBuf,
    pub socket_path: PathBuf,
    pub log_path: PathBuf,
    pub lock_path: PathBuf,
    pub pid_path: PathBuf,
}

impl DaemonPaths {
    pub fn resolve(socket_override: Option<PathBuf>) -> anyhow::Result<Self> {
        let home_dir = resolve_home_dir();
        let default_cache = default_cache_dir_for_home(&home_dir);

        let requested_socket = socket_override
            .map(|path| expand_tilde(path, &home_dir))
            .unwrap_or_else(|| default_socket_path_for_home(&home_dir));
        let socket_path = absolutize(requested_socket)?;

        let cache_dir = socket_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or(default_cache);

        Ok(Self {
            log_path: cache_dir.join(DEFAULT_LOG_FILE),
            lock_path: cache_dir.join(LOCK_FILE),
            pid_path: cache_dir.join(PID_FILE),
            cache_dir,
            socket_path,
        })
    }
}

#[derive(Debug)]
pub struct DaemonSingleton {
    lock_file: File,
    lock_path: PathBuf,
    pid_path: PathBuf,
    socket_path: PathBuf,
    pid: u32,
}

impl DaemonSingleton {
    pub fn acquire(paths: &DaemonPaths) -> Result<Self, SingletonError> {
        std::fs::create_dir_all(&paths.cache_dir).with_context(|| {
            format!(
                "create daemon cache directory '{}'",
                paths.cache_dir.display()
            )
        })?;

        let lock_file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(&paths.lock_path)
            .with_context(|| format!("open daemon lock file '{}'", paths.lock_path.display()))?;

        match lock_file.try_lock_exclusive() {
            Ok(()) => {}
            Err(err) if err.kind() == ErrorKind::WouldBlock => {
                let pid = read_pid_file(&paths.pid_path);
                if daemon_socket_is_healthy(&paths.socket_path) {
                    return Err(SingletonError::AlreadyRunning {
                        pid,
                        socket_path: paths.socket_path.clone(),
                    });
                }

                return Err(SingletonError::StartupInProgress {
                    pid,
                    lock_path: paths.lock_path.clone(),
                    socket_path: paths.socket_path.clone(),
                });
            }
            Err(err) => {
                return Err(err).with_context(|| {
                    format!(
                        "acquire daemon singleton lock '{}'",
                        paths.lock_path.display()
                    )
                })?;
            }
        }

        cleanup_stale_socket(&paths.socket_path)?;

        let pid = std::process::id();
        write_pid_file(&paths.pid_path, pid)?;

        Ok(Self {
            lock_file,
            lock_path: paths.lock_path.clone(),
            pid_path: paths.pid_path.clone(),
            socket_path: paths.socket_path.clone(),
            pid,
        })
    }

    pub fn pid(&self) -> u32 {
        self.pid
    }

    pub fn lock_path(&self) -> &Path {
        &self.lock_path
    }
}

impl Drop for DaemonSingleton {
    fn drop(&mut self) {
        if let Err(err) = std::fs::remove_file(&self.pid_path) {
            if err.kind() != ErrorKind::NotFound {
                tracing::debug!(
                    path = %self.pid_path.display(),
                    error = %err,
                    "failed to remove daemon pid file"
                );
            }
        }

        if let Err(err) = std::fs::remove_file(&self.socket_path) {
            if err.kind() != ErrorKind::NotFound {
                tracing::debug!(
                    path = %self.socket_path.display(),
                    error = %err,
                    "failed to remove daemon socket file"
                );
            }
        }

        if let Err(err) = fs2::FileExt::unlock(&self.lock_file) {
            tracing::debug!(
                path = %self.lock_path.display(),
                error = %err,
                "failed to unlock daemon singleton lock"
            );
        }
    }
}

#[derive(Debug, Error)]
pub enum SingletonError {
    #[error("daemon already running")]
    AlreadyRunning {
        pid: Option<u32>,
        socket_path: PathBuf,
    },
    #[error("daemon startup lock is held")]
    StartupInProgress {
        pid: Option<u32>,
        lock_path: PathBuf,
        socket_path: PathBuf,
    },
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl SingletonError {
    pub fn user_message(&self) -> String {
        match self {
            Self::AlreadyRunning { pid, socket_path } => match pid {
                Some(pid) => format!(
                    "skelesearchd is already running (pid {pid}) at {}. Reuse that daemon, or stop it before launching another.",
                    socket_path.display()
                ),
                None => format!(
                    "skelesearchd is already running at {}. Reuse that daemon, or stop it before launching another.",
                    socket_path.display()
                ),
            },
            Self::StartupInProgress {
                pid,
                lock_path,
                socket_path,
            } => match pid {
                Some(pid) => format!(
                    "another skelesearchd process (pid {pid}) holds startup lock {} but is not serving {} yet; wait a moment and retry.",
                    lock_path.display(),
                    socket_path.display()
                ),
                None => format!(
                    "another skelesearchd process holds startup lock {} but is not serving {} yet; wait a moment and retry.",
                    lock_path.display(),
                    socket_path.display()
                ),
            },
            Self::Other(err) => format!("failed to start skelesearchd singleton: {err:#}"),
        }
    }
}

pub fn default_cache_dir_for_home(home: &Path) -> PathBuf {
    home.join(".cache").join("skelesearch")
}

pub fn default_socket_path_for_home(home: &Path) -> PathBuf {
    default_cache_dir_for_home(home).join(DEFAULT_SOCKET_FILE)
}

pub fn default_log_path_for_home(home: &Path) -> PathBuf {
    default_cache_dir_for_home(home).join(DEFAULT_LOG_FILE)
}

fn resolve_home_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

fn expand_tilde(path: PathBuf, home_dir: &Path) -> PathBuf {
    let Some(raw) = path.to_str() else {
        return path;
    };

    if raw == "~" {
        return home_dir.to_path_buf();
    }

    if let Some(rest) = raw.strip_prefix("~/") {
        return home_dir.join(rest);
    }

    path
}

fn absolutize(path: PathBuf) -> anyhow::Result<PathBuf> {
    if path.is_absolute() {
        return Ok(path);
    }

    let cwd = std::env::current_dir().context("resolve current working directory")?;
    Ok(cwd.join(path))
}

fn read_pid_file(path: &Path) -> Option<u32> {
    let raw = std::fs::read_to_string(path).ok()?;
    raw.trim().parse::<u32>().ok()
}

fn write_pid_file(path: &Path, pid: u32) -> anyhow::Result<()> {
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(path)
        .with_context(|| format!("open daemon pid file '{}'", path.display()))?;
    writeln!(&mut file, "{pid}").with_context(|| format!("write daemon pid file '{}'", path.display()))?;
    file.sync_all()
        .with_context(|| format!("sync daemon pid file '{}'", path.display()))?;
    Ok(())
}

fn cleanup_stale_socket(socket_path: &Path) -> Result<(), SingletonError> {
    if !socket_path.exists() {
        return Ok(());
    }

    if daemon_socket_is_healthy(socket_path) {
        return Err(SingletonError::AlreadyRunning {
            pid: None,
            socket_path: socket_path.to_path_buf(),
        });
    }

    std::fs::remove_file(socket_path)
        .with_context(|| format!("remove stale socket '{}'", socket_path.display()))?;
    tracing::info!(path = %socket_path.display(), "removed stale daemon socket");
    Ok(())
}

#[cfg(unix)]
fn daemon_socket_is_healthy(socket_path: &Path) -> bool {
    use std::os::unix::net::UnixStream;

    UnixStream::connect(socket_path).is_ok()
}

#[cfg(not(unix))]
fn daemon_socket_is_healthy(_socket_path: &Path) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_socket_path_is_under_home_cache() {
        let home = PathBuf::from("/tmp/skelesearch-home");
        assert_eq!(
            default_socket_path_for_home(&home),
            home.join(".cache").join("skelesearch").join("daemon.sock")
        );
    }

    #[test]
    fn resolved_paths_keep_singleton_files_next_to_socket() {
        let temp = tempfile::tempdir().expect("tempdir");
        let socket_path = temp.path().join("custom-daemon.sock");

        let paths = DaemonPaths::resolve(Some(socket_path.clone())).expect("resolve paths");

        assert_eq!(paths.socket_path, socket_path);
        assert_eq!(paths.cache_dir, temp.path());
        assert_eq!(paths.lock_path, temp.path().join("daemon.lock"));
        assert_eq!(paths.pid_path, temp.path().join("daemon.pid"));
    }
}
