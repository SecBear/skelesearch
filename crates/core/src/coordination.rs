use std::{
    fs::{File, OpenOptions},
    io::{ErrorKind, Write},
    path::{Path, PathBuf},
};

use anyhow::Context as _;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

const LOCK_FILE: &str = ".skelesearch.lock";
const STATUS_FILE: &str = "indexing-status.json";

/// Shared cross-process indexing status persisted under `.skelesearch/indexing-status.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedIndexingStatus {
    pub instance_id: String,
    pub pid: u32,
    pub path: String,
    pub provider: String,
    pub trigger: String,
    pub status: String,
    pub started_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub files_total: usize,
    pub files_done: usize,
    pub chunks_done: usize,
    pub cache_hits: usize,
    pub error: Option<String>,
}

/// RAII lease that keeps an exclusive writer lock alive for the indexing operation.
///
/// The lock is held through `lock_file`; dropping this type releases the lock and
/// best-effort removes `indexing-status.json`.
pub struct IndexingLease {
    lock_file: File,
    status_path: PathBuf,
}

impl IndexingLease {
    /// Atomically write the latest shared indexing status.
    pub fn write_status(&self, status: &SharedIndexingStatus) -> anyhow::Result<()> {
        write_status_atomic(&self.status_path, status)
    }
}

pub fn write_file_atomic(path: &Path, payload: &[u8]) -> anyhow::Result<()> {
    let parent = path
        .parent()
        .with_context(|| format!("path has no parent: {}", path.display()))?;
    std::fs::create_dir_all(parent)
        .with_context(|| format!("failed to create parent dir: {}", parent.display()))?;

    let temp_name = format!(
        ".{}.tmp-{}-{}",
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("atomic"),
        std::process::id(),
        Utc::now().timestamp_nanos_opt().unwrap_or_default()
    );
    let temp_path = parent.join(temp_name);

    let mut temp = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temp_path)
        .with_context(|| format!("failed to create temp file at {}", temp_path.display()))?;
    temp.write_all(payload)
        .with_context(|| format!("failed to write temp file at {}", temp_path.display()))?;
    temp.sync_all()
        .with_context(|| format!("failed to fsync temp file at {}", temp_path.display()))?;

    if let Err(err) = std::fs::rename(&temp_path, path) {
        if path.exists() {
            std::fs::remove_file(path)
                .with_context(|| format!("failed to replace file at {}", path.display()))?;
            std::fs::rename(&temp_path, path)
                .with_context(|| format!("failed to move file into place at {}", path.display()))?;
        } else {
            return Err(err).with_context(|| {
                format!(
                    "failed to rename temp file '{}' to '{}'",
                    temp_path.display(),
                    path.display()
                )
            });
        }
    }

    Ok(())
}

impl Drop for IndexingLease {
    fn drop(&mut self) {
        if let Err(err) = std::fs::remove_file(&self.status_path) {
            if err.kind() != ErrorKind::NotFound {
                tracing::debug!(path = %self.status_path.display(), error = %err, "failed to remove shared indexing status");
            }
        }
        if let Err(err) = fs2::FileExt::unlock(&self.lock_file) {
            tracing::debug!(path = %self.status_path.display(), error = %err, "failed to unlock indexing lease");
        }
    }
}

/// Try to acquire the cross-process indexing lease for `storage_dir`.
///
/// Returns `Ok(None)` when another process already holds the writer lock.
pub fn try_acquire_indexing_lease(
    storage_dir: &Path,
    initial: &SharedIndexingStatus,
) -> anyhow::Result<Option<IndexingLease>> {
    std::fs::create_dir_all(storage_dir)
        .with_context(|| format!("failed to create storage dir at {}", storage_dir.display()))?;

    let lock_path = lock_path(storage_dir);
    let status_path = status_path(storage_dir);
    let lock_file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(&lock_path)
        .with_context(|| format!("failed to open indexing lock at {}", lock_path.display()))?;

    match fs2::FileExt::try_lock_exclusive(&lock_file) {
        Ok(()) => {
            let lease = IndexingLease {
                lock_file,
                status_path,
            };
            lease.write_status(initial)?;
            Ok(Some(lease))
        }
        Err(err) if err.kind() == ErrorKind::WouldBlock => Ok(None),
        Err(err) => Err(err)
            .with_context(|| format!("failed to acquire indexing lock at {}", lock_path.display())),
    }
}

/// Read active shared indexing status for `storage_dir`.
///
/// If a status file exists while no lock is held, it is treated as stale and
/// removed best-effort.
pub fn read_shared_indexing_status(
    storage_dir: &Path,
) -> anyhow::Result<Option<SharedIndexingStatus>> {
    let status_path = status_path(storage_dir);
    if !status_path.exists() {
        return Ok(None);
    }

    let lock_held = is_lock_held(storage_dir)?;
    if !lock_held {
        if let Err(err) = std::fs::remove_file(&status_path) {
            if err.kind() != ErrorKind::NotFound {
                tracing::debug!(path = %status_path.display(), error = %err, "failed to remove stale indexing status");
            }
        } else {
            tracing::info!(path = %status_path.display(), "removed stale shared indexing status");
        }
        return Ok(None);
    }

    let bytes = match std::fs::read(&status_path) {
        Ok(bytes) => bytes,
        Err(err) if err.kind() == ErrorKind::NotFound => return Ok(None),
        Err(err) => {
            return Err(err).with_context(|| {
                format!(
                    "failed to read shared indexing status at {}",
                    status_path.display()
                )
            })
        }
    };

    match serde_json::from_slice::<SharedIndexingStatus>(&bytes) {
        Ok(status) => Ok(Some(status)),
        Err(err) => {
            tracing::warn!(path = %status_path.display(), error = %err, "invalid shared indexing status JSON; removing");
            let _ = std::fs::remove_file(&status_path);
            Ok(None)
        }
    }
}

/// Returns true when a different process appears to be actively indexing this
/// storage dir.
pub fn is_indexing_active_elsewhere(storage_dir: &Path) -> anyhow::Result<bool> {
    if !storage_dir.exists() {
        return Ok(false);
    }

    if read_shared_indexing_status(storage_dir)?.is_some() {
        return Ok(true);
    }

    is_lock_held(storage_dir)
}

fn is_lock_held(storage_dir: &Path) -> anyhow::Result<bool> {
    if !storage_dir.exists() {
        return Ok(false);
    }

    let lock_path = lock_path(storage_dir);
    let lock_file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(&lock_path)
        .with_context(|| format!("failed to open indexing lock at {}", lock_path.display()))?;

    match fs2::FileExt::try_lock_shared(&lock_file) {
        Ok(()) => {
            fs2::FileExt::unlock(&lock_file).with_context(|| {
                format!("failed to unlock indexing lock at {}", lock_path.display())
            })?;
            Ok(false)
        }
        Err(err) if err.kind() == ErrorKind::WouldBlock => Ok(true),
        Err(err) => Err(err)
            .with_context(|| format!("failed to check indexing lock at {}", lock_path.display())),
    }
}

fn lock_path(storage_dir: &Path) -> PathBuf {
    storage_dir.join(LOCK_FILE)
}

fn status_path(storage_dir: &Path) -> PathBuf {
    storage_dir.join(STATUS_FILE)
}

fn write_status_atomic(path: &Path, status: &SharedIndexingStatus) -> anyhow::Result<()> {
    let payload =
        serde_json::to_vec_pretty(status).context("failed to serialize shared indexing status")?;
    write_file_atomic(path, &payload)
}
