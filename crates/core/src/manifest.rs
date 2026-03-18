use std::collections::HashSet;
use std::path::Path;
use std::sync::Mutex;

use rusqlite::{params, Connection, OptionalExtension};

/// Metadata about a single indexed file.
pub struct ManifestEntry {
    pub file_path: String,
    pub mtime: i64,
    pub size: i64,
    pub xxhash3: String,
}

/// SQLite-backed manifest of indexed file hashes.
///
/// Lives in a separate `.db` file, never inside CozoDB (ADR-006).
/// `mtime` + `size` are checked first (O(1)); `xxhash3` is consulted only
/// when metadata suggests a change, avoiding unnecessary hashing on unchanged
/// files.
///
/// The inner `Mutex<Connection>` makes `ManifestStore` `Send + Sync`, allowing
/// safe use from multiple threads. WAL mode + a 5-second busy timeout prevent
/// SQLITE_BUSY errors under concurrent access.
pub struct ManifestStore {
    conn: Mutex<Connection>,
}

impl ManifestStore {
    /// Open or create the manifest database at `path`.
    pub fn open(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let conn = Connection::open(path.as_ref())?;
        conn.pragma_update(None, "journal_mode", "wal")?;
        conn.pragma_update(None, "busy_timeout", "5000")?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS file_hashes (
                file_path TEXT PRIMARY KEY,
                mtime     INTEGER NOT NULL,
                size      INTEGER NOT NULL,
                xxhash3   TEXT    NOT NULL
            );
            CREATE TABLE IF NOT EXISTS index_progress (
                run_id     TEXT    NOT NULL,
                batch_idx  INTEGER NOT NULL,
                files      TEXT    NOT NULL,
                status     TEXT    NOT NULL DEFAULT 'pending',
                created_at INTEGER NOT NULL,
                PRIMARY KEY (run_id, batch_idx)
            );",
        )?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// Insert or update the manifest entry for `file_path`.
    pub fn upsert(
        &self,
        file_path: &str,
        mtime: i64,
        size: i64,
        xxhash3: &str,
    ) -> anyhow::Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| anyhow::anyhow!("manifest lock: {e}"))?;
        conn.execute(
            "INSERT INTO file_hashes (file_path, mtime, size, xxhash3) VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(file_path) DO UPDATE SET mtime=excluded.mtime, size=excluded.size, xxhash3=excluded.xxhash3",
            params![file_path, mtime, size, xxhash3],
        )?;
        Ok(())
    }

    /// Returns `true` iff the stored entry for `file_path` exactly matches all
    /// three fields.  Returns `false` if the file is not in the manifest.
    pub fn is_unchanged(
        &self,
        file_path: &str,
        mtime: i64,
        size: i64,
        xxhash3: &str,
    ) -> anyhow::Result<bool> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| anyhow::anyhow!("manifest lock: {e}"))?;
        let result = conn
            .query_row(
                "SELECT mtime, size, xxhash3 FROM file_hashes WHERE file_path = ?1",
                params![file_path],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, String>(2)?,
                    ))
                },
            )
            .optional()?;

        Ok(match result {
            None => false,
            Some((stored_mtime, stored_size, stored_hash)) => {
                stored_mtime == mtime && stored_size == size && stored_hash == xxhash3
            }
        })
    }

    /// All file paths currently recorded in the manifest, sorted ascending.
    pub fn list_paths(&self) -> anyhow::Result<Vec<String>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| anyhow::anyhow!("manifest lock: {e}"))?;
        let mut stmt =
            conn.prepare("SELECT file_path FROM file_hashes ORDER BY file_path")?;
        let paths = stmt
            .query_map([], |row| row.get(0))?
            .collect::<Result<Vec<String>, _>>()?;
        Ok(paths)
    }

    /// Returns manifest paths that are **not** in `visited`.
    ///
    /// These are candidates for deletion (the file was removed or renamed).
    /// A rename is represented as the old path becoming stale while the new
    /// path is discovered by the walker as a fresh entry.
    pub fn stale_paths_against(&self, visited: &HashSet<String>) -> anyhow::Result<Vec<String>> {
        Ok(self
            .list_paths()?
            .into_iter()
            .filter(|path| !visited.contains(path))
            .collect())
    }

    /// Returns `true` iff the stored entry for `file_path` has matching `mtime`
    /// **and** `size`.  Does not read the hash — used as a cheap pre-filter in
    /// Phase 1 of the indexer to skip obviously-unchanged files without reading
    /// file content.
    pub fn mtime_size_unchanged(
        &self,
        file_path: &str,
        mtime: i64,
        size: i64,
    ) -> anyhow::Result<bool> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| anyhow::anyhow!("manifest lock: {e}"))?;
        let result = conn
            .query_row(
                "SELECT mtime, size FROM file_hashes WHERE file_path = ?1",
                rusqlite::params![file_path],
                |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)),
            )
            .optional()?;
        Ok(matches!(result, Some((m, s)) if m == mtime && s == size))
    }

    /// Remove the manifest entry for `file_path`.
    pub fn remove(&self, file_path: &str) -> anyhow::Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| anyhow::anyhow!("manifest lock: {e}"))?;
        conn.execute(
            "DELETE FROM file_hashes WHERE file_path = ?1",
            params![file_path],
        )?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Checkpoint types and methods
// ---------------------------------------------------------------------------

/// A batch that was started but not completed — indicates a previous crash.
#[derive(Debug, Clone)]
pub struct IncompleteBatch {
    pub run_id: String,
    pub batch_idx: i64,
    pub files: Vec<String>,
}

impl ManifestStore {
    /// Record that a batch is about to be processed (write-before-transition).
    /// Uses INSERT OR REPLACE so retries after a crash are idempotent.
    pub fn begin_batch(&self, run_id: &str, batch_idx: usize, files: &[&str]) -> anyhow::Result<()> {
        let conn = self.conn.lock().map_err(|e| anyhow::anyhow!("lock: {e}"))?;
        let files_json = serde_json::to_string(files)?;
        let now = chrono::Utc::now().timestamp();
        conn.execute(
            "INSERT OR REPLACE INTO index_progress (run_id, batch_idx, files, status, created_at) VALUES (?1, ?2, ?3, 'pending', ?4)",
            params![run_id, batch_idx as i64, files_json, now],
        )?;
        Ok(())
    }

    /// Mark a batch as successfully completed.
    pub fn complete_batch(&self, run_id: &str, batch_idx: usize) -> anyhow::Result<()> {
        let conn = self.conn.lock().map_err(|e| anyhow::anyhow!("lock: {e}"))?;
        conn.execute(
            "UPDATE index_progress SET status = 'complete' WHERE run_id = ?1 AND batch_idx = ?2",
            params![run_id, batch_idx as i64],
        )?;
        Ok(())
    }

    /// Find batches that were started but never completed (crash recovery).
    pub fn find_incomplete_batches(&self) -> anyhow::Result<Vec<IncompleteBatch>> {
        let conn = self.conn.lock().map_err(|e| anyhow::anyhow!("lock: {e}"))?;
        let mut stmt = conn.prepare(
            "SELECT run_id, batch_idx, files FROM index_progress WHERE status = 'pending' ORDER BY created_at",
        )?;
        let rows = stmt
            .query_map([], |row| {
                let run_id: String = row.get(0)?;
                let batch_idx: i64 = row.get(1)?;
                let files_json: String = row.get(2)?;
                Ok((run_id, batch_idx, files_json))
            })?
            .filter_map(|r| r.ok())
            .map(|(run_id, batch_idx, files_json)| {
                let files: Vec<String> = serde_json::from_str(&files_json).unwrap_or_default();
                IncompleteBatch { run_id, batch_idx, files }
            })
            .collect();
        Ok(rows)
    }

    /// Remove all completed batch records for a run.
    pub fn clear_completed_batches(&self, run_id: &str) -> anyhow::Result<()> {
        let conn = self.conn.lock().map_err(|e| anyhow::anyhow!("lock: {e}"))?;
        conn.execute(
            "DELETE FROM index_progress WHERE run_id = ?1 AND status = 'complete'",
            params![run_id],
        )?;
        Ok(())
    }
}
