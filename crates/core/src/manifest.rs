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
