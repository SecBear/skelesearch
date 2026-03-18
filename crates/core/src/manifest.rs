use std::collections::HashSet;
use std::path::Path;

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
pub struct ManifestStore {
    conn: sqlite::Connection,
}

impl ManifestStore {
    /// Open or create the manifest database at `path`.
    pub fn open(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let conn = sqlite::open(path.as_ref())?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS file_hashes (
                file_path TEXT PRIMARY KEY,
                mtime     INTEGER NOT NULL,
                size      INTEGER NOT NULL,
                xxhash3   TEXT    NOT NULL
            );",
        )?;
        Ok(Self { conn })
    }

    /// Insert or update the manifest entry for `file_path`.
    pub fn upsert(
        &self,
        file_path: &str,
        mtime: i64,
        size: i64,
        xxhash3: &str,
    ) -> anyhow::Result<()> {
        let mut stmt = self.conn.prepare(
            "INSERT INTO file_hashes (file_path, mtime, size, xxhash3) VALUES (?, ?, ?, ?)
             ON CONFLICT(file_path) DO UPDATE SET mtime=excluded.mtime, size=excluded.size, xxhash3=excluded.xxhash3",
        )?;
        stmt.bind((1, file_path))?;
        stmt.bind((2, mtime))?;
        stmt.bind((3, size))?;
        stmt.bind((4, xxhash3))?;
        stmt.next()?;
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
        let mut stmt = self.conn.prepare(
            "SELECT mtime, size, xxhash3 FROM file_hashes WHERE file_path = ?",
        )?;
        stmt.bind((1, file_path))?;

        use sqlite::State;
        match stmt.next()? {
            State::Done => Ok(false),
            State::Row => {
                let stored_mtime: i64 = stmt.read(0)?;
                let stored_size: i64 = stmt.read(1)?;
                let stored_hash: String = stmt.read(2)?;
                Ok(stored_mtime == mtime && stored_size == size && stored_hash == xxhash3)
            }
        }
    }

    /// All file paths currently recorded in the manifest, sorted ascending.
    pub fn list_paths(&self) -> anyhow::Result<Vec<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT file_path FROM file_hashes ORDER BY file_path")?;

        let mut paths = Vec::new();
        use sqlite::State;
        while let Ok(State::Row) = stmt.next() {
            let path: String = stmt.read(0)?;
            paths.push(path);
        }
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
        let mut stmt = self
            .conn
            .prepare("DELETE FROM file_hashes WHERE file_path = ?")?;
        stmt.bind((1, file_path))?;
        stmt.next()?;
        Ok(())
    }
}
