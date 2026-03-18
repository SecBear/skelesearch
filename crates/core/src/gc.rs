use std::path::Path;
use std::sync::Arc;

use crate::{ManifestStore, StorageBackend};

/// Remove index entries for files that no longer exist on disk.
///
/// Iterates the manifest, checks each recorded path against the filesystem
/// relative to `root`, and deletes chunks/edges/file records plus the manifest
/// entry for every path that is absent.
///
/// Returns the number of files removed.
pub async fn collect_garbage<B: StorageBackend>(
    root: &Path,
    backend: &Arc<B>,
    manifest: &Arc<ManifestStore>,
) -> anyhow::Result<usize> {
    let indexed_paths = manifest.list_paths()?;
    let mut removed = 0usize;
    for file_path in &indexed_paths {
        let abs_path = root.join(file_path);
        if !abs_path.exists() {
            backend.delete_chunks_for_file(file_path).await?;
            backend.delete_edges_for_file(file_path).await?;
            backend.delete_file(file_path).await?;
            manifest.remove(file_path)?;
            removed += 1;
        }
    }
    Ok(removed)
}
