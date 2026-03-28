// Tests for ManifestStore checkpoint-table semantics.
//
// These tests verify that:
//   - the checkpoint table is initialised on first open
//   - batches begun but never completed appear as incomplete
//   - completing a batch removes it from the incomplete set
//   - files whose batch completed are not re-indexed on a subsequent run
//
// NOTE: These tests do NOT exercise real process-kill crash recovery.
// They simulate "interrupted batch" by calling begin_batch without
// complete_batch in the same in-process session.  A separate test
// (not yet written) would need to spawn a child process, kill it, and
// reopen the store to verify cross-process durability.

use skelesearch_core::{CompositeBackend, Indexer, ManifestStore, StorageBackend};
use std::sync::Arc;

struct ZeroProvider(usize);

#[async_trait::async_trait]
impl skelesearch_core::EmbedProvider for ZeroProvider {
    fn dim(&self) -> usize {
        self.0
    }
    async fn embed_batch(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|_| vec![0.0; self.0]).collect())
    }
}

#[tokio::test]
async fn checkpoint_table_created_on_open() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let manifest = ManifestStore::open(dir.path().join("manifest.db"))?;
    let incomplete = manifest.find_incomplete_batches()?;
    assert!(incomplete.is_empty());
    Ok(())
}

#[tokio::test]
async fn incomplete_batch_detected_when_complete_not_called() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let manifest = ManifestStore::open(dir.path().join("manifest.db"))?;

    manifest.begin_batch("run_001", 0, &["src/a.rs", "src/b.rs"])?;

    let incomplete = manifest.find_incomplete_batches()?;
    assert_eq!(incomplete.len(), 1);
    assert_eq!(
        incomplete[0].files,
        vec!["src/a.rs".to_string(), "src/b.rs".to_string()]
    );

    manifest.complete_batch("run_001", 0)?;
    let incomplete = manifest.find_incomplete_batches()?;
    assert!(incomplete.is_empty());
    Ok(())
}

#[tokio::test]
async fn completed_files_not_reindexed_on_subsequent_run() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let repo = dir.path().join("repo");
    std::fs::create_dir_all(&repo)?;
    std::fs::write(repo.join("a.rs"), "fn a() {}")?;
    std::fs::write(repo.join("b.rs"), "fn b() {}")?;

    let idx_dir = dir.path().join("idx");
    std::fs::create_dir_all(&idx_dir)?;
    let backend = Arc::new(CompositeBackend::open(&idx_dir).await?);
    let manifest = Arc::new(ManifestStore::open(idx_dir.join("manifest.db"))?);
    backend.initialize(8).await?;

    let indexer = Indexer::new(Arc::clone(&backend), Arc::clone(&manifest), ZeroProvider(8));
    let r1 = indexer.index_path(&repo).await?;
    assert_eq!(r1.indexed_files, 2);

    let r2 = indexer.index_path(&repo).await?;
    assert_eq!(r2.indexed_files, 0);
    Ok(())
}
