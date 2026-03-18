use skelesearch_core::gc::collect_garbage;
use skelesearch_core::{CozoBackend, ChunkRecord, FileRecord, ManifestStore, StorageBackend};
use std::sync::Arc;

#[tokio::test]
async fn gc_removes_orphaned_chunks() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let idx = dir.path().join("idx");
    std::fs::create_dir_all(&idx)?;
    let backend = Arc::new(CozoBackend::open(idx.join("index.db"))?);
    let manifest = Arc::new(ManifestStore::open(idx.join("manifest.db"))?);
    backend.initialize(8).await?;

    backend
        .upsert_file(&FileRecord {
            file_path: "gone.rs".into(),
            language: "rust".into(),
            last_modified: 100,
            last_indexed: 100,
            chunk_count: 1,
        })
        .await?;
    backend
        .upsert_chunks(&[ChunkRecord {
            file_path: "gone.rs".into(),
            chunk_idx: 0,
            content: "fn gone() {}".into(),
            normalized: "fn gone".into(),
            chunk_type: "code".into(),
            start_line: 1,
            end_line: 1,
            embedding: Some(vec![0.1; 8]),
        }])
        .await?;

    let root = dir.path().join("repo");
    std::fs::create_dir_all(&root)?;
    manifest.upsert("gone.rs", 100, 10, "hash")?;

    let removed = collect_garbage(&root, &backend, &manifest).await?;
    assert_eq!(removed, 1);
    let chunks = backend.get_chunks_for_file("gone.rs").await?;
    assert!(chunks.is_empty());
    Ok(())
}

#[tokio::test]
async fn gc_skips_existing_files() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let idx = dir.path().join("idx");
    std::fs::create_dir_all(&idx)?;
    let backend = Arc::new(CozoBackend::open(idx.join("index.db"))?);
    let manifest = Arc::new(ManifestStore::open(idx.join("manifest.db"))?);
    backend.initialize(8).await?;

    let root = dir.path().join("repo");
    std::fs::create_dir_all(&root)?;
    // Create the file on disk.
    let src_file = root.join("present.rs");
    std::fs::write(&src_file, b"fn present() {}")?;

    backend
        .upsert_file(&FileRecord {
            file_path: "present.rs".into(),
            language: "rust".into(),
            last_modified: 100,
            last_indexed: 100,
            chunk_count: 1,
        })
        .await?;
    manifest.upsert("present.rs", 100, 15, "hash")?;

    let removed = collect_garbage(&root, &backend, &manifest).await?;
    assert_eq!(removed, 0);
    Ok(())
}

#[tokio::test]
async fn gc_empty_index_returns_zero() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let idx = dir.path().join("idx");
    std::fs::create_dir_all(&idx)?;
    let backend = Arc::new(CozoBackend::open(idx.join("index.db"))?);
    let manifest = Arc::new(ManifestStore::open(idx.join("manifest.db"))?);
    backend.initialize(8).await?;

    let root = dir.path().join("repo");
    std::fs::create_dir_all(&root)?;

    let removed = collect_garbage(&root, &backend, &manifest).await?;
    assert_eq!(removed, 0);
    Ok(())
}
