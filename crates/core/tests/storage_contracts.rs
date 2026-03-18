use skelesearch_core::{ChunkRecord, CozoBackend, EdgeRecord, FileRecord, StorageBackend};

#[tokio::test]
async fn cozo_backend_round_trips_storage_backend_contract() -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let backend = CozoBackend::open(temp.path().join("index.db"))?;
    backend.initialize(8).await?;

    backend
        .upsert_file(&FileRecord {
            file_path: "src/lib.rs".into(),
            language: "rust".into(),
            last_modified: 10,
            last_indexed: 10,
            chunk_count: 1,
        })
        .await?;

    backend
        .upsert_chunks(&[ChunkRecord {
            file_path: "src/lib.rs".into(),
            chunk_idx: 0,
            content: "fn alpha() {}".into(),
            normalized: "fn alpha".into(),
            chunk_type: "function".into(),
            start_line: 1,
            end_line: 1,
            embedding: Some(vec![0.1; 8]),
        }])
        .await?;

    backend
        .upsert_edges(&[EdgeRecord {
            from_file: "src/lib.rs".into(),
            from_chunk: 0,
            to_file: "src/search.rs".into(),
            edge_type: "imports".into(),
        }])
        .await?;

    assert_eq!(
        backend.list_indexed_paths().await?,
        vec!["src/lib.rs".to_string()]
    );
    assert_eq!(backend.get_chunks_for_file("src/lib.rs").await?.len(), 1);
    assert_eq!(
        backend.get_imports("src/lib.rs").await?,
        vec!["src/search.rs".to_string()]
    );
    assert_eq!(
        backend.get_importers("src/search.rs").await?,
        vec!["src/lib.rs".to_string()]
    );

    backend.delete_edges_for_file("src/lib.rs").await?;
    backend.delete_chunks_for_file("src/lib.rs").await?;
    backend.delete_file("src/lib.rs").await?;
    assert!(backend.list_indexed_paths().await?.is_empty());
    Ok(())
}

#[tokio::test]
async fn cozo_backend_initializes_and_reports_empty_stats() -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let backend = CozoBackend::open(temp.path().join("index.db"))?;

    // initialize must be idempotent.
    backend.initialize(768).await?;
    backend.initialize(768).await?;

    let stats = backend.stats().await?;
    let hits = backend
        .hybrid_search(&vec![0.0; 768], "missing symbol", 5)
        .await?;

    assert_eq!(stats.indexed_files, 0);
    assert_eq!(stats.total_chunks, 0);
    assert!(stats.last_indexed.is_none());
    assert!(!stats.watching);
    assert!(hits.is_empty());
    Ok(())
}


#[tokio::test]
async fn upsert_chunks_batch_handles_500_chunks() -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let backend = CozoBackend::open(temp.path().join("index.db"))?;
    backend.initialize(8).await?;

    let chunks: Vec<ChunkRecord> = (0..500)
        .map(|i| ChunkRecord {
            file_path: "big_file.rs".into(),
            chunk_idx: i,
            content: format!("fn func_{i}() {{}}"),
            normalized: format!("fn func {i}"),
            chunk_type: "code".into(),
            start_line: i * 10 + 1,
            end_line: (i + 1) * 10,
            embedding: Some(vec![0.1; 8]),
        })
        .collect();

    backend.upsert_chunks(&chunks).await?;
    let stored = backend.get_chunks_for_file("big_file.rs").await?;
    assert_eq!(stored.len(), 500);
    Ok(())
}

#[tokio::test]
async fn upsert_edges_batch_handles_many_edges() -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let backend = CozoBackend::open(temp.path().join("index.db"))?;
    backend.initialize(8).await?;

    let edges: Vec<EdgeRecord> = (0..200)
        .map(|i| EdgeRecord {
            from_file: format!("src/mod_{i}.rs"),
            from_chunk: 0,
            to_file: "src/lib.rs".into(),
            edge_type: "imports".into(),
        })
        .collect();

    backend.upsert_edges(&edges).await?;
    let importers = backend.get_importers("src/lib.rs").await?;
    assert_eq!(importers.len(), 200);
    Ok(())
}