use skelesearch_core::{ChunkRecord, CompositeBackend, EdgeRecord, FileRecord, StorageBackend};

#[tokio::test]
async fn cozo_backend_round_trips_storage_backend_contract() -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let backend = CompositeBackend::open(temp.path()).await?;
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
            description: String::new(),
            chunk_type: "function".into(),
            start_line: 1,
            end_line: 1,
            materialization_tier: 2,
            embedding: Some(vec![0.1; 8]),
            doc_embedding: None,
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
    let backend = CompositeBackend::open(temp.path()).await?;

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
    let backend = CompositeBackend::open(temp.path()).await?;
    backend.initialize(8).await?;

    let chunks: Vec<ChunkRecord> = (0..500)
        .map(|i| ChunkRecord {
            file_path: "big_file.rs".into(),
            chunk_idx: i,
            content: format!("fn func_{i}() {{}}"),
            normalized: format!("fn func {i}"),
            description: String::new(),
            chunk_type: "code".into(),
            start_line: i * 10 + 1,
            end_line: (i + 1) * 10,
            materialization_tier: 2,
            // Use unique embeddings: identical vectors have cosine distance 0, which
            // causes CozoDB's HNSW algorithm to loop indefinitely.
            embedding: Some((0..8).map(|j| if j == i % 8 { 1.0_f32 } else { 0.01 }).collect()),
            doc_embedding: None,
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
    let backend = CompositeBackend::open(temp.path()).await?;
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


use skelesearch_core::{CallEdge, CoChangePair, SymbolDef};
use skelesearch_core::sparse::SparseEmbedding;

// ---------------------------------------------------------------------------
// New tests for previously uncovered StorageBackend methods (Phase 8)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn compute_pagerank_ranks_most_imported_file_highest() -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let backend = CompositeBackend::open(temp.path()).await?;
    backend.initialize(4).await?;

    // Build: a.rs ← b.rs, a.rs ← c.rs  (a is imported by 2 files)
    let edges = vec![
        EdgeRecord { from_file: "b.rs".into(), from_chunk: 0, to_file: "a.rs".into(), edge_type: "imports".into() },
        EdgeRecord { from_file: "c.rs".into(), from_chunk: 0, to_file: "a.rs".into(), edge_type: "imports".into() },
    ];
    backend.upsert_edges(&edges).await?;
    backend.compute_pagerank(None).await?;

    let ranks = backend.get_file_ranks(&["a.rs", "b.rs", "c.rs"]).await?;
    let rank_a = ranks.get("a.rs").copied().unwrap_or(0.0);
    let rank_b = ranks.get("b.rs").copied().unwrap_or(0.0);
    assert!(rank_a > rank_b, "a.rs (2 importers) should rank higher than b.rs (0 importers)");
    Ok(())
}

#[tokio::test]
async fn upsert_cochange_edges_and_get_neighbors() -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let backend = CompositeBackend::open(temp.path()).await?;
    backend.initialize(4).await?;

    let pairs = vec![CoChangePair {
        file_a: "alpha.rs".into(), file_b: "beta.rs".into(),
        cochange_count: 10, jaccard: 0.8,
    }];
    backend.upsert_cochange_edges(&pairs).await?;

    let neighbors = backend.get_cochange_neighbors("alpha.rs", 0.5).await?;
    assert_eq!(neighbors.len(), 1);
    assert_eq!(neighbors[0].0, "beta.rs");
    assert!((neighbors[0].1 - 0.8).abs() < 1e-9);
    Ok(())
}

#[tokio::test]
async fn hnsw_neighbors_excludes_seed_chunk() -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let backend = CompositeBackend::open(temp.path()).await?;
    backend.initialize(4).await?;

    // Index 3 chunks with distinct embeddings.
    let chunks: Vec<ChunkRecord> = (0..3).map(|i| ChunkRecord {
        file_path: "src/lib.rs".into(), chunk_idx: i,
        content: format!("fn f{i}() {{}}"), normalized: format!("fn f{i}"),
        description: String::new(), chunk_type: "function".into(),
        start_line: i + 1, end_line: i + 1, materialization_tier: 2,
        embedding: Some((0..4).map(|j: usize| if j == i % 4 { 1.0_f32 } else { 0.0 }).collect()),
        doc_embedding: None,
    }).collect();
    backend.upsert_chunks(&chunks).await?;

    let neighbors = backend.hnsw_neighbors(&[("src/lib.rs".into(), 0)], 2.0, 5).await?;
    // Seed (file=src/lib.rs, chunk_idx=0) must not appear in its own neighbors.
    assert!(
        neighbors.iter().all(|(fp, ci, _)| !(fp == "src/lib.rs" && *ci == 0)),
        "seed chunk must not appear in its own neighbor list"
    );
    Ok(())
}

#[tokio::test]
async fn sparse_search_orders_by_dot_product() -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let backend = CompositeBackend::open(temp.path()).await?;
    backend.initialize(4).await?;

    // Chunk 0 has strong token 42; chunk 1 has weak token 42.
    backend.store_sparse_vectors("src/a.rs", 0, &SparseEmbedding { indices: vec![42], values: vec![2.0] }).await?;
    backend.store_sparse_vectors("src/b.rs", 0, &SparseEmbedding { indices: vec![42], values: vec![0.5] }).await?;

    // Query with token 42 weight 1.0 → a.rs should score higher.
    let results = backend.sparse_search(&SparseEmbedding { indices: vec![42], values: vec![1.0] }, 5).await?;
    assert!(!results.is_empty(), "sparse_search must return results");
    assert_eq!(results[0].0, "src/a.rs", "a.rs has higher weight and must rank first");
    assert!(results[0].2 > results[1].2, "scores must be descending");
    Ok(())
}

#[tokio::test]
async fn compute_symbol_roles_classifies_entry_node() -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let backend = CompositeBackend::open(temp.path()).await?;
    backend.initialize(4).await?;

    // lib.rs is imported by 3 files and imports 0 → role: entry
    let edges: Vec<EdgeRecord> = (0..3).map(|i| EdgeRecord {
        from_file: format!("src/mod_{i}.rs"), from_chunk: 0,
        to_file: "src/lib.rs".into(), edge_type: "imports".into(),
    }).collect();
    backend.upsert_edges(&edges).await?;
    backend.compute_symbol_roles().await?;

    let roles = backend.get_symbol_roles(&["src/lib.rs"]).await?;
    assert_eq!(roles.get("src/lib.rs").map(|s| s.as_str()), Some("entry"),
        "lib.rs imported by 3, imports 0 → entry role");
    Ok(())
}

#[tokio::test]
async fn deduplicate_chunks_removes_cross_file_duplicates() -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let backend = CompositeBackend::open(temp.path()).await?;
    backend.initialize(4).await?;

    // Two files with identical normalized text → one should be deduplicated.
    for (file, idx) in [("src/a.rs", 0usize), ("src/b.rs", 0usize)] {
        backend.upsert_file(&FileRecord {
            file_path: file.into(), language: "rust".into(),
            last_modified: 1, last_indexed: 1, chunk_count: 1,
        }).await?;
        backend.upsert_chunks(&[ChunkRecord {
            file_path: file.into(), chunk_idx: idx,
            content: "fn dup() {}".into(), normalized: "fn dup identical_normalized_text".into(),
            description: String::new(), chunk_type: "function".into(),
            start_line: 1, end_line: 1, materialization_tier: 2,
            embedding: Some(vec![0.1; 4]), doc_embedding: None,
        }]).await?;
    }

    let removed = backend.deduplicate_chunks().await?;
    assert_eq!(removed, 1, "exactly one duplicate should be removed");
    Ok(())
}

#[tokio::test]
async fn get_repo_map_data_returns_complete_structure() -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let backend = CompositeBackend::open(temp.path()).await?;
    backend.initialize(4).await?;

    backend.upsert_file(&FileRecord {
        file_path: "src/lib.rs".into(), language: "rust".into(),
        last_modified: 1, last_indexed: 1, chunk_count: 2,
    }).await?;
    backend.upsert_symbols(&[SymbolDef {
        file_path: "src/lib.rs".into(), name: "my_fn".into(), kind: "function".into(),
        start_line: 1, end_line: 5,
    }]).await?;
    backend.upsert_edges(&[EdgeRecord {
        from_file: "src/lib.rs".into(), from_chunk: 0,
        to_file: "src/utils.rs".into(), edge_type: "imports".into(),
    }]).await?;

    let data = backend.get_repo_map_data().await?;
    assert!(!data.files.is_empty(), "repo map must include files");
    let lib = data.files.iter().find(|f| f.path == "src/lib.rs")
        .expect("src/lib.rs must appear in repo map");
    assert_eq!(lib.symbols.len(), 1);
    assert_eq!(lib.symbols[0].name, "my_fn");
    assert!(data.import_edges.iter().any(|(from, to)| from == "src/lib.rs" && to == "src/utils.rs"),
        "import edge must appear in repo map");
    Ok(())
}

#[tokio::test]
async fn call_edges_bidirectional_lookup() -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let backend = CompositeBackend::open(temp.path()).await?;
    backend.initialize(4).await?;

    backend.upsert_call_edges(&[CallEdge {
        caller_file: "src/main.rs".into(), caller_symbol: "main".into(),
        callee_name: "helper".into(), start_line: 5,
        callee_file: Some("src/helpers.rs".into()),
        callee_symbol: Some("helper".into()),
        confidence: 0.9, dynamic: false,
    }]).await?;

    let callers = backend.get_callers("src/helpers.rs", "helper").await?;
    assert_eq!(callers.len(), 1);
    assert_eq!(callers[0].caller_file, "src/main.rs");

    let callees = backend.get_callees("src/main.rs", "main").await?;
    assert_eq!(callees.len(), 1);
    assert_eq!(callees[0].callee_name, "helper");
    Ok(())
}

#[tokio::test]
async fn traverse_importers_returns_correct_depths() -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let backend = CompositeBackend::open(temp.path()).await?;
    backend.initialize(4).await?;

    // Build: a ← b ← c  (c imports b, b imports a)
    backend.upsert_edges(&[
        EdgeRecord { from_file: "b.rs".into(), from_chunk: 0, to_file: "a.rs".into(), edge_type: "imports".into() },
        EdgeRecord { from_file: "c.rs".into(), from_chunk: 0, to_file: "b.rs".into(), edge_type: "imports".into() },
    ]).await?;

    // traverse_importers("a.rs", max_depth=2) should return b.rs at depth 1, c.rs at depth 2
    let importers = backend.traverse_importers("a.rs", 2, None).await?;
    let b_depth = importers.iter().find(|(f, _)| f == "b.rs").map(|(_, d)| *d);
    let c_depth = importers.iter().find(|(f, _)| f == "c.rs").map(|(_, d)| *d);
    assert_eq!(b_depth, Some(1), "b.rs is 1 hop from a.rs");
    assert_eq!(c_depth, Some(2), "c.rs is 2 hops from a.rs");
    Ok(())
}

#[tokio::test]
async fn get_chunks_for_files_batch_matches_individual() -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let backend = CompositeBackend::open(temp.path()).await?;
    backend.initialize(4).await?;

    for (file, idx) in [("src/a.rs", 0usize), ("src/a.rs", 1usize), ("src/b.rs", 0usize)] {
        backend.upsert_chunks(&[ChunkRecord {
            file_path: file.into(), chunk_idx: idx,
            content: format!("fn f_{idx}() {{}}"), normalized: format!("fn f_{idx}"),
            description: String::new(), chunk_type: "function".into(),
            start_line: idx + 1, end_line: idx + 1, materialization_tier: 2,
            embedding: None, doc_embedding: None,
        }]).await?;
    }

    let batch = backend.get_chunks_for_files(&["src/a.rs", "src/b.rs"]).await?;
    let a_only = backend.get_chunks_for_file("src/a.rs").await?;
    let b_only = backend.get_chunks_for_file("src/b.rs").await?;
    assert_eq!(batch.len(), a_only.len() + b_only.len(), "batch must equal sum of individual queries");
    Ok(())
}

#[tokio::test]
async fn delete_tier1_leaves_tier2_intact() -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let backend = CompositeBackend::open(temp.path()).await?;
    backend.initialize(4).await?;

    backend.upsert_chunks(&[
        ChunkRecord {
            file_path: "src/lib.rs".into(), chunk_idx: 0,
            content: "tier1 chunk".into(), normalized: "tier1".into(),
            description: String::new(), chunk_type: "code".into(),
            start_line: 1, end_line: 1, materialization_tier: 1,
            embedding: None, doc_embedding: None,
        },
        ChunkRecord {
            file_path: "src/lib.rs".into(), chunk_idx: 1,
            content: "tier2 chunk".into(), normalized: "tier2".into(),
            description: String::new(), chunk_type: "function".into(),
            start_line: 2, end_line: 2, materialization_tier: 2,
            embedding: None, doc_embedding: None,
        },
    ]).await?;

    backend.delete_tier1_chunks_for_file("src/lib.rs").await?;

    let remaining = backend.get_chunks_for_file("src/lib.rs").await?;
    assert_eq!(remaining.len(), 1, "only tier-2 chunk should remain");
    assert_eq!(remaining[0].materialization_tier, 2);
    Ok(())
}

#[tokio::test]
async fn get_chunk_embeddings_returns_exact_stored_vectors() -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let backend = CompositeBackend::open(temp.path()).await?;
    backend.initialize(4).await?;

    let emb = vec![0.25_f32, 0.5, 0.75, 1.0];
    backend.upsert_chunks(&[ChunkRecord {
        file_path: "src/lib.rs".into(), chunk_idx: 0,
        content: "fn f() {}".into(), normalized: "fn f".into(),
        description: String::new(), chunk_type: "function".into(),
        start_line: 1, end_line: 1, materialization_tier: 2,
        embedding: Some(emb.clone()), doc_embedding: None,
    }]).await?;

    let retrieved = backend.get_chunk_embeddings(&[("src/lib.rs".to_string(), 0)]).await?;
    assert_eq!(retrieved.len(), 1);
    for (a, b) in retrieved[0].iter().zip(emb.iter()) {
        assert!((a - b).abs() < 1e-6, "embedding values must round-trip exactly");
    }
    Ok(())
}

#[tokio::test]
async fn fts_tokenizer_matches_camel_case_subwords() -> anyhow::Result<()> {
    // Verifies the CodeAnalyzer tokenizer splits getUserById into [get, user, by, id].
    let temp = tempfile::tempdir()?;
    let backend = CompositeBackend::open(temp.path()).await?;
    backend.initialize(4).await?;

    backend.upsert_chunks(&[ChunkRecord {
        file_path: "src/lib.rs".into(), chunk_idx: 0,
        content: "getUserById".into(),
        normalized: "getUserById".into(), // CodeAnalyzer tokenizes this
        description: String::new(), chunk_type: "function".into(),
        start_line: 1, end_line: 1, materialization_tier: 2,
        embedding: Some(vec![0.1; 4]), doc_embedding: None,
    }]).await?;

    // Each subword must match the chunk.
    for query in ["user", "id", "get"] {
        let results = backend.hybrid_search(&[0.1; 4], query, 5).await?;
        assert!(
            results.iter().any(|r| r.file_path == "src/lib.rs"),
            "query '{}' must find chunk with getUserById via CodeAnalyzer tokenization", query
        );
    }
    Ok(())
}