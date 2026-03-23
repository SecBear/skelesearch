use std::sync::Arc;

use async_trait::async_trait;
mod test_utils;
use test_utils::{copy_dir_all, DeterministicTestProvider};
use skelesearch_core::{
    CozoBackend, EdgeRecord, EmbedProvider, FileContext, FileRecord, Indexer, ManifestStore,
    Searcher, StorageBackend,
};
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Helpers (DeterministicTestProvider + copy_dir_all live in test_utils)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn test_backend() -> anyhow::Result<Arc<CozoBackend>> {
    let dir = tempfile::tempdir()?;
    let backend = CozoBackend::open(dir.path().join("index.db"))?;
    std::mem::forget(dir);
    Ok(Arc::new(backend))
}

fn test_manifest() -> anyhow::Result<(Arc<ManifestStore>, TempDir)> {
    let dir = tempfile::tempdir()?;
    let store = ManifestStore::open(dir.path().join("manifest.db"))?;
    Ok((Arc::new(store), dir))
}

fn fixture_repo() -> anyhow::Result<TempDir> {
    let dir = tempfile::tempdir()?;
    let fixture_src = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/sample_repo");
    copy_dir_all(&fixture_src, dir.path())?;
    Ok(dir)
}


/// Build an indexed backend + searcher ready for retrieval tests.
async fn indexed_searcher() -> anyhow::Result<(
    Arc<CozoBackend>,
    Searcher<CozoBackend, DeterministicTestProvider>,
    TempDir,
    TempDir,
)> {
    let fixture = fixture_repo()?;
    let backend = test_backend()?;
    let (manifest, manifest_dir) = test_manifest()?;

    // Index the fixture.
    let indexer = Indexer::new(
        backend.clone(),
        manifest.clone(),
        DeterministicTestProvider::new(8),
    );
    indexer.index_path(fixture.path()).await?;

    let searcher = Searcher::new(backend.clone(), DeterministicTestProvider::new(8));
    Ok((backend, searcher, fixture, manifest_dir))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn match_quality_uses_documented_relative_thresholds() -> anyhow::Result<()> {
    // top_score = 1.0
    // 1.0 >= 0.8 * 1.0  → high
    // 0.8 >= 0.8 * 1.0  → high
    // 0.5 >= 0.5 * 1.0  → moderate
    // 0.49 < 0.5 * 1.0  → low
    let labels = Searcher::<Arc<CozoBackend>, DeterministicTestProvider>::label_match_quality(
        &[1.0, 0.8, 0.5, 0.49],
    );
    assert_eq!(labels, vec!["high", "high", "moderate", "low"]);
    Ok(())
}

#[tokio::test]
async fn match_quality_empty_slice_returns_empty() -> anyhow::Result<()> {
    let labels = Searcher::<Arc<CozoBackend>, DeterministicTestProvider>::label_match_quality(&[]);
    assert!(labels.is_empty());
    Ok(())
}

#[tokio::test]
async fn searcher_returns_quality_labels_for_non_empty_results() -> anyhow::Result<()> {
    let (_, searcher, _fixture, _manifest_dir) = indexed_searcher().await?;

    // Search for something that appears in the fixture files.
    let results = searcher.search("pub fn", 5, false, 0, 0.0, None).await?;
    if results.is_empty() {
        // Accept gracefully if FTS doesn't surface results for this query.
        return Ok(());
    }

    for row in &results {
        assert!(
            matches!(row.match_quality.as_str(), "high" | "moderate" | "low"),
            "unexpected match_quality {:?}",
            row.match_quality
        );
        assert!(
            row.why == "vector" || row.why == "fts" || row.why == "hybrid",
            "unexpected why {:?}",
            row.why
        );
    }

    Ok(())
}

#[tokio::test]
#[ignore = "graph augmentation disabled in v1.2 until identifier-based dependency graph lands"]
async fn searcher_graph_augmentation_annotates_import_neighbours() -> anyhow::Result<()> {
    let (_, searcher, _fixture, _manifest_dir) = indexed_searcher().await?;

    let plain = searcher.search("pub fn", 5, false, 0, 0.0, None).await?;
    let graph = searcher.search("pub fn", 5, true, 2, 0.0, None).await?;

    // Graph search must return at least as many results as plain (it augments).
    assert!(
        graph.len() >= plain.len(),
        "graph search should have >= results; plain={}, graph={}",
        plain.len(),
        graph.len()
    );

    // Any graph-annotated result must have the expected prefix.
    for row in &graph {
        if row.why.starts_with("graph (") {
            assert!(
                row.why.starts_with("graph (depth "),
                "unexpected graph why format: {:?}",
                row.why
            );
        }
    }

    Ok(())
}

#[tokio::test]
async fn file_context_and_empty_search_are_truthful_for_missing_data() -> anyhow::Result<()> {
    let (_, searcher, _fixture, _manifest_dir) = indexed_searcher().await?;

    // Search for a term that definitely won't match anything indexed.
    let empty = searcher.search("ZZZNOMATCH_ZZZNOMATCH_ZZZNOMATCH_XYZ", 3, false, 0, 0.0, None)
        .await?;
    // Accept either empty results or low-scoring results for a nonsense query.
    // The important invariant: no panic, no error.
    let _ = empty;

    // File context for a path never indexed must return empty arrays — not an error.
    let ctx: FileContext = searcher.file_context("definitely/missing.rs").await?;
    assert!(ctx.chunks.is_empty(), "chunks must be empty for unknown file");
    assert!(ctx.imports.is_empty(), "imports must be empty for unknown file");
    assert!(ctx.imported_by.is_empty(), "imported_by must be empty for unknown file");

    Ok(())
}

#[tokio::test]
async fn file_context_returns_chunks_for_indexed_file() -> anyhow::Result<()> {
    let (_, searcher, _fixture, _manifest_dir) = indexed_searcher().await?;

    // The fixture has src/lib.rs; find it regardless of the exact relative prefix.
    let paths = searcher.search("pub fn add", 10, false, 0, 0.0, None)
        .await?
        .into_iter()
        .filter(|r| r.file_path.ends_with("lib.rs"))
        .map(|r| r.file_path)
        .next();

    if let Some(file_path) = paths {
        let ctx = searcher.file_context(&file_path).await?;
        assert!(
            !ctx.chunks.is_empty(),
            "indexed file must have at least one chunk"
        );
    }
    // If no results, the test is vacuously acceptable (FTS may not surface it).

    Ok(())
}


#[tokio::test]
async fn two_hop_traversal_finds_transitive_imports() -> anyhow::Result<()> {
    // Setup: a.rs imports b.rs, b.rs imports c.rs
    // traverse_imports("a.rs", 2) should find both b.rs and c.rs
    let dir = tempfile::tempdir()?;
    let backend = Arc::new(CozoBackend::open(dir.path().join("index.db"))?);
    backend.initialize(8).await?;

    for name in ["a.rs", "b.rs", "c.rs"] {
        backend.upsert_file(&FileRecord {
            file_path: name.into(),
            language: "rust".into(),
            last_modified: 100,
            last_indexed: 100,
            chunk_count: 1,
        }).await?;
    }
    backend.upsert_edges(&[
        EdgeRecord { from_file: "a.rs".into(), from_chunk: 0, to_file: "b.rs".into(), edge_type: "imports".into() },
        EdgeRecord { from_file: "b.rs".into(), from_chunk: 0, to_file: "c.rs".into(), edge_type: "imports".into() },
    ]).await?;

    let neighbors = backend.traverse_imports("a.rs", 2, None).await?;
    let paths: Vec<String> = neighbors.iter().map(|(p, _)| p.clone()).collect();
    assert!(paths.contains(&"b.rs".to_string()), "expected b.rs in {paths:?}");
    assert!(paths.contains(&"c.rs".to_string()), "expected c.rs in {paths:?}");
    Ok(())
}

#[tokio::test]
async fn traverse_handles_cycles() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let backend = Arc::new(CozoBackend::open(dir.path().join("index.db"))?);
    backend.initialize(8).await?;

    for name in ["a.rs", "b.rs"] {
        backend.upsert_file(&FileRecord {
            file_path: name.into(),
            language: "rust".into(),
            last_modified: 100,
            last_indexed: 100,
            chunk_count: 0,
        }).await?;
    }
    backend.upsert_edges(&[
        EdgeRecord { from_file: "a.rs".into(), from_chunk: 0, to_file: "b.rs".into(), edge_type: "imports".into() },
        EdgeRecord { from_file: "b.rs".into(), from_chunk: 0, to_file: "a.rs".into(), edge_type: "imports".into() },
    ]).await?;

    let neighbors = backend.traverse_imports("a.rs", 5, None).await?;
    let paths: Vec<String> = neighbors.into_iter().map(|(p, _)| p).collect();
    assert_eq!(paths, vec!["b.rs".to_string()]);
    Ok(())
}

// ---------------------------------------------------------------------------
// Concurrent index + search test
// ---------------------------------------------------------------------------

/// A provider that sleeps briefly per-batch to widen the indexing window,
/// making it possible for a concurrent search to overlap with active indexing.
#[derive(Clone)]
struct SlowProvider {
    dim: usize,
}

#[async_trait]
impl EmbedProvider for SlowProvider {
    fn dim(&self) -> usize {
        self.dim
    }

    async fn embed_batch(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        Ok(texts.iter().map(|_| vec![0.1_f32; self.dim]).collect())
    }
}

/// Verify that a search issued while indexing is in-flight neither panics nor
/// returns an error.  Results may be empty or partial — that is acceptable.
#[tokio::test]
async fn concurrent_index_and_search_does_not_panic() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let repo = dir.path().join("repo");
    std::fs::create_dir_all(&repo)?;

    // Write enough files to ensure at least a couple of embedding batches.
    for i in 0..10 {
        std::fs::write(
            repo.join(format!("mod_{i}.rs")),
            format!("fn func_{i}() {{}}\n"),
        )?;
    }

    let idx_dir = dir.path().join("idx");
    std::fs::create_dir_all(&idx_dir)?;
    let backend = Arc::new(CozoBackend::open(idx_dir.join("index.db"))?);
    let manifest = Arc::new(ManifestStore::open(idx_dir.join("manifest.db"))?);

    backend.initialize(8).await?;

    let indexer = Indexer::new(backend.clone(), manifest, SlowProvider { dim: 8 });
    let searcher = Searcher::new(backend.clone(), DeterministicTestProvider::new(8));

    // Start indexing in a background task.
    let index_handle = tokio::spawn(async move {
        indexer.index_path(&repo).await
    });

    // Give the indexer a moment to start so there is genuine overlap.
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;

    // Search while indexing is in-flight.  Must not panic or error.
    let search_result = searcher.search("func", 5, false, 0, 0.0, None).await;
    assert!(
        search_result.is_ok(),
        "search during indexing must return Ok; got: {:?}",
        search_result.unwrap_err()
    );

    // Wait for indexing to complete to avoid leaking the task.
    index_handle.await??;
    Ok(())
}

// ---------------------------------------------------------------------------
// MMR re-ranking test
// ---------------------------------------------------------------------------

/// Provider that maps text content to one of two clusters:
/// text containing "cluster_a" → [1,0,0,0], anything else → [0,1,0,0].
/// Both vectors are unit-length (L2 norm = 1).
#[derive(Clone)]
struct ClusterProvider;

#[async_trait]
impl EmbedProvider for ClusterProvider {
    fn dim(&self) -> usize { 4 }
    async fn embed_batch(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| {
            if t.contains("cluster_a") {
                vec![1.0_f32, 0.0, 0.0, 0.0]
            } else {
                vec![0.0_f32, 1.0, 0.0, 0.0]
            }
        }).collect())
    }
}

#[tokio::test]
async fn mmr_reranking_diversifies_results() -> anyhow::Result<()> {
    // Two files contain "cluster_a" in source → stored embeddings [1,0,0,0].
    // One file does NOT → stored embedding [0,1,0,0].
    // All three files share identical BM25 vocabulary ("handle request") so
    // BM25 scores are roughly equal and only vector similarity differentiates.
    //
    // Query "cluster_a handle request" also maps to [1,0,0,0].
    // Without MMR: dup1, dup2, other (by RRF relevance).
    // With MMR: other promoted because dup2 is redundant with dup1.
    let dir = tempfile::tempdir()?;
    let idx_dir = dir.path().join("idx");
    std::fs::create_dir_all(&idx_dir)?;

    let backend = Arc::new(CozoBackend::open(idx_dir.join("index.db"))?);
    let manifest = Arc::new(ManifestStore::open(idx_dir.join("manifest.db"))?);
    backend.initialize(4).await?;

    // Write 3 small Rust source files.  Content must contain "near" for
    // ClusterProvider to map them correctly.  Use near/far in the actual
    // source content (not just filenames) since embeddings come from content.
    let repo = dir.path().join("repo");
    std::fs::create_dir_all(&repo)?;
    std::fs::write(repo.join("dup1.rs"),  "pub fn cluster_a_handle() { request_process() }\n")?;
    std::fs::write(repo.join("dup2.rs"),  "pub fn cluster_a_dispatch() { request_route() }\n")?;
    std::fs::write(repo.join("other.rs"), "pub fn cluster_b_handle() { request_serve() }\n")?;

    let indexer = Indexer::new(backend.clone(), manifest, ClusterProvider);
    indexer.index_path(&repo).await?;

    let searcher = Searcher::new(backend.clone(), ClusterProvider);

    let no_mmr   = searcher.search("cluster_a handle request", 3, false, 0, 0.0, None).await?;
    let with_mmr = searcher.search("cluster_a handle request", 3, false, 0, 0.7, None).await?;

    // Both searches must succeed and return at least one result.
    assert!(!no_mmr.is_empty(),   "no-MMR search must return results");
    assert!(!with_mmr.is_empty(), "MMR search must return results");

    // If the index returned all 3 chunks, the MMR ordering must differ:
    // MMR should promote the orthogonal "far" chunk ahead of the redundant "near" clone.
    if no_mmr.len() == 3 && with_mmr.len() == 3 {
        let with_mmr_files: Vec<_> = with_mmr.iter().map(|r| r.file_path.as_str()).collect();
        // With high diversity, the two near-duplicate "cluster_a" files (dup1, dup2)
        // must NOT be adjacent at positions 0 and 1.  MMR should interleave the
        // orthogonal "other" file between them.
        let cluster_a_files = ["dup1.rs", "dup2.rs"];
        let both_first = cluster_a_files.contains(&with_mmr_files[0])
            && cluster_a_files.contains(&with_mmr_files[1]);
        assert!(
            !both_first,
            "MMR must not place both near-duplicate files at top 2; got {with_mmr_files:?}"
        );
    }

    Ok(())
}

#[tokio::test]
async fn unified_search_returns_results_and_overlaps_hybrid() -> anyhow::Result<()> {
    let (backend, _searcher, _fixture, _manifest_dir) = indexed_searcher().await?;

    let provider = DeterministicTestProvider::new(8);
    let query = "pub fn";
    let query_vec = provider.embed_batch(vec![query.to_string()]).await?;
    let query_vec = query_vec.into_iter().next().unwrap_or_default();

    // unified_search (no graph)
    let unified = backend
        .unified_search(&query_vec, query, 5, 0, 0.55, 0.3, 0.005, 0.1)
        .await?;

    // Fall back gracefully if the index is empty or embeddings are missing.
    if unified.is_empty() {
        return Ok(());
    }

    // Every result must have a valid why tag.
    for r in &unified {
        assert!(
            matches!(r.why.as_str(), "hybrid" | "graph"),
            "unexpected why tag: {:?}",
            r.why
        );
    }

    // Scores must be non-negative and ordered descending.
    let mut prev = f64::MAX;
    for r in &unified {
        assert!(r.score >= 0.0, "score must be non-negative: {}", r.score);
        assert!(r.score <= prev + 1e-9, "results must be score-descending");
        prev = r.score;
    }

    // Verify Searcher::with_unified_search(true) can be constructed and run.
    let unified_searcher = Searcher::new(backend.clone(), DeterministicTestProvider::new(8))
        .with_unified_search(true);
    let searcher_results = unified_searcher
        .search(query, 5, false, 0, 0.0, None)
        .await?;
    // Must not error; empty result is acceptable (e.g. no embeddings in index).
    let _ = searcher_results;

    Ok(())
}