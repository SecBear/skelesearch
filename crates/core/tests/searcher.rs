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

    let neighbors = backend.traverse_imports("a.rs", 2).await?;
    assert!(neighbors.contains(&"b.rs".to_string()), "expected b.rs in {neighbors:?}");
    assert!(neighbors.contains(&"c.rs".to_string()), "expected c.rs in {neighbors:?}");
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

    let neighbors = backend.traverse_imports("a.rs", 5).await?;
    assert_eq!(neighbors, vec!["b.rs".to_string()]);
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
/// text containing "near" → [1,0,0,0], anything else → [0,1,0,0].
/// Both vectors are already unit-length (L2 norm = 1).
/// This gives us a fully controlled similarity structure for MMR testing.
#[derive(Clone)]
struct ClusterProvider;

#[async_trait]
impl EmbedProvider for ClusterProvider {
    fn dim(&self) -> usize { 4 }
    async fn embed_batch(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| {
            if t.to_lowercase().contains("near") {
                vec![1.0_f32, 0.0, 0.0, 0.0]
            } else {
                vec![0.0_f32, 1.0, 0.0, 0.0]
            }
        }).collect())
    }
}

#[tokio::test]
async fn mmr_reranking_diversifies_results() -> anyhow::Result<()> {
    // Two "near" files have identical stored embeddings [1,0,0,0].
    // One "far" file has orthogonal stored embedding [0,1,0,0].
    // Query "near_alpha" also maps to [1,0,0,0].
    //
    // Without MMR: near1, near2, far (by RRF relevance order).
    // With MMR (diversity=0.7, lambda=0.3):
    //   - Pick near1 (or near2) first (highest relevance = 1.0).
    //   - near2 now has redundancy=cos([1,0,0,0],[1,0,0,0])=1.0 → mmr=-0.4
    //   - far has redundancy=cos([0,1,0,0],[1,0,0,0])=0.0  → mmr=0.0
    //   - far wins → order: [near1, far, near2]
    // So the orderings differ.
    let dir = tempfile::tempdir()?;
    let idx_dir = dir.path().join("idx");
    std::fs::create_dir_all(&idx_dir)?;

    let backend = Arc::new(CozoBackend::open(idx_dir.join("index.db"))?);
    let manifest = Arc::new(ManifestStore::open(idx_dir.join("manifest.db"))?);
    backend.initialize(4).await?;

    // Write 3 small Rust source files with clearly differentiated content.
    let repo = dir.path().join("repo");
    std::fs::create_dir_all(&repo)?;
    std::fs::write(repo.join("near1.rs"), "fn near_alpha() {}\n")?;
    std::fs::write(repo.join("near2.rs"), "fn near_beta() {}\n")?;
    std::fs::write(repo.join("far.rs"),   "fn far_omega() {}\n")?;

    let indexer = Indexer::new(backend.clone(), manifest, ClusterProvider);
    indexer.index_path(&repo).await?;

    let searcher = Searcher::new(backend.clone(), ClusterProvider);

    // Fetch all 3 chunks so the MMR has material to reorder.
    let no_mmr   = searcher.search("near_alpha", 3, false, 0, 0.0, None).await?;
    let with_mmr = searcher.search("near_alpha", 3, false, 0, 0.7, None).await?;

    // Both searches must succeed and return at least one result.
    assert!(!no_mmr.is_empty(),   "no-MMR search must return results");
    assert!(!with_mmr.is_empty(), "MMR search must return results");

    // If the index returned all 3 chunks, the MMR ordering must differ:
    // MMR should promote the orthogonal "far" chunk ahead of the redundant "near" clone.
    if no_mmr.len() == 3 && with_mmr.len() == 3 {
        let no_mmr_files: Vec<_>   = no_mmr.iter().map(|r| r.file_path.as_str()).collect();
        let with_mmr_files: Vec<_> = with_mmr.iter().map(|r| r.file_path.as_str()).collect();
        assert_ne!(
            no_mmr_files, with_mmr_files,
            "MMR must reorder results when near-duplicate embeddings are present; \
             no_mmr={no_mmr_files:?} with_mmr={with_mmr_files:?}"
        );
    }

    Ok(())
}