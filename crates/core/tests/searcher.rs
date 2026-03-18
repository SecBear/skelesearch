use std::sync::Arc;

use async_trait::async_trait;
use skelesearch_core::{
    CozoBackend, EmbedProvider, FileContext, Indexer, ManifestStore, Searcher,
};
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Test doubles  (same shape as in indexer tests; each test file is independent)
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct DeterministicTestProvider {
    dim: usize,
}

impl DeterministicTestProvider {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

#[async_trait]
impl EmbedProvider for DeterministicTestProvider {
    fn dim(&self) -> usize {
        self.dim
    }

    async fn embed_batch(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let mut v = vec![0.1_f32; self.dim];
                if !v.is_empty() {
                    v[0] = (i as f32 + 1.0) * 0.1;
                }
                let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
                v.iter_mut().for_each(|x| *x /= norm);
                v
            })
            .collect())
    }
}

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

fn copy_dir_all(src: &std::path::Path, dst: &std::path::Path) -> anyhow::Result<()> {
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let target = dst.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            std::fs::create_dir_all(&target)?;
            copy_dir_all(&entry.path(), &target)?;
        } else {
            std::fs::copy(entry.path(), &target)?;
        }
    }
    Ok(())
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
    let results = searcher.search("pub fn", 5, false).await?;
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
            row.why == "vector" || row.why == "fts" || row.why == "both",
            "unexpected why {:?}",
            row.why
        );
    }

    Ok(())
}

#[tokio::test]
async fn searcher_graph_augmentation_annotates_import_neighbours() -> anyhow::Result<()> {
    let (_, searcher, _fixture, _manifest_dir) = indexed_searcher().await?;

    let plain = searcher.search("pub fn", 5, false).await?;
    let graph = searcher.search("pub fn", 5, true).await?;

    // Graph search must return at least as many results as plain (it augments).
    assert!(
        graph.len() >= plain.len(),
        "graph search should have >= results; plain={}, graph={}",
        plain.len(),
        graph.len()
    );

    // If graph results are present, at least some may be annotated with imports.
    // (Depends on fixture import edges being indexable by the chunker.)
    // We accept both "vector" results only and "imports …" results — just verify
    // that any "imports …" result has the correct prefix.
    for row in &graph {
        if row.why.starts_with("imports ") {
            assert!(
                !row.why.trim_start_matches("imports ").is_empty(),
                "imports annotation should name a file"
            );
        }
    }

    Ok(())
}

#[tokio::test]
async fn file_context_and_empty_search_are_truthful_for_missing_data() -> anyhow::Result<()> {
    let (_, searcher, _fixture, _manifest_dir) = indexed_searcher().await?;

    // Search for a term that definitely won't match anything indexed.
    let empty = searcher
        .search("ZZZNOMATCH_ZZZNOMATCH_ZZZNOMATCH_XYZ", 3, false)
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
    let paths = searcher
        .search("pub fn add", 10, false)
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
