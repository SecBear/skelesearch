use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use skelesearch_core::{
    CozoBackend, EmbedProvider, IndexResult, Indexer, ManifestStore, StorageBackend,
};
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Test doubles
// ---------------------------------------------------------------------------

/// Returns identical fixed vectors of the given dimensionality for every input.
///
/// Deterministic output allows round-trip tests without a real model.
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
                // Slightly vary vectors per-chunk so FTS and vector scores differ.
                let mut v = vec![0.1_f32; self.dim];
                if !v.is_empty() {
                    v[0] = (i as f32 + 1.0) * 0.1;
                }
                // Normalise to unit length so cosine similarity is well-defined.
                let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
                v.iter_mut().for_each(|x| *x /= norm);
                v
            })
            .collect())
    }
}

/// Counts `embed_batch` calls and total chunk texts seen across all calls.
///
/// Used to verify that the indexer sends chunks in batches rather than one
/// call per chunk.
#[derive(Clone)]
pub struct CountingTestProvider {
    dim: usize,
    calls: Arc<Mutex<usize>>,
    chunks: Arc<Mutex<usize>>,
}

impl CountingTestProvider {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            calls: Arc::new(Mutex::new(0)),
            chunks: Arc::new(Mutex::new(0)),
        }
    }

    /// Number of `embed_batch` calls made so far.
    pub fn call_count(&self) -> usize {
        *self.calls.lock().unwrap()
    }

    /// Total number of chunk texts seen across all calls.
    pub fn chunk_count_seen(&self) -> usize {
        *self.chunks.lock().unwrap()
    }
}

#[async_trait]
impl EmbedProvider for CountingTestProvider {
    fn dim(&self) -> usize {
        self.dim
    }

    async fn embed_batch(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
        *self.calls.lock().unwrap() += 1;
        *self.chunks.lock().unwrap() += texts.len();
        Ok(texts.iter().map(|_| vec![0.1_f32; self.dim]).collect())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create an in-memory-ish CozoBackend in a temp directory.
fn test_backend() -> anyhow::Result<Arc<CozoBackend>> {
    let dir = tempfile::tempdir()?;
    let backend = CozoBackend::open(dir.path().join("index.db"))?;
    // Leak the TempDir so it lives for the duration of the test.
    std::mem::forget(dir);
    Ok(Arc::new(backend))
}

/// Create a ManifestStore in a temp directory.
fn test_manifest() -> anyhow::Result<(Arc<ManifestStore>, TempDir)> {
    let dir = tempfile::tempdir()?;
    let store = ManifestStore::open(dir.path().join("manifest.db"))?;
    Ok((Arc::new(store), dir))
}

/// Copy the on-disk fixture repo into a fresh temp directory so each test
/// gets an independent, writable snapshot.
fn fixture_repo() -> anyhow::Result<TempDir> {
    let dir = tempfile::tempdir()?;
    let fixture_src = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/sample_repo");
    // Recursively copy src/ sub-tree.
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn indexer_skips_unchanged_files_reconciles_renames_and_removes_deleted_paths(
) -> anyhow::Result<()> {
    let fixture = fixture_repo()?;
    let backend = test_backend()?;
    let (manifest, _manifest_dir) = test_manifest()?;
    let provider = DeterministicTestProvider::new(8);
    let indexer = Indexer::new(backend.clone(), manifest.clone(), provider);

    // First pass: index everything.
    let first: IndexResult = indexer.index_path(fixture.path()).await?;
    assert!(first.indexed_files >= 1, "expected at least one file indexed");

    // Rename old.rs → new.rs, simulating a file move.
    std::fs::rename(
        fixture.path().join("src/old.rs"),
        fixture.path().join("src/new.rs"),
    )?;

    // Second pass: should detect the rename (old.rs stale, new.rs fresh).
    let second: IndexResult = indexer.index_path(fixture.path()).await?;
    assert!(second.deleted_files >= 1, "expected at least one deleted path");

    // new.rs must be in the index, old.rs must not.
    let paths = backend.list_indexed_paths().await?;
    assert!(
        paths.iter().any(|p| p.ends_with("new.rs")),
        "new.rs should be indexed; got {paths:?}"
    );
    assert!(
        !paths.iter().any(|p| p.ends_with("old.rs")),
        "old.rs should have been removed; got {paths:?}"
    );

    // Backend must have no chunks or imports for the stale path.
    assert!(
        backend.get_chunks_for_file("src/old.rs").await?.is_empty(),
        "stale chunks must be deleted"
    );
    assert!(
        backend.get_imports("src/old.rs").await?.is_empty(),
        "stale edges must be deleted"
    );

    // Manifest must not record the old path.
    assert!(
        !manifest.list_paths()?.iter().any(|p| p.ends_with("old.rs")),
        "manifest should not record old.rs"
    );

    Ok(())
}

#[tokio::test]
async fn indexer_batches_embeddings_instead_of_one_call_per_chunk() -> anyhow::Result<()> {
    let fixture = fixture_repo()?;
    let (manifest, _manifest_dir) = test_manifest()?;
    let provider = CountingTestProvider::new(8);
    let indexer = Indexer::new(test_backend()?, manifest, provider.clone());

    indexer.index_path(fixture.path()).await?;

    let calls = provider.call_count();
    let total = provider.chunk_count_seen();

    // At least two chunks total (two fixture files, each with at least one chunk).
    // The indexer must have batched them rather than calling once per chunk.
    assert!(
        total >= 2,
        "expected >= 2 chunks total; got {total} — fixture files may be too small"
    );
    assert!(
        calls < total,
        "expected fewer embed_batch calls ({calls}) than chunks ({total})"
    );

    Ok(())
}

#[tokio::test]
async fn indexer_updates_last_indexed_after_successful_index() -> anyhow::Result<()> {
    let fixture = fixture_repo()?;
    let backend = test_backend()?;
    let (manifest, _manifest_dir) = test_manifest()?;
    let provider = DeterministicTestProvider::new(8);
    let indexer = Indexer::new(backend.clone(), manifest, provider);

    indexer.index_path(fixture.path()).await?;

    let stats = backend.stats().await?;
    assert!(
        stats.last_indexed.is_some(),
        "last_indexed must be set after a successful index run"
    );

    Ok(())
}

#[tokio::test]
async fn indexer_second_pass_skips_unchanged_files() -> anyhow::Result<()> {
    let fixture = fixture_repo()?;
    let backend = test_backend()?;
    let (manifest, _manifest_dir) = test_manifest()?;
    let provider = CountingTestProvider::new(8);
    let indexer = Indexer::new(backend.clone(), manifest, provider.clone());

    indexer.index_path(fixture.path()).await?;
    let calls_after_first = provider.call_count();

    // Second pass with no file changes: no embed_batch calls should happen.
    indexer.index_path(fixture.path()).await?;
    let calls_after_second = provider.call_count();

    assert_eq!(
        calls_after_first, calls_after_second,
        "unchanged files must not trigger re-embedding"
    );

    Ok(())
}

#[tokio::test]
async fn indexer_handles_empty_directory_gracefully() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let backend = test_backend()?;
    let (manifest, _manifest_dir) = test_manifest()?;
    let provider = DeterministicTestProvider::new(8);
    let indexer = Indexer::new(backend.clone(), manifest, provider);

    let result = indexer.index_path(dir.path()).await?;

    assert_eq!(result.indexed_files, 0);
    assert_eq!(result.deleted_files, 0);
    assert_eq!(result.total_chunks, 0);

    Ok(())
}


// ---------------------------------------------------------------------------
// BatchTrackingProvider — records call sizes for streaming assertions
// ---------------------------------------------------------------------------

/// Tracks the size of every `embed_batch` call so tests can assert that no
/// single call exceeds the configured `batch_size` limit.
struct BatchTrackingProvider {
    dim: usize,
    call_sizes: std::sync::Mutex<Vec<usize>>,
}

impl BatchTrackingProvider {
    fn new(dim: usize) -> Self {
        Self { dim, call_sizes: std::sync::Mutex::new(Vec::new()) }
    }

    fn call_sizes(&self) -> Vec<usize> {
        self.call_sizes.lock().unwrap().clone()
    }
}

#[async_trait]
impl EmbedProvider for BatchTrackingProvider {
    fn dim(&self) -> usize {
        self.dim
    }

    async fn embed_batch(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
        self.call_sizes.lock().unwrap().push(texts.len());
        Ok(texts.iter().map(|_| vec![0.1_f32; self.dim]).collect())
    }
}

// ---------------------------------------------------------------------------
// Streaming pipeline test
// ---------------------------------------------------------------------------

#[tokio::test]
async fn indexer_processes_in_bounded_file_batches() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let repo = dir.path().join("repo");
    std::fs::create_dir_all(&repo)?;

    // 60 small .rs files: exceeds FILE_BATCH_SIZE of 50, so at least two
    // pipeline batches will run.
    for i in 0..60 {
        std::fs::write(repo.join(format!("mod_{i}.rs")), format!("fn func_{i}() {{}}"))?;
    }

    let idx_dir = dir.path().join("idx");
    std::fs::create_dir_all(&idx_dir)?;
    let backend = std::sync::Arc::new(skelesearch_core::CozoBackend::open(idx_dir.join("index.db"))?);
    let manifest = std::sync::Arc::new(skelesearch_core::ManifestStore::open(idx_dir.join("manifest.db"))?);
    let provider = BatchTrackingProvider::new(8);

    backend.initialize(8).await?;
    let indexer = skelesearch_core::Indexer::new(backend, manifest, provider);
    let result = indexer.index_path(&repo).await?;

    assert_eq!(result.indexed_files, 60, "all 60 files should be indexed");

    // No single embed_batch call may exceed batch_size (64) — the streaming
    // design processes at most FILE_BATCH_SIZE (50) files per outer batch,
    // and each file contributes at most a handful of chunks.
    let sizes = indexer.provider().call_sizes();
    assert!(!sizes.is_empty(), "expected at least one embed call");
    for &size in &sizes {
        assert!(size <= 64, "embed_batch call exceeded batch_size: {size}");
    }

    Ok(())
}