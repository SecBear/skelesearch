use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use rusqlite::{params, Connection, OptionalExtension};
mod test_utils;
use test_utils::{copy_dir_all, DeterministicTestProvider};
use skelesearch_core::{
    CompositeBackend, EmbedProvider, IndexResult, Indexer, ManifestStore, StorageBackend,
};
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Local test doubles (DeterministicTestProvider lives in test_utils)
// ---------------------------------------------------------------------------

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

#[derive(Clone)]
struct NamedCountingProvider {
    name: &'static str,
    dim: usize,
    calls: Arc<Mutex<usize>>,
    chunks: Arc<Mutex<usize>>,
}

impl NamedCountingProvider {
    pub fn new(name: &'static str, dim: usize) -> Self {
        Self {
            name,
            dim,
            calls: Arc::new(Mutex::new(0)),
            chunks: Arc::new(Mutex::new(0)),
        }
    }

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

#[async_trait]
impl EmbedProvider for NamedCountingProvider {
    fn dim(&self) -> usize {
        self.dim
    }

    fn name(&self) -> &str {
        self.name
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

/// Create an in-memory-ish CompositeBackend in a temp directory.
async fn test_backend() -> anyhow::Result<Arc<CompositeBackend>> {
    let dir = tempfile::tempdir()?;
    let backend = CompositeBackend::open(dir.path()).await?;
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

fn seed_embedding_cache_entry(
    db_path: &std::path::Path,
    content_hash: &str,
    dim: usize,
) -> anyhow::Result<()> {
    let conn = Connection::open(db_path)?;
    let bytes: Vec<u8> = std::iter::repeat(0.25_f32)
        .take(dim)
        .flat_map(|f| f.to_le_bytes())
        .collect();
    conn.execute(
        "INSERT OR REPLACE INTO embedding_cache (content_hash, dim, embedding) VALUES (?1, ?2, ?3)",
        params![content_hash, dim as i64, bytes],
    )?;
    Ok(())
}

fn embedding_cache_has_entry(db_path: &std::path::Path, content_hash: &str) -> anyhow::Result<bool> {
    let conn = Connection::open(db_path)?;
    let present: Option<i64> = conn
        .query_row(
            "SELECT 1 FROM embedding_cache WHERE content_hash = ?1",
            params![content_hash],
            |row| row.get(0),
        )
        .optional()?;
    Ok(present.is_some())
}


// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn indexer_skips_unchanged_files_reconciles_renames_and_removes_deleted_paths(
) -> anyhow::Result<()> {
    let fixture = fixture_repo()?;
    let backend = test_backend().await?;
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
    let indexer = Indexer::new(test_backend().await?, manifest, provider.clone());

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
    let backend = test_backend().await?;
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
    let backend = test_backend().await?;
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
async fn stale_refresh_preserves_cache_hits() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let repo = dir.path().join("repo");
    std::fs::create_dir_all(&repo)?;

    std::fs::write(repo.join("alpha.rs"), "fn alpha_one() {}\nfn alpha_two() {}\n")?;
    std::fs::write(repo.join("beta.rs"), "fn beta_one() {}\n")?;

    let idx_dir = dir.path().join("idx");
    std::fs::create_dir_all(&idx_dir)?;
    let backend = Arc::new(CompositeBackend::open(&idx_dir).await?);
    let manifest = Arc::new(ManifestStore::open(idx_dir.join("manifest.db"))?);

    backend.initialize(8).await?;
    let provider = CountingTestProvider::new(8);
    let indexer = Indexer::new(backend.clone(), manifest.clone(), provider.clone());

    indexer.index_path(&repo).await?;
    let first_seen = provider.chunk_count_seen();
    assert!(first_seen > 0, "expected initial chunks for both files");

    std::fs::write(
        repo.join("alpha.rs"),
        "fn alpha_one() {}\nfn alpha_two() {}\nfn alpha_three() {}\n",
    )?;

    indexer.index_path(&repo).await?;
    let second_seen = provider.chunk_count_seen();

    assert_eq!(
        second_seen - first_seen,
        1,
        "stale refresh should reuse cached embeddings for unchanged chunks"
    );
    Ok(())
}

#[tokio::test]
async fn provider_change_clears_embedding_cache_and_file_hashes() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let repo = dir.path().join("repo");
    std::fs::create_dir_all(&repo)?;

    std::fs::write(repo.join("alpha.rs"), "fn alpha_one() {}\nfn alpha_two() {}\n")?;

    let idx_dir = dir.path().join("idx");
    std::fs::create_dir_all(&idx_dir)?;
    let backend = Arc::new(CompositeBackend::open(&idx_dir).await?);
    let manifest = Arc::new(ManifestStore::open(idx_dir.join("manifest.db"))?);
    let manifest_db = idx_dir.join("manifest.db");

    backend.initialize(8).await?;
    let provider_a = NamedCountingProvider::new("provider-a", 8);
    let indexer_a = Indexer::new(backend.clone(), manifest.clone(), provider_a.clone());
    indexer_a.index_path(&repo).await?;
    let first_seen = provider_a.chunk_count_seen();

    let sentinel_hash = "9f5f4b0e0d8d4d6e";
    seed_embedding_cache_entry(&manifest_db, sentinel_hash, 8)?;
    assert!(embedding_cache_has_entry(&manifest_db, sentinel_hash)?);

    let provider_b = NamedCountingProvider::new("provider-b", 8);
    let indexer_b = Indexer::new(backend.clone(), manifest.clone(), provider_b.clone());
    indexer_b.index_path(&repo).await?;

    assert_eq!(
        provider_b.chunk_count_seen(),
        first_seen,
        "provider change must force a full re-embed"
    );
    assert!(
        !embedding_cache_has_entry(&manifest_db, sentinel_hash)?,
        "provider change must clear stale embedding cache rows"
    );
    Ok(())
}

#[tokio::test]
async fn dimension_change_clears_embedding_cache_and_file_hashes() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let repo = dir.path().join("repo");
    std::fs::create_dir_all(&repo)?;

    std::fs::write(repo.join("alpha.rs"), "fn alpha_one() {}\nfn alpha_two() {}\n")?;

    let idx_dir = dir.path().join("idx");
    std::fs::create_dir_all(&idx_dir)?;
    let backend = Arc::new(CompositeBackend::open(&idx_dir).await?);
    let manifest = Arc::new(ManifestStore::open(idx_dir.join("manifest.db"))?);
    let manifest_db = idx_dir.join("manifest.db");

    backend.initialize(8).await?;
    let provider_a = NamedCountingProvider::new("provider-a", 8);
    let indexer_a = Indexer::new(backend.clone(), manifest.clone(), provider_a.clone());
    indexer_a.index_path(&repo).await?;
    let first_seen = provider_a.chunk_count_seen();

    let sentinel_hash = "3c4dd5c1e1b94c4f";
    seed_embedding_cache_entry(&manifest_db, sentinel_hash, 8)?;
    assert!(embedding_cache_has_entry(&manifest_db, sentinel_hash)?);

    backend.initialize(9).await?;
    let provider_b = NamedCountingProvider::new("provider-a", 9);
    let indexer_b = Indexer::new(backend.clone(), manifest.clone(), provider_b.clone());
    indexer_b.index_path(&repo).await?;

    assert_eq!(
        provider_b.chunk_count_seen(),
        first_seen,
        "dimension change must force a full re-embed"
    );
    assert!(
        !embedding_cache_has_entry(&manifest_db, sentinel_hash)?,
        "dimension change must clear stale embedding cache rows"
    );
    Ok(())
}

#[tokio::test]
async fn indexer_handles_empty_directory_gracefully() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let backend = test_backend().await?;
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
    let backend = std::sync::Arc::new(skelesearch_core::CompositeBackend::open(&idx_dir).await?);
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

// ---------------------------------------------------------------------------
// Error-path tests
// ---------------------------------------------------------------------------

/// A provider that returns fewer vectors than texts — triggers the mismatch
/// guard in the indexer's embedding loop.
#[derive(Clone)]
struct ShortProvider {
    dim: usize,
}

#[async_trait]
impl EmbedProvider for ShortProvider {
    fn dim(&self) -> usize {
        self.dim
    }

    async fn embed_batch(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
        // Return one fewer vector than requested to trigger the mismatch guard.
        let count = texts.len().saturating_sub(1);
        Ok((0..count).map(|_| vec![0.1_f32; self.dim]).collect())
    }
}

#[tokio::test]
async fn embedding_count_mismatch_returns_error() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let repo = dir.path().join("repo");
    std::fs::create_dir_all(&repo)?;
    // Three functions → at least three chunks → short provider fires on the first non-empty batch.
    std::fs::write(repo.join("a.rs"), "fn alpha() {}\nfn beta() {}\nfn gamma() {}\n")?;

    let idx_dir = dir.path().join("idx");
    std::fs::create_dir_all(&idx_dir)?;
    let backend = Arc::new(CompositeBackend::open(&idx_dir).await?);
    let manifest = Arc::new(ManifestStore::open(idx_dir.join("manifest.db"))?);

    backend.initialize(8).await?;
    let indexer = Indexer::new(backend, manifest, ShortProvider { dim: 8 });
    let err = indexer.index_path(&repo).await;

    assert!(err.is_err(), "expected Err from count mismatch but got Ok");
    let msg = format!("{:#}", err.unwrap_err());
    assert!(
        msg.contains("embedding count mismatch"),
        "error should mention 'embedding count mismatch', got: {msg}"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Binary-file skip test
// ---------------------------------------------------------------------------

#[tokio::test]
async fn binary_files_are_skipped() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let repo = dir.path().join("repo");
    std::fs::create_dir_all(&repo)?;

    // A normal Rust source file — must appear in the index.
    std::fs::write(repo.join("main.rs"), "fn main() {}\n")?;
    // A binary file containing a null byte — must be skipped entirely.
    std::fs::write(repo.join("data.bin"), b"\x00\x01\x02binary content\xFF")?;

    let idx_dir = dir.path().join("idx");
    std::fs::create_dir_all(&idx_dir)?;
    let backend = Arc::new(CompositeBackend::open(&idx_dir).await?);
    let manifest = Arc::new(ManifestStore::open(idx_dir.join("manifest.db"))?);

    backend.initialize(8).await?;
    let indexer = Indexer::new(backend.clone(), manifest, DeterministicTestProvider::new(8));
    let result = indexer.index_path(&repo).await?;

    // Exactly one file was indexed — the Rust source.
    assert_eq!(result.indexed_files, 1, "only the text file should be indexed");

    // The binary file must have no chunks stored in the backend.
    let binary_chunks = backend.get_chunks_for_file("data.bin").await?;
    assert!(binary_chunks.is_empty(), "binary file must have no stored chunks");

    // The text file must appear in the indexed paths.
    let paths = backend.list_indexed_paths().await?;
    assert!(
        paths.iter().any(|p| p.contains("main.rs")),
        "main.rs should appear in indexed paths; got: {paths:?}"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Embedding cache test
// ---------------------------------------------------------------------------

/// Verifies that the embedding cache prevents redundant provider calls.
///
/// Strategy:
///   1. Index 3 source files — all chunks are embedded and cached.
///   2. Modify ONE file's content so the manifest detects it as changed.
///   3. Index again — only the changed file's chunks need embedding.
///   4. Assert the second run embedded fewer texts than the first.
///
/// The unchanged files are skipped entirely by manifest mtime+size detection.
/// The changed file's new chunks are fresh cache misses, but the volume is
/// strictly less than embedding all three files in the first run.
#[tokio::test]
async fn embedding_cache_reduces_provider_calls() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let repo = dir.path().join("repo");
    std::fs::create_dir_all(&repo)?;

    // Three source files; only one will be modified between runs.
    std::fs::write(repo.join("alpha.rs"), "fn alpha_one() {}\nfn alpha_two() {}\n")?;
    std::fs::write(repo.join("beta.rs"),  "fn beta_one() {}\nfn beta_two() {}\n")?;
    std::fs::write(repo.join("gamma.rs"), "fn gamma_one() {}\nfn gamma_two() {}\n")?;

    let idx_dir = dir.path().join("idx");
    std::fs::create_dir_all(&idx_dir)?;

    // Shared backend and manifest (same manifest DB = same embedding_cache).
    let backend = Arc::new(CompositeBackend::open(&idx_dir).await?);
    let manifest = Arc::new(ManifestStore::open(idx_dir.join("manifest.db"))?);

    // --- Run 1: all files are new, all chunks need embedding ---
    let provider1 = CountingTestProvider::new(8);
    backend.initialize(8).await?;
    let indexer1 = Indexer::new(backend.clone(), manifest.clone(), provider1);
    indexer1.index_path(&repo).await?;
    let first_texts = indexer1.provider().chunk_count_seen();
    assert!(first_texts > 0, "first run must embed at least one chunk");

    // --- Modify only gamma.rs so the manifest sees it as changed ---
    // Sleep briefly to ensure the filesystem mtime advances beyond second
    // granularity used by the manifest's mtime+size fast-skip check.
    std::thread::sleep(std::time::Duration::from_millis(10));
    std::fs::write(
        repo.join("gamma.rs"),
        "fn gamma_one() {}\nfn gamma_two() {}\nfn gamma_three() {}\n",
    )?;

    // --- Run 2: only gamma.rs is re-indexed; alpha+beta are unchanged ---
    let provider2 = CountingTestProvider::new(8);
    let indexer2 = Indexer::new(backend.clone(), manifest.clone(), provider2);
    indexer2.index_path(&repo).await?;
    let second_texts = indexer2.provider().chunk_count_seen();

    assert!(
        second_texts < first_texts,
        "second run should embed fewer texts than first (cache + manifest skips); \
         first={first_texts}, second={second_texts}"
    );
    Ok(())
}
