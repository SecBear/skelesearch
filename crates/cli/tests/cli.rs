// Integration tests for the `skelesearch` CLI binary.
//
// Two fixture variants:
//
//   `indexed_cli_fixture()` — uses DeterministicTestProvider (no model
//   download) so that status, context, clear, and watch tests run without
//   network access.  The dim is set to 768 to match FastEmbedProvider so
//   the CLI binary can open the schema without a mismatch.
//
//   `indexed_cli_fixture_with_model()` — uses FastEmbedProvider.  Required
//   only for the `search` test because that test calls the CLI binary which
//   must embed the query.  Returns `None` when the model is unavailable so
//   the test can skip gracefully.

use std::sync::Arc;

use async_trait::async_trait;
use predicates::prelude::PredicateBooleanExt;
use skelesearch_core::{CozoBackend, EmbedProvider, Indexer, ManifestStore, StorageBackend};
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Test double — deterministic vectors, no model download, dim=768
// ---------------------------------------------------------------------------

/// Produces unit-length vectors of fixed dimensionality.
/// Dimension is 768 so the schema is compatible with FastEmbedProvider.
#[derive(Clone)]
struct DeterministicTestProvider {
    dim: usize,
}

impl DeterministicTestProvider {
    fn new(dim: usize) -> Self {
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
// Fixture source files (shared by both variants)
// ---------------------------------------------------------------------------

fn write_fixture_files(dir: &std::path::Path) {
    let src = dir.join("src");
    std::fs::create_dir_all(&src).unwrap();

    // lib.rs: contains "import edges" multiple times for FTS + uses `graph`
    // module so the chunker records an outbound import edge.
    std::fs::write(
        src.join("lib.rs"),
        r#"// This module tracks import edges between source files.
// Import edges are the core data structure of the dependency graph.
mod graph;

use graph::Edge;

/// Collect all import edges from the parsed source tree.
pub fn collect_import_edges() -> Vec<Edge> {
    graph::all_import_edges()
}

/// Count import edges originating from a given file.
pub fn count_import_edges(file: &str) -> usize {
    collect_import_edges()
        .into_iter()
        .filter(|e| e.from == file)
        .count()
}
"#,
    )
    .unwrap();

    // graph.rs: the module imported by lib.rs.
    std::fs::write(
        src.join("graph.rs"),
        r#"/// A directed import edge between two source files.
#[derive(Debug, Clone)]
pub struct Edge {
    pub from: String,
    pub to: String,
}

/// Return all import edges discovered during analysis.
pub fn all_import_edges() -> Vec<Edge> {
    Vec::new()
}
"#,
    )
    .unwrap();
}

fn build_index_in<P: EmbedProvider + 'static>(
    repo_root: &std::path::Path,
    index_dir: &std::path::Path,
    provider: P,
    dim: usize,
) {
    std::fs::create_dir_all(&index_dir).unwrap();

    let backend = Arc::new(
        CozoBackend::open(index_dir.join("index.db")).expect("open cozo backend"),
    );
    let manifest = Arc::new(
        ManifestStore::open(index_dir.join("manifest.db")).expect("open manifest"),
    );

    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    rt.block_on(async {
        backend.initialize(dim).await.expect("initialize backend");
        let indexer = Indexer::new(backend, manifest, provider);
        indexer.index_path(repo_root).await.expect("index fixture");
    });
}

fn build_index<P: EmbedProvider + 'static>(dir: &std::path::Path, provider: P, dim: usize) {
    build_index_in(dir, &dir.join(".skelesearch"), provider, dim)
}

// ---------------------------------------------------------------------------
// Fixture variants
// ---------------------------------------------------------------------------

/// Fast fixture using DeterministicTestProvider (no model download).
/// Suitable for status, context, clear, and watch tests.
fn indexed_cli_fixture() -> TempDir {
    let dir = tempfile::tempdir().expect("create temp dir");
    write_fixture_files(dir.path());
    // Dim 768 matches FastEmbedProvider so CLI binary opens schema without mismatch.
    build_index(dir.path(), DeterministicTestProvider::new(768), 768);
    dir
}

/// Fixture backed by the real embedding model.
/// Returns `None` when `SKIP_MODEL_DOWNLOAD` is set or when the model
/// cannot be initialised (no network access).
fn indexed_cli_fixture_with_model() -> Option<TempDir> {
    use skelesearch_core::EmbedProvider as _;
    use skelesearch_embed_fastembed::FastEmbedProvider;

    if std::env::var("SKIP_MODEL_DOWNLOAD").is_ok() {
        return None;
    }
    let provider = match FastEmbedProvider::default() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("FastEmbedProvider::default() failed (network?): {e}");
            return None;
        }
    };
    let dim = provider.dim();
    let dir = tempfile::tempdir().expect("create temp dir");
    write_fixture_files(dir.path());
    build_index(dir.path(), provider, dim);
    Some(dir)
}

fn mark_fixture_stale(repo_root: &std::path::Path) {
    let manifest_path = repo_root.join(".skelesearch").join("manifest.db");
    let manifest = ManifestStore::open(&manifest_path).expect("open manifest for stale fixture");
    manifest
        .upsert("src/deleted.rs", 1, 1, "fixture-hash")
        .expect("insert stale manifest row");
}

fn indexed_cli_generation_fixture() -> TempDir {
    let dir = tempfile::tempdir().expect("create temp dir");
    write_fixture_files(dir.path());
    let generation_dir = dir
        .path()
        .join(".skelesearch")
        .join("generations")
        .join("gen-1");
    build_index_in(
        dir.path(),
        &generation_dir,
        DeterministicTestProvider::new(768),
        768,
    );
    std::fs::write(dir.path().join(".skelesearch").join("active-generation"), "gen-1")
        .expect("write active generation pointer");
    dir
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn search_json_contains_required_result_fields() {
    // The search command embeds the query using FastEmbedProvider; skip when
    // the model is not available.
    let repo = match indexed_cli_fixture_with_model() {
        Some(r) => r,
        None => {
            eprintln!("search test skipped: embedding model unavailable");
            return;
        }
    };

    let output = assert_cmd::Command::cargo_bin("skelesearch")
        .unwrap()
        .current_dir(repo.path())
        .args(["search", "import edges", "--json"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "search failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let rows: Vec<serde_json::Value> =
        serde_json::from_slice(&output.stdout).expect("stdout is valid JSON");
    assert!(!rows.is_empty(), "expected at least one result for 'import edges'");

    let row = &rows[0];
    for key in ["file_path", "start_line", "end_line", "content", "score", "match_quality", "why"]
    {
        assert!(row.get(key).is_some(), "missing field: {key}");
    }
}

#[test]
fn status_json_contains_hook_facing_fields() {
    let repo = indexed_cli_fixture();
    let output = assert_cmd::Command::cargo_bin("skelesearch")
        .unwrap()
        .current_dir(repo.path())
        .args(["status", "--json"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "status failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let status: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("stdout is valid JSON");
    for key in [
        "indexed_files",
        "total_chunks",
        "last_indexed",
        "estimated_stale",
        "freshness_state",
        "freshness_checked_at",
        "freshness_error",
        "watching",
    ] {
        assert!(status.get(key).is_some(), "missing field: {key}");
    }
}

#[test]
fn status_json_reports_stale_freshness_for_missing_manifest_file() {
    let repo = indexed_cli_fixture();
    mark_fixture_stale(repo.path());

    let output = assert_cmd::Command::cargo_bin("skelesearch")
        .unwrap()
        .current_dir(repo.path())
        .args(["status", "--json"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "status failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let status: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("stdout is valid JSON");
    assert_eq!(status["freshness_state"], "stale");
    assert_eq!(status["estimated_stale"], 1);
    assert!(status["freshness_checked_at"].is_string());
    assert!(status["freshness_error"].is_null());
}

#[test]
fn status_json_reads_active_generation_index() {
    let repo = indexed_cli_generation_fixture();

    let output = assert_cmd::Command::cargo_bin("skelesearch")
        .unwrap()
        .current_dir(repo.path())
        .args(["status", "--json"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "status failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let status: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("stdout is valid JSON");
    assert_eq!(status["indexed_files"], 2);
    assert_eq!(status["freshness_state"], "fresh");
}

#[test]
fn context_command_prints_file_sections() {
    let repo = indexed_cli_fixture();
    assert_cmd::Command::cargo_bin("skelesearch")
        .unwrap()
        .current_dir(repo.path())
        .args(["context", "src/lib.rs"])
        .assert()
        .success()
        .stdout(
            predicates::str::contains("imports")
                .and(predicates::str::contains("imported_by")),
        );
}

#[test]
fn clear_command_removes_local_index() {
    let repo = indexed_cli_fixture();
    assert_cmd::Command::cargo_bin("skelesearch")
        .unwrap()
        .current_dir(repo.path())
        .args(["clear"])
        .assert()
        .success();

    let output = assert_cmd::Command::cargo_bin("skelesearch")
        .unwrap()
        .current_dir(repo.path())
        .args(["status", "--json"])
        .output()
        .unwrap();
    let status: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("stdout is valid JSON");
    assert_eq!(status["indexed_files"], 0, "indexed_files should be 0 after clear");
}

#[test]
fn index_rejects_unknown_provider() {
    let mut cmd = assert_cmd::Command::cargo_bin("skelesearch").unwrap();
    cmd.args(["index", ".", "--provider", "definitely-not-a-provider"])
        .assert()
        .failure();
}

#[test]
fn watch_command_sets_watching_state() {
    // Uses the fast DTP fixture (no model download).  The watch subprocess
    // will attempt provider init and fall back gracefully if model is absent,
    // writing its PID file before any slow initialisation.
    let repo = indexed_cli_fixture();

    let mut child = std::process::Command::new(assert_cmd::cargo::cargo_bin("skelesearch"))
        .current_dir(repo.path())
        .args(["watch", "."])
        .spawn()
        .unwrap();

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    let mut watching_and_idle = false;
    while std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(200));
        let output = assert_cmd::Command::cargo_bin("skelesearch")
            .unwrap()
            .current_dir(repo.path())
            .args(["status", "--json"])
            .output()
            .unwrap();
        if let Ok(json) = serde_json::from_slice::<serde_json::Value>(&output.stdout) {
            if json["watching"] == true && json["freshness_state"] != "refreshing" {
                watching_and_idle = true;
                break;
            }
        }
    }

    let _ = child.kill();
    let _ = child.wait();

    assert!(
        watching_and_idle,
        "watcher did not report watching=true with non-refreshing freshness_state within 5 seconds"
    );
}

#[test]
fn watch_is_a_subcommand_not_an_index_flag() {
    let mut cmd = assert_cmd::Command::cargo_bin("skelesearch").unwrap();
    cmd.args(["index", ".", "--watch"]).assert().failure();
}
