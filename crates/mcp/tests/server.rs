// Integration tests for the skelesearch MCP server.
//
// All tests are network-independent: they use DeterministicTestProvider instead
// of FastEmbedProvider so the model download step is never needed.  The vector
// search still works because deterministic (non-zero) unit vectors give non-zero
// cosine similarity, producing real result rows from the hybrid search.
//
// `test_server()` creates a SkeleSearchServer pre-indexed with the fixture repo.
// `run_mcp_exchange` spawns the real binary and verifies stdio discipline.

use std::{
    io::Write as _,
    path::PathBuf,
    process::{Command, Stdio},
    sync::Arc,
};

use async_trait::async_trait;
use skelesearch_core::{CozoBackend, EmbedProvider, Indexer, ManifestStore};
use skelesearch_mcp::{
    server::{ArcProvider, SkeleSearchServer},
    tools::{GetFileContextInput, IndexCodebaseInput, IndexStatusInput, SearchCodeInput},
};
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// DeterministicTestProvider
//
// Produces slightly-varied unit vectors without any model loading.  Sufficient
// for exercising the full index + search pipeline in tests.
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct DeterministicTestProvider {
    dim: usize,
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

fn det_provider() -> ArcProvider {
    ArcProvider::new(DeterministicTestProvider { dim: 16 })
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/// Path to the fixture repo used by core tests.
fn fixture_repo_path() -> anyhow::Result<PathBuf> {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let path = base
        .parent() // crates/
        .and_then(|p| p.parent()) // workspace root
        .map(|root| root.join("crates/core/tests/fixtures/sample_repo"))
        .ok_or_else(|| anyhow::anyhow!("could not resolve fixture repo path"))?;
    if !path.exists() {
        anyhow::bail!("fixture repo not found at {}", path.display());
    }
    Ok(path)
}

/// Create a pre-indexed `SkeleSearchServer` backed by temp databases.
///
/// Uses `DeterministicTestProvider` so no network access is needed.
/// Temp dirs are leaked; the test process cleans them up on exit.
async fn test_server() -> anyhow::Result<SkeleSearchServer> {
    let backend_dir = TempDir::new()?;
    let manifest_dir = TempDir::new()?;

    let backend_path = backend_dir.path().join("index.db");
    let manifest_path = manifest_dir.path().join("manifest.db");

    let backend = Arc::new(CozoBackend::open(&backend_path)?);
    let provider = det_provider();

    // Pre-index the fixture so `search_code` tests have real results.
    {
        let manifest = Arc::new(ManifestStore::open(&manifest_path)?);
        let indexer = Indexer::new(Arc::clone(&backend), manifest, provider.clone());
        indexer.index_path(&fixture_repo_path()?).await?;
    }

    let server = SkeleSearchServer::new(backend, &manifest_path, provider);

    // Leak temp dirs — cleaned when the test process exits.
    std::mem::forget(backend_dir);
    std::mem::forget(manifest_dir);

    Ok(server)
}

// ---------------------------------------------------------------------------
// JSON-RPC helpers for the stdio smoke test
// ---------------------------------------------------------------------------

struct McpTranscript {
    stdout: String,
    stderr: String,
}

/// Encode a JSON-RPC message for rmcp's newline-delimited stdio transport.
fn encode_msg(v: serde_json::Value) -> String {
    let mut s = v.to_string();
    s.push('\n');
    s
}

fn initialize_request() -> String {
    encode_msg(serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": { "name": "skelesearch-test", "version": "0.0.0" }
        }
    }))
}

fn initialized_notification() -> String {
    encode_msg(serde_json::json!({
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
        "params": {}
    }))
}

fn list_tools_request() -> String {
    encode_msg(serde_json::json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {}
    }))
}

/// Spawn the `skelesearch-mcp` binary, write `messages` to its stdin, close
/// stdin (EOF → server exits), and collect all stdout/stderr output.
fn run_mcp_exchange(messages: &[String]) -> anyhow::Result<McpTranscript> {
    let bin = env!("CARGO_BIN_EXE_skelesearch-mcp");

    let mut child = Command::new(bin)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| anyhow::anyhow!("failed to spawn {bin}: {e}"))?;

    // Write all messages then drop stdin → EOF.
    {
        let mut stdin = child.stdin.take().expect("stdin not captured");
        for msg in messages {
            stdin.write_all(msg.as_bytes())?;
        }
        // drop → EOF
    }

    let output = child
        .wait_with_output()
        .map_err(|e| anyhow::anyhow!("wait_with_output failed: {e}"))?;

    Ok(McpTranscript {
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn list_tools_exposes_the_v1_tools() -> anyhow::Result<()> {
    let server = test_server().await?;
    let names = server.tool_names().await?;
    assert_eq!(
        names,
        vec![
            "find_symbol",
            "get_file_context",
            "index_codebase",
            "index_status",
            "search_code",
            "smart_search",
        ]
    );
    Ok(())
}

#[tokio::test]
async fn search_code_output_exposes_spec_fields() -> anyhow::Result<()> {
    let server = test_server().await?;
    let rows = server
        .search_code(SearchCodeInput {
            query: "import edges".into(),
            top_k: 3,
            include_graph: true,
            max_depth: None,
            diversity: 0.0,
            max_tokens: None,
            branch_scope: false,
        })
        .await?;
    assert!(
        !rows.is_empty(),
        "expected at least one result from pre-indexed fixture"
    );
    let row = &rows[0];
    assert!(!row.file_path.is_empty());
    assert!(row.end_line >= row.start_line);
    assert!(!row.content.is_empty());
    assert!(row.score > 0.0);
    assert!(!row.match_quality.is_empty());
    assert!(!row.why.is_empty());
    Ok(())
}

#[tokio::test]
async fn get_file_context_returns_empty_arrays_for_unknown_file() -> anyhow::Result<()> {
    let ctx = test_server()
        .await?
        .get_file_context(GetFileContextInput {
            file_path: "missing.rs".into(),
        })
        .await?;
    assert!(
        ctx.chunks.is_empty() && ctx.imports.is_empty() && ctx.imported_by.is_empty(),
        "expected empty arrays for unknown file"
    );
    Ok(())
}

#[tokio::test]
async fn index_codebase_returns_status_indexed_and_chunk_counts() -> anyhow::Result<()> {
    // Fresh server (not pre-indexed) so `indexed > 0` on first run.
    let backend_dir = TempDir::new()?;
    let manifest_dir = TempDir::new()?;
    let backend = Arc::new(CozoBackend::open(backend_dir.path().join("index.db"))?);
    let manifest_path = manifest_dir.path().join("manifest.db");
    let server = SkeleSearchServer::new(backend, &manifest_path, det_provider());
    std::mem::forget(backend_dir);
    std::mem::forget(manifest_dir);

    // Use run_index so the test is network-independent (bypasses provider factory).
    let out = server
        .run_index(&fixture_repo_path()?, det_provider())
        .await?;
    assert!(!out.status.is_empty());
    assert!(out.indexed > 0, "expected indexed > 0 on first run, got {}", out.indexed);
    assert!(out.chunks > 0, "expected chunks > 0, got {}", out.chunks);
    Ok(())
}

#[tokio::test]
async fn index_codebase_rejects_unknown_provider() -> anyhow::Result<()> {
    let err = test_server()
        .await?
        .index_codebase(IndexCodebaseInput {
            path: fixture_repo_path()?.display().to_string(),
            provider: Some("definitely-not-a-provider".into()),
        })
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("provider"),
        "error message should mention 'provider', got: {err}"
    );
    Ok(())
}

#[tokio::test]
async fn index_status_exposes_estimated_stale_and_watching() -> anyhow::Result<()> {
    // test_server() pre-indexed with DeterministicTestProvider; last_indexed should be set.
    let server = test_server().await?;
    let status = server.index_status(IndexStatusInput { path: None }).await?;
    assert_eq!(status.estimated_stale, 0);
    assert!(!status.watching);
    assert!(
        status
            .last_indexed
            .as_ref()
            .map(|s| chrono::DateTime::parse_from_rfc3339(s).is_ok())
            .unwrap_or(false),
        "last_indexed should be RFC 3339, got: {:?}",
        status.last_indexed
    );
    Ok(())
}

/// Smoke test: the binary speaks JSON-RPC on stdout and logs only to stderr.
///
/// rmcp's stdio transport uses newline-delimited JSON (one object per line).
#[test]
fn server_stdio_speaks_json_rpc_without_stdout_logs() -> anyhow::Result<()> {
    let transcript = run_mcp_exchange(&[
        initialize_request(),
        initialized_notification(),
        list_tools_request(),
    ])?;

    assert!(
        !transcript.stdout.is_empty(),
        "stdout should contain JSON-RPC responses.\nstderr: {}",
        transcript.stderr
    );

    // Every non-empty stdout line must be a JSON object.
    for line in transcript.stdout.lines() {
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            assert!(
                trimmed.starts_with('{'),
                "non-JSON line on stdout (log pollution?): {:?}",
                &trimmed[..trimmed.len().min(120)]
            );
        }
    }

    // Logs go to stderr only — no JSON-RPC messages on stderr.
    for line in transcript.stderr.lines() {
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            assert!(
                !trimmed.starts_with('{'),
                "JSON-RPC message found on stderr (stdout/stderr discipline broken): {:?}",
                &trimmed[..trimmed.len().min(120)]
            );
        }
    }

    Ok(())
}
