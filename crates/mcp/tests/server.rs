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
    sync::{Arc, Mutex as StdMutex, OnceLock},
};

use async_trait::async_trait;
use skelesearch_core::{
    try_acquire_indexing_lease, CompositeBackend, EmbedProvider, Indexer, ManifestStore,
    SharedIndexingStatus,
};
use skelesearch_mcp::{
    server::{ArcProvider, SkeleSearchServer},
    tools::{
        GetFileContextInput, GetRepoMapInput, GetSymbolContextInput, IndexCodebaseInput,
        IndexFreshnessState, IndexStatusInput, SearchCodeInput, SmartSearchInput,
    },
};
use skelesearch_service::{
    DaemonCapabilities, DaemonRequest, DaemonResponse, HandshakeResponse, ProjectKey,
    ProtocolFrame, SearchCodeResponse as DaemonSearchCodeResponse,
    SearchResultRow as DaemonSearchResultRow,
};
use tempfile::TempDir;
use tokio::{
    io::{AsyncBufReadExt as _, AsyncWriteExt as _, BufReader},
    net::UnixListener,
};

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

fn env_lock() -> &'static StdMutex<()> {
    static LOCK: OnceLock<StdMutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| StdMutex::new(()))
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

#[cfg(unix)]
async fn spawn_stub_search_daemon(
    socket_path: &std::path::Path,
) -> anyhow::Result<tokio::task::JoinHandle<anyhow::Result<()>>> {
    let listener = UnixListener::bind(socket_path)?;
    Ok(tokio::spawn(async move {
        let (stream, _) = listener.accept().await?;
        let (read_half, mut write_half) = stream.into_split();
        let mut lines = BufReader::new(read_half).lines();
        while let Some(line) = lines.next_line().await? {
            let frame: ProtocolFrame = serde_json::from_str(&line)?;
            let response = match frame {
                ProtocolFrame::Request { id, request } => match request {
                    DaemonRequest::Handshake(_) => ProtocolFrame::Response {
                        id,
                        response: DaemonResponse::Handshake(HandshakeResponse {
                            protocol_version: skelesearch_service::DAEMON_PROTOCOL_VERSION
                                .to_string(),
                            server_name: "stub-daemon".to_string(),
                            server_version: "0.1.0".to_string(),
                            capabilities: DaemonCapabilities {
                                info: true,
                                index_codebase: true,
                                index_status: true,
                                search_code: true,
                                smart_search: false,
                                register_client: true,
                                heartbeat: true,
                                unregister_client: true,
                            },
                        }),
                    },
                    DaemonRequest::SearchCode(_) => ProtocolFrame::Response {
                        id,
                        response: DaemonResponse::SearchCode(DaemonSearchCodeResponse {
                            project_key: ProjectKey {
                                canonical_root: "/tmp/repo".to_string(),
                                logical_id: None,
                            },
                            results: vec![DaemonSearchResultRow {
                                file_path: "src/searcher.rs".to_string(),
                                start_line: 10,
                                end_line: 20,
                                content: "fn search() {}".to_string(),
                                score: 0.9,
                                match_quality: "high".to_string(),
                                why: "semantic".to_string(),
                            }],
                        }),
                    },
                    other => ProtocolFrame::Response {
                        id,
                        response: DaemonResponse::Error(skelesearch_service::DaemonErrorResponse {
                            code: skelesearch_service::DaemonErrorCode::BadRequest,
                            message: format!("unexpected request in stub: {other:?}"),
                            details: None,
                            retryable: false,
                        }),
                    },
                },
                _ => continue,
            };
            let encoded = serde_json::to_string(&response)?;
            write_half.write_all(encoded.as_bytes()).await?;
            write_half.write_all(b"\n").await?;
            write_half.flush().await?;
        }
        Ok(())
    }))
}

/// Create a pre-indexed `SkeleSearchServer` backed by temp databases.
///
/// Uses `DeterministicTestProvider` so no network access is needed.
/// Temp dirs are leaked; the test process cleans them up on exit.
async fn test_server() -> anyhow::Result<SkeleSearchServer> {
    let project_dir = TempDir::new()?;
    copy_dir_all(&fixture_repo_path()?, project_dir.path())?;
    let storage_dir = project_dir.path().join(".skelesearch");
    std::fs::create_dir_all(&storage_dir)?;

    let manifest_path = storage_dir.join("manifest.db");

    let backend = Arc::new(CompositeBackend::open(&storage_dir).await?);
    let provider = det_provider();

    {
        let manifest = Arc::new(ManifestStore::open(&manifest_path)?);
        let indexer = Indexer::new(Arc::clone(&backend), manifest, provider.clone());
        indexer.index_path(project_dir.path()).await?;
    }

    let server = SkeleSearchServer::new(backend, &manifest_path, provider);

    std::mem::forget(project_dir);

    Ok(server)
}

async fn test_server_with_manifest_path() -> anyhow::Result<(SkeleSearchServer, PathBuf)> {
    let project_dir = TempDir::new()?;
    copy_dir_all(&fixture_repo_path()?, project_dir.path())?;
    let storage_dir = project_dir.path().join(".skelesearch");
    std::fs::create_dir_all(&storage_dir)?;

    let manifest_path = storage_dir.join("manifest.db");

    let backend = Arc::new(CompositeBackend::open(&storage_dir).await?);
    let provider = det_provider();

    {
        let manifest = Arc::new(ManifestStore::open(&manifest_path)?);
        let indexer = Indexer::new(Arc::clone(&backend), manifest, provider.clone());
        indexer.index_path(project_dir.path()).await?;
    }

    let server = SkeleSearchServer::new(backend, &manifest_path, provider);

    std::mem::forget(project_dir);

    Ok((server, manifest_path))
}

fn mark_manifest_stale(manifest_path: &std::path::Path) {
    let manifest = ManifestStore::open(manifest_path).expect("open manifest");
    manifest
        .upsert("src/deleted.rs", 1, 1, "fixture-hash")
        .expect("insert stale manifest row");
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
            "find_dependents",
            "find_symbol",
            "find_tests",
            "get_index_status",
            "get_repo_map",
            "get_symbol_info",
            "index",
            "search_code",
        ]
    );
    Ok(())
}

#[tokio::test]
async fn search_code_output_exposes_spec_fields() -> anyhow::Result<()> {
    #[cfg(unix)]
    {
        let _guard = env_lock().lock().expect("env lock");
        let temp = TempDir::new()?;
        let socket_path = temp.path().join("daemon.sock");
        let daemon = spawn_stub_search_daemon(&socket_path).await?;
        std::env::set_var("SKELESEARCH_DAEMON_SOCKET", &socket_path);

        let server = test_server().await?;
        let response = server
            .search_code(SearchCodeInput {
                query: "import edges".into(),
                top_k: 3,
                include_graph: true,
                max_depth: None,
                diversity: 0.0,
                max_tokens: None,
                branch_scope: false,
                session_id: None,
            })
            .await?;
        assert!(
            !response.results.is_empty(),
            "expected at least one result from daemon-backed search"
        );
        let row = &response.results[0];
        assert!(!row.file_path.is_empty());
        assert!(row.end_line >= row.start_line);
        assert!(!row.content.is_empty());
        assert!(row.score > 0.0);
        assert!(!row.match_quality.is_empty());
        assert!(!row.why.is_empty());

        std::env::remove_var("SKELESEARCH_DAEMON_SOCKET");
        daemon.abort();
        return Ok(());
    }

    #[cfg(not(unix))]
    {
        Ok(())
    }
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
    // Fresh server (not pre-indexed) so `indexed_files > 0` on first run.
    let backend_dir = TempDir::new()?;
    let manifest_dir = TempDir::new()?;
    let backend = Arc::new(CompositeBackend::open(backend_dir.path()).await?);
    let manifest_path = manifest_dir.path().join("manifest.db");
    let server = SkeleSearchServer::new(backend, &manifest_path, det_provider());
    std::mem::forget(backend_dir);
    std::mem::forget(manifest_dir);

    // Use run_index so the test is network-independent (bypasses provider factory).
    // run_index returns IndexResult directly after synchronous foreground indexing.
    let out = server
        .run_index(&fixture_repo_path()?, det_provider())
        .await?;
    assert!(
        out.indexed_files > 0,
        "expected indexed_files > 0 on first run, got {}",
        out.indexed_files
    );
    assert!(
        out.total_chunks > 0,
        "expected total_chunks > 0, got {}",
        out.total_chunks
    );
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
    let server = test_server().await?;
    let status = server.index_status(IndexStatusInput { path: None }).await?;
    assert_eq!(status.estimated_stale, 0);
    assert_eq!(status.freshness_state, IndexFreshnessState::Fresh);
    assert!(status.freshness_checked_at.is_some());
    assert_eq!(status.freshness_error, None);
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

#[tokio::test]
async fn index_status_reports_stale_freshness_for_missing_manifest_file() -> anyhow::Result<()> {
    let (server, manifest_path) = test_server_with_manifest_path().await?;
    mark_manifest_stale(&manifest_path);

    let status = server.index_status(IndexStatusInput { path: None }).await?;
    assert!(status.estimated_stale > 0);
    assert_eq!(status.freshness_state, IndexFreshnessState::Stale);
    assert!(status.freshness_checked_at.is_some());
    assert_eq!(status.freshness_error, None);

    Ok(())
}

#[tokio::test]
async fn get_repo_map_prepends_warning_for_unknown_freshness() -> anyhow::Result<()> {
    let (server, manifest_path) = test_server_with_manifest_path().await?;

    let _ = std::fs::remove_file(&manifest_path);
    std::fs::create_dir_all(&manifest_path)?;

    let repo_map = server
        .get_repo_map(GetRepoMapInput {
            max_tokens: 4096,
            include_symbols: true,
            include_edges: true,
            project: None,
        })
        .await?;
    assert!(
        repo_map.starts_with("⚠ Index freshness is unknown"),
        "expected unknown freshness warning, got:\n{repo_map}"
    );

    Ok(())
}

#[tokio::test]
async fn get_repo_map_prepends_warning_for_stale_freshness() -> anyhow::Result<()> {
    let (server, manifest_path) = test_server_with_manifest_path().await?;
    mark_manifest_stale(&manifest_path);

    let repo_map = server
        .get_repo_map(GetRepoMapInput {
            max_tokens: 4096,
            include_symbols: true,
            include_edges: true,
            project: None,
        })
        .await?;
    assert!(
        repo_map.starts_with("⚠ ") && repo_map.contains("file(s) changed since last index"),
        "expected stale freshness warning, got:\n{repo_map}"
    );

    Ok(())
}

#[tokio::test]
async fn get_repo_map_prepends_warning_for_refreshing_freshness() -> anyhow::Result<()> {
    let (server, manifest_path) = test_server_with_manifest_path().await?;
    let storage_dir = manifest_path
        .parent()
        .expect("manifest has storage dir")
        .to_path_buf();

    let now = chrono::Utc::now();
    let status = SharedIndexingStatus {
        instance_id: "test-instance".to_string(),
        pid: std::process::id(),
        path: fixture_repo_path()?.display().to_string(),
        provider: "fastembed".to_string(),
        trigger: "test".to_string(),
        status: "running".to_string(),
        started_at: now,
        updated_at: now,
        files_total: 10,
        files_done: 1,
        chunks_done: 2,
        cache_hits: 0,
        error: None,
    };
    let _lease = try_acquire_indexing_lease(&storage_dir, &status)?
        .expect("acquire lease for refreshing repo-map test");

    let repo_map = server
        .get_repo_map(GetRepoMapInput {
            max_tokens: 4096,
            include_symbols: true,
            include_edges: true,
            project: None,
        })
        .await?;
    assert!(
        repo_map.starts_with("⚠ Index refresh is in progress"),
        "expected refreshing warning, got:\n{repo_map}"
    );

    Ok(())
}

#[tokio::test]
async fn smart_search_exact_symbol_prefers_grep() -> anyhow::Result<()> {
    let server = test_server().await?;
    let out = server
        .smart_search(SmartSearchInput {
            query: "OldStruct".into(),
            top_k: 3,
            include_graph: false,
            diversity: 0.0,
            max_tokens: Some(1024),
            branch_scope: false,
            session_id: None,
            intent: None,
            symbols: vec![],
            scope: None,
            project: None,
        })
        .await?;
    assert_eq!(out.strategy, "grep");
    match out.results {
        skelesearch_mcp::tools::SmartSearchResults::Grep(rows) => {
            assert!(
                !rows.is_empty(),
                "expected grep rows for exact symbol query"
            );
            assert!(rows.iter().any(
                |r| r.file_path.ends_with("src/old.rs") || r.file_path.ends_with("src/lib.rs")
            ));
        }
        other => panic!("expected grep results, got {other:?}"),
    }
    Ok(())
}

#[tokio::test]
async fn get_symbol_context_returns_role_and_context() -> anyhow::Result<()> {
    let server = test_server().await?;
    let ctx = server
        .get_symbol_context(GetSymbolContextInput {
            name: "helper".into(),
            kind: Some("function".into()),
            include_tests: true,
            project: None,
        })
        .await?;
    assert!(ctx.symbol.is_some(), "expected symbol match");
    assert_eq!(ctx.match_count, 1);
    assert!(!ctx.ambiguous);
    assert!(ctx
        .source
        .as_deref()
        .unwrap_or("")
        .contains("pub fn helper"));
    assert!(ctx.role.is_some(), "expected non-null role");
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
