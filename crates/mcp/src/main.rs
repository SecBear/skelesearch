// skelesearch-mcp binary entry point.
//
// Logging MUST go to stderr only; stdout is reserved exclusively for
// JSON-RPC messages sent by rmcp.  Any log line written to stdout would
// corrupt the MCP framing and break the client.
//
// Transport: stdio (default) or Streamable HTTP (--http <addr>)
// Framework: rmcp 1.2 (#[tool_router] + #[tool_handler])

use std::{path::PathBuf, sync::Arc};

use anyhow::Context as _;
use async_trait::async_trait;
use clap::Parser;
use skelesearch_core::EmbedProvider;
use skelesearch_mcp::server::SkeleSearchServer;

// ---------------------------------------------------------------------------
// Startup provider
//
// Loading FastEmbedProvider (ONNX model) takes seconds; the stdio server must
// respond to the MCP `initialize` handshake quickly.  The binary uses a
// zero-cost noop provider so it starts instantly and responds to protocol
// messages without any model loading.  Callers can trigger real indexing via
// the `index_codebase` tool which creates its own provider.
// ---------------------------------------------------------------------------

struct NoopProvider;

#[async_trait]
impl EmbedProvider for NoopProvider {
    fn dim(&self) -> usize {
        1
    }

    async fn embed_batch(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
        // Unit-length 1-dim vectors: safe to pass to cosine similarity.
        Ok(texts.iter().map(|_| vec![1.0_f32]).collect())
    }
}

#[derive(Parser)]
#[command(name = "skelesearch-mcp", version)]
struct Args {
    /// Listen on HTTP (Streamable HTTP transport) instead of stdio.
    /// When set, the server binds to this address (e.g., 127.0.0.1:3000).
    #[arg(long)]
    http: Option<String>,
}

fn main() {
    // Initialise tracing to stderr. When OTEL_EXPORTER_OTLP_ENDPOINT is set
    // and the `otlp` feature is enabled on skelesearch-telemetry, spans are
    // also exported to the configured OTLP collector.
    let _telemetry = skelesearch_telemetry::init_tracing("skelesearch-mcp", "info");

    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    if let Err(e) = rt.block_on(async_main()) {
        eprintln!("skelesearch-mcp: fatal: {e:#}");
        std::process::exit(1);
    }
}

async fn async_main() -> anyhow::Result<()> {
    let args = Args::parse();

    let project_root = find_project_root();
    let skelesearch_dir = resolve_storage_dir(&project_root);

    let backend = Arc::new(
        skelesearch_core::CozoBackend::open(skelesearch_dir.join("index.db"))
            .context("open CozoBackend")?,
    );
    let manifest_path = skelesearch_dir.join("manifest.db");

    let server = SkeleSearchServer::new(backend, manifest_path, NoopProvider);

    if let Some(addr_str) = args.http {
        let addr: std::net::SocketAddr = addr_str.parse()
            .context("invalid --http address")?;
        tracing::info!(
            "skelesearch-mcp starting on HTTP (storage: {})",
            skelesearch_dir.display()
        );
        server.serve_http(addr).await
    } else {
        tracing::info!(
            "skelesearch-mcp starting on stdio (storage: {})",
            skelesearch_dir.display()
        );
        server.serve_stdio().await
    }
}

/// Walk up from cwd until we find a directory containing `.git`, then return
/// that directory.  Falls back to cwd if no `.git` ancestor is found.
fn find_project_root() -> PathBuf {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut dir = cwd.clone();
    loop {
        if dir.join(".git").exists() {
            return dir;
        }
        match dir.parent() {
            Some(parent) => dir = parent.to_path_buf(),
            None => return cwd,
        }
    }
}

/// Determine the `.skelesearch` storage directory. Tries the project root first;
/// falls back to `$HOME/.skelesearch/<hash>` if the primary path is not writable,
/// and finally to a temp directory. The server MUST always reach `serve_stdio()`
/// — a degraded storage directory is better than a dead process.
fn resolve_storage_dir(project_root: &std::path::Path) -> PathBuf {
    let primary = project_root.join(".skelesearch");
    if std::fs::create_dir_all(&primary).is_ok() {
        return primary;
    }
    tracing::warn!(
        path = %primary.display(),
        "cannot create .skelesearch in project root — falling back"
    );

    // Deterministic fallback under $HOME so repeated launches reuse the same DB.
    if let Ok(home) = std::env::var("HOME") {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        project_root.hash(&mut hasher);
        let hash = hasher.finish();
        let fallback = PathBuf::from(home)
            .join(".skelesearch")
            .join(format!("fallback-{hash:016x}"));
        if std::fs::create_dir_all(&fallback).is_ok() {
            tracing::info!(path = %fallback.display(), "using fallback storage dir");
            return fallback;
        }
    }

    // Last resort: temp directory (ephemeral, but the server starts).
    let tmp = std::env::temp_dir().join(".skelesearch-tmp");
    let _ = std::fs::create_dir_all(&tmp);
    tracing::warn!(path = %tmp.display(), "using temp storage dir — index will not persist");
    tmp
}
