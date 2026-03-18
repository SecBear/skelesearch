// skelesearch-mcp binary entry point.
//
// Logging MUST go to stderr only; stdout is reserved exclusively for
// JSON-RPC messages sent by rmcp.  Any log line written to stdout would
// corrupt the MCP framing and break the client.
//
// Transport: stdio (default) or Streamable HTTP (--http <addr>)
// Framework: rmcp 0.16 (#[tool_router] + #[tool_handler])

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
    // Initialise tracing to stderr before touching anything else.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    if let Err(e) = rt.block_on(async_main()) {
        eprintln!("skelesearch-mcp: fatal: {e:#}");
        std::process::exit(1);
    }
}

async fn async_main() -> anyhow::Result<()> {
    let args = Args::parse();

    let project_root = find_project_root();
    let skelesearch_dir = project_root.join(".skelesearch");
    std::fs::create_dir_all(&skelesearch_dir)
        .with_context(|| format!("create .skelesearch dir at {}", skelesearch_dir.display()))?;

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
