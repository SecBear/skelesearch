// skelesearch-mcp binary entry point.
//
// Logging MUST go to stderr only; stdout is reserved exclusively for
// JSON-RPC messages sent by rmcp.  Any log line written to stdout would
// corrupt the MCP framing and break the client.
//
// Transport: stdio (Claude Code / MCP-compatible)
// Framework: rmcp 0.16 (#[tool_router] + #[tool_handler])

use std::sync::Arc;

use anyhow::Context as _;
use async_trait::async_trait;
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
    let backend_dir = tempfile::tempdir().context("create backend temp dir")?;
    let manifest_dir = tempfile::tempdir().context("create manifest temp dir")?;

    let backend = Arc::new(
        skelesearch_core::CozoBackend::open(backend_dir.path().join("index.db"))
            .context("open CozoBackend")?,
    );
    let manifest_path = manifest_dir.path().join("manifest.db");

    let server = SkeleSearchServer::new(backend, manifest_path, NoopProvider);

    tracing::info!("skelesearch-mcp starting on stdio");

    // Keep temp dirs alive until the server exits.
    let _backend_dir = backend_dir;
    let _manifest_dir = manifest_dir;

    server.serve_stdio().await
}
