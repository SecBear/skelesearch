use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Context as _;
use chrono::{DateTime, Utc};
use serde_json::json;
use skelesearch_core::{
    CozoBackend, EmbedProvider, IndexStats, Indexer, ManifestStore, Searcher, StorageBackend,
};
use skelesearch_embed_fastembed::FastEmbedProvider;

use crate::cli::{Cli, Commands};

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub async fn run(cli: Cli) -> anyhow::Result<()> {
    match cli.command {
        Commands::Index { path, provider } => run_index(path, provider).await,
        Commands::Search { query, top_k, graph, json } => {
            run_search(query, top_k as usize, graph, json).await
        }
        Commands::Context { file } => run_context(file).await,
        Commands::Status { path, json } => run_status(path, json).await,
        Commands::Clear { path } => run_clear(path).await,
        Commands::Watch { path, provider } => run_watch(path, provider).await,
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// `.skelesearch/` directory inside `root`.
fn index_dir(root: &Path) -> PathBuf {
    root.join(".skelesearch")
}

/// Resolve the project root: explicit `path` argument or current directory.
fn resolve_root(path: Option<PathBuf>) -> anyhow::Result<PathBuf> {
    match path {
        Some(p) => Ok(p),
        None => std::env::current_dir().context("failed to get current directory"),
    }
}

/// Validate and build a FastEmbedProvider for the given provider name.
///
/// Only `"fastembed"` is supported in v1.  Unknown names are rejected with a
/// clear error so callers get a useful message rather than a panic.
fn make_provider(name: &str) -> anyhow::Result<FastEmbedProvider> {
    match name {
        "fastembed" => {
            FastEmbedProvider::default().context("failed to initialise fastembed provider")
        }
        other => anyhow::bail!(
            "unknown embedding provider: '{}'. Supported providers: fastembed",
            other
        ),
    }
}

/// Open the CozoBackend at `dir/index.db`.
fn open_backend(dir: &Path) -> anyhow::Result<Arc<CozoBackend>> {
    let db_path = dir.join("index.db");
    let backend = CozoBackend::open(&db_path)
        .with_context(|| format!("failed to open index at {}", db_path.display()))?;
    Ok(Arc::new(backend))
}

/// Open the ManifestStore at `dir/manifest.db`.
fn open_manifest(dir: &Path) -> anyhow::Result<Arc<ManifestStore>> {
    let db_path = dir.join("manifest.db");
    let store = ManifestStore::open(&db_path)
        .with_context(|| format!("failed to open manifest at {}", db_path.display()))?;
    Ok(Arc::new(store))
}

// ---------------------------------------------------------------------------
// Watch sentinel: PID-file mechanism
// ---------------------------------------------------------------------------
//
// The `watch` command writes its PID to `.skelesearch/watch.pid`.
// `status --json` checks whether that file exists and the recorded PID is
// still a live process before reporting `watching: true`.
//
// If the watch process crashes, the stale PID file is detected via `kill -0`
// and ignored.

fn pid_file(dir: &Path) -> PathBuf {
    dir.join("watch.pid")
}

/// Check whether a watch process is currently running for this index dir.
fn is_process_watching(dir: &Path) -> bool {
    let pf = pid_file(dir);
    let pid_str = match std::fs::read_to_string(&pf) {
        Ok(s) => s,
        Err(_) => return false,
    };
    let pid: u32 = match pid_str.trim().parse() {
        Ok(p) => p,
        Err(_) => return false,
    };
    process_is_alive(pid)
}

/// Returns `true` if the given PID corresponds to a running process.
#[cfg(unix)]
fn process_is_alive(pid: u32) -> bool {
    // `kill -0 <pid>` succeeds if the process exists and we have permission.
    std::process::Command::new("kill")
        .args(["-0", &pid.to_string()])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[cfg(not(unix))]
fn process_is_alive(_pid: u32) -> bool {
    // Windows: conservative — assume alive if file is present.
    true
}

// ---------------------------------------------------------------------------
// Command handlers
// ---------------------------------------------------------------------------

async fn run_index(path: PathBuf, provider_name: String) -> anyhow::Result<()> {
    // Validate provider before doing any work.
    let provider = make_provider(&provider_name)?;
    let dim = provider.dim();

    let root = std::fs::canonicalize(&path)
        .with_context(|| format!("cannot access path: {}", path.display()))?;
    let dir = index_dir(&root);
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("failed to create index dir: {}", dir.display()))?;

    let backend = open_backend(&dir)?;
    let manifest = open_manifest(&dir)?;

    backend.initialize(dim).await?;

    let indexer = Indexer::new(backend, manifest, provider);
    let result = indexer.index_path(&root).await?;

    println!(
        "indexed {} file(s), {} chunk(s) ({} removed)",
        result.indexed_files, result.total_chunks, result.deleted_files
    );
    Ok(())
}

async fn run_search(
    query: String,
    top_k: usize,
    graph: bool,
    json_output: bool,
) -> anyhow::Result<()> {
    let root = std::env::current_dir().context("failed to get current directory")?;
    let dir = index_dir(&root);

    // Short-circuit: no index at all.
    if !dir.join("index.db").exists() {
        if json_output {
            println!("[]");
        } else {
            println!("No index found. Run `skelesearch index <path>` first.");
        }
        return Ok(());
    }

    let provider = make_provider("fastembed")?;
    let dim = provider.dim();
    let backend = open_backend(&dir)?;
    backend.initialize(dim).await?;

    let searcher = Searcher::new(backend, provider);
    let results = searcher.search(&query, top_k, graph).await.unwrap_or_default();

    if json_output {
        let rows: Vec<serde_json::Value> = results
            .iter()
            .map(|r| {
                json!({
                    "file_path":    r.file_path,
                    "start_line":   r.start_line,
                    "end_line":     r.end_line,
                    "content":      r.content,
                    "score":        r.score,
                    "match_quality": r.match_quality,
                    "why":          r.why,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&rows)?);
    } else {
        if results.is_empty() {
            println!("No results.");
        }
        for (i, r) in results.iter().enumerate() {
            println!(
                "[{}] {} (lines {}-{}) — {} score={:.4}",
                i + 1,
                r.file_path,
                r.start_line,
                r.end_line,
                r.match_quality,
                r.score
            );
            println!("{}", r.content);
            println!();
        }
    }
    Ok(())
}

async fn run_context(file: PathBuf) -> anyhow::Result<()> {
    let root = std::env::current_dir().context("failed to get current directory")?;
    let dir = index_dir(&root);

    // Normalise the file argument to a relative path string the backend can
    // look up.  If the path is absolute and lives under the project root,
    // strip the root prefix; otherwise use the path as supplied.
    let file_key = if file.is_absolute() {
        file.strip_prefix(&root)
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|_| file.to_string_lossy().to_string())
    } else {
        file.to_string_lossy().to_string()
    };

    if !dir.join("index.db").exists() {
        println!("No index found. Run `skelesearch index <path>` first.");
        return Ok(());
    }

    // Context lookup is read-only; we use a minimal no-op provider to satisfy
    // the Searcher generic without downloading a model.
    let backend = open_backend(&dir)?;
    // We need to call initialize so relations exist; borrow the stored dim
    // from the provider.  Since context never calls embed, dim doesn't matter.
    // Use a no-op shim — see NoopProvider below.
    let searcher = Searcher::new(backend, NoopProvider);
    let ctx = searcher.file_context(&file_key).await?;

    println!("=== chunks ({}) ===", ctx.chunks.len());
    for c in &ctx.chunks {
        println!(
            "  [{}] lines {}-{}: {}",
            c.chunk_idx,
            c.start_line,
            c.end_line,
            c.content.lines().next().unwrap_or("").trim()
        );
    }

    println!("\n=== imports ===");
    if ctx.imports.is_empty() {
        println!("  (none)");
    } else {
        for imp in &ctx.imports {
            println!("  {imp}");
        }
    }

    println!("\n=== imported_by ===");
    if ctx.imported_by.is_empty() {
        println!("  (none)");
    } else {
        for by in &ctx.imported_by {
            println!("  {by}");
        }
    }

    Ok(())
}

async fn run_status(path: Option<PathBuf>, json_output: bool) -> anyhow::Result<()> {
    let root = resolve_root(path)?;
    let dir = index_dir(&root);

    let stats = if dir.join("index.db").exists() {
        let backend = open_backend(&dir)?;
        // stats() queries `files` and `chunks` relations.  On an uninitialised
        // db those don't exist yet — return zeros rather than propagating the
        // error, since that situation is equivalent to an empty index.
        backend.stats().await.unwrap_or_else(|_| IndexStats {
            indexed_files: 0,
            total_chunks: 0,
            last_indexed: None::<DateTime<Utc>>,
            watching: false,
        })
    } else {
        IndexStats { indexed_files: 0, total_chunks: 0, last_indexed: None::<DateTime<Utc>>, watching: false }
    };

    // Overlay the watching flag from the PID sentinel.
    let watching = stats.watching || is_process_watching(&dir);

    let last_indexed_str = stats
        .last_indexed
        .map(|dt: DateTime<Utc>| dt.to_rfc3339())
        .unwrap_or_else(|| "null".to_string());

    if json_output {
        // Hook-facing contract consumed by session-start and post-edit-reindex.
        // Field set is stable; do not remove or rename fields.
        let obj = json!({
            "indexed_files":    stats.indexed_files,
            "total_chunks":     stats.total_chunks,
            "last_indexed":     stats.last_indexed.map(|dt: DateTime<Utc>| dt.to_rfc3339()),
            "estimated_stale":  0,
            "watching":         watching,
        });
        println!("{}", serde_json::to_string_pretty(&obj)?);
    } else {
        println!("indexed_files:  {}", stats.indexed_files);
        println!("total_chunks:   {}", stats.total_chunks);
        println!("last_indexed:   {last_indexed_str}");
        println!("watching:       {watching}");
    }
    Ok(())
}

async fn run_clear(path: Option<PathBuf>) -> anyhow::Result<()> {
    let root = resolve_root(path)?;
    let dir = index_dir(&root);

    if dir.exists() {
        std::fs::remove_dir_all(&dir)
            .with_context(|| format!("failed to remove index directory: {}", dir.display()))?;
        println!("Index cleared: {}", dir.display());
    } else {
        println!("Nothing to clear (no index at {}).", dir.display());
    }
    Ok(())
}

async fn run_watch(path: PathBuf, provider_name: String) -> anyhow::Result<()> {
    // Validate provider name immediately so we fail fast on bad input.
    // We do NOT construct the provider yet — writing the PID file first
    // ensures `status --json` reports watching=true even while the model
    // loads, which may take time on first use.

    // Reject unknown providers before doing any I/O.
    if provider_name != "fastembed" {
        anyhow::bail!(
            "unknown embedding provider: '{}'. Supported providers: fastembed",
            provider_name
        );
    }

    let root = std::fs::canonicalize(&path)
        .with_context(|| format!("cannot access path: {}", path.display()))?;
    let dir = index_dir(&root);

    // Write the PID file first so concurrent `status --json` calls see
    // watching=true immediately, before any slow model initialisation.
    std::fs::create_dir_all(&dir)?;
    std::fs::write(pid_file(&dir), std::process::id().to_string())
        .context("failed to write watch PID file")?;

    // Best-effort initial indexing pass: if the provider is unavailable
    // (no model, no network), log a warning and continue watching without
    // an initial index rather than failing the whole command.  The PID
    // file is already written so `status --json` will report watching=true.
    match make_provider(&provider_name) {
        Ok(provider) => {
            let dim = provider.dim();
            let backend = open_backend(&dir)?;
            let manifest = open_manifest(&dir)?;
            if let Err(e) = backend.initialize(dim).await {
                eprintln!("skelesearch watch: backend init failed: {e}");
            } else {
                let indexer = Indexer::new(backend, manifest, provider);
                match indexer.index_path(&root).await {
                    Ok(r) => eprintln!(
                        "skelesearch watch: indexed {} file(s) in {}",
                        r.indexed_files,
                        root.display()
                    ),
                    Err(e) => eprintln!("skelesearch watch: initial index failed: {e}"),
                }
            }
        }
        Err(e) => {
            eprintln!(
                "skelesearch watch: provider unavailable ({e}), watching without initial index"
            );
        }
    }

    eprintln!("skelesearch watch: watching {} (Ctrl-C to stop)", root.display());

    // Wait for termination signal.  File-change re-indexing is a v2 feature;
    // in v1 this loop merely keeps the process alive and the PID sentinel
    // valid so other tools can detect that watching is active.
    tokio::signal::ctrl_c().await.ok();

    // Best-effort cleanup: remove PID file so `status` shows watching=false.
    let _ = std::fs::remove_file(pid_file(&dir));
    Ok(())
}

// ---------------------------------------------------------------------------
// NoopProvider — used for context lookup which never needs embeddings.
// ---------------------------------------------------------------------------

/// A zero-cost [`EmbedProvider`] for read-only operations that never call
/// `embed_batch` (e.g. `context`).  Using this avoids downloading a model
/// just to display import edges.
struct NoopProvider;

#[async_trait::async_trait]
impl skelesearch_core::EmbedProvider for NoopProvider {
    fn dim(&self) -> usize {
        // Dimension is irrelevant for context-only queries.
        0
    }

    async fn embed_batch(&self, _texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
        anyhow::bail!("NoopProvider does not support embedding; use a real provider for search")
    }
}
