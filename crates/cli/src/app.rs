use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Context as _;
use chrono::{DateTime, Utc};
use fs2::FileExt;
use notify::Watcher as _;
use serde_json::json;
use skelesearch_core::{
    CozoBackend, Config, EmbedProvider, IndexStats, Indexer, ManifestStore, Searcher, StorageBackend,
};
use skelesearch_embed_fastembed::provider_from_name;

use crate::cli::{Cli, Commands};
use crate::eval;

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub async fn run(cli: Cli) -> anyhow::Result<()> {
    // Initialise tracing. Verbose flag sets the default level; RUST_LOG
    // overrides it. When OTEL_EXPORTER_OTLP_ENDPOINT is set and the `otlp`
    // feature is enabled on skelesearch-telemetry, spans are also exported.
    let default_level = match cli.verbose {
        0 => "warn",
        1 => "info",
        2 => "debug",
        _ => "trace",
    };
    let _telemetry = skelesearch_telemetry::init_tracing("skelesearch", default_level);

    match cli.command {
        Commands::Index { path, provider } => run_index(path, provider).await,
        Commands::Search { query, top_k, graph, json, diversity, provider, max_tokens, branch, timings } => {
            run_search(query, top_k as usize, graph, json, diversity, provider, max_tokens, branch, timings).await
        }
        Commands::Context { file } => run_context(file).await,
        Commands::Status { path, json } => run_status(path, json).await,
        Commands::Clear { path } => run_clear(path).await,
        Commands::Watch { path, provider } => run_watch(path, provider).await,
        Commands::Gc { path } => run_gc(path).await,
        Commands::Grep { pattern, path, max_results, ignore_case, json } => {
            run_grep(pattern, path, max_results, ignore_case, json).await
        }
        Commands::Symbol { name, kind } => run_symbol(name, kind).await,
        Commands::Eval { eval_set, provider, json } => run_eval(eval_set, provider, json).await,
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

/// Acquire an exclusive lock on `.skelesearch/.skelesearch.lock`.
///
/// The returned `File` MUST be kept alive for the duration of the operation;
/// dropping it releases the lock.
fn acquire_lock(dir: &std::path::Path) -> anyhow::Result<std::fs::File> {
    let lock_path = dir.join(".skelesearch.lock");
    let file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .open(&lock_path)?;
    match file.try_lock_exclusive() {
        Ok(()) => Ok(file),
        Err(_) => anyhow::bail!(
            "Another skelesearch process is running (lock held at {})",
            lock_path.display()
        ),
    }
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

/// Wire expander + reranker onto a [`Searcher`] using project config and env vars.
///
/// Shared between `run_search`, `run_eval`, and any future command that needs
/// the full search pipeline.
fn configure_searcher<B, P>(searcher: Searcher<B, P>, root: &Path) -> Searcher<B, P>
where
    B: StorageBackend + Send + Sync + 'static,
    P: EmbedProvider + Send + Sync + 'static,
{
    let config = Config::load(root).unwrap_or_default();

    // PageRank boost: disabled when config explicitly sets pagerank_boost = false.
    let searcher = match config.search.pagerank_boost {
        Some(false) => searcher.with_pagerank_boost(false),
        _ => searcher,
    };

    // Unified Datalog retrieval: single round-trip FTS + HNSW + graph + PageRank.
    // Also applies any tuning parameters from the config.
    let searcher = match config.search.unified_search {
        Some(true) => searcher.with_unified_search(true).with_search_tuning(&config),
        _ => searcher.with_search_tuning(&config),
    };

    // Query expansion: skip if explicitly disabled; otherwise attach LLMExpander
    // when OPENAI_API_KEY is available.
    let searcher = match config.search.expansion.enabled {
        Some(false) => searcher,
        _ => match std::env::var("OPENAI_API_KEY").ok().filter(|k| !k.is_empty()) {
            Some(key) => searcher.with_expander(Box::new(skelesearch_core::LLMExpander::new(key))),
            None => searcher,
        },
    };

    // Reranker: explicit config wins; fall back to env-var auto-detect.
    if let Some(ref provider_name) = config.search.reranker.provider {
        if provider_name == "local" {
            // Explicit local reranker via config.
            let result = if let Some(ref dir) = config.search.reranker.model_dir {
                // Expand ~ to home directory
                let expanded = if dir.starts_with("~/") {
                    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
                    std::path::PathBuf::from(home).join(&dir[2..])
                } else {
                    std::path::PathBuf::from(dir)
                };
                skelesearch_rerank_local::LocalReranker::new(&expanded)
            } else {
                skelesearch_rerank_local::LocalReranker::default_model()
            };
            match result {
                Ok(r) => {
                    tracing::info!("local reranker enabled");
                    searcher.with_reranker(Box::new(r))
                }
                Err(e) => {
                    tracing::warn!("local reranker failed to load: {e}");
                    searcher
                }
            }
        } else if provider_name == "qwen3" {
            // Explicit Qwen3-Reranker-0.6B via config.
            // Requires the `qwen3` cargo feature.
            #[cfg(feature = "qwen3")]
            {
                let result = if let Some(ref dir) = config.search.reranker.model_dir {
                    skelesearch_rerank_qwen3::Qwen3Reranker::from_path(std::path::Path::new(dir))
                } else {
                    skelesearch_rerank_qwen3::Qwen3Reranker::from_hf()
                };
                match result {
                    Ok(r) => {
                        tracing::info!("Qwen3 reranker enabled");
                        return searcher.with_reranker(Box::new(r));
                    }
                    Err(e) => {
                        tracing::warn!("Qwen3 reranker failed to load: {e}");
                        return searcher;
                    }
                }
            }
            #[cfg(not(feature = "qwen3"))]
            {
                tracing::warn!(
                    "reranker = 'qwen3' in config but the `qwen3` feature is not enabled; \
                     rebuild with --features qwen3"
                );
                searcher
            }
        } else {
            let key_env = config.search.reranker.api_key_env.as_deref().unwrap_or(
                match provider_name.as_str() {
                    "jina"   => "JINA_API_KEY",
                    "cohere" => "COHERE_API_KEY",
                    "voyage" => "VOYAGE_API_KEY",
                    _        => "RERANKER_API_KEY",
                },
            );
            match std::env::var(key_env).ok().filter(|k| !k.is_empty()) {
                Some(key) => match skelesearch_rerank_api::reranker_from_name(provider_name, key) {
                    Ok(r)  => searcher.with_reranker(Box::new(r)),
                    Err(e) => { tracing::warn!("reranker init failed: {e}"); searcher }
                },
                None => {
                    tracing::warn!("reranker configured as '{provider_name}' but {key_env} not set");
                    searcher
                }
            }
        }
    } else {
        // No explicit config — check SKELESEARCH_RERANKER env var first,
        // then fall back to API key auto-detect.
        if std::env::var("SKELESEARCH_RERANKER").ok().as_deref() == Some("qwen3") {
            #[cfg(feature = "qwen3")]
            {
                let result = if let Ok(dir) = std::env::var("SKELESEARCH_RERANKER_MODEL_DIR") {
                    skelesearch_rerank_qwen3::Qwen3Reranker::from_path(std::path::Path::new(&dir))
                } else {
                    skelesearch_rerank_qwen3::Qwen3Reranker::from_hf()
                };
                return match result {
                    Ok(r)  => { tracing::info!("Qwen3 reranker enabled"); searcher.with_reranker(Box::new(r)) }
                    Err(e) => { tracing::warn!("Qwen3 reranker failed: {e}"); searcher }
                };
            }
            #[cfg(not(feature = "qwen3"))]
            tracing::warn!(
                "SKELESEARCH_RERANKER=qwen3 set but the `qwen3` feature is not enabled; \
                 rebuild with --features qwen3"
            );
        }
        // No explicit config — auto-detect from env vars.
        let auto = std::env::var("JINA_API_KEY").ok().filter(|k| !k.is_empty()).map(|k| ("jina", k))
            .or_else(|| std::env::var("COHERE_API_KEY").ok().filter(|k| !k.is_empty()).map(|k| ("cohere", k)))
            .or_else(|| std::env::var("VOYAGE_API_KEY").ok().filter(|k| !k.is_empty()).map(|k| ("voyage", k)));
        match auto {
            Some((name, key)) => match skelesearch_rerank_api::reranker_from_name(name, key) {
                Ok(r)  => {
                    tracing::info!(provider = name, "cloud reranker enabled");
                    searcher.with_reranker(Box::new(r))
                }
                Err(_) => searcher,
            },
            None => {
                // No cloud API keys and no explicit config — skip reranking.
                // Local CPU cross-encoders (MiniLM, gte-modernbert) are either
                // too slow or not code-aware enough to improve results.
                // Users can opt in via .skelesearch.toml [search.reranker].
                searcher
            }
        }
    }
}

async fn run_index(path: PathBuf, provider_name: String) -> anyhow::Result<()> {
    // Validate provider before doing any work.
    let provider = provider_from_name(&provider_name)?;
    let dim = provider.dim();

    let root = std::fs::canonicalize(&path)
        .with_context(|| format!("cannot access path: {}", path.display()))?;
    let dir = index_dir(&root);
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("failed to create index dir: {}", dir.display()))?;
    let _lock = acquire_lock(&dir)?;

    let backend = open_backend(&dir)?;
    let manifest = open_manifest(&dir)?;
    let config = Config::load(&root)?;

    backend.initialize(dim).await?;

    let indexer = Indexer::new(backend, manifest, provider)
        .with_excludes(config.index.exclude.clone())
        .with_include_extensions(config.index.include_extensions.clone())
        .with_symbol_enrichment(config.index.symbol_enrichment)
        .with_scope_prefix(config.index.scope_prefix);
    let indexer = if config.index.summarize {
        if let Ok(key) = std::env::var("OPENAI_API_KEY") {
            indexer.with_summary_provider(Box::new(crate::summary::OpenAISummaryProvider::new(key)))
        } else {
            tracing::warn!("summarize=true but OPENAI_API_KEY is not set, skipping summaries");
            indexer
        }
    } else {
        indexer
    };
    let start = std::time::Instant::now();
    let result = indexer.index_path(&root).await?;
    let elapsed = start.elapsed();

    println!(
        "indexed {} file(s), {} chunk(s) ({} removed, {} cache hits) in {:.1}s",
        result.indexed_files, result.total_chunks, result.deleted_files,
        result.cache_hits, elapsed.as_secs_f64()
    );
    Ok(())
}

async fn run_search(
    query: String,
    top_k: usize,
    graph: bool,
    json_output: bool,
    diversity: f32,
    provider_name: String,
    max_tokens: Option<usize>,
    branch_scope: bool,
    show_timings: bool,
) -> anyhow::Result<()> {
    let root = std::env::current_dir().context("failed to get current directory")?;
    let dir = index_dir(&root);

    // Short-circuit: no index at all.
    if !dir.join("index.db").exists() {
        if json_output {
            eprintln!("{}", serde_json::json!({"error": "No index found. Run `skelesearch index <path>` first."}));
            std::process::exit(1);
        } else {
            anyhow::bail!("No index found. Run `skelesearch index <path>` first.");
        }
    }

    let provider = provider_from_name(&provider_name)?;
    let dim = provider.dim();
    let backend = open_backend(&dir)?;
    backend.initialize(dim).await?;

    let searcher = Searcher::new(backend, provider);

    let searcher = configure_searcher(searcher, &root);

    let start = std::time::Instant::now();
    let config = Config::load(&root).unwrap_or_default();
    let graph_enabled = graph || config.search.graph.enabled == Some(true);
    let graph_depth: usize = if graph_enabled {
        // Prefer an explicit config depth (≥1); fall back to 1 if unset or zero.
        config.search.graph.max_depth.max(1)
    } else {
        0
    };
    let (mut results, timings) = searcher.search_with_timings(&query, top_k, graph_enabled, graph_depth, diversity, max_tokens).await?;
    let elapsed = start.elapsed();
    tracing::info!(
        embed_ms = timings.embed_ms,
        retrieve_ms = timings.retrieve_ms,
        expand_ms = timings.expand_ms,
        rerank_ms = timings.rerank_ms,
        graph_ms = timings.graph_ms,
        total_ms = timings.total_ms,
        results = results.len(),
        "search complete",
    );

    if show_timings {
        use std::io::Write as _;
        let mut stderr = std::io::stderr();
        writeln!(stderr, "Pipeline timings:").ok();
        writeln!(stderr, "  embed:    {}ms", timings.embed_ms).ok();
        writeln!(stderr, "  retrieve: {}ms", timings.retrieve_ms).ok();
        writeln!(stderr, "  expand:   {}ms", timings.expand_ms).ok();
        writeln!(stderr, "  rerank:   {}ms", timings.rerank_ms).ok();
        writeln!(stderr, "  graph:    {}ms", timings.graph_ms).ok();
        writeln!(stderr, "  total:    {}ms", timings.total_ms).ok();
    }

    // Filter to branch-changed files if requested.
    if branch_scope {
        let changed = skelesearch_core::git::changed_files_on_branch(&root)?;
        if !changed.is_empty() {
            results.retain(|r| changed.iter().any(|c| r.file_path.ends_with(c.as_str()) || c.ends_with(&r.file_path)));
        }
    }

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
        if !results.is_empty() {
            println!("({} results in {:.0}ms)", results.len(), elapsed.as_secs_f64() * 1000.0);
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
            estimated_stale: 0,
        })
    } else {
        IndexStats { indexed_files: 0, total_chunks: 0, last_indexed: None::<DateTime<Utc>>, watching: false, estimated_stale: 0 }
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
            "estimated_stale":  open_manifest(&dir).map(|m| m.count_stale(&root).unwrap_or(0)).unwrap_or(0),
            "watching":         watching,
        });
        println!("{}", serde_json::to_string_pretty(&obj)?);
    } else {
        let stale = open_manifest(&dir).map(|m| m.count_stale(&root).unwrap_or(0)).unwrap_or(0);
        println!("indexed_files:  {}", stats.indexed_files);
        println!("total_chunks:   {}", stats.total_chunks);
        println!("last_indexed:   {last_indexed_str}");
        println!("estimated_stale: {stale}");
        println!("watching:       {watching}");
        if stale > 0 {
            println!("hint: {} file(s) may be out of date; run `skelesearch index <path>` to refresh.", stale);
        }
    }
    Ok(())
}

async fn run_clear(path: Option<PathBuf>) -> anyhow::Result<()> {
    let root = resolve_root(path)?;
    let dir = index_dir(&root);

    if dir.exists() {
        // Acquire the write lock before deleting so we don't race with a
        // concurrent index or watch that is mid-write.  The OS keeps the
        // flock alive even after remove_dir_all unlinks the lock file.
        std::fs::create_dir_all(&dir)?;
        let _lock = acquire_lock(&dir)?;
        std::fs::remove_dir_all(&dir)
            .with_context(|| format!("failed to remove index directory: {}", dir.display()))?;
        println!("Index cleared: {}", dir.display());
    } else {
        println!("Nothing to clear (no index at {}).", dir.display());
    }
    Ok(())
}

async fn run_watch(path: PathBuf, provider_name: String) -> anyhow::Result<()> {
    let root = std::fs::canonicalize(&path)
        .with_context(|| format!("cannot access path: {}", path.display()))?;
    let dir = index_dir(&root);
    let config = Config::load(&root)?;

    // Write PID sentinel so concurrent `status --json` calls see watching=true
    // immediately, before any slow model initialisation.
    std::fs::create_dir_all(&dir)?;
    let _lock = acquire_lock(&dir)?;
    std::fs::write(pid_file(&dir), std::process::id().to_string())
        .context("failed to write watch PID file")?;

    // Best-effort initial indexing pass: if the provider is unavailable
    // (no model, no network), log a warning and continue watching without
    // an initial index rather than failing the whole command.
    match provider_from_name(&provider_name) {
        Ok(provider) => {
            let dim = provider.dim();
            let backend = open_backend(&dir)?;
            let manifest = open_manifest(&dir)?;
            if let Err(e) = backend.initialize(dim).await {
                eprintln!("skelesearch watch: backend init failed: {e}");
            } else {
                let indexer = Indexer::new(backend, manifest, provider)
                    .with_excludes(config.index.exclude.clone())
                    .with_include_extensions(config.index.include_extensions.clone())
                    .with_symbol_enrichment(config.index.symbol_enrichment)
                    .with_scope_prefix(config.index.scope_prefix);
                let indexer = if config.index.summarize {
                    if let Ok(key) = std::env::var("OPENAI_API_KEY") {
                        indexer.with_summary_provider(Box::new(crate::summary::OpenAISummaryProvider::new(key)))
                    } else {
                        tracing::warn!("summarize=true but OPENAI_API_KEY is not set, skipping summaries");
                        indexer
                    }
                } else {
                    indexer
                };
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

    // Set up file watcher with a 2-second debounce window.  Rapid edits (e.g.
    // a save-on-format cascade) are coalesced into a single re-index trigger.
    let (sync_tx, sync_rx) = std::sync::mpsc::channel();
    let mut debouncer = notify_debouncer_full::new_debouncer(
        std::time::Duration::from_secs(2),
        None,
        sync_tx,
    )?;
    debouncer.watcher().watch(&root, notify::RecursiveMode::Recursive)?;

    // Bridge the sync mpsc channel to tokio so we can select! with ctrl_c.
    // The bridge thread owns `sync_rx` and forwards every batch until the
    // async receiver is dropped (on loop exit).
    let (async_tx, mut async_rx) = tokio::sync::mpsc::unbounded_channel();
    std::thread::spawn(move || {
        while let Ok(event) = sync_rx.recv() {
            if async_tx.send(event).is_err() {
                break;
            }
        }
    });

    // Sentinel directories: events here are internal write-back and must be
    // ignored to avoid re-index storms triggered by our own index writes.
    let git_dir = root.join(".git");

    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                eprintln!("skelesearch watch: shutting down");
                break;
            }
            Some(event_result) = async_rx.recv() => {
                match event_result {
                    Ok(events) => {
                        // Ignore changes in .skelesearch/ (our own index writes)
                        // and .git/ (VCS bookkeeping that we don't index).
                        let relevant_count = events.iter()
                            .flat_map(|e| &e.paths)
                            .filter(|p| !p.starts_with(&dir) && !p.starts_with(&git_dir))
                            .count();
                        if relevant_count == 0 {
                            continue;
                        }
                        tracing::info!(changed_files = relevant_count, "file changes detected, re-indexing");
                        match provider_from_name(&provider_name) {
                            Ok(provider) => {
                                let dim = provider.dim();
                                let backend = open_backend(&dir)?;
                                let manifest = open_manifest(&dir)?;
                                if let Err(e) = backend.initialize(dim).await {
                                    eprintln!("skelesearch watch: re-index backend init failed: {e}");
                                } else {
                                    let indexer = Indexer::new(backend, manifest, provider)
                                        .with_excludes(config.index.exclude.clone())
                                        .with_include_extensions(config.index.include_extensions.clone());
                                    let indexer = if config.index.summarize {
                                        if let Ok(key) = std::env::var("OPENAI_API_KEY") {
                                            indexer.with_summary_provider(Box::new(crate::summary::OpenAISummaryProvider::new(key)))
                                        } else {
                                            tracing::warn!("summarize=true but OPENAI_API_KEY is not set, skipping summaries");
                                            indexer
                                        }
                                    } else {
                                        indexer
                                    };
                                    match indexer.index_path(&root).await {
                                        Ok(r) => eprintln!(
                                            "skelesearch watch: re-indexed {} file(s), {} chunk(s)",
                                            r.indexed_files, r.total_chunks
                                        ),
                                        Err(e) => eprintln!("skelesearch watch: re-index failed: {e}"),
                                    }
                                }
                            }
                            Err(e) => {
                                eprintln!("skelesearch watch: provider unavailable ({e}), skipping re-index");
                            }
                        }
                    }
                    Err(errs) => {
                        for e in errs {
                            tracing::warn!("file watcher error: {e}");
                        }
                    }
                }
            }
        }
    }

    // Best-effort cleanup: remove PID file so `status` shows watching=false.
    let _ = std::fs::remove_file(pid_file(&dir));
    Ok(())
}

async fn run_gc(path: Option<PathBuf>) -> anyhow::Result<()> {
    let root = resolve_root(path)?;
    let dir = index_dir(&root);
    if !dir.join("index.db").exists() {
        println!("No index found.");
        return Ok(());
    }
    let _lock = acquire_lock(&dir)?;
    let backend = open_backend(&dir)?;
    let manifest = open_manifest(&dir)?;
    let removed = skelesearch_core::gc::collect_garbage(&root, &backend, &manifest).await?;
    println!("Removed {removed} orphaned file(s) from index.");
    Ok(())
}

async fn run_grep(
    pattern: String,
    path: Option<PathBuf>,
    max_results: usize,
    ignore_case: bool,
    json_output: bool,
) -> anyhow::Result<()> {
    let root = resolve_root(path)?;
    let opts = skelesearch_core::GrepOptions {
        max_results,
        case_insensitive: ignore_case,
    };
    let results = skelesearch_core::grep_codebase(&root, &pattern, &opts)?;

    if json_output {
        let rows: Vec<serde_json::Value> = results
            .iter()
            .map(|r| {
                serde_json::json!({
                    "file_path": r.file_path,
                    "line_number": r.line_number,
                    "line_content": r.line_content,
                    "why": "grep",
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&rows)?);
    } else {
        if results.is_empty() {
            println!("No matches.");
        }
        for r in &results {
            println!("{}:{}: {}", r.file_path, r.line_number, r.line_content);
        }
    }
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


async fn run_symbol(name: String, kind: Option<String>) -> anyhow::Result<()> {
    let root = std::env::current_dir()?;
    let dir = index_dir(&root);
    if !dir.join("index.db").exists() {
        anyhow::bail!("No index found in {}. Run `skelesearch index <path>` first.", dir.display());
    }
    let backend = open_backend(&dir)?;
    let results = backend.find_symbols(&name, kind.as_deref()).await?;
    if results.is_empty() {
        println!("No symbols found for {:?}", name);
        return Ok(());
    }
    for sym in &results {
        println!("{} {} @ {}:{}-{}", sym.kind, sym.name, sym.file_path, sym.start_line, sym.end_line);
    }
    Ok(())
}

async fn run_eval(eval_set_path: PathBuf, provider_name: String, json_output: bool) -> anyhow::Result<()> {
    let provider = provider_from_name(&provider_name)?;
    let dim = provider.dim();
    let root = resolve_root(None)?;
    let dir = index_dir(&root);

    if !dir.join("index.db").exists() {
        anyhow::bail!("No index found. Run `skelesearch index <path>` first.");
    }

    let backend = open_backend(&dir)?;
    backend.initialize(dim).await?;
    let searcher = Searcher::new(backend, provider);
    let config = Config::load(&root).unwrap_or_default();
    let searcher = configure_searcher(searcher, &root);
    let eval_graph_enabled = config.search.graph.enabled == Some(true);
    let eval_graph_depth: usize = if eval_graph_enabled { config.search.graph.max_depth.max(1) } else { 0 };
    let eval_data = std::fs::read_to_string(&eval_set_path)
        .with_context(|| format!("failed to read eval set: {}", eval_set_path.display()))?;
    let cases: Vec<eval::EvalCase> = serde_json::from_str(&eval_data)
        .context("failed to parse eval set JSON")?;

    if cases.is_empty() {
        println!("Eval set is empty.");
        return Ok(());
    }

    let mut results = Vec::new();
    for case in &cases {
        let hits = searcher.search(&case.query, 10, eval_graph_enabled, eval_graph_depth, 0.0, None).await?;
        // Deduplicate file paths (multiple chunks from same file).
        let mut unique_files: Vec<String> = Vec::new();
        for h in &hits {
            if !unique_files.contains(&h.file_path) {
                unique_files.push(h.file_path.clone());
            }
        }

        let metrics = eval::CaseMetrics {
            query: case.query.clone(),
            recall_at_5: eval::recall_at_k(&unique_files, &case.expected_files, 5),
            recall_at_10: eval::recall_at_k(&unique_files, &case.expected_files, 10),
            precision_at_5: eval::precision_at_k(&unique_files, &case.expected_files, 5),
            mrr: eval::mrr(&unique_files, &case.expected_files),
            retrieved_files: unique_files,
            category: case.category.clone(),
        };

        if !json_output {
            println!(
                "Q: {} | R@5={:.2} R@10={:.2} P@5={:.2} MRR={:.2}",
                metrics.query, metrics.recall_at_5, metrics.recall_at_10, metrics.precision_at_5, metrics.mrr
            );
        }
        results.push(metrics);
    }

    let agg = eval::aggregate(&results);

    if json_output {
        let output = serde_json::json!({
            "cases": results.iter().map(|r| serde_json::json!({
                "query": r.query,
                "recall_at_5": r.recall_at_5,
                "recall_at_10": r.recall_at_10,
                "precision_at_5": r.precision_at_5,
                "mrr": r.mrr,
                "retrieved_files": r.retrieved_files,
                "category": r.category,
            })).collect::<Vec<_>>(),
            "aggregate": {
                "mean_recall_at_5": agg.mean_recall_at_5,
                "mean_recall_at_10": agg.mean_recall_at_10,
                "mean_precision_at_5": agg.mean_precision_at_5,
                "mean_mrr": agg.mean_mrr,
                "total_cases": agg.total_cases,
            }
        });
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        println!("\n--- Aggregate ---");
        println!("Cases:       {}", agg.total_cases);
        println!("Mean R@5:    {:.3}", agg.mean_recall_at_5);
        println!("Mean R@10:   {:.3}", agg.mean_recall_at_10);
        println!("Mean P@5:    {:.3}", agg.mean_precision_at_5);
        println!("Mean MRR:    {:.3}", agg.mean_mrr);
    }
    Ok(())
}