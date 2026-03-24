# Skelesearch v1 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build skelesearch v1 end-to-end: AST-aware indexing, hybrid retrieval, CLI and MCP surfaces, Claude Code plugin assets, and Nix packaging.

**Architecture:** Keep all storage/query logic in `skelesearch-core`, with `schema.rs` as the sole CozoDB boundary behind `StorageBackend`, `manifest.rs` as the separate SQLite hash manifest, `chunker/` as the tree-sitter + text-splitter layer, and `indexer.rs`/`searcher.rs` as orchestration layers. Ship thin adapters on top: `skelesearch-embed-fastembed` for the default embedding provider, `skelesearch-cli` for local commands, `skelesearch-mcp` for rmcp stdio tools, and repo-root Claude plugin assets for zero-friction agent use. The implementation should follow the researched fit already captured by the spec and ADRs: CozoDB HNSW + FTS inside Datalog, `text-splitter` `CodeSplitter` for chunking, and rmcp 0.16 `tool_box` + `schemars` for tool schemas.

**Tech Stack:** Rust workspace, CozoDB 0.7.6, rusqlite, tree-sitter grammars, text-splitter, fastembed-rs, rmcp 0.16, clap 4, tokio, Nix/crane, Claude Code plugin hooks.

---

## File Structure

### Workspace and shared dependencies
- Modify: `Cargo.toml`
  - Add shared runtime deps needed by the spec (`chrono`, `tracing`, `tracing-subscriber`, `tempfile`) and test deps used across crates (`assert_cmd`, `predicates`).
  - Normalize the `fastembed` workspace dependency so the embed crate owns optionality; do not hardcode embedding-provider behavior into `skelesearch-core`.

### `crates/core`
- Modify: `crates/core/src/lib.rs:1-6`
  - Export the real modules and re-export the public entry points the CLI/MCP crates use.
- Create: `crates/core/src/provider.rs`
  - Define `EmbedProvider` and any provider-neutral config/selection types that belong in core.
- Create: `crates/core/src/schema.rs`
  - Define `StorageBackend`, record types, `CozoBackend`, schema creation, CRUD methods, hybrid search, file-context helpers, and stats.
- Create: `crates/core/src/manifest.rs`
  - Own the separate SQLite manifest (`mtime`, `size`, `xxhash3`) and stale-path reconciliation helpers.
- Create: `crates/core/src/chunker/mod.rs`
  - Parse files, invoke `CodeSplitter`, normalize identifiers for FTS, and return chunk metadata + extracted import edges.
- Create: `crates/core/src/chunker/languages.rs`
  - Define `LanguageConfig`, registry wiring, Tier 1 language configs, and Tier 2 fallback selection.
- Create: `crates/core/src/indexer.rs`
  - Walk repos with `ignore`, classify deltas, reconcile deletions, batch-embed chunks, and upsert backend + manifest state.
- Create: `crates/core/src/searcher.rs`
  - Wrap hybrid search, graph augmentation, `get_file_context`, and match-quality labeling.
- Create: `crates/core/tests/storage_contracts.rs`
- Create: `crates/core/tests/manifest_store.rs`
- Create: `crates/core/tests/chunker.rs`
- Create: `crates/core/tests/indexer.rs`
- Create: `crates/core/tests/searcher.rs`
- Create: `crates/core/tests/fixtures/sample_repo/` (small Rust/TS/Nix/Python sample tree used by integration tests)

### `crates/embed-fastembed`
- Modify: `crates/embed-fastembed/src/lib.rs:1-2`
  - Implement the real `FastEmbedProvider` against `EmbedProvider`.
- Create: `crates/embed-fastembed/tests/provider.rs`

### `crates/cli`
- Modify: `crates/cli/src/main.rs:1-12`
  - Keep bootstrap thin.
- Create: `crates/cli/src/cli.rs`
  - `clap` argument structs and subcommand definitions.
- Create: `crates/cli/src/app.rs`
  - Command dispatch, JSON/plain rendering, provider selection, watch-mode wiring.
- Create: `crates/cli/tests/cli.rs`

### `crates/mcp`
- Modify: `crates/mcp/src/main.rs:1-9`
  - Stdio bootstrap and stderr-only logging.
- Create: `crates/mcp/src/server.rs`
  - `SkeleSearchServer`, shared state, rmcp server handler wiring.
- Create: `crates/mcp/src/tools.rs`
  - Tool input/output structs (`schemars::JsonSchema`) and tool methods.
- Create: `crates/mcp/tests/server.rs`

### Repo-root plugin + packaging assets
- Create: `.claude-plugin/plugin.json`
- Create: `hooks/hooks.json`
- Create: `hooks/session-start`
- Create: `hooks/post-edit-reindex`
- Create: `skills/search-code/SKILL.md`
- Create: `agents/skelesearch-scout.md`
- Create: `CLAUDE.md.template`
- Create: `flake.nix`

---

## Chunk 1: Core Index/Search Engine

### Task 1: Establish core contracts and schema boundary

**Files:**
- Modify: `Cargo.toml`
- Modify: `crates/core/src/lib.rs:1-6`
- Create: `crates/core/src/provider.rs`
- Create: `crates/core/src/schema.rs`
- Test: `crates/core/tests/storage_contracts.rs`

- [ ] **Step 1: Write the failing storage-contract tests**

```rust
#[tokio::test]
async fn cozo_backend_round_trips_storage_backend_contract() -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let backend = CozoBackend::open(temp.path().join("index.db"))?;
    backend.initialize(8).await?;

    backend.upsert_file(&FileRecord {
        file_path: "src/lib.rs".into(),
        language: "rust".into(),
        last_modified: 10,
        last_indexed: 10,
        chunk_count: 1,
    }).await?;
    backend.upsert_chunks(&[ChunkRecord {
        file_path: "src/lib.rs".into(),
        chunk_idx: 0,
        content: "fn alpha() {}".into(),
        normalized: "fn alpha".into(),
        chunk_type: "function".into(),
        start_line: 1,
        end_line: 1,
        embedding: Some(vec![0.1; 8]),
    }]).await?;
    backend.upsert_edges(&[EdgeRecord {
        from_file: "src/lib.rs".into(),
        from_chunk: 0,
        to_file: "src/search.rs".into(),
        edge_type: "imports".into(),
    }]).await?;

    assert_eq!(backend.list_indexed_paths().await?, vec!["src/lib.rs".to_string()]);
    assert_eq!(backend.get_chunks_for_file("src/lib.rs").await?.len(), 1);
    assert_eq!(backend.get_imports("src/lib.rs").await?, vec!["src/search.rs".to_string()]);
    assert_eq!(backend.get_importers("src/search.rs").await?, vec!["src/lib.rs".to_string()]);

    backend.delete_edges_for_file("src/lib.rs").await?;
    backend.delete_chunks_for_file("src/lib.rs").await?;
    backend.delete_file("src/lib.rs").await?;
    assert!(backend.list_indexed_paths().await?.is_empty());
    Ok(())
}

#[tokio::test]
async fn cozo_backend_initializes_and_reports_empty_stats() -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let backend = CozoBackend::open(temp.path().join("index.db"))?;

    backend.initialize(768).await?;
    backend.initialize(768).await?;
    let stats = backend.stats().await?;
    let hits = backend.hybrid_search(&vec![0.0; 768], "missing symbol", 5).await?;

    assert_eq!(stats.indexed_files, 0);
    assert_eq!(stats.total_chunks, 0);
    assert!(stats.last_indexed.is_none());
    assert!(!stats.watching);
    assert!(hits.is_empty());
    Ok(())
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p skelesearch-core cozo_backend_ -- --nocapture`
Expected: FAIL with unresolved imports/types because `schema.rs` and `provider.rs` do not exist yet.

- [ ] **Step 3: Implement the full public contracts first**

```rust
#[async_trait]
pub trait StorageBackend: Send + Sync {
    async fn initialize(&self, dim: usize) -> anyhow::Result<()>;
    async fn upsert_file(&self, record: &FileRecord) -> anyhow::Result<()>;
    async fn delete_file(&self, file_path: &str) -> anyhow::Result<()>;
    async fn list_indexed_paths(&self) -> anyhow::Result<Vec<String>>;
    async fn upsert_chunks(&self, chunks: &[ChunkRecord]) -> anyhow::Result<()>;
    async fn delete_chunks_for_file(&self, file_path: &str) -> anyhow::Result<()>;
    async fn get_chunks_for_file(&self, file_path: &str) -> anyhow::Result<Vec<ChunkRecord>>;
    async fn upsert_edges(&self, edges: &[EdgeRecord]) -> anyhow::Result<()>;
    async fn delete_edges_for_file(&self, file_path: &str) -> anyhow::Result<()>;
    async fn get_importers(&self, file_path: &str) -> anyhow::Result<Vec<String>>;
    async fn get_imports(&self, file_path: &str) -> anyhow::Result<Vec<String>>;
    async fn hybrid_search(&self, query_vec: &[f32], query_str: &str, top_k: usize) -> anyhow::Result<Vec<SearchResult>>;
    async fn stats(&self) -> anyhow::Result<IndexStats>;
}
```

Also define the record types from the schema-level contract (`FileRecord` including `last_indexed`, `ChunkRecord`, `EdgeRecord`, `SearchResult`, `IndexStats`) plus the provider-neutral `EmbedProvider` trait in `provider.rs`; treat the shorter trait snippet in the spec as documentation drift and normalize on the schema/data-model section.

- [ ] **Step 4: Implement `CozoBackend` schema initialization and empty-query behavior**

```rust
let schema = r#"
:create files { file_path: String => language: String, last_modified: Int, last_indexed: Int, chunk_count: Int }
:create chunks { file_path: String, chunk_idx: Int => content: String, normalized: String, chunk_type: String, start_line: Int, end_line: Int, embedding: [Float]? }
:create code_edges { from_file: String, from_chunk: Int, to_file: String => edge_type: String, created_at: Int }
"#;
```

Implementation notes:
- Keep all Cozo-specific query strings in `schema.rs` only.
- `initialize(dim)` must create relations and both indices exactly once.
- `hybrid_search` on an empty index must return `Ok(vec![])`, not a fake record and not an error.
- Do not defer any `StorageBackend` methods to later crates; Chunk 1 owns the full boundary contract.

- [ ] **Step 5: Run targeted and surrounding tests**

Run:
- `cargo test -p skelesearch-core cozo_backend_round_trips_storage_backend_contract -- --exact`
- `cargo test -p skelesearch-core cozo_backend_initializes_and_reports_empty_stats -- --exact`
- `cargo test -p skelesearch-core --test storage_contracts -- --nocapture`

Expected: PASS; schema creation is idempotent, the full trait surface round-trips data through Cozo, and empty search returns no results.

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml crates/core/src/lib.rs crates/core/src/provider.rs crates/core/src/schema.rs crates/core/tests/storage_contracts.rs
git commit -m "feat(core): add storage contracts and cozo backend"
```

### Task 2: Add manifest storage and stale-path reconciliation

**Files:**
- Create: `crates/core/src/manifest.rs`
- Test: `crates/core/tests/manifest_store.rs`

- [ ] **Step 1: Write the failing manifest test**

```rust
#[test]
fn manifest_detects_changed_and_deleted_files() -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let manifest = ManifestStore::open(temp.path().join("manifest.db"))?;

    manifest.upsert("src/lib.rs", 10, 100, "hash-a")?;
    assert!(manifest.is_unchanged("src/lib.rs", 10, 100, "hash-a")?);
    assert!(!manifest.is_unchanged("src/lib.rs", 11, 100, "hash-a")?);
    assert_eq!(manifest.list_paths()?, vec!["src/lib.rs".to_string()]);
    Ok(())
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p skelesearch-core manifest_detects_changed_and_deleted_files -- --exact`
Expected: FAIL because `ManifestStore` is not implemented.

- [ ] **Step 3: Implement the manifest API**

```rust
pub struct ManifestEntry {
    pub file_path: String,
    pub mtime: i64,
    pub size: i64,
    pub xxhash3: String,
}

impl ManifestStore {
    pub fn stale_paths_against(&self, visited: &std::collections::HashSet<String>) -> anyhow::Result<Vec<String>> {
        Ok(self.list_paths()?.into_iter().filter(|path| !visited.contains(path)).collect())
    }
}
```

Implementation notes:
- Use a separate SQLite file, never CozoDB, per ADR-006.
- `stale_paths_against` must drive rename/delete cleanup later; do not bury deletion logic inside the walker.
- Keep manifest rows authoritative for known indexed paths only.

- [ ] **Step 4: Add deletion- and rename-oriented integration tests**

```rust
#[test]
fn stale_paths_against_reports_removed_paths() -> anyhow::Result<()> {
    let mut visited = std::collections::HashSet::new();
    visited.insert("src/new.rs".into());
    assert_eq!(manifest.stale_paths_against(&visited)?, vec!["src/lib.rs".to_string()]);
    Ok(())
}

#[test]
fn stale_paths_against_treats_rename_as_old_path_becoming_stale() -> anyhow::Result<()> {
    let mut visited = std::collections::HashSet::new();
    visited.insert("src/renamed.rs".into());
    let stale = manifest.stale_paths_against(&visited)?;
    assert!(stale.contains(&"src/lib.rs".to_string()));
    Ok(())
}
```

- [ ] **Step 5: Run targeted tests**

Run:
- `cargo test -p skelesearch-core manifest_detects_changed_and_deleted_files -- --exact`
- `cargo test -p skelesearch-core stale_paths_against_reports_removed_paths -- --exact`
- `cargo test -p skelesearch-core stale_paths_against_treats_rename_as_old_path_becoming_stale -- --exact`
- `cargo test -p skelesearch-core --test manifest_store -- --nocapture`

Expected: PASS; unchanged detection, hash changes, and stale-path reporting all behave deterministically for deletions and rename-style path churn.

- [ ] **Step 6: Commit**

```bash
git add crates/core/src/manifest.rs crates/core/tests/manifest_store.rs
git commit -m "feat(core): add manifest store for incremental indexing"
```

### Task 3: Build AST chunking, normalization, and import extraction

**Files:**
- Modify: `Cargo.toml`
- Create: `crates/core/src/chunker/mod.rs`
- Create: `crates/core/src/chunker/languages.rs`
- Modify: `crates/core/src/lib.rs:1-6`
- Test: `crates/core/tests/chunker.rs`

- [ ] **Step 1: Write the failing chunker tests**

```rust
#[test]
fn rust_chunker_preserves_function_boundaries() -> anyhow::Result<()> {
    let source = r#"
fn alpha() { println!("a"); }
fn beta() { println!("b"); }
"#;
    let chunker = Chunker::default();
    let chunks = chunker.chunk_file("src/lib.rs", source)?;

    assert!(chunks.iter().any(|c| c.content.contains("fn alpha")));
    assert!(chunks.iter().any(|c| c.content.contains("fn beta")));
    Ok(())
}

#[test]
fn normalization_splits_mixed_identifiers() {
    assert_eq!(normalize_for_fts("parseHTTPResponse_json"), "parse http response json");
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
- `cargo test -p skelesearch-core rust_chunker_preserves_function_boundaries -- --exact`
- `cargo test -p skelesearch-core normalization_splits_mixed_identifiers -- --exact`

Expected: FAIL because `Chunker`, `LanguageConfig`, and normalization helpers do not exist.

- [ ] **Step 3: Implement language registry and chunker output types**

```rust
pub trait LanguageConfig: Send + Sync {
    fn file_extensions(&self) -> &[&'static str];
    fn language(&self) -> tree_sitter::Language;
    fn chunk_node_kinds(&self) -> &[&'static str];
    fn import_query(&self) -> &str;
}

pub struct ParsedChunk {
    pub chunk_idx: usize,
    pub content: String,
    pub normalized: String,
    pub chunk_type: String,
    pub start_line: usize,
    pub end_line: usize,
}
```

- [ ] **Step 4: Implement Tier 1 language configs and fallback chunking**

Implementation notes:
- Register Rust, Nix, Python, TypeScript, JavaScript, and Go exactly as the spec lists.
- Use `CodeSplitter` for Tier 1 files with a 1,500 non-whitespace character budget.
- Use a Tier 2 sliding-window fallback for all other extensions.
- Extract import edges with tree-sitter queries; return edge candidates alongside chunks.
- Keep identifier normalization separate from raw `content`; never mutate the stored source text.

- [ ] **Step 5: Add Tier 1 coverage, import-extraction, and Tier 2 fallback tests**

```rust
#[test]
fn typescript_import_query_extracts_edges() -> anyhow::Result<()> {
    let source = "import { search } from './search';\nexport const x = search();";
    let edges = Chunker::default().extract_edges("src/app.ts", source)?;
    assert_eq!(edges[0].to_file, "./search");
    Ok(())
}

#[test]
fn nix_python_javascript_and_go_configs_produce_chunks() -> anyhow::Result<()> {
    assert!(!Chunker::default().chunk_file("flake.nix", "let x = 1; in x")?.is_empty());
    assert!(!Chunker::default().chunk_file("app.py", "def run():\n    return 1\n")?.is_empty());
    assert!(!Chunker::default().chunk_file("app.js", "function run() { return 1; }")?.is_empty());
    assert!(!Chunker::default().chunk_file("main.go", "package main\nfunc run() {}")?.is_empty());
    Ok(())
}

#[test]
fn unknown_extension_uses_sliding_window_fallback() -> anyhow::Result<()> {
    let source = "word ".repeat(400);
    let chunks = Chunker::default().chunk_file("notes.txt", &source)?;
    assert!(chunks.len() > 1);
    Ok(())
}
```

- [ ] **Step 6: Run targeted tests**

Run:
- `cargo test -p skelesearch-core --test chunker -- --nocapture`

Expected: PASS; Tier 1 chunk boundaries, all declared Tier 1 languages, normalization, import-edge extraction, and Tier 2 fallback chunking all match the spec.

- [ ] **Step 7: Commit**

```bash
git add Cargo.toml crates/core/src/lib.rs crates/core/src/chunker/mod.rs crates/core/src/chunker/languages.rs crates/core/tests/chunker.rs
git commit -m "feat(core): add ast-aware chunker and language registry"
```

### Task 4: Implement indexing orchestration and retrieval APIs

**Files:**
- Modify: `Cargo.toml`
- Create: `crates/core/src/indexer.rs`
- Create: `crates/core/src/searcher.rs`
- Modify: `crates/core/src/schema.rs`
- Modify: `crates/core/src/manifest.rs`
- Create: `crates/core/tests/indexer.rs`
- Create: `crates/core/tests/searcher.rs`
- Create: `crates/core/tests/fixtures/sample_repo/`

- [ ] **Step 1: Write failing indexer and searcher integration tests**

```rust
#[tokio::test]
async fn indexer_skips_unchanged_files_reconciles_renames_and_removes_deleted_paths() -> anyhow::Result<()> {
    let fixture = fixture_repo()?;
    let backend = test_backend()?;
    let manifest = test_manifest()?;
    let provider = DeterministicTestProvider::new(8);
    let indexer = Indexer::new(backend.clone(), manifest.clone(), provider);

    let first = indexer.index_path(fixture.path()).await?;
    std::fs::rename(fixture.path().join("src/old.rs"), fixture.path().join("src/new.rs"))?;
    let second = indexer.index_path(fixture.path()).await?;

    assert!(first.indexed_files >= 1);
    assert!(second.deleted_files >= 1);
    assert!(backend.list_indexed_paths().await?.contains(&"src/new.rs".to_string()));
    assert!(!backend.list_indexed_paths().await?.contains(&"src/old.rs".to_string()));
    assert!(backend.get_chunks_for_file("src/old.rs").await?.is_empty());
    assert!(backend.get_imports("src/old.rs").await?.is_empty());
    assert!(!manifest.list_paths()?.contains(&"src/old.rs".to_string()));
    Ok(())
}

#[tokio::test]
async fn searcher_returns_provenance_graph_hits_and_quality_buckets() -> anyhow::Result<()> {
    let plain = searcher.search("import edges", 5, false).await?;
    let graph = searcher.search("import edges", 5, true).await?;
    assert!(!plain.is_empty());
    assert!(plain.iter().all(|row| matches!(row.match_quality.as_str(), "high" | "moderate" | "low")));
    assert!(plain.iter().all(|row| row.why == "vector" || row.why == "fts" || row.why == "both"));
    assert!(graph.iter().any(|row| row.why.starts_with("imports ")));
    Ok(())
}

#[tokio::test]
async fn match_quality_uses_documented_relative_thresholds() -> anyhow::Result<()> {
    let labels = Searcher::label_match_quality(&[1.0, 0.8, 0.5, 0.49]);
    assert_eq!(labels, vec!["high", "high", "moderate", "low"]);
    Ok(())
}

#[tokio::test]
async fn indexer_batches_embeddings_instead_of_one_call_per_chunk() -> anyhow::Result<()> {
    let provider = CountingTestProvider::new(8);
    let indexer = Indexer::new(test_backend()?, test_manifest()?, provider.clone());
    indexer.index_path(fixture_repo()?.path()).await?;
    assert!(provider.call_count() < provider.chunk_count_seen());
    Ok(())
}

#[tokio::test]
async fn indexer_updates_last_indexed_after_successful_index() -> anyhow::Result<()> {
    let backend = test_backend()?;
    let provider = DeterministicTestProvider::new(8);
    let indexer = Indexer::new(backend.clone(), test_manifest()?, provider);
    indexer.index_path(fixture_repo()?.path()).await?;
    let stats = backend.stats().await?;
    assert!(stats.last_indexed.is_some());
    Ok(())
}

#[tokio::test]
async fn file_context_and_empty_search_are_truthful_for_missing_data() -> anyhow::Result<()> {
    assert!(searcher.search("definitely missing symbol", 3, false).await?.is_empty());
    let ctx = searcher.file_context("missing.rs").await?;
    assert!(ctx.chunks.is_empty() && ctx.imports.is_empty() && ctx.imported_by.is_empty());
    Ok(())
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
- `cargo test -p skelesearch-core indexer_skips_unchanged_files_reconciles_renames_and_removes_deleted_paths -- --exact`
- `cargo test -p skelesearch-core searcher_returns_provenance_graph_hits_and_quality_buckets -- --exact`
- `cargo test -p skelesearch-core match_quality_uses_documented_relative_thresholds -- --exact`
- `cargo test -p skelesearch-core indexer_batches_embeddings_instead_of_one_call_per_chunk -- --exact`
- `cargo test -p skelesearch-core indexer_updates_last_indexed_after_successful_index -- --exact`
- `cargo test -p skelesearch-core file_context_and_empty_search_are_truthful_for_missing_data -- --exact`

Expected: FAIL because `Indexer`, `Searcher`, and fixture helpers do not exist.

- [ ] **Step 3: Implement `Indexer` around the manifest + chunker + backend contracts**

```rust
pub struct Indexer<B, P> {
    backend: std::sync::Arc<B>,
    manifest: ManifestStore,
    provider: P,
    batch_size: usize,
}
```

Implementation notes:
- Walk with `ignore` and respect gitignore tiers.
- Use metadata (`mtime` + `size`) before xxHash3.
- Delete chunks and edges before re-indexing modified files.
- Reconcile `known_paths - visited_paths` after the walk to handle renames/deletes.
- Batch embedding calls; never call the provider once per chunk.

- [ ] **Step 4: Implement `Searcher` as the read-path wrapper**

```rust
pub struct Searcher<B, P> {
    backend: std::sync::Arc<B>,
    provider: P,
}

pub async fn search(&self, query: &str, top_k: usize, include_graph: bool) -> anyhow::Result<Vec<SearchResult>>
pub async fn file_context(&self, file_path: &str) -> anyhow::Result<FileContext>
```

Implementation notes:
- `Searcher` owns query embedding through `EmbedProvider`; it must produce `query_vec` before calling `StorageBackend::hybrid_search`.
- Derive `match_quality` using the current relative bands from the spec (`high >= 0.8 * top_score`, `moderate >= 0.5 * top_score`, otherwise `low`).
- `get_file_context` for an unindexed file must return empty arrays, not an error.
- Keep all Cozo-specific Datalog for hybrid search and graph augmentation in `schema.rs`; `searcher.rs` stays backend-agnostic and shapes results only.
- Base retrieval provenance stays `vector | fts | both`; graph augmentation appends one-hop results annotated as `imports <target>`.
- Empty search results must return `[]`, never a placeholder row or a hidden error.

- [ ] **Step 5: Run targeted tests and the full core suite**

Run:
- `cargo test -p skelesearch-core indexer_skips_unchanged_files_reconciles_renames_and_removes_deleted_paths -- --exact`
- `cargo test -p skelesearch-core searcher_returns_provenance_graph_hits_and_quality_buckets -- --exact`
- `cargo test -p skelesearch-core match_quality_uses_documented_relative_thresholds -- --exact`
- `cargo test -p skelesearch-core indexer_batches_embeddings_instead_of_one_call_per_chunk -- --exact`
- `cargo test -p skelesearch-core indexer_updates_last_indexed_after_successful_index -- --exact`
- `cargo test -p skelesearch-core file_context_and_empty_search_are_truthful_for_missing_data -- --exact`
- `cargo test -p skelesearch-core`

Expected: PASS; incremental indexing, rename/delete reconciliation, retrieval provenance, threshold labeling, embedding batching, last-indexed updates, and file context all work against real temp databases and fixture repos.

- [ ] **Step 6: Commit**

```bash
git add crates/core/src/indexer.rs crates/core/src/searcher.rs crates/core/src/schema.rs crates/core/src/manifest.rs crates/core/tests/indexer.rs crates/core/tests/searcher.rs crates/core/tests/fixtures/sample_repo
git commit -m "feat(core): add indexing and retrieval orchestration"
```

---

## Chunk 2: Providers and User-Facing Binaries

### Task 5: Implement the real fastembed provider

**Files:**
- Modify: `Cargo.toml`
- Modify: `crates/embed-fastembed/src/lib.rs:1-2`
- Create: `crates/embed-fastembed/tests/provider.rs`

- [ ] **Step 1: Write the failing provider test**

```rust
#[tokio::test]
async fn fastembed_provider_returns_one_vector_per_input_in_order() -> anyhow::Result<()> {
    let provider = FastEmbedProvider::default()?;
    let ab = provider.embed_batch(vec!["fn alpha() {}".into(), "fn beta() {}".into()]).await?;
    let ba = provider.embed_batch(vec!["fn beta() {}".into(), "fn alpha() {}".into()]).await?;

    assert_eq!(ab.len(), 2);
    assert_eq!(ab[0].len(), provider.dim());
    assert_eq!(ab[0], ba[1]);
    assert_eq!(ab[1], ba[0]);
    Ok(())
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p skelesearch-embed-fastembed fastembed_provider_returns_one_vector_per_input_in_order -- --exact`
Expected: FAIL because `FastEmbedProvider` is still a stub.

- [ ] **Step 3: Implement `FastEmbedProvider` against the core trait**

```rust
pub struct FastEmbedProvider {
    model: fastembed::TextEmbedding,
    dim: usize,
}
```

Implementation notes:
- Default to `jina-embeddings-v2-base-code` per ADR-004.
- Cache `dim` once; do not infer it from every batch call.
- Preserve input order in outputs.
- Surface model-load failures as real errors.

- [ ] **Step 4: Run targeted tests**

Run:
- `cargo test -p skelesearch-embed-fastembed fastembed_provider_returns_one_vector_per_input_in_order -- --exact`
- `cargo test -p skelesearch-embed-fastembed`

Expected: PASS; returned vector count, dimensionality, and input-order preservation match the trait contract.

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml crates/embed-fastembed/src/lib.rs crates/embed-fastembed/tests/provider.rs
git commit -m "feat(embed): add fastembed provider"
```

### Task 6: Build the CLI surface, including `watch` as a separate subcommand

**Files:**
- Modify: `Cargo.toml`
- Modify: `crates/cli/Cargo.toml`
- Modify: `crates/cli/src/main.rs:1-12`
- Create: `crates/cli/src/cli.rs`
- Create: `crates/cli/src/app.rs`
- Create: `crates/cli/tests/cli.rs`

- [ ] **Step 1: Write the failing CLI smoke tests**

Define any helpers (`indexed_cli_fixture`, temp-index bootstrap) inside `crates/cli/tests/cli.rs` so the task stays self-contained.

```rust
#[test]
fn search_json_contains_required_result_fields() {
    let repo = indexed_cli_fixture();
    let output = assert_cmd::Command::cargo_bin("skelesearch").unwrap()
        .current_dir(repo.path())
        .args(["search", "import edges", "--json"])
        .output()
        .unwrap();
    assert!(output.status.success());

    let rows: Vec<serde_json::Value> = serde_json::from_slice(&output.stdout).unwrap();
    let row = &rows[0];
    for key in ["file_path", "start_line", "end_line", "content", "score", "match_quality", "why"] {
        assert!(row.get(key).is_some(), "missing {key}");
    }
}

#[test]
fn status_json_contains_hook_facing_fields() {
    let repo = indexed_cli_fixture();
    let output = assert_cmd::Command::cargo_bin("skelesearch").unwrap()
        .current_dir(repo.path())
        .args(["status", "--json"])
        .output()
        .unwrap();
    let status: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    for key in ["indexed_files", "total_chunks", "last_indexed", "estimated_stale", "watching"] {
        assert!(status.get(key).is_some(), "missing {key}");
    }
}

#[test]
fn context_command_prints_file_sections() {
    let repo = indexed_cli_fixture();
    assert_cmd::Command::cargo_bin("skelesearch").unwrap()
        .current_dir(repo.path())
        .args(["context", "src/lib.rs"])
        .assert()
        .success()
        .stdout(predicates::str::contains("imports").and(predicates::str::contains("imported_by")));
}

#[test]
fn clear_command_removes_local_index() {
    let repo = indexed_cli_fixture();
    assert_cmd::Command::cargo_bin("skelesearch").unwrap()
        .current_dir(repo.path())
        .args(["clear"])
        .assert()
        .success();

    let output = assert_cmd::Command::cargo_bin("skelesearch").unwrap()
        .current_dir(repo.path())
        .args(["status", "--json"])
        .output()
        .unwrap();
    let status: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(status["indexed_files"], 0);
}

#[test]
fn index_rejects_unknown_provider() {
    let mut cmd = assert_cmd::Command::cargo_bin("skelesearch").unwrap();
    cmd.args(["index", ".", "--provider", "definitely-not-a-provider"]).assert().failure();
}

#[test]
fn watch_command_sets_watching_state() {
    let repo = indexed_cli_fixture();
    let mut child = std::process::Command::new(assert_cmd::cargo::cargo_bin("skelesearch"))
        .current_dir(repo.path())
        .args(["watch", "."])
        .spawn()
        .unwrap();

    // Poll until the watcher reports itself as active (max 5s)
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    let mut watching = false;
    while std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(200));
        let output = assert_cmd::Command::cargo_bin("skelesearch").unwrap()
            .current_dir(repo.path())
            .args(["status", "--json"])
            .output()
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        if json["watching"] == true {
            watching = true;
            break;
        }
    }
    assert!(watching, "watcher did not become active within 5 seconds");
    let _ = child.kill();
    let _ = child.wait();
}

#[test]
fn watch_is_a_subcommand_not_an_index_flag() {
    let mut cmd = assert_cmd::Command::cargo_bin("skelesearch").unwrap();
    cmd.args(["index", ".", "--watch"]).assert().failure();
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p skelesearch-cli cli -- --nocapture`
Expected: FAIL because the binary is a stub and the command surface is not implemented.

- [ ] **Step 3: Implement the clap types and thin bootstrap**

```rust
#[derive(clap::Subcommand)]
enum Commands {
    Index { path: std::path::PathBuf, #[arg(long)] provider: Option<String> },
    Search { query: String, #[arg(long, default_value_t = 5)] top_k: u32, #[arg(long)] graph: bool, #[arg(long)] json: bool },
    Context { file: std::path::PathBuf },
    Status { path: Option<std::path::PathBuf>, #[arg(long)] json: bool },
    Clear { path: Option<std::path::PathBuf> },
    Watch { path: std::path::PathBuf, #[arg(long)] provider: Option<String> },
}
```

- [ ] **Step 4: Implement command handlers with real core calls**

Implementation notes:
- `index` and `watch` must select a provider explicitly, defaulting to fastembed.
- Accept only the documented provider values and reject unknown provider strings with a real error.
- `search --json` must emit the exact result fields callers rely on: `file_path`, `start_line`, `end_line`, `content`, `score`, `match_quality`, `why`.
- `status --json` is a hook-facing contract consumed by `session-start` and `post-edit-reindex`; it must appear in CLI help text so the shipped interface stays discoverable even though the high-level spec synopsis omitted flags.
- `context` must remain implemented and human-readable in v1; verify it prints chunk/import/importer sections for a known indexed file.
- `clear` must remove the local index for the requested path and leave `status --json` reporting an empty index afterward.
- `watch` must be its own subcommand per ADR-008.
- Provider selection should go through the provider-neutral types introduced in core, not ad-hoc string handling in the binary.

- [ ] **Step 5: Run targeted tests and one real smoke command**

Run:
- `cargo test -p skelesearch-cli cli -- --nocapture`
- `cargo run -p skelesearch-cli -- status --json`

Expected: PASS; clap shape matches the spec and the JSON commands expose the documented fields used by callers.

- [ ] **Step 6: Commit**

```bash
git add crates/cli/src/main.rs crates/cli/src/cli.rs crates/cli/src/app.rs crates/cli/tests/cli.rs
git commit -m "feat(cli): add skelesearch command surface"
```

### Task 7: Implement the rmcp stdio server and tool schemas

**Files:**
- Modify: `Cargo.toml`
- Modify: `crates/mcp/Cargo.toml`
- Modify: `crates/mcp/src/main.rs:1-9`
- Create: `crates/mcp/src/server.rs`
- Create: `crates/mcp/src/tools.rs`
- Create: `crates/mcp/tests/server.rs`

- [ ] **Step 1: Write the failing MCP tests**

Define any helpers (`test_server`, `fixture_repo_path`, `run_mcp_exchange`, JSON-RPC request builders) inside `crates/mcp/tests/server.rs` so the task stays self-contained.

```rust
#[tokio::test]
async fn list_tools_exposes_the_four_v1_tools() -> anyhow::Result<()> {
    let server = test_server().await?;
    let names = server.tool_names().await?;
    assert_eq!(names, vec!["get_file_context", "index_codebase", "index_status", "search_code"]);
    Ok(())
}

#[tokio::test]
async fn search_code_output_exposes_spec_fields() -> anyhow::Result<()> {
    let server = test_server().await?;
    let rows = server.search_code(SearchCodeInput {
        query: "import edges".into(),
        top_k: 3,
        include_graph: true,
    }).await?;
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
    let ctx = test_server().await?.get_file_context(GetFileContextInput { file_path: "missing.rs".into() }).await?;
    assert!(ctx.chunks.is_empty() && ctx.imports.is_empty() && ctx.imported_by.is_empty());
    Ok(())
}

#[tokio::test]
async fn index_codebase_returns_status_indexed_and_chunk_counts() -> anyhow::Result<()> {
    let out = test_server().await?.index_codebase(IndexCodebaseInput {
        path: fixture_repo_path()?.display().to_string(),
        provider: Some("fastembed".into()),
    }).await?;
    assert!(!out.status.is_empty());
    assert!(out.indexed > 0);
    assert!(out.chunks > 0);
    Ok(())
}

#[tokio::test]
async fn index_codebase_rejects_unknown_provider() -> anyhow::Result<()> {
    let err = test_server().await?.index_codebase(IndexCodebaseInput {
        path: fixture_repo_path()?.display().to_string(),
        provider: Some("definitely-not-a-provider".into()),
    }).await.unwrap_err();
    assert!(err.to_string().contains("provider"));
    Ok(())
}

#[tokio::test]
async fn index_status_exposes_estimated_stale_and_watching() -> anyhow::Result<()> {
    let server = test_server().await?;
    server.index_codebase(IndexCodebaseInput { path: fixture_repo_path()?.display().to_string(), provider: Some("fastembed".into()) }).await?;
    let status = server.index_status(IndexStatusInput { path: None }).await?;
    assert_eq!(status.estimated_stale, 0);
    assert!(!status.watching);
    assert!(status.last_indexed.as_ref().map(|s| chrono::DateTime::parse_from_rfc3339(s).is_ok()).unwrap_or(false));
    Ok(())
}

#[test]
fn server_stdio_speaks_json_rpc_without_stdout_logs() -> anyhow::Result<()> {
    let transcript = run_mcp_exchange("skelesearch-mcp", &[initialize_request(), list_tools_request()])?;
    assert!(transcript.stdout.starts_with("Content-Length:"));
    assert!(!transcript.stderr.contains("Content-Length:"));
    Ok(())
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p skelesearch-mcp server -- --nocapture`
Expected: FAIL because the server, tool schemas, and rmcp wiring do not exist.

- [ ] **Step 3: Implement tool structs and server state**

```rust
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct SearchCodeInput {
    pub query: String,
    #[serde(default = "default_top_k")]
    pub top_k: u32,
    #[serde(default)]
    pub include_graph: bool,
}

pub struct IndexStatusOutput {
    pub indexed_files: u32,
    pub total_chunks: u32,
    pub last_indexed: Option<String>,
    pub estimated_stale: u32,
    pub watching: bool,
}
```

Implementation notes:
- Define caller-visible output types for all four tools, not just `SearchCodeInput`.
- `index_codebase` must accept `path` plus optional `provider` and return `{status, indexed, chunks}`.
- The MCP `provider` field must flow through the same provider-neutral selection path the CLI uses and reject unknown providers clearly.
- `search_code` output must include `file_path`, line range, `content`, `score`, `match_quality`, and `why`.
- `index_status` must include `estimated_stale`, `watching`, and ISO8601 `last_indexed`.
- [ ] **Step 4: Implement rmcp bootstrap with stdout discipline**

```rust
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_writer(std::io::stderr).init();
    let config = ServerConfig::from_env_or_defaults()?;
    let server = SkeleSearchServer::new(config).await?;
    let peer = server.serve((tokio::io::stdin(), tokio::io::stdout())).await?;
    peer.waiting().await?;
    Ok(())
}
```

Implementation notes:
- Keep logs on stderr only; stdout is reserved for JSON-RPC.
- Expose exactly the four v1 tools from the spec.
- `search_code` results are candidates, not asserted truth; preserve `match_quality` and `why`.

- [ ] **Step 5: Run targeted tests and a real stdio smoke check**

Run:
- `cargo test -p skelesearch-mcp list_tools_exposes_the_four_v1_tools -- --exact`
- `cargo test -p skelesearch-mcp search_code_output_exposes_spec_fields -- --exact`
- `cargo test -p skelesearch-mcp get_file_context_returns_empty_arrays_for_unknown_file -- --exact`
- `cargo test -p skelesearch-mcp index_codebase_returns_status_indexed_and_chunk_counts -- --exact`
- `cargo test -p skelesearch-mcp index_codebase_rejects_unknown_provider -- --exact`
- `cargo test -p skelesearch-mcp index_status_exposes_estimated_stale_and_watching -- --exact`
- `cargo test -p skelesearch-mcp server_stdio_speaks_json_rpc_without_stdout_logs -- --exact`
- `cargo test -p skelesearch-mcp server -- --nocapture`

Expected: PASS; tool schemas match the spec, `index_codebase` and `get_file_context` keep their caller-visible contracts, invalid providers fail transparently, and the stdio server proves stdout/stderr discipline under a real JSON-RPC startup path.

- [ ] **Step 6: Commit**

```bash
git add crates/mcp/src/main.rs crates/mcp/src/server.rs crates/mcp/src/tools.rs crates/mcp/tests/server.rs
git commit -m "feat(mcp): add rmcp tool server"
```

---

## Chunk 3: Claude Plugin Assets, Packaging, and Acceptance

### Task 8: Add Claude Code plugin metadata, hooks, skill, and scout agent

**Files:**
- Modify: `Cargo.toml`
- Create: `.claude-plugin/plugin.json`
- Create: `hooks/hooks.json`
- Create: `hooks/session-start`
- Create: `hooks/post-edit-reindex`
- Create: `skills/search-code/SKILL.md`
- Create: `agents/skelesearch-scout.md`
- Create: `CLAUDE.md.template`

- [ ] **Step 1: Write the failing artifact checks**

Fail fast on missing assets:

```bash
test -f .claude-plugin/plugin.json
test -f hooks/hooks.json
test -x hooks/session-start
test -x hooks/post-edit-reindex
test -f skills/search-code/SKILL.md
test -f agents/skelesearch-scout.md
test -f CLAUDE.md.template
```

- [ ] **Step 2: Implement the metadata and hook wiring exactly as the spec defines**

Implementation notes:
- `Cargo.toml` becomes the single source of truth for plugin metadata; add the canonical author/homepage/repository values there first, then copy them into `plugin.json` so placeholders never ship.
- `plugin.json` is metadata only.
- `hooks/hooks.json` must wire `SessionStart` with matcher `startup|clear|compact` and `PostToolUse` with matcher `Write|Edit`.
- Hook command paths in `hooks/hooks.json` must use `${CLAUDE_PLUGIN_ROOT}` exactly as the spec shows.
- Hook script filenames must be extensionless.
- `session-start` must self-derive `PLUGIN_ROOT` before trusting `CLAUDE_PLUGIN_ROOT`.
- `session-start` must use `printf`, not a heredoc, when emitting hook JSON.
- `session-start` must cap `additionalContext` under 100 words.
- `post-edit-reindex` must skip work when `status --json` reports `watching: true`.
- The skill description in `skills/search-code/SKILL.md` must describe triggering conditions only, not workflow.
- The skill body, agent body, and `CLAUDE.md.template` must preserve the “results are candidates to verify” framing.
- `CLAUDE.md.template` must stay under 10 lines.

- [ ] **Step 3: Smoke-test the hook scripts directly**

Run:
- `PATH="$PWD/target/debug:$PATH" ./hooks/session-start`
- `PATH="$PWD/target/debug:$PATH" ./hooks/post-edit-reindex`

Expected:
- `session-start` exits 0 and either emits valid `hookSpecificOutput` JSON or no output when unconfigured.
- when `session-start` emits JSON, `additionalContext` stays under 100 words.
- `post-edit-reindex` exits 0 immediately when watch mode is active and otherwise spawns incremental indexing without blocking.

- [ ] **Step 4: Verify the generated text artifacts with machine-checked assertions**

Run:
- `python -m json.tool .claude-plugin/plugin.json`
- `python -m json.tool hooks/hooks.json`
- `python - <<'PY'
import json, tomllib
from pathlib import Path
cargo = tomllib.loads(Path('Cargo.toml').read_text())
plugin = json.loads(Path('.claude-plugin/plugin.json').read_text())
hooks = json.loads(Path('hooks/hooks.json').read_text())
skill = Path('skills/search-code/SKILL.md').read_text()
agent = Path('agents/skelesearch-scout.md').read_text()
template = Path('CLAUDE.md.template').read_text()
author = cargo['workspace']['package']['authors'][0]
assert plugin['author']['name'] == author.split(' <')[0]
assert plugin['author']['email'] == author.split('<')[1].rstrip('>')
assert plugin['homepage'] == cargo['workspace']['package']['homepage']
assert plugin['repository'] == cargo['workspace']['package']['repository']
assert hooks['hooks']['SessionStart'][0]['matcher'] == 'startup|clear|compact'
assert hooks['hooks']['SessionStart'][0]['hooks'][0]['command'] == '"${CLAUDE_PLUGIN_ROOT}/hooks/session-start"'
assert hooks['hooks']['SessionStart'][0]['hooks'][0]['async'] is False
assert hooks['hooks']['PostToolUse'][0]['matcher'] == 'Write|Edit'
assert hooks['hooks']['PostToolUse'][0]['hooks'][0]['command'] == '"${CLAUDE_PLUGIN_ROOT}/hooks/post-edit-reindex"'
assert hooks['hooks']['PostToolUse'][0]['hooks'][0]['async'] is True
assert 'Use when' in skill and 'CANDIDATES to verify' in skill
assert 'Never modify files' in agent
assert 'Results are candidates' in template
assert len(template.splitlines()) <= 10
PY`

Expected: JSON files parse cleanly, hook matchers/command strings/async flags match the spec, plugin metadata matches Cargo exactly, and the skill/agent/template preserve the required wording and size constraints.

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml .claude-plugin/plugin.json hooks/hooks.json hooks/session-start hooks/post-edit-reindex skills/search-code/SKILL.md agents/skelesearch-scout.md CLAUDE.md.template
git commit -m "feat(plugin): add claude plugin assets and hooks"
```

### Task 9: Add Nix packaging for CLI and MCP binaries

**Files:**
- Create: `flake.nix`

- [ ] **Step 1: Write the failing packaging check**

Run: `nix build .#skelesearch-cli`
Expected: FAIL because `flake.nix` does not exist yet.

- [ ] **Step 2: Implement the flake outputs**

```nix
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    crane.url = "github:ipetkov/crane";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, crane, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        craneLib = crane.lib.${system};
        src = craneLib.cleanCargoSource ./.;
        commonArgs = {
          inherit src;
          nativeBuildInputs = [ pkgs.cmake pkgs.pkg-config ];
        };
      in {
        packages.default = self.packages.${system}.skelesearch-cli;
        packages.mcp-server = self.packages.${system}.skelesearch-mcp;
        packages.skelesearch-cli = craneLib.buildPackage (commonArgs // { cargoExtraArgs = "-p skelesearch-cli --features skelesearch-core/storage-rocksdb"; });
        packages.skelesearch-mcp = craneLib.buildPackage (commonArgs // { cargoExtraArgs = "-p skelesearch-mcp --features skelesearch-core/storage-rocksdb"; });
        devShells.default = pkgs.mkShell {
          buildInputs = [ pkgs.rustc pkgs.cargo pkgs.cmake pkgs.pkg-config pkgs.clang ];
        };
      });
}
```

Implementation notes:
- Declare the `inputs` explicitly; the flake must be runnable from this chunk alone.
- Use `crane` for Rust builds.
- Keep SQLite as the dev default; do not force RocksDB into developer builds.
- Package outputs should build release binaries with `cargoExtraArgs = "-p <crate> --features skelesearch-core/storage-rocksdb"`.
- Ensure the dev shell includes cmake and a C++20-capable toolchain for optional RocksDB builds.
- Expose the named outputs the spec promises: `packages.default`, `packages.mcp-server`, `packages.skelesearch-cli`, and `packages.skelesearch-mcp`.

- [ ] **Step 3: Run the packaging checks**

Run:
- `nix flake show`
- `nix build .#skelesearch-cli`
- `nix build .#skelesearch-mcp`
- `nix build .#mcp-server`
- `nix develop -c cargo test -p skelesearch-core storage_contracts -- --nocapture`

Expected: `nix flake show` lists the named package outputs and both binaries build successfully, including the `mcp-server` alias; packaged release binaries use `storage-rocksdb`, while the dev shell keeps the default SQLite path for local iteration.

- [ ] **Step 4: Commit**

```bash
git add flake.nix
git commit -m "build: add nix flake packaging"
```

### Task 10: Run end-to-end acceptance coverage against the spec scenarios

**Files:**
- Modify: `crates/core/tests/fixtures/sample_repo/` as needed
- Modify: `crates/cli/tests/cli.rs`
- Modify: `crates/mcp/tests/server.rs`

- [ ] **Step 1: Add explicit acceptance tests for the v1 scenarios**

Focus on these checks from the spec:
- first-run indexing stores files/chunks and, for the large generated Rust fixture, reports `indexed_files >= 490`
- search returns relevant hits for “import edges”
- `get_file_context` returns imports/importers
- incremental reindex updates changed files and removes deleted files
- CLI `status --json` and MCP `index_status` stay aligned on counts
- SessionStart injects compact context under 100 words and completes within `< 100ms` on the reference setup
- PostToolUse async reindex updates status without blocking
- plugin install plus `.mcp.json` reaches the first successful `search_code` call within the acceptance window
- a per-project `.mcp.json` pointing at `skelesearch-mcp` is valid and sufficient to configure the MCP server

- [ ] **Step 2: Run the modified automated suites**

Run:
- `cargo test -p skelesearch-core`
- `cargo test -p skelesearch-cli`
- `cargo test -p skelesearch-mcp`

Expected: PASS; no mocks, real temp databases, real fixture repos, real command/server surfaces, and explicit hook-behavior assertions where the tests can own them.

- [ ] **Step 3: Run the full acceptance smoke sequence manually**

Run:
- `ROOT="$PWD" && TMPDIR="$(mktemp -d)" && cp -R . "$TMPDIR/repo" && (cd "$TMPDIR/repo" && PATH="$ROOT/target/debug:$PATH" "$ROOT/hooks/session-start")`
- `/usr/bin/time -p cargo run -p skelesearch-cli -- index crates/core`
- `cargo run -p skelesearch-cli -- search "import edges" --json`
- `cargo run -p skelesearch-cli -- context crates/core/src/searcher.rs`
- `cargo run -p skelesearch-cli -- status --json`
- `claude plugin install "$PWD"`
- `python - <<'PY'
import json, pathlib
pathlib.Path('.mcp.json').write_text(json.dumps({"mcpServers": {"skelesearch": {"command": "skelesearch-mcp", "args": []}}}, indent=2) + "\n")
PY`
- `python -m json.tool .mcp.json`
- `manual: start Claude in the project root and invoke search_code with query "import edges"`
- `rm .mcp.json`
- `python - <<'PY'
import subprocess, time
start = time.perf_counter()
subprocess.run(['./hooks/session-start'], check=True)
print((time.perf_counter() - start) * 1000)
PY`
- `touch crates/core/src/lib.rs && PATH="$PWD/target/debug:$PATH" ./hooks/post-edit-reindex`
- `cargo run -p skelesearch-cli -- status --json`

Expected:
- in an unindexed repo, `session-start` exits 0, spawns background indexing, and emits the fallback first-run context instead of failing.
- `/usr/bin/time -p` shows the index command meeting the applicable acceptance target for the chosen fixture, and `status --json` reports the indexed-file count truthfully.
- `search` returns JSON rows with `file_path`, line range, `score`, `match_quality`, and `why`.
- `context` returns chunks/imports/imported_by.
- `status` reports indexed counts, `estimated_stale`, and `watching`.
- `claude plugin install "$PWD"` succeeds without changing the hook wiring shape.
- `.mcp.json` matches the install shape from the spec and remains in place long enough for the first Claude `search_code` call.
- after plugin install plus `.mcp.json` configuration, the first `search_code` call succeeds in a Claude session for the project.
- the Python timing probe for `session-start` reports `< 100ms` on the reference setup.
- `post-edit-reindex` returns immediately and the changed file is reflected by `status --json` within 60 seconds.

- [ ] **Step 4: Commit**

```bash
git add crates/core/tests crates/cli/tests/cli.rs crates/mcp/tests/server.rs
git commit -m "test: add end-to-end acceptance coverage"
```

---

## Final Verification Checklist

Run these before claiming the implementation is complete:

```bash
cargo test -p skelesearch-core
cargo test -p skelesearch-embed-fastembed
cargo test -p skelesearch-cli
cargo test -p skelesearch-mcp
cargo run -p skelesearch-cli -- status --json
PATH="$PWD/target/debug:$PATH" ./hooks/session-start
PATH="$PWD/target/debug:$PATH" ./hooks/post-edit-reindex
nix flake show
nix build .#skelesearch-cli
nix build .#skelesearch-mcp
nix build .#mcp-server
```

Expected:
- all modified test suites pass
- CLI JSON output remains parseable and includes the documented fields
- MCP server passes a real stdio startup test without stdout pollution
- SessionStart emits either no output or valid hook JSON with `additionalContext` under 100 words
- PostToolUse remains async/fire-and-forget and does not block the caller
- flake outputs expose `packages.default`, `packages.mcp-server`, `packages.skelesearch-cli`, and `packages.skelesearch-mcp`
- no step introduces a compatibility shim, duplicate API, or legacy command surface

## Notes for the implementing agent
- Respect ADR-003: no SPLADE in v1.
- Respect ADR-008: `watch` is a subcommand, not a flag on `index`.
- Keep embedding dimensionality runtime-configurable; never hardcode 768 outside the fastembed default provider.
- Keep Cozo-specific details isolated to `schema.rs`; do not leak query strings into CLI or MCP crates.
- When adding tests a second time for the same pattern, extract shared temp-repo helpers instead of cloning setup code.
