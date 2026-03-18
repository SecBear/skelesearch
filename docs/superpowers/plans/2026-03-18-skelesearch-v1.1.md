# skelesearch v1.1 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden skelesearch for production use (streaming pipeline, batched writes, crash safety, concurrent access) and add table-stakes features (grep, 9 languages, config file, logging, GC) plus competitive differentiators (symbol search, multi-hop graph, search router).

**Architecture:** Phase 1 fixes production blockers sequentially in core (schema.rs, manifest.rs, indexer.rs). Phase 2 adds 4 independent feature tracks in parallel (grep, languages, config, logging+GC). Phase 3 adds 3 differentiators in parallel (symbols, graph traversal, router). Phase 4 is a polish batch. Each phase must complete and pass all tests before the next begins.

**Tech Stack:** Rust workspace, CozoDB 0.7.6, rusqlite (replacing `sqlite` crate for manifest), tree-sitter, text-splitter, fastembed-rs, rmcp 0.16, clap 4, tokio, tracing, regex, toml, fs2.

**Spec:** `docs/superpowers/specs/2026-03-18-skelesearch-v1.1-gaps.md`
**ADRs:** `DECISIONS.md` (ADR-001 through ADR-010)
**v1 baseline:** 42 tests passing across skelesearch-core (27), skelesearch-cli (7), skelesearch-mcp (7), plus 1 doctest.

---

## Execution Strategy

```
Phase 1: Tier 0 production blockers — sequential (with Tasks 1+2 parallelizable)
  Task 1: ManifestStore rusqlite migration + WAL + Mutex ──┐
  Task 2: Batched CozoDB multi-row :put ──────────────────┤
                                                           ├→ Task 3: Streaming pipeline
                                                           └→ Task 4: Crash-safe indexing

Phase 2: Tier 1 table stakes — 4 parallel agents
  Agent A: Task 5 — Regex/literal search (grep_code tool + CLI grep)
  Agent B: Task 6 — 9 new language configs
  Agent C: Task 7 — Config file + estimated_stale
  Agent D: Task 8 — CLI logging + GC + lock file

Phase 3: Tier 2 differentiators — 3 parallel agents
  Agent A: Task 9  — Symbol search
  Agent B: Task 10 — Multi-hop graph traversal (level-batched BFS)
  Agent C: Task 11 — Search strategy router (smart_search)

Phase 4: Tier 3 polish — single agent batch
  Task 12: Fix docs, tool descriptions, hooks, watch PID, chunker errors
```

## Verification Commands

After each phase completes, run:
```bash
cargo test -p skelesearch-core -p skelesearch-mcp -p skelesearch-cli 2>&1
cargo clippy --workspace -- -D warnings 2>&1
```

## New Workspace Dependencies

Add to `Cargo.toml` (workspace root) `[workspace.dependencies]`:
```toml
# Grep / regex (T1-1)
regex = "1"

# Config (T1-3)
toml = "0.8"

# Logging — move from mcp's direct deps to workspace
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

# File locking (T1-6 / P0-4b)
fs2 = "0.4"

# New tree-sitter grammars (T1-2) — verify exact versions against tree-sitter 0.26 on crates.io
tree-sitter-java = "0.23"
tree-sitter-c = "0.23"
tree-sitter-cpp = "0.23"
tree-sitter-ruby = "0.23"
tree-sitter-php = "0.23"
tree-sitter-c-sharp = "0.23"
tree-sitter-kotlin = "0.23"
tree-sitter-swift = "0.23"
tree-sitter-scala = "0.23"
```

Note: Tree-sitter grammar crate names and versions vary. The implementing agent MUST check crates.io for the actual published crate name and a version compatible with `tree-sitter = "0.26"`. Some may use `0.24` or different naming (e.g., `tree-sitter-c_sharp`). Adjust accordingly.

## File Structure

### New files
```
crates/core/src/grep.rs          — regex/literal file search (T1-1)
crates/core/src/config.rs        — .skelesearch.toml loading (T1-3)
crates/core/src/gc.rs            — index garbage collection (T1-5)
crates/core/src/symbols.rs       — symbol extraction + CozoDB relation (T2-1)
crates/core/src/router.rs        — search strategy router (T2-3)
crates/core/tests/grep.rs        — grep tests
crates/core/tests/config.rs      — config tests
crates/core/tests/gc.rs          — GC tests
crates/core/tests/symbols.rs     — symbol tests
crates/core/tests/router.rs      — router tests
crates/core/tests/crash_safety.rs — crash-safety tests (P0-3)
```

### Modified files
```
Cargo.toml                       — workspace deps (all phases)
crates/core/Cargo.toml           — swap sqlite→rusqlite, add deps per phase
crates/core/src/lib.rs           — pub mod for new modules
crates/core/src/manifest.rs      — rusqlite + WAL + Mutex + checkpoint table (P0-4, P0-3)
crates/core/src/schema.rs        — batched upserts, symbols relation (P0-2, T2-1)
crates/core/src/indexer.rs       — streaming pipeline, crash safety (P0-1, P0-3)
crates/core/src/searcher.rs      — multi-hop BFS (T2-2)
crates/core/src/chunker/mod.rs   — symbol extraction hook (T2-1)
crates/core/src/chunker/languages.rs — 9 new language configs (T1-2)
crates/core/src/provider.rs      — (no changes)
crates/core/tests/indexer.rs     — streaming pipeline tests
crates/core/tests/manifest_store.rs — WAL + concurrent access tests
crates/core/tests/storage_contracts.rs — batched upsert tests
crates/core/tests/searcher.rs    — multi-hop tests
crates/mcp/src/tools.rs          — grep_code, find_symbol, smart_search inputs/outputs
crates/mcp/src/server.rs         — new tool handlers, Arc<ManifestStore>, estimated_stale
crates/mcp/tests/server.rs       — tests for new tools
crates/cli/src/cli.rs            — grep, gc, symbol subcommands
crates/cli/src/app.rs            — new command handlers, tracing init, lock file
crates/cli/Cargo.toml            — add deps
crates/mcp/Cargo.toml            — move tracing to workspace ref
skills/search-code/SKILL.md      — fix "FAISS-backed" → CozoDB HNSW
hooks/session-start              — remove python3 dependency
```

---

## Chunk 1: Phase 1 — Production Blockers

### Task 1: ManifestStore → rusqlite + WAL + Mutex (P0-4a)

**Rationale:** The current `sqlite` crate produces a `!Send` Connection, forcing the MCP server into a `spawn_blocking` workaround. Switching to `rusqlite` (already a workspace dep, used internally by CozoDB) gives us a `Send` Connection. Wrapping in `Mutex<Connection>` makes ManifestStore `Send + Sync`. Adding WAL mode + busy_timeout eliminates SQLITE_BUSY under concurrent access.

**Files:**
- Modify: `crates/core/Cargo.toml` — replace `sqlite = "0.32"` with `rusqlite = { workspace = true }`
- Modify: `crates/core/src/manifest.rs` — rewrite internals using rusqlite + Mutex
- Test: `crates/core/tests/manifest_store.rs` — existing tests must pass, add concurrent access test

**Important:** The public API of ManifestStore does NOT change. All existing tests must pass unchanged.

- [ ] **Step 1: Update core Cargo.toml**

In `crates/core/Cargo.toml`, replace:
```toml
sqlite = "0.32"
```
with:
```toml
rusqlite = { workspace = true }
```

- [ ] **Step 2: Write concurrent-access test**

Add to `crates/core/tests/manifest_store.rs`:
```rust
#[test]
fn concurrent_manifest_access_no_busy_errors() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("manifest.db");

    // Open two ManifestStore instances to the same file
    let store1 = ManifestStore::open(&db_path).unwrap();
    let store2 = ManifestStore::open(&db_path).unwrap();

    // Interleave writes — should not get SQLITE_BUSY
    for i in 0..100 {
        let path = format!("file_{i}.rs");
        store1.upsert(&path, i as i64, 100, "hash_a").unwrap();
        store2.upsert(&path, i as i64, 200, "hash_b").unwrap();
    }

    // Both stores see the same final state
    let paths = store1.list_paths().unwrap();
    assert_eq!(paths.len(), 100);
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test -p skelesearch-core manifest_store -- --nocapture 2>&1`
Expected: compilation failure (`sqlite` crate removed, `rusqlite` not yet used in manifest.rs)

- [ ] **Step 4: Rewrite manifest.rs with rusqlite + Mutex**

Replace the entire `manifest.rs` implementation. Key changes:
- `use rusqlite::{Connection, params, OptionalExtension};`
- `use std::sync::Mutex;`
- `conn: Mutex<Connection>` instead of `conn: sqlite::Connection`
- Constructor sets WAL mode + busy_timeout:
  ```rust
  let conn = Connection::open(path.as_ref())?;
  conn.pragma_update(None, "journal_mode", "wal")?;
  conn.pragma_update(None, "busy_timeout", "5000")?;
  conn.execute_batch(
      "CREATE TABLE IF NOT EXISTS file_hashes (
          file_path TEXT PRIMARY KEY,
          mtime     INTEGER NOT NULL,
          size      INTEGER NOT NULL,
          xxhash3   TEXT    NOT NULL
      );"
  )?;
  Ok(Self { conn: Mutex::new(conn) })
  ```
- Each method acquires the lock: `let conn = self.conn.lock().map_err(|e| anyhow::anyhow!("manifest lock: {e}"))?;`
- `upsert` uses `conn.execute(sql, params![...])` instead of prepare+bind+next
- `is_unchanged` uses `conn.query_row(...).optional()?` pattern
- `list_paths` uses `conn.prepare(sql)?.query_map([], |row| row.get(0))?`
- `stale_paths_against` and `remove` follow the same pattern

Preserve the exact same public API:
```rust
pub fn open(path: impl AsRef<Path>) -> anyhow::Result<Self>
pub fn upsert(&self, file_path: &str, mtime: i64, size: i64, xxhash3: &str) -> anyhow::Result<()>
pub fn is_unchanged(&self, file_path: &str, mtime: i64, size: i64, xxhash3: &str) -> anyhow::Result<bool>
pub fn list_paths(&self) -> anyhow::Result<Vec<String>>
pub fn stale_paths_against(&self, visited: &HashSet<String>) -> anyhow::Result<Vec<String>>
pub fn remove(&self, file_path: &str) -> anyhow::Result<()>
```

- [ ] **Step 5: Run all manifest tests**

Run: `cargo test -p skelesearch-core manifest_store -- --nocapture 2>&1`
Expected: all tests pass including the new concurrent access test.

- [ ] **Step 6: Run full test suite to verify no regressions**

Run: `cargo test -p skelesearch-core -p skelesearch-mcp -p skelesearch-cli 2>&1`
Expected: 42+ tests pass (42 existing + 1 new concurrent test).

- [ ] **Step 7: Commit**

```bash
git add crates/core/Cargo.toml crates/core/src/manifest.rs crates/core/tests/manifest_store.rs
git commit -m "refactor(manifest): switch to rusqlite + WAL + Mutex for concurrent access (P0-4a)"
```

---

### Task 2: Batched CozoDB multi-row :put (P0-2)

**Rationale:** `upsert_chunks` currently issues one `:put` per chunk. 500k chunks = 500k transactions. CozoScript natively supports multi-row inserts: `<- [[r1], [r2], ...]`. Batch 500 rows per query.

**Files:**
- Modify: `crates/core/src/schema.rs` — rewrite `upsert_chunks` and `upsert_edges` to use multi-row `:put`
- Test: `crates/core/tests/storage_contracts.rs` — existing tests pass, add batch-size test

**Can run in parallel with Task 1** (different files, no shared changes).

- [ ] **Step 1: Write batch performance test**

Add to `crates/core/tests/storage_contracts.rs`:
```rust
#[tokio::test]
async fn upsert_chunks_batch_handles_500_chunks() -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let backend = CozoBackend::open(temp.path().join("index.db"))?;
    backend.initialize(8).await?;

    let chunks: Vec<ChunkRecord> = (0..500)
        .map(|i| ChunkRecord {
            file_path: "big_file.rs".into(),
            chunk_idx: i,
            content: format!("fn func_{i}() {{}}"),
            normalized: format!("fn func {i}"),
            chunk_type: "code".into(),
            start_line: i * 10 + 1,
            end_line: (i + 1) * 10,
            embedding: Some(vec![0.1; 8]),
        })
        .collect();

    // Should complete without error — previously would be 500 separate transactions
    backend.upsert_chunks(&chunks).await?;

    let stored = backend.get_chunks_for_file("big_file.rs").await?;
    assert_eq!(stored.len(), 500);
    Ok(())
}

#[tokio::test]
async fn upsert_edges_batch_handles_many_edges() -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let backend = CozoBackend::open(temp.path().join("index.db"))?;
    backend.initialize(8).await?;

    let edges: Vec<EdgeRecord> = (0..200)
        .map(|i| EdgeRecord {
            from_file: format!("src/mod_{i}.rs"),
            from_chunk: 0,
            to_file: "src/lib.rs".into(),
            edge_type: "imports".into(),
        })
        .collect();

    backend.upsert_edges(&edges).await?;

    let importers = backend.get_importers("src/lib.rs").await?;
    assert_eq!(importers.len(), 200);
    Ok(())
}
```

- [ ] **Step 2: Run tests to confirm they pass (slowly, as baseline)**

Run: `cargo test -p skelesearch-core storage_contracts -- --nocapture 2>&1`
Expected: tests pass but `upsert_chunks_batch_handles_500_chunks` may be slow (500 individual transactions).

- [ ] **Step 3: Rewrite upsert_chunks for multi-row :put**

In `crates/core/src/schema.rs`, replace the `upsert_chunks` implementation inside `impl StorageBackend for CozoBackend`. Key approach:

```rust
async fn upsert_chunks(&self, chunks: &[ChunkRecord]) -> anyhow::Result<()> {
    let dim = self.dim.load(Ordering::Relaxed);
    const BATCH_SIZE: usize = 500;

    for batch in chunks.chunks(BATCH_SIZE) {
        // Build multi-row data: <- [[row1], [row2], ...]
        let rows: Vec<Vec<DataValue>> = batch
            .iter()
            .map(|c| {
                vec![
                    Self::dv_str(&c.file_path),
                    Self::dv_int(c.chunk_idx as i64),
                    Self::dv_str(&c.content),
                    Self::dv_str(&c.normalized),
                    Self::dv_str(&c.chunk_type),
                    Self::dv_int(c.start_line as i64),
                    Self::dv_int(c.end_line as i64),
                    Self::embedding_to_dv(&c.embedding, dim),
                ]
            })
            .collect();

        let data = DataValue::List(
            rows.into_iter()
                .map(|r| DataValue::List(r))
                .collect(),
        );

        let mut p = BTreeMap::new();
        p.insert("rows".into(), data);

        self.run_mut(
            "?[file_path, chunk_idx, content, normalized, chunk_type, start_line, end_line, embedding] <- $rows \
             :put chunks { file_path, chunk_idx => content, normalized, chunk_type, start_line, end_line, embedding }",
            p,
        )?;
    }
    Ok(())
}
```

- [ ] **Step 4: Rewrite upsert_edges for multi-row :put**

Same pattern for `upsert_edges`:
```rust
async fn upsert_edges(&self, edges: &[EdgeRecord]) -> anyhow::Result<()> {
    let now = chrono::Utc::now().timestamp();
    const BATCH_SIZE: usize = 500;

    for batch in edges.chunks(BATCH_SIZE) {
        let rows: Vec<Vec<DataValue>> = batch
            .iter()
            .map(|e| {
                vec![
                    Self::dv_str(&e.from_file),
                    Self::dv_int(e.from_chunk as i64),
                    Self::dv_str(&e.to_file),
                    Self::dv_str(&e.edge_type),
                    Self::dv_int(now),
                ]
            })
            .collect();

        let data = DataValue::List(
            rows.into_iter()
                .map(|r| DataValue::List(r))
                .collect(),
        );

        let mut p = BTreeMap::new();
        p.insert("rows".into(), data);

        self.run_mut(
            "?[from_file, from_chunk, to_file, edge_type, created_at] <- $rows \
             :put code_edges { from_file, from_chunk, to_file => edge_type, created_at }",
            p,
        )?;
    }
    Ok(())
}
```

- [ ] **Step 5: Run all storage contract tests**

Run: `cargo test -p skelesearch-core storage_contracts -- --nocapture 2>&1`
Expected: all tests pass.

- [ ] **Step 6: Run full test suite**

Run: `cargo test -p skelesearch-core -p skelesearch-mcp -p skelesearch-cli 2>&1`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add crates/core/src/schema.rs crates/core/tests/storage_contracts.rs
git commit -m "perf(schema): batched multi-row CozoDB :put for chunks and edges (P0-2)"
```

---

### Task 3: Streaming indexing pipeline (P0-1)

**Rationale:** Indexer currently collects ALL chunk texts into `Vec<String>` before embedding. OOM on large repos. Fix: process files in bounded batches — walk → chunk → embed → upsert per batch, never holding more than `batch_size * avg_chunks` in memory.

**Depends on:** Task 2 (batched upserts make per-batch upserts efficient).

**Files:**
- Modify: `crates/core/src/indexer.rs` — restructure to process files in bounded batches
- Test: `crates/core/tests/indexer.rs` — verify embed_batch is called in bounded chunks

- [ ] **Step 1: Write streaming pipeline test**

Add to `crates/core/tests/indexer.rs` a test that verifies the indexer processes in bounded batches rather than collecting all texts upfront. Use a mock provider that tracks call sizes:

```rust
/// Provider that records the size of each embed_batch call.
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

#[async_trait::async_trait]
impl EmbedProvider for BatchTrackingProvider {
    fn dim(&self) -> usize { self.dim }
    async fn embed_batch(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
        self.call_sizes.lock().unwrap().push(texts.len());
        Ok(texts.iter().map(|_| vec![0.1; self.dim]).collect())
    }
}

#[tokio::test]
async fn indexer_embeds_in_bounded_batches() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let repo = dir.path().join("repo");
    std::fs::create_dir_all(&repo)?;

    // Create enough files that total chunks exceed batch_size
    for i in 0..20 {
        let content = format!("fn func_{i}() {{}}\nfn other_{i}() {{}}");
        std::fs::write(repo.join(format!("mod_{i}.rs")), content)?;
    }

    let idx_dir = dir.path().join("idx");
    std::fs::create_dir_all(&idx_dir)?;
    let backend = Arc::new(CozoBackend::open(idx_dir.join("index.db"))?);
    let manifest = Arc::new(ManifestStore::open(idx_dir.join("manifest.db"))?);
    let provider = BatchTrackingProvider::new(8);

    backend.initialize(8).await?;

    let indexer = Indexer::new(backend, manifest, provider);
    let result = indexer.index_path(&repo).await?;
    assert!(result.indexed_files > 0);

    // With batch_size=64 and ~20 files producing ~2 chunks each (~40 chunks total),
    // all should fit in one batch. But with 200 files producing 400+ chunks,
    // we'd see multiple calls. The key invariant: no single call exceeds batch_size.
    let sizes = indexer.provider().call_sizes();
    for &size in &sizes {
        assert!(size <= 64, "embed_batch call exceeded batch_size: {size}");
    }
    Ok(())
}
```

Note: The test needs `indexer.provider()` accessor. Add a `pub fn provider(&self) -> &P` method to `Indexer` if it doesn't exist.

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `cargo test -p skelesearch-core indexer -- --nocapture 2>&1`
Expected: may need `provider()` accessor added; test should pass once added since current batch_size=64 already batches the embed calls. The real fix is restructuring the pipeline so texts aren't collected upfront.

- [ ] **Step 3: Restructure indexer.rs for streaming pipeline**

Replace the current 5-phase approach with a file-batch approach:

```
// Current (P0-1 problem):
// Phase 1: Walk ALL files, chunk ALL, collect ALL work
// Phase 2: Delete old data for ALL re-indexed files
// Phase 3: Embed ALL texts in batches of batch_size
// Phase 4: Upsert ALL files, chunks, edges
// Phase 5: Reconcile deletions

// New (streaming):
// Phase 1: Walk ALL files, collect metadata only (path, mtime, size, hash, lang) — no content
// Phase 2: Process files in bounded batches of FILE_BATCH_SIZE:
//   For each file batch:
//     a. Read file content, chunk, extract edges
//     b. Delete old chunks/edges for these files
//     c. Embed this batch's texts
//     d. Upsert files, chunks, edges for this batch
//     e. Update manifest for this batch
// Phase 3: Reconcile deletions (same as before)
```

Key constants:
```rust
/// Maximum files processed in one pipeline batch.
const FILE_BATCH_SIZE: usize = 50;
```

The `batch_size` field on Indexer remains for controlling embed sub-batching within a file batch. The new `FILE_BATCH_SIZE` controls how many files' content is held in memory simultaneously.

Add a `pub fn provider(&self) -> &P` accessor to Indexer for test observability.

- [ ] **Step 4: Run all indexer tests**

Run: `cargo test -p skelesearch-core indexer -- --nocapture 2>&1`
Expected: all tests pass.

- [ ] **Step 5: Run full test suite**

Run: `cargo test -p skelesearch-core -p skelesearch-mcp -p skelesearch-cli 2>&1`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add crates/core/src/indexer.rs crates/core/tests/indexer.rs
git commit -m "perf(indexer): streaming file-batch pipeline for bounded memory (P0-1)"
```

---

### Task 4: Crash-safe indexing (P0-3)

**Rationale:** Delete-then-embed-then-upsert risks data loss on crash. Fix: add a checkpoint table to the manifest SQLite that records batch intent before processing. On restart, incomplete batches are re-processed.

**Depends on:** Task 1 (manifest uses rusqlite), Task 3 (pipeline is batch-structured).

**Files:**
- Modify: `crates/core/src/manifest.rs` — add `index_progress` table and checkpoint methods
- Modify: `crates/core/src/indexer.rs` — integrate checkpoint writes around each batch
- Create: `crates/core/tests/crash_safety.rs` — test checkpoint recovery

- [ ] **Step 1: Write crash safety tests**

Create `crates/core/tests/crash_safety.rs`:
```rust
use skelesearch_core::{CozoBackend, ManifestStore, Indexer};
use std::sync::Arc;

// A provider that returns zero vectors (no model needed)
struct ZeroProvider(usize);
#[async_trait::async_trait]
impl skelesearch_core::EmbedProvider for ZeroProvider {
    fn dim(&self) -> usize { self.0 }
    async fn embed_batch(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|_| vec![0.0; self.0]).collect())
    }
}

#[tokio::test]
async fn checkpoint_table_created_on_open() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let manifest = ManifestStore::open(dir.path().join("manifest.db"))?;
    // Verify checkpoint methods exist and don't error on empty state
    let incomplete = manifest.find_incomplete_batches()?;
    assert!(incomplete.is_empty());
    Ok(())
}

#[tokio::test]
async fn incomplete_batch_detected_after_simulated_crash() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let manifest = ManifestStore::open(dir.path().join("manifest.db"))?;

    // Simulate: a batch was started but never completed
    manifest.begin_batch("run_001", 0, &["src/a.rs", "src/b.rs"])?;

    // On "restart", find_incomplete_batches should return this batch
    let incomplete = manifest.find_incomplete_batches()?;
    assert_eq!(incomplete.len(), 1);
    assert_eq!(incomplete[0].files, vec!["src/a.rs", "src/b.rs"]);

    // After completing it, no more incomplete batches
    manifest.complete_batch("run_001", 0)?;
    let incomplete = manifest.find_incomplete_batches()?;
    assert!(incomplete.is_empty());
    Ok(())
}

#[tokio::test]
async fn completed_files_not_reindexed_on_restart() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let repo = dir.path().join("repo");
    std::fs::create_dir_all(&repo)?;
    std::fs::write(repo.join("a.rs"), "fn a() {}")?;
    std::fs::write(repo.join("b.rs"), "fn b() {}")?;

    let idx_dir = dir.path().join("idx");
    std::fs::create_dir_all(&idx_dir)?;

    let backend = Arc::new(CozoBackend::open(idx_dir.join("index.db"))?);
    let manifest = Arc::new(ManifestStore::open(idx_dir.join("manifest.db"))?);
    let provider = ZeroProvider(8);

    backend.initialize(8).await?;
    let indexer = Indexer::new(Arc::clone(&backend), Arc::clone(&manifest), provider);

    // First index: both files indexed
    let r1 = indexer.index_path(&repo).await?;
    assert_eq!(r1.indexed_files, 2);

    // Second index: nothing changed, zero files indexed
    let r2 = indexer.index_path(&repo).await?;
    assert_eq!(r2.indexed_files, 0);
    Ok(())
}
```

- [ ] **Step 2: Run tests to verify compilation and expected failures**

Run: `cargo test -p skelesearch-core crash_safety -- --nocapture 2>&1`
Expected: compilation failure (missing `begin_batch`, `complete_batch`, `find_incomplete_batches` methods).

- [ ] **Step 3: Add checkpoint table and methods to ManifestStore**

Add to `manifest.rs`:

New table in the constructor's `execute_batch`:
```sql
CREATE TABLE IF NOT EXISTS index_progress (
    run_id    TEXT    NOT NULL,
    batch_idx INTEGER NOT NULL,
    files     TEXT    NOT NULL,
    status    TEXT    NOT NULL DEFAULT 'pending',
    created_at INTEGER NOT NULL,
    PRIMARY KEY (run_id, batch_idx)
);
```

New public struct and methods:
```rust
#[derive(Debug, Clone)]
pub struct IncompleteBatch {
    pub run_id: String,
    pub batch_idx: i64,
    pub files: Vec<String>,
}

impl ManifestStore {
    pub fn begin_batch(&self, run_id: &str, batch_idx: usize, files: &[&str]) -> anyhow::Result<()> { ... }
    pub fn complete_batch(&self, run_id: &str, batch_idx: usize) -> anyhow::Result<()> { ... }
    pub fn find_incomplete_batches(&self) -> anyhow::Result<Vec<IncompleteBatch>> { ... }
    pub fn clear_completed_batches(&self, run_id: &str) -> anyhow::Result<()> { ... }
}
```

- `begin_batch`: INSERT with status='pending', files as JSON array, created_at as unix timestamp.
- `complete_batch`: UPDATE status='complete' WHERE run_id AND batch_idx.
- `find_incomplete_batches`: SELECT WHERE status='pending'.
- `clear_completed_batches`: DELETE WHERE run_id AND status='complete'.

- [ ] **Step 4: Integrate checkpoints into indexer pipeline**

In `indexer.rs`, wrap each file-batch with checkpoint writes:
```rust
// Before processing batch:
let file_paths: Vec<&str> = batch_files.iter().map(|f| f.rel_path.as_str()).collect();
self.manifest.begin_batch(&run_id, batch_idx, &file_paths)?;

// ... process batch (chunk, embed, upsert) ...

// After successful batch:
self.manifest.complete_batch(&run_id, batch_idx)?;
```

At the start of `index_path`, check for incomplete batches and re-process:
```rust
let incomplete = self.manifest.find_incomplete_batches()?;
for batch in &incomplete {
    // Re-process these files by including them in the work list
    // even if their manifest hash hasn't changed
}
```

Generate `run_id` as a timestamp-based string: `format!("run_{}", chrono::Utc::now().timestamp_millis())`.

- [ ] **Step 5: Run crash safety tests**

Run: `cargo test -p skelesearch-core crash_safety -- --nocapture 2>&1`
Expected: all 3 tests pass.

- [ ] **Step 6: Run full test suite**

Run: `cargo test -p skelesearch-core -p skelesearch-mcp -p skelesearch-cli 2>&1`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add crates/core/src/manifest.rs crates/core/src/indexer.rs crates/core/tests/crash_safety.rs
git commit -m "feat(core): crash-safe indexing with checkpoint table (P0-3)"
```

---

## Chunk 2: Phase 2 — Table Stakes (4 Parallel Agents)

**Prerequisites:** All Phase 1 tasks complete. Full test suite passes.

Each agent works on independent files. Merge conflicts are limited to additive changes in `core/lib.rs`, `Cargo.toml` files, and `core/Cargo.toml`.

### Task 5: Regex/literal search — grep_code MCP tool + CLI grep (T1-1)

**Agent A.** New `grep_code` MCP tool and `grep` CLI subcommand. Uses `ignore` (already a dep) for file walking + `regex` crate for pattern matching. Returns results in the same `SearchResult` shape with `why: "grep"`.

**Files:**
- Modify: `Cargo.toml` — add `regex = "1"` to workspace deps
- Modify: `crates/core/Cargo.toml` — add `regex = { workspace = true }`
- Create: `crates/core/src/grep.rs` — file-walking regex search
- Modify: `crates/core/src/lib.rs` — add `pub mod grep;` and re-export
- Create: `crates/core/tests/grep.rs` — unit tests
- Modify: `crates/mcp/src/tools.rs` — add `GrepCodeInput`, `GrepCodeRow`
- Modify: `crates/mcp/src/server.rs` — add `grep_code` tool handler
- Modify: `crates/mcp/tests/server.rs` — test grep_code tool
- Modify: `crates/cli/src/cli.rs` — add `Grep` subcommand
- Modify: `crates/cli/src/app.rs` — add `run_grep` handler

- [ ] **Step 1: Write grep core tests**

Create `crates/core/tests/grep.rs`:
```rust
use skelesearch_core::grep::{grep_codebase, GrepOptions, GrepMatch};
use std::path::Path;

#[test]
fn grep_finds_exact_string() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("main.rs"), "fn main() {\n    println!(\"hello\");\n}").unwrap();

    let results = grep_codebase(dir.path(), "println", &GrepOptions::default()).unwrap();
    assert_eq!(results.len(), 1);
    assert!(results[0].line_content.contains("println"));
    assert_eq!(results[0].line_number, 2);
}

#[test]
fn grep_finds_regex_pattern() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("lib.rs"), "fn foo_bar() {}\nfn baz_qux() {}").unwrap();

    let results = grep_codebase(dir.path(), r"fn \w+_bar", &GrepOptions::default()).unwrap();
    assert_eq!(results.len(), 1);
    assert!(results[0].line_content.contains("foo_bar"));
}

#[test]
fn grep_respects_gitignore() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join(".gitignore"), "target/\n").unwrap();
    std::fs::create_dir_all(dir.path().join("target")).unwrap();
    std::fs::write(dir.path().join("target/gen.rs"), "fn generated() {}").unwrap();
    std::fs::write(dir.path().join("src.rs"), "fn source() {}").unwrap();

    let results = grep_codebase(dir.path(), "fn ", &GrepOptions::default()).unwrap();
    assert_eq!(results.len(), 1);
    assert!(results[0].file_path.contains("src.rs"));
}

#[test]
fn grep_limits_results() {
    let dir = tempfile::tempdir().unwrap();
    let content: String = (0..100).map(|i| format!("fn func_{i}() {{}}\n")).collect();
    std::fs::write(dir.path().join("big.rs"), content).unwrap();

    let opts = GrepOptions { max_results: 5, ..Default::default() };
    let results = grep_codebase(dir.path(), "fn func_", &opts).unwrap();
    assert_eq!(results.len(), 5);
}
```

- [ ] **Step 2: Implement grep.rs**

Create `crates/core/src/grep.rs`:
```rust
use std::path::Path;
use regex::Regex;
use ignore::WalkBuilder;

#[derive(Debug, Clone)]
pub struct GrepMatch {
    pub file_path: String,
    pub line_number: usize,
    pub line_content: String,
}

#[derive(Debug, Clone)]
pub struct GrepOptions {
    pub max_results: usize,
    pub case_insensitive: bool,
}

impl Default for GrepOptions {
    fn default() -> Self {
        Self { max_results: 50, case_insensitive: false }
    }
}

pub fn grep_codebase(root: &Path, pattern: &str, opts: &GrepOptions) -> anyhow::Result<Vec<GrepMatch>> {
    let re = if opts.case_insensitive {
        Regex::new(&format!("(?i){pattern}"))?
    } else {
        Regex::new(pattern)?
    };

    let mut results = Vec::new();
    let walker = WalkBuilder::new(root).build();

    for entry in walker {
        if results.len() >= opts.max_results { break; }
        let entry = entry?;
        if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) { continue; }

        let content = match std::fs::read_to_string(entry.path()) {
            Ok(c) => c,
            Err(_) => continue, // skip binary/unreadable files
        };

        let rel_path = entry.path()
            .strip_prefix(root)
            .unwrap_or(entry.path())
            .to_string_lossy()
            .to_string();

        for (i, line) in content.lines().enumerate() {
            if results.len() >= opts.max_results { break; }
            if re.is_match(line) {
                results.push(GrepMatch {
                    file_path: rel_path.clone(),
                    line_number: i + 1,
                    line_content: line.to_string(),
                });
            }
        }
    }
    Ok(results)
}
```

- [ ] **Step 3: Add pub mod grep to lib.rs and re-export**

Add to `crates/core/src/lib.rs`:
```rust
pub mod grep;
```

- [ ] **Step 4: Run grep tests**

Run: `cargo test -p skelesearch-core grep -- --nocapture 2>&1`
Expected: all 4 tests pass.

- [ ] **Step 5: Add grep_code MCP tool**

Add to `crates/mcp/src/tools.rs`:
```rust
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GrepCodeInput {
    /// Regex or literal pattern to search for.
    pub pattern: String,
    /// Directory to search (defaults to project root).
    pub path: Option<String>,
    /// Maximum results (default: 50).
    #[serde(default = "default_grep_max")]
    pub max_results: usize,
    /// Case-insensitive search.
    #[serde(default)]
    pub case_insensitive: bool,
}

fn default_grep_max() -> usize { 50 }

#[derive(Debug, Serialize, JsonSchema)]
pub struct GrepCodeRow {
    pub file_path: String,
    pub line_number: usize,
    pub line_content: String,
    pub why: String,
}
```

Add to `server.rs`: `grep_code` method that calls `grep_codebase()` and maps results to `GrepCodeRow` with `why: "grep"`.

Add MCP tool declaration in the `#[tool_router]` block:
```rust
#[tool(name = "grep_code")]
async fn mcp_grep_code(&self, Parameters(input): Parameters<GrepCodeInput>) -> Result<String, String> { ... }
```

- [ ] **Step 6: Add grep CLI subcommand**

In `crates/cli/src/cli.rs`, add to `Commands` enum:
```rust
/// Search files for a regex or literal pattern.
Grep {
    /// Regex pattern to search for.
    pattern: String,
    /// Directory to search (defaults to current directory).
    path: Option<std::path::PathBuf>,
    /// Maximum number of results.
    #[arg(long, default_value_t = 50)]
    max_results: usize,
    /// Case-insensitive matching.
    #[arg(short, long)]
    ignore_case: bool,
    /// Output as JSON.
    #[arg(long)]
    json: bool,
},
```

In `app.rs`, add `run_grep` handler.

- [ ] **Step 7: Add MCP server test for grep_code**

In `crates/mcp/tests/server.rs`:
```rust
#[tokio::test]
async fn grep_code_finds_pattern_in_indexed_files() { ... }
```

- [ ] **Step 8: Run full test suite**

Run: `cargo test -p skelesearch-core -p skelesearch-mcp -p skelesearch-cli 2>&1`
Expected: all tests pass.

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "feat(core): add grep_code regex search tool and CLI grep subcommand (T1-1)"
```

---

### Task 6: Language expansion — 9 new LanguageConfig impls (T1-2)

**Agent B.** Add Java, C, C++, Ruby, PHP, C#, Kotlin, Swift, Scala to the language registry.

**Files:**
- Modify: `Cargo.toml` — add 9 tree-sitter grammar workspace deps
- Modify: `crates/core/Cargo.toml` — add 9 grammar deps
- Modify: `crates/core/src/chunker/languages.rs` — add 9 LanguageConfig impls + registry entries
- Modify: `crates/core/src/indexer.rs` — extend `language_for()` match arms
- Create: `crates/core/tests/languages.rs` — per-language chunk + import tests

**Important:** Check crates.io for exact crate names and versions compatible with `tree-sitter = "0.26"`. Some grammars may use different names (e.g., `tree-sitter-c-sharp` vs `tree-sitter-csharp`). If a grammar crate is not published or incompatible, skip it and document why.

- [ ] **Step 1: Write per-language tests**

Create `crates/core/tests/languages.rs` with one test per language:
```rust
use skelesearch_core::Chunker;

fn assert_chunks_and_imports(filename: &str, source: &str, expect_chunks: bool, expect_imports: bool) {
    let chunker = Chunker::default();
    let chunks = chunker.chunk_file(filename, source).unwrap();
    assert!(!chunks.is_empty(), "expected chunks for {filename}");
    if expect_chunks {
        assert!(chunks.iter().any(|c| c.chunk_type == "code"), "expected code chunks for {filename}");
    }
    if expect_imports {
        let edges = chunker.extract_edges(filename, source).unwrap();
        assert!(!edges.is_empty(), "expected import edges for {filename}");
    }
}

#[test]
fn java_chunks_and_imports() {
    assert_chunks_and_imports("Main.java",
        "import java.util.List;\n\npublic class Main {\n    public static void main(String[] args) {\n        System.out.println(\"hello\");\n    }\n}",
        true, true);
}

#[test]
fn c_chunks() {
    assert_chunks_and_imports("main.c",
        "#include <stdio.h>\n\nint main() {\n    printf(\"hello\");\n    return 0;\n}",
        true, true);
}

#[test]
fn cpp_chunks() {
    assert_chunks_and_imports("main.cpp",
        "#include <iostream>\n\nint main() {\n    std::cout << \"hello\";\n    return 0;\n}",
        true, true);
}

#[test]
fn ruby_chunks() {
    assert_chunks_and_imports("app.rb",
        "require 'json'\n\ndef greet(name)\n  puts \"Hello #{name}\"\nend",
        true, true);
}

// Similar tests for PHP, C#, Kotlin, Swift, Scala...
```

- [ ] **Step 2: Add grammar deps and implement configs**

For each language, add:
1. Grammar crate to workspace `Cargo.toml` and `crates/core/Cargo.toml`
2. A `struct XxxConfig` implementing `LanguageConfig` in `languages.rs`
3. Registry entry in `config_for_extension()`
4. Extension mapping in `language_for()` in `indexer.rs`

Each config needs:
- `file_extensions()` — e.g., `&["java"]`
- `language()` — e.g., `tree_sitter_java::LANGUAGE.into()`
- `chunk_node_kinds()` — e.g., `&["method_declaration", "class_declaration"]`
- `import_query()` — e.g., `"(import_declaration) @path"` for Java

Research the correct tree-sitter node kinds for each language by checking the grammar's `node-types.json` or running `tree-sitter parse` on sample files.

- [ ] **Step 3: Run language tests**

Run: `cargo test -p skelesearch-core languages -- --nocapture 2>&1`
Expected: all per-language tests pass.

- [ ] **Step 4: Run full test suite**

Run: `cargo test -p skelesearch-core -p skelesearch-mcp -p skelesearch-cli 2>&1`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(chunker): add 9 new language configs — Java, C, C++, Ruby, PHP, C#, Kotlin, Swift, Scala (T1-2)"
```

---

### Task 7: Config file + estimated_stale (T1-3 + T1-7)

**Agent C.** Add `.skelesearch.toml` loading and fix the hardcoded `estimated_stale: 0`.

**Files:**
- Modify: `Cargo.toml` — add `toml = "0.8"` to workspace deps
- Modify: `crates/core/Cargo.toml` — add `toml = { workspace = true }`
- Create: `crates/core/src/config.rs` — TOML config loading
- Modify: `crates/core/src/lib.rs` — add `pub mod config;`
- Modify: `crates/core/src/manifest.rs` — add `count_stale` method
- Modify: `crates/mcp/src/server.rs` — use `count_stale` for `estimated_stale`
- Modify: `crates/cli/src/app.rs` — load config, use for index and status
- Create: `crates/core/tests/config.rs` — config parsing tests
- Modify: `crates/core/tests/manifest_store.rs` — test count_stale

- [ ] **Step 1: Write config parsing tests**

Create `crates/core/tests/config.rs`:
```rust
use skelesearch_core::config::Config;

#[test]
fn parse_minimal_config() {
    let toml = "";
    let config = Config::from_str(toml).unwrap();
    assert_eq!(config.index.batch_size, 64); // default
    assert_eq!(config.index.provider, "fastembed"); // default
}

#[test]
fn parse_full_config() {
    let toml = r#"
[index]
provider = "fastembed"
batch_size = 128
exclude = ["vendor/", "*.generated.*"]

[search]
default_top_k = 10
"#;
    let config = Config::from_str(toml).unwrap();
    assert_eq!(config.index.batch_size, 128);
    assert_eq!(config.index.exclude, vec!["vendor/", "*.generated.*"]);
    assert_eq!(config.search.default_top_k, 10);
}

#[test]
fn config_loads_from_project_root() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join(".skelesearch.toml"), "[index]\nbatch_size = 256\n").unwrap();
    let config = Config::load(dir.path()).unwrap();
    assert_eq!(config.index.batch_size, 256);
}

#[test]
fn config_returns_defaults_when_no_file() {
    let dir = tempfile::tempdir().unwrap();
    let config = Config::load(dir.path()).unwrap();
    assert_eq!(config.index.batch_size, 64);
}
```

- [ ] **Step 2: Implement config.rs**

```rust
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Config {
    pub index: IndexConfig,
    pub search: SearchConfig,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct IndexConfig {
    pub provider: String,
    pub batch_size: usize,
    pub exclude: Vec<String>,
    pub index_dir: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct SearchConfig {
    pub default_top_k: usize,
}

impl Default for Config { ... } // provider="fastembed", batch_size=64, default_top_k=5
impl Default for IndexConfig { ... }
impl Default for SearchConfig { ... }

impl Config {
    pub fn from_str(s: &str) -> anyhow::Result<Self> {
        Ok(toml::from_str(s)?)
    }

    pub fn load(project_root: &Path) -> anyhow::Result<Self> {
        let path = project_root.join(".skelesearch.toml");
        if path.exists() {
            let content = std::fs::read_to_string(&path)?;
            Self::from_str(&content)
        } else {
            Ok(Self::default())
        }
    }
}
```

- [ ] **Step 3: Add count_stale to ManifestStore**

Add to `manifest.rs`:
```rust
/// Count files where stored mtime differs from current filesystem mtime.
/// This is an O(n) scan of the manifest — fast for typical repo sizes.
pub fn count_stale(&self, root: &Path) -> anyhow::Result<usize> {
    let conn = self.conn.lock().map_err(|e| anyhow::anyhow!("lock: {e}"))?;
    let mut stmt = conn.prepare("SELECT file_path, mtime FROM file_hashes")?;
    let mut count = 0usize;
    let mut rows = stmt.query([])?;
    while let Some(row) = rows.next()? {
        let file_path: String = row.get(0)?;
        let stored_mtime: i64 = row.get(1)?;
        let abs_path = root.join(&file_path);
        if let Ok(meta) = std::fs::metadata(&abs_path) {
            let current_mtime = meta.modified().ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0);
            if current_mtime != stored_mtime {
                count += 1;
            }
        } else {
            // File deleted — counts as stale
            count += 1;
        }
    }
    Ok(count)
}
```

- [ ] **Step 4: Write estimated_stale test**

Add to `crates/core/tests/manifest_store.rs`:
```rust
#[test]
fn count_stale_detects_modified_files() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("repo");
    std::fs::create_dir_all(&root).unwrap();

    // Create files and index them
    std::fs::write(root.join("a.rs"), "fn a() {}").unwrap();
    std::fs::write(root.join("b.rs"), "fn b() {}").unwrap();

    let manifest = ManifestStore::open(dir.path().join("manifest.db")).unwrap();
    manifest.upsert("a.rs", 1000, 10, "hash_a").unwrap();
    manifest.upsert("b.rs", 1000, 10, "hash_b").unwrap();

    // Touch one file (change mtime)
    std::fs::write(root.join("a.rs"), "fn a_changed() {}").unwrap();

    let stale = manifest.count_stale(&root).unwrap();
    assert!(stale >= 1, "expected at least 1 stale file, got {stale}");
}
```

- [ ] **Step 5: Wire estimated_stale into MCP server and CLI**

In `server.rs` `index_status`:
```rust
// Replace hardcoded 0:
estimated_stale: 0,
// With:
estimated_stale: self.manifest().map(|m| m.count_stale(&root).unwrap_or(0)).unwrap_or(0),
```

This requires the server to know the project root. Add a `root_path: Arc<PathBuf>` field to `SkeleSearchServer`.

In `cli/app.rs` `run_status`, similarly call `manifest.count_stale(&root)`.

- [ ] **Step 6: Run all tests**

Run: `cargo test -p skelesearch-core -p skelesearch-mcp -p skelesearch-cli 2>&1`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat(core): add .skelesearch.toml config file and fix estimated_stale (T1-3, T1-7)"
```

---

### Task 8: CLI logging + GC + lock file (T1-4 + T1-5 + T1-6)

**Agent D.** Add tracing-based logging, `gc` command, and lock file for index concurrency.

**Files:**
- Modify: `Cargo.toml` — add `tracing`, `tracing-subscriber`, `fs2` to workspace deps
- Modify: `crates/core/Cargo.toml` — add `tracing`, `fs2`
- Modify: `crates/cli/Cargo.toml` — add `tracing`, `tracing-subscriber`, `fs2`
- Modify: `crates/mcp/Cargo.toml` — switch tracing/tracing-subscriber to workspace refs
- Create: `crates/core/src/gc.rs` — garbage collection logic
- Modify: `crates/core/src/lib.rs` — add `pub mod gc;`
- Modify: `crates/core/src/indexer.rs` — add `#[instrument]` on key methods
- Modify: `crates/cli/src/cli.rs` — add `Gc` subcommand, `--verbose` global flag
- Modify: `crates/cli/src/app.rs` — tracing init, lock file, gc handler
- Create: `crates/core/tests/gc.rs` — GC tests

- [ ] **Step 1: Write GC tests**

Create `crates/core/tests/gc.rs`:
```rust
use skelesearch_core::gc::collect_garbage;
use skelesearch_core::{CozoBackend, ManifestStore, ChunkRecord, FileRecord};
use std::sync::Arc;

#[tokio::test]
async fn gc_removes_orphaned_chunks() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let idx = dir.path().join("idx");
    std::fs::create_dir_all(&idx)?;

    let backend = Arc::new(CozoBackend::open(idx.join("index.db"))?);
    let manifest = Arc::new(ManifestStore::open(idx.join("manifest.db"))?);
    backend.initialize(8).await?;

    // Index a file via the backend
    backend.upsert_file(&FileRecord {
        file_path: "gone.rs".into(), language: "rust".into(),
        last_modified: 100, last_indexed: 100, chunk_count: 1,
    }).await?;
    backend.upsert_chunks(&[ChunkRecord {
        file_path: "gone.rs".into(), chunk_idx: 0,
        content: "fn gone() {}".into(), normalized: "fn gone".into(),
        chunk_type: "code".into(), start_line: 1, end_line: 1,
        embedding: Some(vec![0.1; 8]),
    }]).await?;

    // File doesn't exist on disk — GC should remove it
    let root = dir.path().join("repo");
    std::fs::create_dir_all(&root)?; // empty repo
    manifest.upsert("gone.rs", 100, 10, "hash")?;

    let removed = collect_garbage(&root, &backend, &manifest).await?;
    assert_eq!(removed, 1);

    // Verify it's gone from backend
    let chunks = backend.get_chunks_for_file("gone.rs").await?;
    assert!(chunks.is_empty());
    Ok(())
}
```

- [ ] **Step 2: Implement gc.rs**

```rust
use std::path::Path;
use std::sync::Arc;
use crate::{ManifestStore, StorageBackend};

/// Remove index entries for files that no longer exist on disk.
/// Returns the number of files removed.
pub async fn collect_garbage<B: StorageBackend>(
    root: &Path,
    backend: &Arc<B>,
    manifest: &Arc<ManifestStore>,
) -> anyhow::Result<usize> {
    let indexed_paths = manifest.list_paths()?;
    let mut removed = 0usize;

    for file_path in &indexed_paths {
        let abs_path = root.join(file_path);
        if !abs_path.exists() {
            backend.delete_chunks_for_file(file_path).await?;
            backend.delete_edges_for_file(file_path).await?;
            backend.delete_file(file_path).await?;
            manifest.remove(file_path)?;
            removed += 1;
        }
    }
    Ok(removed)
}
```

- [ ] **Step 3: Add tracing instrumentation to indexer**

In `crates/core/src/indexer.rs`, add `use tracing::{info, debug, instrument, warn};` and annotate:
```rust
#[instrument(skip(self), fields(root = %root.display()))]
pub async fn index_path(&self, root: &Path) -> anyhow::Result<IndexResult> {
    // ... existing code, with:
    debug!(file = %fw.rel_path, chunks = chunks.len(), "chunked file");
    info!(files = result.indexed_files, chunks = result.total_chunks, "indexing complete");
    warn!(file = %rel_path, "parse failed, using fallback chunker");
    // etc.
}
```

- [ ] **Step 4: Add --verbose flag and tracing init to CLI**

In `cli.rs`, add global args to `Cli`:
```rust
pub struct Cli {
    /// Verbosity: -v for info, -vv for debug, -vvv for trace.
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    pub verbose: u8,
    #[command(subcommand)]
    pub command: Commands,
}
```

Add `Gc` subcommand to `Commands`:
```rust
/// Remove index entries for deleted files.
Gc {
    path: Option<std::path::PathBuf>,
},
```

In `app.rs`, init tracing at the top of `run()`:
```rust
let level = match cli.verbose {
    0 => "warn",
    1 => "info",
    2 => "debug",
    _ => "trace",
};
tracing_subscriber::fmt()
    .with_env_filter(tracing_subscriber::EnvFilter::new(level))
    .with_writer(std::io::stderr)
    .init();
```

- [ ] **Step 5: Add lock file to index and watch commands**

In `app.rs`, before `run_index` does any work:
```rust
use fs2::FileExt;

fn acquire_lock(dir: &Path) -> anyhow::Result<std::fs::File> {
    let lock_path = dir.join(".skelesearch.lock");
    let file = std::fs::OpenOptions::new().create(true).write(true).open(&lock_path)?;
    match file.try_lock_exclusive() {
        Ok(()) => Ok(file),
        Err(_) => anyhow::bail!("Another skelesearch process is running (lock held)"),
    }
}
```

Call `let _lock = acquire_lock(&dir)?;` at the start of `run_index` and `run_watch`. The lock is released when `_lock` is dropped.

- [ ] **Step 6: Run all tests**

Run: `cargo test -p skelesearch-core -p skelesearch-mcp -p skelesearch-cli 2>&1`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat(cli): add tracing logging, gc command, and lock file (T1-4, T1-5, T1-6)"
```

---

## Chunk 3: Phase 3 — Differentiators + Phase 4 — Polish

**Prerequisites:** All Phase 1 and Phase 2 tasks complete. Full test suite passes.

### Task 9: Symbol search (T2-1)

**Agent A.** Extract symbol definitions during chunking, store in new CozoDB `symbols` relation, expose via `find_symbol` MCP tool and CLI subcommand.

**Files:**
- Create: `crates/core/src/symbols.rs` — symbol extraction from tree-sitter AST
- Modify: `crates/core/src/lib.rs` — add `pub mod symbols;`
- Modify: `crates/core/src/schema.rs` — add `symbols` relation, `upsert_symbols`, `find_symbols`
- Modify: `crates/core/src/chunker/mod.rs` — call symbol extraction during chunking
- Modify: `crates/core/src/indexer.rs` — upsert symbols alongside chunks
- Modify: `crates/mcp/src/tools.rs` — add `FindSymbolInput`, `SymbolRow`
- Modify: `crates/mcp/src/server.rs` — add `find_symbol` tool
- Modify: `crates/cli/src/cli.rs` — add `Symbol` subcommand
- Modify: `crates/cli/src/app.rs` — add `run_symbol` handler
- Create: `crates/core/tests/symbols.rs` — symbol tests

- [ ] **Step 1: Write symbol extraction tests**

Create `crates/core/tests/symbols.rs`:
```rust
use skelesearch_core::symbols::{extract_symbols, SymbolDef};

#[test]
fn extract_rust_symbols() {
    let source = "pub struct Foo {}\nfn bar() {}\nimpl Foo { fn baz(&self) {} }";
    let symbols = extract_symbols("lib.rs", source).unwrap();
    let names: Vec<&str> = symbols.iter().map(|s| s.name.as_str()).collect();
    assert!(names.contains(&"Foo"));
    assert!(names.contains(&"bar"));
    assert!(names.contains(&"baz"));
}

#[test]
fn extract_python_symbols() {
    let source = "class MyClass:\n    def method(self):\n        pass\n\ndef free_func():\n    pass";
    let symbols = extract_symbols("app.py", source).unwrap();
    let names: Vec<&str> = symbols.iter().map(|s| s.name.as_str()).collect();
    assert!(names.contains(&"MyClass"));
    assert!(names.contains(&"method"));
    assert!(names.contains(&"free_func"));
}
```

- [ ] **Step 2: Implement symbols.rs**

```rust
use crate::chunker::languages::config_for_extension;
use tree_sitter::{Parser, Query, QueryCursor};
use streaming_iterator::StreamingIterator;

#[derive(Debug, Clone)]
pub struct SymbolDef {
    pub file_path: String,
    pub name: String,
    pub kind: String,      // "function", "struct", "class", "method", "trait", "enum"
    pub start_line: usize,
    pub end_line: usize,
}

pub fn extract_symbols(filename: &str, source: &str) -> anyhow::Result<Vec<SymbolDef>> {
    let ext = std::path::Path::new(filename)
        .extension().and_then(|e| e.to_str()).unwrap_or("");
    let cfg = match config_for_extension(ext) {
        Some(c) => c,
        None => return Ok(vec![]),
    };

    let mut parser = Parser::new();
    parser.set_language(&cfg.language())?;
    let tree = parser.parse(source.as_bytes(), None)
        .ok_or_else(|| anyhow::anyhow!("parse failed for {filename}"))?;

    let mut symbols = Vec::new();
    // Walk AST nodes matching chunk_node_kinds, extract their name child
    collect_symbols(&tree.root_node(), source, filename, cfg.chunk_node_kinds(), &mut symbols);
    Ok(symbols)
}

fn collect_symbols(node: &tree_sitter::Node, source: &str, filename: &str,
                   kinds: &[&str], out: &mut Vec<SymbolDef>) {
    if kinds.contains(&node.kind()) {
        // Look for a name child (typically "name" or "identifier")
        if let Some(name_node) = node.child_by_field_name("name") {
            if let Ok(name) = name_node.utf8_text(source.as_bytes()) {
                out.push(SymbolDef {
                    file_path: filename.to_string(),
                    name: name.to_string(),
                    kind: normalize_kind(node.kind()),
                    start_line: node.start_position().row + 1,
                    end_line: node.end_position().row + 1,
                });
            }
        }
    }
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i) {
            collect_symbols(&child, source, filename, kinds, out);
        }
    }
}
```

- [ ] **Step 3: Add symbols relation to schema.rs**

In `initialize()`, after creating `code_edges`:
```rust
self.run_mut_ignore(
    ":create symbols { file_path: String, name: String => kind: String, start_line: Int, end_line: Int }",
    "already exists",
)?;
```

Add `StorageBackend` trait methods:
```rust
async fn upsert_symbols(&self, symbols: &[SymbolDef]) -> anyhow::Result<()>;
async fn delete_symbols_for_file(&self, file_path: &str) -> anyhow::Result<()>;
async fn find_symbols(&self, name: &str, kind: Option<&str>) -> anyhow::Result<Vec<SymbolDef>>;
```

Use batched multi-row `:put` (same pattern as Task 2).

- [ ] **Step 4: Wire symbol extraction into indexer**

In `indexer.rs`, after chunking each file:
```rust
let symbols = symbols::extract_symbols(&fw.rel_path, &source).unwrap_or_default();
// ... later, after upsert_chunks:
backend.delete_symbols_for_file(&fw.rel_path).await?;
backend.upsert_symbols(&symbols).await?;
```

- [ ] **Step 5: Add find_symbol MCP tool and CLI subcommand**

MCP tool `find_symbol` with input `{ name: String, kind: Option<String> }`.
CLI subcommand `symbol <name> [--kind <kind>]`.

- [ ] **Step 6: Run all tests**

Run: `cargo test -p skelesearch-core -p skelesearch-mcp -p skelesearch-cli 2>&1`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat(core): add symbol search with CozoDB symbols relation (T2-1)"
```

---

### Task 10: Multi-hop graph traversal — level-batched BFS (T2-2)

**Agent B.** Replace one-hop `augment_with_graph` with configurable-depth level-batched BFS. Pattern from skg-state-cozo's `traverse()`.

**Files:**
- Modify: `crates/core/src/schema.rs` — add `traverse_imports` method to StorageBackend
- Modify: `crates/core/src/searcher.rs` — replace `augment_with_graph` with BFS
- Modify: `crates/mcp/src/tools.rs` — add `max_depth` to `SearchCodeInput`
- Modify: `crates/core/tests/searcher.rs` — multi-hop tests

- [ ] **Step 1: Write multi-hop traversal tests**

Add to `crates/core/tests/searcher.rs`:
```rust
#[tokio::test]
async fn two_hop_traversal_finds_transitive_imports() -> anyhow::Result<()> {
    // Setup: A imports B, B imports C
    // Search hits A. With depth=2, graph results should include B and C.
    let dir = tempfile::tempdir()?;
    let backend = Arc::new(CozoBackend::open(dir.path().join("index.db"))?);
    backend.initialize(8).await?;

    // Create files A, B, C with chunks
    for name in ["a.rs", "b.rs", "c.rs"] {
        backend.upsert_file(&FileRecord {
            file_path: name.into(), language: "rust".into(),
            last_modified: 100, last_indexed: 100, chunk_count: 1,
        }).await?;
        backend.upsert_chunks(&[ChunkRecord {
            file_path: name.into(), chunk_idx: 0,
            content: format!("// {name}"), normalized: name.into(),
            chunk_type: "code".into(), start_line: 1, end_line: 1,
            embedding: Some(vec![0.1; 8]),
        }]).await?;
    }

    // Edges: a→b, b→c
    backend.upsert_edges(&[
        EdgeRecord { from_file: "a.rs".into(), from_chunk: 0, to_file: "b.rs".into(), edge_type: "imports".into() },
        EdgeRecord { from_file: "b.rs".into(), from_chunk: 0, to_file: "c.rs".into(), edge_type: "imports".into() },
    ]).await?;

    // Traverse from a.rs with depth=2 should find b.rs and c.rs
    let neighbors = backend.traverse_imports("a.rs", 2).await?;
    assert!(neighbors.contains(&"b.rs".to_string()));
    assert!(neighbors.contains(&"c.rs".to_string()));
    Ok(())
}

#[tokio::test]
async fn traverse_handles_cycles() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let backend = Arc::new(CozoBackend::open(dir.path().join("index.db"))?);
    backend.initialize(8).await?;

    // a→b, b→a (cycle)
    for name in ["a.rs", "b.rs"] {
        backend.upsert_file(&FileRecord {
            file_path: name.into(), language: "rust".into(),
            last_modified: 100, last_indexed: 100, chunk_count: 1,
        }).await?;
    }
    backend.upsert_edges(&[
        EdgeRecord { from_file: "a.rs".into(), from_chunk: 0, to_file: "b.rs".into(), edge_type: "imports".into() },
        EdgeRecord { from_file: "b.rs".into(), from_chunk: 0, to_file: "a.rs".into(), edge_type: "imports".into() },
    ]).await?;

    // Should not infinite loop, should return just ["b.rs"]
    let neighbors = backend.traverse_imports("a.rs", 5).await?;
    assert_eq!(neighbors, vec!["b.rs".to_string()]);
    Ok(())
}
```

- [ ] **Step 2: Add traverse_imports to StorageBackend and CozoBackend**

Add to `StorageBackend` trait:
```rust
async fn traverse_imports(&self, file_path: &str, max_depth: usize) -> anyhow::Result<Vec<String>>;
```

Implement in CozoBackend using level-batched BFS (from skg-state-cozo pattern):
```rust
async fn traverse_imports(&self, file_path: &str, max_depth: usize) -> anyhow::Result<Vec<String>> {
    if max_depth == 0 { return Ok(vec![]); }

    let mut visited: HashSet<String> = HashSet::new();
    visited.insert(file_path.to_string());
    let mut frontier = vec![file_path.to_string()];
    let mut result: Vec<String> = Vec::new();

    for _ in 0..max_depth {
        if frontier.is_empty() { break; }

        let frontier_dv = DataValue::List(
            frontier.iter().map(|k| Self::dv_str(k)).collect()
        );
        let mut p = BTreeMap::new();
        p.insert("frontier".into(), frontier_dv);

        let rows = self.run_imm(
            "?[to_file] := *code_edges[from_file, _, to_file, _, _], is_in(from_file, $frontier)",
            p,
        )?;

        frontier.clear();
        for row in &rows.rows {
            if let Ok(to_file) = Self::str_col(&row[0]) {
                if !visited.contains(&to_file) {
                    visited.insert(to_file.clone());
                    result.push(to_file.clone());
                    frontier.push(to_file);
                }
            }
        }
    }
    Ok(result)
}
```

Don't forget to add the Arc blanket impl delegation.

- [ ] **Step 3: Update searcher to use traverse_imports**

Replace `augment_with_graph` in `searcher.rs`:
- Accept a `max_depth` parameter (default 2 instead of 1)
- Call `backend.traverse_imports(file_path, max_depth)` for each file in primary results
- Collect neighbor chunks with `why: "imports <path> (depth N)"`

Update the `search` method signature to accept `max_depth: usize`.

- [ ] **Step 4: Update MCP tool to accept max_depth**

In `tools.rs`, add to `SearchCodeInput`:
```rust
/// Maximum graph traversal depth (default: 2). Set to 0 to disable graph augmentation.
#[serde(default = "default_max_depth")]
pub max_depth: usize,
```

- [ ] **Step 5: Run all tests**

Run: `cargo test -p skelesearch-core -p skelesearch-mcp -p skelesearch-cli 2>&1`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(search): multi-hop level-batched BFS graph traversal (T2-2)"
```

---

### Task 11: Search strategy router — smart_search (T2-3)

**Agent C.** New `smart_search` MCP tool that analyzes the query and routes to grep or semantic search.

**Files:**
- Create: `crates/core/src/router.rs` — query classification + routing
- Modify: `crates/core/src/lib.rs` — add `pub mod router;`
- Modify: `crates/mcp/src/tools.rs` — add `SmartSearchInput`, `SmartSearchOutput`
- Modify: `crates/mcp/src/server.rs` — add `smart_search` tool
- Create: `crates/core/tests/router.rs` — routing tests

- [ ] **Step 1: Write routing classification tests**

Create `crates/core/tests/router.rs`:
```rust
use skelesearch_core::router::{classify_query, QueryStrategy};

#[test]
fn literal_string_routes_to_grep() {
    assert_eq!(classify_query("ERR_INVALID_HANDLE"), QueryStrategy::Grep);
}

#[test]
fn regex_pattern_routes_to_grep() {
    assert_eq!(classify_query(r"fn \w+_test"), QueryStrategy::Grep);
}

#[test]
fn file_path_routes_to_grep() {
    assert_eq!(classify_query("src/main.rs"), QueryStrategy::Grep);
}

#[test]
fn natural_language_routes_to_semantic() {
    assert_eq!(classify_query("how does the authentication system work"), QueryStrategy::Semantic);
}

#[test]
fn short_identifier_routes_to_grep() {
    assert_eq!(classify_query("StorageBackend"), QueryStrategy::Grep);
}

#[test]
fn question_routes_to_semantic() {
    assert_eq!(classify_query("where is the database connection configured"), QueryStrategy::Semantic);
}
```

- [ ] **Step 2: Implement router.rs**

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum QueryStrategy {
    Grep,
    Semantic,
}

/// Classify a query as grep-appropriate or semantic-appropriate.
///
/// Heuristics:
/// - Contains regex special chars (except common English punctuation): Grep
/// - Looks like a file path (contains / or \, ends with extension): Grep
/// - Single token, looks like an identifier (PascalCase, snake_case, SCREAMING_CASE): Grep
/// - Contains 4+ words, question words, or natural language connectors: Semantic
/// - Short phrases without special chars: Semantic (safer default)
pub fn classify_query(query: &str) -> QueryStrategy {
    let trimmed = query.trim();

    // Regex special chars (beyond what appears in normal English)
    if trimmed.contains('\\') || trimmed.contains('[') || trimmed.contains('^')
       || trimmed.contains('$') || trimmed.contains('+') || trimmed.contains('|') {
        return QueryStrategy::Grep;
    }

    // File path pattern
    if trimmed.contains('/') || trimmed.contains('\\')
       || trimmed.ends_with(".rs") || trimmed.ends_with(".py")
       || trimmed.ends_with(".ts") || trimmed.ends_with(".js") {
        return QueryStrategy::Grep;
    }

    let words: Vec<&str> = trimmed.split_whitespace().collect();

    // Single-token identifier (no spaces, contains _ or has mixed case)
    if words.len() == 1 {
        let w = words[0];
        if w.contains('_') || (w.chars().any(|c| c.is_uppercase()) && w.chars().any(|c| c.is_lowercase())) {
            return QueryStrategy::Grep;
        }
        // ALL_CAPS
        if w.len() > 2 && w.chars().all(|c| c.is_uppercase() || c == '_') {
            return QueryStrategy::Grep;
        }
    }

    // Natural language indicators
    let nl_words = ["how", "what", "where", "why", "when", "does", "is", "are", "the", "this"];
    if words.len() >= 3 && words.iter().any(|w| nl_words.contains(&w.to_lowercase().as_str())) {
        return QueryStrategy::Semantic;
    }

    // Default: 3+ words → semantic, fewer → grep
    if words.len() >= 3 {
        QueryStrategy::Semantic
    } else {
        QueryStrategy::Grep
    }
}
```

- [ ] **Step 3: Add smart_search MCP tool**

In `tools.rs`:
```rust
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SmartSearchInput {
    pub query: String,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default)]
    pub include_graph: bool,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct SmartSearchOutput {
    pub strategy: String, // "grep" or "semantic"
    pub results: Vec<serde_json::Value>, // SearchCodeRow or GrepCodeRow
}
```

In `server.rs`, the `smart_search` handler calls `classify_query()`, then dispatches to either `grep_code()` or `search_code()`.

- [ ] **Step 4: Run all tests**

Run: `cargo test -p skelesearch-core -p skelesearch-mcp -p skelesearch-cli 2>&1`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(search): add smart_search strategy router (T2-3)"
```

---

### Task 12: Tier 3 polish batch (T3-1 through T3-7)

**Single agent.** Quick fixes for docs, tool descriptions, hooks, watch PID, and chunker error visibility.

**Files:**
- Modify: `skills/search-code/SKILL.md` — fix "FAISS-backed" → "CozoDB HNSW-indexed"
- Modify: `crates/mcp/src/server.rs` — improve MCP tool descriptions with examples
- Modify: `hooks/session-start` — remove python3 dependency, use shell-native JSON
- Modify: `crates/cli/src/app.rs` — fix `process_is_alive` to use `libc::kill(pid, 0)` instead of spawning a process
- Modify: `crates/core/src/indexer.rs` — log warnings on chunk parse failures, track count
- Modify: `crates/core/src/schema.rs` — add `parse_errors` to `IndexStats`

- [ ] **Step 1: Fix SKILL.md**

In `skills/search-code/SKILL.md`, replace any occurrence of "FAISS-backed" or "FAISS" with "CozoDB HNSW-indexed".

- [ ] **Step 2: Improve MCP tool descriptions**

In the `#[tool_router]` block in `server.rs`, update tool doc comments:
```rust
/// Semantic and full-text hybrid search over the indexed codebase.
///
/// Returns ranked code chunks matching the query. Results include file path,
/// line numbers, content, match quality label, and retrieval provenance.
///
/// Example queries:
/// - "how does authentication work" (semantic)
/// - "database connection pool" (semantic)
///
/// Results are candidates for context, not guaranteed answers.
/// Always verify by reading the full file.
#[tool(name = "search_code")]
```

Similar improvements for `index_codebase`, `index_status`, `get_file_context`, `grep_code`, `find_symbol`, `smart_search`.

- [ ] **Step 3: Fix session-start hook**

Replace python3 JSON parsing with shell-native approach:
```bash
# Before (requires python3):
# status=$(skelesearch status --json | python3 -c "import json,sys; ...")

# After (shell-native):
status_json=$(skelesearch status --json 2>/dev/null)
if [ $? -eq 0 ] && [ -n "$status_json" ]; then
    indexed=$(echo "$status_json" | grep -o '"indexed_files":[0-9]*' | cut -d: -f2)
    # ...
fi
```

- [ ] **Step 4: Fix watch PID check**

In `app.rs`, replace the process-spawning `process_is_alive`:
```rust
#[cfg(unix)]
fn process_is_alive(pid: u32) -> bool {
    // kill(pid, 0) checks if process exists without sending a signal
    unsafe { libc::kill(pid as i32, 0) == 0 }
}
```

Add `libc` as a dev/optional dependency, or use the nix crate, or use `std::process::Command::new("kill")` but check it properly. The simplest portable approach that doesn't need a new dep:
```rust
#[cfg(unix)]
fn process_is_alive(pid: u32) -> bool {
    std::path::Path::new(&format!("/proc/{pid}")).exists()
        || std::process::Command::new("kill")
            .args(["-0", &pid.to_string()])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
}
```

- [ ] **Step 5: Add chunker error visibility**

In `indexer.rs`, replace:
```rust
let chunks = chunker.chunk_file(&rel_path, &source).unwrap_or_default();
```
with:
```rust
let chunks = match chunker.chunk_file(&rel_path, &source) {
    Ok(c) => c,
    Err(e) => {
        tracing::warn!(file = %rel_path, error = %e, "chunk parse failed, skipping");
        parse_errors += 1;
        continue;
    }
};
```

Add `parse_errors` counter to `IndexResult`:
```rust
pub struct IndexResult {
    pub indexed_files: usize,
    pub deleted_files: usize,
    pub total_chunks: usize,
    pub parse_errors: usize,
}
```

Add `estimated_stale` and `parse_errors` to `IndexStats` if not already present.

- [ ] **Step 6: Run full test suite**

Run: `cargo test -p skelesearch-core -p skelesearch-mcp -p skelesearch-cli 2>&1`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "fix(polish): SKILL.md docs, tool descriptions, hooks, PID check, chunker errors (T3)"
```

---

## Final Verification

After all 12 tasks complete:

```bash
# Full test suite
cargo test -p skelesearch-core -p skelesearch-mcp -p skelesearch-cli 2>&1

# Clippy
cargo clippy --workspace -- -D warnings 2>&1

# Build release
cargo build --release -p skelesearch-cli -p skelesearch-mcp 2>&1

# Smoke test: index this repo and search
./target/release/skelesearch index . -vv
./target/release/skelesearch search "storage backend trait"
./target/release/skelesearch grep "StorageBackend"
./target/release/skelesearch status --json
./target/release/skelesearch gc
```

Expected test count: ~70+ (42 existing + ~30 new across all tasks).
