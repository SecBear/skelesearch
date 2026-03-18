# skelesearch v1.2 Production Readiness Plan

**Status:** In Progress
**Created:** 2026-03-18
**Context:** Comprehensive review identified 6 critical bugs, 7 high-severity issues, and 25 additional improvements needed before skelesearch is production-ready as a memory/code-search layer for agentic systems.

## Architecture Decisions (from research)

### AD-1: RRF Fusion Moves to Rust
CozoDB Datalog lacks window functions (no RANK/ROW_NUMBER). The current in-Datalog RRF formula `1/(60 + 1/(bm25+0.001))` inverts BM25 signal. **Fix:** Run two separate CozoDB queries (FTS + HNSW), collect results in Rust, sort to assign 1-indexed ranks, apply weighted RRF: `score(d) = 0.55/(60+rank_bm25) + 0.45/(60+rank_vec)`. BM25 weighted higher for code (exact identifiers matter). Missing-list sentinel = 1000.

### AD-2: HNSW Parameters
Current m=50, ef_construction=20 is inverted. **Fix:** m=32, ef_construction=128, ef_search=64. Consensus from OpenSearch portfolio learning, Qdrant defaults, and Malkov & Yashunin 2018.

### AD-3: Import Graph Strategy
Full import resolution across 15 languages is not worth the cost. LocAgent (ACL 2025) ablation shows import edges alone provide minimal retrieval value — invoke/inherit edges matter more. **Phase 0:** Disable dead import graph (edges stored but never match file paths). **Future:** Implement aider-style identifier-based dependency graph (extract symbol defs+refs per file, link by identifier co-occurrence). For TS/JS: `oxc_resolver` crate. For Rust: DIY module walker (~300 lines).

### AD-4: MCP Server Persistent Storage
Store index at `{project_root}/.skelesearch/` (same as CLI). Project root discovery: 3-tier fallback — (1) tool argument with absolute path, (2) MCP `roots/list` capability, (3) inherited cwd + `.git` walk-up. CLI and MCP share same CozoDB SQLite DB with WAL mode + busy_timeout=5000ms.

### AD-5: Streaming Indexing Pipeline
4-stage bounded-channel pipeline: walker/reader --cap=64--> chunker --cap=16--> embedder --cap=8--> upserter. `tokio::task::spawn_blocking` for CPU stages (not rayon). Memory ceiling ~1.7MB O(1). Batch accumulation in embedder stage (EMBED_BATCH_SIZE=64).

---

## Phase 0: Critical Correctness Bugs

### 0.1: Fix MCP server ephemeral storage
- **Files:** `crates/mcp/src/main.rs`, `crates/mcp/src/server.rs`, `crates/mcp/Cargo.toml`
- **Change:** Replace `tempfile::tempdir()` with persistent `.skelesearch/` in project root. Accept `--project-root` CLI arg or discover via cwd/.git walk-up. Remove `tempfile` from non-dev deps. Add `ServerInfo` name+version.
- **Tests:** Existing MCP tests must still pass. Add test that index persists across server restarts.

### 0.2: Fix RRF hybrid search formula
- **Files:** `crates/core/src/schema.rs`, `crates/core/src/searcher.rs`
- **Change:** Split `hybrid_search` in schema.rs into `fts_search` (FTS-only query returning chunk_id + bm25_score) and `vector_search` (HNSW-only query returning chunk_id + cosine_distance). Add `rank_fuse_rrf()` function in searcher.rs that: (1) calls both queries, (2) sorts each to assign 1-indexed ranks, (3) applies weighted RRF with k=60, w_bm25=0.55, w_vec=0.45, (4) returns merged+sorted results. The `why` field should indicate which retrieval paths contributed ("vector", "fts", or "hybrid").
- **Tests:** Update searcher tests. Add test that verifies BM25-only matches and vector-only matches both appear in hybrid results with correct relative ordering.

### 0.3: Fix HNSW parameters
- **Files:** `crates/core/src/schema.rs`
- **Change:** In HNSW creation: `m: 32, ef_construction: 128`. In HNSW queries: `ef: 64`. Make ef_search configurable via SearchConfig.
- **Tests:** Existing search tests must pass (re-create index with new params).

### 0.4: Disable import graph (honest about broken state)
- **Files:** `crates/core/src/searcher.rs`, `crates/core/src/chunker/mod.rs`
- **Change:** In searcher.rs, disable graph augmentation (the edges never match file paths, so traverse_imports always returns empty — the feature is dead code producing zero value). Set `include_graph` to no-op with a doc comment explaining why and referencing the planned identifier-based approach. Keep edge extraction code in chunker but add `// TODO(v2): edges store raw import text, not resolved paths — see AD-3` comment.
- **Tests:** Remove or update graph augmentation tests to reflect disabled state.

### 0.5: Wire symbol extraction into indexer
- **Files:** `crates/core/src/indexer.rs`, `crates/core/src/gc.rs`
- **Change:** After chunking each file in the indexer pipeline, call `extract_symbols(lang, source)` and `backend.upsert_symbols(file_path, symbols)`. In gc.rs, add `backend.delete_symbols_for_file(path)` alongside existing chunk/edge/file deletion.
- **Tests:** Add integration test: index a file with known symbols, verify `find_symbols` returns them. Verify GC removes symbols when file deleted.

### 0.6: Add embedding count assertion
- **Files:** `crates/core/src/indexer.rs`
- **Change:** After `embed_batch()`, assert `embeddings.len() == texts.len()`. On mismatch, return error (not silent zero vectors).
- **Tests:** Add test with a provider that returns wrong-length embeddings, verify error.

### 0.7: Skip binary files in indexer
- **Files:** `crates/core/src/indexer.rs`
- **Change:** After reading file content, check for null bytes in first 8KB. If found, skip with `tracing::debug!` log. Do NOT pass through `String::from_utf8_lossy`.
- **Tests:** Add test with a binary file in the fixture repo, verify it's skipped (0 chunks indexed).

---

## Phase 1: Production Hardening (P0 blockers)

### 1.1: Streaming indexing pipeline
- **Files:** `crates/core/src/indexer.rs` (major rewrite)
- **Change:** Replace collect-all-then-batch with 4-stage bounded-channel pipeline per AD-5. Stage types: ReadFile, ChunkedFile, UpsertBatch. Decreasing channel capacities 64→16→8. `tokio::task::spawn_blocking` for chunking and embedding.
- **Tests:** Existing indexer tests must pass. Add test for 1000-file repo staying under 10MB RSS.

### 1.2: Batched CozoDB multi-row upserts
- **Files:** `crates/core/src/schema.rs`
- **Change:** Replace single-row `:put` loops with multi-row `:put` using `$rows` parameter. Use CozoDB chained queries for atomic file+chunks+edges upsert. Batch size ~100 rows.
- **Tests:** Existing storage_contracts tests must pass. Add benchmark comparing single-row vs batched.

### 1.3: Crash-safe indexing with recovery
- **Files:** `crates/core/src/indexer.rs`, `crates/core/src/manifest.rs`
- **Change:** Use manifest batch tracking in actual recovery path. On startup, call `find_incomplete_batches()` and re-index those files. Move crash recovery to per-upsert-batch granularity.
- **Tests:** Integration test: begin indexing, simulate crash (drop midway), reopen, verify recovery re-indexes incomplete files.

### 1.4: Concurrent access safety
- **Files:** `crates/core/src/schema.rs`, `crates/core/src/manifest.rs`, `crates/cli/src/app.rs`
- **Change:** Add flock-based lock file for write operations (index, gc, clear). Set busy_timeout=5000ms on CozoDB SQLite backend. Allow concurrent readers during indexing.
- **Tests:** Test concurrent CLI search during indexing doesn't error.

### 1.5: Fix run_mut_ignore error swallowing
- **Files:** `crates/core/src/schema.rs`
- **Change:** Narrow `run_mut_ignore` to only swallow exact expected messages ("already exists" for index creation). Do not match on "conflict" — that could mask dimension mismatch errors.
- **Tests:** Test that re-opening an index with different dimensions returns an error.

---

## Phase 2: Correctness & Quality

### 2.1: Wire config.exclude to WalkBuilder
- **Files:** `crates/core/src/indexer.rs`
- **Change:** Pass `IndexConfig.exclude` patterns to `ignore::WalkBuilder` via `add_ignore()` or glob matching.

### 2.2: Fix GC to clean up symbols
- **Files:** `crates/core/src/gc.rs`
- **Change:** Add `backend.delete_symbols_for_file(path)` call in `collect_garbage()`.

### 2.3: Fix NoopProvider search guard
- **Files:** `crates/mcp/src/server.rs`
- **Change:** Check if real provider exists before search. Return clear error if index is empty/uninitialized.

### 2.4: Fix SmartSearchOutput typing
- **Files:** `crates/mcp/src/tools.rs`
- **Change:** Replace `serde_json::Value` with tagged enum for grep vs semantic results.

### 2.5: Fix serde serialization error swallowing
- **Files:** `crates/mcp/src/server.rs`
- **Change:** Replace `unwrap_or_default()` with `.map_err(|e| e.to_string())`.

### 2.6: Fix watch command or honest help text
- **Files:** `crates/cli/src/app.rs`, `crates/cli/src/cli.rs`
- **Change:** Update help text to honestly describe v1 stub behavior. Add `--help` note about v2 file watching.

### 2.7: Fix searcher 'why' field accuracy
- **Files:** `crates/core/src/searcher.rs`
- **Change:** After RRF fusion, set `why` based on which lists contributed: "vector" (vector-only), "fts" (FTS-only), "hybrid" (both).

---

## Phase 3: Test Suite Hardening

### 3.1: Extract shared test utilities
- **Files:** `crates/core/src/test_utils.rs` (new), update 4 test files
- **Change:** Move `DeterministicTestProvider` and `copy_dir_all` to shared module.

### 3.2: Add error-path tests
- **Tests:** Unreadable files, permission denied, corrupted DB, invalid regex in grep, embedding provider failure.

### 3.3: Fix crash_safety tests
- **Change:** Either implement real process-kill recovery test or rename to `manifest_checkpoint.rs`.

### 3.4: Fix silent-skip test pattern
- **Files:** `crates/embed-fastembed/tests/provider.rs`
- **Change:** Replace `Ok(())` on model unavailable with `#[ignore]` attribute.

### 3.5: Add concurrent index+search test
- **Test:** Search during active indexing returns partial but valid results.

### 3.6: Add binary file and Unicode tests
- **Tests:** Binary skip in indexer, Unicode in grep patterns, Unicode identifiers in chunker.

---

## Phase 4: Integration & Polish

### 4.1: Fix Claude plugin manifest
### 4.2: Fix post-edit-reindex python3 dependency
### 4.3: Expand CLAUDE.md.template
### 4.4: Expand skill definition for MCP tools
### 4.5: Add installation quickstart
### 4.6: Enrich MCP tool descriptions for LLMs
### 4.7: Add MCP server identity
### 4.8: Extract shared make_provider

---

## Phase 5: Competitive Features (Post-Launch)

### 5.1: MMR diversity re-ranking
### 5.2: Embedding cache
### 5.3: HTTP transport for MCP server
### 5.4: Codex/OpenClaw integration docs
### 5.5: ANN optimization for large codebases
