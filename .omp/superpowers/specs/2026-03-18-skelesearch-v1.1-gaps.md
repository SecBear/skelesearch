# skelesearch v1.1 — Feature Gap Spec (DRAFT)
*Date: 2026-03-18*

## Context

v1 shipped with the right architecture (CozoDB HNSW+FTS+graph, StorageBackend trait, provider-agnostic embeddings, MCP+CLI dual surface, Claude Code plugin). The competitive audit against 7 tools (claude-context, Code-Index-MCP, Greptile, Sourcegraph Cody, Cursor, Aider, Bloop) and an internal code audit surfaced gaps in three categories:

1. **Production blockers** — will crash or corrupt under real-world load
2. **Table-stakes features** — every competitor ships these; agents expect them
3. **Differentiators** — features that would make skelesearch the clear winner in its niche

## Skelegent Reuse Opportunities

The sibling `skelegent` repo (at `/Users/bear/dev/golden-neuron/skelegent/` and `/Users/bear/dev/golden-neuron/extras/`) has recently landed several features directly applicable to skelesearch's gaps:

| skelegent Component | Location | skelesearch Gap Solved |
|---|---|---|
| `skg-state-cozo` graph traversal | `extras/state/skg-state-cozo/src/store.rs` | Level-batched BFS for import graph (T2-2) |
| `skg-state-cozo` schema DDL pattern | `extras/state/skg-state-cozo/src/schema.rs` | Idempotent init + separate index DDL |
| `skg-state-cozo` RRF hybrid search | `extras/state/skg-state-cozo/src/search.rs` | Search quality reference impl |
| `EmbedRequest`/`EmbedResponse` types | `skelegent/turn/skg-turn/src/embedding.rs` | Provider-neutral embed contract (P0-1) |
| `EmbedMiddleware` chain | `skelegent/turn/skg-turn/src/infer_middleware.rs` | Batch-chunking middleware seam (P0-1) |
| `ProviderError` with `is_retryable()` | `skelegent/turn/skg-turn/src/provider.rs` | Robust error classification |
| `OtelEmbedMiddleware` | `extras/hooks/skg-hook-otel/src/lib.rs` | gen_ai.* span attributes for indexing (T1-4) |
| WAL-mode SQLite + Mutex pattern | `extras/run/skg-run-sqlite/src/store.rs` | Concurrent access safety (P0-4) |
| Checkpoint schema + migration pattern | `extras/run/skg-run-sqlite/src/schema.rs` | Crash-safe indexing (P0-3) |
| Write-before-transition pattern | `extras/orch/skg-orch-sqlite/src/controller.rs` | Crash-safe indexing (P0-3) |
| `cozo_params!` macro | `extras/state/skg-state-cozo/src/engine.rs` | Cleaner CozoDB query construction |

**Key finding:** skelegent's CozoDB upserts are also single-row — this gap is shared. Neither repo has batched CozoDB writes yet. skelesearch should build multi-row `:put` and upstream the pattern. The CozoScript `<- [[r1], [r2], ...]` syntax supports this natively.

**Key finding:** skelegent has no embed batching/streaming. `EmbedRequest.texts` accepts a full `Vec<String>` but nothing chunks large requests. The `EmbedMiddleware` trait is the right seam — skelesearch should build a batch-chunking middleware and contribute it back.

## Niche Definition

skelesearch occupies a unique position: **local-first, zero-dependency, graph-aware semantic code search for AI coding agents**. No competitor combines all four properties. The strategy is NOT to compete with Sourcegraph (enterprise cross-repo) or Cursor (IDE-locked) or Greptile (SaaS code review), but to be the best single-binary code search primitive that any agent on any machine can use without cloud credentials, Docker, or a server.

---

## Tier 0: Production Blockers (must fix before real use)

### P0-1: Streaming indexing pipeline
**Problem:** Indexer collects ALL chunk texts into `Vec<String>` before embedding. OOM on repos > ~10k files.
**What competitors do:** Cursor uses content-hash embedding cache + incremental sync. Bloop streamed chunks through Qdrant.
**skelegent reuse:** Adopt `EmbedRequest`/`EmbedResponse`/`Embedding` types from `skg-turn/src/embedding.rs` (clean, serde-enabled, no framework coupling). Build a batch-chunking `EmbedMiddleware` that splits large `Vec<String>` into fixed-size batches (100 texts per call), merges responses. This middleware doesn't exist in skelegent yet — build it here, contribute upstream.
**Fix:** Process files in bounded batches (e.g., 100 files at a time). Embed each batch, upsert, then drop before loading the next. Never hold more than `batch_size * avg_chunk_count` chunks in memory. Use a channel-based pipeline: chunk producer -> embed batcher -> index writer.
**Acceptance:** Index the Linux kernel's `drivers/` subtree (~15k files) without exceeding 2GB RSS.

### P0-2: Batched CozoDB upserts
**Problem:** `upsert_chunks()` issues one `:put` per chunk. 500k chunks = 500k transactions = minutes of I/O.
**What competitors do:** SQLite FTS5 tools batch inserts in transactions. Qdrant batches vector upserts.
**skelegent reuse:** `skg-state-cozo` has the same single-row gap. The `cozo_params!` macro and `CozoEngine::run_mutation()` pattern are clean. Build multi-row `:put` using CozoScript's `<- [[r1], [r2], ...]` syntax. Batch 500-1000 rows per query.
**Fix:** Add `upsert_chunks_batch(&self, chunks: &[ChunkRecord])` that builds a single multi-row `:put` script. Same for edges.
**Acceptance:** Index 50k chunks in under 30 seconds.

### P0-3: Crash-safe indexing
**Problem:** Delete-then-embed-then-upsert pipeline. Crash between delete and upsert = lost data until next full re-index.
**skelegent reuse:** Adopt the **write-before-transition** pattern from `skg-orch-sqlite/controller.rs`. Persist intent (list of files about to be processed) BEFORE processing. On restart, check for incomplete batches and re-process. Adopt the WAL-mode SQLite pattern and `pragma user_version` migration tracking from `skg-run-sqlite/schema.rs`.
**Fix:** Add an `index_progress` table to the manifest SQLite DB: `(run_id TEXT, batch_idx INTEGER, files JSON, status TEXT, created_at INTEGER)`. On each batch start, INSERT with status='pending'. On batch completion, UPDATE to 'complete'. On startup, find incomplete batches and re-process. Use upsert-then-delete ordering (write new data before removing old). ~100-150 lines of Rust, no skg-run-core dependency needed.
**Acceptance:** Kill the indexer mid-run 100 times; the index is never in a state where previously-indexed files have zero chunks.

### P0-4: Concurrent access safety
**Problem:** ManifestStore has no busy_timeout. Watch mode + post-edit-reindex can race. SQLITE_BUSY with no retry.
**skelegent reuse:** Adopt `Mutex<Connection>` pattern from `skg-run-sqlite/store.rs` and WAL mode from its schema initialization: `pragma journal_mode=wal` + `pragma busy_timeout=5000`.
**Fix:** Configure `busy_timeout(5000)` on SQLite connections. Add a lock file (`$INDEX_DIR/.skelesearch.lock`) that index/watch commands acquire exclusively with `flock()`. post-edit-reindex checks the lock before spawning.
**Acceptance:** Run `skelesearch index .` and `skelesearch watch .` simultaneously; no SQLITE_BUSY errors.

---

## Tier 1: Table-Stakes Features (agents expect these)

### T1-1: Regex/literal search
**Problem:** All queries go through embedding + FTS hybrid. Agents frequently need exact-match: "find this error string", "find all TODO comments", "where is ERR_INVALID_HANDLE defined".
**What competitors do:** Cursor has Instant Grep (ripgrep). Sourcegraph uses Zoekt trigram engine. Bloop had Tantivy regex. Code-Index-MCP has fuzzy search.
**Approach:** Add a `grep_code` tool to MCP and `grep` subcommand to CLI. Use the `grep` crate or `ignore`+regex for file walking + pattern matching. Return results in the same `SearchResult` shape with `why: "grep"`. Keep it as a separate MCP tool — agents choose between semantic and grep based on query type. Consider a `smart_search` tool later (T2-3) that auto-routes.

### T1-2: More language configs (Tier 1 expansion)
**Problem:** 6 Tier 1 languages (Rust, Nix, Python, TS, JS, Go). Competitors support 14-48.
**Priority additions:** Java, C, C++, Ruby, PHP, C#, Kotlin, Swift, Scala — these cover >95% of professional codebases.
**Approach:** Each is a `LanguageConfig` impl with extensions, tree-sitter grammar, chunk_node_kinds, and import_query. Grammar crates added to workspace deps. Per ADR-010, individual crates not a bundle.
**Acceptance:** `chunk_file("Main.java", ...)` produces function-boundary chunks. Import extraction works for `import` statements.

### T1-3: Config file support
**Problem:** Batch size hardcoded at 64. Provider is CLI-arg only. No way to set project-level defaults.
**What competitors do:** Cursor uses project settings. Code-Index-MCP has JSON config. Aider reads .aider.conf.yml.
**Approach:** `.skelesearch.toml` in project root. Fields: `provider`, `batch_size`, `exclude_patterns`, `index_dir`. CLI args override config. Config loaded by both CLI and MCP.
**Format:**
```toml
[index]
provider = "fastembed"
batch_size = 128
exclude = ["vendor/", "*.generated.*"]

[search]
default_top_k = 10
```

### T1-4: CLI logging and --verbose
**Problem:** CLI uses raw `println!`. No debug output. No way to diagnose slow indexing or missed files.
**skelegent reuse:** Adopt the tracing-based pattern from `skg-hook-otel`. Use `tracing` crate as foundation. The OtelEmbedMiddleware's `gen_ai.*` span attributes are a good reference for what to instrument on embedding calls. For CLI, the progress-to-tracing bridge pattern (from the streaming-observation design) informs how indexing progress events get structured output.
**Fix:** Add `tracing_subscriber` init to CLI with `--verbose` / `-v` flag. Default: warn. `-v`: info. `-vv`: debug. Instrument indexer with `#[tracing::instrument]` on key methods. Add span attributes: files_processed, chunks_embedded, embedding_latency_ms.
**Acceptance:** `skelesearch index . -vv` shows per-file processing, batch timing, skip reasons.

### T1-5: Index cleanup / GC
**Problem:** No way to reclaim space from deleted files except re-indexing. No TTL, no prune.
**Approach:** `skelesearch gc` command. Reads manifest, compares against current filesystem, removes orphaned chunks/edges/files from CozoDB.
**Acceptance:** Delete 1000 files, run `gc`, CozoDB file size decreases.

### T1-6: Reindex debounce / lock file
**Problem:** post-edit-reindex spawns `skelesearch index .` on every edit. Rapid saves = multiple concurrent indexers.
**Fix:** Lock file at `$INDEX_DIR/.skelesearch.lock`. `index` command acquires it exclusively with flock(). If locked, exit 0 silently (another indexer is running). post-edit-reindex checks the lock before spawning. This also solves the concurrent write issue from P0-4.
**Acceptance:** Save 10 files in 2 seconds. Only 1 index process runs.

### T1-7: estimated_stale actually computed
**Problem:** `estimated_stale` is hardcoded to 0 in IndexStats.
**Fix:** Quick scan: count files where `mtime > last_indexed` using the manifest. Doesn't require rehashing — just metadata comparison.
**Acceptance:** Touch 5 files after indexing. `status --json` reports `estimated_stale: 5`.

---

## Tier 2: Competitive Differentiators

### T2-1: Symbol search (find definition / references)
**Problem:** No dedicated symbol-level search. Can search for "struct Foo" via text but can't do structured "find all call sites of function X".
**What competitors do:** Bloop had go-to-definition via tree-sitter. Sourcegraph has full code navigation. Code-Index-MCP has symbol resolution.
**Approach:** During chunking, extract symbol definitions (function names, struct names, type names) and store as a new relation in CozoDB: `symbols { file_path, symbol_name, kind, start_line, end_line }`. New MCP tool: `find_symbol`. CLI: `skelesearch symbol <name>`.
**Acceptance:** `find_symbol("StorageBackend")` returns the trait definition in schema.rs with line numbers.

### T2-2: Agentic search (multi-hop graph traversal)
**Problem:** Current graph augmentation is one-hop only. No iterative expansion along import chains.
**What competitors do:** Greptile's agent follows code references beyond initial hits. Sourcegraph's agentic context gathering traverses dependency graphs.
**skelegent reuse:** Adopt the **level-batched BFS** algorithm from `skg-state-cozo/src/store.rs`. It runs O(depth) Datalog queries instead of O(nodes), using `is_in(from_key, $frontier)` to fan out the entire frontier per level. Cycle-safe via visited HashSet. This is the right CozoDB-native traversal pattern.
**Fix:** After initial hybrid search, traverse import edges N hops deep (configurable, default 2) using level-batched BFS. Deduplicate against primary results. Score graph-discovered results lower. Annotate with `why: "imports <path> (depth N)"`.
**Acceptance:** Search for "database connection" where DB code imports a config module. Graph results include the config module.

### T2-3: Search strategy router
**Problem:** Agent must choose between semantic search and grep. No auto-detection.
**What competitors do:** Cursor's agent autonomously picks between semantic search and Instant Grep.
**Approach:** New MCP tool `smart_search` that analyzes the query: if it looks like a literal/regex (contains special chars, is a known symbol name, is a file path), route to grep. If it's natural language, route to hybrid search. Expose the routing decision in the result.

### T2-4: Multi-repo workspace support
**Deferred to v2.** Single-repo covers 90% of the agent use case.

### T2-5: Reranking pipeline
**Deferred to v2.** Need retrieval quality benchmarks first to prove RRF is insufficient.

---

## Tier 3: Polish and Robustness

### T3-1: Fix SKILL.md documentation
Says "FAISS-backed" — should say CozoDB HNSW.

### T3-2: Add README.md

### T3-3: session-start: remove python3 dependency
Use shell-native JSON extraction or call `skelesearch status --json` and parse with `grep`/`sed`.

### T3-4: MCP tool descriptions
Add query examples, limitations, and "results are candidates" framing to tool descriptions.

### T3-5: Error path testing
Tests for: corrupt CozoDB database, disk full during indexing, permission errors, binary files, concurrent access.

### T3-6: Watch mode stale PID cleanup
Check if PID is alive with `kill(pid, 0)` on Unix.

### T3-7: Chunker error visibility
Log warnings on parse failures. Track count in `status --json` as `parse_errors: N`.

---

## Implementation Strategy

### What to build fresh (in skelesearch)
- Batched CozoDB multi-row `:put` (P0-2) — CozoScript natively supports it but neither repo uses it
- Batch-chunking embed middleware (P0-1) — doesn't exist in skelegent either
- Index progress checkpoint table (P0-3) — adopt the pattern, not the crate
- `grep_code` MCP tool + CLI `grep` subcommand (T1-1)
- Language configs for 9 new languages (T1-2)
- `.skelesearch.toml` config loading (T1-3)
- `gc` command (T1-5)
- `estimated_stale` computation (T1-7)
- Symbol extraction during chunking (T2-1)

### What to adopt from skelegent (patterns, not dependencies)
- WAL-mode SQLite + `busy_timeout` + `Mutex<Connection>` (P0-4) — from `skg-run-sqlite`
- Write-before-transition crash safety (P0-3) — from `skg-orch-sqlite`
- Level-batched BFS traversal (T2-2) — from `skg-state-cozo`
- `cozo_params!` macro for query construction — from `skg-state-cozo`
- `tracing` + `#[instrument]` pattern (T1-4) — from `skg-hook-otel`
- `pragma user_version` migration tracking — from `skg-run-sqlite`

### What to NOT depend on (too heavy, wrong abstraction)
- `skg-state-cozo` crate itself — skelesearch's StorageBackend is simpler and purpose-built. Importing skg-state-cozo would pull in `layer0` and the full StateStore trait, which is overengineered for code search.
- `skg-turn` Provider trait — skelesearch's `EmbedProvider` trait is simpler (just embed, no infer). Copy the type definitions, not the trait hierarchy.
- `skg-run-core` — skelesearch's indexer doesn't need a 5-state run machine. A 3-state checkpoint (pending/complete/failed) suffices.

---

## Feature Matrix: skelesearch v1 vs v1.1 vs Competitors

| Feature | v1 | v1.1 | claude-ctx | Code-Index | Cody | Cursor |
|---|---|---|---|---|---|---|
| Hybrid BM25+vector | Y | Y | Y | Y | Partial | Partial |
| Import graph queries | Y | Y | N | N | Y (static) | N |
| Zero external deps | Y | Y | N | N | N | N |
| MCP server | Y | Y | Y | Y | Y | Consumer |
| Claude plugin | Y | Y | Y | Y | N | N |
| Regex/literal search | **N** | **Y** | N | Y | N | Y |
| Large repo (>10k files) | **N** | **Y** | Y | Y | Y | Y |
| Crash-safe indexing | **N** | **Y** | Y | ? | Y | Y |
| Symbol search | N | **Y** | N | Y | Y | N |
| Config file | **N** | **Y** | Y | Y | Y | Y |
| Languages (Tier 1) | 6 | **15+** | 14 | 48 | All | All |
| Concurrent access safe | **N** | **Y** | Y | ? | Y | Y |
| Index GC/cleanup | **N** | **Y** | ? | ? | Y | Y |
| Multi-hop graph traversal | 1-hop | **2-hop** | N | N | Static | N |
| Reranking pipeline | N | N | N | Y | Y | N |
| Multi-repo | N | N | N | Y | Y | N |

---

## Open Questions

1. **Should regex search be a separate MCP tool (`grep_code`) or a mode on `search_code`?** Recommendation: separate tool. Cleaner for agent tool selection. `smart_search` (T2-3) can auto-route later.

2. **How many Tier 1 languages in v1.1?** Adding 9 (Java, C, C++, Ruby, PHP, C#, Kotlin, Swift, Scala) brings us to 15. Each needs a tree-sitter grammar crate, chunk_node_kinds, and import_query. Effort: ~2 hours per language.

3. **Should we implement symbol search as a CozoDB relation or derive it from chunk metadata?** Recommendation: CozoDB relation. It's the right abstraction and CozoDB handles it well.

4. **Config format — TOML or JSON?** Recommendation: TOML (Rust convention, human-friendly). Agents don't need to generate config files — they use MCP tools.

5. **Workspace package metadata**: The current `authors`, `homepage`, `repository` fields use placeholder values. What are the real canonical values?

6. **Should skelesearch depend on any skelegent crates?** Recommendation: **No.** Copy patterns, not crates. skelesearch must remain zero-dependency and self-contained per its niche definition. Adding `layer0` or `skg-turn` as deps would undermine the single-binary story.
