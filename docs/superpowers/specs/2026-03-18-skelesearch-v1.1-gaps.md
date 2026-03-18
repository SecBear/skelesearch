# skelesearch v1.1 — Feature Gap Spec (DRAFT)
*Date: 2026-03-18*

## Context

v1 shipped with the right architecture (CozoDB HNSW+FTS+graph, StorageBackend trait, provider-agnostic embeddings, MCP+CLI dual surface, Claude Code plugin). The competitive audit against 7 tools (claude-context, Code-Index-MCP, Greptile, Sourcegraph Cody, Cursor, Aider, Bloop) and an internal code audit surfaced gaps in three categories:

1. **Production blockers** — will crash or corrupt under real-world load
2. **Table-stakes features** — every competitor ships these; agents expect them
3. **Differentiators** — features that would make skelesearch the clear winner in its niche

## Niche Definition

skelesearch occupies a unique position: **local-first, zero-dependency, graph-aware semantic code search for AI coding agents**. No competitor combines all four properties. The strategy is NOT to compete with Sourcegraph (enterprise cross-repo) or Cursor (IDE-locked) or Greptile (SaaS code review), but to be the best single-binary code search primitive that any agent on any machine can use without cloud credentials, Docker, or a server.

---

## Tier 0: Production Blockers (must fix before real use)

### P0-1: Streaming indexing pipeline
**Problem:** Indexer collects ALL chunk texts into `Vec<String>` before embedding. OOM on repos > ~10k files.
**What competitors do:** Cursor uses content-hash embedding cache + incremental sync. Bloop streamed chunks through Qdrant.
**Fix:** Process files in bounded batches (e.g., 100 files at a time). Embed each batch, upsert, then drop before loading the next. Never hold more than `batch_size * avg_chunk_count` chunks in memory.
**Acceptance:** Index the Linux kernel's `drivers/` subtree (~15k files) without exceeding 2GB RSS.

### P0-2: Batched CozoDB upserts
**Problem:** `upsert_chunks()` issues one `:put` per chunk. 500k chunks = 500k transactions = minutes of I/O.
**What competitors do:** SQLite FTS5 tools batch inserts in transactions. Qdrant batches vector upserts.
**Fix:** Build multi-row `:put` queries with parameterized data arrays. Batch 500-1000 rows per transaction.
**Acceptance:** Index 50k chunks in under 30 seconds (currently unbounded).

### P0-3: Crash-safe indexing
**Problem:** Delete-then-embed-then-upsert pipeline. Crash between delete and upsert = lost data until next full re-index.
**What competitors do:** claude-context uses Merkle trees for atomic state tracking. Cursor caches by content hash.
**Fix:** Upsert-then-delete pattern: write new chunks first, then remove stale ones. Or: defer manifest update until all CozoDB writes succeed (manifest is the commit point). On next run, files with stale manifest entries get re-indexed.
**Acceptance:** Kill the indexer mid-run 100 times; the index is never in a state where indexed files have zero chunks.

### P0-4: Concurrent access safety
**Problem:** ManifestStore has no busy_timeout. Watch mode + post-edit-reindex can race. SQLITE_BUSY with no retry.
**Fix:** Configure `busy_timeout(5000)` on SQLite connections. Add a lock file (`$INDEX_DIR/.skelesearch.lock`) that index/watch commands acquire exclusively. post-edit-reindex checks the lock before spawning.
**Acceptance:** Run `skelesearch index .` and `skelesearch watch .` simultaneously; no SQLITE_BUSY errors.

---

## Tier 1: Table-Stakes Features (agents expect these)

### T1-1: Regex/literal search
**Problem:** All queries go through embedding + FTS hybrid. Agents frequently need exact-match: "find this error string", "find all TODO comments", "where is ERR_INVALID_HANDLE defined".
**What competitors do:** Cursor has Instant Grep (ripgrep). Sourcegraph uses Zoekt trigram engine. Bloop had Tantivy regex. Code-Index-MCP has fuzzy search.
**Approach:** Add a `grep` tool to MCP and `grep` subcommand to CLI. Use the `grep` crate or `ignore`+regex for file walking + pattern matching. Return results in the same `SearchResult` shape with `why: "grep"`.
**Open question:** Should `search` auto-detect regex patterns (e.g., `/pattern/`) and route to grep, or keep grep as a separate tool?

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
**Fix:** Add `tracing_subscriber` init to CLI with `--verbose` / `-v` flag. Default: warn. `-v`: info. `-vv`: debug. Structured JSON logs with `--log-json`.
**Acceptance:** `skelesearch index . -vv` shows per-file processing, batch timing, skip reasons.

### T1-5: Index cleanup / GC
**Problem:** No way to reclaim space from deleted files except re-indexing. No TTL, no prune.
**Approach:** `skelesearch gc` command. Reads manifest, compares against current filesystem, removes orphaned chunks/edges/files from CozoDB.
**Acceptance:** Delete 1000 files, run `gc`, CozoDB file size decreases.

### T1-6: Reindex debounce / lock file
**Problem:** post-edit-reindex spawns `skelesearch index .` on every edit. Rapid saves = multiple concurrent indexers.
**Fix:** Lock file at `$INDEX_DIR/.skelesearch.lock`. `index` command acquires it exclusively with flock(). If locked, exit 0 silently (another indexer is running). post-edit-reindex checks the lock before spawning.
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

### T2-2: Agentic search (iterative graph traversal)
**Problem:** Search returns top-k flat results. No iterative expansion along import edges.
**What competitors do:** Greptile's agent follows code references beyond initial hits. Sourcegraph's agentic context gathering.
**Approach:** After initial hybrid search, for `include_graph: true`, follow import edges 2 hops deep. Deduplicate. Score graph-discovered results lower. This is a Datalog strength — recursive queries in CozoDB are one-liners.
**Acceptance:** Search for "database connection" in a project where the DB code imports a config module. Graph results include the config module even though it doesn't mention "database".

### T2-3: Search strategy router
**Problem:** Agent must choose between semantic search and grep. No auto-detection.
**What competitors do:** Cursor's agent autonomously picks between semantic search and Instant Grep.
**Approach:** New MCP tool `smart_search` that analyzes the query: if it looks like a literal/regex (contains special chars, is a symbol name, is a path), route to grep. If it's natural language, route to hybrid search. Expose the routing decision in the result.
**Open question:** Is this a separate tool or does `search_code` gain a `strategy: auto | semantic | grep` parameter?

### T2-4: Multi-repo workspace support
**Problem:** Single-repo only. Can't search across related repos.
**What competitors do:** Code-Index-MCP supports multi-repo workspaces. Sourcegraph is built for cross-repo.
**Approach:** `.skelesearch.toml` gains a `[workspace]` section listing paths. Index/search operates across all listed repos. Results include repo origin.
**Deferred?** This may be v2. Single-repo is the core use case for now.

### T2-5: Reranking pipeline
**Problem:** Results are ranked by raw RRF score. No learned reranking.
**What competitors do:** Code-Index-MCP has multi-strategy reranking (TF-IDF, Cohere, Cross-Encoder). Sourcegraph has an ML ranking layer.
**Approach:** Optional reranker trait. Default: none (raw RRF is already good). Optional: cross-encoder reranking using a small model. Feature-gated, not in the default binary.
**Deferred?** Likely v2. Need benchmarks first to prove RRF is insufficient.

---

## Tier 3: Polish and Robustness

### T3-1: Fix SKILL.md documentation
**Problem:** Says "FAISS-backed" — should say CozoDB. Stale documentation.

### T3-2: Add README.md
**Problem:** No README. CLAUDE.md is the only docs.

### T3-3: session-start: remove python3 dependency
**Problem:** Hook uses python3 to parse JSON. Fragile on minimal systems.
**Fix:** Use shell-native JSON extraction: `grep -o '"indexed_files":[0-9]*'` or similar.

### T3-4: MCP tool descriptions
**Problem:** Tool descriptions are minimal. Should include query examples, limitations, and the "results are candidates" framing.

### T3-5: Error path testing
**Problem:** No tests for corrupt DB, disk full, permission errors, binary files, concurrent access.

### T3-6: Watch mode stale PID cleanup
**Problem:** SIGKILL leaves a stale PID file. `is_process_watching()` doesn't verify the process is alive.
**Fix:** Check if PID is alive with `kill(pid, 0)` on Unix.

### T3-7: Chunker error visibility
**Problem:** `chunker.chunk_file().unwrap_or_default()` silently swallows parse failures. Files that fail to parse are indexed as having zero chunks — invisible data loss.
**Fix:** Log a warning and track the count. `status --json` should report `parse_errors: N`.

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
| Reranking pipeline | N | N | N | Y | Y | N |
| Multi-repo | N | N | N | Y | Y | N |
| Agentic iterative search | N | Partial | N | N | Y | Y |

---

## Open Questions

1. **Should regex search be a separate MCP tool (`grep_code`) or a mode on `search_code`?** Separate tool is cleaner for agent tool selection. But `search_code` with `strategy: auto` reduces agent decision burden.

2. **How many Tier 1 languages in v1.1?** Adding 9 (Java, C, C++, Ruby, PHP, C#, Kotlin, Swift, Scala) brings us to 15. Each needs a tree-sitter grammar crate, chunk_node_kinds, and import_query. Effort: ~2 hours per language.

3. **Should we implement symbol search as a CozoDB relation or derive it from chunk metadata?** CozoDB relation is cleaner but adds another index to maintain. Deriving from chunk types + names is simpler but less precise.

4. **Is multi-repo workspace in scope for v1.1 or deferred to v2?** Single-repo covers 90% of the agent use case. Multi-repo adds configuration complexity.

5. **Should the config file be TOML (Rust convention) or JSON (easier for agents to generate)?** TOML is more human-friendly. JSON is more machine-friendly. Could support both with serde.

6. **Workspace package metadata**: The current `authors`, `homepage`, `repository` fields use placeholder values. What are the real canonical values?
