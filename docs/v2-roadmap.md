# skelesearch v2 Roadmap

> The definitive open-source semantic code search engine for coding agents.
> Narrow, hard, excellent. Code retrieval, not general RAG.

Last updated: 2026-03-19 (post-strategic-research consolidation)

---

## Current State (v1.5)

**Shipped (all committed, unpushed):**
- Tree-sitter AST chunking (15 languages + sliding-window fallback)
- Hybrid BM25 + vector search with RRF fusion
- LLM query expansion (OpenAI, gated by Grep/Symbol/Semantic classifier)
- Multi-query keyword expansion with camelCase/PascalCase splitting
- Cross-encoder reranking (Jina/Cohere/Voyage unified client)
- MMR diversity reranking
- Embedding cache (content-hash keyed, SQLite-backed)
- Local ONNX embeddings (jina-v2-base-code, 768-dim)
- Cloud embeddings (OpenAI 1536-dim, Voyage code-3 1024-dim)
- Import graph with heuristic resolvers (Rust/TS/JS/Python/Go + 8 fallback languages)
- PageRank on file-level import graph (pure-Rust power iteration)
- Post-reranker test-file penalty (0.3x) + doc-file penalty (0.5x)
- Strong signal detection (skip reranker when BM25 dominant)
- Position-aware rerank blending (top 3: 25/75, mid: 40/60, tail: 60/40)
- File extension allowlist (~80 extensions, configurable)
- FTS query sanitization (strips dots, hyphens, reserved keywords)
- MCP server (stdio + HTTP transport, session dedup, branch scope)
- CLI with index/search/grep/symbol/context/status/gc/watch/clear/eval
- Production observability (#[instrument] spans)
- File watcher re-indexing (notify, 2s debounce)
- Token-budget-aware retrieval (--max-tokens, default 8192 for MCP)
- Diff-aware branch-scoped search (--branch)
- Provider manifest (stores which model indexed, auto-detect on search)
- Eval framework (R@5, R@10, MRR) + 96 benchmark cases across 6 repos
- Zero-config pipeline wiring (auto-detect API keys)

**Eval baselines (commit `9e31546`):**

| Corpus | Cases | R@5 | R@10 | MRR |
|---|---|---|---|---|
| 6-repo benchmark (voyage-full avg) | 96 | 84.0% | 87.2% | 87.8% |
| 6-repo benchmark (no-expansion avg) | 96 | 80.3% | 80.3% | 87.7% |
| skelegent (15 cases) | 15 | 70.0% | 70.0% | 80.0% |

---

## Strategic Positioning

**Thesis:** skelesearch wins on five dimensions no competitor covers together:
1. Task-specialized retrieval (graph traversal where semantic search can't work)
2. Structural code graph (imports, symbols, PageRank — not just embeddings)
3. Compact, high-precision context (fewer better chunks > more comprehensive chunks)
4. Repo-specific ranking priors (test/doc penalties, barrel-file detection, recency)
5. Operational truthfulness (freshness, provenance, deterministic staleness detection)

**Competitive landscape (validated Mar 2026):**
- No competitor offers task-specialized retrieval plans
- No competitor publishes standard IR metrics — first mover on public benchmark is a moat
- Claude Code's "agentic beats RAG" narrative is the philosophical competitor, not other tools
- Augment Code is the biggest blind spot (opaque, $252M funding, may build similar)
- Cursor's utility-aligned custom embeddings are a genuine moat; skelesearch differentiates on graph + plans
- Sourcegraph Cody is the closest architectural competitor; their public freeze is an opportunity

**Product split (validated):** Keep skelesearch narrow on code retrieval. Branch a
sibling product for long-horizon multimodal memory later. Zero overlap between
memory systems (Mem0, Zep, Letta) and code search. Share only true substrate.

---

## Implementation Queue

### Phase A: Retrieval Quality + Eval Rigor (next)

#### A1: Contextual Chunk Descriptions (2-3 days) — estimated -49% retrieval failure
Prepend a 50-100 token context prefix to each chunk before embedding. Anthropic's
Contextual Retrieval pattern. Highest expected impact per effort.

**What:**
- At chunking time, generate a prefix: `"{file_path} | {parent_scope} | {chunk_type}: {symbol_name}"`
- Prepend to the `normalized` field that feeds both BM25 FTS and vector embedding
- No LLM needed — derive from AST metadata already extracted during chunking
- Example: `"src/server.rs | impl Server | function: run — Accepts connections with semaphore limit and exponential backoff on accept errors"`

**Evidence:** Anthropic Contextual Retrieval (Sep 2024): -49% retrieval failure with
prefix alone, -67% combined with reranking. Chroma Context Rot (Jul 2025): topically
related distractors hurt more than irrelevant content — precision matters more than recall.

**Files:** `crates/core/src/chunker.rs`, `crates/core/src/indexer.rs`

#### A2: tree-sitter tags.scm Integration (2 days) — richer symbols, graph foundation
Replace hand-rolled `extract_symbols` with `tree-sitter-tags` crate for definition +
reference extraction across 14/15 languages.

**What:**
- Add `tree-sitter-tags = "0.26"` dependency (version-locked to tree-sitter 0.26.x)
- `TagsConfiguration::new()` per language in `OnceLock`, `TagsContext::new()` per thread
- Map `syntax_type_id` via `config.syntax_type_name()` into existing `normalize_kind()`
- Extract both `@definition.*` and `@reference.*` captures
- Replace `symbols.rs` manual traversal

**Critical finding:** `@reference.call` is single-file only — does NOT produce
cross-file edges. Cross-file call graph needs additional import-join post-processing
(~1-2 weeks, deferred to Phase B).

**Evidence:** 14/15 target languages have tags.scm. tree-sitter-nix has none (skip).
tree-sitter-java/cpp TAGS_QUERY needs verification before shipping.

**Files:** `crates/core/src/symbols.rs`, `crates/core/Cargo.toml`

#### A3: Barrel-File Penalty (1-2 days) — fixes httpx MRR (62% → target 80%+)
Detect and downrank re-export files (`__init__.py`, `index.ts`, `mod.rs`).

**What:**
- Filename pre-filter: `index.ts`, `__init__.py`, `mod.rs`, `barrel.ts`
- Content analysis: tree-sitter count `reexport_count > declaration_count AND reexport_count >= 3`
- Apply 0.4x score penalty alongside test/doc penalties (post-reranker)

**Evidence:** eslint-plugin-barrel-files ratio heuristic. httpx `__init__.py` re-exports
every public symbol, outranking actual implementations on keyword-dense queries.

**Files:** `crates/core/src/searcher.rs`, `crates/core/src/indexer.rs`

#### A4: Eval Expansion + Metrics Upgrade (3-4 days)
Scale to 250+ categorized cases. Add Precision@5. Adopt SWE-PolyBench.

**What:**
- Expand benchmark corpus to 250+ cases with explicit `category` field per case
  (symbol_lookup, implementation, cross_file, architecture, error_handling, config, vocabulary_gap)
- Add Precision@5 metric to eval framework (token efficiency proxy)
- Reframe metric hierarchy: (1) file-level R@5 as correctness floor, (2) P@5 for
  token efficiency, (3) MRR for ranking quality
- Adopt SWE-PolyBench (2,110 multi-language issues) as external independent benchmark
- Add held-out test set (20% of cases) to detect eval-set overfitting
- Publish results in README

**Evidence:** ContextBench (Feb 2026): file-level R@5 strongest predictor of agent task
completion. 96 cases = ~70% power to detect 5% improvement. Need 250+ for statistical rigor.

**Files:** `benchmarks/cases/`, `crates/core/src/eval.rs`, `crates/cli/src/eval.rs`

### Phase B: Graph Depth + Retrieval Plans (after Phase A)

#### B1: Cross-File Call Graph Edges (1-2 weeks)
Join intra-file `@reference.call` captures from tags.scm against imported symbols
table to produce cross-file call edges. Feeds richer graph for PageRank + impact set.

**What:**
- For each `@reference.call` tag, look up name in file's imported symbols
- If match found, emit `EdgeRecord` with `edge_type = "calls"`
- Store alongside existing `"imports"` edges in `code_edges` relation
- Recompute PageRank with combined import + call edges

**Files:** `crates/core/src/indexer.rs`, `crates/core/src/schema.rs`

#### B2: `find_impact_set` Retrieval Plan (3-5 days)
The one plan where task specialization is causally necessary. CodeCompass: graph-based
navigation achieves 99.4% on hidden dependency tasks vs BM25's 76.2%.

**What:**
- Add `traverse_importers()` reverse BFS to `StorageBackend` trait
- New MCP tool: `find_impact_set(file_path, symbol?) → {direct_importers, transitive_importers, tests, configs}`
- Group results by ripple level (distance from changed file)
- Include test files that import the target (test-to-source edge detection)

**Evidence:** CodeCompass (arXiv:2602.20048, Feb 2026): 23pp gap where BM25 gives zero
improvement. LaToza & Myers 2010: reachability questions hardest category with weakest tool support.

**Files:** `crates/core/src/schema.rs`, `crates/core/src/searcher.rs`, `crates/mcp/src/server.rs`

#### B3: `find_test_context` Retrieval Plan (2-3 days)
Find tests covering a file or symbol. High-frequency agent workflow.

**What:**
- Reverse-lookup: which test files import the target file?
- Symbol-level: which test functions reference the target symbol?
- Return test files + specific test function chunks

**Files:** `crates/core/src/schema.rs`, `crates/mcp/src/server.rs`

### Phase C: Scale + Distribution (after Phase B)

#### C1: LanceDB+Tantivy Backend (1 week) — derisk CozoDB dependency
Feature-gated alternative. CozoDB's `graph-algo` is already broken; project stalled.

**What:**
- `crates/core/src/lance_backend.rs` implementing `StorageBackend` (16+ methods)
- LanceDB for vector storage, Tantivy for BM25 FTS
- Pure-Rust graph storage for edges + PageRank
- `open_backend()` selects based on config or `--backend` flag

**Evidence:** CozoDB last release v0.7.6 (Dec 2023). graph-algo broken.
Scale: HNSW erratic at 700K-1.2M vectors without int8 quantization.

**Files:** `crates/core/src/lance_backend.rs` (new), `crates/core/Cargo.toml`

#### C2: npm Distribution (2-3 days) — adoption
`npx -y @skelesearch/mcp` one-liner for Claude Code, Codex, OMP.

**What:**
- npm package wrapping native binary (postinstall downloads platform binary)
- Homebrew tap: `brew install skelesearch`
- GitHub Releases with prebuilt binaries (Linux x86_64, macOS arm64/x86_64)
- cargo install from crates.io

#### C3: Lite Mode (2 days) — zero-config DX
BM25-only search with no API keys, no model downloads.

**What:**
- `skelesearch search "query" .` works with BM25 + AST chunking only
- `--provider none` = BM25-only mode
- Instant first run, quality degrades gracefully

### Phase D: Advanced (ongoing)

- ColBERT/LateOn-Code integration (next-plaid + next-plaid-onnx)
- Contextual chunk descriptions via LLM (upgrade from AST-only to LLM-generated summaries)
- Git recency boost (Cody/Sourcegraph pattern)
- Co-change index from git log
- `find_defect_context` retrieval plan (bug localization from stack traces)
- `trace_architecture` retrieval plan (graph traversal for onboarding/planning)
- Matryoshka adaptive dimensions
- Retrieval feedback loop (log query→used_results, tune RRF weights per-repo)
- Multi-repo indexing
- VS Code extension
- SWE-PolyBench adapter for external benchmark comparison
- Public benchmark publication (competitive moat)

---

## Priority Matrix

| # | Feature | Effort | Impact | Risk |
|---|---|---|---|---|
| A1 | Contextual chunk descriptions | 2-3d | **-49% retrieval failure** | LOW |
| A2 | tree-sitter tags.scm | 2d | richer symbols, graph foundation | LOW |
| A3 | Barrel-file penalty | 1-2d | **httpx MRR fix** | LOW |
| A4 | Eval expansion + metrics | 3-4d | statistical rigor, competitive moat | LOW |
| B1 | Cross-file call graph | 1-2w | deeper graph, better PageRank | MEDIUM |
| B2 | find_impact_set plan | 3-5d | **23pp gap where BM25 fails** | MEDIUM |
| B3 | find_test_context plan | 2-3d | high-frequency agent workflow | LOW |
| C1 | LanceDB+Tantivy backend | 1w | derisk CozoDB, scale to 100K files | MEDIUM |
| C2 | npm distribution | 2-3d | adoption critical | LOW |
| C3 | Lite mode | 2d | zero-config DX | LOW |

---

## Key Decisions (updated)

| Decision | Choice | Rationale |
|---|---|---|
| Primary metric | file-level R@5 | ContextBench: strongest predictor of agent task completion |
| Product scope | Code retrieval only | Validated: memory systems are separate domain (Mem0, Zep, Letta) |
| Context format | Compact precision > rich packets | Anthropic PTC pattern, Chroma Context Rot, Lost in the Middle |
| Retrieval plans | Router + PlanConfig, not separate code paths | 4/5 plans are config tuning; only find_impact_set needs graph BFS |
| Cross-file calls | Import-join post-processing on tags.scm refs | @reference.call is single-file only — RED for direct cross-file edges |
| CozoDB mitigation | Pure-Rust PageRank + StorageBackend trait | graph-algo feature broken, LanceDB+Tantivy at Phase C |
| Eval strategy | Internal 250+ cases + SWE-PolyBench external | No competitor publishes IR metrics — first mover is a moat |
| Scale threshold | int8 quantization at >500K vectors | HNSW erratic at 700K-1.2M without it |

---

## Target Metrics

| Metric | Current (v1.5) | After Phase A | After Phase B | Best-in-class |
|---|---|---|---|---|
| R@5 (voyage-full) | 84.0% | 88-92% | 92-95% | ~90% (Cursor est.) |
| MRR (voyage-full) | 87.8% | 90-93% | 93-96% | ~85% (Cursor est.) |
| P@5 | not tracked | tracked | optimized | not published |
| Eval cases | 96 | 250+ | 300+ | — |
| Languages (graph) | 5+8 fallback | 14 (tags.scm) | 14 | ~5 (competitors) |
