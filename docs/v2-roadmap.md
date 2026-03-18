# skelesearch v2 Roadmap — Beat Every Competitor

> Goal: The definitive open-source semantic code search engine for agentic systems.
> No competitor has all of these. We will.

## Current state (v1.2)

**What we have:**
- Tree-sitter AST chunking (15 languages + sliding-window fallback)
- Hybrid BM25 + vector search with Reciprocal Rank Fusion
- MMR diversity reranking
- Embedding cache (content-hash keyed, SQLite-backed)
- Local ONNX embeddings (jina-v2-base-code, 768-dim)
- Cloud embeddings (OpenAI text-embedding-3-small, 1536-dim)
- MCP server (stdio + HTTP transport) with smart_search, search_code, find_symbol, get_file_context
- CLI with full index/search/grep/symbol/context/status/gc/watch/clear commands
- Production observability (#[instrument] spans, timing, cache counters)
- Post-edit hooks for automatic background re-indexing
- CozoDB single-DB for both HNSW vector and BM25 FTS

**What's missing (competitive gaps):**
- No file-watcher re-indexing (watch is a PID stub)
- No cross-encoder reranker
- No eval framework
- No token-budget-aware output
- No query decomposition/expansion
- No provider name stored in manifest
- No diff-aware branch-scoped retrieval

---

## Phase 1: Production Polish (1-2 days)
*Close the gaps that make dogfooding painful.*

### 1.1 Store provider/model in manifest
- Add `metadata` table to ManifestStore: `key TEXT PRIMARY KEY, value TEXT`
- On index: store `provider_name`, `model_name`, `dim` in metadata
- On search: read metadata, auto-select provider (fallback to fastembed)
- Eliminates "indexed with openai, searched with fastembed" dimension mismatch
- **Files:** `manifest.rs`, `indexer.rs`, `app.rs`, `server.rs`

### 1.2 File watcher re-indexing
- Replace the `watch` PID-only stub with actual `notify`-based file watching
- On `Modify`/`Create`/`Remove` events: debounce (2s), run `index_path` on changed files
- Use `notify-debouncer-full` (already in workspace deps)
- GC on file removal (already implemented as `gc::collect_garbage`)
- **Files:** `app.rs` (run_watch), possibly extract to `crates/core/src/watcher.rs`

### 1.3 Token-budget-aware retrieval
- Add `--max-tokens` flag to CLI search and MCP search_code/smart_search
- Greedy selection: sort by score, accumulate token counts (approximate via `content.len() / 4`), stop at budget
- Return `truncated: bool` in output so agent knows budget was hit
- Default: unlimited (backwards compatible). Recommended: 8192 for agent use.
- **Files:** `searcher.rs`, `cli.rs`, `app.rs`, `tools.rs`, `server.rs`

---

## Phase 2: Retrieval Quality (1 week)
*Match Cursor/Copilot retrieval quality.*

### 2.1 Cross-encoder reranker
- Add `crates/rerank/` crate with `Reranker` trait
- Implement `JinaReranker` using Jina Reranker v3 (ONNX, 0.6B params) or Qwen3-Reranker (Apache 2.0)
- Pipeline: initial retrieval (top-50) → reranker scores all 50 → return top-K
- Add `--rerank` flag to CLI, `rerank: bool` to MCP (default: true when reranker available)
- **Expected impact:** 5-15% nDCG@10 improvement. Distinguishes "similar function" from "correct function."

### 2.2 Multi-query expansion
- When `smart_search` receives a NL query, generate 2-3 variant queries:
  - Original NL query → embed for vector search
  - Extract keywords → use for BM25 (already done)
  - Generate a "hypothetical code snippet" prompt → embed for vector search (HyDE-lite)
- Merge results from all queries via RRF
- No LLM required for basic expansion (keyword extraction + synonym mapping)
- **Files:** `searcher.rs` (add `multi_query_search`), `server.rs` (smart_search path)

### 2.3 Diff-aware branch-scoped retrieval
- Add `--branch` flag: scope search to files changed on current branch
- Implementation: `git diff --name-only HEAD...$(git merge-base HEAD main)`
- Filter HNSW and FTS results to only include chunks from changed files
- Massive precision improvement for agent tasks on feature branches
- **Files:** `searcher.rs`, `cli.rs`, `app.rs`, `tools.rs`

---

## Phase 3: Late Interaction / ColBERT (2-3 weeks)
*Leapfrog bi-encoder quality. No competitor in our space has this.*

### 3.1 LateOn-Code integration
- Add `crates/embed-lateon/` crate implementing a new `ColBERTProvider` trait
- LateOn-Code models (17M and 130M params) run via ONNX locally
- Each chunk produces N vectors (one per token) instead of one vector
- Storage: multi-vector column in CozoDB or separate SQLite table
- **Key challenge:** CozoDB's HNSW is designed for single-vector. Options:
  - Store mean-pooled single vector for HNSW coarse search, then late-interaction rerank on top-100
  - Or use Next-Plaid (LightOn's Rust multi-vector DB) alongside CozoDB
- ColBERT MaxSim scoring: `score = sum(max(q_i · d_j for all j) for all i)`
- **Expected impact:** 70% win rate vs pure grep (ColGrep benchmark). Best retrieval quality for code.

### 3.2 Hybrid regex + semantic search
- When query contains regex-like patterns (`/pattern/`, `func_name`, `ClassName`), split into:
  - Regex component → fast grep filter
  - Semantic component → embedding search on grep-filtered candidates
- This is what ColGrep does and it's the most novel approach in the space
- Subsumes traditional grep rather than replacing it
- **Files:** `searcher.rs` (add `hybrid_regex_semantic_search`), `cli.rs`, `tools.rs`

---

## Phase 4: Eval Framework (3-5 days)
*Can't improve what you can't measure. No open-source competitor has this.*

### 4.1 Morph Labs-style auto-eval
- Given a repo + set of resolved GitHub issues/PRs:
  - Extract the files/functions modified in each fix
  - Generate NL queries from issue titles/descriptions
  - Ground truth: the files/functions that were actually modified
- Measure: Recall@5, Recall@10, MRR, NDCG@10
- **Output:** `skelesearch eval --repo . --issues issues.json`

### 4.2 CoIR benchmark runner
- Implement CoIR (Code Information Retrieval) benchmark evaluation
- Measures our embedding model + retrieval pipeline against standard datasets
- Publish results in README to establish credibility
- **Crate:** `github.com/CoIR-team/coir` (Python, but we can wrap)

### 4.3 RepoBench-R evaluation
- Cross-file snippet retrieval within a repo — closest to our actual use case
- Use as regression test: any change to retrieval pipeline must not regress RepoBench-R scores

---

## Phase 5: Advanced Features (2-4 weeks)
*Market leadership. Things nobody else has.*

### 5.1 Matryoshka adaptive dimensions
- Support MRL-compatible models (nomic-embed-text-v1.5, Jina v3)
- Two-phase retrieval: coarse search at 128 dims over full index, then full-dim rerank top-100
- 4-8x storage reduction with <5% quality loss
- Store both truncated and full embeddings, or just full and truncate at query time
- **Expected impact:** Faster search, lower memory, enables larger repos

### 5.2 Call graph extraction
- Extract function call edges from tree-sitter ASTs
- Store as `edge_type: "calls"` in CozoDB `code_edges` relation
- Post-retrieval expansion: found function X → also pull callers/callees of X
- **Expected impact:** Answers "how is this function used?" without separate LSP
- **Files:** `chunker/`, `schema.rs`

### 5.3 Retrieval feedback loop (lightweight)
- Log `(query, retrieved_chunks, session_id)` to `~/.skelesearch/feedback.db`
- Track which results the agent actually used (MCP can see tool call sequences)
- Monthly: analyze unused results as hard negatives, surface precision problems
- No model retraining — just adjust RRF weights (vector_weight vs fts_weight) per-repo
- **Files:** `searcher.rs`, `server.rs`, new `feedback.rs`

### 5.4 Multi-repo indexing
- Allow indexing multiple repos into a single searchable namespace
- Use case: monorepo with multiple packages, or related repos
- Implementation: prefix file paths with repo name, add `--repo` filter to search
- **Files:** `indexer.rs`, `searcher.rs`, `cli.rs`

### 5.5 Streaming index updates via MCP
- Add `index_file` MCP tool: index a single file without full project scan
- Agent can call this after writing a file to immediately update the index
- Faster than post-edit hook (no process spawn, no disk walk)
- **Files:** `server.rs`, `tools.rs`

---

## Phase 6: Ecosystem (ongoing)
*Distribution and adoption.*

### 6.1 Package distribution
- Homebrew formula (tap or core)
- cargo install from crates.io
- Nix flake (already exists)
- Pre-built binaries for Linux x86_64, macOS arm64/x86_64, Windows

### 6.2 IDE integrations
- VS Code extension (MCP client that connects to skelesearch-mcp)
- Neovim plugin (MCP client)
- JetBrains plugin

### 6.3 Agent framework integrations
- Claude Code: MCP config + hooks (already done)
- Codex: MCP config
- Cursor: MCP config
- Windsurf: MCP config
- OMP/OpenClaw: MCP config + integration docs (already done)
- Continue.dev: context provider plugin
- Aider: custom command integration

### 6.4 Documentation
- Architecture deep-dive (how hybrid search works)
- Embedding model comparison guide (local vs cloud, quality vs cost)
- Benchmarking your own repo guide
- Contributing guide for adding languages/providers

---

## Priority Matrix

| Feature | Effort | Impact | Competitors Have It | Priority |
|---------|--------|--------|-------------------|----------|
| Provider in manifest | 1 day | high | N/A (our bug) | P0 |
| File watcher reindex | 2 days | high | grepai, cocoindex | P0 |
| Token-budget output | 1 day | high | Probe, ColGrep | P0 |
| Cross-encoder reranker | 3 days | high | DeepContext, Cursor | P1 |
| Multi-query expansion | 2 days | medium | Cursor, Cody | P1 |
| Diff-aware retrieval | 2 days | high | Cursor, Copilot | P1 |
| Eval framework | 3 days | critical | Claude Context | P1 |
| LateOn-Code / ColBERT | 2-3 weeks | very high | **nobody in our space** | P1 |
| Hybrid regex+semantic | 3 days | high | ColGrep only | P2 |
| Matryoshka dimensions | 2 days | medium | Copilot | P2 |
| Call graph extraction | 2 weeks | medium | grepai | P2 |
| Feedback loop | 2 weeks | medium | nobody OSS | P3 |
| Multi-repo | 1 week | medium | Kit | P3 |
| Streaming index via MCP | 2 days | medium | nobody | P3 |
| Homebrew/crates.io | 2 days | critical (adoption) | grepai, Probe | P1 |
| VS Code extension | 1 week | high (adoption) | Claude Context | P2 |

---

## What Makes This "Can't Live Without"

The thesis: **agents spend 60%+ of their first turn finding context** (Claude Code's own data). Every token wasted on search is a token not spent on reasoning. skelesearch cuts that to near-zero with:

1. **Instant concept search** — "find retry logic" works even when the code uses `with_backoff` and `attempt_loop`
2. **Precision over recall** — token-budget-aware output + MMR diversity + reranking = only relevant results
3. **Zero maintenance** — file watcher keeps index fresh, embedding cache makes re-index instant
4. **Universal agent compatibility** — MCP server works with every major agent (Claude Code, Codex, Cursor, Windsurf, OMP)
5. **Best-in-class retrieval** — late interaction (ColBERT) + hybrid BM25+vector + reranking = quality that matches Cursor's internal pipeline, as an open-source standalone tool

The moat: nobody else combines late-interaction retrieval + hybrid search + local+cloud embeddings + MCP + production quality in one open-source package. Cursor has the quality but it's proprietary and locked to their IDE. We make that quality available to every agent.

---

## Technical Decisions Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Vector DB | CozoDB (HNSW + FTS in one DB) | Single dependency. Migration path to LanceDB+Tantivy documented. |
| Chunking | tree-sitter AST | Consensus approach. 15 languages. Sliding-window fallback. |
| Fusion | RRF (Reciprocal Rank Fusion) | Proven, parameter-free. DBSF considered for v2. |
| Default embedding | jina-v2-base-code (local) | Code-specialized, 768-dim, fast ONNX. |
| Cloud embedding | OpenAI text-embedding-3-small | Cheapest, widest adoption. voyage-code-3 for quality. |
| Reranker (planned) | Qwen3-Reranker (Apache-2.0) | Open license. Jina v3 is CC BY-NC. |
| Late interaction (planned) | LateOn-Code 130M | Best code ColBERT. ONNX. MIT license. |
| Language | Rust | Performance, single binary, no runtime deps. |
| Interface | MCP + CLI | MCP for agents, CLI for humans. HTTP for non-subprocess consumers. |

---

*Last updated: 2026-03-18*
*Sources: Parallel deep research across academic papers, competitor repos, pricing pages, and blog posts.*
*Companion: `docs/competitive-landscape.md`, `docs/future-improvements.md`*
