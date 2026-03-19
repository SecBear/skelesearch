# skelesearch v2 Roadmap — Beat Every Competitor

> Goal: The definitive open-source semantic code search engine for agentic systems.
> No competitor has all of these. We will.

## Current state (v1.3)

**Shipped:**
- Tree-sitter AST chunking (15 languages + sliding-window fallback)
- Hybrid BM25 + vector search with Reciprocal Rank Fusion
- MMR diversity reranking
- Embedding cache (content-hash keyed, SQLite-backed)
- Multi-query expansion (keyword extraction for BM25 boost)
- Local ONNX embeddings (jina-v2-base-code, 768-dim)
- Cloud embeddings (OpenAI text-embedding-3-small, 1536-dim)
- MCP server (stdio + HTTP transport)
- CLI with index/search/grep/symbol/context/status/gc/watch/clear/eval
- Production observability (#[instrument] spans, timing, cache counters)
- File watcher re-indexing (notify, 2s debounce)
- Token-budget-aware retrieval (--max-tokens)
- Diff-aware branch-scoped search (--branch)
- Provider manifest (stores which model indexed, auto-detect on search)
- Reranker trait + pipeline stage (NoopReranker, ready for concrete impl)
- Eval framework (Recall@5, Recall@10, MRR)
- Post-edit hooks for automatic background re-indexing

**Eval baseline (skelegent, 370 files, 2753 chunks, OpenAI embeddings):**
- R@5=0.567, R@10=0.600, MRR=0.454 (15 cases)
- Perfect on symbol/keyword queries, zero on vocabulary-mismatch conceptual queries

**Root cause of failures:** Not embedding quality — vocabulary mismatch between
natural language ("remember things") and code identifiers (`StateStore`). Needs
query expansion at the architecture level, not a model swap.

---

## Implementation Queue (priority order)

### P0: LLM Query Expansion (2 days) — estimated +20-30% R@5
The single highest-impact change. Bridges the vocabulary gap that causes all 4 eval failures.

**What:**
- `QueryExpander` trait in `crates/core/src/expander.rs`
- Query classifier (10-line heuristic: camelCase/snake_case → symbol, stop words + >3 words → conceptual)
- `LLMQueryExpander` implementation using OpenAI completions (or any configured LLM)
- Prompt: "Given this code search query, list 3-5 code identifiers/keywords that might appear in relevant source files. Query: {query}. Keywords:"
- Gate: symbol queries skip expansion entirely, only conceptual queries get LLM call
- Search with both original embedding AND expanded-keyword BM25

**Evidence:** QECK paper: +64% precision for code search via keyword expansion. Jina benchmarks: +1.5 to +6.5 NDCG.

**Files:** `crates/core/src/expander.rs` (new), `crates/core/src/searcher.rs`, `crates/core/src/lib.rs`, `crates/mcp/src/server.rs` (smart_search path), `crates/cli/src/app.rs`

### P1: Jina Reranker API (2 days) — estimated +5-15% nDCG
Concrete implementation of the Reranker trait using Jina's code-specific reranker.

**What:**
- `crates/rerank-api/` crate implementing `Reranker` trait
- Jina reranker-v2-base-multilingual (code-specific, free 10M tokens)
- Unified REST client covering Jina/Cohere/Voyage (nearly identical APIs)
- Config: `[search.reranker]` in .skelesearch.toml
- CLI: `--reranker jina` (default: none)
- MCP: auto-enable when configured

**API format (shared across all 3 providers):**
```
POST https://api.jina.ai/v1/rerank
{ "model": "jina-reranker-v2-base-multilingual", "query": "...", "documents": [...], "top_n": 10 }
→ { "results": [{ "index": 0, "relevance_score": 0.84 }] }
```

**Evidence:** Dedicated rerankers beat LLM-as-reranker by 12-15% NDCG at 25-60x lower cost.

**Pricing:**
| Provider | Free Tier | Paid | Code-specific |
|---|---|---|---|
| Jina | 10M tokens | $0.02/1M tok | **yes** (v2) |
| Voyage | 200M tokens | $0.05/1M tok | no |
| Cohere | 1K calls/mo | $2/1K searches | no |

### P2: voyage-code-3 Embedding Provider (1 day) — estimated +5-10% R@5
Best-in-class code embedding model. 13.8% better than OpenAI on code retrieval.

**What:**
- `crates/embed-voyage/` crate implementing `EmbedProvider`
- voyage-code-3: 1024-dim, 32K context, 300+ programming languages
- API: `POST https://api.voyageai.com/v1/embeddings`
- Auth: `VOYAGE_API_KEY` env var
- CLI: `--provider voyage`

**Evidence:** Voyage AI's 32-dataset suite: 92.12% vs OpenAI 78.48%.

### P3: Session Dedup (1 day) — DX improvement
Prevent agents from re-reading the same code across multiple searches in one session.

**What:**
- Add `session_id: Option<String>` to SearchCodeInput and SmartSearchInput
- Server-side `HashMap<String, HashSet<u64>>` tracking content hashes per session
- After ranking, deprioritize (not exclude) already-seen results
- CLI: `--session <id>` flag

**Evidence:** Only Probe has this. Agents run 3-4 rapid searches; without dedup they
waste context on repeated code blocks. Server-side tracking survives context compaction.

### P4: Expand Eval to 100+ Cases (2 days) — measurement quality
Current 15-case eval has enormous confidence intervals. Need reliable measurement
before optimizing further.

**What:**
- Auto-generate eval cases from resolved GitHub issues (Morph Labs methodology)
- Cover 3+ repos (skelegent, skelesearch itself, a well-known OSS project)
- Mix: 30% symbol, 30% implementation, 20% architectural, 20% hard conceptual
- Run against CodeSearchNet subset for external benchmark comparability
- Publish results in README

### P5: LanceDB+Tantivy Backend (1 week) — perf + maintainability
Feature-gated alternative to CozoDB. Don't remove CozoDB — let both coexist.

**What:**
- `crates/core/src/lance_backend.rs` implementing `StorageBackend` (16 methods)
- LanceDB for vector storage (IVF+PQ, columnar, metadata filtering)
- Tantivy for BM25 FTS (3x faster than Elasticsearch, 14.7K stars)
- `open_backend()` selects based on `.skelesearch.toml` config or `--backend` flag
- Manual BFS for `traverse_imports` (replaces CozoDB Datalog recursive query)

**Blast radius:** 1 new file + 2 construction sites. Everything else is behind the trait.

**Evidence:** CozoDB last release v0.7.6 (Dec 2023). Tantivy actively maintained (14.7K stars).
LanceDB: 9.5K stars, 2000+ commits, Rust-native.

### P6: ColBERT/LateOn-Code (2-3 weeks) — estimated +15-25% R@5
Late interaction retrieval. No OSS MCP code search tool has this. The moat.

**What:**
- `crates/embed-lateon/` crate using `next-plaid` + `next-plaid-onnx`
- LateOn-Code 130M (ONNX, Apache-2.0, ModernBERT-based)
- PLAID algorithm for multi-vector storage (product quantization, mmap, SIMD MaxSim)
- Architecture: next-plaid alongside CozoDB/LanceDB, fuse ColBERT + BM25 scores
- rusqlite 0.38 upgrade already done (unblocks next-plaid dependency)

**Evidence:** ColGrep benchmark: 70% win rate vs grep, 15.7% average token savings.

### P7: Distribution (3 days) — adoption
Zero-friction install for every agent platform.

**What:**
- npm package wrapping native binary (esbuild/turbo pattern: postinstall downloads platform binary)
- `npx -y @skelesearch/mcp` one-liner for Claude Code, Codex, OMP
- Homebrew tap: `brew install skelesearch`
- GitHub Releases with prebuilt binaries (Linux x86_64, macOS arm64/x86_64)
- cargo install from crates.io

### P8: Lite Mode (2 days) — zero-config DX
BM25-only search with no API keys, no model downloads, no indexing wait.

**What:**
- `skelesearch search "query" .` works with BM25 + AST chunking only
- No embedding provider needed — tree-sitter chunks stored in CozoDB/LanceDB FTS
- Instant first run (index on first search, incremental thereafter)
- Add `--provider fastembed|openai|voyage|none` where `none` = BM25-only
- Quality degrades gracefully; configure a provider for semantic upgrade

**Evidence:** Probe wins adoption with zero-config (`npx` one-liner). Our moat is
quality, but users need to try it first.

### P9: Advanced (ongoing)
- Call graph extraction (tree-sitter → function call edges)
- Matryoshka adaptive dimensions (CodeSage v2, nomic-embed-text-v1.5)
- Retrieval feedback loop (log query→used_results, tune RRF weights per-repo)
- Multi-repo indexing
- Streaming index via MCP (`index_file` tool for single-file updates)
- cAST dynamic chunk sizing (ChunkHound algorithm)
- VS Code extension

---

## Priority Matrix

| # | Feature | Effort | R@5 Impact | DX Impact | Competitors Have It |
|---|---|---|---|---|---|
| P0 | LLM query expansion | 2d | **+20-30%** | medium | Cursor (proprietary) |
| P1 | Jina reranker | 2d | **+5-15%** | medium | Continue.dev only |
| P2 | voyage-code-3 | 1d | +5-10% | none | Continue.dev |
| P3 | Session dedup | 1d | none | **high** | Probe only |
| P4 | Expand eval | 2d | measurement | none | Claude Context |
| P5 | LanceDB backend | 1w | perf | none | nobody (novel) |
| P6 | ColBERT | 2-3w | **+15-25%** | none | **nobody in MCP** |
| P7 | npm distribution | 3d | none | **critical** | Probe, grepai |
| P8 | Lite mode | 2d | none | **critical** | Probe |
| P9 | Advanced features | ongoing | varies | varies | varies |

## Target Metrics

| Metric | Current | After P0-P1 | After P0-P6 | Best-in-class (Cursor) |
|---|---|---|---|---|
| R@5 | 0.567 | 0.80-0.85 | 0.90+ | ~0.90 (estimated) |
| R@10 | 0.600 | 0.85-0.90 | 0.95+ | ~0.95 |
| MRR | 0.454 | 0.70-0.80 | 0.85+ | ~0.85 |

---

## What Makes This "Can't Live Without"

Your agent spends half its tokens LOOKING for code. skelesearch finds it in one shot.

- **97% token reduction** vs grep-only workflows (grepai benchmark on 155K LOC)
- **12.5% higher accuracy** vs grep (Cursor's own A/B test)
- **Zero context pollution** — token budget + session dedup + precision-first ranking
- **Universal** — works with Claude Code, Codex, OMP, Cursor, Windsurf via MCP
- **Private** — local-first, your code never leaves your machine (unless you choose cloud embeddings)
- **Modular** — swap embedding models, rerankers, and storage backends independently

## Technical Decisions Log

| Decision | Choice | Rationale |
|---|---|---|
| Default storage | CozoDB (HNSW + FTS) | Single dependency. LanceDB+Tantivy as feature-gated alternative. |
| Chunking | tree-sitter AST | Consensus. 15 languages. Sliding-window fallback. |
| Fusion | RRF | Proven, parameter-free. DBSF considered for v2. |
| Default local embedding | jina-v2-base-code | Code-specialized, 768-dim, fast ONNX. Upgrade to CodeRankEmbed planned. |
| Cloud embedding | OpenAI text-embedding-3-small | Cheapest. voyage-code-3 for quality (P2). |
| Reranker | Jina API (code-specific, free tier) | Best code reranker. Cohere/Voyage as alternatives. |
| Late interaction | LateOn-Code 130M via next-plaid | Best code ColBERT. ONNX. Apache-2.0. PLAID storage. |
| Query expansion | LLM keyword extraction, gated by query classifier | QECK: +64% precision. HyDE rejected (Continue.dev removed it). |
| Language | Rust | Performance, single binary, no runtime deps. |
| Interface | MCP + CLI | MCP for agents, CLI for humans. HTTP for non-subprocess. |
| Distribution | npm-wrapped native binary | `npx -y @skelesearch/mcp` one-liner. Homebrew + cargo install as alternatives. |

---

*Last updated: 2026-03-19. Revised from original roadmap based on consolidated research
from 6 internal research tasks, 1 external analysis, and eval data from 15-case skelegent benchmark.*
