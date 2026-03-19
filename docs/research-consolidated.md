# Consolidated Research — March 2026

Cross-referenced findings from 6 parallel research tasks + external analysis.

## Consensus (all sources agree)

1. **Tree-sitter AST chunking is correct.** Industry standard. No change needed.
2. **Hybrid BM25+vector with RRF fusion is correct.** Table stakes architecture.
3. **jina-v2-base-code is stale.** voyage-code-3 (API) or CodeRankEmbed (local) are the upgrades.
4. **Cross-encoder reranker is high-leverage.** Pipeline: retrieve → RRF → rerank → MMR → budget.
5. **CozoDB is risky long-term.** StorageBackend trait contains the blast radius. LanceDB+Tantivy is the migration path.
6. **Cursor's feedback loop is the real moat.** Can't replicate without usage data + fine-tuning.
7. **Eval corpus needs expansion.** 15 cases insufficient for reliable measurement.

## Disagreements (resolved with evidence)

### "Embedding model swap is the highest-leverage fix"
**External research says:** Swap model immediately, +0.1 to +0.2 R@5.
**Our eval data says:** All 4 failures are vocabulary mismatch, not embedding quality. "remember things" vs `StateStore` — no embedding model bridges that gap without query expansion.
**Verdict:** Model swap helps ~5-10%. LLM query expansion helps ~20-30%. **Architecture > model.**

### "HyDE dramatically improves recall"
**External research says:** Implement HyDE for code search.
**Our research found:** Continue.dev tried HyDE and removed it. Elastic blog: marginal gains when hybrid search already in play. LLM hallucinated code snippets may embed into wrong neighborhoods.
**Verdict:** **LLM keyword expansion > HyDE.** Constrained prompt asking for synonyms outperforms free-form code generation.

### "No incremental indexing is a hard blocker"
**External says:** skelesearch does full re-indexing.
**Fact:** We have embedding cache (content-hash keyed SQLite), manifest mtime/hash tracking, and file watcher with 2s debounce. Re-index of unchanged 370-file repo: 0.0s.
**Verdict:** **Factually wrong about our current state.**

## New Signals Worth Acting On

### From external research
- **cAST algorithm (ChunkHound)** — dynamic chunk sizing on tree-sitter. Worth investigating for v2.
- **Qwen3-Embedding** — reportedly SOTA on multilingual+code. Add as provider option.
- **usearch** — Unum's Rust HNSW, int8/binary quantization. More optimized than CozoDB's HNSW. Relevant for LanceDB migration.
- **BGE-Reranker-v2-m3** — open weights (Apache 2.0), strong code support. Self-hostable alternative to API rerankers.
- **CodeSage Large v2** — Matryoshka + The Stack v2 training. Good MRL-compatible option.

### From our research
- **Jina Reranker API** — free 10M tokens, code-specific model, identical REST API across providers.
- **LLM keyword extraction gated by query classification** — simplest effective expansion (10-line classifier + 1 LLM call for conceptual queries only).
- **Session dedup** — only Probe has this. Server-side tracking survives context compaction.
- **npm-wrapped native binary** — esbuild pattern for zero-config `npx -y @skelesearch/mcp`.
- **Lite mode** — BM25-only without API keys for instant first run. Semantic upgrade when configured.

## Revised Priority Stack

| # | Action | Impact | Effort | Evidence |
|---|---|---|---|---|
| 1 | LLM query expansion (keyword extraction, gated) | **+20-30% R@5** | 2 days | QECK paper: +64% precision for code search |
| 2 | Jina reranker API (concrete Reranker impl) | **+5-15% nDCG** | 2 days | Dedicated rerankers beat LLMs by 12-15% |
| 3 | voyage-code-3 provider | **+5-10% R@5** | 1 day | 13.8% better than OpenAI on code |
| 4 | Session dedup | **DX** (less context pollution) | 1 day | Only Probe has this; agents need it |
| 5 | Expand eval to 100+ cases | **measurement quality** | 2 days | 15 cases = unreliable confidence intervals |
| 6 | LanceDB+Tantivy backend | **perf + maintainability** | 1 week | Tantivy 3x faster BM25, LanceDB actively maintained |
| 7 | ColBERT/LateOn-Code | **+15-25% R@5** | 2-3 weeks | 70% win rate vs grep (ColGrep benchmark) |
| 8 | npm distribution | **adoption** | 3 days | `npx -y @skelesearch/mcp` one-liner |
| 9 | Lite mode (BM25-only) | **zero-config DX** | 2 days | Probe's zero-config wins adoption |

## LanceDB Migration Analysis

**Can we swap CozoDB for LanceDB without rewriting everything?** Yes.

- `StorageBackend` trait (16 methods) is the exact boundary needed
- Blast radius: 1 new file + 2 construction sites (`open_backend()`)
- Everything else (Indexer, Searcher, Reranker, Manifest, CLI, MCP) is generic over the trait
- `traverse_imports` Datalog query becomes manual BFS (~20 lines)
- Feature-gate: `--features lance` vs default CozoDB, both can coexist

**Recommendation:** Build `LanceBackend` as a separate feature. Don't remove CozoDB yet — let users choose. Compare eval scores on same dataset. Ship the winner as default.

---
*Consolidated 2026-03-19 from: 4 internal research tasks (reranker APIs, LLM expansion, modular architectures, DX strategy), 1 external analysis, eval data from 15-case skelegent benchmark.*
