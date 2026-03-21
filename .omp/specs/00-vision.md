# skelesearch — Vision

The best code retrieval engine for coding agents. Graph-aware, structure-aware,
token-budget-aware. First retrieval tool on the ContextBench and CoIR leaderboards.

## Linear Project
- **Team:** Personal Development (PER)
- **Project:** skelesearch
- **URL:** https://linear.app/seabear/project/skelesearch-2a7e3caa17c5

## Product Split
- **skelesearch (public, Apache-2.0):** Core engine, CLI, local MCP, benchmarks
- **skelesearch-cloud (private):** Hosted API, billing, webhook indexing, dashboard
- See `docs/repo-structure-plan.md` for architecture

## Requirements
- [x] Hybrid BM25 + vector + RRF retrieval with cross-encoder reranking
- [x] AST-aware chunking (15 languages via tree-sitter)
- [x] Import graph + PageRank + depth-decayed graph augmentation
- [x] MCP server (stdio + HTTP) with 8 tools
- [x] CLI with index/search/grep/symbol/context/eval/watch
- [x] CoIR benchmark adapter (embed-only + hybrid modes)
- [x] ContextBench adapter (file + line F1 metrics)
- [ ] Full CoIR 10-task run with publishable scores
- [ ] Full ContextBench verified run (500 instances)
- [ ] npm distribution (`npx -y @skelesearch/mcp`)
- [ ] Cross-file call graph edges (Phase B1)
- [ ] Hosted service MVP (API + webhook indexing + billing)
- [ ] Public launch (repo, blog, landing page, leaderboard submission)

## Non-goals
- General document/PDF/note search (that's gno's lane)
- Web UI / second-brain product
- Conversational memory (separate future product)
- SSO / enterprise features before product-market fit
- Supporting every embedding model — focus on fastembed (local) + voyage (cloud)

## Key Metrics
- **CoIR nDCG@10:** Current 0.442 (CoSQA only). Target: full 10-task average.
- **ContextBench Line F1:** Target > 0.25. Matching agent SOTA (0.34) is headline.
- **Internal R@5:** Current 86.7% (240 cases, voyage-full).
- **MRR weak repos:** zod 63.6%, httpx 65.2% — target > 75%.

## Competitive Position
- No competitor publishes standard IR metrics
- First mover on ContextBench + CoIR leaderboards = defensible moat
- Differentiation: graph + retrieval plans + published metrics + local-first
