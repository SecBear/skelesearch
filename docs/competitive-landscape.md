# Competitive Landscape — March 2026

## Direct Open-Source Competitors

| Tool | Stars | Language | Approach | MCP | Hybrid Search | File Watcher | License |
|------|-------|----------|----------|-----|---------------|-------------|---------|
| Claude Context (Zilliz) | 5,680 | Python | AST + OpenAI + Milvus Cloud | yes | no | no | MIT |
| grepai | 1,482 | Go | AST + Ollama/OpenAI + local | yes | no | **yes** | MIT |
| SeaGOAT | 1,278 | Python | Line-level + ChromaDB | no | regex+semantic | daemon | MIT |
| Kit (Cased) | 1,277 | Python | AST + configurable + ChromaDB | yes | no | no | MIT |
| cocoindex-code | 994 | Rust | AST + MiniLM/LiteLLM | yes | no | **yes** | Apache-2.0 |
| Probe | 501 | Rust | AST + BM25/TF-IDF (no embeddings) | yes | keyword only | N/A | Apache-2.0 |
| codeqai | 497 | Python | AST + sentence-transformers/FAISS | no | no | no | Apache-2.0 |
| DeepContext MCP | 270 | TypeScript | AST + Jina + Turbopuffer Cloud | yes | **yes** (reranker) | no | Apache-2.0 |
| semantic-search-mcp | 4 | Rust | AST + FastEmbed/ONNX + SQLite FTS5 | yes | **yes** (RRF) | no | MIT |
| mcp-vector-search | 25 | TypeScript | AST + CodeT5+ + LanceDB + KuzuDB | yes | no | no | Elastic-2.0 |
| Bloop (DEAD) | 9,516 | Rust | ONNX + Qdrant + Tantivy | no | yes | N/A | Apache-2.0 |

### Key observations
- **No open-source tool has all of:** AST chunking + hybrid BM25+vector + local+cloud embeddings + embedding cache + MCP + MMR diversity.
- **grepai** is the closest competitor in adoption — Go binary, Homebrew, call graph, file watcher. But no hybrid search.
- **cocoindex-code** is closest in tech stack — Rust engine, tree-sitter, MCP. But no BM25 hybrid.
- **semantic-search-mcp** is closest in architecture — hybrid RRF, FastEmbed ONNX, tree-sitter. But 4 stars, early.
- **Bloop** was the gold standard but is dead. Its architecture (Rust + Qdrant + Tantivy) is the reference.

## Commercial / IDE Built-in

| Tool | Approach | Published Claims | Key Insight |
|------|----------|-----------------|-------------|
| Cursor | Custom embedding + merkle reindex + Turbopuffer | 12.5% accuracy lift vs grep-only | Most sophisticated. $300M+ ARR. |
| Copilot | Custom contrastive model + Matryoshka dims | 37.6% retrieval quality lift, 8x smaller index | Best published benchmarks. |
| Cody (Sourcegraph) | Multi-retriever + transformer ranker | ~20% improvement with BM25F | RecSys '24 paper. Academic rigor. |
| Claude Code | NO EMBEDDINGS. Agentic grep/find/read. | 60%+ of first turn spent finding context | Anti-embedding. Token-expensive. |
| Codex | No semantic indexing. Agentic shell tools. | None | Community wants semantic search. |
| Windsurf | Hybrid + RL-trained retrieval subagent (SWE-grep) | 27% more code completed | Most innovative retrieval. |
| Continue.dev | Configurable RAG (transformers.js / voyage-code-3) | None | Most configurable. |
| Aider | Tree-sitter repo map + PageRank graph ranking | None | 42K stars, no embeddings. |
| Greptile | LLM-generated docstrings → embed docstrings | Scored 2.17/5 on Morph eval (lowest) | Unique approach, poor results. |

### Key insight from commercial
- **Precision > recall.** Windsurf proved context pollution is more harmful than missing context.
- **Hybrid always wins.** Cursor, Cody, Windsurf all converge on hybrid retrieval.
- **Claude Code and Codex explicitly lack semantic search** — both have open issues requesting it. This is our insertion point.

## Benchmarks

### Code Retrieval Benchmarks
| Benchmark | Source | What | Metrics | Relevance |
|-----------|--------|------|---------|-----------|
| CoIR | ACL 2025 | 10-dataset code retrieval (text→code, code→code) | NDCG@10 | HIGH — BEIR equivalent for code |
| RepoBench-R | ICLR 2024 | Cross-file snippet retrieval within a repo | Recall/Precision | VERY HIGH — closest to our use case |
| CodeSearchNet | 2019 | NL→code search, 2M pairs, 6 langs | MRR | Classic, widely cited |
| SWE-bench | 2023+ | File localization for real bug fixes | File accuracy | VERY HIGH — real repos |
| CodeRAG-Bench | NAACL 2025 | Does retrieval help generation? | Pass@1 | Shows naive RAG can hurt |
| CrossCodeEval | NeurIPS 2023 | Cross-file code completion requiring retrieval | Exact Match | Core use case |
| CoSQA+ | 2024 | Real Bing search queries → code | MRR, NDCG@10 | Realistic queries |
| CodeScaleBench | Sourcegraph 2025 | 220 tasks, cross-repo navigation | Task completion | Enterprise scale |

### Custom Evaluation Approaches
| Method | Who | How |
|--------|-----|-----|
| Morph Labs auto-eval | Morph Labs | Generate repo-specific questions from resolved GitHub issues. Score 1-5. |
| Voyage Code Suite | Voyage AI | 32-dataset suite. voyage-code-3: 92.12%, OpenAI: 78.48%. |
| Cursor Context Bench | Cursor | Internal: accuracy answering codebase questions with/without semantic search. |
| Sourcegraph BM25F eval | Sourcegraph | Internal quality metrics. ~20% improvement with BM25F. |

### Recommended KPIs for skelesearch
- Recall@5, Recall@10 on repo-specific eval set
- MRR on concept queries (NL → code function)
- Tokens-per-task (measures retrieval efficiency)
- Time-to-first-relevant-file
- Staleness incidents (how often index misses recent changes)

## Top Embedding Models for Code (March 2026)
| Model | Provider | Dims | Context | Access | Verdict |
|-------|----------|------|---------|--------|---------|
| voyage-code-3 | Voyage AI | 1024 | 32K | API ($0.18/MTok) | Best proprietary. 13.8% over OpenAI. |
| text-embedding-3-large | OpenAI | 3072 | 8191 | API ($0.13/MTok) | Strong default. |
| text-embedding-3-small | OpenAI | 1536 | 8191 | API ($0.02/MTok) | Best budget. |
| Nomic Embed Code | Nomic AI | 768 | 2048 | Open (Apache-2.0, 7B) | Best open-source large. |
| CodeRankEmbed | Nomic AI | 768 | 8192 | Open (MIT, 137M) | Best open-source small. |
| Jina Code V2 | Jina AI | 768 | 8192 | Open (Apache-2.0) | Current skelesearch default. |
| LateOn-Code | LightOn | ColBERT | N/A | Open weights | Novel late interaction. |

## skelesearch Positioning
**Unique combination:** CozoDB HNSW+FTS (both vector and BM25 in one DB) + tree-sitter AST chunking + embedding cache + MMR diversity + local ONNX + cloud embeddings. No competitor has all six.

**Market gap:** No open-source, MCP-compatible semantic code search engine combines all of: AST-aware chunking + hybrid BM25+vector + local+cloud embeddings + incremental indexing + production observability.

---
*Last updated: 2026-03-18. Source: parallel deep research across GitHub, academic papers, competitor blogs, and pricing pages.*
