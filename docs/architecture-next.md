# Architecture Next — Getting to 0.9 and Beating Cursor on DX

## Current State
- R@5=0.567, R@10=0.600, MRR=0.454 on 15 eval cases (skelegent repo)
- Perfect on symbol/keyword queries, zero on vocabulary-mismatch conceptual queries
- Single-shot pipeline: embed → HNSW+BM25 → RRF → MMR → budget filter

## Root Cause of Failures
All 4 zero-score queries fail because neither the embedding nor BM25 can bridge
the vocabulary gap between natural language and code identifiers:
- "remember things" → code: `StateStore`, `MemoryTier`, `Lifetime`
- "approval flow" → code: `ApprovalRequest`, `PendingToolCall`
- "secrets leaking" → code: `RedactionMiddleware`, `ExfilGuard`
- "hand off work" → code: `Effect::Handoff`, `Delegate`

## The Three Moves to 0.9

### 1. LLM Query Expansion (biggest impact, 2 days)
When a query looks conceptual (heuristic: has stop words, >3 words, no camelCase),
call the LLM to generate 3-5 code-vocabulary synonyms:
- "how do agents remember things" → "state persistence, memory store, StateStore, session"
- "prevent secrets leaking" → "redaction, secret filtering, credential masking, ExfilGuard"

Then search with both original AND expanded terms (BM25 gets keywords, vector gets both embeddings).

**Evidence:** Jina AI benchmarks show +1.5 NDCG for strong models, +6.5 for weaker ones.
QECK paper showed 64% precision improvement for code search via keyword expansion.

**Implementation:**
- Add `QueryExpander` trait to `crates/core/src/`
- Implement `LLMQueryExpander` that calls the configured embedding provider's inference API
  (OpenAI completions for openai provider, or a configurable endpoint)
- Gate on query classification: symbol queries skip expansion entirely
- Add `--expand` flag (default: true for smart_search, false for search_code)

### 2. API-Based Reranker (medium impact, 2 days)
The Reranker trait exists. Add a concrete implementation using Jina Reranker API:
- Free tier: 10M tokens
- Endpoint: POST https://api.jina.ai/v1/rerank
- Code-specific model: jina-reranker-v2-base-multilingual
- Request: `{model, query, documents, top_n}`

All three cloud reranker APIs (Jina, Cohere, Voyage) share nearly identical REST interfaces.
One implementation with provider-specific config covers all three.

**Evidence:** Dedicated rerankers beat LLM-as-reranker by 12-15% NDCG at 25-60x lower cost.

### 3. Session-Aware Dedup (low effort, high DX impact, 1 day)
Track which chunks an agent has already seen in a session. On subsequent searches,
either exclude or deprioritize already-seen results.

**Evidence:** Only Probe has shipped this. Agents run 3-4 rapid searches; without dedup
they re-read the same code blocks. Server-side tracking survives context compaction.

**Implementation:**
- Add `session_id: Option<String>` to search_code and smart_search
- Server-side `HashMap<String, HashSet<content_hash>>` tracks seen chunks per session
- After ranking, move already-seen results to bottom or exclude

## Modularity Gaps to Fix

### Current trait boundaries (good)
- `EmbedProvider` — 2 impls, factory, CLI flag, runtime swap ✓
- `StorageBackend` — trait exists, 1 impl, documented migration path ✓
- `Reranker` — trait exists, NoopReranker only, not in CLI/MCP ✗

### Missing trait boundaries (need)
| Trait | What it abstracts | Priority |
|---|---|---|
| `QueryExpander` | LLM expansion, keyword extraction, HyDE | P0 |
| `RerankerProvider` | Jina/Cohere/Voyage/local ONNX reranker | P0 |
| `Chunker` | Tree-sitter vs sliding window vs custom | P2 |
| `ScoreFuser` | RRF vs DBSF vs weighted sum | P2 |
| `Pipeline` | Compose stages declaratively | P3 |

### Configuration target (.skelesearch.toml)
```toml
[index]
exclude = ["target/", "node_modules/"]
provider = "openai"            # embedding provider
# provider = "fastembed"       # local alternative

[search]
top_k = 10
diversity = 0.3
max_tokens = 8192

[search.reranker]
provider = "jina"              # or "cohere", "voyage", "none"
api_key_env = "JINA_API_KEY"   # env var name
top_n = 50                     # rerank this many candidates

[search.expansion]
enabled = true                 # LLM query expansion for conceptual queries
provider = "openai"            # which LLM to use for expansion
```

## DX Strategy — Beating Cursor

### The insight
Cursor's advantage is IDE integration (sees your open files, cursor position, edit history).
skelesearch can't match that as a standalone tool. But we can beat Cursor on:
1. **Universality** — works with ANY agent (Claude Code, Codex, OMP, Cursor itself)
2. **Zero vendor lock-in** — swap embedding models, DBs, rerankers independently
3. **Token efficiency** — maxTokens budget + session dedup = less context pollution
4. **Privacy** — local-first with optional cloud enhancement

### Distribution (the real gap)
Probe wins adoption because: `npx -y @probelabs/probe mcp`

skelesearch needs the same:
```bash
# Target: one command, works everywhere
claude mcp add skelesearch -- npx -y @skelesearch/mcp

# Implementation: npm package wrapping native binary (esbuild pattern)
# postinstall script downloads prebuilt binary for platform
# npm package is a thin shell wrapper
```

### Lite mode (zero-config)
Ship a mode that works WITHOUT embeddings (BM25 + AST chunking only).
No API keys, no model downloads, no indexing wait. Just:
```bash
skelesearch search "retry logic" .
```
Works in seconds. Add `--provider openai` or configure .skelesearch.toml for
semantic upgrade. Quality degrades gracefully without embeddings.

### Claude Code pain points we already solve
| Pain point | Claude Code today | skelesearch |
|---|---|---|
| Grep false negatives | ~50% failure rate | Hybrid BM25+vector catches what grep misses |
| Context flooding (100K+ matches) | No truncation, crashes SDK | maxTokens budget, ranked results |
| No semantic understanding | "completely blind" to structure | Embedding-based concept search |
| Creates instead of finding | grep fails → assumes nonexistent | Semantic search finds related code |
| Context degradation over sessions | Loses track after compactions | Server-side session state (planned) |

## Reranker API Comparison
| Provider | Best Model | Pricing | Free Tier | Latency | Code-specific |
|---|---|---|---|---|---|
| Jina | reranker-v2-base-multilingual | $0.02/1M tok | 10M tokens | fast | **yes** |
| Cohere | rerank-v3.5 | $2/1K searches | 1K calls/mo | 172ms | no |
| Voyage | rerank-2.5 | $0.05/1M tok | 200M tokens | unknown | no |

**Recommendation:** Jina (code-specific model + generous free tier) as default,
Voyage (cheapest) as alternative, Cohere (highest quality) as premium option.

---
*Synthesized from 4 parallel research tasks, 2026-03-19*
*Sources: Jina/Cohere/Voyage API docs, Claude Code GitHub issues, Probe/ColGrep repos,
Haystack/LangChain/Continue.dev architectures, Cursor/Windsurf/Cody blog posts*
