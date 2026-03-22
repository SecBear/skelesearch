# Agent-Facing Design Principles

**Status:** Active design document. All architecture decisions must pass these criteria.
**Date:** 2026-03-22
**Evidence:** Agent trace analysis (14 real queries from OMP sessions) + competitive analysis.

---

## Core Asymmetry

Humans can't read code fluently. Agents can.

This means every design decision that bridges natural language → code (description
embeddings, code summaries, NL explanations) solves a **human** problem. skelesearch
is agent-facing. Design for the agent's strengths, not the human's weaknesses.

## How Agents Query Code Search

From trace analysis (OMP sessions, 2026-03-22):

| Pattern | % | Example | What works |
|---|---|---|---|
| keyword_code | 64% | `"RRF reciprocal rank fusion weight BM25 HNSW"` | BM25 + code HNSW |
| symbol | 21% | `"EmbedProvider trait fastembed"` | BM25 exact match |
| conceptual | 14% | `"how does chunking split source code"` | Query expansion → BM25 |

Agents:
- Issue **multiple targeted queries**, not one broad one
- Pack **symbol names + descriptive keywords** together (6.6 words avg)
- Don't retry — get what they need first try
- Don't paste code snippets as queries
- Don't browse results — they consume directly into context window

## Agent vs Human Design Tradeoffs

| Dimension | Human-facing | Agent-facing (skelesearch) |
|---|---|---|
| Query precision | Low — vague NL | High — multiple targeted queries |
| Noise tolerance | High — humans skim | **Low — wasted context tokens** |
| Recall vs precision | Recall wins | **Precision wins** (top 3-5 must be right) |
| Code readability | Need NL bridge | **Read code natively** |
| Result format | Summaries, highlights | **Raw code with location** |
| Multi-hop | User does manually | **Agent does multiple queries** |
| Session state | Stateless | **Session-aware dedup matters** |

## Priority Order for Agent Value

1. **Chunking quality** — complete functions, not fragments. AST-aware chunking is
   worth more than any embedding improvement. An agent can't act on half a function.

2. **Symbol-level retrieval** — agents know names. `find_symbols("ConnectionPool")`
   is the agent's natural query mode. Most competitors don't have this.

3. **Import graph traversal** — "what depends on this?" is the structural query
   agents need before editing. Blast radius analysis.

4. **Precision-oriented reranking** — agents care about top 3-5 being right.
   Cross-encoder reranking on raw code > better first-stage recall.

5. **Session-aware dedup** — not re-returning chunks already in context window.
   Pure agent-facing value that human-facing tools don't need.

6. **Query expansion** — turn conceptual queries into precise symbol queries at
   query time. This solves the vocabulary mismatch at the right layer — no LLM
   dependency in the index path.

## What We Should NOT Do

- **Replace code embeddings with description embeddings.** Agents read code natively.
  Code-structural similarity in the embedding space is what agents rely on for the
  64% keyword_code query pattern.

- **Add LLM dependency to the index path.** Indexing must be fast, offline-capable,
  and deterministic. LLM calls in the index path add latency, cost, and a failure mode.

- **Optimize for conceptual queries at the expense of structural.** Our eval was 78%
  conceptual — that's a bias, not reality. Agents are 64% keyword_code + 21% symbol.

## Where Descriptions Fit (If At All)

1. **Result metadata** — 1-line summary helps agent decide which chunks to pull into
   context without reading full code. This is a formatting optimization, not indexing.

2. **Additional BM25 field** — FTS on descriptions gives conceptual queries a BM25
   path without touching HNSW. Cheap, no embedding cost.

3. **NOT as HNSW embedding source.** Code embeddings are primary. Period.
