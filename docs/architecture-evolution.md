# Architecture Evolution: Progressive Materialization

**Status:** Design direction. Not yet implemented.
**Date:** 2026-03-22
**Context:** Ablation study showed skelesearch's pipeline features (graph, expansion, reranker)
add marginal value over bare hybrid retrieval. The architecture needs to evolve from
"flat vector store with bolted-on features" to "progressively materialized knowledge graph."

---

## Core Insight

CozoDB already blurs the line between vector DB, graph DB, and relational store. We're
treating it as a vector store with side tables. The competitive advantage isn't in the
embedding model or reranker (commodity) — it's in structural analysis + Datalog-powered
retrieval that combines vector similarity with graph traversal in a single query.

## Three-Tier Materialization

### Tier 1: Immediate (seconds)

What happens on first connect / `index` command:
- Naive chunking — token-window splits or file-level chunks
- Embed and index into HNSW
- Basic FTS index on normalized text
- **User can search immediately**

This is what every other tool does. It's the baseline.

### Tier 2: Background (minutes)

Runs after Tier 1 is serving, progressively replaces naive chunks:
- AST-aware chunking via tree-sitter (function/class boundaries)
- Import edge resolution (file-level dependency graph)
- Call site extraction (@reference.call joins)
- Symbol enrichment in FTS normalized field
- Path + type prefix for BM25 disambiguation

As each file completes Tier 2 analysis, its chunks are upgraded in place.
Queries prefer Tier 2 chunks when available. Users see results get better
over time without any "switch."

### Tier 3: Deep Analysis (minutes to hours, cached)

Runs after Tier 2 completes for the whole repo:
- Cross-file dependency graph (full symbol resolution)
- PageRank / importance scoring over the import graph
- Semantic clustering of related code
- Module-level summary embeddings
- Common query pattern bundles (pre-materialized retrieval sets)

Tier 3 results are cached keyed by content hashes. Persist across sessions
in default mode (Tier 1 only); future: persistent caching for long-running sessions.

### Change Detection

When a user returns and the repo has changed:
1. Diff file content hashes against stored manifest
2. Invalidate affected chunks, edges, and downstream analysis
3. Tier 1 naive chunks regenerate in seconds (user never waits)
4. Tier 2 structural analysis trickles in behind it
5. Tier 3 only recomputes affected clusters

This is Nix-style content addressing applied to individual files,
not whole corpora.

## Datalog-Powered Retrieval (The Real Differentiator)

Nobody else does this. Current tools: embed query → ANN search → return top-k.

With CozoDB, a single Datalog query can:
1. Find chunks semantically similar to the query (HNSW)
2. Walk the call graph outward 2 hops from matched symbols
3. Pull in type definitions those symbols reference
4. Rank by fusion of vector similarity + graph distance + BM25

In a standard vector DB + separate graph DB, that's three round trips
and manual stitching. In CozoDB it's one query.

### What We Do Now (Broken)

```
Query → embed → [HNSW + BM25 parallel] → RRF fusion → PageRank boost
      → graph augment (fetch ALL chunks from imported files) → MMR → rerank
```

Graph augmentation is a post-hoc bolt-on: run vector search, THEN walk
the graph, THEN merge. The graph phase doesn't know what the query is
about — it just dumps everything from imported files.

### What We Should Do

```datalog
# Single Datalog query: hybrid retrieval + graph walk + type resolution
# CozoDB can do this natively.

vec_hits[fp, ci, score] :=
    ~chunks:embedding{ fp, ci | query: $q_vec, k: 50, ef: 100,
                       bind_distance: d, radius: 0.8 },
    score = 1.0 / (60.0 + d)

fts_hits[fp, ci, score] :=
    ~chunks:text{ fp, ci | query: $q_str, k: 50,
                  score_kind: 'tf_idf', bind_score: s },
    score = 1.0 / (60.0 + 1.0 / (s + 0.001))

# RRF fusion
base[fp, ci, sum(score)] := vec_hits[fp, ci, score]
base[fp, ci, sum(score)] := fts_hits[fp, ci, score]

# Graph walk: importers/callers of top hits (1 hop)
graph[fp, ci, parent_score * 0.5] :=
    base[target_fp, _, parent_score],
    parent_score > $threshold,
    *code_edges[fp, _, target_fp, _, _],
    *chunks[fp, ci, _, _, _, _, _, emb],
    !is_null(emb)

# Union base + graph hits
?[fp, ci, content, score, why] :=
    base[fp, ci, score],
    *chunks[fp, ci, content, _, _, _, _, _],
    why = 'hybrid'
?[fp, ci, content, score, why] :=
    graph[fp, ci, score],
    *chunks[fp, ci, content, _, _, _, _, _],
    why = 'graph',
    not base[fp, ci, _]  # don't duplicate

:order -score
:limit $top_k
```

This is one round-trip. The graph walk happens inside the query engine,
not as a separate Rust function with N+1 queries.

## Precomputed Query Bundles

If the system notices agents frequently retrieve certain file clusters
together (auth code, config code, database code), pre-materialize those
bundles as stored relations. When a query hits a known cluster, serve
from the precomputed bundle instead of the full pipeline.

This is a query cache informed by usage patterns. Nobody else does this
because nobody else has the structural metadata to identify the clusters.

## Implementation Path

1. **Add `materialization_tier` column to chunks** — 1, 2, or 3.
   Retrieval queries prefer higher tiers.
2. **Background worker** — after initial index, run Tier 2 analysis per file.
   Write upgraded chunks with tier=2.
3. **Single Datalog retrieval query** — replace the current
   hybrid_search → augment_with_graph → rerank pipeline with one query.
4. **Change detection** — content-hash-based invalidation on re-index.
5. **Tier 3 caching** — persist across sessions.
6. **Query bundle detection** — log retrieval patterns, identify clusters,
   pre-materialize.

## What This Means for the Current Code

- `searcher.rs` search pipeline should converge toward a single Datalog query
  instead of sequential Rust phases
- `augment_with_graph` should be absorbed into the retrieval query
- `schema.rs` hybrid_search should compose vector + FTS + graph in Datalog
- The current architecture is a stepping stone, not the destination
