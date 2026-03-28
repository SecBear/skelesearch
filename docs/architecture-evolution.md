# Architecture Evolution: Progressive Materialization

**Status:** Partially implemented. CompositeBackend is live on `main`. Three-tier
materialization and query-time graph integration are future work.
**Last updated:** 2026-03-28
**Context:** Ablation study showed skelesearch's pipeline features (graph, expansion, reranker)
add marginal value over bare hybrid retrieval. The architecture needs to evolve from
"flat vector store with bolted-on features" to "progressively materialized knowledge graph."

---

## Core Insight

The competitive advantage isn't in the embedding model or reranker (commodity) — it's in
structural analysis + **graph-integrated retrieval** that combines vector similarity with
import graph traversal in a single pipeline pass.

Current state (CompositeBackend, live on `main`):
- LanceDB: HNSW vector storage + relational tables via Apache Arrow
- Tantivy: BM25 full-text search with a code-aware tokenizer (CamelCase/snake_case splitting)
- petgraph: in-memory import graph for BFS traversal and PageRank

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

## Graph-Integrated Retrieval (The Real Differentiator)

Nobody else does this. Current tools: embed query → ANN search → return top-k.

With CompositeBackend, the retrieval pipeline can:
1. Find chunks semantically similar to the query (LanceDB HNSW)
2. Walk the import graph outward via petgraph BFS
3. Pull in type definitions those symbols reference
4. Rank by fusion of vector similarity + graph distance + BM25 (RRF)

### What We Do Now

```
Query → embed → [HNSW + BM25 parallel] → RRF fusion → PageRank boost
      → graph augment (fetch ALL chunks from imported files) → MMR → rerank
```

Graph augmentation is a post-hoc bolt-on: run vector search, THEN walk
the graph, THEN merge. The graph phase doesn't know what the query is
about — it just dumps everything from imported files.

### What We Should Do

Tight graph-vector fusion: use the query embedding to score graph-expanded
results instead of treating all expanded nodes equally. The BFS walk from
high-scoring seed chunks produces candidates; a second-pass vector similarity
against the query decides which graph nodes are actually relevant.

This stays in Rust (petgraph + LanceDB vector lookup), so it's one async
pipeline, not three sequential round-trips.

```
Query → embed
      → [HNSW top-50 + BM25 top-50] → RRF fusion → seed set
      → BFS(seeds, depth=2) → candidate expansion
      → vector re-score(candidates, query) → graph_score * graph_weight + vec_score
      → merge + rank → MMR → rerank
```

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
3. **Query-time graph re-scoring** — replace `augment_with_graph`'s flat BFS
   dump with a scored second-pass vector lookup over expanded candidates.
4. **Change detection** — content-hash-based invalidation on re-index.
5. **Tier 3 caching** — persist PageRank scores and cluster assignments
   across sessions in LanceDB.
6. **Query bundle detection** — log retrieval patterns, identify clusters,
   pre-materialize in a `bundles` LanceDB table.

## What This Means for the Current Code

- `searcher.rs` `augment_with_graph` should evolve to query-guided graph
  traversal (score candidates by vector similarity, not just BFS depth)
- `composite.rs` `traverse_importers` + `hnsw_neighbors` already compose
  for graph-guided HNSW search — the wiring in `searcher.rs` needs to use it
- The current architecture is a stepping stone, not the destination
