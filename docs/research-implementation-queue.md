# Research-Driven Implementation Queue

> Findings from Hippocampus paper analysis (arXiv 2602.13594), Anatomy of Agentic
> Memory survey (arXiv 2602.19320), and four deep-dive research tracks on evaluation
> methodology, multi-graph architectures (MAGMA), system cost patterns, and context
> saturation.
>
> Ordered by signal-to-effort ratio. Integrates with existing v2-roadmap.md phases.

Last updated: 2026-03-18

---

## Strategic Conclusions

1. **Hippocampus (DWM/binary signatures): Do not implement.** Different domain
   (conversational memory), binary signatures regress code retrieval quality,
   append-only DWM incompatible with code re-indexing. No borrowable components.

2. **Context saturation does not threaten code search.** 1M tokens ≈ 50K LOC.
   Production codebases exceed this. NoLiMa (ICML 2025): all 18 models drop below
   50% at 32K tokens on non-literal retrieval. Chroma Context Rot: topically similar
   distractors (the code case) hurt most. Retrieval is durable.

3. **"Agentic beats RAG" is a false dichotomy.** Every top SWE-bench system uses
   structured retrieval internally. Claude Code abandoned static vector-DB RAG, not
   retrieval itself. The emerging pattern (SWE-grep, WarpGrep) is specialized
   retrieval subagents — exactly skelesearch's architecture.

4. **MAGMA multi-graph pattern validates skelesearch's schema.** The `code_edges`
   relation already has `edge_type` but all queries wildcard it. Depth decay is the
   single most justified formula change.

5. **Cloud reranker is the dominant latency cost** (400-750ms P50 vs 5-50ms for
   local HNSW+BM25). Local ONNX cross-encoder would be 10-30ms.

---

## Queue

### R1: Depth-Decayed Graph Augmentation (2-4 hours)

**Source:** MAGMA fusion formula, SpIDER (+13% with graph expansion post-retrieval)
**Impact:** Fixes indefensible flat 0.5× scoring where depth-3 nodes equal direct importers
**Risk:** LOW — formula change only, no schema change

**What:**
- Change `traverse_imports` return type: `Vec<String>` → `Vec<(String, usize)>` to
  match `traverse_importers` (which already returns depth)
- Update BFS in `CozoBackend::traverse_imports` to track depth per node
- Update `augment_with_graph` in `searcher.rs`:
  ```
  // Old: graph_score = best_score * 0.5
  // New: graph_score = best_score * base_weight * γ^depth
  //   where base_weight = 0.6, γ = 0.7
  //   depth 1: 0.42×, depth 2: 0.29×, depth 3: 0.21×
  ```
- Update Arc<B> blanket impl for new signature
- Update all callers of `traverse_imports` (searcher + any tests)

**Files:** `crates/core/src/schema.rs` (trait + impl + blanket), `crates/core/src/searcher.rs`

**Eval gate:** Run zod + httpx benchmarks before/after. Depth-decayed scoring should
reduce false positives from deep transitive imports.

---

### R2: Edge-Type-Aware Graph Queries (3-5 hours)

**Source:** MAGMA edge-type partitioned multigraph, GraphCodeAgent dual-graph pattern
**Impact:** Prerequisite for Phase B1 (call graph). Currently adding call edges would
  pollute import-only PageRank and BFS traversal.
**Risk:** LOW — backwards-compatible with default `edge_types: &["imports"]`
**Depends on:** R1 (same functions being modified)

**What:**
- Add `edge_types` parameter to `StorageBackend` trait methods:
  - `traverse_imports(from, max_depth, edge_types: &[&str])` → default `&["imports"]`
  - `traverse_importers(from, max_depth, edge_types: &[&str])` → default `&["imports"]`
  - `compute_pagerank(edge_types: &[&str])` → default `&["imports"]`
- Update three CozoDB queries that currently wildcard `edge_type`:
  - `traverse_imports`: `*code_edges[f, _, t, _, _]` → filter by `edge_type in [...]`
  - `traverse_importers`: same pattern
  - `compute_pagerank`: same pattern
- Update Arc<B> blanket impl
- No caller changes needed — all current callers pass default

**Files:** `crates/core/src/schema.rs` (trait + impl + blanket)

---

### R3: Local ONNX Cross-Encoder Reranker (2-3 days)

**Source:** System cost analysis — cloud reranker adds 400-750ms P50 per query.
  Local ONNX INT8 runs in 5-30ms. Nixiesearch benchmark confirms.
**Impact:** 10-20× latency reduction on reranked queries. Enables reranking by default
  without API key requirement. Moves skelesearch toward zero-config excellence.
**Risk:** MEDIUM — new crate, ONNX model bundling, model selection

**What:**
- New crate: `crates/rerank-local/` implementing `Reranker` trait
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` ONNX (22MB, INT8)
  - Alternative: `BAAI/bge-reranker-v2-m3` for multilingual code
- Use `ort` (ONNX Runtime) crate — same runtime as fastembed, shared session
- Tokenize query+document pairs → run inference → return scores
- `reranker_from_name("local")` returns local reranker
- Update MCP auto-configure: prefer local reranker when no cloud API key set
- Feature-gate behind `rerank-local` feature flag

**Files:** `crates/rerank-local/` (new), `crates/core/src/reranker.rs` (trait already
  defined), `crates/mcp/src/server.rs` (auto-configure priority), `Cargo.toml` workspace

**Eval gate:** Compare local vs cloud reranker accuracy on full 240-case benchmark.
  Acceptable if MRR delta < 2pp (latency gain >> accuracy loss).

**ADR needed:** ADR-011: Local ONNX cross-encoder as default reranker

---

### R4: Latency Telemetry in MCP Response (3-5 hours)

**Source:** Survey finding: "system-level costs are frequently overlooked."
  Agent developers cannot optimize what they cannot measure.
**Impact:** Agent frameworks can make informed decisions about reranker, expansion, graph
**Risk:** LOW — additive metadata, no behavior change

**What:**
- Add timing struct to `Searcher::search` return:
  ```rust
  pub struct SearchTimings {
      pub embed_ms: u64,      // query embedding
      pub retrieve_ms: u64,   // HNSW + BM25 + RRF
      pub expand_ms: u64,     // LLM query expansion (0 if skipped)
      pub rerank_ms: u64,     // cross-encoder (0 if skipped)
      pub graph_ms: u64,      // graph augmentation (0 if disabled)
      pub total_ms: u64,
  }
  ```
- Instrument each phase with `std::time::Instant`
- Return timings alongside results in `search_code` MCP tool response as
  `_timings` field (underscore prefix = metadata convention)
- Log timings via `tracing::info!` span

**Files:** `crates/core/src/searcher.rs`, `crates/mcp/src/server.rs`

---

### R5: Async PageRank (Lazy Consolidation) (2-3 hours)

**Source:** Survey's write-consolidate lifecycle analysis. PageRank is computed
  synchronously at index end, blocking for large repos.
**Impact:** Unblocks index completion for 10K+ file repos. PageRank boost is
  non-critical — stale ranks are better than blocking.
**Risk:** LOW — PageRank is a score boost, not a filter. Stale ranks degrade
  gracefully (no boost applied to new files).

**What:**
- In `Indexer::index_path`, spawn `compute_pagerank` as a background task
  instead of awaiting it inline
- Add `pagerank_stale: bool` flag to `IndexStats` so callers know when
  ranks are being recomputed
- Log completion via `tracing::info!`

**Files:** `crates/core/src/indexer.rs`, `crates/core/src/schema.rs` (IndexStats)

---

### R6: Eval Annotation Audit with LLM Judge (1-2 days)

**Source:** Survey finding: lexical metrics diverge from semantic judgments due to
  annotation incompleteness. skelesearch's exact-path-match is immune to F1 failure
  modes but vulnerable to missing valid alternative files in `expected_files`.
**Impact:** Improves eval corpus quality. Detects cases where skelesearch returns
  correct files that are scored as misses.
**Risk:** LOW — offline audit, not a runtime change

**What:**
- Script: `benchmarks/scripts/audit-annotations.py`
- For each case where skelesearch returns a top-3 file NOT in `expected_files`:
  - Ask LLM: "Given this query, is {file} relevant to answering it? Here are the
    first 50 lines of the file."
  - If judge says yes → flag as annotation gap
- Run on the full 240-case corpus with current voyage-full results
- Update `expected_files` for validated gaps
- Report: how many cases were annotation errors vs retrieval failures

**Files:** `benchmarks/scripts/audit-annotations.py` (new), `benchmarks/cases/` (updates)

---

### R7: Per-Edge-Type PageRank (3-5 hours)

**Source:** MAGMA: separate graph signals measure different things. Import centrality ≠
  call centrality. Research finding: "must resolve before Phase B1 merges."
**Impact:** Prevents call graph edges from polluting import-based PageRank
**Risk:** LOW — additive signal, backwards-compatible
**Depends on:** R2 (edge-type-aware queries)

**What:**
- `compute_pagerank` already takes `edge_types` from R2
- Add `file_call_ranks` relation (or reuse `file_ranks` with a `rank_type` column)
- Searcher applies both boosts independently:
  ```
  import_boost = 1.0 + 0.3 * ln(1 + import_pr / median_import_pr)
  call_boost = 1.0 + 0.2 * ln(1 + call_pr / median_call_pr)
  combined_boost = import_boost * call_boost
  ```
- Only matters once Phase B1 lands — implement alongside or immediately before

**Files:** `crates/core/src/schema.rs`, `crates/core/src/searcher.rs`

---

### R8: Benchmark Corpus Hardening (1-2 days)

**Source:** Context saturation research: mini-redis hits 100% R@5 (cannot discriminate
  improvements). Survey: benchmark saturation is a validity threat.
**Impact:** Ensures future improvements are measurable
**Risk:** LOW — eval infrastructure only

**What:**
- Add 2-3 larger repos to benchmark corpus (target: 50K+ LOC each)
  - Candidates: FastAPI (Python, ~60K LOC), Axum (Rust, ~40K LOC),
    Next.js (TS, large), Gin (Go, ~30K LOC)
- Add 40 cases per new repo (same category distribution)
- Implement held-out test split (20% of cases) to detect eval-set overfitting
- Update saturation analysis: flag repos where R@5 = 100% as non-discriminative

**Files:** `benchmarks/manifests/repos.toml`, `benchmarks/cases/` (new files),
  `benchmarks/scripts/report.ts` (saturation flag)

---

### R9: Query Embedding Cache (2-3 hours)

**Source:** Nixiesearch benchmark: cloud embedding P99 up to 5s. Agent loops issue
  variant queries ("error handling in HTTP" → "HTTP error handling") that map to
  similar embeddings.
**Impact:** Eliminates cloud embedding latency on repeated/similar queries
**Risk:** LOW — LRU cache, transparent

**What:**
- Add LRU cache (capacity: 256) keyed on normalized query string in `Searcher`
- Before calling `provider.embed_batch`, check cache
- Cache hit → skip embedding API entirely
- Already partially addressed by embedding cache for chunks — this is for queries

**Files:** `crates/core/src/searcher.rs`

---

## Integration with Existing Roadmap

| Research Item | Roadmap Phase | Relationship |
|---|---|---|
| R1: Depth decay | Pre-B1 | Fixes existing deficiency, prerequisite for B1 quality |
| R2: Edge-type queries | Pre-B1 | Must land before call graph edges |
| R3: Local reranker | Phase A (parallel) | Independent, improves DX + latency |
| R4: Latency telemetry | Phase A (parallel) | Independent, observability |
| R5: Async PageRank | Phase A (parallel) | Independent, indexing performance |
| R6: Eval audit | Phase A4 extension | Improves eval corpus before B1 tuning |
| R7: Per-edge PageRank | Phase B1 (co-land) | Required when call edges land |
| R8: Corpus hardening | Phase A4 extension | Ensures improvements are measurable |
| R9: Query embed cache | Phase A (parallel) | Independent, latency |

**Recommended execution order:**
1. R1 + R2 (same files, do together) — graph scoring foundation
2. R4 + R5 + R9 (independent, parallelizable) — latency + observability
3. R3 (local reranker) — biggest DX improvement
4. R6 + R8 (eval hardening) — before tuning Phase B1
5. R7 (per-edge PageRank) — co-land with Phase B1

---

## Key Decisions from Research

| Decision | Choice | Source |
|---|---|---|
| Hippocampus DWM | Do not implement | Different domain, regresses code retrieval |
| Context saturation | Not a threat at production scale | NoLiMa, Chroma, LongCodeBench |
| Multi-graph abstraction | Adopt (near-zero cost) | MAGMA, schema already supports it |
| Depth decay formula | `base_weight × γ^depth` (γ=0.7) | MAGMA beam search, deferred in code |
| Default reranker | Local ONNX > cloud API | 10-20× latency reduction |
| Eval methodology | Add LLM judge for annotation audit | Survey metric validity findings |
| Benchmark saturation | Add larger repos, flag 100% R@5 repos | Survey saturation analysis |
| Semantic graph edges | Skip (O(n²), HNSW covers at query time) | MAGMA: lowest priority edge type |
| "Agentic beats RAG" | False dichotomy — retrieval is durable | LaRA, SWE-grep, ContextBench |

---

## New ADRs Needed

- **ADR-011:** Local ONNX cross-encoder as default reranker (R3)
- **ADR-012:** Depth-decayed graph scoring with edge-type weights (R1+R2)

---

## Papers Referenced

- Hippocampus (arXiv 2602.13594) — DWM for agentic memory. Not applicable.
- Anatomy of Agentic Memory (arXiv 2602.19320) — Survey, eval methodology.
- MAGMA (arXiv 2601.03236) — Multi-graph agentic memory. Graph scoring applicable.
- NoLiMa (ICML 2025, arXiv 2502.05167) — Long context degradation.
- LaRA (ICML 2025, arXiv 2502.09977) — RAG vs long context, no silver bullet.
- GrepRAG (arXiv 2601.23254) — Lexical retrieval competitive with graph-RAG for code.
- ContextBench (arXiv 2602.05892) — Code retrieval benchmark, F1=0.344 best.
- LongCodeBench (arXiv 2602.17183) — Recognition-generation gap at long context.
- GraphCodeBERT (arXiv 2009.08366) — Data flow graph MRR=0.700.
- GRACE (arXiv 2509.05980) — 5-graph code search, +8.19% EM.
- SpIDER (arXiv 2512.16956) — Graph expansion post-retrieval, +13%.
- FPGraphCS (doi:10.3390/app16010012) — Multi-graph fusion, early > late.
- Chroma Context Rot (2025) — Topically similar distractors hurt most.
- Anthropic Context Engineering (2025) — Just-in-time retrieval endorsed.
- SWE-grep / Cognition (Oct 2025) — RL-trained retrieval subagent.
