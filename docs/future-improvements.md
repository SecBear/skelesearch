# Future Improvements

Research and ideas deferred from v1. These are documented here so the context
behind each deferral is not lost.

---

## SPLADE Sparse Retrieval (post-v1)

**What:** Replace CozoDB FTS (BM25) with SPLADE learned sparse embeddings for the sparse
retrieval leg of hybrid search.

**Why deferred:** No code-domain SPLADE benchmarks exist. Both available fastembed-rs SPLADE
models are trained on English NLP (not code). CozoDB has no native inverted index for SPLADE
vectors, so it would require a hand-rolled posting list. SPLADE is 24× slower than BM25 at
query time on CPU.

**Research foundation:**
- **Seismic** (SIGIR 2024 Best Paper Runner-up): state-of-the-art Rust implementation of
  SPLADE retrieval using geometrically cohesive block clustering for sub-millisecond latency.
  Crate: `seismic` on crates.io. Paper: https://arxiv.org/html/2404.18812v1
  GitHub: https://github.com/TusKANNy/seismic

- **fastembed-rs API** (already researched):
  ```rust
  let model = SparseTextEmbedding::try_new(
      SparseInitOptions::new(SparseModel::BGEM3)
  )?;
  let embeddings: Vec<SparseEmbedding> = model.embed(texts, None)?;
  // SparseEmbedding { indices: Vec<usize>, values: Vec<f32> }
  ```
  Only two models: `SPLADEPPV1` and `BGEM3`. BGE-M3 requires separate model instances
  for dense and sparse (no unified single call in fastembed-rs as of 2026-03).

- **DBSF** as an alternative to RRF for fusion: normalizes scores using 3-sigma rule.
  Performs better than RRF when dense and sparse scores have very different distributions.
  Qdrant added DBSF in v1.11. Reference: https://medium.com/plain-simple-software/distribution-based-score-fusion-dbsf-a-new-approach-to-vector-search-ranking-f87c37488b18

**When to revisit:** If a code-specialized SPLADE model appears, or if users report that
BM25 misses semantically related functions due to vocabulary mismatch (e.g., searching
"authentication" doesn't match functions named `auth_*`).

---

## AST-level function diffing for incremental indexing

**What:** Instead of re-indexing changed files at file granularity, diff old vs new parse tree
to identify which function/impl blocks actually changed, and re-embed only those chunks.

**Why deferred:** Significant implementation complexity (store old parse tree or chunk
boundaries per file, use tree-sitter incremental re-parse, diff old vs new named node list).
Benefit only materializes in watch mode on large files with many functions.

**Reference:** tree-sitter supports incremental parsing via `ts_parser_parse(old_tree, edit)`.
rust-analyzer's salsa-based incremental analysis is the gold standard for this pattern.

---

## tree-sitter-stack-graphs for precise import resolution

**What:** Use GitHub's tree-sitter-stack-graphs to build precise cross-file import graphs
(resolving `use foo::Bar` to the actual file/module that defines `Bar`).

**Why deferred:** Requires writing per-language `.tsg` rule files (non-trivial for Rust).
Rust `.tsg` rules don't exist yet upstream. TypeScript and Python rules are open source.

**Reference:**
- https://github.com/github/stack-graphs
- Crate: `tree-sitter-stack-graphs` on crates.io
- Related: `tree-sitter-graph` DSL for constructing arbitrary graphs from ASTs

**When to revisit:** Once v1 import extraction (simple query-based) proves insufficient for
"find all files that transitively import this module" queries.

---

## type-sitter typed AST wrappers

**What:** Use `type-sitter` to generate strongly-typed Rust wrappers from `node-types.json`
for each grammar. Replaces string-based `node.kind() == "function_item"` comparisons with
typed `FunctionItem::try_from(node)` patterns.

**Why deferred:** Additional build step, unfamiliar API. String-based kind matching is
sufficient and understandable for v1.

**Reference:** https://github.com/Jakobeha/type-sitter

---

## CozoDB → LanceDB + Tantivy migration path

**What:** If CozoDB's apparent development stagnation (last release v0.7.6, December 2023)
becomes a real problem (security issues, incompatibility with new Rust editions, etc.), the
migration path is:

- Replace `crates/core/src/schema.rs` with a LanceDB implementation of `StorageBackend`
- Use LanceDB for HNSW vector storage + hybrid search
- Use Tantivy for BM25 FTS
- Port the Datalog graph traversal queries to a hand-rolled BFS/DFS over a relational edge table
- RRF fusion stays the same (already implemented independently)

The `code-sage` project (https://github.com/faxioman/code-sage) uses exactly this stack
(USearch + Tantivy + Sled) and is a useful reference implementation.

**Blast radius:** `schema.rs` only. The `StorageBackend` trait boundary is the migration seam.

---

## BGE-M3 unified dense+sparse

**What:** Once fastembed-rs adds a unified multi-output API for BGE-M3 (single model call
returning both dense and sparse vectors), add it as an embedding option. BGE-M3 with both
outputs would give better hybrid search quality with only one ONNX inference pass.

**Reference:** fastembed-rs issue tracker (feature request for unified multi-output, Sept 2024).

---

## Diff-aware retrieval mode (v2)

**What:** A query mode that restricts the search corpus to files modified in the current
git branch (`git diff main --name-only`). When an agent is working on a branch, "what
changed" is a stronger locality signal than full-corpus semantic similarity.

**Why deferred:** Branch-scoped retrieval requires integrating with `git` at query time
and maintaining a per-branch filtered view of the index. v1 full-corpus search is the
right default.

**Evidence:** gstack's `/qa` mode uses `git diff main` as its primary context scoping
mechanism and is the most-used retrieval pattern in that tool. Retrieval adds the most
value when the relevant code is scattered with no obvious locality — diff-scoped queries
reduce the search space to the region where an agent is already working.

**Approach:** Add `branch_scope: bool` parameter to `search_code`. When true, run
`git diff --name-only HEAD...$(git merge-base HEAD main)` to get the changed file set,
then add a `file_path IN (...)` filter to the HNSW and FTS queries before RRF fusion.

---

## Closed-loop retrieval self-verification (v2)

**What:** After `search_code` returns results, automatically re-query with broadened
terms if the top result's `match_quality` is "low" (no high-confidence hits). Return
results from both queries merged with provenance.

**Why deferred:** Requires defining quality thresholds and fallback query strategies.
v1 returns all results with `match_quality` labels and lets the agent decide.

**Evidence:** Cherny and Steinberger (Jan 2026) both emphasize that tools must give
agents a way to self-verify. Sycophantic agents will use a "low" match result without
questioning it unless the tool provides an explicit signal and a next step.

**Approach:** If top result score < threshold: retry with query decomposed into keywords,
merge results, add `source: "broadened_query"` annotation to distinguish. Return combined
ranked results with per-result provenance.

---

## Retrieval feedback loop (v2)

**What:** A lightweight mechanism for recording "this result was irrelevant" per-query,
stored in `~/.local/share/skelesearch/<hash>/feedback.db`. Use accumulated feedback to
adjust RRF weights (vector vs FTS leg) per-repo over time.

**Evidence:** gstack persists Greptile false positives to `~/.gstack/greptile-history.md`
to filter future runs — a cheap feedback-loop without retraining. Same principle applies
to retrieval weight tuning.

---

## Call graph edges (v2)

**What:** Extract function call edges in addition to import edges. Store as `edge_type: "calls"`
in `code_edges`. Enable "find all callers of this function" queries.

**Why deferred:** Import edges are extractable with simple tree-sitter queries. Call edges
require tracking which identifier at a call site refers to which function definition —
a name resolution problem that requires either stack graphs or a language server.

**Approach for Rust:** Use tree-sitter to find `call_expression` nodes, extract the function
name, then match against indexed function chunk names. Approximate (no type-based disambiguation)
but useful for most cases.
