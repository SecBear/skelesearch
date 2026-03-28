> **ARCHIVED (2026-03-28):** CozoDB replaced by CompositeBackend (ADR-011). See `docs/composite-backend.md`.

# CozoDB Patterns for skelesearch

Source of truth for how skelesearch uses CozoDB. Read this before modifying
`crates/core/src/schema.rs`, `crates/core/src/searcher.rs`, or `crates/core/src/indexer.rs`.

## Version and Status

CozoDB v0.7.6 (Dec 2023, stalled). All code isolated behind `StorageBackend` trait.
See ADR-002 in `DECISIONS.md` for rationale and migration path.

**Broken:** `::algo` fixed rules (PageRank, shortest_path, etc.) — do NOT use.
**Working:** Recursive Datalog, HNSW, FTS, LSH, triggers, time travel — all functional.

## Relations

```
files       { file_path: String => language: String, last_modified: Int, last_indexed: Int, chunk_count: Int }
chunks      { file_path: String, chunk_idx: Int => content: String, normalized: String, chunk_type: String, start_line: Int, end_line: Int, embedding: <F32; dim> }
code_edges  { from_file: String, from_chunk: Int, to_file: String => edge_type: String, created_at: Int }
symbols     { file_path: String, name: String, start_line: Int => kind: String, end_line: Int }
file_ranks  { file_path: String => pagerank: Float }
```

## Indices on `chunks`

| Name | Type | Purpose |
|---|---|---|
| `chunks:semantic` | HNSW | Vector search. m=32, ef_construction=128, Cosine distance. |
| `chunks:text` | FTS | Full-text search. Simple tokenizer, Lowercase + AlphaNumOnly, TF-IDF scoring. |
| `chunks:dedup` | LSH | Near-duplicate detection. n_gram=5, n_perm=128, threshold=0.85. |

## Datalog Query Patterns

### Parameterized queries

```rust
let mut p = BTreeMap::new();
p.insert("key".into(), Self::dv_str(value));
self.run_imm("?[x] := *relation[$key, x]", p)?;  // read
self.run_mut("?[x, y] <- $rows :put relation { x => y }", p)?;  // write
```

Use `run_imm` for reads, `run_mut` for writes. Both are synchronous.

### Proximity search

```datalog
-- HNSW vector search
~chunks:semantic{ file_path, chunk_idx | query: $qv, k: 50, ef: 64, bind_distance: dist }

-- FTS search (MUST specify score_kind: 'tf_idf' — default is raw TF)
~chunks:text{ file_path, chunk_idx | query: $qs, k: 50, score_kind: 'tf_idf', bind_score: bm25 }

-- LSH near-duplicate search
~chunks:dedup{ file_path, chunk_idx | query: $content, k: 1 }
```

**Filter predicates** can restrict results during search (not post-hoc):
```datalog
~chunks:semantic{ fp, ci | query: $qv, k: 50, ef: 64, bind_distance: dist, filter: chunk_type = 'function' }
```

### HNSW graph walking (zero-cost seed expansion)

The HNSW index is queryable as a stored relation. **Column naming convention:**
CozoDB names adjacency columns as `fr_{key_column_name}` / `to_{key_column_name}`.

For `chunks { file_path: String, chunk_idx: Int => ... }`:
```datalog
*chunks:semantic{ layer: 0, fr_file_path, fr_chunk_idx, to_file_path, to_chunk_idx, dist, ignore_link }
```

Always bind `ignore_link: false` explicitly — leaving it unbound makes the filter a no-op.

### Recursive Datalog (replaces Rust BFS loops)

CozoDB supports recursive rules natively. This is a core language feature, NOT affected
by the broken `::algo` crate. Use it for graph traversal instead of Rust-side BFS:

```datalog
-- Forward traversal: files reachable from $start via code_edges
-- NOTE: CozoDB does not allow literals in rule heads. Use d = 1 in the body.
reach[to_file, d] := *code_edges[$start, _, to_file, _, _], to_file != $start, d = 1
reach[to_file, d] := reach[mid, prev], d = prev + 1, d <= $max_depth,
    *code_edges[mid, _, to_file, _, _], to_file != $start
?[to_file, min(depth)] := reach[to_file, depth]
```

This replaces O(max_depth) queries with 1 query. CozoDB handles cycles via stratification.

### Batch operations with `is_in`

```datalog
-- Batch lookup by key list
?[fp, ci, content] := *chunks[fp, ci, content, _, _, _, _, _], is_in(fp, $file_paths)

-- Batch lookup by composite key pairs
?[fp, ci, emb] := key <- $keys, key = [fp, ci], *chunks[fp, ci, _, _, _, _, _, emb]
```

Prefer `is_in` over Rust-side loops. One query for N items, not N queries.

### Multi-rule queries (combine what would be separate queries)

```datalog
-- Two counts in one RTT
total[count(fp)] := *chunks[fp, _, _, _, _, _, _, _]
with_emb[count(fp)] := *chunks[fp, _, _, _, _, _, _, emb], !is_null(emb)
?[t, e] := total[t], with_emb[e]
```

### Deletion

Output column names **must** match the relation's key column names:

```datalog
-- CORRECT: column names match relation keys (file_path, chunk_idx)
?[file_path, chunk_idx] <- $keys :rm chunks

-- WRONG: aliased names cause parse error at runtime
-- ?[fp, ci] <- $keys :rm chunks { file_path: fp, chunk_idx: ci }
```

## Anti-Patterns (DO NOT)

| Anti-pattern | Correct approach |
|---|---|
| Rust BFS loop with one query per depth | Single recursive Datalog query |
| Sequential FTS + HNSW queries | `std::thread::scope` for parallel execution |
| `get_chunks_for_file` in a loop | `get_chunks_for_files` with `is_in` batch query |
| `is_in(fp, $fps)` then filter by chunk_idx in Rust | Batch fetch with `is_in` and filter Rust-side, or use specific key queries |
| Literal values in rule heads (`reach[x, 1] := ...`) | Bind in body instead: `reach[x, d] := ..., d = 1` |
| Omitting `score_kind: 'tf_idf'` on FTS queries | Always specify — default is raw TF, not TF-IDF |
| Using `!ignore_link` as negation-as-failure | Bind `ignore_link: false` in the relation pattern |
| Using `fr_k` / `fr__field` for HNSW graph columns | Use `fr_{column_name}` (e.g., `fr_file_path`, `fr_chunk_idx`) |
| Using aliased columns in `:rm` (`?[fp, ci] <- $keys :rm rel { col: fp }`) | Output names must match relation key names: `?[col] <- $keys :rm rel` |
| Multiple count/stat queries in sequence | Combine into multi-rule single query |
| `::algo` fixed rules (PageRank, etc.) | Broken in v0.7.6. Compute in Rust, store back. |

## Performance Guidelines

1. **Minimize round-trips.** Each `run_imm`/`run_mut` call is a full Datalog parse-plan-execute cycle. Combine queries when the data is needed together.
2. **Push filtering into CozoDB.** Use `filter:` on proximity search, `is_in` for set membership, relation pattern matching for exact keys. Don't fetch-then-filter in Rust.
3. **Parallelize independent reads.** `DbInstance` is `Send + Sync`. Use `std::thread::scope` for synchronous calls, `tokio::join!` for async calls.
4. **Batch writes.** Use `:put` with `$rows` parameter for bulk upserts. Batch size 500 is the current default.
5. **Background heavy computation.** PageRank and LSH dedup run as `tokio::spawn` background tasks — they're quality boosts, not blocking operations.

## StorageBackend Trait

All CozoDB access goes through this trait. When adding new functionality:

1. Add the method to `StorageBackend` trait (line ~81-155 in schema.rs).
2. Implement on `CozoBackend`.
3. Add `Arc<B>` delegation in the blanket impl (end of schema.rs).
4. Use Datalog-native patterns (recursive rules, is_in, multi-rule) over Rust loops.

## Index Tuning

| Parameter | Current | Notes |
|---|---|---|
| HNSW m | 32 | 2x default. Good for <500K chunks. Could reduce to 16 for faster builds. |
| HNSW ef_construction | 128 | Build-time quality. Higher = better recall, slower index. |
| HNSW ef (query) | 64 | Query-time quality. Higher = better recall, slower search. |
| FTS tokenizer | Simple | Splits on whitespace/punctuation. AlphaNumOnly strips underscores. |
| FTS score_kind | tf_idf | MUST be specified explicitly — CozoDB defaults to raw TF. |
| LSH n_gram | 5 | Shingle size for MinHash. |
| LSH threshold | 0.85 | Jaccard similarity threshold for near-duplicate detection. |
