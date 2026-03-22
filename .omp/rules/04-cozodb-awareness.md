---
description: "Enforce CozoDB architectural awareness when modifying storage, search, or indexing code"
alwaysApply: true
---
# CozoDB Architectural Awareness

When modifying any of these files, you MUST read `docs/cozodb-patterns.md` first:
- `crates/core/src/schema.rs` — all CozoDB queries and the StorageBackend trait
- `crates/core/src/searcher.rs` — search pipeline, graph augmentation, MMR
- `crates/core/src/indexer.rs` — data flow into CozoDB

## Mandatory checks before writing CozoDB queries

1. **No Rust-side loops issuing queries.** Use `is_in($list)`, `key <- $keys` destructuring, or recursive Datalog instead of per-item query loops. Every `run_imm`/`run_mut` in a loop is a potential N+1.
2. **No sequential independent queries.** If two reads are independent, use `std::thread::scope` (for sync calls) or `tokio::join!` (for async calls).
3. **Combine related counts/stats.** Multiple aggregation queries on the same relations should be multi-rule single queries.
4. **FTS queries MUST specify `score_kind: 'tf_idf'`.** CozoDB defaults to raw TF.
5. **HNSW graph columns use `fr_{column_name}` convention** (e.g., `fr_file_path`, `fr_chunk_idx`), NOT `fr_k`/`fr__field`.
6. **Bind `ignore_link: false` explicitly** when querying HNSW graph relations. Unbound `!ignore_link` is NAF, which is always true.
7. **New StorageBackend methods** need: trait declaration + CozoBackend impl + Arc<B> delegation.
