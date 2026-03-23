# CozoDB Limitations & Workarounds

**Purpose:** Document every CozoDB limitation encountered during skelesearch development.
This is the spec for what a purpose-built replacement database would need to solve.
When we eventually build or adopt a replacement, every entry here is a requirement.

See also: `docs/cozodb-patterns.md` (usage patterns), `DECISIONS.md` ADR-002 (why CozoDB).

---

## FTS Limitations

### No custom tokenizer support
**Problem:** FTS only supports `Simple` and `Raw` tokenizers with `Lowercase`/`AlphaNumOnly`
filters. No n-gram, trigram, or subword tokenization. Cannot match partial identifiers
like `erById` against `getUserById` via FTS alone.

**Workaround:** Pre-compute bigram concatenations in the `normalized` field at index time.
`getUserById` → `get user by id getuser userby byid`. Storage overhead ~2x on normalized field.

**Ideal:** Custom tokenizer pipeline: `CamelCaseSplit → Lowercase → NGram(3) → AlphaNumOnly`.
Or a dedicated code-aware tokenizer that understands identifier conventions.

### No FTS field-level boosting
**Problem:** Cannot boost matches in file path vs code content within a single FTS query.
A match on `src/auth/login.rs` in the path prefix should score higher than matching `login`
in a code comment, but CozoDB FTS treats the entire `normalized` field uniformly.

**Workaround:** File path is prepended to `normalized`, so it participates in BM25. But
there's no way to weight it differently from code content.

**Ideal:** Per-field BM25 with configurable boost weights, like Elasticsearch/Tantivy.

### No FTS query escaping
**Problem:** CozoDB's FTS query mini-language interprets dots, hyphens, and special characters
as operators. `std::io::Error` breaks the parser. We must sanitize all queries before FTS.

**Workaround:** `sanitize_fts_query()` strips all non-alphanumeric non-whitespace characters.
This loses precision — `io.Reader` and `io.Writer` become `io Reader` and `io Writer`,
which are ambiguous.

**Ideal:** Proper query escaping or a raw-text match mode that treats input as literal.

---

## HNSW Limitations

### No filtered HNSW search
**Problem:** Cannot combine vector similarity with metadata predicates in a single HNSW query.
Want: "find chunks similar to X where language='rust' and file_path starts with 'src/'".
Instead: run HNSW, then filter in Rust. If the answer is at position 51 and ef_search=50,
it's lost.

**Workaround:** Over-fetch (2-5x top_k) from HNSW, then filter and truncate in Rust.

**Ideal:** Predicate pushdown into HNSW traversal (Qdrant, Weaviate style).

### No multi-vector HNSW
**Problem:** Can only index one vector per chunk. Cannot store both a code embedding and a
docstring embedding for the same chunk and query against either.

**Workaround:** Embed the combined code+context once. Or maintain two separate HNSW indexes
(doubles storage, complex query orchestration).

**Ideal:** Named vector fields per record with independent HNSW indexes, queryable separately
or in ensemble.

### HNSW graph neighbor query is fragile
**Problem:** `hnsw_neighbors` query uses `seed <- $seeds, seed = [fp, ci]` destructuring
which silently produces empty results if column order is wrong. No error, just zero results.
Errors are swallowed.

**Workaround:** Integration tests. But silent failures mean bugs hide for weeks.

**Ideal:** Typed HNSW neighbor API that validates input structure at compile time.

---

## Schema / DDL Limitations

### `:rm` requires exact column names
**Problem:** CozoDB's `:rm` directive for deletions requires output column names to exactly
match the relation's key column names. `?[fp, ci] <- $keys :rm chunks { file_path: fp }`
silently fails or errors. Must be `?[file_path, chunk_idx] <- $keys :rm chunks`.

**Workaround:** Documented in `docs/cozodb-patterns.md` anti-patterns table.

**Ideal:** Either error on mismatch or support column aliasing in `:rm`.

### No schema migrations
**Problem:** No ALTER TABLE equivalent. Adding a column requires creating a new relation,
copying data, dropping the old one, and renaming. In production with user data, this is
a migration nightmare.

**Workaround:** Design schema upfront. Accept that changes require full re-index.

**Ideal:** `ALTER RELATION ADD COLUMN` with default values.

### No concurrent write access
**Problem:** SQLite backend uses exclusive locking. CLI and MCP server cannot both index
simultaneously. Even read queries can conflict during compaction.

**Workaround:** File lock detection + retry. Or WAL mode (not exposed by CozoDB API).

**Ideal:** WAL mode with concurrent readers + single writer, or true MVCC.

---

## Query Language Limitations

### No built-in PageRank/graph algorithms
**Problem:** `::algo PageRank` and other fixed rules are documented but broken in v0.7.6.
Cannot compute PageRank over the import graph within CozoDB.

**Workaround:** Compute PageRank in Rust, store results in a separate relation.

**Ideal:** Working `::algo` fixed rules, or extensible UDF system for custom graph algorithms.

### No EXPLAIN or query plan visibility
**Problem:** Cannot see how CozoDB plans or optimizes a query. Performance debugging is
trial-and-error.

**Workaround:** Benchmark queries externally. Profile with tracing spans.

**Ideal:** `EXPLAIN` output showing scan strategy, index usage, join order.

### Recursive Datalog performance is unpredictable
**Problem:** Recursive queries (BFS/DFS over edges) sometimes degrade to O(N²) without
clear reason. No ability to hint at traversal strategy.

**Workaround:** Cap recursion depth. Fall back to Rust-side BFS for critical paths.

**Ideal:** Stratification hints, materialized recursive views, or explicit BFS/DFS operators.

---

## LSH Limitations

### LSH dedup counts are inflated
**Problem:** LSH MinHash creates multiple hash bands per chunk. The same chunk appears in
multiple hash buckets. Deletion counts report the same chunk removal multiple times.

**Workaround:** Track unique (file_path, chunk_idx) pairs in a HashSet during dedup.

**Ideal:** LSH API that returns unique candidate pairs, not per-band duplicates.

---

## What a Replacement Needs

A purpose-built code search database should provide:

1. **Hybrid search primitive**: Vector (HNSW/IVF) + BM25 + filtered, in one query
2. **Code-aware FTS**: CamelCase/snake_case tokenization, n-gram, field boosting
3. **Predicate pushdown on vector search**: metadata filters during HNSW traversal
4. **Graph storage + traversal**: first-class edges with recursive queries, working PageRank
5. **Concurrent access**: WAL or MVCC, multiple readers + single writer minimum
6. **Schema evolution**: column add/remove without full re-index
7. **Observable queries**: EXPLAIN, query cost estimation, index usage stats
8. **Embeddable**: single-file database, no external process, Rust-native

Candidates to evaluate: LanceDB (clean boundary exists via `StorageBackend` trait),
DuckDB (SQL + vector), custom Tantivy+HNSW composite, or a bespoke engine.
