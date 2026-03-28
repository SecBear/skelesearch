# CompositeBackend Architecture

**Introduced:** ADR-011 (2026-03-28)
**Replaces:** CozoDB (ADR-002). See `docs/archive/cozodb-limitations.md` for the full list of limitations that motivated this replacement.

`CompositeBackend` orchestrates three specialized engines behind the `StorageBackend` trait:

| Engine | Purpose |
|---|---|
| LanceDB 0.27 | Vector storage + relational tables (Apache Arrow) |
| Tantivy 0.25 | BM25 full-text search with code-aware tokenizer |
| petgraph 0.8 | In-memory import graph (BFS, PageRank) |

---

## On-Disk Layout

```
<index_dir>/
  lance/     — LanceDB dataset directory (one sub-directory per table)
  tantivy/   — Tantivy index directory (meta.json + segment files)
```

`CompositeBackend::open(dir)` expects a **directory**, not a `.db` file.
Existing CozoDB `index.db` files are not migrated. Callers must re-index with `skelesearch index`.

---

## LanceDB Tables (Arrow Schemas)

All schemas are defined in `crates/core/src/backend/schemas.rs`.
Embedding dimensions (`dim`) are runtime-configurable — never hardcoded.

### `files`

| Column | Type | Nullable | Notes |
|---|---|---|---|
| `file_path` | Utf8 | no | Primary key |
| `language` | Utf8 | no | tree-sitter language name |
| `last_modified` | Int64 | no | Unix timestamp |
| `last_indexed` | Int64 | no | Unix timestamp |
| `chunk_count` | UInt64 | no | |

### `chunks`

| Column | Type | Nullable | Notes |
|---|---|---|---|
| `file_path` | Utf8 | no | |
| `chunk_idx` | UInt32 | no | |
| `content` | Utf8 | no | Raw source text |
| `normalized` | Utf8 | no | Pre-processed for FTS |
| `description` | Utf8 | no | LLM-generated summary (empty string if none) |
| `chunk_type` | Utf8 | no | `function`, `class`, `block`, etc. |
| `start_line` | UInt32 | no | |
| `end_line` | UInt32 | no | |
| `embedding` | FixedSizeList\<Float32\>[dim] | yes | None until embedded |
| `doc_embedding` | FixedSizeList\<Float32\>[dim] | yes | Dual-embedding slot |
| `materialization_tier` | UInt8 | no | 0 = raw, 1 = tier1, 2 = tier2 |

### `code_edges`

| Column | Type | Nullable | Notes |
|---|---|---|---|
| `from_file` | Utf8 | no | Importing file |
| `from_chunk` | UInt32 | no | Chunk containing the import statement |
| `to_file` | Utf8 | no | Imported file |
| `edge_type` | Utf8 | no | e.g. `import`, `reexport` |

### `call_edges`

| Column | Type | Nullable | Notes |
|---|---|---|---|
| `caller_file` | Utf8 | no | |
| `caller_symbol` | Utf8 | no | |
| `callee_name` | Utf8 | no | |
| `start_line` | UInt32 | no | Call site line |
| `callee_file` | Utf8 | yes | Resolved, or null if unresolved |
| `callee_symbol` | Utf8 | yes | Resolved, or null if unresolved |
| `confidence` | Float64 | no | |
| `dynamic` | Boolean | no | True for dynamic dispatch |

### `symbols`

| Column | Type | Nullable | Notes |
|---|---|---|---|
| `file_path` | Utf8 | no | |
| `name` | Utf8 | no | |
| `kind` | Utf8 | no | `function`, `class`, `constant`, etc. |
| `start_line` | UInt32 | no | |
| `end_line` | UInt32 | no | |

### `cochange_edges`

| Column | Type | Nullable | Notes |
|---|---|---|---|
| `file_a` | Utf8 | no | |
| `file_b` | Utf8 | no | |
| `cochange_count` | UInt64 | no | Git co-commit count |
| `jaccard` | Float64 | no | Jaccard similarity score |

### `sparse_index`

SPLADE sparse embedding index. Stored in LanceDB but only loaded into memory lazily.

| Column | Type | Nullable |
|---|---|---|
| `file_path` | Utf8 | no |
| `chunk_idx` | UInt32 | no |
| `token_id` | UInt32 | no |
| `weight` | Float32 | no |

### `pagerank_scores`

| Column | Type | Nullable |
|---|---|---|
| `file_path` | Utf8 | no |
| `score` | Float64 | no |

### `symbol_roles`

| Column | Type | Nullable |
|---|---|---|
| `file_path` | Utf8 | no |
| `role` | Utf8 | no |

### `doc_chunks` (optional dual-embedding table)

Separate table to avoid wide rows when dual_embedding is disabled.

| Column | Type | Nullable |
|---|---|---|
| `file_path` | Utf8 | no |
| `chunk_idx` | UInt32 | no |
| `embedding` | FixedSizeList\<Float32\>[dim] | no |

---

## Tantivy Index

Defined in `crates/core/src/backend/tantivy_idx.rs`. Stored at `<index_dir>/tantivy/`.

### Fields

| Field | Tantivy type | Options | Notes |
|---|---|---|---|
| `file_path` | text | STORED + FAST | Raw (no tokenization). FAST for term deletion. |
| `chunk_idx` | u64 | STORED + FAST | FAST for efficient deletion filter. |
| `normalized` | text | indexed | Tokenized with `CodeAnalyzer` for BM25. |
| `description` | text | indexed | Tokenized with `en_stem` (English stemmer) for LLM summaries. |
| `chunk_type` | text | STORED | Not indexed; used for Rust-side filtering only. |
| `tier` | u64 | STORED + FAST | Materialization tier; FAST for tier1 deletion. |

### CodeAnalyzer Tokenizer

The `code` tokenizer splits identifiers on CamelCase, snake_case, kebab-case, and numeric boundaries using the regex `[A-Z]?[a-z]+|[A-Z]+|[0-9]+`, followed by `LowerCaser`.

Example: `getUserById` → `["get", "user", "by", "id"]`

This was the primary motivation for replacing CozoDB's FTS (which lacked custom tokenizer support — see `docs/archive/cozodb-limitations.md` §FTS Limitations).

The `en_stem` tokenizer (`SimpleTokenizer` + `LowerCaser` + `Stemmer(English)`) is used for the `description` field.

---

## petgraph Import Graph

Defined in `crates/core/src/backend/graph.rs`.

The import graph is held entirely **in memory** as a `petgraph::DiGraph<String, String>`:
- Nodes: file paths (`String`)
- Edges: directed, labeled with `edge_type` (e.g. `"import"`, `"reexport"`)
- `node_index: HashMap<String, NodeIndex>` provides O(1) file_path → node lookup

### Lifecycle

- Loaded from the `code_edges` LanceDB table at `CompositeBackend::open` time.
- Updated in-place by `upsert_edges` and `delete_edges_for_file`.
- Never written back to disk — the LanceDB table is the persistent source of truth.

### Operations

| Method | Description |
|---|---|
| `bfs_forward(start, max_depth, edge_types)` | Files reachable from `start` (imports) |
| `bfs_reverse(start, max_depth, edge_types)` | Files that (transitively) import `start` |
| `add_edge(from, to, edge_type)` | Deduplicates — no parallel edges for same (from, to, type) |
| `remove_edges_for_file(path)` | Removes all incident edges (both directions) |

BFS and PageRank run entirely in-memory — no I/O per hop.

---

## Re-Index Requirement

`CompositeBackend` uses a different on-disk format from CozoDB. If an existing `index.db` is detected at the index directory root and no `lance/` sub-directory exists, `open()` returns an error:

```
Found legacy CozoDB index at <path>. CompositeBackend uses a different on-disk format.
Re-index with `skelesearch index`.
```

---

## Build Dependency: `protoc`

`lance-encoding` (a transitive dependency of `lancedb`) requires the Protocol Buffer compiler at build time.

```sh
# macOS
brew install protobuf

# Ubuntu/Debian
apt install protobuf-compiler

# Nix (flake.nix already includes it)
```
