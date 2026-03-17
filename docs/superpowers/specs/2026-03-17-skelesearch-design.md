# skelesearch — Design Spec
*Date: 2026-03-17*

## Overview

skelesearch is a standalone semantic code-search tool and MCP server for AI coding agents.
It indexes a codebase with tree-sitter AST-aware chunking, stores chunks in an embedded
CozoDB database (HNSW vector index + FTS + import graph), and exposes hybrid BM25+dense
retrieval via both an MCP server (for Claude Code and other agents) and a CLI.

**Differentiators over existing tools (claude-context, grepai, Code-Index-MCP):**
- Zero cloud dependencies — fastembed-rs in-process embeddings with no external services
- Graph layer — import/call edges stored as CozoDB Datalog relations, traversable at query time
- Hybrid retrieval — BM25 (CozoDB FTS) + dense HNSW fused with RRF in a single Datalog query
- Fully embeddable — no Docker, no daemon, one RocksDB file per indexed project
- Provider-agnostic — swap in OpenAI, Voyage, Ollama, or fastembed via a trait

---

## Architecture

### Repo layout

```
skelesearch/
  Cargo.toml                  workspace
  flake.nix                   Nix packaging (CLI + MCP server binaries)
  CLAUDE.md                   AI agent context file
  DECISIONS.md                Architectural decision log
  docs/
    future-improvements.md    Deferred research and v2+ ideas
    superpowers/specs/        Design documents
  crates/
    core/                     skelesearch-core  (library)
    embed-fastembed/          skelesearch-embed-fastembed  (library, optional)
    mcp/                      skelesearch-mcp  (binary)
    cli/                      skelesearch-cli  (binary)
```

### Crate responsibilities

| Crate | Type | Purpose |
|---|---|---|
| `skelesearch-core` | lib | Schema, Indexer, Searcher, LanguageConfig trait, EmbedProvider trait, manifest |
| `skelesearch-embed-fastembed` | lib | FastEmbedProvider impl (jina-v2-base-code default) |
| `skelesearch-mcp` | binary | rmcp 0.16 MCP server, 4 tools |
| `skelesearch-cli` | binary | clap CLI, mirrors MCP tools |

### External dependencies on skelegent/extras

- `skg-mcp` — MCP server infrastructure (optional, may use rmcp directly)
- `skg-provider-ollama` — Ollama embedding provider (optional feature)
- `skg-provider-anthropic` — Anthropic/OpenAI-compatible embedding (optional feature)

---

## Data Model

### CozoDB schema

```datalog
# Files — one row per indexed file, for incremental change detection
:create files {
    file_path: String
    =>
    language: String,
    last_modified: Int,
    last_indexed: Int,
    chunk_count: Int,
}

# Chunks — primary storage for code segments
:create chunks {
    file_path: String, chunk_idx: Int
    =>
    content: String,          -- raw source text
    normalized: String,       -- camelCase/snake_case normalized for FTS
    chunk_type: String,       -- "function", "impl", "class", "module", "other"
    start_line: Int,
    end_line: Int,
    embedding: [Float]?,      -- null until embedded; dim set at index creation time
}

# Import/call graph edges
:create code_edges {
    from_file: String, from_chunk: Int,
    to_file: String,
    =>
    edge_type: String,        -- "imports" (v1), "calls" (v2)
    created_at: Int,
}
```

### CozoDB indices

```datalog
# Dense vector index (dim is configurable, set at index creation from EmbedProvider::dim())
::hnsw create chunks:semantic {
    dim: $DIM,
    dtype: F32,
    fields: [embedding],
    distance: Cosine,
    m: 50,
    ef_construction: 20,
    filter: !is_null(embedding),
}

# Full-text search index (BM25)
::fts create chunks:text {
    extractor: normalized,
    tokenizer: Simple,
    filters: [Lowercase, AlphaNumOnly],
}
```

### Hash manifest

A separate SQLite file (`~/.local/share/skelesearch/<project-hash>/manifest.db`) stores:
```sql
CREATE TABLE file_hashes (
    file_path TEXT PRIMARY KEY,
    mtime     INTEGER NOT NULL,
    size      INTEGER NOT NULL,
    xxhash3   TEXT NOT NULL
);
```

mtime+size are checked first (fast); xxHash3 is computed only if metadata changed.
This avoids a Datalog round-trip for the O(n) file scan on each index run.

### Index storage location

`~/.local/share/skelesearch/<sha256-of-abs-path>/`
- `index.db` — CozoDB RocksDB store
- `manifest.db` — SQLite hash manifest

One directory per indexed project, derived from the absolute path. No configuration needed.

---

## Ingestion Pipeline

```
index_codebase(path)
  1. Walk with `ignore` crate (WalkParallel) — respects all gitignore tiers
  2. For each file:
       a. Check mtime+size against manifest → skip if unchanged
       b. Compute xxHash3 → skip if hash matches
       c. Detect language from extension
       d. Parse with tree-sitter via LanguageConfig
       e. Chunk using text-splitter CodeSplitter (recursive merge, non-whitespace char budget)
       f. Normalize content (snake_case → spaces, camelCase → camel case)
       g. Extract import edges via per-language Query patterns
  3. Batch-embed new chunks via EmbedProvider (configurable batch size)
  4. Upsert chunk rows + embedding vectors to CozoDB
  5. Write import edges to code_edges relation
  6. Update file row + manifest hash
```

### LanguageConfig trait

```rust
pub trait LanguageConfig: Send + Sync {
    fn file_extensions(&self) -> &[&'static str];
    fn language(&self) -> tree_sitter::Language;
    fn chunk_node_kinds(&self) -> &[&'static str];
    fn import_query(&self) -> &str;   // S-expression tree-sitter query
}
```

Tier 1 languages at v1 (all have stable tree-sitter grammars):

| Language | Extensions | Chunk boundaries | Import query target |
|---|---|---|---|
| Rust | `.rs` | `function_item`, `impl_item`, `struct_item`, `trait_item`, `enum_item` | `use_declaration` |
| Nix | `.nix` | `function_expression`, `let_expression`, `attrset_expression` | `inherit`, `with_expression` |
| Python | `.py` | `function_definition`, `class_definition` | `import_statement`, `import_from_statement` |
| TypeScript | `.ts`, `.tsx` | `function_declaration`, `class_declaration`, `method_definition` | `import_declaration` |
| JavaScript | `.js`, `.jsx` | `function_declaration`, `class_declaration` | `import_declaration` |
| Go | `.go` | `function_declaration`, `method_declaration` | `import_declaration` |

Tier 2 fallback: `SlidingWindowChunker` (512 non-whitespace chars, 64 overlap) for all other extensions.

Adding a new Tier 1 language = implement `LanguageConfig` for that language. Register in a
`HashMap<&str, Box<dyn LanguageConfig>>` keyed by file extension. No other changes needed.

### EmbedProvider trait

```rust
#[async_trait]
pub trait EmbedProvider: Send + Sync {
    fn dim(&self) -> usize;
    async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>>;
}
```

Built-in providers:
- `FastEmbedProvider` (`skelesearch-embed-fastembed`) — `jina-embeddings-v2-base-code` (768-dim), in-process ONNX
- `OllamaProvider` (optional feature, via `skg-provider-ollama`) — `nomic-embed-text`, `mxbai-embed-large`
- `OpenAIProvider` (optional feature, via `skg-provider-anthropic`) — `text-embedding-3-small`, `voyage-code-3`

---

## Retrieval

### Hybrid search (BM25 + HNSW → RRF)

```datalog
# Vector retrieval leg
vec_hits[rank, file_path, chunk_idx] :=
    ~chunks:semantic{ file_path, chunk_idx |
        query: $query_vec, k: 50, ef: 50, bind_distance: dist },
    rank = rank_of(dist)  # ascending distance = better rank

# BM25 retrieval leg
fts_hits[rank, file_path, chunk_idx] :=
    ~chunks:text{ file_path, chunk_idx |
        query: $query_str, k: 50, bind_score: score },
    rank = rank_of(score)  # descending score = better rank

# RRF fusion (k=60)
rrf_score[file_path, chunk_idx, score] :=
    vec_hits[vr, file_path, chunk_idx] or fts_hits[fr, file_path, chunk_idx],
    score = ifelse(is_bound(vr), 1.0/(60+vr), 0.0)
           + ifelse(is_bound(fr), 1.0/(60+fr), 0.0)

# Final results with content
?[score, file_path, chunk_idx, content, start_line, end_line, chunk_type] :=
    rrf_score[file_path, chunk_idx, score],
    *chunks[file_path, chunk_idx, content, _, chunk_type, start_line, end_line, _],
    :order -score
    :limit $top_k
```

### Graph-augmented retrieval

When `include_graph: true`, a second Datalog query fetches callers/importers of top results:

```datalog
?[file_path, chunk_idx, content] :=
    *code_edges[file_path, chunk_idx, $target_file, 'imports'],
    *chunks[file_path, chunk_idx, content, _, _, _, _, _]
```

Results are appended to the response with `"why": "imports <target>"` annotation.

---

## MCP Tools

Exposed via `rmcp` 0.16 over stdio. Schema auto-generated from `schemars::JsonSchema` on input structs.

| Tool | Input | Output |
|---|---|---|
| `index_codebase` | `path: String`, `provider?: String` | Status string, chunk count |
| `search_code` | `query: String`, `top_k?: u32`, `include_graph?: bool` | JSON array of `{file, start_line, end_line, content, score, why}` |
| `get_file_context` | `file_path: String` | All chunks for file + its imports + importers |
| `index_status` | `path?: String` | `{indexed_files, total_chunks, last_indexed, stale_files}` |

### MCP server startup

```rust
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let server = SkeleSearchServer::new(config).await?;
    let peer = server.serve((tokio::io::stdin(), tokio::io::stdout())).await?;
    peer.waiting().await?;
    Ok(())
}
```

---

## CLI Commands

```
skelesearch index <path> [--provider fastembed|ollama|openai|voyage] [--watch]
skelesearch search "<query>" [--top-k 5] [--graph] [--json]
skelesearch context <file>
skelesearch status [<path>]
skelesearch clear [<path>]
```

`--watch` starts a background file watcher (`notify` + `notify-debouncer-full`) that re-indexes
changed files on save. Debounce window: 1 second. Off by default in v1.

---

## Incremental Indexing

Strategy: **file-level delta with metadata-filtered cleanup**

1. On each run: walk with `ignore`, check mtime+size first (fast pre-filter), then xxHash3
2. Classify files: `added | modified | deleted | unchanged`
3. For `modified` and `deleted`: `DELETE FROM chunks WHERE file_path = ?` + delete manifest row
4. For `added` and `modified`: parse, chunk, embed, upsert
5. Persist updated manifest to SQLite

Import edge cleanup follows the same pattern: edges are deleted by `from_file` before re-extraction.

**No Merkle tree needed** — that is a client-server sync mechanism. Flat manifest is sufficient
for local single-machine indexing.

---

## Nix Packaging

```nix
flake.nix outputs:
  packages.default           = skelesearch-cli binary
  packages.mcp-server        = skelesearch-mcp binary
  packages.skelesearch-cli   = alias for default
  packages.skelesearch-mcp   = explicit name

  devShells.default          = Rust toolchain + cmake (for RocksDB)
```

Builds via `crane` (incremental Rust builds in Nix). The RocksDB dependency requires cmake
and a C++20-capable compiler — both are available in `pkgs.buildInputs` on darwin and linux.

Use `storage-sqlite` feature during development (faster compile), `storage-rocksdb` in release.

---

## Known Risks

| Risk | Mitigation |
|---|---|
| CozoDB stalled at v0.7.6 (Dec 2023) | All CozoDB code isolated in `core/src/schema.rs` behind `StorageBackend` trait. Migration to LanceDB+Tantivy is a single-file change. |
| RocksDB compile time (~10 min cold) | Use SQLite backend for dev builds. CI uses incremental caching. |
| Embedding bottleneck (5-50 chunks/sec) | Async batch embedding with configurable batch size. fastembed in-process eliminates HTTP overhead. |
| jina-v2-base-code 768-dim vs 1536-dim providers | Dim is configured at index creation from `EmbedProvider::dim()`. Re-index required if provider changes. |
