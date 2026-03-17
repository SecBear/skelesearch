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
  plugin/                     Claude Code plugin (hooks + skills + scout agent)
    plugin.json
    hooks/
      session-start.sh
      post-edit-reindex.sh
    skills/
      search-code/
        SKILL.md
    agents/
      skelesearch-scout.md
    CLAUDE.md.template
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
     Collect the full set of visited paths into a HashSet<String>.
  2. For each visited file:
       a. Check mtime+size against manifest → skip if unchanged
       b. Compute xxHash3 → skip if hash matches
       c. For modified files: delete chunks + edges by file_path before re-indexing
       d. Detect language from extension
       e. Parse with tree-sitter via LanguageConfig
       f. Chunk using text-splitter CodeSplitter (recursive merge, non-whitespace char budget)
       g. Normalize content (snake_case → spaces, camelCase → camel case)
       h. Extract import edges via per-language Query patterns
  2.5 Reconcile against manifest (handles renames and deletes):
       - Query manifest for all known file_paths
       - Diff: known_paths - visited_paths = stale paths
       - For each stale path: delete chunks, delete edges, delete manifest row, delete files row
       (A rename is a delete of the old path + add of the new path; the walk naturally
        visits the new path and never visits the old path, so reconciliation handles it.)
  3. Batch-embed new/modified chunks via EmbedProvider (configurable batch size)
  4. Upsert chunk rows + embedding vectors to CozoDB
  5. Write import edges to code_edges relation
  6. Update file row + manifest hash for all added/modified files
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

CozoDB's disjunctive `or` semantics differ from SQL FULL OUTER JOIN — variables in one
branch are not in scope in the other. RRF fusion is implemented with two separate scored
rules that are unioned, then aggregated by doc id:

```datalog
# Vector retrieval leg — assigns rank 1..k by ascending distance
vec_scored[file_path, chunk_idx, score] :=
    ~chunks:semantic{ file_path, chunk_idx |
        query: $query_vec, k: 50, ef: 50, bind_distance: dist },
    score = 1.0 / (60.0 + dist * 50.0)  # approximate rank via distance

# BM25 retrieval leg — assigns score by BM25 rank
fts_scored[file_path, chunk_idx, score] :=
    ~chunks:text{ file_path, chunk_idx |
        query: $query_str, k: 50, bind_score: bm25 },
    score = 1.0 / (60.0 + 1.0 / (bm25 + 0.001))  # rank approximated from score

# Union both legs, aggregate RRF contributions per doc
rrf[file_path, chunk_idx, sum(score)] :=
    vec_scored[file_path, chunk_idx, score]
rrf[file_path, chunk_idx, sum(score)] :=
    fts_scored[file_path, chunk_idx, score]

# Final results with content
?[rrf_score, file_path, chunk_idx, content, start_line, end_line, chunk_type] :=
    rrf[file_path, chunk_idx, rrf_score],
    *chunks[file_path, chunk_idx, content, _, chunk_type, start_line, end_line, _],
    :order -rrf_score
    :limit $top_k
```

Note: The exact rank-to-score mapping will be validated against CozoDB 0.7.6 query
semantics during implementation. The key invariant is: chunks appearing in both legs
receive a higher combined score than chunks appearing in only one.

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
| `index_codebase` | `path: String`, `provider?: String` | `{status: String, indexed: u32, chunks: u32}` |
| `search_code` | `query: String`, `top_k?: u32`, `include_graph?: bool` | JSON array of `{file, start_line, end_line, content, score, why}` |
| `get_file_context` | `file_path: String` | `{chunks: [...], imports: [...], imported_by: [...]}`. Returns empty arrays (not an error) if the file has not been indexed. |
| `index_status` | `path?: String` | `{indexed_files: u32, total_chunks: u32, last_indexed: String (ISO8601), estimated_stale: u32}` |

`estimated_stale` in `index_status` is computed by comparing mtime+size from the manifest
against current filesystem state — it is a fast estimate (no full walk, no re-hashing)
and may not reflect renames or deletions detected only during a full walk.

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
skelesearch index <path> [--provider fastembed|ollama|openai|voyage]
skelesearch search "<query>" [--top-k 5] [--graph] [--json]
skelesearch context <file>
skelesearch status [<path>]
skelesearch clear [<path>]
skelesearch watch <path> [--provider fastembed|ollama|openai|voyage]
```

`watch` is a separate subcommand (not a flag on `index`) that starts a persistent background
file watcher. Uses `notify` 6.x + `notify-debouncer-full` (handles vim rename-over-tempfile).
Debounce window: 1 second. Opt-in in v1 — not invoked automatically by `index`.

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

## StorageBackend Trait

All CozoDB-specific code lives in `crates/core/src/schema.rs` behind this trait.
The `CozoBackend` struct in that file is the only implementation in v1. Migration to
LanceDB+Tantivy means providing a second implementation — no other files change.

```rust
#[async_trait]
pub trait StorageBackend: Send + Sync {
    /// Create schema (relations + indices) if not already present.
    /// `dim` must match `EmbedProvider::dim()` — stored in index metadata.
    async fn initialize(&self, dim: usize) -> Result<()>;

    // --- File records ---
    async fn upsert_file(&self, record: &FileRecord) -> Result<()>;
    async fn delete_file(&self, file_path: &str) -> Result<()>;
    /// Returns all file_paths currently tracked in the backend.
    async fn list_indexed_paths(&self) -> Result<Vec<String>>;

    // --- Chunks ---
    async fn upsert_chunks(&self, chunks: &[ChunkRecord]) -> Result<()>;
    async fn delete_chunks_for_file(&self, file_path: &str) -> Result<()>;
    async fn get_chunks_for_file(&self, file_path: &str) -> Result<Vec<ChunkRecord>>;

    // --- Edges ---
    async fn upsert_edges(&self, edges: &[EdgeRecord]) -> Result<()>;
    async fn delete_edges_for_file(&self, file_path: &str) -> Result<()>;
    /// Returns file_paths that import `file_path`.
    async fn get_importers(&self, file_path: &str) -> Result<Vec<String>>;
    /// Returns file_paths imported by `file_path`.
    async fn get_imports(&self, file_path: &str) -> Result<Vec<String>>;

    // --- Search ---
    async fn hybrid_search(
        &self,
        query_vec: &[f32],
        query_str: &str,
        top_k: usize,
    ) -> Result<Vec<SearchResult>>;

    // --- Status ---
    async fn stats(&self) -> Result<IndexStats>;
}

pub struct FileRecord {
    pub file_path: String,
    pub language: String,
    pub last_modified: i64,
    pub chunk_count: usize,
}

pub struct ChunkRecord {
    pub file_path: String,
    pub chunk_idx: usize,
    pub content: String,
    pub normalized: String,
    pub chunk_type: String,
    pub start_line: usize,
    pub end_line: usize,
    pub embedding: Option<Vec<f32>>,
}

pub struct EdgeRecord {
    pub from_file: String,
    pub from_chunk: usize,
    pub to_file: String,
    pub edge_type: String,  // "imports"
}

pub struct SearchResult {
    pub file_path: String,
    pub chunk_idx: usize,
    pub content: String,
    pub start_line: usize,
    pub end_line: usize,
    pub chunk_type: String,
    pub score: f32,
    pub why: String,  // "vector", "fts", or "both"
}

pub struct IndexStats {
    pub indexed_files: u32,
    pub total_chunks: u32,
    pub last_indexed: Option<chrono::DateTime<chrono::Utc>>,
}
```

---

## Claude Code Plugin Layer

The plugin layer gives skelesearch zero-friction UX inside Claude Code. The agent never
needs to decide "should I index this?" or "should I search before editing?" — both happen
automatically through two complementary mechanisms: **hooks** (guaranteed injection at session
boundaries) and **PROACTIVELY descriptions** (in-session auto-dispatch when context is relevant).

### Plugin directory layout

```
plugin/
  plugin.json                    # Plugin manifest (name, description, version)
  hooks/
    session-start.sh             # SessionStart hook: auto-index + inject context
    post-edit-reindex.sh         # PostToolUse hook: incremental reindex after Write/Edit
  skills/
    search-code/
      SKILL.md                   # Natural language code search (PROACTIVELY framed)
  agents/
    skelesearch-scout.md         # Haiku scout agent for codebase exploration
  CLAUDE.md.template             # CLAUDE.md snippet for project-level agent guidance
```

### SessionStart hook

`hooks/session-start.sh` runs when Claude Code starts a session. It calls
`skelesearch status --json` on the current directory, then emits a JSON
`additionalContext` payload to stdout. Claude receives this as implicit system
context before any user message — the agent never has to think "should I check the index?"

Behavior:

1. **If indexed**: emits `additionalContext` with live stats:
   ```json
   {
     "type": "additionalContext",
     "content": "skelesearch: 1,247 files indexed across 8,432 chunks (last indexed: 2h ago). Use search_code for semantic queries before modifying existing code."
   }
   ```

2. **If not indexed**: spawns `skelesearch index . &` asynchronously, emits:
   ```json
   {
     "type": "additionalContext",
     "content": "skelesearch: indexing this codebase in the background (first run). search_code will be available once complete — use grep for now."
   }
   ```

3. **If skelesearch not installed / MCP unconfigured**: emits nothing. Hook exit code 0.
   This is a graceful no-op — the hook must never error on systems without skelesearch.

### PostToolUse hook

`hooks/post-edit-reindex.sh` fires after every `Write` or `Edit` tool completion.
It runs `skelesearch index . --incremental` as a fire-and-forget background process.
The manifest's mtime+size pre-filter means unchanged files are skipped in microseconds;
only genuinely modified files trigger re-embedding.

Hook wiring in `plugin.json`:
```json
{
  "PostToolUse": [{
    "matcher": "Write|Edit",
    "hooks": [{
      "type": "command",
      "command": "bash ${PLUGIN_DIR}/hooks/post-edit-reindex.sh"
    }]
  }]
}
```

### search-code skill

`skills/search-code/SKILL.md` frontmatter:

```yaml
---
name: search-code
description: >
  Search the indexed codebase with natural language. Use this PROACTIVELY:
  - BEFORE modifying existing code — understand what's already there
  - When you need to find code by intent rather than by exact symbol name
  - When grep would require knowing the exact identifier in advance
  - When starting work in an unfamiliar part of the codebase
  - When you need to understand how a file fits into the broader system
  Prefer grep/Bash for: finding all occurrences of an exact known symbol,
  file existence checks, line-count operations.
allowed-tools: mcp__skelesearch__search_code, mcp__skelesearch__get_file_context
---
```

The `description` field is what Claude reads when deciding whether to invoke this skill.
The "BEFORE modifying" framing is the trigger that makes agents reach for semantic search
proactively, not reactively. This mirrors the technique used in `claude-context`.

### skelesearch-scout subagent

`agents/skelesearch-scout.md`:

```yaml
---
name: skelesearch-scout
description: >
  Use this agent PROACTIVELY when the user wants to understand how existing code
  works, find where something is implemented, or before making changes to existing
  features. Uses semantic search over the indexed codebase and returns a concise
  summary with file locations and line numbers.
tools: mcp__skelesearch__search_code, mcp__skelesearch__get_file_context, Read
model: haiku
color: blue
maxTurns: 5
---

You are a code navigation agent. Given a topic or question, search the indexed
codebase and return a concise summary of relevant chunks with file paths and
line numbers. Never modify files — read and report only.
```

The scout uses `haiku` for speed and cost. The `PROACTIVELY` keyword in `description`
causes the parent agent to dispatch it automatically when working in unfamiliar code.
It complements the SessionStart hook: the hook fires once per session to establish
index context; the scout fires per-task when semantic search is relevant.

### MCP tool description (search_code)

The `search_code` tool's description, in `crates/mcp/src/main.rs`, must include the
proactive framing — this is what non-plugin consumers (Cursor, Windsurf, any MCP client)
see:

```
Search the indexed codebase with natural language.

Use this BEFORE making changes to any existing code — understand what's already there.
Use this when you don't know what a feature is called (intent search).
Use this when grep would require knowing the exact symbol name in advance.

Returns: file path, line range, code chunk, relevance score.
Prefer grep/Bash for: all occurrences of an exact known symbol, file existence checks.
```

### CLAUDE.md template

`CLAUDE.md.template` — projects can append this snippet to their own CLAUDE.md:

```markdown
## Code Search

This project is indexed with skelesearch. Before modifying existing code, use
`search_code` (or `/search-code` skill) to understand what's already there.
This finds code by intent — complementary to grep, not a replacement.

Prefer `search_code` when: exploring unfamiliar code, finding where a concept is
implemented, understanding how files relate to each other.
Prefer grep when: finding all occurrences of a symbol name you already know exactly.
```

Keep this under 10 lines. Project CLAUDE.md should stay under 200 lines total for
reliable adherence — Claude Code truncates context beyond that.

### Installation

```bash
# Install plugin globally (available in all Claude Code sessions)
claude plugin install path/to/skelesearch/plugin/

# Or install from the published repo:
claude plugin install github:you/skelesearch

# Configure MCP server per-project (.mcp.json at repo root):
{
  "mcpServers": {
    "skelesearch": {
      "command": "skelesearch-mcp",
      "args": []
    }
  }
}
```

After plugin install: the SessionStart hook fires on every `claude` invocation in any
directory, checks the index, and injects context automatically. The MCP server config
in `.mcp.json` gives the agent the actual search tools. Both are needed; either alone
is incomplete.

---

## Known Risks

| Risk | Mitigation |
|---|---|
| CozoDB stalled at v0.7.6 (Dec 2023) | All CozoDB code isolated in `core/src/schema.rs` behind `StorageBackend` trait. Migration to LanceDB+Tantivy is a single-file change. |
| RocksDB compile time (~10 min cold) | Use SQLite backend for dev builds. CI uses incremental caching. |
| Embedding bottleneck (5-50 chunks/sec) | Async batch embedding with configurable batch size. fastembed in-process eliminates HTTP overhead. |
| jina-v2-base-code 768-dim vs 1536-dim providers | Dim is configured at index creation from `EmbedProvider::dim()`. Re-index required if provider changes. |
