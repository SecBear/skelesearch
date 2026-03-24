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
  .claude-plugin/             Claude Code plugin manifest
    plugin.json               (pure metadata: name, version, description, author)
  hooks/                      Hook scripts + wiring (repo root, not plugin/ subdir)
    hooks.json                Hook event wiring (SessionStart, PostToolUse)
    session-start             SessionStart script (no .sh — prevents Windows bash prepend)
    post-edit-reindex         PostToolUse script (no .sh)
  skills/                     Claude Code skills (auto-discovered from this directory)
    search-code/
      SKILL.md
  agents/                     Claude Code agent definitions
    skelesearch-scout.md
  CLAUDE.md.template          CLAUDE.md snippet for project-level guidance
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

**Target chunk size:** 1,500 non-whitespace characters for Tier 1 AST-aware chunking
(≈ 500 tokens; large enough for a coherent function body, small enough for focused retrieval).
The CodeSplitter will split oversized AST nodes and merge small siblings to stay near this budget.

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
    *code_edges[file_path, chunk_idx, $target_file, 'imports', _],
    *chunks[file_path, chunk_idx, content, _, _, _, _, _]
```

Results are appended to the response with `"why": "imports <target>"` annotation.

---

## MCP Tools

Exposed via `rmcp` 0.16 over stdio. Schema auto-generated from `schemars::JsonSchema` on input structs.

| Tool | Input | Output |
|---|---|---|
| `index_codebase` | `path: String`, `provider?: String` | `{status: String, indexed: u32, chunks: u32}` |
| `search_code` | `query: String`, `top_k?: u32`, `include_graph?: bool` | JSON array of `{file_path, start_line, end_line, content, score, match_quality, why}` |
| `get_file_context` | `file_path: String` | `{chunks: [...], imports: [...], imported_by: [...]}`. Returns empty arrays (not an error) if the file has not been indexed. |
| `index_status` | `path?: String` | `{indexed_files: u32, total_chunks: u32, last_indexed: String (ISO8601), estimated_stale: u32, watching: bool}` |

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
    pub match_quality: String, // "high" | "moderate" | "low" derived from RRF score thresholds
    pub why: String,           // "vector" | "fts" | "both"
}
// match_quality thresholds (to be calibrated during implementation):
// "high"     — score >= 0.8 × top result's score (clearly relevant)
// "moderate" — score >= 0.5 × top result's score (probably relevant)
// "low"      — score < 0.5 × top result's score (marginal — agent should treat skeptically)

pub struct IndexStats {
    pub indexed_files: u32,
    pub total_chunks: u32,
    pub last_indexed: Option<chrono::DateTime<chrono::Utc>>,
    pub watching: bool,   // true if a watch process is active for this directory
}
```

---

## Claude Code Plugin Layer

The plugin layer gives skelesearch zero-friction UX inside Claude Code. The agent never
needs to decide "should I index this?" or "should I search before editing?" — both happen
automatically through two complementary mechanisms: **hooks** (guaranteed injection at session
boundaries) and **PROACTIVELY descriptions** (in-session auto-dispatch when context is relevant).

The plugin root is the **skelesearch repo root itself**. Claude Code discovers
`.claude-plugin/plugin.json` at the repo root on `claude plugin install github:you/skelesearch`.

### `.claude-plugin/plugin.json`

Pure metadata only. Hooks, skills, and agents are NOT declared here — they are
auto-discovered from their directories by Claude Code.

```json
{
  "name": "skelesearch",
  "description": "Semantic code search for AI coding agents — AST chunking, hybrid BM25+HNSW retrieval",
  "version": "0.1.0",
  "author": { "name": "you", "email": "you@example.com" },
  "homepage": "https://github.com/you/skelesearch",
  "repository": "https://github.com/you/skelesearch",
  "license": "MIT",
  "keywords": ["code-search", "semantic", "mcp", "claude-code"]
}
```

### `hooks/hooks.json`

Hooks are declared here, NOT in `plugin.json`. Format confirmed from official Anthropic
plugins (ralph-wiggum, superpowers v5):

```json
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "startup|clear|compact",
        "hooks": [
          {
            "type": "command",
            "command": "\"${CLAUDE_PLUGIN_ROOT}/hooks/session-start\"",
            "async": false
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "command",
            "command": "\"${CLAUDE_PLUGIN_ROOT}/hooks/post-edit-reindex\"",
            "async": true
          }
        ]
      }
    ]
  }
}
```

Key notes:
- `async: false` on SessionStart — must block until context is injected before Claude responds
- `async: true` on PostToolUse — fire-and-forget; indexing must not block the agent's next turn
- `matcher` is a regex — `startup|clear|compact` limits SessionStart to session-open events,
  not every prompt. Without `matcher`, the hook would fire on every tool call.
- `${CLAUDE_PLUGIN_ROOT}` is the env var Claude Code injects pointing to the installed plugin root
- Script filenames are extensionless (`session-start` not `session-start.sh`) to prevent
  Windows from auto-prepending `bash` to `.sh` filenames

### `hooks/session-start`

Runs at session start (`async: false` — blocks). Calls `skelesearch status --json`,
emits `hookSpecificOutput` JSON to stdout.

**Output format** (confirmed from Anthropic's official plugins):
```json
{
  "hookSpecificOutput": {
    "hookEventName": "SessionStart",
    "additionalContext": "<compact status string, under 100 words>"
  }
}
```

Use `printf` (not `cat <<EOF` heredoc) to produce this JSON — bash 5.3+ hangs on
heredoc expansion when content exceeds ~512 bytes. Self-derive plugin root before
trusting the env var (needed during install/testing):

```bash
PLUGIN_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="${CLAUDE_PLUGIN_ROOT:-$PLUGIN_ROOT}"
```

**Behavior:**

1. **If indexed**: emits `additionalContext` with live stats:
   > "skelesearch: 1,247 files indexed, 8,432 chunks (last: 2h ago). search_code
   > available. Use before modifying existing code — results are candidates to verify."

2. **If not indexed**: spawns `skelesearch index . &` asynchronously, emits:
   > "skelesearch: indexing in background (first run). Use grep for now — search_code
   > will be available when indexing completes."

3. **If skelesearch not installed or no code files in cwd**: exit 0, no output.
   Never error — graceful no-op when unconfigured.

**Context size discipline**: keep `additionalContext` under 100 words. Injecting too much
at session start degrades response quality (40-60% context utilization target per
ACE/FCA research). Status + one-line instruction is sufficient; CLAUDE.md handles conventions.

### `hooks/post-edit-reindex`

Fires after Write or Edit tool calls (`async: true` — does not block). Runs
`skelesearch index .` as a fire-and-forget background process. Indexing is always
incremental by design — unchanged files are skipped in microseconds.

Before spawning, checks `skelesearch status --json | jq -r '.watching'` — if `true`,
the watch daemon is already handling changes and the hook exits 0 immediately.

### `skills/search-code/SKILL.md`

```yaml
---
name: search-code
description: >
  Use when the task requires finding code by intent rather than by exact symbol name,
  or before modifying existing code to understand what's already there. Use when the
  relevant code is not known in advance (no exact identifier to grep for), when
  exploring an unfamiliar module, or when understanding how a file relates to the
  broader system.
allowed-tools: mcp__skelesearch__search_code, mcp__skelesearch__get_file_context
---

Invoke `search_code` to locate code by intent. Results are CANDIDATES to verify —
not ground truth to act on directly. Confirm the result is the right symbol/file
before modifying.

Use `get_file_context` to understand a file's imports and importers (graph context).
Prefer grep/Bash for: finding all occurrences of an exact known symbol, file existence.
```

**Description discipline** (from superpowers writing-skills): the `description` field
MUST describe ONLY triggering conditions — not the workflow or what the skill does.
Summarizing the workflow in the description causes Claude to shortcut and skip the body.
This is a documented failure mode. The current description above lists conditions only.

### `agents/skelesearch-scout.md`

```yaml
---
name: skelesearch-scout
description: |
  Use this agent PROACTIVELY when the user wants to understand how existing code
  works, find where something is implemented, or before making changes to existing
  features. Uses semantic search over the indexed codebase.
tools: mcp__skelesearch__search_code, mcp__skelesearch__get_file_context, Read
model: haiku
color: blue
maxTurns: 5
---

You are a code navigation agent. Given a topic or question, search the indexed
codebase and return a concise summary of relevant chunks with file paths and
line numbers. Never modify files — read and report only.
```

### MCP tool description (`search_code`)

Framed as success criteria (not imperative) — agents respond better to goals than
instructions (Karpathy Jan 2026, Cherny Jan 2026). The `CANDIDATES` framing
counteracts agents' tendency to silently run with wrong assumptions:

```
Locate code by semantic intent. Returns candidate locations so you can verify
them, then read precisely.

Use BEFORE modifying existing code — confirm what's already there.
Use when you don't know the exact symbol name (intent search).
Use when grep would require knowing the identifier in advance.

Results include: file_path, line range, content, score, match_quality, why.
Treat results as CANDIDATES to verify — not ground truth to act on directly.
Confirm the right file/symbol before modifying.
Prefer grep/Bash for: all occurrences of a known exact symbol, file existence.
```

### CLAUDE.md template

```markdown
## Code Search

This project is indexed with skelesearch. Before modifying existing code, use
`search_code` to understand what's already there. Results are candidates — verify
before acting. Use `get_file_context` to see a file's imports and importers.

Prefer `search_code` when: exploring unfamiliar code, finding where a concept
is implemented. Prefer grep when: finding all occurrences of a known exact symbol.
```

Keep under 10 lines. Project CLAUDE.md should stay under 200 lines total for reliable
adherence. Coding conventions belong in a separate `standards/` directory injected
per task, not loaded into CLAUDE.md on every session.

### Installation

```bash
# Install plugin globally
claude plugin install path/to/skelesearch/
# Or from published repo:
claude plugin install github:you/skelesearch

# Configure MCP server per-project (.mcp.json at repo root):
{
  "mcpServers": {
    "skelesearch": { "command": "skelesearch-mcp", "args": [] }
  }
}
```

The SessionStart hook fires on every `claude` session start, checks the index, and
injects context automatically. The `.mcp.json` config gives the agent the actual search
tools. Both are needed; either alone is incomplete.

---

## Functional Requirements

v1 = in scope for first release; v2 = planned future; out-of-scope = explicitly excluded.

| ID | Requirement | Scope |
|---|---|---|
| FR-001 | Index a directory with tree-sitter AST-aware chunking (Tier 1 languages) | v1 |
| FR-002 | Store chunks in CozoDB with HNSW vector index + BM25 FTS index | v1 |
| FR-003 | Embed chunks in-process with fastembed-rs (jina-v2-base-code, 768-dim) | v1 |
| FR-004 | Pluggable EmbedProvider trait (Ollama, OpenAI/Voyage swappable) | v1 |
| FR-005 | Hybrid BM25 + HNSW retrieval fused with RRF in a single Datalog query | v1 |
| FR-006 | Incremental indexing: skip unchanged files via mtime+size+xxHash3 manifest | v1 |
| FR-007 | Handle file renames and deletions via manifest reconciliation | v1 |
| FR-008 | Expose 4 MCP tools via rmcp 0.16 (index_codebase, search_code, get_file_context, index_status) | v1 |
| FR-009 | CLI: index, search, context, status, clear, watch subcommands | v1 |
| FR-010 | Claude Code plugin: SessionStart hook (auto-index + context injection) | v1 |
| FR-011 | Claude Code plugin: PostToolUse hook (incremental reindex after Write/Edit) | v1 |
| FR-012 | Import edge storage (code_edges) and one-hop graph-augmented retrieval | v1 |
| FR-013 | Nix flake packaging for CLI and MCP server binaries | v1 |
| FR-014 | SPLADE sparse retrieval via Seismic (pending code-domain benchmarks) | v2 |
| FR-015 | Diff-aware retrieval mode (restrict search to files modified in current branch) | v2 |
| FR-016 | Call graph edges via stack-graphs | v2 |
| FR-017 | Closed-loop self-verification (re-query with broadened terms when results < threshold) | v2 |
| FR-018 | "This result was irrelevant" feedback signal for RRF weight tuning | v2 |

**Out of scope (v1):** Docker/daemon deployment; model fine-tuning; web UI or IDE extension;
multi-user shared indices; Windows native support.

### Acceptance Scenarios

**Given** a Git repo with 500 Rust files,
**when** `skelesearch index .` is run for the first time,
**then** all `.rs` files are chunked, embedded, and stored; `index_status` reports
≥ 490 indexed files; the process completes in < 10 minutes (fastembed, SQLite backend).

**Given** a session where Claude Code has the skelesearch plugin installed,
**when** a new Claude session opens in an indexed directory,
**then** the SessionStart hook fires within 100ms and `additionalContext` includes
the file count and a directive to use `search_code` before modifying code.

**Given** a `search_code` call with `query: "how are import edges extracted"`,
**when** the index contains the skelesearch codebase itself,
**then** at least one result in the top 3 has `match_quality: "high"` and references
the `LanguageConfig::import_query` function.

**Given** a file is edited via the Edit tool while the plugin is active,
**when** the PostToolUse hook fires,
**then** `skelesearch index .` runs asynchronously; `index_status` reports the
changed file as updated within 60 seconds (fastembed, SQLite).

---

## Success Criteria

All latency targets are wall-clock on aarch64 macOS (M-series) with fastembed-rs
(CPU only) and SQLite backend unless noted.

| ID | Criterion | Target |
|---|---|---|
| SC-001 | `search_code` latency, top-5, hybrid retrieval, warm index | < 200ms |
| SC-002 | `get_file_context` latency | < 50ms |
| SC-003 | `index_status` latency | < 10ms |
| SC-004 | SessionStart hook end-to-end (spawn + status check + printf) | < 100ms |
| SC-005 | Incremental index, 10 modified files in a 100k-file repo | < 30s |
| SC-006 | First-run full index throughput (fastembed batch 64) | ≥ 5 chunks/sec |
| SC-007 | MCP server cold-start to first `search_code` response | < 500ms |
| SC-008 | Plugin install + `.mcp.json` config to first successful `search_code` | ≤ 5 minutes |
| SC-009 | Manual relevance eval: 10-query test suite on skelesearch's own codebase | ≥ 70% top-3 relevant |

---

## Known Risks

| Risk | Mitigation |
|---|---|
| CozoDB stalled at v0.7.6 (Dec 2023) | All CozoDB code isolated in `core/src/schema.rs` behind `StorageBackend` trait. Migration to LanceDB+Tantivy is a single-file change. |
| RocksDB compile time (~10 min cold) | Use SQLite backend for dev builds. CI uses incremental caching. |
| Embedding bottleneck (5-50 chunks/sec) | Async batch embedding with configurable batch size. fastembed in-process eliminates HTTP overhead. |
| jina-v2-base-code 768-dim vs 1536-dim providers | Dim is configured at index creation from `EmbedProvider::dim()`. Re-index required if provider changes. |
| Plugin hook env var (`$CLAUDE_PLUGIN_ROOT`) may not be set in all execution contexts | Hook self-derives root: `PLUGIN_ROOT="$(cd "$(dirname "$0")/.." && pwd)"` as fallback before trusting env var. |
| `hooks.json` format may differ across Claude Code versions | Hook wiring confirmed from official Anthropic plugins (ralph-wiggum, superpowers v5). Pin to the confirmed format; validate during installation. |
| Agent sycophancy: agents act on plausibly-right results without verifying | `match_quality` field in `SearchResult` + "CANDIDATES" framing in tool description creates natural verification step. |
| Over-injection at SessionStart degrading context quality | `additionalContext` capped at 100 words. Status + one-line directive only. |
