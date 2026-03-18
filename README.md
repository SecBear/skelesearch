# skelesearch

Local semantic code search and memory layer for agentic systems.

skelesearch indexes a codebase into chunked embeddings and a full-text index,
then exposes hybrid (BM25 + vector) search over both CLI and MCP. Designed for
Claude Code, Codex, OMP/OpenClaw, and any MCP-compatible agent.

## Why

Agents work better with retrieval. `grep` finds exact strings; skelesearch
finds *semantically related code* — functions that do similar things, error
handling patterns, related types — even when the query doesn't share keywords
with the result.

## Quick start

```bash
# Build
cargo install --path crates/cli --features storage-sqlite

# Index your project
cd /path/to/project
skelesearch index .

# Search
skelesearch search "retry logic with backoff" --json
```

The index lives in `.skelesearch/` at the project root. Add it to `.gitignore`.

## How it works

1. **Chunking** — tree-sitter parses 15 languages into AST-aware chunks
   (functions, structs, impl blocks). Unknown languages fall back to a sliding
   window.

2. **Embedding** — each chunk is embedded via
   [jina-embeddings-v2-base-code](https://huggingface.co/jinaai/jina-embeddings-v2-base-code)
   (768-dim, code-specialized ONNX model, runs locally via fastembed-rs). An
   **embedding cache** (SQLite-backed, keyed by content hash) skips
   re-embedding unchanged chunks on subsequent runs.

3. **Storage** — CozoDB stores chunks, embeddings (HNSW index), full-text
   (BM25), import graph edges, and symbol definitions.

4. **Search** — queries run through both HNSW vector search and BM25 full-text
   search. Results are fused via **Reciprocal Rank Fusion** (RRF), then
   optionally reranked with **Maximal Marginal Relevance** (MMR) for diversity.

## CLI

```
skelesearch index <path>              Index a directory
skelesearch search <query>            Hybrid semantic + FTS search
  --top-k N                             Max results (default: 5)
  --diversity 0.3                       MMR re-ranking (0=off, 1=max diversity)
  --json                                JSON output
skelesearch grep <pattern>            Regex search over indexed files
  -i, --ignore-case                     Case insensitive
  --max-results N                       Limit (default: 50)
  --json                                JSON output
skelesearch symbol <name>             Find symbol definitions
  --kind function|struct|class|...      Filter by kind
skelesearch context <file>            Show chunks + import graph for a file
skelesearch status [--json]           Index statistics
skelesearch gc                        Remove entries for deleted files
skelesearch clear                     Delete the entire local index
skelesearch watch <path>              Maintain a watch sentinel (v1: no auto-reindex)
```

## MCP server

For agent integration (Claude Code, Codex, OMP):

```bash
# stdio transport (default)
skelesearch-mcp

# HTTP transport (for non-subprocess consumers)
skelesearch-mcp --http 127.0.0.1:3000
```

### MCP tools

| Tool | Description |
|------|-------------|
| `smart_search` | Auto-classifies query → grep or semantic search. **Recommended default.** |
| `search_code` | Hybrid BM25 + vector search with MMR diversity control |
| `find_symbol` | Look up definitions by name, optionally filtered by kind |
| `get_file_context` | All chunks + import edges for a specific file |
| `index_codebase` | Trigger indexing from within a session |
| `index_status` | Check whether the index is current |

### Claude Code config

```json
{
  "mcpServers": {
    "skelesearch": {
      "command": "skelesearch-mcp",
      "args": []
    }
  }
}
```

See [docs/integrations.md](docs/integrations.md) for Codex, OMP, HTTP, and CLI
integration guides.

## Supported languages

Rust, Python, TypeScript, JavaScript, Go, Java, C, C++, Nix, Ruby, PHP, C#,
Kotlin, Swift, Scala. Unknown extensions use a sliding-window fallback.

## Architecture

```
crates/
  core/           Schema, indexer, searcher, chunker, manifest, GC, grep, symbols
  embed-fastembed/ FastEmbed ONNX provider (jina-v2-base-code)
  mcp/            MCP server (stdio + HTTP transport)
  cli/            CLI binary
```

Key design boundaries:
- `StorageBackend` trait — all CozoDB access is behind this; migration to
  LanceDB+Tantivy is a single-file change
- `EmbedProvider` trait — swap embedding models without touching indexing logic
- Manifest (SQLite) — change detection, crash recovery, embedding cache

## Observability

```bash
# Debug-level span traces for every index/search operation
RUST_LOG=skelesearch_core=debug skelesearch index .
RUST_LOG=skelesearch_core=debug skelesearch search "error handling"
```

CLI output includes elapsed time and embedding cache statistics. MCP
`index_codebase` returns `cache_hits` in the response. All hot-path functions
carry `#[tracing::instrument]` spans — wire up `tracing-opentelemetry` for
Jaeger/OTLP export when needed.

## Configuration

Optional `.skelesearch.toml` at project root:

```toml
[index]
exclude = ["target/", "node_modules/", "*.lock"]

[search]
top_k = 10
```

## License

MIT OR Apache-2.0
