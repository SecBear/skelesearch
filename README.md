# skelesearch

Local semantic code search and memory layer for agentic systems.

skelesearch indexes a codebase into chunked embeddings and a full-text index,
then exposes hybrid (BM25 + vector) search over both CLI and MCP. Designed for
Claude Code, Codex, and any MCP-compatible agent.

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

2. **Embedding** — three providers:
   - **fastembed** (default) — [jina-embeddings-v2-base-code](https://huggingface.co/jinaai/jina-embeddings-v2-base-code)
     (768-dim, code-specialized ONNX model, runs locally via fastembed-rs)
   - **openai** — `text-embedding-3-small` via the OpenAI API (`--provider openai`);
     requires `OPENAI_API_KEY` in the environment
   - **voyage** — `voyage-code-3` via the Voyage AI API (`--provider voyage`);
     requires `VOYAGE_API_KEY` in the environment

   An **embedding cache** (SQLite-backed, keyed by content hash) skips
   re-embedding unchanged chunks on subsequent runs. The **provider manifest**
   records which model was used at index time; search auto-detects it.

3. **Storage** — CozoDB stores chunks, embeddings (HNSW index), full-text
   (BM25), import graph edges, and symbol definitions.

4. **Search** — queries run through both HNSW vector search and BM25 full-text
   search. Results are fused via **Reciprocal Rank Fusion** (RRF), then
   optionally reranked with **Maximal Marginal Relevance** (MMR) for diversity.
   A **reranker** pipeline stage (cross-encoder, e.g. Jina, Qwen3) is pluggable
   via the `Reranker` trait and builder — first-party models ship separately.

## CLI

```
skelesearch index <path>              Index a directory
  --provider fastembed|openai|voyage   Embedding backend (default: fastembed)
skelesearch search <query>            Hybrid semantic + FTS search
  --top-k N                             Max results (default: 5)
  --diversity 0.3                       MMR re-ranking (0=off, 1=max diversity)
  --max-tokens N                        Cap output to a token budget
  --branch                              Scope results to files changed on current git branch
  --provider fastembed|openai|voyage   Must match the provider used at index time
  --json                                JSON output
skelesearch grep <pattern>            Regex search over indexed files
  -i, --ignore-case                     Case insensitive
  --max-results N                       Limit (default: 50)
  --json                                JSON output
skelesearch symbol <name>             Find symbol definitions
  --kind function|struct|class|...      Filter by kind
skelesearch context <file>            Show chunks + import graph for a file
skelesearch eval <eval_set.json>      Measure retrieval quality (Recall@5, Recall@10, MRR)
  --json                                JSON output
skelesearch status [--json]           Index statistics
skelesearch gc                        Remove entries for deleted files
skelesearch clear                     Delete the entire local index
skelesearch watch <path>              Re-index on file changes (2s debounce)
```

## MCP server

For agent integration (Claude Code, Codex, OMP):

```bash
# stdio transport (default)
skelesearch-mcp

# HTTP transport (for non-subprocess consumers)
skelesearch-mcp --http 127.0.0.1:3000
```

### Install for global OMP use

> Recommended when you want skelesearch available to all OMP sessions across directories.

```bash
cargo install --path crates/mcp --root ~/.local --force --features storage-rocksdb
# or use the helper script
./scripts/install-mcp.sh
```

This installs `skelesearch-mcp` to `~/.local/bin/skelesearch-mcp`. Add that binary to
`~/.omp/agent/mcp.json` for system-wide use. After rebuilding or reinstalling, restart OMP.


### MCP tools

| Tool | Description |
|------|-------------|
| `search_code` | Primary hybrid code search tool. Supports `top_k`, graph expansion, diversity, token budget, branch scope, and session dedup. |
| `find_symbol` | Look up definitions by exact symbol name, optionally filtered by kind. |
| `get_symbol_info` | One-call symbol context bundle: source, imports, dependents, and tests. |
| `find_dependents` | Discover files that import or otherwise depend on a target file. |
| `find_tests` | Find test files that cover a source file. |
| `get_repo_map` | Return a fast structural map of the indexed repo (tree, roles, symbols, edges). |
| `index` | Start background indexing for a project. |
| `get_index_status` | Check indexing progress and freshness. |

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
  core/             Schema, indexer, searcher, chunker, manifest, GC, grep, symbols
  embed-fastembed/  FastEmbed ONNX provider (jina-v2-base-code)
  embed-openai/     OpenAI API provider (text-embedding-3-small)
  embed-voyage/     Voyage AI provider (voyage-code-3)
  mcp/              MCP server (stdio + HTTP transport)
  cli/              CLI binary
```

Key design boundaries:
- `StorageBackend` trait — all CozoDB access is behind this; migration to
  LanceDB+Tantivy is a single-file change
- `EmbedProvider` trait — swap embedding models without touching indexing logic
- `Reranker` trait — pluggable cross-encoder stage after RRF fusion
- Manifest (SQLite) — change detection, crash recovery, embedding cache, provider record

## Observability

```bash
# Debug-level span traces for every index/search operation
RUST_LOG=skelesearch_core=debug skelesearch index .
RUST_LOG=skelesearch_core=debug skelesearch search "error handling"
```

CLI output includes elapsed time and embedding cache statistics. MCP
`get_index_status` reports indexing progress and freshness. All hot-path functions
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

## Performance

Profiled on skelegent (370 files, 2753 chunks):

| Provider | First index | Re-index (cached) | Search |
|----------|-------------|-------------------|--------|
| OpenAI `text-embedding-3-small` | 52.6 s | 0 s | 429 ms |
| fastembed (local CPU) | 118 s | 0.1 s | 27 ms |

OpenAI cost: ~$0.01 per full index of a medium codebase. The embedding cache
makes re-indexes essentially free regardless of provider.

## Benchmarks

The reproducible eval and benchmark framework lives under `benchmarks/`. It
supports a local corpus of cloned repos, config/profile matrices, version/binary
comparison, and normalized run artifacts for reporting. Start with
`benchmarks/README.md` and `benchmarks/manifests/repos.toml`.

## License

Dual-licensed under **MIT OR Apache-2.0**. See [LICENSE-MIT](LICENSE-MIT) and
[LICENSE-APACHE](LICENSE-APACHE).
