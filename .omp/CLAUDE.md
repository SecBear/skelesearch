@DECISIONS.md

# skelesearch

Semantic code-search MCP server and CLI. Indexes codebases with tree-sitter AST-aware
chunking, stores in embedded CozoDB (HNSW + FTS + import graph), exposes hybrid BM25+dense
retrieval via MCP and CLI.

## Before making architectural changes

**Read DECISIONS.md first.** Every major dependency choice and design pattern has a recorded
decision with rationale. Before switching a dependency, changing the storage backend, or
adding a new retrieval strategy, check whether an ADR already covers that decision.

Notably:
- CozoDB is used intentionally despite stalled development (see ADR-002). The mitigation
  is the `StorageBackend` trait in `crates/core/src/schema.rs`.
- SPLADE is intentionally absent from v1 (see ADR-003). Research is in `docs/future-improvements.md`.
- Embedding dimensions are runtime-configurable — do not hardcode them anywhere.

## Codebase structure

```
crates/
  core/               Library — schema, indexer, searcher, chunker, manifest, provider trait
  embed-fastembed/    Optional library — fastembed-rs provider (jina-v2-base-code default)
  mcp/                Binary — rmcp 0.16 MCP server (stdio transport)
  cli/                Binary — clap CLI
```

## Design spec

Full design: `.omp/superpowers/specs/2026-03-17-skelesearch-design.md`

Future improvements and deferred research: `docs/future-improvements.md`

## Build notes

- Use `storage-sqlite` feature during development (fast compile). Switch to `storage-rocksdb` for release.
- RocksDB requires cmake and a C++20 compiler. On macOS: `CXXFLAGS="-std=c++20"` if needed.
- Cold RocksDB build: ~10 minutes. Subsequent builds are incremental.
