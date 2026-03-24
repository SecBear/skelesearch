# Contributing to skelesearch

Thank you for your interest in contributing to skelesearch! This document covers the
process for contributing and the standards we maintain.

## Project overview

skelesearch is a local-first, graph-aware semantic code search tool with no
required hosted service by default. It ships as a Rust workspace with core,
embedding provider, reranker, telemetry, CLI, and MCP server crates.

## Getting started

### Prerequisites

- **Rust 1.92+** (edition 2021)
- **cmake** (required by fastembed-rs ONNX runtime)
- A working internet connection for downloading crate dependencies

### With Nix (recommended)

If you have [Nix](https://nixos.org/) installed:

```bash
nix develop
```

This provides the full development environment including Rust, cmake, clippy,
and rustfmt.

### Without Nix

Install Rust via [rustup](https://rustup.rs/) and ensure you have the stable
toolchain with clippy and rustfmt components:

```bash
rustup component add clippy rustfmt
```

### Fork and branch workflow

1. Fork the repository on GitHub.
2. Clone your fork locally:
   ```bash
   git clone https://github.com/<your-username>/skelesearch.git
   cd skelesearch
   ```
3. Create a feature branch from `main`:
   ```bash
   git checkout -b feat/my-feature main
   ```
4. Make your changes, following the conventions below.
5. Push your branch and open a Pull Request against `main`.

## Conventions

### Rust standards

- **Edition 2021**, resolver 2, minimum Rust 1.92
- **`#[async_trait]`** for async trait methods
- **`thiserror`** for error types
- **`schemars`** for JSON Schema derivation on MCP tool inputs
- No `unwrap()` in library code

### Workspace structure

| Crate | Purpose |
|---|---|
| `skelesearch-core` | Storage, indexing, search, chunking, manifest |
| `skelesearch-embed-fastembed` | Default embedding provider (jina-v2-base-code) |
| `skelesearch-embed-openai` | OpenAI embedding provider (text-embedding-3-small) |
| `skelesearch-embed-voyage` | Voyage AI embedding provider (voyage-code-3) |
| `skelesearch-cli` | CLI binary with clap subcommands |
| `skelesearch-mcp` | MCP server with rmcp 1.2 |
| `skelesearch-rerank-api` | Cloud cross-encoder reranker (API-based) |
| `skelesearch-rerank-local` | Local ONNX cross-encoder reranker (experimental, disabled by default) |
| `skelesearch-telemetry` | Shared tracing setup (fmt + optional OTLP) |
### Documentation

- Inline `///` doc comments on **every** public item.
- When adding or changing public API, update all documentation surfaces in the
  same commit.

## Commit messages

We use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).

Format:

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

Types: `feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `ci`, `perf`.

Scope is typically the crate name without the `skelesearch-` prefix (e.g., `core`,
`mcp`, `cli`, `embed`). Use no scope for workspace-wide changes.

Examples:

```
feat(core): add symbol search with CozoDB symbols relation
fix(mcp): handle empty index in grep_code tool
docs: update SKILL.md for CozoDB HNSW
```

## Running checks

Before submitting a PR, run full verification:

```bash
cargo test -p skelesearch-core -p skelesearch-mcp -p skelesearch-cli
cargo clippy --workspace -- -D warnings
```

## Pull request process

1. Ensure all CI checks pass.
2. Keep PRs focused -- one concern per PR.
3. Add or update tests for any behavioral changes.

## License

By contributing to skelesearch, you agree that your contributions will be dual
licensed under the [MIT License](./LICENSE-MIT) and the
[Apache License 2.0](./LICENSE-APACHE), at the user's option. This is the same
license used by the project itself.


## Changelog

We track changes in git commit history. A formal CHANGELOG will be added before the first tagged release.