# Benchmarks

This directory contains the reproducible evaluation framework for skelesearch.

## Purpose

The benchmark system answers three questions:
1. Does skelesearch improve or regress across versions?
2. Which retrieval profile wins on which repository and language?
3. How does skelesearch compare to automatable competitor baselines?

## Tracked vs ignored

Tracked:
- `manifests/` — repo corpus definitions
- `configs/` — benchmark profile configs
- `cases/` — labeled eval sets per repo
- `scripts/` — runners, comparators, report generation
- `schemas/` — result artifact formats
- `reports/README.md` — reporting conventions

Ignored local state:
- `repos/` — cloned benchmark corpus
- `runs/` — raw benchmark outputs
- `tmp/` — temporary artifacts and copied binaries

## Benchmark matrix

Each run varies along three axes:
- **Repo** — benchmark target repository
- **Profile** — skelesearch config/toggle profile
- **Binary** — specific skelesearch executable or competitor adapter

This allows:
- current vs previous git SHA comparisons
- feature-toggle comparisons (`no-expansion`, `no-reranker`, etc.)
- per-language/per-repo analysis

## Starter corpus

Initial benchmark repos:
- Rust: `mini-redis`, `hyperfine`
- TypeScript: `hono`, `zod`
- Python: `httpx`
- Go: `cobra`

## Eval case format

Each eval file is a JSON array of cases. Each case has:
- `query` — natural language search query
- `expected_files` — ordered or unordered list of defensible target files
- `category` — rough bucket (`symbol lookup`, `implementation lookup`, etc.)
- `notes` — why those files are expected

Example:

```json
[
  {
    "query": "How does graceful shutdown propagate from the server listener to active connection handlers",
    "expected_files": ["src/server.rs", "src/shutdown.rs"],
    "category": "architecture conceptual lookup",
    "notes": "server::run creates the broadcast channel and shutdown.rs implements per-handler shutdown reception."
  }
]
```

Case quality rules:
- validate expected files against the actual cloned repo
- avoid trivia and purely lexical string-match cases
- prefer 1-2 expected files; use 3 only when necessary
- avoid hyphenated query tokens until the current FTS parser bug is fixed

## Workflow

1. Clone/update repos from `manifests/repos.toml`
2. Run a benchmark cell for a chosen binary/profile/repo/eval-set combination
3. Store normalized JSON output under `runs/`
4. Compare runs across profiles or versions
5. Generate a Markdown report summarizing deltas and regressions

## Scope

Version 1 is intentionally narrow:
- internal skelesearch comparisons first
- competitor adapters second
- starter corpus first, larger eval corpus later

See `docs/superpowers/plans/2026-03-19-benchmarks-and-eval-infrastructure.md` for the implementation plan.
