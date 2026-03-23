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


## Quick eval (development feedback loop)

For rapid iteration, use the quick-eval script instead of running the full suite:

```bash
# First time: clone benchmark repos
bun benchmarks/scripts/clone-repos.ts

# Quick eval: 3 diagnostic repos (~3-5 min, no API keys)
./benchmarks/scripts/quick-eval.sh

# Quick eval with voyage embeddings (~5-10 min, needs API keys)
./benchmarks/scripts/quick-eval.sh --provider voyage

# Tag a run for comparison
./benchmarks/scripts/quick-eval.sh --tag post-depth-decay

# Single repo deep dive
./benchmarks/scripts/quick-eval.sh --repo zod

# Full 6-repo suite (~10-15 min)
./benchmarks/scripts/quick-eval.sh --full

# Full suite with voyage
./benchmarks/scripts/quick-eval.sh --full --provider voyage
```

**Diagnostic subset:** zod (63.6% MRR, weakest), httpx (65.2% MRR, second
weakest), cobra (96.3% MRR, regression canary). If zod/httpx improve without
cobra regressing, the change is good.

**Machine requirements:**
- macOS M4 Pro 24GB: fastembed runs well, ~3-5 min for quick eval
- NixOS bearbrick (7800X3D): same or faster, run `nix develop` first
- No GPU required — fastembed uses ONNX on CPU

## External benchmarks

### CoIR (Code Information Retrieval)
Standard code retrieval benchmark, on MTEB leaderboard. 10 sub-tasks.

```bash
# Single task smoke test (~20 min)
uv run --with coir-eval --with fastembed --with numpy \
  python3 benchmarks/scripts/coir-eval.py \
  --mode embed-only --provider fastembed --tasks cosqa \
  --output benchmarks/runs/coir-fastembed-cosqa.json

# Full 10-task suite (~2-3 hours)
uv run --with coir-eval --with fastembed --with numpy \
  python3 benchmarks/scripts/coir-eval.py \
  --mode embed-only --provider fastembed --tasks all \
  --output benchmarks/runs/coir-fastembed-full.json
```

### ContextBench
Real GitHub issue retrieval. File/Block/Line F1. No retrieval-only baselines published.

```bash
# Pilot (~25 min, 9 instances)
uv run --with datasets --with huggingface_hub \
  python3 benchmarks/scripts/contextbench-eval.py \
  --binary ./target/release/skelesearch \
  --languages python,go,rust --limit-per-lang 3 \
  --cache-dir benchmarks/contextbench-repos \
  --output benchmarks/runs/contextbench-pilot.json --provider fastembed

# Full verified (500 instances, ~2-4 hours with caching)
# Same command without --limit-per-lang
```