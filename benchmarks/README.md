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

See `.omp/superpowers/plans/2026-03-19-benchmarks-and-eval-infrastructure.md` for the implementation plan.


## Quick eval (development feedback loop)

Current scripts:

```bash
# Re-index the six hand-written eval repos (Voyage provider)
source .env && export VOYAGE_API_KEY
./benchmarks/scripts/reindex.sh

# Run the 240-case suite and print R@5 / Acc@5 / MRR
./benchmarks/scripts/eval.py

# Overnight / brick run: build, tests, re-index, eval, ContextBench quick-bench, SWE-bench sample
nohup ./benchmarks/scripts/overnight.sh > overnight.log 2>&1 &
tail -f overnight.log
```

**Latest 240-case baseline (main, 2026-03-23):** R@5 84.5%, Acc@5 73.8%, MRR 0.837.

**Latest ContextBench quick-bench (30 cached Python instances):** R@5 77.8%, Acc@5 63.3%, MRR 0.794, region overlap 23.6%.

**Latest SWE-bench sample (44/50 completed):** Acc@1 61.4%, Acc@5 79.5%, MRR 0.688. Six instances timed out on per-instance indexing in large astropy/django commits.

**Machine requirements:**
- macOS M4 Pro 24GB: voyage indexing works, but large TS repos (zod) take several minutes
- NixOS bearbrick (7800X3D): preferred for overnight runs
- No GPU required for the current Voyage/FastEmbed paths

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