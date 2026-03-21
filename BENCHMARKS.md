# Benchmark Results

Curated record of benchmark scores across skelesearch versions. Each entry is
committed alongside the code that produced it, so `git log --follow BENCHMARKS.md`
shows the evolution and `git diff` between entries shows what changed.

Raw per-instance results live in `benchmarks/runs/` (gitignored). This file
captures the aggregate numbers that matter.

---

## Latest Results

### CoIR (Code Information Retrieval) — embed-only mode

Measures pure embedding quality on standard code retrieval tasks. Directly
comparable to published Voyage/OpenAI/Jina scores on the MTEB leaderboard.

| Date | Commit | Provider | Task | nDCG@10 | R@10 | Notes |
|---|---|---|---|---|---|---|
| 2026-03-18 | `6b53d61` | fastembed (jina-v2-base-code, 768d) | CoSQA | 0.442 | 0.742 | First run on bearbrick (NixOS, 7800X3D) |

**Reference scores (published, same CoSQA task):**

| Model | nDCG@10 | Source |
|---|---|---|
| SFR-Embedding-Code-2B (Salesforce) | 0.363 | CoIR paper (ACL 2025) |
| Voyage-Code-002 | ~0.298 | MTEB leaderboard |
| OpenAI Ada-002 | ~0.289 | MTEB leaderboard |
| BGE-Base-en-v1.5 | ~0.253 | MTEB leaderboard |

### ContextBench — retrieval-only mode

Measures file/line retrieval quality on real GitHub issues. No published
retrieval-only baselines exist — skelesearch is the first tool measured here.

| Date | Commit | Provider | Instances | File R@5 | File F1 | Line F1 | Notes |
|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | Pilot run pending |

**Reference scores (published, full agentic systems):**

| Agent | Context F1 | Efficiency | Source |
|---|---|---|---|
| Claude Sonnet 4.5 | 0.344 | 0.658 | ContextBench leaderboard |
| Devstral 2 | 0.332 | 0.616 | ContextBench leaderboard |
| GPT-5 | 0.312 | 0.591 | ContextBench leaderboard |

### Internal Eval (240 cases, 6 repos)

Custom benchmark: 40 cases per repo across mini-redis, hyperfine, hono, zod,
httpx, cobra. Categories: implementation, cross_file, symbol_lookup,
error_handling, architecture, utility_helper, config_init, vocabulary_gap.

| Date | Commit | Provider | Profile | R@5 | R@10 | MRR | P@5 | Cases |
|---|---|---|---|---|---|---|---|---|
| 2026-03-15 | `e931f51` | voyage-code-3 | voyage-full | 86.7% | 89.5% | 80.6% | — | 240 |

**Per-repo breakdown (voyage-full, 240 cases):**

| Repo | Language | R@5 | R@10 | MRR | Cases |
|---|---|---|---|---|---|
| mini-redis | Rust | 97.5% | 97.5% | 93.8% | 40 |
| cobra | Go | 92.9% | 92.9% | 96.3% | 40 |
| hono | TypeScript | 88.8% | 91.3% | 82.6% | 40 |
| hyperfine | Rust | 85.0% | 88.8% | 82.2% | 40 |
| httpx | Python | 85.8% | 85.8% | 65.2% | 40 |
| zod | TypeScript | 70.4% | 80.8% | 63.6% | 40 |

---

## How to reproduce

```bash
# CoIR (embed-only, no API keys needed)
uv run --with coir-eval --with fastembed --with numpy \
  python3 benchmarks/scripts/coir-eval.py \
  --mode embed-only --provider fastembed \
  --tasks cosqa \
  --output benchmarks/runs/coir-fastembed-cosqa.json

# ContextBench (retrieval-only, no API keys with fastembed)
uv run --with datasets --with huggingface_hub \
  python3 benchmarks/scripts/contextbench-eval.py \
  --binary ./target/release/skelesearch \
  --languages python,typescript,go,rust,javascript \
  --cache-dir benchmarks/contextbench-repos \
  --output benchmarks/runs/contextbench-verified.json \
  --provider fastembed

# Internal eval (requires API keys for voyage provider)
bun benchmarks/scripts/run-eval.ts \
  --binary ./target/release/skelesearch \
  --repo mini-redis \
  --profile benchmarks/configs/voyage-full.toml \
  --eval benchmarks/cases/rust/mini-redis.json \
  --output benchmarks/runs/mini-redis-voyage-full.json \
  --provider voyage
```

## Changelog

- **2026-03-18** — First CoIR run (CoSQA, fastembed). nDCG@10 = 0.442.
- **2026-03-15** — Internal eval baseline (240 cases, voyage-full). R@5 = 86.7%.
