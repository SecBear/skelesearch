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
| 2026-03-18 | `6b53d61` | fastembed (jina-v2-base-code, 768d) | CoSQA | 0.442 | 0.742 | First run on AMD 7800X3D / NixOS |

**Note:** This section records skelesearch's measured CoSQA score for its current
embedding setup. Published leaderboard comparisons age quickly as providers ship new
models, so treat the absolute score as the durable signal unless the compared models
and evaluation setup are explicitly refreshed in the same commit.
### ContextBench — retrieval-only mode

Measures file/line retrieval quality on real GitHub issues. No published
retrieval-only baselines exist — skelesearch is the first tool measured here.

| Date | Commit | Provider | Instances | File R@5 | Acc@5 | MRR | Region overlap | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-03-23 | `987c1ec` | voyage-code-3 | 30 (django+astropy cached subset) | 77.8% | 63.3% | 0.794 | 23.6% | quick-bench cached-only sample; avg search 2354ms/query |
| 2026-03-24 | `c792da8` | voyage-code-3 | 30 (django+astropy cached subset) | 77.8% | 63.3% | 0.800 | 23.6% | overnight run (AMD 7800X3D / NixOS); avg search 2659ms/query |
**Important:** ContextBench `File R@5`, `Acc@5`, and `MRR` here are retrieval-only metrics.
They are **not comparable** to published ContextBench `Context F1` scores for full
agentic systems, which measure end-to-end generated output quality rather than retrieval.
Use this section only for skelesearch-vs-skelesearch comparisons unless another tool
publishes retrieval-only ContextBench numbers.
### Internal Eval (240 cases, 6 repos)

Custom benchmark: 40 cases per repo across mini-redis, hyperfine, hono, zod,
httpx, cobra. Categories: implementation, cross_file, symbol_lookup,
error_handling, architecture, utility_helper, config_init, vocabulary_gap.

| Date | Commit | Provider | Profile | R@5 | Acc@5 | MRR | Cases | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-03-24 | `c792da8` | voyage-code-3 | overnight (AMD 7800X3D / NixOS) | 84.7% | 73.8% | 0.842 | 240 | search quality fixes (barrel/doc penalties). Pre-scope-chain prefix experiment baseline. |
| 2026-03-23 | `9359eea` | voyage-code-3 | latest main | 84.5% | 73.8% | 0.837 | 240 | after chunk merge, schema migration, intent routing, MCP/server fixes |
| 2026-03-15 | `e931f51` | voyage-code-3 | voyage-full | 83.3% | — | 0.851 | 240 | pre-Acc@5 baseline |

**scope-chain prefix A/B test (fastembed, local, 4 repos):**

| Config | R@5 | MRR | Cases | Notes |
|---|---|---|---|---|
| `scope_prefix: false` (default) | 78.4% | 0.719 | 160 | hono/httpx/hyperfine/zod, fastembed |
| `scope_prefix: true` | 77.1% | 0.719 | 160 | Same repos. R@5 -1.3%, MRR unchanged |

> **Conclusion:** Scope prefixes are neutral-to-slightly-negative with jina-v2-base-code
> (code-specific model trained on raw code). Feature is gated behind `scope_prefix = true`
> in config, default off. Needs Voyage eval before enabling by default.

**PageRank ablation (fastembed, 240 cases, 6 repos):**

| Config | R@5 | MRR | Notes |
|---|---|---|---|
| pagerank_boost: true (old default) | 81.0% | 0.766 | Hub files promoted above relevant results |
| pagerank_boost: false (new default) | 81.0% | 0.799 | +3.4pp MRR. No change to R@5. |

**2026-03-22 per-repo breakdown — true local (no reranker) vs Voyage reranker:**

| Repo | Language | R@5 (local) | R@5 (Voyage) | Δ R@5 | MRR (local) | MRR (Voyage) | ms/40q (local) |
|---|---|---|---|---|---|---|---|
| mini-redis | Rust | 92.5% | 93.8% | +1.3 | 0.724 | 0.913 | 3,779 |
| cobra | Go | 90.4% | 90.4% | +0.0 | 0.908 | 0.938 | 3,615 |
| httpx | Python | 89.0% | 89.0% | +0.0 | 0.868 | 0.842 | 4,398 |
| hyperfine | Rust | 70.0% | 81.3% | +11.3 | 0.625 | 0.801 | 3,794 |
| hono | TypeScript | 69.2% | 86.3% | +17.1 | 0.528 | 0.803 | 6,799 |
| zod | TypeScript | 52.9% | (429 rate-limited) | — | 0.535 | — | 7,373 |

 > **Analysis:** Reranker adds +10.9pp R@5 on average. Impact is uneven: httpx/cobra
 > gain nothing, mini-redis gains 1.3pp, while hono (+17.1pp) and hyperfine (+11.3pp)
 > improve the most. TypeScript repos benefit strongly overall.
 >
>
> **Reranker findings (2026-03-22):**
> - MiniLM-L6-v2 (22M, CPU): **hurts results** (-3.8pp R@5 avg). Demotes code below docs.
> - gte-modernbert-base (149M, CPU): 3-5s/query. Unusable without GPU.
> - gte-modernbert-base (149M, CoreML M4 Pro): ~342ms warm, but 12s cold start per CLI invocation.
> - No open-weight code-aware CPU reranker exists in the ecosystem (reranker research).
> - Recommended: cloud reranker when key available, no reranker otherwise.
>
> **LSH dedup (fixed in `bf04d4e`):** Was broken since introduction (CozoDB `:rm` syntax).
> Now working. Impact is mixed: hono R@10 +2.5pp, zod R@5 -1.2pp. The 0.85 similarity
> threshold may be too aggressive for code — similar-looking validators in zod are
> semantically distinct. Average: -0.2pp R@5, +0.1pp R@10 (net neutral). Needs threshold tuning.
**2026-03-15 per-repo breakdown (voyage-full, 240 cases):**

| Repo | Language | R@5 | R@10 | MRR | Cases |
|---|---|---|---|---|---|
| mini-redis | Rust | 97.5% | 97.5% | 93.8% | 40 |
| cobra | Go | 92.9% | 92.9% | 96.3% | 40 |
| hono | TypeScript | 88.8% | 91.3% | 82.6% | 40 |
| hyperfine | Rust | 85.0% | 88.8% | 82.2% | 40 |
| httpx | Python | 85.8% | 85.8% | 65.2% | 40 |
| zod | TypeScript | 70.4% | 80.8% | 63.6% | 40 |


### SWE-bench Lite file localization (sample)

> Current script checks out per-instance commits and re-indexes, so runtime is dominated by indexing on large repos. Results below are from a 50-instance sample where 44 completed and 6 timed out at the 900s per-instance indexing limit.

| Date | Commit | Provider | Instances | Acc@1 | Acc@5 | R@5 | MRR | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-03-23 | `987c1ec` | voyage-code-3 | 44/50 completed | 61.4% | 79.5% | 79.5% | 0.688 | 6 skipped due per-instance indexing timeout on large astropy/django commits |
| 2026-03-24 | `c792da8` | voyage-code-3 | 36/50 completed | 66.7% | 91.7% | 91.7% | 0.746 | overnight run (AMD 7800X3D / NixOS); 14 skipped (indexing timeout) |

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

## Per-repo breakdown (2026-03-24 overnight, voyage-code-3, AMD 7800X3D / NixOS)

| Repo | Language | R@5 | Acc@5 | MRR | Cases |
|---|---|---|---|---|---|
| mini-redis | Rust | 97.5% | 95.0% | 0.923 | 40 |
| httpx | Python | 88.8% | 80.0% | 0.846 | 40 |
| hyperfine | Rust | 87.5% | 75.0% | 0.855 | 40 |
| cobra | Go | 82.1% | 70.0% | 0.880 | 40 |
| hono | TypeScript | 80.0% | 67.5% | 0.823 | 40 |
| zod | TypeScript | 72.5% | 55.0% | 0.726 | 40 |

> **Weakest:** zod (72.5% R@5) and hono (80.0% R@5) — both TypeScript.
> TypeScript repos remain the primary improvement target.

## Changelog

- **2026-03-24** — PageRank ablation (fastembed, 240 cases). R@5 unchanged (81.0%),
  MRR +3.4pp without PageRank (0.766→0.799). PageRank boost promotes hub files above
  relevant results. Disabled by default.
- **2026-03-24** — scope-chain prefix experiment A/B test (bdfb0ca). fastembed: R@5 78.4%→77.1%
  (-1.3%), MRR unchanged. Feature gated behind `scope_prefix = true`, default off.
  Code-specific models (jina-v2-base-code) don't benefit; needs Voyage eval.
- **2026-03-24** — Overnight run on AMD 7800X3D / NixOS (c792da8). Internal eval: R@5=84.7% (+1.4%),
  MRR=0.842. ContextBench: R@5=77.8%, MRR=0.800. SWE-bench: Acc@5=91.7% (36/50
  completed, 14 timed out during indexing — indexing timeout). Search quality fixes (barrel/doc
  penalties) improved R@5 slightly but MRR stayed flat.
- **2026-03-22** — True local-only run (d509642). R@5=77.3%, MRR=0.698 (240 cases,
  all 6 repos including zod). Previous "local" run was contaminated by Voyage reranker
  auto-detected from VOYAGE_API_KEY in environment. Adapter now strips cloud keys.
  Reranker adds +10.9pp R@5 on average; TypeScript repos benefit most.
- **2026-03-22** — Post-CozoDB optimization run (fastembed + Voyage reranker auto-detected).
  R@5=88.2%, MRR=0.859 (200 cases, zod rate-limited). Key changes since baseline:
  TF-IDF scoring fix (was raw TF), HNSW proximity graph expansion, recursive Datalog
  traversal, batch chunk fetching, parallel FTS+HNSW, LSH dedup.
- **2026-03-18** — First CoIR run (CoSQA, fastembed). nDCG@10 = 0.442.
- **2026-03-15** — Internal eval baseline (240 cases, voyage-full). R@5 = 86.7%.