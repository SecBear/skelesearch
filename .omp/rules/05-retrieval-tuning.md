---
description: "Fast feedback loop for tuning retrieval parameters against the 6-repo benchmark suite"
alwaysApply: false
globs: ["crates/core/src/searcher.rs", "crates/core/src/schema.rs", "crates/core/src/config.rs", "benchmarks/**"]
---
# Retrieval Tuning Protocol

## Key Parameters (all in SearchConfig → Datalog query in schema.rs)

| Parameter | Config key | Current | What it does |
|---|---|---|---|
| FTS fusion weight | `fts_weight` | 0.55 | BM25 contribution to score-based fusion |
| HNSW fusion weight | `vec_weight` | 0.45 | Vector similarity contribution (= 1 - fts_weight) |
| Graph score multiplier | `graph_score_factor` | 0.3 | Graph-expanded chunk score = parent × this |
| Graph relevance threshold | `graph_min_score` | 0.005 | Minimum parent score to trigger graph walk |
| Graph cosine sim threshold | `graph_sim_threshold` | 0.25 | Minimum query-chunk similarity for graph results |
| Graph max results | `graph_max_results` | 20 | Cap on graph-added chunks |
| PageRank coefficient | `pagerank_factor` | 0.1 | Linear boost: 1.0 + factor × pr |
| Score normalization | — | bm25/(bm25+1) | BM25 → [0,1) sigmoid. HNSW: 1-dist |

## Fast Feedback Loop

### Prerequisites
- Voyage API key set (`VOYAGE_API_KEY`) — 200M free tokens, ~59 full runs
- Release binary built: `cargo build --release --features storage-sqlite`
- All 6 repos cloned: `bun benchmarks/scripts/clone-repos.ts`

### One-parameter sweep
```bash
# 1. Change parameter in crates/core/src/schema.rs or config.rs
# 2. Rebuild (~15s incremental)
cargo build --release --features storage-sqlite

# 3. Run full eval with Voyage embedding (fast, ~5-8 min all 6 repos)
./benchmarks/scripts/quick-eval.sh --full --provider voyage --tag <param-value> --profile unified

# 4. Compare against baseline
cat benchmarks/runs/*-<param-value>-*.json | python3 -c "
import json, sys, glob
for f in sorted(glob.glob('benchmarks/runs/*-<param-value>-*.json')):
    d = json.load(open(f))
    print(f'{d[\"repo_id\"]:>12}: R@5={d[\"aggregate\"][\"mean_recall_at_5\"]:.1%} MRR={d[\"aggregate\"][\"mean_mrr\"]:.3f}')
"
```

### Automated sweep (when parameters are in TOML)
```bash
# Generate configs, run all, compare
benchmarks/scripts/param-sweep.sh --param fts_weight --values "0.4,0.5,0.55,0.6,0.7" --provider voyage
```

### Baseline numbers (2026-03-22, fastembed, no reranker)

| Repo | R@5 | MRR | Notes |
|---|---|---|---|
| mini-redis | 92.5% | 0.724 | Small Rust, 194 chunks |
| hyperfine | 70.0% | 0.625 | Medium Rust, 293 chunks |
| hono | 69.2% | 0.528 | Large TS, 3635 chunks |
| zod | 52.9% | 0.535 | Large TS, 4028 chunks |
| httpx | 89.0% | 0.868 | Medium Python, ~500 chunks |
| cobra | 90.4% | 0.908 | Medium Go, ~400 chunks |

### Unified query baseline (fastembed, no reranker)

| Repo | R@5 | MRR |
|---|---|---|
| mini-redis | 91.3% | 0.820 |
| cobra | 90.4% | 0.894 |

## Rules

1. **Never tune on one repo.** Always run all 6. Small repos (mini-redis) and large repos (zod, hono) behave differently.
2. **Track every run.** Tag with parameter value. Results go to `benchmarks/runs/`.
3. **MRR matters more than R@5 for agents.** Agents read top-1 first. A +0.05 MRR gain is worth a -1pp R@5 cost.
4. **Document what you tried.** Add results to BENCHMARKS.md even for negative results.
5. **Parameters should live in TOML, not code.** If you're recompiling to change a parameter, extract it to config first.
