# Pipeline Polish Benchmark — commit `4d33ea3`

Date: 2026-03-19
Changes: Post-reranker penalties, doc downranking, camelCase splitting,
file extension allowlist. Eval corpus expanded to 96 cases.

## Delta from previous baseline (`9f44e90`)

| Repo | Profile | R@5 delta | MRR delta | Notes |
|---|---|---|---|---|
| mini-redis | voyage-full | 0 | 0 | Perfect: 100%/100% both runs |
| mini-redis | no-expansion | 0 | 0 | Perfect |
| hyperfine | no-expansion | +7.8pp | **+17.4pp** | MRR 76.4% → 93.8% |
| hyperfine | voyage-full | +7.8pp | **+15.1pp** | MRR 75.6% → 90.6% |
| hono | no-expansion | 0 | **+42.4pp** | MRR 38.9% → 81.3% |
| hono | voyage-full | -1.0pp | **+23.0pp** | MRR 70.8% → 93.8% |
| zod | no-expansion | +13.5pp | **+40.3pp** | MRR 33.6% → 74.0% |
| zod | voyage-full | +30.2pp | **+46.7pp** | MRR 34.0% → 80.7% |
| httpx | no-expansion | -8.9pp | **+16.6pp** | MRR 46.4% → 63.0% |
| httpx | voyage-full | -8.9pp | **+17.0pp** | MRR 45.0% → 62.0% |
| cobra | no-expansion | +3.1pp | **+43.8pp** | MRR 50.0% → 93.8% |
| cobra | voyage-full | +3.1pp | **+43.8pp** | MRR 50.0% → 93.8% |

**Mean MRR across all 12 runs: 60.1% → 85.5% (+42%)**

## Full results (96 eval cases per profile)

| Repo | Profile | R@5 | R@10 | MRR | Cases |
|---|---|---|---|---|---|
| mini-redis | no-expansion | 100.0% | 100.0% | 100.0% | 16 |
| mini-redis | voyage-full | 100.0% | 100.0% | 100.0% | 16 |
| hyperfine | no-expansion | 82.8% | 84.4% | 93.8% | 16 |
| hyperfine | voyage-full | 82.8% | 84.4% | 90.6% | 16 |
| hono | no-expansion | 66.7% | 66.7% | 81.3% | 16 |
| hono | voyage-full | 82.3% | 87.5% | 93.8% | 16 |
| zod | no-expansion | 55.2% | 55.2% | 74.0% | 16 |
| zod | voyage-full | 71.9% | 77.1% | 80.7% | 16 |
| httpx | no-expansion | 82.8% | 84.4% | 63.0% | 16 |
| httpx | voyage-full | 82.8% | 84.4% | 62.0% | 16 |
| cobra | no-expansion | 78.1% | 78.1% | 93.8% | 16 |
| cobra | voyage-full | 78.1% | 78.1% | 93.8% | 16 |

## Per-language averages

| Language | Avg R@5 | Avg R@10 | Avg MRR |
|---|---|---|---|
| Rust | 91.4% | 92.2% | 96.1% |
| Go | 78.1% | 78.1% | 93.8% |
| TypeScript | 69.0% | 71.6% | 82.4% |
| Python | 82.8% | 84.4% | 62.5% |

## skelegent eval (15 cases)

| Metric | Previous | Now | Delta |
|---|---|---|---|
| R@5 | 0.700 | 0.700 | 0 |
| R@10 | 0.700 | 0.700 | 0 |
| MRR | 0.606 | **0.833** | **+37%** |

Notable improvements:
- Provider trait: MRR 0.50 → 1.00
- Dispatcher trait: MRR 1.00 → 1.00 (maintained)
- Multiple queries went from MRR 0.25-0.50 → 1.00

Still 2 zero-recall queries (pure vocabulary gap, unchanged).

## What drove the improvement

1. **Post-reranker penalties (+~30pp MRR)**: Moving test/doc penalties after
   reranker blending was the single largest fix. Previously, the reranker's
   75% weight at top positions completely overrode the 0.3x test penalty.

2. **Doc-file penalty (+~10pp MRR for zod/httpx)**: The 0.5x penalty for
   markdown/README/CHANGELOG files prevents NL-rich docs from outranking
   source code for "where is X implemented" queries.

3. **Extension filter (zod)**: Removing SVGs and non-code files from the index
   eliminated the catastrophic "refine/superRefine" failure where 3 SVG files
   were the only results returned.

4. **camelCase splitting**: Helps with queries like "superRefine" where the
   FTS tokenizer can't decompose compound identifiers. Contributes to overall
   BM25 matching quality.

## Remaining weaknesses

- **httpx MRR (62-63%)**: Still the weakest. Test files with dense keyword
  overlap still outrank source even with post-reranker penalty. May need
  stronger test penalty (0.2x?) or barrel-file (__init__.py) penalty.

- **zod R@5 (55-72%)**: Extension filter fixed the SVG issue, but many
  queries still miss because v3/v4 code coexists and v3's monolithic
  types.ts competes with v4 files.

- **2 zero-recall skelegent queries**: Pure vocabulary gap, needs graph
  augmentation (PageRank) or contextual chunk descriptions to bridge.
