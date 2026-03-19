# Full Benchmark Baseline — commit `9f44e90`

Date: 2026-03-19
Pipeline: FTS sanitization fix + all previous pipeline polish (test-file downranking,
strong signal detection, position-aware rerank blending, workspace cross-crate resolution).

## Configuration

- **Embedding**: Voyage code-3 (1024-dim) for all runs
- **no-expansion**: No LLM query expansion, no reranker, no graph
- **voyage-full**: LLM expansion (OpenAI) + Voyage reranker + import graph (depth 1) + symbol enrichment

## Results: 6 repos x 2 profiles (36 eval cases)

| Repo | Profile | R@5 | R@10 | MRR | Duration |
|---|---|---|---|---|---|
| mini-redis | no-expansion | 100.0% | 100.0% | 100.0% | 2.0s |
| mini-redis | voyage-full | 100.0% | 100.0% | 100.0% | 6.3s |
| hyperfine | no-expansion | 75.0% | 83.3% | 76.4% | 2.3s |
| hyperfine | voyage-full | 75.0% | 83.3% | 75.6% | 10.4s |
| hono | no-expansion | 66.7% | 66.7% | 38.9% | 2.9s |
| hono | voyage-full | **83.3%** | **91.7%** | **70.8%** | 9.1s |
| zod | no-expansion | 41.7% | 66.7% | 33.6% | 189.7s |
| zod | voyage-full | 41.7% | **75.0%** | 34.0% | 12.0s |
| httpx | no-expansion | 91.7% | 100.0% | 46.4% | 28.2s |
| httpx | voyage-full | 91.7% | 100.0% | 45.0% | 8.6s |
| cobra | no-expansion | 75.0% | 75.0% | 50.0% | 20.7s |
| cobra | voyage-full | 75.0% | 75.0% | 50.0% | 8.7s |

## Per-language averages

| Language | Runs | Avg R@5 | Avg R@10 | Avg MRR |
|---|---|---|---|---|
| Rust | 4 | 87.5% | 91.7% | 88.0% |
| Python | 2 | 91.7% | 100.0% | 45.7% |
| Go | 2 | 75.0% | 75.0% | 50.0% |
| TypeScript | 4 | 58.3% | 75.0% | 44.4% |

## Aggregate (all 12 runs)

| Metric | Value |
|---|---|
| Mean R@5 | 76.4% |
| Mean R@10 | 84.7% |
| Mean MRR | 60.1% |

## voyage-full vs no-expansion delta

| Repo | R@5 delta | R@10 delta | MRR delta |
|---|---|---|---|
| mini-redis | 0 | 0 | 0 |
| hyperfine | 0 | 0 | -0.8pp |
| hono | **+16.6pp** | **+25.0pp** | **+31.9pp** |
| zod | 0 | **+8.3pp** | +0.4pp |
| httpx | 0 | 0 | -1.4pp |
| cobra | 0 | 0 | 0 |

## Key findings

1. **voyage-full helps most on TypeScript** — hono shows the largest improvement (+16.6pp R@5,
   +31.9pp MRR). The LLM expander bridges vocabulary gaps between framework-specific queries
   and source code terms.

2. **Rust performs best** — mini-redis is a perfect score, hyperfine is strong. Rust code has
   clear naming conventions that align well with semantic search.

3. **Zod is the hardest repo** — 41.7% R@5 even with the full pipeline. Zod's codebase is
   heavily abstracted with generic names (`parse`, `transform`, `refine`) that don't align
   with natural language queries about specific features.

4. **Python (httpx) has high recall but low MRR** — we find the right files but not at the
   top of the result list. Ranking quality needs improvement for Python codebases.

5. **Go (cobra) shows no delta between profiles** — cobra's codebase uses straightforward
   naming that works well with basic BM25, so expansion/reranking don't add value.

6. **Duration: voyage-full is faster than no-expansion for large repos** — because no-expansion
   must re-embed queries through fastembed locally, while voyage-full hits the API. For small
   repos (mini-redis) the API overhead is visible.

## skelegent eval (15 cases, separate from benchmark corpus)

| Metric | Value |
|---|---|
| R@5 | 0.700 |
| R@10 | 0.700 |
| MRR | 0.606 |

- **Unchanged from previous run** — same 2 zero-recall queries remain:
  - "how do agents remember things across conversations" (expects `state.rs`, `effect.rs`)
  - "how does one agent hand off work to another agent" (expects `dispatch.rs`, `effect.rs`)
- These are pure vocabulary gap: no code token matches "remember" or "hand off".
  LLM expansion was active but did not bridge the gap.

## Improvement opportunities (ordered by expected impact)

1. **tree-sitter tags.scm** — richer def/ref extraction would give the graph layer more
   edges to traverse. Currently limited to import resolution only.
2. **PageRank on file graph** — structural centrality scoring would help for queries about
   core abstractions (dispatch, effect) that are heavily imported.
3. **Contextual chunk descriptions** — LLM-generated per-chunk summaries (Anthropic pattern)
   would provide natural language anchors for conceptual queries.
4. **Expand eval to 100+ cases** — 36 benchmark + 15 skelegent = 51 total. Need 2x more
   to get statistically significant deltas.
5. **Fix zod/httpx MRR** — investigate why correct files rank low despite being found.
