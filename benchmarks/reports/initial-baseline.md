# skelesearch Benchmark Report

Generated: 2026-03-19T18:46:32.215Z


---

## Summary

| repo       | tool        | version     | profile      | language   | R@5     | R@10    | MRR     | cases | duration |
| ---------- | ----------- | ----------- | ------------ | ---------- | ------- | ------- | ------- | ----- | -------- |
| mini-redis | skelesearch | git:3429987 | no-expansion | rust       | 100.00% | 100.00% | 100.00% | 6     | 8.4s     |
| mini-redis | skelesearch | git:3429987 | no-reranker  | rust       | 100.00% | 100.00% | 100.00% | 6     | 2.5s     |
| mini-redis | skelesearch | git:3429987 | voyage-full  | rust       | 100.00% | 100.00% | 100.00% | 6     | 10.3s    |
| hono       | skelesearch | git:3429987 | no-expansion | typescript | 66.67%  | 66.67%  | 38.89%  | 6     | 181.7s   |
| hono       | skelesearch | git:3429987 | no-reranker  | typescript | 66.67%  | 66.67%  | 38.89%  | 6     | 2.2s     |
| hono       | skelesearch | git:3429987 | voyage-full  | typescript | 66.67%  | 66.67%  | 38.89%  | 6     | 9.2s     |

---

## Per-language averages

| language   | runs | avg R@5 | avg R@10 | avg MRR |
| ---------- | ---- | ------- | -------- | ------- |
| rust       | 3    | 100.00% | 100.00%  | 100.00% |
| typescript | 3    | 66.67%  | 66.67%   | 38.89%  |

---

## Best profile per repo

| repo       | best profile | MRR     | R@5     |
| ---------- | ------------ | ------- | ------- |
| hono       | no-expansion | 38.89%  | 66.67%  |
| mini-redis | no-expansion | 100.00% | 100.00% |

---

## Notes

- **6** run artifacts loaded across **2** repos and **3** profiles.
- All runs use the same tool (`skelesearch`).
- Language is inferred from the `eval_set` path segment (`cases/<lang>/...`). Runs with no recognisable segment appear under `unknown`.
- Duration includes index + search time for the full eval set.


## Interpretation

- `mini-redis` is currently an easy corpus for skelesearch: all three tested profiles hit 100% across the 6 starter cases.
- `hono` is materially harder: all three tested profiles tie at R@5=66.67 / MRR=38.89 on the 6 starter cases.
- The current benchmark runner measures full cell time (index + eval), so reuse-index runs are much faster than fresh-index runs.
- TypeScript retrieval is visibly polluted by nearby test files in some cases (`compose.test.ts`, `context.test.ts`, etc.). This suggests we should add future file-category signals or test-file downranking rather than only tuning embeddings.
- Graph impact is not yet a meaningful differentiator on these starter runs; workspace- and import-resolution limitations still cap the structural signal.
- Competitor comparisons are not yet included; this baseline is for internal skelesearch profile comparison only.
