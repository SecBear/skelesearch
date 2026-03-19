# skelesearch Benchmark Report

Generated: 2026-03-19T20:30:26.237Z


---

## Summary

| repo       | tool        | version     | profile      | language   | R@5     | R@10    | MRR     | cases | duration |
| ---------- | ----------- | ----------- | ------------ | ---------- | ------- | ------- | ------- | ----- | -------- |
| hono       | skelesearch | git:0eea7d5 | voyage-full  | typescript | 83.33%  | 91.67%  | 70.83%  | 6     | 8.9s     |
| mini-redis | skelesearch | git:0eea7d5 | voyage-full  | rust       | 100.00% | 100.00% | 100.00% | 6     | 9.5s     |
| mini-redis | skelesearch | git:0eea7d5 | no-expansion | rust       | 100.00% | 100.00% | 100.00% | 6     | 10.1s    |
| hono       | skelesearch | git:0eea7d5 | no-expansion | typescript | 66.67%  | 66.67%  | 38.89%  | 6     | 190.9s   |
| hyperfine  | skelesearch | git:0eea7d5 | no-expansion | rust       | 75.00%  | 83.33%  | 76.39%  | 6     | 18.2s    |

---

## Per-language averages

| language   | runs | avg R@5 | avg R@10 | avg MRR |
| ---------- | ---- | ------- | -------- | ------- |
| rust       | 3    | 91.67%  | 94.44%   | 92.13%  |
| typescript | 2    | 75.00%  | 79.17%   | 54.86%  |

---

## Best profile per repo

| repo       | best profile | MRR     | R@5     |
| ---------- | ------------ | ------- | ------- |
| hono       | voyage-full  | 70.83%  | 83.33%  |
| hyperfine  | no-expansion | 76.39%  | 75.00%  |
| mini-redis | voyage-full  | 100.00% | 100.00% |

---

## Notes

- **5** run artifacts loaded across **3** repos and **2** profiles.
- All runs use the same tool (`skelesearch`).
- Language is inferred from the `eval_set` path segment (`cases/<lang>/...`). Runs with no recognisable segment appear under `unknown`.
- Duration includes index + search time for the full eval set.

