# Benchmark Reports

This directory documents reporting conventions for benchmark output.

## Durable vs ephemeral artifacts

Ephemeral/local:
- raw run artifacts under `benchmarks/runs/`
- temporary comparison outputs
- copied binaries and local work products under `benchmarks/tmp/`

Durable/check-in candidates:
- named Markdown summaries when they capture a meaningful milestone
- baseline reports referenced from roadmap or release notes

## Report expectations

A useful report should include:
- benchmark date and binary/version
- corpus subset used
- profiles compared
- aggregate Recall@5, Recall@10, and MRR
- latency/indexing notes when available
- regressions and improvements by case
- known limitations affecting interpretation

## Naming

Recommended durable report names:
- `initial-baseline.md`
- `voyage-vs-fastembed.md`
- `graph-layer-regression-check.md`

Keep raw machine output in `benchmarks/runs/`, not here.
