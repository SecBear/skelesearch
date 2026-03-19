# Benchmark Adapters

Each adapter encapsulates the tool-specific benchmark pipeline behind a common
`BenchmarkAdapter` interface defined in `types.ts`. The orchestrator
(`run-eval.ts`) handles argument parsing, path resolution, output writing, and
summary display; adapters own everything tool-specific.

## Interface

```ts
interface BenchmarkAdapter {
  id: string;
  run(input: BenchmarkRunInput): Promise<BenchmarkRunResult>;
}
```

`BenchmarkRunResult.artifact` is the fully-assembled JSON object written to the
output file. Its shape must be compatible with `compare-runs.ts` and `report.ts`.

## Implemented

| Adapter | File | Status |
|---------|------|--------|
| skelesearch | `skelesearch.ts` | Active |

## Future candidates

The following tools are plausible future adapters. None are implemented yet.

- **gnosh** — different index/query pipeline; adapter would wrap its CLI similarly
- **probe** — code search tool; adapter would translate eval cases to probe queries
- **semantic-search-mcp** — MCP server wrapper; adapter may need async transport

Add a new adapter by:
1. Creating `adapters/<tool>.ts` that exports a `BenchmarkAdapter` instance
2. Importing it in `run-eval.ts` and selecting it (e.g. via `--tool` flag)
3. Ensuring the returned `artifact` includes at minimum the fields consumed by
   `compare-runs.ts`: `tool`, `repo_id`, `profile`, `aggregate`, `environment`
