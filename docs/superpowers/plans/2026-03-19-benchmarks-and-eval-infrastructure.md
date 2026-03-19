# Benchmarks and Eval Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible benchmark and evaluation system for skelesearch that can clone a curated multi-language repo corpus, run evals across config profiles and binaries/versions, and produce comparable reports for skelesearch and competitor baselines.

**Architecture:** Add a tracked `benchmarks/` framework plus ignored local corpus/run directories. Use a declarative repo manifest, declarative profile configs, and a small runner layer that shells out to specific binaries and records normalized JSON results. Keep the first version narrow: internal skelesearch comparison first, competitor adapters second.

**Tech Stack:** Rust workspace (existing CLI), Bun/TypeScript scripts for orchestration, TOML/JSON manifests, git, local filesystem.

---

## File Structure

```text
benchmarks/
  README.md                        # How the benchmark system works
  manifests/
    repos.toml                     # Curated repo corpus manifest
  configs/
    voyage-full.toml               # Full skelesearch profile
    no-expansion.toml              # Expansion disabled
    no-reranker.toml               # Reranker disabled
    no-graph.toml                  # Graph augmentation disabled
    no-symbol-enrichment.toml      # Symbol BM25 enrichment disabled (runner flag)
  cases/
    rust/
      mini-redis.json              # Eval cases per repo
      hyperfine.json
    typescript/
      hono.json
      zod.json
    python/
      httpx.json
    go/
      cobra.json
  scripts/
    clone-repos.ts                 # Clone/update corpus from repos.toml
    run-eval.ts                    # Run one benchmark matrix cell
    compare-runs.ts                # Compare two result files / snapshots
    report.ts                      # Generate Markdown summary report
  schemas/
    repo-manifest.schema.json      # Optional manifest schema
    eval-run.schema.json           # Result artifact schema
  reports/
    README.md                      # Report conventions; generated reports stay ignored or explicitly checked in
  repos/                           # IGNORED: cloned target repos
  runs/                            # IGNORED: raw benchmark outputs
  tmp/                             # IGNORED: temp work dirs, binary copies
```

Also modify:
- `.gitignore` — ignore `benchmarks/repos/`, `benchmarks/runs/`, `benchmarks/tmp/`
- `README.md` — add a short “Benchmarks” section pointing to `benchmarks/README.md`
- `crates/cli/src/cli.rs` and `crates/cli/src/app.rs` — only if needed for minimal profile toggles not currently exposed
- `crates/core/src/config.rs` — only if needed for profile-controlled toggles not currently configurable

---

## Chunk 1: Benchmark Skeleton and Manifest

### Task 1: Create benchmark directory skeleton

**Files:**
- Create: `benchmarks/README.md`
- Create: `benchmarks/manifests/repos.toml`
- Create: `benchmarks/reports/README.md`
- Modify: `.gitignore`
- Modify: `README.md`

- [ ] **Step 1: Add ignored local benchmark directories to `.gitignore`**

Add:
```gitignore
benchmarks/repos/
benchmarks/runs/
benchmarks/tmp/
```

- [ ] **Step 2: Create `benchmarks/README.md`**

Document:
- purpose of the system
- tracked vs ignored directories
- benchmark matrix dimensions: repo, profile, binary
- current repo corpus
- how to clone, run, compare, report
- versioning policy for snapshots

- [ ] **Step 3: Create `benchmarks/reports/README.md`**

Document whether generated reports are ephemeral or promoted into git as named snapshots.

- [ ] **Step 4: Create `benchmarks/manifests/repos.toml` with starter corpus**

Use a schema like:
```toml
[[repo]]
id = "mini-redis"
url = "https://github.com/tokio-rs/mini-redis.git"
rev = "main"
language = "rust"
license = "MIT"
tags = ["async", "server", "pubsub"]
recommended = true

[[repo]]
id = "hyperfine"
url = "https://github.com/sharkdp/hyperfine.git"
rev = "master"
language = "rust"
license = "MIT OR Apache-2.0"
tags = ["cli", "benchmarking", "process"]
recommended = true

[[repo]]
id = "hono"
url = "https://github.com/honojs/hono.git"
rev = "main"
language = "typescript"
license = "MIT"
tags = ["router", "middleware", "web"]
recommended = true

[[repo]]
id = "zod"
url = "https://github.com/colinhacks/zod.git"
rev = "main"
language = "typescript"
license = "MIT"
tags = ["types", "validation", "schema"]
recommended = true

[[repo]]
id = "httpx"
url = "https://github.com/encode/httpx.git"
rev = "master"
language = "python"
license = "BSD-3-Clause"
tags = ["http", "client", "async"]
recommended = true

[[repo]]
id = "cobra"
url = "https://github.com/spf13/cobra.git"
rev = "main"
language = "go"
license = "Apache-2.0"
tags = ["cli", "flags", "commands"]
recommended = true
```

- [ ] **Step 5: Add short benchmark pointer to `README.md`**

One short section only; do not duplicate full docs.

- [ ] **Step 6: Verify file layout only**

Run:
```bash
git diff --stat
```
Expected: only the new benchmark skeleton docs/manifest and `.gitignore` changes.

- [ ] **Step 7: Commit**

```bash
git add .gitignore README.md benchmarks/
git commit -m "chore: add benchmark and eval framework skeleton"
```

---

## Chunk 2: Clone/Update Corpus Manager

### Task 2: Implement repo clone/update script

**Files:**
- Create: `benchmarks/scripts/clone-repos.ts`
- Test manually against: `benchmarks/manifests/repos.toml`

- [ ] **Step 1: Parse `benchmarks/manifests/repos.toml`**

Implement a small Bun script that reads the manifest and produces strongly typed repo entries.

- [ ] **Step 2: Clone or update repos into `benchmarks/repos/<id>`**

Behavior:
- if repo missing: `git clone <url> benchmarks/repos/<id>`
- if repo exists: `git fetch --all --tags --prune`
- checkout specified `rev`
- if `rev` is a branch name, reset to `origin/<rev>`
- record final checked-out commit SHA in output

- [ ] **Step 3: Add safe output and summary**

Print per-repo:
- id
- url
- requested rev
- resolved SHA
- language
- path

- [ ] **Step 4: Add `--only <id1,id2>` support**

Useful for iteration and CI.

- [ ] **Step 5: Add `--dry-run` support**

Show intended actions without cloning.

- [ ] **Step 6: Manual verification**

Run:
```bash
bun benchmarks/scripts/clone-repos.ts --only mini-redis,hono --dry-run
bun benchmarks/scripts/clone-repos.ts --only mini-redis,hono
```
Expected:
- repos cloned to `benchmarks/repos/mini-redis` and `benchmarks/repos/hono`
- summary prints exact SHAs

- [ ] **Step 7: Commit**

```bash
git add benchmarks/scripts/clone-repos.ts benchmarks/manifests/repos.toml
git commit -m "feat: add benchmark corpus clone manager"
```

---

## Chunk 3: Config Profile Matrix

### Task 3: Define benchmark profile configs

**Files:**
- Create: `benchmarks/configs/voyage-full.toml`
- Create: `benchmarks/configs/no-expansion.toml`
- Create: `benchmarks/configs/no-reranker.toml`
- Create: `benchmarks/configs/no-graph.toml`
- Create: `benchmarks/configs/no-symbol-enrichment.toml`
- Modify: `crates/core/src/config.rs` only if needed
- Modify: `crates/cli/src/cli.rs` and `crates/cli/src/app.rs` only if needed

- [ ] **Step 1: Audit what is already configurable**

Verify current support for:
- expansion on/off
- reranker configured/not configured
- graph on/off
- provider selection
- max_tokens

- [ ] **Step 2: Decide the minimal new toggles needed**

Current likely gap:
- symbol enrichment is index-time, not query-time
- graph may be query-time only

Do **not** add a large settings surface. Add only what the benchmark runner needs.

Recommended additions if needed:
- `index.symbol_enrichment: bool` default `true`
- `search.graph.enabled: bool` default `false` or use existing CLI flag in benchmark runner

- [ ] **Step 3: Create profile TOMLs**

Examples:

`voyage-full.toml`
```toml
[index]
symbol_enrichment = true

[search]
top_k = 10
max_tokens = 8192

[search.graph]
enabled = true
max_depth = 1

[search.reranker]
provider = "voyage"

[search.expansion]
enabled = true
```

`no-expansion.toml`
```toml
[index]
symbol_enrichment = true

[search.graph]
enabled = true
max_depth = 1

[search.reranker]
provider = "voyage"

[search.expansion]
enabled = false
```

`no-reranker.toml`
```toml
[index]
symbol_enrichment = true

[search.graph]
enabled = true
max_depth = 1

[search.expansion]
enabled = true
```

`no-graph.toml`
```toml
[index]
symbol_enrichment = true

[search.expansion]
enabled = true

[search.reranker]
provider = "voyage"
```

`no-symbol-enrichment.toml`
```toml
[index]
symbol_enrichment = false

[search.graph]
enabled = false

[search.expansion]
enabled = true

[search.reranker]
provider = "voyage"
```

- [ ] **Step 4: Add only the minimal code required to honor these profiles**

If current config already supports a toggle, do not add another.

- [ ] **Step 5: Verify profile parsing**

Run:
```bash
cargo test -p skelesearch-core --features storage-sqlite
cargo check --workspace --features storage-sqlite
```

- [ ] **Step 6: Commit**

```bash
git add benchmarks/configs/ crates/core/src/config.rs crates/cli/src/cli.rs crates/cli/src/app.rs
git commit -m "feat: add benchmark profile configs and minimal toggle support"
```

---

## Chunk 4: Eval Runner and Result Artifact Format

### Task 4: Implement benchmark run artifact schema

**Files:**
- Create: `benchmarks/schemas/eval-run.schema.json`
- Create: `benchmarks/runs/.gitkeep` only if needed (prefer ignored directory without file)

- [ ] **Step 1: Define normalized result schema**

Use a schema like:
```json
{
  "tool": "skelesearch",
  "tool_version": "git:a666029",
  "binary": "./target/release/skelesearch",
  "repo_id": "mini-redis",
  "repo_path": "benchmarks/repos/mini-redis",
  "repo_sha": "...",
  "profile": "voyage-full",
  "eval_set": "benchmarks/cases/rust/mini-redis.json",
  "started_at": "2026-03-19T...Z",
  "duration_ms": 12345,
  "index": {
    "indexed_files": 123,
    "total_chunks": 456,
    "cache_hits": 0,
    "cache_misses": 456,
    "resolved_import_edges": 78
  },
  "aggregate": {
    "mean_recall_at_5": 0.75,
    "mean_recall_at_10": 0.81,
    "mean_mrr": 0.63,
    "total_cases": 18
  },
  "cases": [...],
  "environment": {
    "provider": "voyage",
    "reranker": "voyage",
    "expansion": true,
    "graph": true,
    "symbol_enrichment": true
  }
}
```

- [ ] **Step 2: Keep schema small and stable**

Do not encode raw logs or large traces into the run artifact.

### Task 5: Implement `run-eval.ts`

**Files:**
- Create: `benchmarks/scripts/run-eval.ts`

- [ ] **Step 1: Accept required inputs**

CLI flags:
- `--binary <path>`
- `--repo <id or path>`
- `--profile <name or path>`
- `--eval <path>`
- `--output <path>`
- optional `--tool skelesearch|gnosh|probe|...` default `skelesearch`

- [ ] **Step 2: For skelesearch runs, execute a full run**

Behavior:
1. ensure repo exists under `benchmarks/repos/`
2. write/copy the selected profile to repo-local `.skelesearch.toml`
3. clear repo-local `.skelesearch/` before fresh index unless `--reuse-index`
4. run index command
5. run eval command
6. capture duration and aggregate metrics
7. write normalized result JSON

- [ ] **Step 3: Preserve repo cleanliness**

Do not leave benchmark-only config files lying around in tracked repos unless they are in ignored benchmark clones. Since `benchmarks/repos/` is ignored, writing `.skelesearch.toml` there is acceptable.

- [ ] **Step 4: Manual verification**

Run on one repo:
```bash
bun benchmarks/scripts/run-eval.ts \
  --binary ./target/release/skelesearch \
  --repo mini-redis \
  --profile benchmarks/configs/voyage-full.toml \
  --eval benchmarks/cases/rust/mini-redis.json \
  --output benchmarks/runs/mini-redis-voyage-full-current.json
```

Expected:
- index runs
- eval runs
- output artifact matches schema

- [ ] **Step 5: Commit**

```bash
git add benchmarks/scripts/run-eval.ts benchmarks/schemas/eval-run.schema.json
git commit -m "feat: add benchmark runner and result artifact format"
```

---

## Chunk 5: Comparison and Reporting

### Task 6: Implement run comparison tool

**Files:**
- Create: `benchmarks/scripts/compare-runs.ts`

- [ ] **Step 1: Compare two run artifacts**

Inputs:
- `--base <run.json>`
- `--candidate <run.json>`

Output:
- aggregate delta table
- per-case delta summary
- regressions list
- improvements list

At minimum compare:
- R@5
- R@10
- MRR
- index duration
- search/eval duration if available

- [ ] **Step 2: Print machine-readable JSON and human-readable Markdown**

Support `--format json|md|text`.

### Task 7: Implement report generator

**Files:**
- Create: `benchmarks/scripts/report.ts`
- Create: `benchmarks/reports/latest.md` only as generated output example if desired

- [ ] **Step 1: Aggregate many run artifacts into one report**

Group by:
- repo
- profile
- binary/tool

Show:
- best profile per repo
- regressions vs baseline
- global average
- per-language average

- [ ] **Step 2: Manual verification**

Run against 2-3 run artifacts and verify the report is readable.

- [ ] **Step 3: Commit**

```bash
git add benchmarks/scripts/compare-runs.ts benchmarks/scripts/report.ts benchmarks/reports/README.md
git commit -m "feat: add benchmark comparison and reporting tools"
```

---

## Chunk 6: Seed Eval Cases

### Task 8: Add first corpus of eval cases

**Files:**
- Create: `benchmarks/cases/rust/mini-redis.json`
- Create: `benchmarks/cases/rust/hyperfine.json`
- Create: `benchmarks/cases/typescript/hono.json`
- Create: `benchmarks/cases/typescript/zod.json`
- Create: `benchmarks/cases/python/httpx.json`
- Create: `benchmarks/cases/go/cobra.json`

- [ ] **Step 1: Define eval case schema and document it in `benchmarks/README.md`**

Recommended fields:
```json
{
  "id": "mini-redis-001",
  "repo": "mini-redis",
  "language": "rust",
  "category": "architecture",
  "query": "how does publish subscribe work",
  "expected_files": ["src/cmd/subscribe.rs", "src/db.rs"],
  "notes": "Pubsub flow spans command parsing and DB subscription state"
}
```

- [ ] **Step 2: Add starter cases per repo**

Target 8-12 cases per repo in v1 of the benchmark harness. Categories per repo must include:
- symbol lookup
- implementation lookup
- architecture/conceptual lookup
- cross-file relationship

This yields ~50-70 cases initially.

- [ ] **Step 3: Validate cases manually by reading the repos**

Do not LLM-hallucinate expected files. Every expected file must be checked against the repo.

- [ ] **Step 4: Run one real benchmark end-to-end**

Clone one repo, run one profile, verify the system works.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/cases/ benchmarks/README.md
git commit -m "feat: add starter multi-language eval corpus"
```

---

## Chunk 7: Competitor Adapter Layer (Minimum Viable)

### Task 9: Define competitor adapter interface

**Files:**
- Create: `benchmarks/scripts/adapters/README.md`
- Create: `benchmarks/scripts/adapters/types.ts`
- Create: `benchmarks/scripts/adapters/skelesearch.ts`
- Optionally create: `benchmarks/scripts/adapters/gnosh.ts`

- [ ] **Step 1: Define a narrow adapter contract**

The adapter must answer:
- how to index a repo
- how to query a repo
- how to normalize output to `expected_files`-comparable format

Keep it tiny. Example:
```ts
interface BenchmarkAdapter {
  id: string
  prepare(repoPath: string, profilePath?: string): Promise<void>
  index(repoPath: string): Promise<IndexStats>
  eval(repoPath: string, evalPath: string): Promise<EvalResult>
}
```

- [ ] **Step 2: Implement skelesearch adapter**

Move skelesearch-specific runner logic behind the adapter.

- [ ] **Step 3: Add Gnosh adapter only if it is actually automatable**

Important: Gnosh is not a code-search peer; label it clearly as a general retrieval baseline.

Use Gnosh only if:
- local indexing/querying is scriptable
- result files can be normalized to file paths
- setup cost is acceptable

Otherwise document it and defer.

- [ ] **Step 4: Commit**

```bash
git add benchmarks/scripts/adapters/
git commit -m "feat: add benchmark adapter interface and skelesearch adapter"
```

---

## Chunk 8: First End-to-End Benchmark Snapshot

### Task 10: Produce initial snapshot and comparison workflow

**Files:**
- Create: `benchmarks/reports/initial-baseline.md` or similar if intentionally checked in
- Generate ignored files under `benchmarks/runs/`

- [ ] **Step 1: Run benchmark matrix on a minimal subset**

Run at least:
- 2 repos
- 3 profiles (`voyage-full`, `no-expansion`, `no-reranker`)
- current binary only

- [ ] **Step 2: Generate report**

Show:
- which profile wins on which repo
- latency tradeoffs
- whether expansion/reranker materially help per language

- [ ] **Step 3: Document current known limitations**

Specifically note:
- Rust cross-crate workspace imports are not yet fully resolved
- graph impact is currently under-realized on workspace-heavy repos
- competitor comparison is preliminary until adapters mature

- [ ] **Step 4: Commit only durable docs, not local corpora/runs**

```bash
git add benchmarks/README.md benchmarks/reports/*.md
git commit -m "docs: record initial benchmark workflow and baseline findings"
```

---

## Non-Goals for This Plan

Do **not** include in this plan:
- web dashboard for benchmarks
- automatic benchmark execution in CI on every PR
- large competitor fleet integration
- 100+ eval cases immediately
- cloud benchmark services
- benchmarking proprietary hosted tools

Those are follow-on plans.

---

## Follow-on Plans (separate)

1. **Expand eval corpus to 100+ cases**
   - after harness is proven
2. **Workspace Rust cross-crate resolution**
   - to make graph benchmarks meaningful on repos like skelegent
3. **Strong Signal Detection + rerank blending**
   - adopt Gnosh’s best query-pipeline ideas
4. **Competitor adapter expansion**
   - Probe, Gnosh, semantic-search-mcp, maybe grepai

---

## Acceptance Criteria for This Plan

This plan is complete when:
- `benchmarks/` exists with a documented, reproducible structure
- repos can be cloned from a manifest into ignored local corpus storage
- evals can be run against a chosen binary/profile/repo combination
- results are written as normalized JSON artifacts
- runs can be compared across profiles and versions
- at least 6 starter repos are defined in the manifest
- at least one end-to-end benchmark run has been completed successfully
- no benchmark corpus clones or generated run artifacts are committed to git
