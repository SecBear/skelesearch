#!/usr/bin/env bun
/**
 * report.ts — aggregate multiple eval-run artifacts into a Markdown report.
 *
 * Usage:
 *   bun benchmarks/scripts/report.ts --inputs run1.json,run2.json [--output report.md]
 *   bun benchmarks/scripts/report.ts --dir benchmarks/runs [--output report.md]
 */

import { readdirSync, readFileSync } from "fs";
import { basename, extname, join, resolve } from "path";

// ---------------------------------------------------------------------------
// Types mirroring eval-run.schema.json
// ---------------------------------------------------------------------------

interface IndexStats {
  indexed_files: number;
  total_chunks: number;
  cache_hits: number | null;
  cache_misses: number | null;
  resolved_import_edges: number | null;
}

interface AggregateMetrics {
  mean_recall_at_5: number;
  mean_recall_at_10: number;
  mean_mrr: number;
  total_cases: number;
}

interface Environment {
  provider: string;
  reranker: string | null;
  expansion: boolean | null;
  graph: boolean;
  symbol_enrichment: boolean;
}

interface EvalRun {
  tool: string;
  tool_version: string;
  binary: string;
  repo_id: string;
  repo_path: string;
  repo_sha: string | null;
  profile: string;
  eval_set: string;
  started_at: string;
  duration_ms: number;
  index: IndexStats;
  aggregate: AggregateMetrics;
  cases: unknown[];
  environment: Environment;
}

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

interface Args {
  inputs: string[] | null; // explicit file list
  dir: string | null;      // scan a directory
  output: string | null;   // destination file; null → stdout
}

function parseArgs(argv: string[]): Args {
  const args = argv.slice(2);
  const result: Args = { inputs: null, dir: null, output: null };

  for (let i = 0; i < args.length; i++) {
    const flag = args[i];
    switch (flag) {
      case "--inputs":
        result.inputs = (args[++i] ?? "").split(",").map((p) => p.trim()).filter(Boolean);
        break;
      case "--dir":
        result.dir = args[++i];
        break;
      case "--output":
        result.output = args[++i];
        break;
      default:
        process.stderr.write(`Unknown flag: ${flag}\n`);
        process.exit(1);
    }
  }

  if (!result.inputs && !result.dir) {
    process.stderr.write(
      "ERROR: Provide either --inputs <paths> or --dir <directory>\n\n" +
        "Usage:\n" +
        "  bun report.ts --inputs run1.json,run2.json [--output report.md]\n" +
        "  bun report.ts --dir benchmarks/runs [--output report.md]\n"
    );
    process.exit(1);
  }

  return result;
}

// ---------------------------------------------------------------------------
// I/O helpers
// ---------------------------------------------------------------------------

function loadRun(path: string): EvalRun {
  let text: string;
  try {
    text = readFileSync(path, "utf8");
  } catch (e) {
    process.stderr.write(`ERROR: Cannot read file: ${path}\n  ${e}\n`);
    process.exit(1);
  }

  let data: unknown;
  try {
    data = JSON.parse(text);
  } catch (e) {
    process.stderr.write(`ERROR: Invalid JSON in: ${path}\n  ${e}\n`);
    process.exit(1);
  }

  // Minimal structural validation — enough to surface malformed artifacts early.
  const required = [
    "tool", "tool_version", "repo_id", "profile", "eval_set",
    "started_at", "duration_ms", "aggregate",
  ] as const;
  for (const key of required) {
    if ((data as Record<string, unknown>)[key] === undefined) {
      process.stderr.write(
        `ERROR: Missing required field "${key}" in artifact: ${path}\n`
      );
      process.exit(1);
    }
  }

  return data as EvalRun;
}

function collectPaths(args: Args): string[] {
  if (args.inputs) return args.inputs.map((p) => resolve(p));

  const dir = resolve(args.dir!);
  let entries: string[];
  try {
    entries = readdirSync(dir);
  } catch (e) {
    process.stderr.write(`ERROR: Cannot read directory: ${dir}\n  ${e}\n`);
    process.exit(1);
  }

  return entries
    .filter((f) => extname(f) === ".json")
    .map((f) => join(dir, f));
}

// ---------------------------------------------------------------------------
// Inference helpers
// ---------------------------------------------------------------------------

/**
 * Infer programming language from the eval_set path.
 * Expects a segment like `cases/<lang>/...` anywhere in the path.
 */
function inferLanguage(evalSet: string): string {
  // Normalise separators so the regex works on Windows paths too.
  const normalised = evalSet.replace(/\\/g, "/");
  const m = normalised.match(/\/cases\/([^/]+)\//);
  return m ? m[1] : "unknown";
}

/** Extract a short, human-readable profile name from a profile file path. */
function profileName(profile: string): string {
  return basename(profile, extname(profile));
}

// ---------------------------------------------------------------------------
// Markdown formatting
// ---------------------------------------------------------------------------

/** Left-pad a number to a fixed decimal width for table alignment. */
function pct(n: number, decimals = 3): string {
  return (n * 100).toFixed(decimals - 1) + "%";
}

function msToSec(ms: number): string {
  return (ms / 1000).toFixed(1) + "s";
}

/** Build a Markdown table from headers and rows of strings. */
function markdownTable(headers: string[], rows: string[][]): string {
  const widths = headers.map((h, i) =>
    Math.max(h.length, ...rows.map((r) => (r[i] ?? "").length))
  );
  const pad = (s: string, w: number) => s.padEnd(w);
  const sep = widths.map((w) => "-".repeat(w));

  const lines = [
    `| ${headers.map((h, i) => pad(h, widths[i])).join(" | ")} |`,
    `| ${sep.join(" | ")} |`,
    ...rows.map((r) => `| ${r.map((c, i) => pad(c ?? "", widths[i])).join(" | ")} |`),
  ];
  return lines.join("\n");
}

// ---------------------------------------------------------------------------
// Report generation
// ---------------------------------------------------------------------------

interface RunRow {
  run: EvalRun;
  language: string;
  profileShort: string;
}

function buildReport(rows: RunRow[]): string {
  const generatedAt = new Date().toISOString();
  const sections: string[] = [];

  // ── Title ──────────────────────────────────────────────────────────────────
  sections.push(`# skelesearch Benchmark Report\n\nGenerated: ${generatedAt}\n`);

  // ── Summary table ──────────────────────────────────────────────────────────
  const summaryHeaders = [
    "repo", "tool", "version", "profile", "language",
    "R@5", "R@10", "MRR", "cases", "duration",
  ];
  const summaryRows = rows.map(({ run, language, profileShort }) => [
    run.repo_id,
    run.tool,
    run.tool_version,
    profileShort,
    language,
    pct(run.aggregate.mean_recall_at_5),
    pct(run.aggregate.mean_recall_at_10),
    pct(run.aggregate.mean_mrr),
    String(run.aggregate.total_cases),
    msToSec(run.duration_ms),
  ]);

  sections.push("## Summary\n\n" + markdownTable(summaryHeaders, summaryRows));

  // ── Per-language averages ──────────────────────────────────────────────────
  const byLanguage = new Map<string, RunRow[]>();
  for (const row of rows) {
    const bucket = byLanguage.get(row.language) ?? [];
    bucket.push(row);
    byLanguage.set(row.language, bucket);
  }

  const langHeaders = ["language", "runs", "avg R@5", "avg R@10", "avg MRR"];
  const langRows: string[][] = [];
  for (const [lang, langRuns] of [...byLanguage.entries()].sort()) {
    const n = langRuns.length;
    const avg = (key: (r: RunRow) => number) =>
      pct(langRuns.reduce((s, r) => s + key(r), 0) / n);
    langRows.push([
      lang,
      String(n),
      avg((r) => r.run.aggregate.mean_recall_at_5),
      avg((r) => r.run.aggregate.mean_recall_at_10),
      avg((r) => r.run.aggregate.mean_mrr),
    ]);
  }

  if (langRows.length > 0) {
    sections.push("## Per-language averages\n\n" + markdownTable(langHeaders, langRows));
  }

  // ── Best profile per repo ──────────────────────────────────────────────────
  const byRepo = new Map<string, RunRow[]>();
  for (const row of rows) {
    const bucket = byRepo.get(row.run.repo_id) ?? [];
    bucket.push(row);
    byRepo.set(row.run.repo_id, bucket);
  }

  const bestHeaders = ["repo", "best profile", "MRR", "R@5"];
  const bestRows: string[][] = [];
  for (const [repoId, repoRuns] of [...byRepo.entries()].sort()) {
    // Sort: descending MRR, tie-break descending R@5
    const sorted = [...repoRuns].sort((a, b) => {
      const mrrDiff = b.run.aggregate.mean_mrr - a.run.aggregate.mean_mrr;
      if (mrrDiff !== 0) return mrrDiff;
      return b.run.aggregate.mean_recall_at_5 - a.run.aggregate.mean_recall_at_5;
    });
    const best = sorted[0];
    bestRows.push([
      repoId,
      best.profileShort,
      pct(best.run.aggregate.mean_mrr),
      pct(best.run.aggregate.mean_recall_at_5),
    ]);
  }

  sections.push("## Best profile per repo\n\n" + markdownTable(bestHeaders, bestRows));

  // ── Notes ─────────────────────────────────────────────────────────────────
  const totalRuns = rows.length;
  const uniqueRepos = new Set(rows.map((r) => r.run.repo_id)).size;
  const uniqueProfiles = new Set(rows.map((r) => r.profileShort)).size;
  const uniqueTools = new Set(rows.map((r) => r.run.tool)).size;
  const toolNote =
    uniqueTools > 1
      ? `Results span **${uniqueTools} distinct tools**; see the tool column above.`
      : `All runs use the same tool (\`${rows[0].run.tool}\`).`;

  sections.push(
    "## Notes\n\n" +
      `- **${totalRuns}** run artifact${totalRuns === 1 ? "" : "s"} loaded` +
      ` across **${uniqueRepos}** repo${uniqueRepos === 1 ? "" : "s"}` +
      ` and **${uniqueProfiles}** profile${uniqueProfiles === 1 ? "" : "s"}.\n` +
      `- ${toolNote}\n` +
      `- Language is inferred from the \`eval_set\` path segment (\`cases/<lang>/...\`).` +
      ` Runs with no recognisable segment appear under \`unknown\`.\n` +
      `- Duration includes index + search time for the full eval set.\n`
  );

  return sections.join("\n\n---\n\n") + "\n";
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const args = parseArgs(process.argv);
  const paths = collectPaths(args);

  if (paths.length === 0) {
    process.stderr.write("ERROR: No input files found. Check --inputs or --dir.\n");
    process.exit(1);
  }

  const runs: EvalRun[] = paths.map(loadRun);

  const rows: RunRow[] = runs.map((run) => ({
    run,
    language: inferLanguage(run.eval_set),
    profileShort: profileName(run.profile),
  }));

  const report = buildReport(rows);

  if (args.output) {
    const outPath = resolve(args.output);
    await Bun.write(outPath, report);
    process.stderr.write(`Report written to: ${outPath}\n`);
  } else {
    process.stdout.write(report);
  }
}

main();
