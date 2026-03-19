#!/usr/bin/env bun
/**
 * compare-runs.ts — compare two benchmark run artifacts.
 *
 * Usage:
 *   bun benchmarks/scripts/compare-runs.ts \
 *     --base <run.json> \
 *     --candidate <run.json> \
 *     [--format text|md|json]
 */

import { resolve } from "path";
import { existsSync } from "fs";

// ---------------------------------------------------------------------------
// Types — aligned with eval-run.schema.json
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

interface CaseResult {
  query: string;
  recall_at_5: number;
  recall_at_10: number;
  mrr: number;
  retrieved_files: string[];
}

interface EvalRun {
  tool: string;
  tool_version: string;
  binary: string;
  repo_id: string;
  repo_path: string;
  repo_sha?: string | null;
  profile: string;
  eval_set: string;
  started_at: string;
  duration_ms: number;
  index: IndexStats;
  aggregate: AggregateMetrics;
  cases: CaseResult[];
  environment: {
    provider: string;
    reranker: string | null;
    expansion: boolean | null;
    graph: boolean;
    symbol_enrichment: boolean;
  };
}

// ---------------------------------------------------------------------------
// Comparison types
// ---------------------------------------------------------------------------

interface AggregateDelta {
  mean_recall_at_5: number;
  mean_recall_at_10: number;
  mean_mrr: number;
  duration_ms: number;
  indexed_files: number;
  total_chunks: number;
}

interface CaseDelta {
  query: string;
  recall_at_5_delta: number;
  recall_at_10_delta: number;
  mrr_delta: number;
  base: { recall_at_5: number; recall_at_10: number; mrr: number };
  candidate: { recall_at_5: number; recall_at_10: number; mrr: number };
}

interface CompareResult {
  base: { path: string; tool_version: string };
  candidate: { path: string; tool_version: string };
  aggregate_delta: AggregateDelta;
  improved_cases: CaseDelta[];
  regressed_cases: CaseDelta[];
  unchanged_cases: number;
  missing_in_candidate: string[];
  missing_in_base: string[];
  warnings: string[];
}

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

type Format = "text" | "md" | "json";

interface Args {
  base: string;
  candidate: string;
  format: Format;
}

function parseArgs(argv: string[]): Args {
  const result: Partial<Args> = { format: "text" };
  const args = argv.slice(2);

  for (let i = 0; i < args.length; i++) {
    const flag = args[i];
    switch (flag) {
      case "--base":
        result.base = args[++i];
        break;
      case "--candidate":
        result.candidate = args[++i];
        break;
      case "--format": {
        const fmt = args[++i];
        if (fmt !== "text" && fmt !== "md" && fmt !== "json") {
          process.stderr.write(`Invalid --format "${fmt}"; expected text|md|json\n`);
          process.exit(1);
        }
        result.format = fmt;
        break;
      }
      default:
        process.stderr.write(`Unknown flag: ${flag}\n`);
        process.exit(1);
    }
  }

  const missing: string[] = [];
  if (!result.base) missing.push("--base");
  if (!result.candidate) missing.push("--candidate");
  if (missing.length > 0) {
    process.stderr.write(
      `Missing required flags: ${missing.join(", ")}\n\n` +
        `Usage:\n` +
        `  bun compare-runs.ts --base <run.json> --candidate <run.json> [--format text|md|json]\n`
    );
    process.exit(1);
  }

  return result as Args;
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

async function loadRun(path: string): Promise<EvalRun> {
  const abs = resolve(path);
  if (!existsSync(abs)) {
    process.stderr.write(`File not found: ${abs}\n`);
    process.exit(1);
  }
  try {
    return await Bun.file(abs).json() as EvalRun;
  } catch (e) {
    process.stderr.write(`Failed to parse JSON from ${abs}: ${e}\n`);
    process.exit(1);
  }
}

// ---------------------------------------------------------------------------
// Comparison logic
// ---------------------------------------------------------------------------

const EPS = 1e-9;

/** Classify a case delta: positive = improved, negative = regressed, 0 = unchanged. */
function classifyDelta(d: CaseDelta): "improved" | "regressed" | "unchanged" {
  // Primary signal: MRR delta. Secondary: recall deltas.
  if (d.mrr_delta > EPS) return "improved";
  if (d.mrr_delta < -EPS) return "regressed";
  // MRR is equal — check recall
  const recallSum = d.recall_at_5_delta + d.recall_at_10_delta;
  if (recallSum > EPS) return "improved";
  if (recallSum < -EPS) return "regressed";
  return "unchanged";
}

function compare(basePath: string, candidatePath: string, base: EvalRun, candidate: EvalRun): CompareResult {
  const warnings: string[] = [];

  if (base.repo_id !== candidate.repo_id) {
    warnings.push(`repo_id differs: base="${base.repo_id}" candidate="${candidate.repo_id}"`);
  }
  if (base.profile !== candidate.profile) {
    warnings.push(`profile differs: base="${base.profile}" candidate="${candidate.profile}"`);
  }

  // Build query → case maps
  const baseMap = new Map<string, CaseResult>(base.cases.map((c) => [c.query, c]));
  const candidateMap = new Map<string, CaseResult>(candidate.cases.map((c) => [c.query, c]));

  const baseQueries = new Set(baseMap.keys());
  const candidateQueries = new Set(candidateMap.keys());

  const missing_in_candidate = [...baseQueries].filter((q) => !candidateQueries.has(q));
  const missing_in_base = [...candidateQueries].filter((q) => !baseQueries.has(q));

  if (missing_in_candidate.length > 0) {
    warnings.push(
      `${missing_in_candidate.length} query/queries in base but not candidate: ` +
        missing_in_candidate.map((q) => `"${q}"`).join(", ")
    );
  }
  if (missing_in_base.length > 0) {
    warnings.push(
      `${missing_in_base.length} query/queries in candidate but not base: ` +
        missing_in_base.map((q) => `"${q}"`).join(", ")
    );
  }

  // Per-case deltas over intersection
  const improved: CaseDelta[] = [];
  const regressed: CaseDelta[] = [];
  let unchanged = 0;

  for (const query of baseQueries) {
    if (!candidateQueries.has(query)) continue;
    const b = baseMap.get(query)!;
    const c = candidateMap.get(query)!;
    const delta: CaseDelta = {
      query,
      recall_at_5_delta: c.recall_at_5 - b.recall_at_5,
      recall_at_10_delta: c.recall_at_10 - b.recall_at_10,
      mrr_delta: c.mrr - b.mrr,
      base: { recall_at_5: b.recall_at_5, recall_at_10: b.recall_at_10, mrr: b.mrr },
      candidate: { recall_at_5: c.recall_at_5, recall_at_10: c.recall_at_10, mrr: c.mrr },
    };
    const cls = classifyDelta(delta);
    if (cls === "improved") improved.push(delta);
    else if (cls === "regressed") regressed.push(delta);
    else unchanged++;
  }

  // Sort by |MRR delta| descending
  const byAbsMrr = (a: CaseDelta, b: CaseDelta) =>
    Math.abs(b.mrr_delta) - Math.abs(a.mrr_delta);
  improved.sort(byAbsMrr);
  regressed.sort(byAbsMrr);

  const aggregate_delta: AggregateDelta = {
    mean_recall_at_5: candidate.aggregate.mean_recall_at_5 - base.aggregate.mean_recall_at_5,
    mean_recall_at_10: candidate.aggregate.mean_recall_at_10 - base.aggregate.mean_recall_at_10,
    mean_mrr: candidate.aggregate.mean_mrr - base.aggregate.mean_mrr,
    duration_ms: candidate.duration_ms - base.duration_ms,
    indexed_files: candidate.index.indexed_files - base.index.indexed_files,
    total_chunks: candidate.index.total_chunks - base.index.total_chunks,
  };

  return {
    base: { path: basePath, tool_version: base.tool_version },
    candidate: { path: candidatePath, tool_version: candidate.tool_version },
    aggregate_delta,
    improved_cases: improved,
    regressed_cases: regressed,
    unchanged_cases: unchanged,
    missing_in_candidate,
    missing_in_base,
    warnings,
  };
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

/** Format a numeric delta with a leading sign and fixed decimal places. */
function fmtDelta(v: number, decimals = 4): string {
  const s = v.toFixed(decimals);
  return v >= 0 ? `+${s}` : s;
}

/** Format a plain number (no sign). */
function fmt(v: number, decimals = 4): string {
  return v.toFixed(decimals);
}

/** Format an integer delta. */
function fmtInt(v: number): string {
  return v >= 0 ? `+${v}` : `${v}`;
}

// ---------------------------------------------------------------------------
// Text output
// ---------------------------------------------------------------------------

function renderText(r: CompareResult): string {
  const lines: string[] = [];

  lines.push("=== Benchmark Comparison ===");
  lines.push(`  base:      ${r.base.path}  (${r.base.tool_version})`);
  lines.push(`  candidate: ${r.candidate.path}  (${r.candidate.tool_version})`);
  lines.push("");

  if (r.warnings.length > 0) {
    lines.push("WARNINGS:");
    for (const w of r.warnings) lines.push(`  ! ${w}`);
    lines.push("");
  }

  lines.push("--- Aggregate Metrics ---");
  const d = r.aggregate_delta;
  lines.push(
    `  mean_recall@5   ${fmtDelta(d.mean_recall_at_5)}  (delta)`
  );
  lines.push(
    `  mean_recall@10  ${fmtDelta(d.mean_recall_at_10)}  (delta)`
  );
  lines.push(`  mean_mrr        ${fmtDelta(d.mean_mrr)}  (delta)`);
  lines.push(`  duration_ms     ${fmtInt(d.duration_ms)}  (delta)`);
  lines.push(`  indexed_files   ${fmtInt(d.indexed_files)}  (delta)`);
  lines.push(`  total_chunks    ${fmtInt(d.total_chunks)}  (delta)`);
  lines.push("");

  const totalCompared =
    r.improved_cases.length + r.regressed_cases.length + r.unchanged_cases;
  lines.push(
    `--- Per-Case Results  (intersection: ${totalCompared} queries) ---`
  );
  lines.push(
    `  improved:  ${r.improved_cases.length}   regressed: ${r.regressed_cases.length}   unchanged: ${r.unchanged_cases}`
  );

  if (r.missing_in_candidate.length > 0) {
    lines.push(
      `  missing in candidate: ${r.missing_in_candidate.length} ` +
        `(${r.missing_in_candidate.map((q) => `"${q}"`).join(", ")})`
    );
  }
  if (r.missing_in_base.length > 0) {
    lines.push(
      `  missing in base: ${r.missing_in_base.length} ` +
        `(${r.missing_in_base.map((q) => `"${q}"`).join(", ")})`
    );
  }

  if (r.regressed_cases.length > 0) {
    lines.push("");
    lines.push("REGRESSIONS (sorted by |MRR delta|):");
    for (const c of r.regressed_cases) {
      lines.push(`  "${c.query}"`);
      lines.push(
        `    mrr: ${fmt(c.base.mrr)} → ${fmt(c.candidate.mrr)}  (${fmtDelta(c.mrr_delta)})`
      );
      lines.push(
        `    recall@5: ${fmt(c.base.recall_at_5)} → ${fmt(c.candidate.recall_at_5)}  (${fmtDelta(c.recall_at_5_delta)})`
      );
      lines.push(
        `    recall@10: ${fmt(c.base.recall_at_10)} → ${fmt(c.candidate.recall_at_10)}  (${fmtDelta(c.recall_at_10_delta)})`
      );
    }
  }

  if (r.improved_cases.length > 0) {
    lines.push("");
    lines.push("IMPROVEMENTS (sorted by |MRR delta|):");
    for (const c of r.improved_cases) {
      lines.push(`  "${c.query}"`);
      lines.push(
        `    mrr: ${fmt(c.base.mrr)} → ${fmt(c.candidate.mrr)}  (${fmtDelta(c.mrr_delta)})`
      );
      lines.push(
        `    recall@5: ${fmt(c.base.recall_at_5)} → ${fmt(c.candidate.recall_at_5)}  (${fmtDelta(c.recall_at_5_delta)})`
      );
      lines.push(
        `    recall@10: ${fmt(c.base.recall_at_10)} → ${fmt(c.candidate.recall_at_10)}  (${fmtDelta(c.recall_at_10_delta)})`
      );
    }
  }

  lines.push("");
  return lines.join("\n");
}

// ---------------------------------------------------------------------------
// Markdown output
// ---------------------------------------------------------------------------

function renderMd(r: CompareResult): string {
  const lines: string[] = [];

  lines.push("## Benchmark Comparison");
  lines.push("");
  lines.push(`| | Path | Version |`);
  lines.push(`|---|---|---|`);
  lines.push(`| **base** | \`${r.base.path}\` | \`${r.base.tool_version}\` |`);
  lines.push(`| **candidate** | \`${r.candidate.path}\` | \`${r.candidate.tool_version}\` |`);
  lines.push("");

  if (r.warnings.length > 0) {
    lines.push("> **Warnings**");
    for (const w of r.warnings) lines.push(`> - ${w}`);
    lines.push("");
  }

  lines.push("### Aggregate Metrics");
  lines.push("");
  lines.push("| Metric | Delta |");
  lines.push("|---|---|");
  const d = r.aggregate_delta;
  lines.push(`| mean_recall@5 | ${fmtDelta(d.mean_recall_at_5)} |`);
  lines.push(`| mean_recall@10 | ${fmtDelta(d.mean_recall_at_10)} |`);
  lines.push(`| mean_mrr | ${fmtDelta(d.mean_mrr)} |`);
  lines.push(`| duration_ms | ${fmtInt(d.duration_ms)} |`);
  lines.push(`| indexed_files | ${fmtInt(d.indexed_files)} |`);
  lines.push(`| total_chunks | ${fmtInt(d.total_chunks)} |`);
  lines.push("");

  const totalCompared =
    r.improved_cases.length + r.regressed_cases.length + r.unchanged_cases;
  lines.push("### Per-Case Results");
  lines.push("");
  lines.push(
    `Intersection: **${totalCompared}** queries — ` +
      `improved: **${r.improved_cases.length}**, ` +
      `regressed: **${r.regressed_cases.length}**, ` +
      `unchanged: **${r.unchanged_cases}**`
  );
  lines.push("");

  if (r.missing_in_candidate.length > 0) {
    lines.push(
      `- **Missing in candidate (${r.missing_in_candidate.length}):** ` +
        r.missing_in_candidate.map((q) => `\`${q}\``).join(", ")
    );
  }
  if (r.missing_in_base.length > 0) {
    lines.push(
      `- **Missing in base (${r.missing_in_base.length}):** ` +
        r.missing_in_base.map((q) => `\`${q}\``).join(", ")
    );
  }

  if (r.regressed_cases.length > 0) {
    lines.push("");
    lines.push("#### Regressions");
    lines.push("");
    lines.push("| Query | MRR delta | Recall@5 delta | Recall@10 delta |");
    lines.push("|---|---|---|---|");
    for (const c of r.regressed_cases) {
      lines.push(
        `| \`${c.query}\` | ${fmtDelta(c.mrr_delta)} | ${fmtDelta(c.recall_at_5_delta)} | ${fmtDelta(c.recall_at_10_delta)} |`
      );
    }
    lines.push("");
  }

  if (r.improved_cases.length > 0) {
    lines.push("");
    lines.push("#### Improvements");
    lines.push("");
    lines.push("| Query | MRR delta | Recall@5 delta | Recall@10 delta |");
    lines.push("|---|---|---|---|");
    for (const c of r.improved_cases) {
      lines.push(
        `| \`${c.query}\` | ${fmtDelta(c.mrr_delta)} | ${fmtDelta(c.recall_at_5_delta)} | ${fmtDelta(c.recall_at_10_delta)} |`
      );
    }
    lines.push("");
  }

  return lines.join("\n");
}

// ---------------------------------------------------------------------------
// JSON output
// ---------------------------------------------------------------------------

function renderJson(r: CompareResult): string {
  return JSON.stringify(r, null, 2);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const args = parseArgs(process.argv);

  const [base, candidate] = await Promise.all([
    loadRun(args.base),
    loadRun(args.candidate),
  ]);

  const result = compare(args.base, args.candidate, base, candidate);

  let output: string;
  switch (args.format) {
    case "text":
      output = renderText(result);
      break;
    case "md":
      output = renderMd(result);
      break;
    case "json":
      output = renderJson(result);
      break;
  }

  process.stdout.write(output);
}

main();
