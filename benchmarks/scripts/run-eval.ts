#!/usr/bin/env bun
/**
 * run-eval.ts — execute one benchmark cell: binary × repo × profile × eval set.
 *
 * Usage:
 *   bun benchmarks/scripts/run-eval.ts \
 *     --binary ./target/release/skelesearch \
 *     --repo mini-redis \
 *     --profile benchmarks/configs/voyage-full.toml \
 *     --eval benchmarks/cases/rust/mini-redis.json \
 *     --output benchmarks/runs/mini-redis-voyage-full.json \
 *     [--provider voyage] \
 *     [--reuse-index]
 */

import { existsSync, mkdirSync, rmSync, cpSync } from "fs";
import { join, resolve, dirname, basename, isAbsolute } from "path";

// ---------------------------------------------------------------------------
// Minimal TOML parser — handles [section], [section.sub], key = value pairs.
// Only parses scalar values (string, bool, integer) at arbitrary nesting depth.
// Sufficient for the benchmark profile TOML structure.
// ---------------------------------------------------------------------------

type TomlValue = string | boolean | number | null;
type TomlDoc = { [key: string]: TomlValue | TomlDoc };

function parseToml(src: string): TomlDoc {
  const doc: TomlDoc = {};
  let section: string[] = [];

  for (const rawLine of src.split("\n")) {
    const line = rawLine.trim();
    if (line === "" || line.startsWith("#")) continue;

    // Section header: [a.b.c]
    if (line.startsWith("[") && line.endsWith("]") && !line.startsWith("[[")) {
      const path = line.slice(1, -1).trim();
      section = path.split(".").map((s) => s.trim());
      // Ensure nested objects exist
      let node: TomlDoc = doc;
      for (const key of section) {
        if (typeof node[key] !== "object" || node[key] === null) {
          node[key] = {};
        }
        node = node[key] as TomlDoc;
      }
      continue;
    }

    // Key = value
    const eqIdx = line.indexOf("=");
    if (eqIdx === -1) continue;

    const key = line.slice(0, eqIdx).trim();
    const valRaw = line.slice(eqIdx + 1).trim();
    const value = parseTomlScalar(valRaw);

    // Navigate to current section and set key
    let node: TomlDoc = doc;
    for (const seg of section) {
      if (typeof node[seg] !== "object" || node[seg] === null) {
        node[seg] = {};
      }
      node = node[seg] as TomlDoc;
    }
    node[key] = value;
  }

  return doc;
}

function parseTomlScalar(raw: string): TomlValue {
  if (raw === "true") return true;
  if (raw === "false") return false;
  if (/^".*"$/.test(raw)) return raw.slice(1, -1);
  if (/^'.*'$/.test(raw)) return raw.slice(1, -1);
  const n = Number(raw);
  if (!isNaN(n) && raw !== "") return n;
  return raw; // fall through: return as-is (e.g., unquoted identifiers)
}

/** Safely read a nested TOML path, returning undefined if absent. */
function tomlGet(doc: TomlDoc, ...path: string[]): TomlValue | undefined {
  let node: TomlDoc | TomlValue = doc;
  for (const seg of path) {
    if (typeof node !== "object" || node === null) return undefined;
    node = (node as TomlDoc)[seg];
    if (node === undefined) return undefined;
  }
  return node as TomlValue;
}

// ---------------------------------------------------------------------------
// Repos manifest parser — reuses the [[repo]] table-array format from
// clone-repos.ts; we only need id → path resolution here.
// ---------------------------------------------------------------------------

interface RepoManifestEntry {
  id: string;
  url: string;
}

function parseReposManifest(src: string): RepoManifestEntry[] {
  const repos: RepoManifestEntry[] = [];
  let current: Partial<RepoManifestEntry> | null = null;

  for (const rawLine of src.split("\n")) {
    const line = rawLine.trim();
    if (line === "[[repo]]") {
      if (current?.id) repos.push(current as RepoManifestEntry);
      current = {};
      continue;
    }
    if (!current || line === "" || line.startsWith("#")) continue;
    const eqIdx = line.indexOf("=");
    if (eqIdx === -1) continue;
    const key = line.slice(0, eqIdx).trim();
    const valRaw = line.slice(eqIdx + 1).trim();
    if (key === "id" || key === "url") {
      (current as Record<string, string>)[key] = valRaw
        .replace(/^"/, "")
        .replace(/"$/, "");
    }
  }
  if (current?.id) repos.push(current as RepoManifestEntry);
  return repos;
}

// ---------------------------------------------------------------------------
// Shell execution helpers
// ---------------------------------------------------------------------------

interface RunResult {
  ok: boolean;
  stdout: string;
  stderr: string;
  code: number;
}

function run(args: string[], cwd?: string): RunResult {
  const proc = Bun.spawnSync(args, {
    cwd,
    stdout: "pipe",
    stderr: "pipe",
  });
  const stdout = new TextDecoder().decode(proc.stdout).trimEnd();
  const stderr = new TextDecoder().decode(proc.stderr).trimEnd();
  return {
    ok: proc.exitCode === 0,
    stdout,
    stderr,
    code: proc.exitCode ?? 1,
  };
}

function runOrDie(args: string[], cwd?: string, context?: string): string {
  const r = run(args, cwd);
  if (!r.ok) {
    const ctx = context ? ` (${context})` : "";
    const detail = r.stderr || r.stdout;
    process.stderr.write(
      `\nERROR: Command failed${ctx}:\n  ${args.join(" ")}\n${detail}\n`
    );
    process.exit(1);
  }
  return r.stdout;
}

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

interface Args {
  binary: string;
  repo: string;
  profile: string;
  eval: string;
  output: string;
  provider: string;
  reuseIndex: boolean;
}

function parseArgs(argv: string[]): Args {
  const result: Partial<Args> = { provider: "voyage", reuseIndex: false };
  const args = argv.slice(2);

  for (let i = 0; i < args.length; i++) {
    const flag = args[i];
    switch (flag) {
      case "--binary":
        result.binary = args[++i];
        break;
      case "--repo":
        result.repo = args[++i];
        break;
      case "--profile":
        result.profile = args[++i];
        break;
      case "--eval":
        result.eval = args[++i];
        break;
      case "--output":
        result.output = args[++i];
        break;
      case "--provider":
        result.provider = args[++i];
        break;
      case "--reuse-index":
        result.reuseIndex = true;
        break;
      default:
        process.stderr.write(`Unknown flag: ${flag}\n`);
        process.exit(1);
    }
  }

  const missing: string[] = [];
  for (const req of ["binary", "repo", "profile", "eval", "output"] as const) {
    if (!result[req]) missing.push(`--${req}`);
  }
  if (missing.length > 0) {
    process.stderr.write(
      `Missing required flags: ${missing.join(", ")}\n\nUsage:\n` +
        `  bun run-eval.ts --binary <path> --repo <id|path> --profile <path>\n` +
        `                  --eval <path> --output <path> [--provider <name>]\n` +
        `                  [--reuse-index]\n`
    );
    process.exit(1);
  }

  return result as Args;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const args = parseArgs(process.argv);
  const startedAt = new Date();
  const startMs = Date.now();

  // --- Resolve binary ---
  const binaryPath = resolve(args.binary);
  if (!existsSync(binaryPath)) {
    process.stderr.write(`Binary not found: ${binaryPath}\n`);
    process.exit(1);
  }

  // --- Resolve repo path ---
  // Determine script location to find project root reliably.
  const scriptDir = dirname(resolve(import.meta.url.replace("file://", "")));
  const projectRoot = resolve(scriptDir, "../..");

  let repoPath: string;
  const manifestPath = join(projectRoot, "benchmarks/manifests/repos.toml");

  // Is `--repo` a known manifest id?
  let repoId = args.repo;
  const candidateFromManifest = join(projectRoot, "benchmarks/repos", args.repo);

  if (existsSync(manifestPath)) {
    const manifestSrc = await Bun.file(manifestPath).text();
    const entries = parseReposManifest(manifestSrc);
    const entry = entries.find((e) => e.id === args.repo);
    if (entry) {
      repoPath = candidateFromManifest;
      if (!existsSync(repoPath)) {
        process.stderr.write(
          `Repo "${args.repo}" is listed in the manifest but has not been cloned.\n` +
            `Run: bun benchmarks/scripts/clone-repos.ts --only ${args.repo}\n`
        );
        process.exit(1);
      }
    } else {
      // Not a manifest id — treat as direct path
      repoId = basename(resolve(args.repo));
      repoPath = resolve(args.repo);
    }
  } else {
    // No manifest — treat as direct path
    repoId = basename(resolve(args.repo));
    repoPath = resolve(args.repo);
  }

  if (!existsSync(repoPath)) {
    process.stderr.write(`Repo path does not exist: ${repoPath}\n`);
    process.exit(1);
  }

  // --- Validate other inputs ---
  const profilePath = resolve(args.profile);
  if (!existsSync(profilePath)) {
    process.stderr.write(`Profile not found: ${profilePath}\n`);
    process.exit(1);
  }

  const evalPath = resolve(args.eval);
  if (!existsSync(evalPath)) {
    process.stderr.write(`Eval set not found: ${evalPath}\n`);
    process.exit(1);
  }

  const outputPath = resolve(args.output);
  mkdirSync(dirname(outputPath), { recursive: true });

  // --- Read and apply profile ---
  const profileSrc = await Bun.file(profilePath).text();
  const profileDoc = parseToml(profileSrc);
  const tomlDest = join(repoPath, ".skelesearch.toml");
  await Bun.write(tomlDest, profileSrc);

  // --- Wipe index unless --reuse-index ---
  const indexDir = join(repoPath, ".skelesearch");
  if (!args.reuseIndex && existsSync(indexDir)) {
    rmSync(indexDir, { recursive: true, force: true });
  }

  // --- Index ---
  console.log(`[index] ${repoId} with provider=${args.provider}`);
  runOrDie(
    [binaryPath, "index", repoPath, "--provider", args.provider],
    repoPath,
    "index"
  );

  // --- Status ---
  console.log(`[status] reading index statistics`);
  const statusResult = run(
    [binaryPath, "status", "--json"],
    repoPath
  );
  if (!statusResult.ok) {
    process.stderr.write(
      `status command failed:\n${statusResult.stderr || statusResult.stdout}\n`
    );
    process.exit(1);
  }

  let indexStats: { indexed_files: number; total_chunks: number } = {
    indexed_files: 0,
    total_chunks: 0,
  };
  try {
    const statusJson = JSON.parse(statusResult.stdout);
    indexStats = {
      indexed_files: statusJson.indexed_files ?? 0,
      total_chunks: statusJson.total_chunks ?? 0,
    };
  } catch {
    process.stderr.write(
      `WARNING: Could not parse status JSON: ${statusResult.stdout}\n`
    );
  }

  // --- Eval ---
  console.log(`[eval] running eval set: ${args.eval}`);
  const evalResult = run(
    [binaryPath, "eval", evalPath, "--provider", args.provider, "--json"],
    repoPath
  );
  if (!evalResult.ok) {
    process.stderr.write(
      `eval command failed:\n${evalResult.stderr || evalResult.stdout}\n`
    );
    process.exit(1);
  }

  let evalJson: {
    aggregate: {
      mean_recall_at_5: number;
      mean_recall_at_10: number;
      mean_mrr: number;
      total_cases: number;
    };
    cases: unknown[];
  };
  try {
    evalJson = JSON.parse(evalResult.stdout);
  } catch {
    process.stderr.write(
      `Could not parse eval JSON output:\n${evalResult.stdout}\n`
    );
    process.exit(1);
  }

  // --- Tool version ---
  // Prefer git short-SHA when the binary lives inside this project.
  let toolVersion: string;
  const gitResult = run(["git", "rev-parse", "--short", "HEAD"], projectRoot);
  if (gitResult.ok && gitResult.stdout) {
    toolVersion = `git:${gitResult.stdout}`;
  } else {
    toolVersion = args.binary;
  }

  // --- Repo SHA ---
  let repoSha: string | null = null;
  const repoGitResult = run(["git", "rev-parse", "HEAD"], repoPath);
  if (repoGitResult.ok && repoGitResult.stdout) {
    repoSha = repoGitResult.stdout;
  }

  // --- Extract environment from profile TOML ---
  const rerankerProvider = tomlGet(profileDoc, "search", "reranker", "provider");
  const expansionEnabled = tomlGet(profileDoc, "search", "expansion", "enabled");
  const graphEnabled = tomlGet(profileDoc, "search", "graph", "enabled");
  const symbolEnrichment = tomlGet(profileDoc, "index", "symbol_enrichment");

  const environment = {
    provider: args.provider,
    reranker: typeof rerankerProvider === "string" ? rerankerProvider : null,
    expansion: typeof expansionEnabled === "boolean" ? expansionEnabled : null,
    graph: graphEnabled === true,
    symbol_enrichment: symbolEnrichment !== false, // default true per profile convention
  };

  // --- Assemble artifact ---
  const durationMs = Date.now() - startMs;
  const artifact = {
    tool: "skelesearch",
    tool_version: toolVersion,
    binary: args.binary,
    repo_id: repoId,
    repo_path: repoPath,
    repo_sha: repoSha,
    profile: args.profile,
    eval_set: args.eval,
    started_at: startedAt.toISOString(),
    duration_ms: durationMs,
    index: {
      indexed_files: indexStats.indexed_files,
      total_chunks: indexStats.total_chunks,
      cache_hits: null,
      cache_misses: null,
      resolved_import_edges: null,
    },
    aggregate: evalJson.aggregate,
    cases: evalJson.cases,
    environment,
  };

  await Bun.write(outputPath, JSON.stringify(artifact, null, 2) + "\n");

  // --- Summary ---
  const ag = artifact.aggregate;
  const r5 = (ag.mean_recall_at_5 * 100).toFixed(1);
  const r10 = (ag.mean_recall_at_10 * 100).toFixed(1);
  const mrr = ag.mean_mrr.toFixed(3);
  console.log(
    `\nDone  repo=${repoId}  profile=${basename(args.profile, ".toml")}` +
      `  R@5=${r5}%  R@10=${r10}%  MRR=${mrr}` +
      `  cases=${ag.total_cases}  ms=${durationMs}` +
      `\n→ ${outputPath}`
  );
}

main().catch((err) => {
  process.stderr.write(`Fatal: ${err.message ?? err}\n`);
  process.exit(1);
});
