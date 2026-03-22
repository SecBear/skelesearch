/**
 * skelesearch.ts — BenchmarkAdapter implementation for the skelesearch binary.
 *
 * Owns the full skelesearch pipeline:
 *   1. Copy benchmark profile into the repo clone as .skelesearch.toml
 *   2. Optionally wipe the existing index
 *   3. Run `skelesearch index`
 *   4. Run `skelesearch status --json` to gather index statistics
 *   5. Run `skelesearch eval --json` to produce recall/MRR metrics
 *   6. Detect tool version (git SHA when running from source)
 *   7. Assemble and return the artifact
 */

import { existsSync, rmSync } from "fs";
import { join, resolve, dirname, basename } from "path";
import type { BenchmarkAdapter, BenchmarkRunInput, BenchmarkRunResult } from "./types.ts";

// ---------------------------------------------------------------------------
// Shell execution helpers
// ---------------------------------------------------------------------------

interface RunResult {
  ok: boolean;
  stdout: string;
  stderr: string;
  code: number;
}

function run(args: string[], cwd?: string, env?: Record<string, string | undefined>): RunResult {
  const proc = Bun.spawnSync(args, {
    cwd,
    env: env ?? undefined,
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

function runOrDie(args: string[], cwd?: string, context?: string, env?: Record<string, string | undefined>): string {
  const r = run(args, cwd, env);
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
// Minimal TOML parser — handles [section], [section.sub], key = value pairs.
// Only parses scalar values (string, bool, integer) at arbitrary nesting depth.
// Sufficient for extracting environment metadata from benchmark profile TOML.
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
// Adapter
// ---------------------------------------------------------------------------

export const skelesearchAdapter: BenchmarkAdapter = {
  id: "skelesearch",

  async run(input: BenchmarkRunInput): Promise<BenchmarkRunResult> {
    const {
      binary,
      repoId,
      repoPath,
      profilePath,
      evalPath,
      outputPath,
      provider,
      reuseIndex,
    } = input;

    const startedAt = new Date();
    const startMs = Date.now();

    // Project root: adapters/ → scripts/ → benchmarks/ → project root
    const adapterDir = dirname(resolve(import.meta.url.replace("file://", "")));
    const projectRoot = resolve(adapterDir, "../../..");

    // --- Copy profile into repo clone ---
    const profileSrc = await Bun.file(profilePath).text();
    const profileDoc = parseToml(profileSrc);
    const tomlDest = join(repoPath, ".skelesearch.toml");
    await Bun.write(tomlDest, profileSrc);

    // --- Build clean env for skelesearch subprocesses ---
    // If the profile doesn't explicitly configure a cloud reranker, strip cloud
    // API keys so auto-detection doesn't silently activate one. This prevents
    // benchmark results from being contaminated by environment-leaked keys.
    const rerankerCfg = tomlGet(profileDoc, "search", "reranker", "provider");
    const skelEnv: Record<string, string | undefined> = { ...process.env };
    if (!rerankerCfg) {
      // No explicit reranker in profile — strip all cloud reranker keys
      for (const key of ["VOYAGE_API_KEY", "JINA_API_KEY", "COHERE_API_KEY"]) {
        delete skelEnv[key];
      }
    }
    // Always set RUST_LOG for telemetry capture
    skelEnv["RUST_LOG"] = skelEnv["RUST_LOG"] ?? "skelesearch=info";

    // --- Wipe index unless --reuse-index ---
    const indexDir = join(repoPath, ".skelesearch");
    if (!reuseIndex && existsSync(indexDir)) {
      rmSync(indexDir, { recursive: true, force: true });
    }

    // --- Index ---
    console.log(`[index] ${repoId} with provider=${provider}`);
    runOrDie([binary, "index", repoPath, "--provider", provider], repoPath, "index", skelEnv);

    // --- Status ---
    console.log(`[status] reading index statistics`);
    const statusResult = run([binary, "status", "--json"], repoPath, skelEnv);
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
    console.log(`[eval] running eval set: ${evalPath}`);
    const evalResult = run(
      [binary, "eval", evalPath, "--provider", provider, "--json"],
      repoPath,
      skelEnv
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
      toolVersion = binary;
    }

    // --- Repo SHA ---
    let repoSha: string | null = null;
    const repoGitResult = run(["git", "rev-parse", "HEAD"], repoPath);
    if (repoGitResult.ok && repoGitResult.stdout) {
      repoSha = repoGitResult.stdout;
    }

    // --- Extract environment metadata from profile TOML ---
    const rerankerProvider = tomlGet(profileDoc, "search", "reranker", "provider");
    const expansionEnabled = tomlGet(profileDoc, "search", "expansion", "enabled");
    const graphEnabled = tomlGet(profileDoc, "search", "graph", "enabled");
    const symbolEnrichment = tomlGet(profileDoc, "index", "symbol_enrichment");

    const environment = {
      provider,
      reranker: typeof rerankerProvider === "string"
        ? rerankerProvider
        : rerankerCfg ? String(rerankerCfg) : null,
      expansion: typeof expansionEnabled === "boolean" ? expansionEnabled : null,
      graph: graphEnabled === true,
      symbol_enrichment: symbolEnrichment !== false, // default true per profile convention
    };

    // --- Assemble artifact ---
    const durationMs = Date.now() - startMs;
    const artifact = {
      tool: "skelesearch",
      tool_version: toolVersion,
      binary,
      repo_id: repoId,
      repo_path: repoPath,
      repo_sha: repoSha,
      profile: profilePath,
      eval_set: evalPath,
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

    const ag = artifact.aggregate;
    const r5 = (ag.mean_recall_at_5 * 100).toFixed(1);
    const r10 = (ag.mean_recall_at_10 * 100).toFixed(1);
    const mrr = ag.mean_mrr.toFixed(3);
    const summary =
      `\nDone  repo=${repoId}  profile=${basename(profilePath, ".toml")}` +
      `  R@5=${r5}%  R@10=${r10}%  MRR=${mrr}` +
      `  cases=${ag.total_cases}  ms=${durationMs}` +
      `\n→ ${outputPath}`;

    return { artifact, summary };
  },
};
