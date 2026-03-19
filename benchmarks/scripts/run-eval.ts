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

import { existsSync, mkdirSync } from "fs";
import { join, resolve, dirname, basename } from "path";
import { skelesearchAdapter } from "./adapters/skelesearch.ts";

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

  // --- Run adapter ---
  const { artifact, summary } = await skelesearchAdapter.run({
    binary: binaryPath,
    repoId,
    repoPath,
    profilePath,
    evalPath,
    outputPath,
    provider: args.provider,
    reuseIndex: args.reuseIndex,
  });

  // --- Write output ---
  await Bun.write(outputPath, JSON.stringify(artifact, null, 2) + "\n");

  // --- Print summary ---
  console.log(summary);
}

main().catch((err) => {
  process.stderr.write(`Fatal: ${err.message ?? err}\n`);
  process.exit(1);
});
