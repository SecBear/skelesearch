#!/usr/bin/env bun
/**
 * clone-repos.ts — clone or update benchmark corpus repos defined in
 * benchmarks/manifests/repos.toml into benchmarks/repos/<id>.
 *
 * Usage:
 *   bun benchmarks/scripts/clone-repos.ts [--only id1,id2,...] [--dry-run]
 */

import { existsSync } from "fs";
import { join, resolve } from "path";

// ---------------------------------------------------------------------------
// Minimal TOML parser for the [[repo]] manifest format.
// Only handles the subset used in repos.toml: string, bool, and string-array
// values inside [[repo]] table-arrays.
// ---------------------------------------------------------------------------

interface RepoEntry {
  id: string;
  url: string;
  rev: string;
  language: string;
  license: string;
  tags: string[];
  recommended: boolean;
}

function parseReposToml(src: string): RepoEntry[] {
  const repos: RepoEntry[] = [];
  let current: Partial<RepoEntry> | null = null;

  for (const rawLine of src.split("\n")) {
    const line = rawLine.trim();

    if (line === "[[repo]]") {
      if (current) {
        assertComplete(current);
        repos.push(current as RepoEntry);
      }
      current = {};
      continue;
    }

    if (!current || line === "" || line.startsWith("#")) continue;

    const eqIdx = line.indexOf("=");
    if (eqIdx === -1) continue;

    const key = line.slice(0, eqIdx).trim() as keyof RepoEntry;
    const valRaw = line.slice(eqIdx + 1).trim();

    if (key === "tags") {
      // e.g. ["async", "server", "pubsub"]
      const inner = valRaw.replace(/^\[/, "").replace(/\]$/, "");
      current.tags = inner
        .split(",")
        .map((s) => s.trim().replace(/^"/, "").replace(/"$/, ""))
        .filter(Boolean);
    } else if (key === "recommended") {
      current.recommended = valRaw === "true";
    } else {
      // String value: strip surrounding quotes
      const str = valRaw.replace(/^"/, "").replace(/"$/, "");
      (current as Record<string, unknown>)[key] = str;
    }
  }

  if (current) {
    assertComplete(current);
    repos.push(current as RepoEntry);
  }

  return repos;
}

function assertComplete(r: Partial<RepoEntry>): asserts r is RepoEntry {
  for (const field of ["id", "url", "rev", "language", "license"] as const) {
    if (!r[field]) {
      throw new Error(
        `repos.toml: incomplete [[repo]] entry — missing field "${field}" near id=${r.id ?? "(unknown)"}`
      );
    }
  }
}

// ---------------------------------------------------------------------------
// Shell helper — runs a command synchronously, returns { ok, stdout, stderr }
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
    throw new Error(
      `Command failed${ctx}: ${args.join(" ")}\n${r.stderr || r.stdout}`
    );
  }
  return r.stdout;
}

// ---------------------------------------------------------------------------
// Branch vs SHA/tag detection — after fetch, check if origin/<rev> exists
// ---------------------------------------------------------------------------

function isRemoteBranch(repoDir: string, rev: string): boolean {
  const r = run(
    ["git", "show-ref", "--verify", "--quiet", `refs/remotes/origin/${rev}`],
    repoDir
  );
  return r.ok;
}

// ---------------------------------------------------------------------------
// Core clone/update logic
// ---------------------------------------------------------------------------

interface RepoResult {
  id: string;
  language: string;
  requestedRev: string;
  resolvedSha: string;
  path: string;
  status: "cloned" | "updated" | "dry-run";
}

function processRepo(
  repo: RepoEntry,
  reposDir: string,
  dryRun: boolean
): RepoResult {
  const repoDir = join(reposDir, repo.id);
  const absPath = resolve(repoDir);

  if (dryRun) {
    const exists = existsSync(repoDir);
    console.log(
      `[dry-run] ${repo.id}: would ${exists ? "fetch+checkout" : "clone"} ${repo.url} @ ${repo.rev} → ${repoDir}`
    );
    return {
      id: repo.id,
      language: repo.language,
      requestedRev: repo.rev,
      resolvedSha: "(dry-run)",
      path: absPath,
      status: "dry-run",
    };
  }

  let status: RepoResult["status"];

  if (!existsSync(repoDir)) {
    // Clone fresh
    console.log(`[clone] ${repo.id}: cloning from ${repo.url}`);
    runOrDie(
      ["git", "clone", "--no-local", repo.url, repoDir],
      undefined,
      `clone ${repo.id}`
    );
    status = "cloned";
  } else {
    // Repo exists — sanity-check it's not dirty before touching it
    const dirty = run(
      ["git", "status", "--porcelain"],
      repoDir
    );
    if (!dirty.ok) {
      throw new Error(
        `${repo.id}: cannot read git status in ${repoDir}. Is it a valid git repo?`
      );
    }
    if (dirty.stdout.length > 0) {
      throw new Error(
        `${repo.id}: working tree is dirty at ${repoDir}.\nUncommitted changes:\n${dirty.stdout}\nClean or stash before running.`
      );
    }
    console.log(`[fetch] ${repo.id}: fetching ${repo.url}`);
    runOrDie(
      ["git", "fetch", "--all", "--tags", "--prune"],
      repoDir,
      `fetch ${repo.id}`
    );
    status = "updated";
  }

  // Checkout or reset to requested rev
  if (isRemoteBranch(repoDir, repo.rev)) {
    // Branch: create/reset local branch tracking origin/<rev>
    console.log(`[checkout] ${repo.id}: branch ${repo.rev} → origin/${repo.rev}`);
    // Detach first to avoid "cannot force-update checked out branch" errors
    runOrDie(["git", "checkout", "--detach"], repoDir, `detach ${repo.id}`);
    runOrDie(
      ["git", "reset", "--hard", `origin/${repo.rev}`],
      repoDir,
      `reset ${repo.id}`
    );
  } else {
    // SHA or tag: direct checkout
    console.log(`[checkout] ${repo.id}: rev ${repo.rev}`);
    runOrDie(
      ["git", "checkout", repo.rev],
      repoDir,
      `checkout ${repo.id} @ ${repo.rev}`
    );
  }

  const resolvedSha = runOrDie(
    ["git", "rev-parse", "HEAD"],
    repoDir,
    `rev-parse ${repo.id}`
  ).slice(0, 12); // short SHA is enough for display; full SHA stored internally

  return {
    id: repo.id,
    language: repo.language,
    requestedRev: repo.rev,
    resolvedSha,
    path: absPath,
    status,
  };
}

// ---------------------------------------------------------------------------
// Summary table
// ---------------------------------------------------------------------------

function printSummary(results: RepoResult[]): void {
  const cols = {
    id: Math.max(2, ...results.map((r) => r.id.length)),
    language: Math.max(8, ...results.map((r) => r.language.length)),
    rev: Math.max(3, ...results.map((r) => r.requestedRev.length)),
    sha: Math.max(12, ...results.map((r) => r.resolvedSha.length)),
    status: Math.max(6, ...results.map((r) => r.status.length)),
  };

  const pad = (s: string, n: number) => s.padEnd(n);
  const header = [
    pad("id", cols.id),
    pad("language", cols.language),
    pad("rev", cols.rev),
    pad("sha", cols.sha),
    pad("status", cols.status),
    "path",
  ].join("  ");
  const sep = "-".repeat(header.length);

  console.log("\n" + sep);
  console.log(header);
  console.log(sep);
  for (const r of results) {
    console.log(
      [
        pad(r.id, cols.id),
        pad(r.language, cols.language),
        pad(r.requestedRev, cols.rev),
        pad(r.resolvedSha, cols.sha),
        pad(r.status, cols.status),
        r.path,
      ].join("  ")
    );
  }
  console.log(sep + "\n");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const args = process.argv.slice(2);

  // Parse flags
  let onlyIds: Set<string> | null = null;
  let dryRun = false;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--dry-run") {
      dryRun = true;
    } else if (args[i] === "--only") {
      const val = args[i + 1];
      if (!val || val.startsWith("--")) {
        throw new Error("--only requires a comma-separated list of repo ids");
      }
      onlyIds = new Set(val.split(",").map((s) => s.trim()).filter(Boolean));
      i++;
    } else {
      throw new Error(`Unknown flag: ${args[i]}`);
    }
  }

  // Load manifest
  const scriptDir = new URL(".", import.meta.url).pathname;
  const manifestPath = resolve(scriptDir, "../manifests/repos.toml");
  if (!existsSync(manifestPath)) {
    throw new Error(`Manifest not found: ${manifestPath}`);
  }
  const tomlSrc = await Bun.file(manifestPath).text();
  const allRepos = parseReposToml(tomlSrc);

  // Validate --only ids
  if (onlyIds) {
    const validIds = new Set(allRepos.map((r) => r.id));
    const unknown = [...onlyIds].filter((id) => !validIds.has(id));
    if (unknown.length > 0) {
      throw new Error(
        `Unknown repo id(s): ${unknown.join(", ")}\nValid ids: ${[...validIds].join(", ")}`
      );
    }
  }

  const selected = onlyIds
    ? allRepos.filter((r) => onlyIds!.has(r.id))
    : allRepos;

  if (selected.length === 0) {
    throw new Error("No repos selected.");
  }

  // Ensure benchmarks/repos/ exists
  const reposDir = resolve(scriptDir, "../repos");
  if (!dryRun) {
    Bun.spawnSync(["mkdir", "-p", reposDir]);
  }

  console.log(
    `Processing ${selected.length} repo(s)${dryRun ? " [dry-run]" : ""}...`
  );

  const results: RepoResult[] = [];
  const errors: { id: string; message: string }[] = [];

  for (const repo of selected) {
    try {
      const result = processRepo(repo, reposDir, dryRun);
      results.push(result);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      errors.push({ id: repo.id, message });
      console.error(`\n[error] ${repo.id}: ${message}`);
    }
  }

  if (results.length > 0) {
    printSummary(results);
  }

  if (errors.length > 0) {
    process.exit(1);
  }
}

main().catch((err) => {
  console.error(`Fatal: ${err instanceof Error ? err.message : String(err)}`);
  process.exit(1);
});
