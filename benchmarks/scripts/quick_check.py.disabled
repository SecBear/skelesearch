#!/usr/bin/env python3
"""
quick_check.py — Stratified random sampler + eval runner for skelesearch.

Called by quick-check.sh; not intended for direct invocation.

Exit codes:
  0 — all repos evaluated and aggregate R@5 >= threshold
  1 — evaluation ran but R@5 fell below threshold (or a repo failed)
  2 — fatal configuration error (missing binary, bad args, etc.)
"""

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

# Canonical repo → case-file mapping.  Extend here when repos are added.
REPO_CASES: dict[str, str] = {
    "mini-redis": "benchmarks/cases/rust/mini-redis.json",
    "hyperfine":  "benchmarks/cases/rust/hyperfine.json",
    "hono":       "benchmarks/cases/typescript/hono.json",
    "zod":        "benchmarks/cases/typescript/zod.json",
    "httpx":      "benchmarks/cases/python/httpx.json",
    "cobra":      "benchmarks/cases/go/cobra.json",
}

ALL_REPOS = list(REPO_CASES)

# Longest repo name + colon, used to align summary lines.
_LABEL_WIDTH = max(len(r) for r in ALL_REPOS) + 1  # +1 for ":"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Quick stratified eval sampler for skelesearch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--per-repo", type=int, default=3, metavar="N",
        help="Cases to sample per repo (default: 3)",
    )
    p.add_argument(
        "--full", action="store_true",
        help="Run all cases in each file (ignores --per-repo)",
    )
    p.add_argument(
        "--repo", metavar="NAME",
        help=f"Evaluate a single repo.  Valid: {', '.join(ALL_REPOS)}",
    )
    p.add_argument(
        "--seed", default="42", metavar="INT|random",
        help='RNG seed for stratified sampling; "random" for non-deterministic (default: 42)',
    )
    p.add_argument(
        "--provider", default="fastembed", metavar="NAME",
        help="Embedding provider (must match the index; default: fastembed)",
    )
    p.add_argument(
        "--unified", action="store_true",
        help="Activate unified search path (copies benchmarks/configs/unified.toml to each repo)",
    )
    p.add_argument(
        "--binary", default="./target/release/skelesearch", metavar="PATH",
        help="Path to the skelesearch binary (default: ./target/release/skelesearch)",
    )
    p.add_argument(
        "--project-root", default=".", metavar="DIR",
        help="Project root directory (default: current directory)",
    )
    p.add_argument(
        "--threshold", type=float, default=0.75, metavar="FLOAT",
        help="Minimum aggregate R@5 fraction for a passing exit code (default: 0.75)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def sample_cases(cases: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Return at most *n* cases, proportionally sampled from each category.

    For each category we take ceil(n * category_proportion) items, then trim
    the overall list to exactly *n* (in shuffled order) if over-allocated.
    This preserves category distribution even for small *n*.
    """
    if n >= len(cases):
        return list(cases)

    # Group by category, preserving original order within each group.
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for c in cases:
        by_cat[c.get("category", "unknown")].append(c)

    total = len(cases)
    sampled: list[dict] = []
    for cat_items in by_cat.values():
        quota = max(1, math.ceil(n * len(cat_items) / total))
        shuffled = list(cat_items)
        rng.shuffle(shuffled)
        sampled.extend(shuffled[:quota])

    # Shuffle the combined list and trim to exactly n.
    rng.shuffle(sampled)
    return sampled[:n]


# ---------------------------------------------------------------------------
# Index presence check
# ---------------------------------------------------------------------------

def has_index(repo_path: Path) -> bool:
    """Return True iff the repo has an existing skelesearch index."""
    return (repo_path / ".skelesearch" / "index.db").exists()


# ---------------------------------------------------------------------------
# Eval runner
# ---------------------------------------------------------------------------

def run_eval(
    binary: Path,
    repo_path: Path,
    cases_file: Path,
    provider: str,
) -> dict | None:
    """Invoke `skelesearch eval` and return the parsed JSON aggregate.

    Returns None when the subprocess fails or produces unparseable output.
    The binary is always run with *repo_path* as cwd so it picks up the
    repo's .skelesearch.toml and index without any extra flags.
    """
    cmd = [str(binary), "eval", str(cases_file), "--provider", provider, "--json"]
    try:
        result = subprocess.run(
            cmd,
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        print(f"  WARN: eval timed out for {repo_path.name}", file=sys.stderr)
        return None

    if result.returncode != 0:
        snippet = (result.stderr or result.stdout or "").strip()[:300]
        print(f"  WARN: eval failed for {repo_path.name} (exit {result.returncode}): {snippet}",
              file=sys.stderr)
        return None

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        print(f"  WARN: could not parse eval JSON for {repo_path.name}: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_ms(ms: float) -> str:
    """Human-readable duration: '<1000ms' stays as ms, else seconds."""
    return f"{ms:.0f}ms" if ms < 1000 else f"{ms / 1000:.1f}s"


def fmt_label(repo: str) -> str:
    """Left-aligned 'repo:' with consistent padding."""
    label = repo + ":"
    return label.ljust(_LABEL_WIDTH + 1)  # +1 for one space after widest label


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()

    # Resolve binary path relative to project root when not absolute.
    bin_path = Path(args.binary)
    if not bin_path.is_absolute():
        bin_path = root / bin_path
    bin_path = bin_path.resolve()

    if not bin_path.exists():
        print(f"ERROR: Binary not found: {bin_path}", file=sys.stderr)
        sys.exit(2)
    if not os.access(bin_path, os.X_OK):
        print(f"ERROR: Binary not executable: {bin_path}", file=sys.stderr)
        sys.exit(2)

    # Validate --repo before doing any work.
    if args.repo and args.repo not in REPO_CASES:
        print(f"ERROR: Unknown repo '{args.repo}'. Valid: {', '.join(ALL_REPOS)}", file=sys.stderr)
        sys.exit(2)

    # Parse seed.
    if args.seed == "random":
        seed: int | None = None
        seed_label = "random"
    else:
        try:
            seed = int(args.seed)
        except ValueError:
            print(f"ERROR: --seed must be an integer or 'random', got: {args.seed}", file=sys.stderr)
            sys.exit(2)
        seed_label = str(seed)

    # Validate unified config exists before touching any repo.
    unified_toml = root / "benchmarks/configs/unified.toml"
    if args.unified and not unified_toml.exists():
        print(f"ERROR: Unified config not found: {unified_toml}", file=sys.stderr)
        sys.exit(2)

    repos = [args.repo] if args.repo else ALL_REPOS
    per_repo = args.per_repo
    mode_label = "all cases" if args.full else f"{per_repo} cases/repo"

    print(f"QUICK CHECK ({mode_label}, seed={seed_label})")

    # Accumulators for the summary line.
    total_hit = 0
    total_cases_run = 0
    total_mrr_sum = 0.0
    repos_evaluated = 0
    any_failure = False

    wall_start = time.time()

    with tempfile.TemporaryDirectory(prefix="skelesearch-quick-") as tmpdir:
        tmp = Path(tmpdir)
        rng = random.Random(seed)  # one RNG so per-repo sampling is independent but deterministic

        for repo in repos:
            cases_path = root / REPO_CASES[repo]
            repo_path = root / "benchmarks/repos" / repo

            # --- Pre-flight checks ---
            if not cases_path.exists():
                print(f"  {fmt_label(repo)} SKIP (no case file)")
                continue

            if not repo_path.exists():
                print(f"  {fmt_label(repo)} SKIP (repo not cloned)")
                continue

            if not has_index(repo_path):
                print(f"  {fmt_label(repo)} SKIP (no index — run: skelesearch index {repo_path})")
                continue

            # --- Load and sample ---
            with open(cases_path) as fh:
                all_cases: list[dict] = json.load(fh)

            sampled = all_cases if args.full else sample_cases(all_cases, per_repo, rng)

            # --- Write temp eval file ---
            tmp_cases = tmp / f"{repo}-cases.json"
            with open(tmp_cases, "w") as fh:
                json.dump(sampled, fh)

            # --- Optionally install unified search config ---
            if args.unified:
                shutil.copy2(unified_toml, repo_path / ".skelesearch.toml")

            # --- Run eval ---
            t0 = time.time()
            result = run_eval(bin_path, repo_path, tmp_cases, args.provider)
            elapsed_ms = (time.time() - t0) * 1000

            if result is None:
                print(f"  {fmt_label(repo)} FAILED")
                any_failure = True
                continue

            agg = result.get("aggregate", {})
            r5: float = agg.get("mean_recall_at_5", 0.0)
            mrr: float = agg.get("mean_mrr", 0.0)
            n_cases: int = agg.get("total_cases", len(sampled))
            hit = round(r5 * n_cases)

            total_hit += hit
            total_cases_run += n_cases
            total_mrr_sum += mrr
            repos_evaluated += 1

            print(f"  {fmt_label(repo)} {hit}/{n_cases} R@5 ({mrr:.3f} MRR) {fmt_ms(elapsed_ms)}")

    wall_elapsed = time.time() - wall_start

    # --- Summary ---
    if total_cases_run == 0:
        print("  No repos evaluated — nothing to check.")
        sys.exit(2)

    avg_r5 = total_hit / total_cases_run
    avg_mrr = total_mrr_sum / repos_evaluated
    print(
        f"TOTAL: {total_hit}/{total_cases_run} "
        f"R@5={avg_r5 * 100:.1f}% "
        f"MRR={avg_mrr:.3f} "
        f"elapsed={wall_elapsed:.1f}s"
    )

    # Exit 1 on any failure (eval error OR quality regression).
    passed = avg_r5 >= args.threshold and not any_failure
    if not passed:
        if any_failure:
            print("FAIL: one or more repos failed to evaluate", file=sys.stderr)
        else:
            print(
                f"FAIL: R@5 {avg_r5 * 100:.1f}% < threshold {args.threshold * 100:.0f}%",
                file=sys.stderr,
            )
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
