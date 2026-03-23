#!/usr/bin/env python3
"""Run eval across all 6 repos. Assumes repos are already indexed.

Reports R@5, Acc@5 (strict: ALL gold files in top-5), and MRR.
Usage: source .env && export VOYAGE_API_KEY && ./benchmarks/scripts/eval.sh
"""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
BINARY = ROOT / "target" / "release" / "skelesearch"

REPOS = {
    "mini-redis": "rust",
    "hyperfine": "rust",
    "cobra": "go",
    "httpx": "python",
    "hono": "typescript",
    "zod": "typescript",
}

def main():
    if not BINARY.exists():
        print(f"ERROR: Binary not found at {BINARY}")
        print("Run: cargo build --release --features storage-sqlite")
        sys.exit(1)

    if not os.environ.get("VOYAGE_API_KEY"):
        print("ERROR: VOYAGE_API_KEY not set. Run: source .env && export VOYAGE_API_KEY")
        sys.exit(1)

    print(f"{'REPO':<12} {'R@5':>8} {'Acc@5':>8} {'MRR':>8} {'CASES':>8}")
    print(f"{'----':<12} {'---':>8} {'-----':>8} {'---':>8} {'-----':>8}")

    totals = {"r5": 0, "a5": 0, "mrr": 0, "count": 0}

    for repo, lang in REPOS.items():
        cases_file = ROOT / "benchmarks" / "cases" / lang / f"{repo}.json"
        repo_dir = ROOT / "benchmarks" / "repos" / repo

        if not cases_file.exists():
            print(f"{repo:<12} {'NO CASES':>8}")
            continue

        if not (repo_dir / ".skelesearch").exists():
            print(f"{repo:<12} {'NO INDEX':>8}")
            continue

        # Run eval
        try:
            result = subprocess.run(
                [str(BINARY), "eval", str(cases_file), "--provider", "voyage", "--json"],
                capture_output=True, text=True, timeout=300,
                cwd=str(repo_dir),
                env={**os.environ, "RUST_LOG": "skelesearch=warn"},
            )
        except subprocess.TimeoutExpired:
            print(f"{repo:<12} {'TIMEOUT':>8}")
            continue

        if result.returncode != 0:
            err = result.stderr.strip().split('\n')[0][:60] if result.stderr else "unknown"
            print(f"{repo:<12} {'ERROR':>8}  {err}")
            continue

        try:
            output = json.loads(result.stdout)
        except json.JSONDecodeError:
            print(f"{repo:<12} {'BADJSON':>8}")
            continue

        agg = output["aggregate"]
        cases_out = output["cases"]
        r5 = agg["mean_recall_at_5"]
        mrr = agg["mean_mrr"]
        total = agg["total_cases"]

        # Compute Acc@5 by joining gold set with results
        gold = json.loads(cases_file.read_text())
        gold_by_query = {g["query"]: set(g.get("expected_files", [])) for g in gold}

        acc5_hits = 0
        for c in cases_out:
            expected = gold_by_query.get(c["query"], set())
            retrieved = set(c.get("retrieved_files", [])[:5])
            if expected and expected.issubset(retrieved):
                acc5_hits += 1
        a5 = acc5_hits / len(cases_out) if cases_out else 0

        print(f"{repo:<12} {r5:>7.1%} {a5:>7.1%} {mrr:>8.3f} {total:>8}")

        totals["r5"] += r5
        totals["a5"] += a5
        totals["mrr"] += mrr
        totals["count"] += 1

    print()
    n = totals["count"]
    if n > 0:
        print(f"{'AVERAGE':<12} {totals['r5']/n:>7.1%} {totals['a5']/n:>7.1%} {totals['mrr']/n:>8.3f}")
        print()
        print("Previous baseline: R@5=83.3% MRR=0.851 (Acc@5 not measured)")


if __name__ == "__main__":
    main()
