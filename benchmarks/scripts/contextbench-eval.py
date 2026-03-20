#!/usr/bin/env python3
"""Run skelesearch against ContextBench verified instances.

Usage:
    uv run --with datasets --with huggingface_hub python3 benchmarks/scripts/contextbench-eval.py \
        --binary ./target/release/skelesearch \
        --languages python,typescript,go,rust,javascript \
        --limit-per-lang 10 \
        --output benchmarks/runs/contextbench-pilot.json

Requires: git, skelesearch binary, VOYAGE_API_KEY + OPENAI_API_KEY in env.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path


def load_dataset_filtered(languages: list[str], limit_per_lang: int | None):
    """Load ContextBench verified and filter to target languages."""
    from datasets import load_dataset

    ds = load_dataset("Contextbench/ContextBench", "contextbench_verified", split="train")
    by_lang: dict[str, list] = defaultdict(list)
    for row in ds:
        if row["language"] in languages:
            by_lang[row["language"]].append(row)

    selected = []
    for lang in languages:
        pool = by_lang.get(lang, [])
        if limit_per_lang:
            pool = pool[:limit_per_lang]
        selected.extend(pool)

    return selected


def extract_gold_files(row) -> list[str]:
    """Extract unique gold file paths from gold_context."""
    gc = row["gold_context"]
    if isinstance(gc, str):
        gc = json.loads(gc)
    return list(set(e["file"] for e in gc))


def clone_at_commit(repo_url: str, commit: str, dest: Path) -> bool:
    """Shallow clone a repo and checkout a specific commit."""
    try:
        # Clone with minimal depth, then fetch the specific commit
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", repo_url, str(dest)],
            capture_output=True, timeout=120, check=True,
        )
        subprocess.run(
            ["git", "-C", str(dest), "checkout", commit],
            capture_output=True, timeout=60, check=True,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"  WARN: clone/checkout failed for {repo_url}@{commit[:8]}: {e}", file=sys.stderr)
        return False


def run_skelesearch(binary: str, project_dir: Path, query: str, provider: str = "fastembed", top_k: int = 10) -> list[str]:
    """Index and search with skelesearch, return retrieved file paths."""
    env = os.environ.copy()

    # Index
    idx_result = subprocess.run(
        [binary, "index", str(project_dir), "--provider", provider],
        capture_output=True, timeout=900, env=env,
    )
    if idx_result.returncode != 0:
        print(f"  WARN: index failed: {idx_result.stderr[:200]}", file=sys.stderr)
        return []

    # Search
    search_result = subprocess.run(
        [binary, "search", query, "--top-k", str(top_k), "--json"],
        capture_output=True, timeout=60, env=env, cwd=str(project_dir),
    )
    if search_result.returncode != 0:
        print(f"  WARN: search failed: {search_result.stderr[:200]}", file=sys.stderr)
        return []

    try:
        results = json.loads(search_result.stdout)
        if isinstance(results, list):
            return [r.get("file_path", r.get("file", "")) for r in results]
        return []
    except json.JSONDecodeError:
        return []


def recall_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    if not expected:
        return 1.0
    top_k = retrieved[:k]
    hits = sum(1 for e in expected if any(r.endswith(e) or e.endswith(r) for r in top_k))
    return hits / len(expected)


def precision_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for r in top_k if any(r.endswith(e) or e.endswith(r) for e in expected))
    return hits / len(top_k)


def mrr(retrieved: list[str], expected: list[str]) -> float:
    for i, r in enumerate(retrieved):
        if any(r.endswith(e) or e.endswith(r) for e in expected):
            return 1.0 / (i + 1)
    return 0.0


def main():
    parser = argparse.ArgumentParser(description="Run skelesearch against ContextBench")
    parser.add_argument("--binary", required=True, help="Path to skelesearch binary")
    parser.add_argument("--languages", default="python,typescript,go,rust,javascript")
    parser.add_argument("--limit-per-lang", type=int, default=10)
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--cache-dir", default=None, help="Dir to cache repo clones")
    parser.add_argument("--provider", default="fastembed", help="Embedding provider")
    args = parser.parse_args()
    args.binary = str(Path(args.binary).resolve())

    languages = [l.strip() for l in args.languages.split(",")]
    instances = load_dataset_filtered(languages, args.limit_per_lang)
    print(f"Loaded {len(instances)} instances across {languages}")

    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(tempfile.mkdtemp(prefix="cb_"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Cache dir: {cache_dir}")

    results = []
    lang_metrics: dict[str, list] = defaultdict(list)

    for i, row in enumerate(instances):
        instance_id = row["instance_id"]
        repo = row["repo"]
        commit = row["base_commit"]
        lang = row["language"]
        query = row["problem_statement"]
        gold_files = extract_gold_files(row)

        print(f"[{i+1}/{len(instances)}] {instance_id} ({lang}, {len(gold_files)} gold files)")

        # Clone/checkout
        repo_dir = cache_dir / f"{repo.replace('/', '_')}_{commit[:8]}"
        if not repo_dir.exists():
            if not clone_at_commit(row["repo_url"], commit, repo_dir):
                results.append({
                    "instance_id": instance_id, "language": lang,
                    "status": "clone_failed", "r5": 0, "r10": 0, "p5": 0, "mrr": 0,
                })
                continue

        # Clear any previous index
        skel_dir = repo_dir / ".skelesearch"
        if skel_dir.exists():
            shutil.rmtree(skel_dir)

        # Run skelesearch
        t0 = time.time()
        retrieved = run_skelesearch(args.binary, repo_dir, query[:2000], provider=args.provider)  # cap query length
        elapsed = time.time() - t0

        r5 = recall_at_k(retrieved, gold_files, 5)
        r10 = recall_at_k(retrieved, gold_files, 10)
        p5 = precision_at_k(retrieved, gold_files, 5)
        m = mrr(retrieved, gold_files)

        result = {
            "instance_id": instance_id,
            "language": lang,
            "repo": repo,
            "status": "ok",
            "r5": round(r5, 4),
            "r10": round(r10, 4),
            "p5": round(p5, 4),
            "mrr": round(m, 4),
            "gold_files": gold_files,
            "retrieved_files": retrieved[:10],
            "elapsed_s": round(elapsed, 1),
        }
        results.append(result)
        lang_metrics[lang].append(result)

        print(f"  R@5={r5:.2f} R@10={r10:.2f} P@5={p5:.2f} MRR={m:.2f} ({elapsed:.1f}s)")

    # Aggregate
    print("\n=== AGGREGATE ===")
    all_ok = [r for r in results if r["status"] == "ok"]
    if all_ok:
        mean_r5 = sum(r["r5"] for r in all_ok) / len(all_ok)
        mean_r10 = sum(r["r10"] for r in all_ok) / len(all_ok)
        mean_p5 = sum(r["p5"] for r in all_ok) / len(all_ok)
        mean_mrr = sum(r["mrr"] for r in all_ok) / len(all_ok)
        print(f"Overall: R@5={mean_r5:.3f} R@10={mean_r10:.3f} P@5={mean_p5:.3f} MRR={mean_mrr:.3f} ({len(all_ok)} instances)")

        for lang in sorted(lang_metrics.keys()):
            ok = [r for r in lang_metrics[lang] if r["status"] == "ok"]
            if ok:
                lr5 = sum(r["r5"] for r in ok) / len(ok)
                lmrr = sum(r["mrr"] for r in ok) / len(ok)
                print(f"  {lang}: R@5={lr5:.3f} MRR={lmrr:.3f} ({len(ok)} instances)")

    failed = sum(1 for r in results if r["status"] != "ok")
    if failed:
        print(f"\n{failed} instances failed (clone/index/search)")

    # Write output
    output = {
        "benchmark": "ContextBench Verified",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tool": "skelesearch",
        "instances": len(results),
        "succeeded": len(all_ok),
        "failed": failed,
        "aggregate": {
            "mean_r5": round(mean_r5, 4) if all_ok else 0,
            "mean_r10": round(mean_r10, 4) if all_ok else 0,
            "mean_p5": round(mean_p5, 4) if all_ok else 0,
            "mean_mrr": round(mean_mrr, 4) if all_ok else 0,
        },
        "per_language": {
            lang: {
                "mean_r5": round(sum(r["r5"] for r in ok) / len(ok), 4),
                "mean_mrr": round(sum(r["mrr"] for r in ok) / len(ok), 4),
                "count": len(ok),
            }
            for lang in sorted(lang_metrics.keys())
            if (ok := [r for r in lang_metrics[lang] if r["status"] == "ok"])
        },
        "results": results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
