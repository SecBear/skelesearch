#!/usr/bin/env python3
"""SWE-bench Lite file localization evaluation for skelesearch.

Evaluates skelesearch's ability to localize which files need modification
given a GitHub issue description. Uses SWE-bench Lite (300 instances, Python).

Usage:
    uv run --with datasets --with huggingface_hub \
        python3 benchmarks/scripts/swebench-eval.py \
        --binary ./target/release/skelesearch \
        --provider fastembed \
        --output benchmarks/runs/swebench-fastembed.json

Requires:
    - datasets (HuggingFace)
    - A built skelesearch binary
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path


def load_swebench_lite():
    """Load SWE-bench Lite from HuggingFace."""
    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    return list(ds)


def extract_gold_files(instance: dict) -> list[str]:
    """Extract files modified in the reference patch."""
    patch = instance.get("patch", "")
    files = set()
    for line in patch.split("\n"):
        if line.startswith("diff --git a/"):
            # diff --git a/foo/bar.py b/foo/bar.py
            parts = line.split(" ")
            if len(parts) >= 4:
                # Take the b/ path (destination)
                path = parts[3]
                if path.startswith("b/"):
                    path = path[2:]
                files.add(path)
    return sorted(files)


def clone_at_commit(repo: str, commit: str, target_dir: Path, timeout: int = 300) -> bool:
    """Shallow clone a repo at a specific commit."""
    try:
        # Clone with depth 1, then fetch the specific commit
        repo_url = f"https://github.com/{repo}.git"
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(target_dir)],
            capture_output=True, text=True, timeout=timeout, check=True,
        )
        subprocess.run(
            ["git", "fetch", "--depth", "1", "origin", commit],
            capture_output=True, text=True, timeout=timeout, check=True,
            cwd=str(target_dir),
        )
        subprocess.run(
            ["git", "checkout", commit],
            capture_output=True, text=True, timeout=60, check=True,
            cwd=str(target_dir),
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"  Clone failed: {e}", file=sys.stderr)
        return False


def run_skelesearch(binary: str, project_dir: Path, query: str,
                    provider: str = "fastembed", top_k: int = 10,
                    index_timeout: int = 900, max_files: int = 20000) -> list[dict]:
    """Index and search with skelesearch. Returns list of result dicts."""
    env = os.environ.copy()
    env["RUST_LOG"] = "skelesearch=info"
    # Unset cloud reranker keys for pure local evaluation
    for key in ["VOYAGE_API_KEY", "JINA_API_KEY", "COHERE_API_KEY"]:
        env.pop(key, None)

    skel_dir = project_dir / ".skelesearch"
    if not skel_dir.exists():
        # Count files
        file_count = sum(1 for _ in project_dir.rglob("*.py"))
        if file_count > max_files:
            print(f"  SKIP: {file_count} Python files exceeds limit of {max_files}", file=sys.stderr)
            return []

        try:
            subprocess.run(
                [binary, "index", str(project_dir), "--provider", provider],
                capture_output=True, text=True, env=env, timeout=index_timeout,
            )
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT: indexing exceeded {index_timeout}s", file=sys.stderr)
            return []

    # Search
    try:
        result = subprocess.run(
            [binary, "search", query[:4000], "--provider", provider,
             "--top-k", str(top_k), "--json"],
            capture_output=True, text=True, env=env, timeout=120,
            cwd=str(project_dir),
        )
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: search exceeded 120s", file=sys.stderr)
        return []

    if result.returncode != 0:
        print(f"  Search error: {result.stderr[:200]}", file=sys.stderr)
        return []

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return []


def acc_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    """Strict Acc@k: 1.0 if ALL gold files appear in top-k, else 0.0."""
    top_k_set = set(retrieved[:k])
    return 1.0 if all(g in top_k_set for g in gold) else 0.0


def recall_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    """Fraction of gold files that appear in top-k."""
    if not gold:
        return 1.0
    top_k_set = set(retrieved[:k])
    return sum(1 for g in gold if g in top_k_set) / len(gold)


def mrr(retrieved: list[str], gold: list[str]) -> float:
    """Mean Reciprocal Rank of first gold file."""
    for i, f in enumerate(retrieved):
        if f in gold:
            return 1.0 / (i + 1)
    return 0.0


def main():
    parser = argparse.ArgumentParser(description="SWE-bench Lite file localization eval")
    parser.add_argument("--binary", required=True)
    parser.add_argument("--provider", default="fastembed")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None, help="Max instances to evaluate")
    parser.add_argument("--index-timeout", type=int, default=900)
    parser.add_argument("--max-files", type=int, default=20000)
    args = parser.parse_args()
    args.binary = str(Path(args.binary).resolve())

    instances = load_swebench_lite()
    if args.limit:
        instances = instances[:args.limit]
    print(f"Loaded {len(instances)} SWE-bench Lite instances")

    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(tempfile.mkdtemp(prefix="swebench_"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Cache dir: {cache_dir}")

    results = []
    # Group by repo for index reuse
    repo_groups = defaultdict(list)
    for inst in instances:
        repo_groups[inst["repo"]].append(inst)

    completed = 0
    skipped = 0

    for repo, repo_instances in repo_groups.items():
        print(f"\n=== {repo} ({len(repo_instances)} instances) ===")

        for inst in repo_instances:
            instance_id = inst["instance_id"]
            commit = inst["base_commit"]
            query = inst["problem_statement"]
            gold = extract_gold_files(inst)

            if not gold:
                print(f"  [{instance_id}] No gold files in patch, skipping")
                skipped += 1
                continue

            print(f"  [{instance_id}] {len(gold)} gold files")

            # Clone at the base commit
            repo_dir = cache_dir / f"{repo.replace('/', '_')}_{commit[:8]}"
            if not repo_dir.exists():
                if not clone_at_commit(repo, commit, repo_dir):
                    results.append({"instance_id": instance_id, "status": "clone_failed",
                                    "acc1": 0, "acc3": 0, "acc5": 0, "recall5": 0, "mrr": 0})
                    skipped += 1
                    continue

            # Run skelesearch
            t0 = time.time()
            chunks = run_skelesearch(
                args.binary, repo_dir, query[:4000],
                provider=args.provider, top_k=args.top_k,
                index_timeout=args.index_timeout, max_files=args.max_files,
            )
            elapsed = time.time() - t0

            if not chunks:
                results.append({"instance_id": instance_id, "status": "skipped",
                                "acc1": 0, "acc3": 0, "acc5": 0, "recall5": 0, "mrr": 0})
                skipped += 1
                continue

            # Deduplicate retrieved files (multiple chunks from same file)
            seen = set()
            retrieved_files = []
            for c in chunks:
                fp = c.get("file_path", c.get("file", ""))
                if fp and fp not in seen:
                    seen.add(fp)
                    retrieved_files.append(fp)

            # Compute metrics
            a1 = acc_at_k(retrieved_files, gold, 1)
            a3 = acc_at_k(retrieved_files, gold, 3)
            a5 = acc_at_k(retrieved_files, gold, 5)
            r5 = recall_at_k(retrieved_files, gold, 5)
            m = mrr(retrieved_files, gold)

            results.append({
                "instance_id": instance_id,
                "repo": repo,
                "status": "ok",
                "gold_files": gold,
                "retrieved_files": retrieved_files[:10],
                "acc1": a1, "acc3": a3, "acc5": a5,
                "recall5": r5, "mrr": m,
                "elapsed_s": round(elapsed, 1),
                "n_gold": len(gold),
            })
            completed += 1

            # Stream progress
            print(f"    Acc@1={a1:.0f} Acc@5={a5:.0f} R@5={r5:.1%} MRR={m:.3f} ({elapsed:.0f}s)")

    # Aggregate
    ok_results = [r for r in results if r["status"] == "ok"]
    if ok_results:
        avg_acc1 = sum(r["acc1"] for r in ok_results) / len(ok_results)
        avg_acc3 = sum(r["acc3"] for r in ok_results) / len(ok_results)
        avg_acc5 = sum(r["acc5"] for r in ok_results) / len(ok_results)
        avg_r5 = sum(r["recall5"] for r in ok_results) / len(ok_results)
        avg_mrr = sum(r["mrr"] for r in ok_results) / len(ok_results)
    else:
        avg_acc1 = avg_acc3 = avg_acc5 = avg_r5 = avg_mrr = 0.0

    summary = {
        "benchmark": "SWE-bench Lite file localization",
        "provider": args.provider,
        "total_instances": len(instances),
        "completed": completed,
        "skipped": skipped,
        "acc_at_1": round(avg_acc1 * 100, 1),
        "acc_at_3": round(avg_acc3 * 100, 1),
        "acc_at_5": round(avg_acc5 * 100, 1),
        "recall_at_5": round(avg_r5 * 100, 1),
        "mrr": round(avg_mrr, 3),
        "results": results,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"SWE-bench Lite File Localization Results")
    print(f"{'='*60}")
    print(f"Instances: {completed} completed, {skipped} skipped")
    print(f"Acc@1: {avg_acc1*100:.1f}%")
    print(f"Acc@3: {avg_acc3*100:.1f}%")
    print(f"Acc@5: {avg_acc5*100:.1f}%")
    print(f"Recall@5: {avg_r5*100:.1f}%")
    print(f"MRR: {avg_mrr:.3f}")
    print(f"\nSOTA reference: SweRank Acc@1=83.2%, Acc@5=96.0%")
    print(f"BM25 baseline: ~20-30%")
    print(f"\nResults saved to {output}")


if __name__ == "__main__":
    main()
