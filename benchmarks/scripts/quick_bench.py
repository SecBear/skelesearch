#!/usr/bin/env python3
"""Fast feedback from ContextBench — multi-language, research-grade ground truth.

Randomly samples N instances across 8 languages, clones/caches repos, runs
skelesearch search, reports results in ~10-20s (using pre-cached repos).

For coding agents: run this after changes to verify no regression.
Exit code 0 = pass, 1 = regression.

Usage (via quick-bench.sh wrapper):
    ./benchmarks/scripts/quick-bench.sh              # 10 random instances ~15s
    ./benchmarks/scripts/quick-bench.sh --n 20       # more instances
    ./benchmarks/scripts/quick-bench.sh --full        # all cached instances

Dataset: ContextBench (1,136 instances, 66 repos, 8 languages)
  - Python, JavaScript, TypeScript, Go, C, Rust, C++, Java
  - Human-annotated gold contexts with sub-file line ranges
  - Real GitHub issues as queries (forward-looking)
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path


def load_contextbench():
    """Load ContextBench dataset from HuggingFace."""
    from datasets import load_dataset
    return list(load_dataset("Contextbench/ContextBench", "default", split="train"))


def extract_gold_files(instance: dict) -> list[str]:
    """Extract gold file paths from the gold_context field."""
    gc = instance.get("gold_context", [])
    if isinstance(gc, str):
        gc = json.loads(gc)
    if not isinstance(gc, list):
        return []

    files = set()
    for entry in gc:
        if isinstance(entry, dict):
            fp = entry.get("file", "")
            # Strip workspace prefix: /workspace/Owner__Repo__0.1/path → path
            if "/workspace/" in fp:
                parts = fp.split("/", 3)
                if len(parts) >= 4:
                    fp = parts[3]
            if fp:
                files.add(fp)
    return sorted(files)


def extract_gold_regions(instance: dict) -> list[dict]:
    """Extract gold regions with sub-file line ranges."""
    gc = instance.get("gold_context", [])
    if isinstance(gc, str):
        gc = json.loads(gc)
    if not isinstance(gc, list):
        return []

    regions = []
    for entry in gc:
        if isinstance(entry, dict) and "start_line" in entry:
            fp = entry.get("file", "")
            if "/workspace/" in fp:
                parts = fp.split("/", 3)
                if len(parts) >= 4:
                    fp = parts[3]
            regions.append({
                "file": fp,
                "start_line": entry["start_line"],
                "end_line": entry["end_line"],
            })
    return regions


def clone_or_cache_repo(repo: str, cache_dir: Path, timeout: int = 300) -> Path | None:
    """Clone repo if not cached. Returns repo dir path."""
    repo_dir_name = repo.replace("/", "_")
    repo_dir = cache_dir / repo_dir_name
    if (repo_dir / ".git").exists():
        return repo_dir

    repo_url = f"https://github.com/{repo}.git"
    try:
        subprocess.run(
            ["git", "clone", "--quiet", repo_url, str(repo_dir)],
            capture_output=True, text=True, timeout=timeout, check=True,
        )
        return repo_dir
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"  Clone failed for {repo}: {e}", file=sys.stderr)
        return None


def checkout_commit(repo_dir: Path, commit: str) -> bool:
    """Checkout a specific commit, preserving .skelesearch cache."""
    try:
        subprocess.run(
            ["git", "checkout", "--force", "--quiet", commit],
            capture_output=True, text=True, timeout=30, check=True,
            cwd=str(repo_dir),
        )
        subprocess.run(
            ["git", "clean", "-fdxq", "--exclude=.skelesearch"],
            capture_output=True, text=True, timeout=30, check=True,
            cwd=str(repo_dir),
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def run_search(binary: str, repo_dir: Path, query: str,
               provider: str, top_k: int = 10) -> list[dict]:
    """Search and return raw chunk results."""
    env = os.environ.copy()
    env["RUST_LOG"] = "skelesearch=warn"

    try:
        result = subprocess.run(
            [binary, "search", query[:4000], "--provider", provider,
             "--top-k", str(top_k), "--json"],
            capture_output=True, text=True, timeout=30,
            cwd=str(repo_dir), env=env,
        )
    except subprocess.TimeoutExpired:
        return []

    if result.returncode != 0:
        return []

    try:
        data = json.loads(result.stdout)
        return data.get("results", data) if isinstance(data, dict) else data
    except json.JSONDecodeError:
        return []


def dedup_files(chunks: list[dict]) -> list[str]:
    """Deduplicate chunk results to file list."""
    seen = set()
    files = []
    for c in chunks:
        fp = c.get("file_path", c.get("file", ""))
        if fp and fp not in seen:
            seen.add(fp)
            files.append(fp)
    return files


def region_overlap(chunks: list[dict], gold_regions: list[dict]) -> float:
    """Fraction of gold regions that overlap with any returned chunk."""
    if not gold_regions:
        return 1.0

    hits = 0
    for region in gold_regions:
        gold_file = region["file"]
        gold_start = region["start_line"]
        gold_end = region["end_line"]

        for chunk in chunks:
            chunk_file = chunk.get("file_path", "")
            chunk_start = chunk.get("start_line", 0)
            chunk_end = chunk.get("end_line", 0)

            if chunk_file == gold_file:
                # Check line range overlap
                if chunk_start <= gold_end and chunk_end >= gold_start:
                    hits += 1
                    break

    return hits / len(gold_regions)


def acc_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    return 1.0 if all(g in set(retrieved[:k]) for g in gold) else 0.0


def recall_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return 1.0
    return sum(1 for g in gold if g in set(retrieved[:k])) / len(gold)


def mrr(retrieved: list[str], gold: list[str]) -> float:
    for i, fp in enumerate(retrieved):
        if fp in gold:
            return 1.0 / (i + 1)
    return 0.0


def main():
    parser = argparse.ArgumentParser(description="Quick ContextBench feedback")
    parser.add_argument("--binary", required=True)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--provider", default="voyage")
    parser.add_argument("--cache-dir", default="benchmarks/swebench-repos")
    parser.add_argument("--threshold", type=float, default=30,
                        help="Minimum R@5 (%) to pass")
    parser.add_argument("--cached-only", action="store_true",
                        help="Only use already-cached repos (no cloning)")
    parser.add_argument("--lang", default=None,
                        help="Filter to specific language")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("Loading ContextBench...", end=" ", flush=True)
    instances = load_contextbench()
    print(f"{len(instances)} instances, {len(set(i['repo'] for i in instances))} repos")

    # Filter by language if specified
    if args.lang:
        instances = [i for i in instances if i.get("language", "") == args.lang]
        print(f"Filtered to {args.lang}: {len(instances)} instances")

    # Find cached repos
    cached_repos = {}
    for d in cache_dir.iterdir():
        if d.is_dir() and (d / ".git").exists():
            repo_name = d.name.replace("_", "/", 1)
            cached_repos[repo_name] = d

    if args.cached_only:
        instances = [i for i in instances if i["repo"] in cached_repos]
        print(f"Cached repos available: {len(cached_repos)}, matching instances: {len(instances)}")
    else:
        print(f"Cached repos: {len(cached_repos)}")

    if not instances:
        print("No instances available", file=sys.stderr)
        sys.exit(2)

    # Sample — stratify by language for diversity
    if args.full:
        sample = instances
    else:
        rng = random.Random(args.seed)
        # Group by language, sample proportionally
        by_lang = {}
        for inst in instances:
            lang = inst.get("language", "unknown")
            by_lang.setdefault(lang, []).append(inst)

        sample = []
        per_lang = max(1, args.n // len(by_lang))
        for lang, lang_instances in sorted(by_lang.items()):
            n = min(per_lang, len(lang_instances))
            sample.extend(rng.sample(lang_instances, n))

        # If we need more to reach N, fill from largest pools
        while len(sample) < args.n and len(sample) < len(instances):
            remaining = [i for i in instances if i not in sample]
            if not remaining:
                break
            sample.append(rng.choice(remaining))

        sample = sample[:args.n]

    # Sort by repo for minimal checkout churn
    sample.sort(key=lambda i: (i["repo"], i.get("created_at", "")))

    lang_dist = Counter(i.get("language", "?") for i in sample)
    print(f"\nQUICK BENCH ({len(sample)} instances, seed={args.seed})")
    print(f"Languages: {dict(lang_dist)}")
    print(f"Provider: {args.provider}")
    print()

    results = []
    skip = 0
    total_ms = 0

    for idx, inst in enumerate(sample):
        instance_id = inst["instance_id"]
        repo = inst["repo"]
        commit = inst["base_commit"]
        lang = inst.get("language", "?")
        query = inst["problem_statement"][:4000]
        gold_files = extract_gold_files(inst)
        gold_regions = extract_gold_regions(inst)

        if not gold_files:
            skip += 1
            continue

        # Get or clone repo
        if repo in cached_repos:
            repo_dir = cached_repos[repo]
        elif not args.cached_only:
            repo_dir = clone_or_cache_repo(repo, cache_dir)
            if repo_dir:
                cached_repos[repo] = repo_dir
            else:
                skip += 1
                continue
        else:
            skip += 1
            continue

        # Checkout
        if not checkout_commit(repo_dir, commit):
            print(f"  [{idx+1}/{len(sample)}] SKIP {instance_id[:35]} (checkout failed)")
            skip += 1
            continue

        # Index (incremental, fast after first run)
        env = os.environ.copy()
        env["RUST_LOG"] = "skelesearch=warn"
        try:
            subprocess.run(
                [args.binary, "index", str(repo_dir), "--provider", args.provider],
                capture_output=True, text=True, env=env, timeout=120,
            )
        except subprocess.TimeoutExpired:
            print(f"  [{idx+1}/{len(sample)}] SKIP {instance_id[:35]} (index timeout)")
            skip += 1
            continue

        # Search
        t0 = time.time()
        chunks = run_search(args.binary, repo_dir, query, args.provider)
        search_ms = (time.time() - t0) * 1000
        total_ms += search_ms

        retrieved_files = dedup_files(chunks)
        r5 = recall_at_k(retrieved_files, gold_files, 5)
        a5 = acc_at_k(retrieved_files, gold_files, 5)
        m = mrr(retrieved_files, gold_files)
        ro = region_overlap(chunks, gold_regions)

        status = "✓" if r5 > 0 else "✗"
        print(f"  [{idx+1}/{len(sample)}] {status} [{lang:>4}] R@5={r5:.0%} RO={ro:.0%} MRR={m:.2f} {search_ms:.0f}ms  {instance_id[:40]}")

        results.append({
            "instance_id": instance_id,
            "language": lang,
            "repo": repo,
            "recall5": r5,
            "acc5": a5,
            "mrr": m,
            "region_overlap": ro,
            "search_ms": search_ms,
            "gold_files": len(gold_files),
            "gold_regions": len(gold_regions),
        })

    total = len(results)
    if total == 0:
        print("\nNo instances evaluated")
        sys.exit(2)

    avg_r5 = sum(r["recall5"] for r in results) / total
    avg_a5 = sum(r["acc5"] for r in results) / total
    avg_mrr = sum(r["mrr"] for r in results) / total
    avg_ro = sum(r["region_overlap"] for r in results) / total
    avg_ms = total_ms / total

    # Per-language breakdown
    by_lang = {}
    for r in results:
        by_lang.setdefault(r["language"], []).append(r)

    print(f"\n{'='*65}")
    print(f"RESULTS: {total} instances ({skip} skipped)")
    print(f"  R@5:            {avg_r5:.1%}")
    print(f"  Acc@5:          {avg_a5:.1%}")
    print(f"  MRR:            {avg_mrr:.3f}")
    print(f"  Region overlap: {avg_ro:.1%}")
    print(f"  Avg latency:    {avg_ms:.0f}ms/query")
    print(f"  Total time:     {total_ms/1000:.1f}s")

    if len(by_lang) > 1:
        print(f"\n  Per-language R@5:")
        for lang in sorted(by_lang.keys()):
            lr = by_lang[lang]
            lr5 = sum(r["recall5"] for r in lr) / len(lr)
            print(f"    {lang:>12}: {lr5:.0%} ({len(lr)} instances)")

    passed = avg_r5 * 100 >= args.threshold
    print(f"\n  {'PASS' if passed else 'FAIL'} (R@5={avg_r5:.1%}, threshold={args.threshold:.0f}%)")

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
