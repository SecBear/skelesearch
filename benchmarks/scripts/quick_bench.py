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

Performance notes:
  --cached-only skips git checkout and re-index entirely (fast path, ~2s/instance).
  --data-cache  caches the HuggingFace dataset locally as parquet (~14s first run,
                <1s thereafter).
  --skip-index  skip skelesearch index step (implied by --cached-only).
"""

import argparse
import json
import os
import random
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_contextbench(cache_file: Path | None = None) -> list[dict]:
    """Load ContextBench dataset, using local parquet cache if available.

    On first run with --data-cache, downloads from HuggingFace and saves to
    parquet. Subsequent runs load from disk (<1s vs ~14s cold).
    """
    if cache_file and cache_file.exists():
        t0 = time.time()
        try:
            import pandas as pd
            df = pd.read_parquet(cache_file)
            elapsed = time.time() - t0
            # Convert to list of dicts; parquet preserves types but gold_context
            # may be a list-column — keep as-is (JSON string fields preserved).
            instances = df.to_dict("records")
            print(f"loaded {len(instances)} instances from cache ({elapsed:.1f}s)")
            return instances
        except Exception as e:
            print(f"cache load failed ({e}), falling back to HuggingFace", file=sys.stderr)

    from datasets import load_dataset
    t0 = time.time()
    instances = list(load_dataset("Contextbench/ContextBench", "default", split="train"))
    elapsed = time.time() - t0
    print(f"loaded {len(instances)} instances from HuggingFace ({elapsed:.1f}s)")

    if cache_file:
        try:
            import pandas as pd
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(instances).to_parquet(cache_file, index=False)
            print(f"  cached to {cache_file}")
        except Exception as e:
            print(f"  cache write failed: {e}", file=sys.stderr)

    return instances


# ---------------------------------------------------------------------------
# Gold context extraction
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Repo management
# ---------------------------------------------------------------------------

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


def detect_provider(repo_dir: Path) -> str | None:
    """Infer the embedding provider from the existing .skelesearch index.

    Checks the dimension of stored embeddings:
      384  → fastembed (BAAI/bge-small-en-v1.5)
      1024 → voyage (voyage-code-3) or openai
      1536 → openai (text-embedding-3-small)

    Returns None if the index does not exist or is empty.
    """
    db_path = repo_dir / ".skelesearch" / "manifest.db"
    if not db_path.exists():
        return None
    try:
        conn = sqlite3.connect(str(db_path))
        row = conn.execute("SELECT dim FROM embedding_cache LIMIT 1").fetchone()
        conn.close()
        if row is None:
            return None
        dim = row[0]
        if dim == 384:
            return "fastembed"
        elif dim == 1024:
            return "voyage"
        elif dim == 1536:
            return "openai"
        else:
            return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    # Performance flags
    parser.add_argument("--data-cache", default=None, metavar="PATH",
                        help="Path to parquet cache for the dataset. "
                             "First run downloads and saves; subsequent runs load from disk. "
                             "Saves ~14s per invocation.")
    parser.add_argument("--skip-index", action="store_true", default=None,
                        help="Skip git checkout and skelesearch index steps. "
                             "Searches the repo at its current HEAD with the existing index. "
                             "Automatically enabled when --cached-only is set. "
                             "Cuts per-instance time from 26-300s to ~2s.")
    parser.add_argument("--auto-provider", action="store_true",
                        help="Auto-detect the embedding provider from the existing index. "
                             "Overrides --provider when --skip-index is active.")
    args = parser.parse_args()

    # --skip-index is implied by --cached-only (repos are already indexed at HEAD)
    if args.skip_index is None:
        args.skip_index = args.cached_only

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # --------------- Load dataset ---------------
    print("Loading ContextBench...", end=" ", flush=True)
    t_dataset_start = time.time()
    data_cache = Path(args.data_cache) if args.data_cache else None
    instances = load_contextbench(data_cache)
    t_dataset = time.time() - t_dataset_start

    repo_count = len(set(i["repo"] for i in instances))
    # (load_contextbench already prints instance count and timing)
    print(f"  {repo_count} repos, dataset load: {t_dataset:.1f}s")

    # --------------- Filter by language ---------------
    if args.lang:
        instances = [i for i in instances if i.get("language", "") == args.lang]
        print(f"Filtered to {args.lang}: {len(instances)} instances")

    # --------------- Find cached repos ---------------
    cached_repos: dict[str, Path] = {}
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

    # --------------- Sample ---------------
    if args.full:
        sample = instances
    else:
        rng = random.Random(args.seed)
        # Group by language, sample proportionally
        by_lang: dict[str, list] = {}
        for inst in instances:
            lang = inst.get("language", "unknown")
            by_lang.setdefault(lang, []).append(inst)

        sample = []
        per_lang = max(1, args.n // len(by_lang))
        for lang, lang_instances in sorted(by_lang.items()):
            n = min(per_lang, len(lang_instances))
            sample.extend(rng.sample(lang_instances, n))

        # Fill to reach N from largest pools
        while len(sample) < args.n and len(sample) < len(instances):
            remaining = [i for i in instances if i not in sample]
            if not remaining:
                break
            sample.append(rng.choice(remaining))

        sample = sample[:args.n]

    # Sort by repo for minimal checkout churn (matters when not skipping index)
    sample.sort(key=lambda i: (i["repo"], i.get("created_at", "")))

    lang_dist = Counter(i.get("language", "?") for i in sample)
    print(f"\nQUICK BENCH ({len(sample)} instances, seed={args.seed})")
    print(f"Languages: {dict(lang_dist)}")
    print(f"Provider: {args.provider}")
    if args.skip_index:
        print("Mode: skip-index (searching at HEAD with existing index)")
    print()

    # --------------- Per-instance evaluation ---------------
    results = []
    skip = 0
    total_search_ms = 0
    total_index_ms = 0
    total_checkout_ms = 0

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

        # --------------- Get or clone repo ---------------
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

        # --------------- Provider detection ---------------
        # When skipping index, the existing index may use a different provider
        # than --provider. Auto-detect if requested, or warn on mismatch.
        effective_provider = args.provider
        if args.skip_index and args.auto_provider:
            detected = detect_provider(repo_dir)
            if detected and detected != args.provider:
                effective_provider = detected
        elif args.skip_index:
            # Best-effort: detect silently to avoid dimension mismatch errors
            detected = detect_provider(repo_dir)
            if detected:
                effective_provider = detected

        # --------------- Checkout (skipped in fast path) ---------------
        if not args.skip_index:
            t0 = time.time()
            if not checkout_commit(repo_dir, commit):
                print(f"  [{idx+1}/{len(sample)}] SKIP {instance_id[:35]} (checkout failed)")
                skip += 1
                continue
            total_checkout_ms += (time.time() - t0) * 1000

        # --------------- Index (skipped in fast path) ---------------
        if not args.skip_index:
            env = os.environ.copy()
            env["RUST_LOG"] = "skelesearch=warn"
            t0 = time.time()
            try:
                subprocess.run(
                    [args.binary, "index", str(repo_dir), "--provider", args.provider],
                    capture_output=True, text=True, env=env, timeout=120,
                )
            except subprocess.TimeoutExpired:
                print(f"  [{idx+1}/{len(sample)}] SKIP {instance_id[:35]} (index timeout)")
                skip += 1
                continue
            total_index_ms += (time.time() - t0) * 1000

        # --------------- Search ---------------
        t0 = time.time()
        chunks = run_search(args.binary, repo_dir, query, effective_provider)
        search_ms = (time.time() - t0) * 1000
        total_search_ms += search_ms

        retrieved_files = dedup_files(chunks)
        r5 = recall_at_k(retrieved_files, gold_files, 5)
        a5 = acc_at_k(retrieved_files, gold_files, 5)
        m = mrr(retrieved_files, gold_files)
        ro = region_overlap(chunks, gold_regions)

        status = "✓" if r5 > 0 else "✗"
        index_note = "" if args.skip_index else f" idx={total_index_ms/max(idx+1-skip,1):.0f}ms"
        print(f"  [{idx+1}/{len(sample)}] {status} [{lang:>4}] R@5={r5:.0%} RO={ro:.0%} "
              f"MRR={m:.2f} srch={search_ms:.0f}ms{index_note}  {instance_id[:40]}")

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
    avg_search_ms = total_search_ms / total

    # Per-language breakdown
    by_lang: dict[str, list] = {}
    for r in results:
        by_lang.setdefault(r["language"], []).append(r)

    print(f"\n{'='*65}")
    print(f"RESULTS: {total} instances ({skip} skipped)")
    print(f"  R@5:            {avg_r5:.1%}")
    print(f"  Acc@5:          {avg_a5:.1%}")
    print(f"  MRR:            {avg_mrr:.3f}")
    print(f"  Region overlap: {avg_ro:.1%}")
    print(f"  Avg search:     {avg_search_ms:.0f}ms/query")

    # Timing breakdown
    total_wall = t_dataset + total_search_ms / 1000
    if not args.skip_index:
        total_wall += (total_checkout_ms + total_index_ms) / 1000
        print(f"  Avg checkout:   {total_checkout_ms/max(total,1):.0f}ms/instance")
        print(f"  Avg index:      {total_index_ms/max(total,1):.0f}ms/instance")
    print(f"  Dataset load:   {t_dataset:.1f}s")
    print(f"  Total search:   {total_search_ms/1000:.1f}s")
    print(f"  Est. total:     {total_search_ms/1000 + t_dataset:.1f}s")

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
