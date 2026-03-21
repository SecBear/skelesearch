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
from collections import defaultdict
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


def extract_gold_spans(row) -> list[dict]:
    """Extract gold spans (file + line ranges) from gold_context."""
    gc = row["gold_context"]
    if isinstance(gc, str):
        gc = json.loads(gc)
    return [
        {
            "file": e["file"],
            "start_line": e["start_line"],
            "end_line": e["end_line"],
        }
        for e in gc
    ]


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


def count_source_files(project_dir: Path) -> int:
    """Quick count of source files in a repo (no hidden dirs, no vendor)."""
    count = 0
    skip_dirs = {'.git', 'node_modules', 'vendor', '.venv', '__pycache__', 'build', 'dist'}
    for root, dirs, files in os.walk(project_dir):
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith('.')]
        count += len(files)
    return count


def run_skelesearch(
    binary: str,
    project_dir: Path,
    query: str,
    provider: str = "fastembed",
    top_k: int = 10,
    use_cache: bool = True,
    index_timeout: int = 3600,
    max_files: int = 50000,
) -> list[dict]:
    """Index (if needed) and search with skelesearch.

    Returns list of dicts with file, start_line, end_line per retrieved chunk.
    Caching: if .skelesearch/ already exists and use_cache is True, skip indexing.
    Repos with more than max_files source files are skipped (too large for CPU indexing).
    """
    env = os.environ.copy()
    # Enable tracing output from skelesearch
    env.setdefault("RUST_LOG", "skelesearch=info")
    skel_dir = project_dir / ".skelesearch"

    if not use_cache or not skel_dir.exists():
        # Check repo size before indexing
        file_count = count_source_files(project_dir)
        if file_count > max_files:
            print(f"  SKIP: {file_count} files exceeds limit of {max_files}", file=sys.stderr)
            return []
        print(f"  Indexing {file_count} files...")

        # Need to index — wipe any stale state first
        if skel_dir.exists():
            shutil.rmtree(skel_dir)

        # Stream output so we can see progress and diagnose hangs
        idx_proc = subprocess.Popen(
            [binary, "index", str(project_dir), "--provider", provider],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=env, text=True,
        )
        try:
            stdout, _ = idx_proc.communicate(timeout=index_timeout)
            if idx_proc.returncode != 0:
                print(f"  WARN: index failed (exit {idx_proc.returncode}): {stdout[:300]}", file=sys.stderr)
                return []
            # Print last line of output (summary)
            last_line = stdout.strip().split('\n')[-1] if stdout.strip() else ''
            if last_line:
                print(f"  Index: {last_line}")
        except subprocess.TimeoutExpired:
            idx_proc.kill()
            idx_proc.wait()
            print(f"  WARN: index timed out after {index_timeout}s", file=sys.stderr)
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
            return [
                {
                    "file": r.get("file_path", r.get("file", "")),
                    "start_line": r.get("start_line", 0),
                    "end_line": r.get("end_line", 0),
                }
                for r in results
            ]
        return []
    except json.JSONDecodeError:
        return []


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _file_match(retrieved: str, gold: str) -> bool:
    """Fuzzy path match: one path is a suffix of the other."""
    return retrieved.endswith(gold) or gold.endswith(retrieved)


def recall_at_k(retrieved_files: list[str], gold_files: list[str], k: int) -> float:
    if not gold_files:
        return 1.0
    top_k = retrieved_files[:k]
    hits = sum(1 for g in gold_files if any(_file_match(r, g) for r in top_k))
    return hits / len(gold_files)


def precision_at_k(retrieved_files: list[str], gold_files: list[str], k: int) -> float:
    top_k = retrieved_files[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for r in top_k if any(_file_match(r, g) for g in gold_files))
    return hits / len(top_k)


def mrr(retrieved_files: list[str], gold_files: list[str]) -> float:
    for i, r in enumerate(retrieved_files):
        if any(_file_match(r, g) for g in gold_files):
            return 1.0 / (i + 1)
    return 0.0


def file_f1(retrieved_files: list[str], gold_files: list[str]) -> tuple[float, float, float]:
    """Return (precision, recall, f1) at the file level."""
    if not retrieved_files and not gold_files:
        return 1.0, 1.0, 1.0
    if not retrieved_files:
        return 0.0, 0.0, 0.0
    if not gold_files:
        return 0.0, 1.0, 0.0  # retrieved something but nothing expected

    tp_r = sum(1 for r in retrieved_files if any(_file_match(r, g) for g in gold_files))
    tp_g = sum(1 for g in gold_files if any(_file_match(r, g) for r in retrieved_files))

    precision = tp_r / len(retrieved_files)
    recall = tp_g / len(gold_files)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def line_overlap(
    retrieved_spans: list[dict],
    gold_spans: list[dict],
) -> tuple[float, float, float]:
    """Compute line-level precision, recall, F1 across all files.

    Each span is {file, start_line, end_line}.  We build sets of (file, line_num)
    pairs and compute standard P/R/F1 over those sets.
    """
    retrieved_lines: set[tuple[str, int]] = set()
    for s in retrieved_spans:
        f = s["file"]
        for ln in range(s["start_line"], s["end_line"] + 1):
            retrieved_lines.add((f, ln))

    gold_lines: set[tuple[str, int]] = set()
    for s in gold_spans:
        f = s["file"]
        for ln in range(s["start_line"], s["end_line"] + 1):
            gold_lines.add((f, ln))

    if not retrieved_lines and not gold_lines:
        return 1.0, 1.0, 1.0
    if not retrieved_lines:
        return 0.0, 0.0, 0.0
    if not gold_lines:
        return 0.0, 1.0, 0.0

    tp = len(retrieved_lines & gold_lines)
    precision = tp / len(retrieved_lines)
    recall = tp / len(gold_lines)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run skelesearch against ContextBench")
    parser.add_argument("--binary", required=True, help="Path to skelesearch binary")
    parser.add_argument("--languages", default="python,typescript,go,rust,javascript")
    parser.add_argument("--limit-per-lang", type=int, default=10)
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--cache-dir", default=None, help="Dir to cache repo clones")
    parser.add_argument("--provider", default="fastembed", help="Embedding provider")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to retrieve (default: 10)")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force re-index even if .skelesearch/ already exists",
    )
    parser.add_argument(
        "--index-timeout",
        type=int,
        default=3600,
        help="Timeout in seconds for indexing a repo (default: 3600)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=50000,
        help="Skip repos with more source files than this (default: 50000)",
    )
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
        gold_spans = extract_gold_spans(row)

        print(f"[{i+1}/{len(instances)}] {instance_id} ({lang}, {len(gold_files)} gold files)")

        # Clone/checkout
        repo_dir = cache_dir / f"{repo.replace('/', '_')}_{commit[:8]}"
        if not repo_dir.exists():
            if not clone_at_commit(row["repo_url"], commit, repo_dir):
                results.append({
                    "instance_id": instance_id, "language": lang,
                    "status": "clone_failed",
                    "r5": 0, "r10": 0, "p5": 0, "mrr": 0,
                    "file_precision": 0, "file_recall": 0, "file_f1": 0,
                    "line_precision": 0, "line_recall": 0, "line_f1": 0,
                })
                continue

        # Determine whether we have a warm index cache
        skel_dir = repo_dir / ".skelesearch"
        cache_hit = skel_dir.exists() and not args.no_cache
        if cache_hit:
            print(f"  Using cached index for {repo}@{commit[:8]}")

        # Run skelesearch (index + search, or search-only on cache hit)
        t0 = time.time()
        chunks = run_skelesearch(
            args.binary,
            repo_dir,
            query[:2000],  # cap query length
            provider=args.provider,
            top_k=args.top_k,
            use_cache=not args.no_cache,
            index_timeout=args.index_timeout,
            max_files=args.max_files,
        )
        elapsed = time.time() - t0

        retrieved_files = [c["file"] for c in chunks]

        # Legacy metrics (preserved)
        r5 = recall_at_k(retrieved_files, gold_files, 5)
        r10 = recall_at_k(retrieved_files, gold_files, 10)
        p5 = precision_at_k(retrieved_files, gold_files, 5)
        m = mrr(retrieved_files, gold_files)

        # New metrics
        fp, fr, ff1 = file_f1(retrieved_files, gold_files)
        lp, lr, lf1 = line_overlap(chunks, gold_spans)

        print(
            f"  File: P={fp:.2f} R={fr:.2f} F1={ff1:.2f} | "
            f"Line: P={lp:.2f} R={lr:.2f} F1={lf1:.2f} | ({elapsed:.1f}s)"
        )

        result = {
            "instance_id": instance_id,
            "language": lang,
            "repo": repo,
            "status": "ok",
            # Legacy metrics
            "r5": round(r5, 4),
            "r10": round(r10, 4),
            "p5": round(p5, 4),
            "mrr": round(m, 4),
            # File-level F1
            "file_precision": round(fp, 4),
            "file_recall": round(fr, 4),
            "file_f1": round(ff1, 4),
            # Line-level F1
            "line_precision": round(lp, 4),
            "line_recall": round(lr, 4),
            "line_f1": round(lf1, 4),
            # Provenance
            "gold_files": gold_files,
            "retrieved_files": retrieved_files[:args.top_k],
            "elapsed_s": round(elapsed, 1),
        }
        results.append(result)
        lang_metrics[lang].append(result)

    # Aggregate
    print("\n=== AGGREGATE ===")
    all_ok = [r for r in results if r["status"] == "ok"]
    failed = sum(1 for r in results if r["status"] != "ok")

    aggregate: dict = {}
    if all_ok:
        mean_r5 = safe_mean([r["r5"] for r in all_ok])
        mean_r10 = safe_mean([r["r10"] for r in all_ok])
        mean_p5 = safe_mean([r["p5"] for r in all_ok])
        mean_mrr = safe_mean([r["mrr"] for r in all_ok])
        mean_file_f1 = safe_mean([r["file_f1"] for r in all_ok])
        mean_line_precision = safe_mean([r["line_precision"] for r in all_ok])
        mean_line_recall = safe_mean([r["line_recall"] for r in all_ok])
        mean_line_f1 = safe_mean([r["line_f1"] for r in all_ok])

        print(
            f"Overall: R@5={mean_r5:.3f} R@10={mean_r10:.3f} P@5={mean_p5:.3f} MRR={mean_mrr:.3f} "
            f"FileF1={mean_file_f1:.3f} LineF1={mean_line_f1:.3f} ({len(all_ok)} instances)"
        )

        for lang in sorted(lang_metrics.keys()):
            ok = [r for r in lang_metrics[lang] if r["status"] == "ok"]
            if ok:
                lr5 = safe_mean([r["r5"] for r in ok])
                lmrr = safe_mean([r["mrr"] for r in ok])
                lff1 = safe_mean([r["file_f1"] for r in ok])
                llf1 = safe_mean([r["line_f1"] for r in ok])
                print(
                    f"  {lang}: R@5={lr5:.3f} MRR={lmrr:.3f} "
                    f"FileF1={lff1:.3f} LineF1={llf1:.3f} ({len(ok)} instances)"
                )

        aggregate = {
            "mean_r5": round(mean_r5, 4),
            "mean_r10": round(mean_r10, 4),
            "mean_p5": round(mean_p5, 4),
            "mean_mrr": round(mean_mrr, 4),
            "mean_file_f1": round(mean_file_f1, 4),
            "mean_line_precision": round(mean_line_precision, 4),
            "mean_line_recall": round(mean_line_recall, 4),
            "mean_line_f1": round(mean_line_f1, 4),
        }

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
        "aggregate": aggregate,
        "per_language": {
            lang: {
                "mean_r5": round(safe_mean([r["r5"] for r in ok]), 4),
                "mean_mrr": round(safe_mean([r["mrr"] for r in ok]), 4),
                "mean_file_f1": round(safe_mean([r["file_f1"] for r in ok]), 4),
                "mean_line_f1": round(safe_mean([r["line_f1"] for r in ok]), 4),
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
