#!/usr/bin/env python3
# uv run --with openai python3 benchmarks/scripts/audit-annotations.py
"""Audit eval case annotations for completeness using an LLM judge.

Detects cases where skelesearch returns correct files that are scored as
misses purely because `expected_files` annotations are incomplete.

Usage:
    uv run --with openai python3 benchmarks/scripts/audit-annotations.py \\
      --run benchmarks/runs/mini-redis-voyage-full.json \\
      --cases benchmarks/cases/rust/mini-redis.json \\
      --repos-dir benchmarks/repos \\
      --output benchmarks/audit-results.json

Required env vars:
    OPENAI_API_KEY
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Audit eval annotation completeness via LLM judge.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--run", required=True, metavar="PATH",
                   help="Path to the skelesearch run results JSON.")
    p.add_argument("--cases", required=True, metavar="PATH",
                   help="Path to the eval cases JSON (expected_files ground truth).")
    p.add_argument("--repos-dir", required=True, metavar="DIR",
                   help="Base directory containing cloned repos (e.g. benchmarks/repos/).")
    p.add_argument("--output", required=True, metavar="PATH",
                   help="Destination path for the audit report JSON.")
    p.add_argument("--top-k", type=int, default=3, metavar="N",
                   help="Number of top-ranked returned files to audit per case (default: 3).")
    p.add_argument("--model", default="gpt-4o-mini",
                   help="OpenAI model to use for judging (default: gpt-4o-mini).")
    p.add_argument("--file-lines", type=int, default=80,
                   help="Max lines of each candidate file to send to the LLM (default: 80).")
    p.add_argument("--sleep", type=float, default=0.5,
                   help="Seconds to sleep between LLM calls to avoid rate limits (default: 0.5).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_run(path: str) -> dict:
    """Load a skelesearch run JSON produced by run-eval.ts."""
    with open(path) as f:
        data = json.load(f)
    # Normalise: run-eval.ts uses 'cases'; contextbench uses 'results'.
    if "cases" not in data and "results" in data:
        data["cases"] = data["results"]
    if "cases" not in data:
        raise ValueError(f"Run JSON {path!r} has neither 'cases' nor 'results' key.")
    return data


def load_cases(path: str) -> dict[str, list[str]]:
    """Return a mapping of query -> expected_files from the eval cases JSON."""
    with open(path) as f:
        cases = json.load(f)
    mapping: dict[str, list[str]] = {}
    for case in cases:
        q = case.get("query", "")
        if q:
            mapping[q] = case.get("expected_files", [])
    return mapping


# ---------------------------------------------------------------------------
# File reading
# ---------------------------------------------------------------------------

def read_file_head(repo_dir: Path, rel_path: str, max_lines: int) -> str | None:
    """Read up to max_lines lines from a file inside repo_dir.

    Returns None when the file does not exist (caller should note the gap).
    """
    target = repo_dir / rel_path
    if not target.exists():
        return None
    lines: list[str] = []
    try:
        with open(target, errors="replace") as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                lines.append(line)
    except OSError:
        return None
    return "".join(lines)


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

def ask_llm(client, model: str, query: str, file_path: str, content: str) -> tuple[str, str]:
    """Ask the LLM whether a file is relevant to a query.

    Returns (verdict, reason) where verdict is 'YES' or 'NO'.
    Raises on unrecoverable API errors.
    """
    prompt = (
        f"Given the query '{query}', is the file '{file_path}' relevant?\n\n"
        f"Here is the file content (first {len(content.splitlines())} lines):\n"
        f"```\n{content}\n```\n\n"
        "Answer YES or NO with a one-line reason."
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a code-relevance judge. "
                    "Assess whether a source file is relevant to a given code search query. "
                    "Be concise: answer YES or NO on the first line, then one sentence of reasoning."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=80,
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    # Parse: first token is YES/NO, rest is the reason.
    first_line = raw.split("\n")[0].strip()
    verdict = "YES" if first_line.upper().startswith("YES") else "NO"
    # Reason: everything after the first word on the first line, or second line.
    parts = first_line.split(None, 1)
    reason = parts[1].strip(" :-") if len(parts) > 1 else ""
    if not reason and "\n" in raw:
        reason = raw.split("\n", 1)[1].strip()
    return verdict, reason


# ---------------------------------------------------------------------------
# Core audit logic
# ---------------------------------------------------------------------------

def audit(
    run_data: dict,
    case_map: dict[str, list[str]],
    repos_dir: Path,
    client,
    model: str,
    top_k: int,
    file_lines: int,
    sleep_s: float,
) -> dict:
    """Run the full audit, returning the report dict."""
    # Derive repo_id from the run metadata; fall back to repos_dir children.
    repo_id: str | None = run_data.get("repo_id")
    # repo_path in the run JSON is an absolute path; extract just the name.
    repo_path_field: str | None = run_data.get("repo_path")
    if repo_id is None and repo_path_field:
        repo_id = Path(repo_path_field).name

    if repo_id and (repos_dir / repo_id).is_dir():
        repo_dir = repos_dir / repo_id
    else:
        # Best-effort: use repos_dir itself if it looks like a single repo.
        repo_dir = repos_dir

    cases = run_data["cases"]

    total_cases = len(cases)
    cases_with_disagreements = 0
    total_disagreements = 0
    annotation_gaps = 0
    details: list[dict] = []

    for case in cases:
        query: str = case.get("query") or case.get("instance_id") or ""
        # retrieved_files is ordered best-first; gold_files is the contextbench key.
        retrieved: list[str] = (
            case.get("retrieved_files")
            or case.get("returned_files")
            or []
        )
        if not retrieved:
            continue

        # Resolve expected files: prefer case_map (authoritative annotations),
        # fall back to gold_files embedded in the run record.
        expected: list[str] = (
            case_map.get(query)
            or case.get("gold_files")
            or []
        )
        expected_set = set(expected)

        # Audit only the top-k returned files.
        top_returned = retrieved[:top_k]
        candidate_files = [f for f in top_returned if f not in expected_set]

        if not candidate_files:
            continue

        case_had_disagreement = False

        for rank_0, file_path in enumerate(top_returned):
            if file_path in expected_set:
                continue

            returned_rank = rank_0 + 1  # 1-indexed

            content = read_file_head(repo_dir, file_path, file_lines)
            if content is None:
                # File missing from cloned repo — note it, skip LLM call.
                details.append({
                    "query": query,
                    "file": file_path,
                    "returned_rank": returned_rank,
                    "score": None,
                    "judge_verdict": "SKIP",
                    "judge_reason": f"File not found in repo dir {repo_dir}",
                })
                total_disagreements += 1
                case_had_disagreement = True
                continue

            try:
                verdict, reason = ask_llm(client, model, query, file_path, content)
            except Exception as exc:
                print(f"  WARN: LLM call failed for {file_path!r}: {exc}", file=sys.stderr)
                verdict, reason = "ERROR", str(exc)

            time.sleep(sleep_s)

            details.append({
                "query": query,
                "file": file_path,
                "returned_rank": returned_rank,
                # No per-file scores in the current run format; use rank as proxy.
                "score": None,
                "judge_verdict": verdict,
                "judge_reason": reason,
            })
            total_disagreements += 1
            case_had_disagreement = True
            if verdict == "YES":
                annotation_gaps += 1

        if case_had_disagreement:
            cases_with_disagreements += 1

    return {
        "total_cases": total_cases,
        "cases_with_disagreements": cases_with_disagreements,
        "total_disagreements": total_disagreements,
        "annotation_gaps": annotation_gaps,
        "details": details,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "ERROR: OPENAI_API_KEY environment variable is not set.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        from openai import OpenAI
    except ImportError:
        print(
            "ERROR: 'openai' package is not installed. "
            "Run: uv run --with openai python3 ...",
            file=sys.stderr,
        )
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    repos_dir = Path(args.repos_dir)
    if not repos_dir.is_dir():
        print(f"ERROR: --repos-dir {repos_dir!r} does not exist.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading run: {args.run}", file=sys.stderr)
    run_data = load_run(args.run)

    print(f"Loading cases: {args.cases}", file=sys.stderr)
    case_map = load_cases(args.cases)

    n_run_cases = len(run_data.get("cases", []))
    print(
        f"Run has {n_run_cases} cases; annotation map has {len(case_map)} entries.",
        file=sys.stderr,
    )

    print(
        f"Auditing top-{args.top_k} returned files per case "
        f"using {args.model} ...",
        file=sys.stderr,
    )

    report = audit(
        run_data=run_data,
        case_map=case_map,
        repos_dir=repos_dir,
        client=client,
        model=args.model,
        top_k=args.top_k,
        file_lines=args.file_lines,
        sleep_s=args.sleep,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    print(
        f"\nDone. {report['total_cases']} cases, "
        f"{report['cases_with_disagreements']} with disagreements, "
        f"{report['annotation_gaps']} likely annotation gaps.",
        file=sys.stderr,
    )
    print(f"Report written to: {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
