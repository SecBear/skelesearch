#!/usr/bin/env python3
"""Generate eval gold sets from git history with sub-file precision.

Extracts meaningful commits from a git repo, parses diff hunks for exact line
ranges, and writes eval cases compatible with `skelesearch eval`.

Usage:
    python3 benchmarks/scripts/generate-gold-set.py \\
        --repo benchmarks/repos/ripgrep \\
        --output benchmarks/cases/rust/ripgrep.json \\
        --max-commits 500 \\
        --max-cases 40
"""

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class DiffRegion:
    file: str
    start_line: int
    end_line: int


@dataclass
class EvalCase:
    id: str
    commit_hash: str
    query: str
    expected_files: list[str]
    expected_regions: list[DiffRegion]
    complexity: str  # low | medium | high
    category: str    # keyword_code | symbol | cross_file | conceptual

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "commit_hash": self.commit_hash,
            "query": self.query,
            "expected_files": self.expected_files,
            "expected_regions": [
                {"file": r.file, "start_line": r.start_line, "end_line": r.end_line}
                for r in self.expected_regions
            ],
            "complexity": self.complexity,
            "category": self.category,
        }


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def git(repo: Path, *args: str) -> str:
    """Run a git command in repo, return stdout. Raises on non-zero exit."""
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed:\n{result.stderr.strip()}"
        )
    return result.stdout


def get_head_commit(repo: Path) -> str:
    return git(repo, "rev-parse", "HEAD").strip()


def get_commits(repo: Path, max_commits: int) -> list[tuple[str, str, str]]:
    """Return list of (hash, subject, iso_date) for the last max_commits non-merge commits."""
    raw = git(repo, "log", "--format=%H|%s|%aI", "--no-merges", f"-{max_commits}")
    results = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|", 2)
        if len(parts) != 3:
            continue
        results.append((parts[0], parts[1], parts[2]))
    return results


def get_numstat(repo: Path, commit_hash: str) -> list[tuple[int, int, str]]:
    """
    Return list of (added, deleted, filepath) for each file in the commit.
    Binary files are represented as '-' counts and are excluded by the caller.
    Renames come back as 'old => new' when --no-renames is NOT used; we parse
    the new name.
    """
    try:
        raw = git(repo, "diff", "--numstat", f"{commit_hash}~1", commit_hash)
    except RuntimeError:
        # First commit — no parent; skip.
        return []

    results = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t", 2)
        if len(parts) != 3:
            continue
        added_str, deleted_str, filepath = parts
        # Binary files are marked with '-'
        if added_str == "-" or deleted_str == "-":
            continue
        try:
            added = int(added_str)
            deleted = int(deleted_str)
        except ValueError:
            continue
        # Handle renames: "src/{old => new}/file.rs" or "old/path => new/path"
        # git --numstat with -M produces {old => new} notation; extract the new name.
        resolved = _resolve_rename_path(filepath)
        results.append((added, deleted, resolved))
    return results


def _resolve_rename_path(filepath: str) -> str:
    """
    Resolve git rename notations:
      - "src/{foo => bar}/file.rs"  → "src/bar/file.rs"
      - "old/path => new/path"      → "new/path"
    """
    # Arrow notation without braces: entire path replaced
    if "=>" in filepath and "{" not in filepath:
        parts = filepath.split("=>", 1)
        return parts[1].strip()

    # Brace notation: replace the {old => new} segment
    m = re.search(r"\{([^}]*) => ([^}]*)\}", filepath)
    if m:
        old_seg, new_seg = m.group(1), m.group(2)
        # Replace the brace group with the new segment
        resolved = filepath[: m.start()] + new_seg + filepath[m.end() :]
        # Collapse any double slashes from empty segments
        resolved = re.sub(r"/+", "/", resolved).strip("/")
        return resolved

    return filepath


def get_diff_regions(repo: Path, commit_hash: str, filepath: str) -> list[DiffRegion]:
    """
    Parse `git diff -U0` hunk headers for filepath and return the new-side line
    ranges that changed. Only the new file side (+) is meaningful for search.

    Hunk header format:  @@ -old_start[,old_count] +new_start[,new_count] @@
    When count is absent, it defaults to 1 (single-line change).
    When count is 0, it means the hunk is a pure deletion — skip (no new lines).
    """
    try:
        raw = git(
            repo,
            "diff",
            "-U0",
            f"{commit_hash}~1",
            commit_hash,
            "--",
            filepath,
        )
    except RuntimeError:
        return []

    regions: list[DiffRegion] = []
    # Pattern: @@ -old_start[,old_count] +new_start[,new_count] @@
    hunk_re = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
    for line in raw.splitlines():
        m = hunk_re.match(line)
        if not m:
            continue
        new_start = int(m.group(1))
        # count absent → 1; count = 0 → pure deletion, skip
        count_str = m.group(2)
        if count_str is not None:
            new_count = int(count_str)
        else:
            new_count = 1  # single-line shorthand

        if new_count == 0:
            # Pure deletion — no lines exist in the new file to search for.
            continue

        new_end = new_start + new_count - 1
        regions.append(DiffRegion(file=filepath, start_line=new_start, end_line=new_end))

    return regions


# ---------------------------------------------------------------------------
# Commit filtering
# ---------------------------------------------------------------------------

# Messages containing these words (in the subject) are skipped wholesale.
_SKIP_WORDS = re.compile(
    r"\b(bump|version|semver|release|changelog|format|fmt|lint|clippy|"
    r"ci|merge|chore|doc|docs|readme|typo|whitespace|spelling|revert)\b",
    re.IGNORECASE,
)

# Messages must contain at least one of these to be kept.
_KEEP_WORDS = re.compile(
    r"\b(fix|add|implement|handle|support|refactor|improve|optimis|optimiz|"
    r"parse|check|validate|detect|update|remove|replace|migrate|extend|"
    r"enable|disable|expose|wrap|unwrap|rework|rewrite|introduce|use|allow)\b",
    re.IGNORECASE,
)

# File patterns that indicate test / CI / doc files.
_TEST_OR_CI_PATTERNS = re.compile(
    r"(^|/)("
    r"test[s_/]|spec[s_/]|__tests__|\.github/|\.circleci/|\.travis|"
    r"Makefile|CMakeLists|CHANGELOG|LICENSE|\.md$|\.txt$|\.rst$|"
    r"\.adoc$|benches?/|examples?/"
    r")",
    re.IGNORECASE,
)

# Conventional commit prefixes to strip when transforming to query.
_CC_PREFIX = re.compile(
    r"^(fix|feat|feature|chore|docs|style|refactor|test|build|ci|perf|revert)"
    r"(\([^)]+\))?[!]?\s*:\s*",
    re.IGNORECASE,
)

# Ticket/issue reference patterns.
_TICKET = re.compile(
    r"(\(#\d+\)|\[#\d+\]|#\d+|\b[A-Z]+-\d+\b|GH-\d+|ISSUE-\d+)",
    re.IGNORECASE,
)


def _is_test_or_ci_file(path: str) -> bool:
    return bool(_TEST_OR_CI_PATTERNS.search(path))


def should_include_commit(subject: str, numstat: list[tuple[int, int, str]]) -> bool:
    """
    Return True if this commit is worth turning into an eval case.

    Exclusion criteria (any one disqualifies):
    - subject matches skip words
    - subject doesn't match any keep word
    - 0 or >20 files changed
    - all changed files are test/CI/doc files
    - all changed files were deletions (added=0)

    The caller has already filtered merge commits via --no-merges.
    """
    subj_lower = subject.lower()
    if _SKIP_WORDS.search(subj_lower):
        return False
    if not _KEEP_WORDS.search(subj_lower):
        return False

    source_files = [
        fp for (added, deleted, fp) in numstat
        if not _is_test_or_ci_file(fp) and added > 0
    ]
    if not source_files:
        return False

    file_count = len(source_files)
    if file_count < 1 or file_count > 20:
        return False

    return True


# ---------------------------------------------------------------------------
# Query transformation
# ---------------------------------------------------------------------------

# Overly generic subjects that produce useless queries even after cleaning.
_TOO_GENERIC = re.compile(
    r"^(fix bug|bug fix|update|cleanup|clean up|minor fix|small fix|fixes|"
    r"improve|improvements?|misc|various|tweak|nit|nits)$",
    re.IGNORECASE,
)

# Identifiers that suggest code-symbol queries: CamelCase or snake_case with at
# least two segments.
_CODE_SYMBOL = re.compile(r"\b([A-Z][a-zA-Z0-9]+[A-Z][a-zA-Z0-9]*|[a-z_]+_[a-z_]+)\b")


def transform_query(subject: str) -> Optional[str]:
    """
    Convert a git commit subject into a natural-language search query.

    Returns None if the result is too short or too generic to be useful.
    """
    q = subject

    # Strip conventional-commit prefix: "fix(parser): " → ""
    q = _CC_PREFIX.sub("", q)

    # Strip ticket references
    q = _TICKET.sub("", q)

    # Strip trailing punctuation
    q = q.rstrip(".").strip()

    # Collapse multiple spaces
    q = re.sub(r"\s+", " ", q).strip()

    # Require at least 3 words
    if len(q.split()) < 3:
        return None

    # Reject generic queries
    if _TOO_GENERIC.match(q):
        return None

    return q


def categorize(query: str, file_count: int) -> str:
    """Assign a category label to the query."""
    q = query.lower()
    if file_count > 1:
        return "cross_file"
    if re.search(r"\b(where|find|locate)\b", q):
        return "symbol"
    if re.search(r"\b(how|why|explain|what)\b", q):
        return "conceptual"
    if _CODE_SYMBOL.search(query):
        return "keyword_code"
    return "keyword_code"


def complexity(file_count: int) -> str:
    if file_count == 1:
        return "low"
    if file_count <= 5:
        return "medium"
    return "high"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def generate(
    repo: Path,
    max_commits: int,
    max_cases: int,
    verbose: bool,
) -> tuple[list[EvalCase], dict]:
    """
    Walk the git log, filter and transform commits into eval cases.

    Returns (cases, metadata_dict).
    """
    commits = get_commits(repo, max_commits)
    head = get_head_commit(repo)

    total_analyzed = 0
    skipped_no_parent = 0
    skipped_filter = 0
    skipped_query = 0
    skipped_regions = 0
    candidate_cases: list[EvalCase] = []

    for commit_hash, subject, _date in commits:
        total_analyzed += 1

        # Get file-level stats; skip first commit (no parent).
        try:
            numstat = get_numstat(repo, commit_hash)
        except RuntimeError:
            skipped_no_parent += 1
            continue

        if not numstat:
            skipped_no_parent += 1
            continue

        # Apply commit-level filter.
        if not should_include_commit(subject, numstat):
            skipped_filter += 1
            if verbose:
                print(f"  SKIP (filter): {commit_hash[:8]} {subject!r}", file=sys.stderr)
            continue

        # Build query.
        query = transform_query(subject)
        if query is None:
            skipped_query += 1
            if verbose:
                print(f"  SKIP (query):  {commit_hash[:8]} {subject!r}", file=sys.stderr)
            continue

        # Gather source files (non-test, non-doc, must have added lines).
        source_files = [
            fp for (added, _deleted, fp) in numstat
            if not _is_test_or_ci_file(fp) and added > 0
        ]
        if not source_files:
            skipped_filter += 1
            continue

        # Build per-file diff regions.
        regions: list[DiffRegion] = []
        for fp in source_files:
            regions.extend(get_diff_regions(repo, commit_hash, fp))

        if not regions:
            skipped_regions += 1
            if verbose:
                print(f"  SKIP (regions): {commit_hash[:8]} {subject!r}", file=sys.stderr)
            continue

        file_count = len(source_files)
        case = EvalCase(
            id=commit_hash[:8],
            commit_hash=commit_hash,
            query=query,
            expected_files=source_files,
            expected_regions=regions,
            complexity=complexity(file_count),
            category=categorize(query, file_count),
        )
        candidate_cases.append(case)

        if verbose:
            print(
                f"  KEEP: {commit_hash[:8]} [{case.complexity}/{case.category}] {query!r}",
                file=sys.stderr,
            )

    # Sort: medium complexity first, then low, then high — medium gives richer
    # signal than trivial single-file changes but avoids noisy bulk refactors.
    COMPLEXITY_ORDER = {"medium": 0, "low": 1, "high": 2}
    candidate_cases.sort(key=lambda c: COMPLEXITY_ORDER[c.complexity])

    # Deduplicate by id (shouldn't happen, but git history can surprise).
    seen_ids: set[str] = set()
    deduped: list[EvalCase] = []
    for c in candidate_cases:
        if c.id not in seen_ids:
            seen_ids.add(c.id)
            deduped.append(c)

    final_cases = deduped[:max_cases]

    # Category distribution for metadata.
    category_dist: dict[str, int] = {}
    for c in final_cases:
        category_dist[c.category] = category_dist.get(c.category, 0) + 1

    metadata = {
        "repository": repo.name,
        "repo_commit": head,
        "max_commits_analyzed": max_commits,
        "total_commits_analyzed": total_analyzed,
        "skipped_no_parent": skipped_no_parent,
        "skipped_filter": skipped_filter,
        "skipped_query": skipped_query,
        "skipped_no_regions": skipped_regions,
        "candidates_found": len(candidate_cases),
        "cases_generated": len(final_cases),
        "category_distribution": category_dist,
    }

    return final_cases, metadata


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate eval gold sets from git history with sub-file precision.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--repo",
        required=True,
        type=Path,
        help="Path to the git repository to analyze",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output JSON path (created with parent dirs)",
    )
    parser.add_argument(
        "--max-commits",
        type=int,
        default=500,
        help="Max commits to walk from HEAD (default: 500)",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=40,
        help="Max eval cases to emit (default: 40)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-commit decisions to stderr",
    )

    args = parser.parse_args()

    repo = args.repo.resolve()
    if not (repo / ".git").exists():
        print(f"ERROR: {repo} is not a git repository", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing {repo} (max {args.max_commits} commits → {args.max_cases} cases)…")

    cases, metadata = generate(
        repo=repo,
        max_commits=args.max_commits,
        max_cases=args.max_cases,
        verbose=args.verbose,
    )

    # Build output document.
    output = {
        "metadata": metadata,
        "cases": [c.to_dict() for c in cases],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    # Summary
    print(f"\nSummary for {metadata['repository']}:")
    print(f"  Commits analyzed:    {metadata['total_commits_analyzed']}")
    print(f"  Skipped (no parent): {metadata['skipped_no_parent']}")
    print(f"  Skipped (filter):    {metadata['skipped_filter']}")
    print(f"  Skipped (query):     {metadata['skipped_query']}")
    print(f"  Skipped (no diff):   {metadata['skipped_no_regions']}")
    print(f"  Candidates found:    {metadata['candidates_found']}")
    print(f"  Cases emitted:       {metadata['cases_generated']}")
    print(f"  Category distribution:")
    for cat, count in sorted(metadata["category_distribution"].items()):
        print(f"    {cat:<20} {count}")
    print(f"\nOutput: {args.output}")
    print(f"Repo HEAD at generation: {metadata['repo_commit']}")


if __name__ == "__main__":
    main()
