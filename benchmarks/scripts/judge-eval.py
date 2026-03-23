#!/usr/bin/env python3
"""
LLM-as-judge eval for skelesearch chunk quality.

Unlike file-level R@5 which only checks "is the right file in results?",
this eval asks: "can an LLM answer the query from the returned chunks alone?"

Measures what agents actually care about: context sufficiency.

Usage:
    python3 benchmarks/scripts/judge-eval.py \
        --binary ./target/release/skelesearch \
        --cases benchmarks/cases/typescript/zod.json \
        --repo-dir benchmarks/repos/zod \
        --provider voyage \
        --output benchmarks/runs/judge-zod.json

    # Quick mode: judge only failing cases from a previous file-level eval
    python3 benchmarks/scripts/judge-eval.py \
        --binary ./target/release/skelesearch \
        --cases benchmarks/cases/typescript/zod.json \
        --repo-dir benchmarks/repos/zod \
        --provider voyage \
        --output benchmarks/runs/judge-zod.json \
        --only-failing benchmarks/runs/zod-baseline.json

Requires: ANTHROPIC_API_KEY or OPENAI_API_KEY for LLM judge calls.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """You are evaluating whether a code search tool returned sufficient context
to answer a developer's question. You will receive:

1. A QUERY: the developer's question about the codebase
2. CHUNKS: the code snippets returned by the search tool
3. GOLD FILES: the files that are known to contain the answer

Score the retrieval result:
- "sufficient" (1.0): The chunks contain enough code to fully answer the query.
  The developer could read these chunks and understand the answer without
  looking at any other files.
- "partial" (0.5): The chunks contain some relevant code but are missing key
  pieces. The developer would need to look at additional files.
- "insufficient" (0.0): The chunks do not contain the relevant code. The
  developer could not answer the query from these chunks.

Respond with ONLY a JSON object:
{"score": <0.0|0.5|1.0>, "reason": "<one sentence explaining your judgment>"}"""

JUDGE_USER = """QUERY: {query}

GOLD FILES (known correct): {gold_files}

RETURNED CHUNKS ({num_chunks} chunks, {total_tokens} tokens):
{chunks_text}

Score this retrieval result. Does the returned context contain enough code
to answer the query? Respond with JSON only."""


# ---------------------------------------------------------------------------
# LLM judge call
# ---------------------------------------------------------------------------

def call_judge(query: str, chunks: list[dict], gold_files: list[str]) -> dict:
    """Call LLM to judge chunk sufficiency. Returns {score, reason}."""
    chunks_text = ""
    total_tokens = 0
    for i, chunk in enumerate(chunks[:10]):  # Cap at 10 chunks
        fp = chunk.get("file_path", "unknown")
        start = chunk.get("start_line", "?")
        end = chunk.get("end_line", "?")
        content = chunk.get("content", "")
        tokens = len(content.split())
        total_tokens += tokens
        chunks_text += f"\n--- Chunk {i+1}: {fp} (lines {start}-{end}, ~{tokens} tokens) ---\n"
        chunks_text += content + "\n"

    prompt = JUDGE_USER.format(
        query=query,
        gold_files=", ".join(gold_files),
        num_chunks=min(len(chunks), 10),
        total_tokens=total_tokens,
        chunks_text=chunks_text if chunks_text else "(no chunks returned)",
    )

    # Try Anthropic first, fall back to OpenAI
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if anthropic_key:
        return _call_anthropic(anthropic_key, prompt)
    elif openai_key:
        return _call_openai(openai_key, prompt)
    else:
        # Fallback: heuristic judge (no API key)
        return _heuristic_judge(chunks, gold_files)


def _call_anthropic(api_key: str, prompt: str) -> dict:
    """Call Claude as judge."""
    import urllib.request
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps({
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 200,
            "system": JUDGE_SYSTEM,
            "messages": [{"role": "user", "content": prompt}],
        }).encode(),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            text = data["content"][0]["text"]
            # Parse JSON from response
            return json.loads(text)
    except Exception as e:
        return {"score": -1, "reason": f"judge error: {e}"}


def _call_openai(api_key: str, prompt: str) -> dict:
    """Call GPT-4o-mini as judge."""
    import urllib.request
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps({
            "model": "gpt-4o-mini",
            "max_tokens": 200,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        }).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            text = data["choices"][0]["message"]["content"]
            return json.loads(text)
    except Exception as e:
        return {"score": -1, "reason": f"judge error: {e}"}


def _heuristic_judge(chunks: list[dict], gold_files: list[str]) -> dict:
    """Fallback when no API key: check if gold files appear in returned chunks."""
    if not chunks:
        return {"score": 0.0, "reason": "no chunks returned"}
    returned_files = {c.get("file_path", "") for c in chunks}
    gold_set = set(gold_files)
    overlap = returned_files & gold_set
    if len(overlap) == len(gold_set):
        return {"score": 1.0, "reason": f"all {len(gold_set)} gold files present in chunks (heuristic)"}
    elif overlap:
        return {"score": 0.5, "reason": f"{len(overlap)}/{len(gold_set)} gold files present (heuristic)"}
    else:
        return {"score": 0.0, "reason": "no gold files in returned chunks (heuristic)"}


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def run_search(binary: str, repo_dir: str, query: str, provider: str, top_k: int = 5) -> list[dict]:
    """Run skelesearch search and return chunk results."""
    try:
        result = subprocess.run(
            [binary, "search", query[:4000], "--provider", provider,
             "--top-k", str(top_k), "--json"],
            capture_output=True, text=True, timeout=120,
            cwd=repo_dir,
        )
        if result.returncode != 0:
            return []
        data = json.loads(result.stdout)
        return data.get("results", data) if isinstance(data, dict) else data
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LLM-as-judge eval for chunk quality")
    parser.add_argument("--binary", required=True)
    parser.add_argument("--cases", required=True, help="Eval cases JSON file")
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--provider", default="fastembed")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only-failing", default=None,
                        help="Path to file-level eval JSON; only judge cases that failed R@5")
    args = parser.parse_args()

    cases = json.load(open(args.cases))
    if isinstance(cases, dict):
        cases = cases.get("cases", cases.get("queries", []))

    # Filter to failing cases if requested
    failing_queries = None
    if args.only_failing:
        prev = json.load(open(args.only_failing))
        prev_cases = prev.get("cases", [])
        failing_queries = set()
        for c in prev_cases:
            if c.get("recall_at_5", 1.0) < 1.0:
                failing_queries.add(c.get("query", ""))

    if args.limit:
        cases = cases[:args.limit]

    results = []
    sufficient = 0
    partial = 0
    insufficient = 0
    errors = 0

    print(f"Judging {len(cases)} cases with LLM-as-judge")
    print(f"Provider: {args.provider}, top-k: {args.top_k}")
    print()

    for i, case in enumerate(cases):
        query = case.get("query", case.get("question", ""))
        gold_files = case.get("gold_files", case.get("expected_files", []))

        if failing_queries is not None and query not in failing_queries:
            continue

        # Search
        chunks = run_search(args.binary, args.repo_dir, query, args.provider, args.top_k)

        # Judge
        judgment = call_judge(query, chunks, gold_files)
        score = judgment.get("score", -1)
        reason = judgment.get("reason", "")

        if score >= 1.0:
            sufficient += 1
            label = "SUFFICIENT"
        elif score >= 0.5:
            partial += 1
            label = "PARTIAL"
        elif score >= 0.0:
            insufficient += 1
            label = "INSUFFICIENT"
        else:
            errors += 1
            label = "ERROR"

        print(f"  [{i+1}/{len(cases)}] {label} ({score:.1f}) {query[:60]}...")
        if score < 1.0:
            print(f"           {reason}")

        results.append({
            "query": query,
            "gold_files": gold_files,
            "returned_files": list({c.get("file_path", "") for c in chunks[:10]}),
            "num_chunks": len(chunks),
            "judge_score": score,
            "judge_reason": reason,
        })

        # Rate limit
        time.sleep(0.5)

    total = sufficient + partial + insufficient
    if total == 0:
        total = 1

    output = {
        "tool": "skelesearch",
        "provider": args.provider,
        "judge": "llm",
        "aggregate": {
            "total_cases": len(results),
            "sufficient": sufficient,
            "partial": partial,
            "insufficient": insufficient,
            "errors": errors,
            "sufficiency_rate": sufficient / total,
            "partial_rate": partial / total,
            "context_score": (sufficient * 1.0 + partial * 0.5) / total,
        },
        "cases": results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print()
    print(f"=== Results ===")
    print(f"Sufficient: {sufficient}/{total} ({sufficient/total:.1%})")
    print(f"Partial:    {partial}/{total} ({partial/total:.1%})")
    print(f"Insufficient: {insufficient}/{total} ({insufficient/total:.1%})")
    print(f"Context Score: {output['aggregate']['context_score']:.3f}")
    print(f"Errors: {errors}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
