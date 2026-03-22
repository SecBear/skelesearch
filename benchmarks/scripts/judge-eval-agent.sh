#!/usr/bin/env bash
set -euo pipefail

# judge-eval-agent.sh — Use pi (omp) subagent as LLM judge for chunk quality.
#
# Spawns a pi subagent per query to judge whether returned chunks are sufficient.
# Uses the session's existing auth (no separate API key needed).
#
# Usage:
#   ./benchmarks/scripts/judge-eval-agent.sh \
#     --binary ./target/release/skelesearch \
#     --cases benchmarks/cases/typescript/zod.json \
#     --repo benchmarks/repos/zod \
#     --provider voyage \
#     --limit 10

BINARY=""
CASES=""
REPO=""
PROVIDER="voyage"
TOP_K=5
LIMIT=""
OUTPUT=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --binary) BINARY="$2"; shift 2 ;;
    --cases) CASES="$2"; shift 2 ;;
    --repo) REPO="$2"; shift 2 ;;
    --provider) PROVIDER="$2"; shift 2 ;;
    --top-k) TOP_K="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    *) echo "Unknown: $1"; exit 1 ;;
  esac
done

if [[ -z "$BINARY" || -z "$CASES" || -z "$REPO" ]]; then
  echo "Usage: $0 --binary <path> --cases <json> --repo <dir> [--provider voyage] [--limit N]"
  exit 1
fi

BINARY="$(cd "$(dirname "$BINARY")" && pwd)/$(basename "$BINARY")"
REPO="$(cd "$REPO" && pwd)"
[[ -z "$OUTPUT" ]] && OUTPUT="benchmarks/runs/judge-$(basename "$REPO")-$(date +%Y%m%d-%H%M%S).json"

echo "=== LLM-as-Judge Eval ==="
echo "Binary: $BINARY"
echo "Cases: $CASES"
echo "Repo: $REPO"
echo "Provider: $PROVIDER"
echo "Output: $OUTPUT"
echo ""

# Extract cases with python
python3 << PYEOF
import json, subprocess, sys, os, time
from pathlib import Path

cases = json.load(open("$CASES"))
if isinstance(cases, dict):
    cases = cases.get("cases", cases.get("queries", []))

limit = int("${LIMIT:-0}") or len(cases)
cases = cases[:limit]

results = []
sufficient = partial = insufficient = errors = 0

for i, case in enumerate(cases):
    query = case.get("query", case.get("question", ""))
    gold_files = case.get("gold_files", case.get("expected_files", []))

    # 1. Search
    try:
        r = subprocess.run(
            ["$BINARY", "search", query[:4000], "--provider", "$PROVIDER",
             "--top-k", "$TOP_K", "--json"],
            capture_output=True, text=True, timeout=120, cwd="$REPO",
        )
        chunks = json.loads(r.stdout).get("results", []) if r.returncode == 0 else []
    except:
        chunks = []

    # 2. Format chunks for judge
    chunks_text = ""
    for ci, chunk in enumerate(chunks[:10]):
        fp = chunk.get("file_path", "?")
        sl = chunk.get("start_line", "?")
        el = chunk.get("end_line", "?")
        content = chunk.get("content", "")[:2000]
        chunks_text += f"\\n--- Chunk {ci+1}: {fp} (lines {sl}-{el}) ---\\n{content}\\n"

    # 3. Call pi subagent as judge
    judge_prompt = f"""You are judging code search quality. A developer asked:

QUERY: {query}

The search tool returned these code chunks:
{chunks_text if chunks_text else "(no chunks returned)"}

GOLD FILES (known correct answers): {', '.join(gold_files)}

Score the retrieval:
- "sufficient" (1.0): Chunks contain enough code to fully answer the query
- "partial" (0.5): Some relevant code but missing key pieces
- "insufficient" (0.0): Chunks do not contain the relevant code

Respond with ONLY JSON: {{"score": <number>, "reason": "<one sentence>"}}"""

    try:
        jr = subprocess.run(
            ["pi", "-p", "--model", "anthropic/claude-sonnet-4-20250514", judge_prompt],
            capture_output=True, text=True, timeout=60,
        )
        # Parse JSON from output
        out = jr.stdout.strip()
        # Find JSON in output
        import re
        m = re.search(r'\{[^}]*"score"[^}]*\}', out)
        if m:
            judgment = json.loads(m.group())
        else:
            judgment = {"score": -1, "reason": f"no JSON in judge output: {out[:100]}"}
    except Exception as e:
        judgment = {"score": -1, "reason": f"judge error: {e}"}

    score = judgment.get("score", -1)
    reason = judgment.get("reason", "")

    if score >= 1.0: sufficient += 1; label = "SUFFICIENT"
    elif score >= 0.5: partial += 1; label = "PARTIAL"
    elif score >= 0.0: insufficient += 1; label = "INSUFFICIENT"
    else: errors += 1; label = "ERROR"

    print(f"  [{i+1}/{len(cases)}] {label} ({score:.1f}) {query[:60]}...")
    if score < 1.0:
        print(f"           {reason[:100]}")

    results.append({
        "query": query,
        "gold_files": gold_files,
        "returned_files": list({c.get("file_path", "") for c in chunks[:10]}),
        "num_chunks": len(chunks),
        "judge_score": score,
        "judge_reason": reason,
    })

total = max(sufficient + partial + insufficient, 1)
output = {
    "tool": "skelesearch",
    "provider": "$PROVIDER",
    "judge": "claude-sonnet-4-20250514",
    "aggregate": {
        "total_cases": len(results),
        "sufficient": sufficient, "partial": partial, "insufficient": insufficient,
        "errors": errors,
        "sufficiency_rate": sufficient / total,
        "context_score": (sufficient + partial * 0.5) / total,
    },
    "cases": results,
}

Path("$OUTPUT").parent.mkdir(parents=True, exist_ok=True)
with open("$OUTPUT", "w") as f:
    json.dump(output, f, indent=2)

print()
print(f"=== Results ===")
print(f"Sufficient: {sufficient}/{total} ({sufficient/total:.0%})")
print(f"Partial:    {partial}/{total} ({partial/total:.0%})")
print(f"Insufficient: {insufficient}/{total} ({insufficient/total:.0%})")
print(f"Context Score: {output['aggregate']['context_score']:.3f}")
print(f"Output: $OUTPUT")
PYEOF
