#!/usr/bin/env bash
set -euo pipefail

# param-sweep.sh — Automated parameter sweep for unified search tuning.
#
# Calls the skelesearch binary directly (bypasses the adapter's TOML overwrite).
# Indexes once per repo with Voyage, then sweeps search parameters via TOML.
#
# Usage:
#   ./benchmarks/scripts/param-sweep.sh                     # default sweep
#   ./benchmarks/scripts/param-sweep.sh --provider fastembed # local-only
#   ./benchmarks/scripts/param-sweep.sh --repos mini-redis,cobra  # subset
#
# Output: benchmarks/runs/sweep-<timestamp>.csv

BINARY="./target/release/skelesearch"
PROVIDER="voyage"
REPOS="mini-redis hyperfine hono zod httpx cobra"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUTPUT_CSV="benchmarks/runs/sweep-${TIMESTAMP}.csv"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_BASE="$(cd "$SCRIPT_DIR/../repos" && pwd)"
CASES_BASE="$(cd "$SCRIPT_DIR/../cases" && pwd)"

while [[ $# -gt 0 ]]; do
  case $1 in
    --binary) BINARY="$2"; shift 2 ;;
    --provider) PROVIDER="$2"; shift 2 ;;
    --repos) REPOS=$(echo "$2" | tr ',' ' '); shift 2 ;;
    *) echo "Unknown: $1"; exit 1 ;;
  esac
done

BINARY="$(cd "$(dirname "$BINARY")" && pwd)/$(basename "$BINARY")"

# Map repo → language for finding eval cases
declare -A REPO_LANG=(
  [mini-redis]=rust [hyperfine]=rust
  [hono]=typescript [zod]=typescript
  [httpx]=python [cobra]=go
)

# Define sweep: each line is "tag fts_weight graph_score_factor graph_min_score pagerank_factor"
CONFIGS=(
  "baseline      0.55 0.3  0.005 0.1"
  "fts60         0.60 0.3  0.005 0.1"
  "fts50         0.50 0.3  0.005 0.1"
  "fts70         0.70 0.3  0.005 0.1"
  "fts40         0.40 0.3  0.005 0.1"
  "graph50       0.55 0.5  0.005 0.1"
  "graph20       0.55 0.2  0.005 0.1"
  "graph10       0.55 0.1  0.005 0.1"
  "gms01         0.55 0.3  0.01  0.1"
  "gms001        0.55 0.3  0.001 0.1"
  "pr00          0.55 0.3  0.005 0.0"
  "pr02          0.55 0.3  0.005 0.2"
  "pr05          0.55 0.3  0.005 0.5"
  "best_guess    0.60 0.2  0.01  0.05"
)

echo "tag,repo,R@5,R@10,MRR,cases,ms" > "$OUTPUT_CSV"
echo "Sweep: ${#CONFIGS[@]} configs x $(echo $REPOS | wc -w | tr -d ' ') repos"
echo "Provider: $PROVIDER"
echo "Output: $OUTPUT_CSV"
echo ""

# Step 1: Index all repos once (reuses existing index if present)
echo "=== Indexing repos ==="
for repo in $REPOS; do
  REPO_DIR="$REPO_BASE/$repo"
  if [[ ! -d "$REPO_DIR/.skelesearch" ]]; then
    echo "  Indexing $repo..."
    (cd "$REPO_DIR" && "$BINARY" index . --provider "$PROVIDER" 2>&1 | tail -1)
  else
    echo "  $repo: index exists, reusing"
  fi
done
echo ""

# Step 2: Sweep configs — only search parameters change, no re-index needed
for cfg_line in "${CONFIGS[@]}"; do
  read -r TAG FW GSF GMS PRF <<< "$cfg_line"
  echo "=== $TAG (fts=$FW graph=$GSF gms=$GMS pr=$PRF) ==="

  for repo in $REPOS; do
    REPO_DIR="$REPO_BASE/$repo"
    LANG="${REPO_LANG[$repo]}"
    EVAL_FILE="$CASES_BASE/$LANG/$repo.json"

    if [[ ! -f "$EVAL_FILE" ]]; then
      echo "  $repo: no eval cases, skipping"
      continue
    fi

    # Write tuned config
    cat > "$REPO_DIR/.skelesearch.toml" << EOF
[index]
symbol_enrichment = true
[search]
unified_search = true
fts_weight = $FW
graph_score_factor = $GSF
graph_min_score = $GMS
pagerank_factor = $PRF
[search.expansion]
enabled = false
[search.graph]
enabled = true
max_depth = 1
EOF

    # Run eval directly
    START_MS=$(($(date +%s%N)/1000000))
    RESULT=$(cd "$REPO_DIR" && "$BINARY" eval "$EVAL_FILE" --provider "$PROVIDER" --json 2>/dev/null) || RESULT=""
    END_MS=$(($(date +%s%N)/1000000))
    ELAPSED=$((END_MS - START_MS))

    if [[ -z "$RESULT" ]]; then
      echo "  $repo: FAILED"
      echo "$TAG,$repo,0,0,0,0,$ELAPSED" >> "$OUTPUT_CSV"
      continue
    fi

    # Parse JSON output
    R5=$(echo "$RESULT" | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'{d[\"aggregate\"][\"mean_recall_at_5\"]*100:.1f}')")
    R10=$(echo "$RESULT" | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'{d[\"aggregate\"][\"mean_recall_at_10\"]*100:.1f}')")
    MRR=$(echo "$RESULT" | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'{d[\"aggregate\"][\"mean_mrr\"]:.3f}')")
    CASES=$(echo "$RESULT" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['aggregate']['total_cases'])")

    echo "  $repo: R@5=${R5}% R@10=${R10}% MRR=$MRR (${ELAPSED}ms)"
    echo "$TAG,$repo,$R5,$R10,$MRR,$CASES,$ELAPSED" >> "$OUTPUT_CSV"
  done
  echo ""
done

echo "=== Sweep complete ==="
echo "Results: $OUTPUT_CSV"
echo ""
# Summary
python3 << 'PYEOF'
import csv, sys
from collections import defaultdict

rows = list(csv.DictReader(open(sys.argv[1] if len(sys.argv) > 1 else "$OUTPUT_CSV")))
tags = defaultdict(list)
for r in rows:
    tags[r["tag"]].append(r)

print(f"{'tag':>15} {'avg_R@5':>8} {'avg_R@10':>9} {'avg_MRR':>8} {'avg_ms':>8}")
print("-" * 55)
for tag, items in tags.items():
    valid = [i for i in items if float(i["R@5"]) > 0]
    if not valid:
        continue
    ar5 = sum(float(i["R@5"]) for i in valid) / len(valid)
    ar10 = sum(float(i["R@10"]) for i in valid) / len(valid)
    amrr = sum(float(i["MRR"]) for i in valid) / len(valid)
    ams = sum(int(i["ms"]) for i in valid) / len(valid)
    print(f"{tag:>15} {ar5:>7.1f}% {ar10:>8.1f}% {amrr:>8.3f} {ams:>7.0f}")
PYEOF
