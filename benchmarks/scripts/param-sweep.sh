#!/usr/bin/env bash
set -euo pipefail

# param-sweep.sh — Automated parameter sweep for unified search tuning.
#
# Generates TOML configs, runs eval on all 6 repos per config, outputs CSV.
# Uses Voyage embedding for speed (API, not local CPU).
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

while [[ $# -gt 0 ]]; do
  case $1 in
    --binary) BINARY="$2"; shift 2 ;;
    --provider) PROVIDER="$2"; shift 2 ;;
    --repos) REPOS=$(echo "$2" | tr ',' ' '); shift 2 ;;
    *) echo "Unknown: $1"; exit 1 ;;
  esac
done

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
echo "Sweep: ${#CONFIGS[@]} configs × $(echo $REPOS | wc -w | tr -d ' ') repos"
echo "Output: $OUTPUT_CSV"
echo ""

for cfg_line in "${CONFIGS[@]}"; do
  read -r TAG FW GSF GMS PRF <<< "$cfg_line"
  echo "=== Config: $TAG (fts=$FW graph=$GSF gms=$GMS pr=$PRF) ==="

  for repo in $REPOS; do
    # Generate TOML config
    cat > "benchmarks/repos/$repo/.skelesearch.toml" << EOF
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

    # Run eval (reuse index — only search params change)
    RESULT=$(env -u OPENAI_API_KEY ./benchmarks/scripts/quick-eval.sh \
      --repo "$repo" --tag "sweep-${TAG}" --profile unified \
      --provider "$PROVIDER" 2>&1 | grep '^Done' || echo "FAILED")

    if [[ "$RESULT" == "FAILED" ]]; then
      echo "  $repo: FAILED"
      echo "$TAG,$repo,0,0,0,0,0" >> "$OUTPUT_CSV"
    else
      # Parse: Done  repo=X  profile=Y  R@5=Z%  R@10=W%  MRR=V  cases=N  ms=M
      R5=$(echo "$RESULT" | grep -oP 'R@5=\K[0-9.]+')
      R10=$(echo "$RESULT" | grep -oP 'R@10=\K[0-9.]+')
      MRR=$(echo "$RESULT" | grep -oP 'MRR=\K[0-9.]+')
      CASES=$(echo "$RESULT" | grep -oP 'cases=\K[0-9]+')
      MS=$(echo "$RESULT" | grep -oP 'ms=\K[0-9]+')
      echo "  $repo: R@5=${R5}% MRR=$MRR"
      echo "$TAG,$repo,$R5,$R10,$MRR,$CASES,$MS" >> "$OUTPUT_CSV"
    fi
  done
  echo ""
done

echo "=== Sweep complete ==="
echo "Results: $OUTPUT_CSV"
echo ""
echo "Analyze:"
echo "  python3 -c \"import csv; rows=list(csv.DictReader(open('$OUTPUT_CSV'))); [print(f'{r[\\\"tag\\\"]:>15} avg_R@5={sum(float(x[\\\"R@5\\\"]) for x in rows if x[\\\"tag\\\"]==r[\\\"tag\\\"])/len([x for x in rows if x[\\\"tag\\\"]==r[\\\"tag\\\"]]):.1f}%') for r in {d[\\\"tag\\\"]:d for d in rows}.values()]\""
