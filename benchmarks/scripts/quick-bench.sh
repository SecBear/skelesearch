#!/usr/bin/env bash
set -euo pipefail

# quick-bench.sh — Fast feedback from ContextBench (~10-20s with cached repos).
#
# Uses ContextBench: 1,136 instances, 66 repos, 8 languages (Python, JS, TS,
# Go, C, Rust, C++, Java). Human-annotated gold contexts with sub-file spans.
# Real GitHub issues as queries. Research-grade ground truth.
#
# For coding agents: run after changes to verify no regression.
# Exit code 0 = pass, 1 = regression, 2 = setup error.
#
# Performance: dataset is cached to benchmarks/data/contextbench.parquet after
# the first run (~14s). With --cached-only, skips git checkout and re-index,
# reducing per-instance time from 26-300s → ~2s.
#
# Usage:
#   ./benchmarks/scripts/quick-bench.sh                # 10 instances, ~15s
#   ./benchmarks/scripts/quick-bench.sh --n 20          # more instances
#   ./benchmarks/scripts/quick-bench.sh --cached-only   # skip cloning + reindex
#   ./benchmarks/scripts/quick-bench.sh --lang rust     # Rust instances only
#   ./benchmarks/scripts/quick-bench.sh --full          # all cached instances

BINARY="./target/release/skelesearch"
ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --binary) BINARY="$2"; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

BINARY="$(cd "$(dirname "$BINARY")" && pwd)/$(basename "$BINARY")"

if [[ ! -x "$BINARY" ]]; then
  echo "ERROR: Binary not found: $BINARY"
  echo "Run: cargo build --release --features storage-sqlite"
  exit 2
fi

# VOYAGE_API_KEY is required for search unless repos are indexed with fastembed.
if [[ -z "${VOYAGE_API_KEY:-}" ]]; then
  echo "WARNING: VOYAGE_API_KEY not set. Search may fail if repos are indexed with Voyage."
  echo "  Set VOYAGE_API_KEY or re-index repos with: skelesearch index <repo> --provider fastembed"
fi

# Data cache: parquet file avoids 14s HuggingFace download on repeated runs.
DATA_CACHE="$(dirname "$0")/../data/contextbench.parquet"

exec env VOYAGE_API_KEY="$VOYAGE_API_KEY" uv run --with datasets --with huggingface_hub --with pandas --with pyarrow \
  python3 "$(dirname "$0")/quick_bench.py" \
  --binary "$BINARY" \
  --provider voyage \
  --cache-dir "$(dirname "$0")/../swebench-repos" \
  --data-cache "$DATA_CACHE" \
  "${ARGS[@]}"