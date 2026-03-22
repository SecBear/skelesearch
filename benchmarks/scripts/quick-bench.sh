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
# Usage:
#   ./benchmarks/scripts/quick-bench.sh                # 10 instances, ~15s
#   ./benchmarks/scripts/quick-bench.sh --n 20          # more instances
#   ./benchmarks/scripts/quick-bench.sh --cached-only   # skip cloning new repos
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

if [[ -z "${VOYAGE_API_KEY:-}" ]]; then
  echo "ERROR: VOYAGE_API_KEY not set (needed for fast Voyage embedding)"
  exit 2
fi

exec uv run --with datasets --with huggingface_hub \
  python3 "$(dirname "$0")/quick_bench.py" \
  --binary "$BINARY" \
  --provider voyage \
  --cache-dir "$(dirname "$0")/../swebench-repos" \
  "${ARGS[@]}"
