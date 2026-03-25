#!/usr/bin/env bash
set -euo pipefail

# overnight.sh — Full rebuild, re-index, eval, and SWE-bench run.
# Run on brick overnight: nohup ./benchmarks/scripts/overnight.sh > overnight.log 2>&1 &
#
# Prerequisites: source .env && export VOYAGE_API_KEY

cd "$(dirname "$0")/../.."
ROOT="$(pwd)"
BIN="$ROOT/target/release/skelesearch"
LOG="$ROOT/benchmarks/runs/overnight-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$ROOT/benchmarks/runs"

exec > >(tee "$LOG") 2>&1
echo "=== Overnight run started at $(date) ==="
echo "Root: $ROOT"

# 0. Preflight
if [[ -z "${VOYAGE_API_KEY:-}" ]]; then
  echo "FATAL: VOYAGE_API_KEY not set"
  exit 1
fi

# 1. Build
echo ""
echo "=== Step 1: Build ==="
cargo build --release --features storage-rocksdb
echo "Binary: $BIN ($(date))"

# 2. Unit tests
echo ""
echo "=== Step 2: Unit tests ==="
cargo test -p skelesearch-core --lib 2>&1 | tail -3
cargo test -p skelesearch-mcp 2>&1 | tail -3

# 3. Re-index hand-written eval repos
echo ""
echo "=== Step 3: Re-index eval repos ==="
for repo in mini-redis hyperfine cobra httpx hono zod; do
  echo "--- $repo ---"
  rm -rf "$ROOT/benchmarks/repos/$repo/.skelesearch"
  RUST_LOG=skelesearch=warn "$BIN" index "$ROOT/benchmarks/repos/$repo" --provider voyage
done

# 4. Eval (hand-written cases)
echo ""
echo "=== Step 4: Eval (240 cases, 6 repos) ==="
python3 "$ROOT/benchmarks/scripts/eval.py"

# 5. Re-index SWE-bench repos (large, takes 30-60 min)
echo ""
echo "=== Step 5: Re-index SWE-bench repos ==="
SWEBENCH="$ROOT/benchmarks/swebench-repos"
if [[ -d "$SWEBENCH" ]]; then
  for repo_dir in "$SWEBENCH"/*/; do
    repo_name=$(basename "$repo_dir")
    # Reset to default branch
    (cd "$repo_dir" && git checkout main 2>/dev/null || git checkout master 2>/dev/null || true) 2>/dev/null
    echo "--- $repo_name ---"
    rm -rf "$repo_dir/.skelesearch"
    RUST_LOG=skelesearch=warn "$BIN" index "$repo_dir" --provider voyage || echo "FAILED: $repo_name"
  done
else
  echo "No SWE-bench repos found at $SWEBENCH — skipping"
fi

# 6. ContextBench quick-bench (all cached repos)
echo ""
echo "=== Step 6: ContextBench quick-bench ==="
if [[ -f "$ROOT/benchmarks/scripts/quick_bench.py" ]]; then
  VOYAGE_API_KEY="$VOYAGE_API_KEY" uv run --with datasets --with huggingface_hub \
    python3 "$ROOT/benchmarks/scripts/quick_bench.py" \
    --binary "$BIN" \
    --n 30 \
    --provider voyage \
    --cache-dir "$SWEBENCH" \
    --cached-only \
    --threshold 20 || echo "Quick-bench completed (may have failed threshold)"
else
  echo "quick_bench.py not found — skipping"
fi

# 7. SWE-bench eval (if adapter exists)
echo ""
echo "=== Step 7: SWE-bench eval ==="
if [[ -f "$ROOT/benchmarks/scripts/swebench-eval.py" ]]; then
  VOYAGE_API_KEY="$VOYAGE_API_KEY" uv run --with datasets --with huggingface_hub \
    python3 "$ROOT/benchmarks/scripts/swebench-eval.py" \
    --binary "$BIN" \
    --provider voyage \
    --cache-dir "$SWEBENCH" \
    --output "$ROOT/benchmarks/runs/swebench-$(date +%Y%m%d).json" \
    --limit 50 || echo "SWE-bench eval completed (may have errors)"
else
  echo "swebench-eval.py not found — skipping"
fi

echo ""
echo "=== Overnight run completed at $(date) ==="
echo "Log: $LOG"
