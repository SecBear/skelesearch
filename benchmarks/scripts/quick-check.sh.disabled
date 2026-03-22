#!/usr/bin/env bash
set -euo pipefail

# quick-check.sh — Fast feedback eval (~30-60s).
#
# Picks N stratified-random cases per repo, runs skelesearch eval against
# the existing index, reports results, and exits non-zero on quality regression.
# Designed as a pre-commit hook or developer sanity check.
#
# Usage:
#   ./benchmarks/scripts/quick-check.sh                  # 3 cases/repo, ~30s
#   ./benchmarks/scripts/quick-check.sh --full            # all cases, ~5-10 min
#   ./benchmarks/scripts/quick-check.sh --per-repo 5      # 5 cases/repo
#   ./benchmarks/scripts/quick-check.sh --repo mini-redis # single repo
#   ./benchmarks/scripts/quick-check.sh --seed 42         # reproducible (default)
#   ./benchmarks/scripts/quick-check.sh --seed random     # non-deterministic
#   ./benchmarks/scripts/quick-check.sh --provider voyage # cloud embeddings
#   ./benchmarks/scripts/quick-check.sh --unified         # unified search path

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BINARY="${PROJECT_ROOT}/target/release/skelesearch"

# Pass-through arguments to the Python sampler.
PY_ARGS=()

# Intercept --binary before forwarding everything else, so we can use it
# for the pre-flight check below as well.
REMAINING=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --binary)
      BINARY="$2"
      PY_ARGS+=("--binary" "$2")
      shift 2
      ;;
    --help|-h)
      grep '^#' "$0" | grep -v '^#!/' | sed 's/^# \?//'
      exit 0
      ;;
    *)
      PY_ARGS+=("$1")
      shift
      ;;
  esac
done

# Pre-flight: binary must exist and be executable.
if [[ ! -x "$BINARY" ]]; then
  echo "ERROR: Binary not found or not executable: $BINARY"
  echo "       Run: cargo build --release"
  exit 2
fi

# Pre-flight: at least one benchmark repo must be present.
if [[ ! -d "${PROJECT_ROOT}/benchmarks/repos/mini-redis" ]]; then
  echo "ERROR: Benchmark repos not cloned."
  echo "       Run: bun benchmarks/scripts/clone-repos.ts"
  exit 2
fi

exec python3 -u \
  "${SCRIPT_DIR}/quick_check.py" \
  --project-root "${PROJECT_ROOT}" \
  --binary "${BINARY}" \
  "${PY_ARGS[@]}"
