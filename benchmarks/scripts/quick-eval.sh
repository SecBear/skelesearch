#!/usr/bin/env bash
set -euo pipefail

# quick-eval.sh — Fast feedback loop for skelesearch development.
#
# Runs the diagnostic subset (3 repos, ~3-5 min) instead of the full 6-repo
# suite (~15 min). Catches regressions AND measures improvements on the
# weakest repos.
#
# Usage:
#   ./benchmarks/scripts/quick-eval.sh                    # fastembed, no API keys
#   ./benchmarks/scripts/quick-eval.sh --provider voyage  # voyage, needs API keys
#   ./benchmarks/scripts/quick-eval.sh --full             # all 6 repos
#   ./benchmarks/scripts/quick-eval.sh --repo zod         # single repo
#
# Requirements:
#   - Built binary at ./target/release/skelesearch (or pass --binary)
#   - bun installed
#   - Repos cloned: bun benchmarks/scripts/clone-repos.ts

BINARY="./target/release/skelesearch"
PROVIDER="fastembed"
PROFILE_NAME="fastembed-full"
MODE="quick"  # quick | full | single
SINGLE_REPO=""
TAG=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --binary) BINARY="$2"; shift 2 ;;
    --provider)
      PROVIDER="$2"
      if [[ "$PROVIDER" == "voyage" ]]; then
        PROFILE_NAME="voyage-full"
      fi
      shift 2
      ;;
    --profile) PROFILE_NAME="$2"; shift 2 ;;
    --full) MODE="full"; shift ;;
    --repo) MODE="single"; SINGLE_REPO="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --help|-h)
      echo "Usage: $0 [--provider fastembed|voyage] [--full] [--repo NAME] [--binary PATH] [--tag LABEL]"
      echo ""
      echo "Modes:"
      echo "  (default)     Quick: zod + httpx + cobra (~3-5 min)"
      echo "  --full        All 6 repos (~10-15 min)"
      echo "  --repo NAME   Single repo"
      echo ""
      echo "Providers:"
      echo "  fastembed     Local ONNX, no API keys (default)"
      echo "  voyage        Cloud, needs VOYAGE_API_KEY + OPENAI_API_KEY"
      echo ""
      echo "Options:"
      echo "  --binary PATH   Path to skelesearch binary (default: ./target/release/skelesearch)"
      echo "  --profile NAME  Config profile name (default: fastembed-full or voyage-full)"
      echo "  --tag LABEL     Label for this run (e.g., 'post-depth-decay', used in output filenames)"
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

PROFILE="benchmarks/configs/${PROFILE_NAME}.toml"
if [[ ! -f "$PROFILE" ]]; then
  echo "ERROR: Profile not found: $PROFILE"
  echo "Available: $(ls benchmarks/configs/*.toml | xargs -n1 basename | sed 's/.toml//' | tr '\n' ' ')"
  exit 1
fi

if [[ ! -x "$BINARY" ]]; then
  echo "ERROR: Binary not found or not executable: $BINARY"
  echo "Run: cargo build --release"
  exit 1
fi

# Determine repos to run
case $MODE in
  quick)
    # Diagnostic subset: weakest 2 + strongest 1 (regression canary)
    REPOS="zod httpx cobra"
    echo "=== Quick eval: diagnostic subset (zod + httpx + cobra) ==="
    ;;
  full)
    REPOS="mini-redis hyperfine hono zod httpx cobra"
    echo "=== Full eval: all 6 repos ==="
    ;;
  single)
    REPOS="$SINGLE_REPO"
    echo "=== Single repo: $SINGLE_REPO ==="
    ;;
esac

echo "Provider: $PROVIDER | Profile: $PROFILE_NAME | Binary: $BINARY"
echo ""

# Build output suffix
SUFFIX="${PROFILE_NAME}"
if [[ -n "$TAG" ]]; then
  SUFFIX="${TAG}-${SUFFIX}"
fi

mkdir -p benchmarks/runs

FAILED=0
for repo in $REPOS; do
  case $repo in
    mini-redis|hyperfine) eval_file="benchmarks/cases/rust/${repo}.json" ;;
    hono|zod)             eval_file="benchmarks/cases/typescript/${repo}.json" ;;
    httpx)                eval_file="benchmarks/cases/python/${repo}.json" ;;
    cobra)                eval_file="benchmarks/cases/go/${repo}.json" ;;
    *)
      echo "WARN: Unknown repo '$repo', skipping"
      continue
      ;;
  esac

  if [[ ! -f "$eval_file" ]]; then
    echo "WARN: Eval file not found: $eval_file, skipping"
    continue
  fi

  output="benchmarks/runs/${repo}-${SUFFIX}.json"
  echo "--- $repo ---"
  if bun benchmarks/scripts/run-eval.ts \
    --binary "$BINARY" \
    --repo "$repo" \
    --profile "$PROFILE" \
    --eval "$eval_file" \
    --output "$output" \
    --provider "$PROVIDER" \
    --reuse-index; then
    echo ""
  else
    echo "FAILED: $repo"
    FAILED=$((FAILED + 1))
    echo ""
  fi
done

echo "=== Generating report ==="
bun benchmarks/scripts/report.ts --dir benchmarks/runs

if [[ $FAILED -gt 0 ]]; then
  echo ""
  echo "WARNING: $FAILED repo(s) failed"
  exit 1
fi
