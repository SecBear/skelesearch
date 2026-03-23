#!/usr/bin/env bash
set -euo pipefail

# Re-index all 6 eval repos with Voyage.
# Usage: source .env && export VOYAGE_API_KEY && ./benchmarks/scripts/reindex.sh

cd "$(dirname "$0")/../.."
BINARY="./target/release/skelesearch"

if [[ ! -x "$BINARY" ]]; then
  echo "ERROR: Binary not found. Run: cargo build --release --features storage-sqlite"
  exit 1
fi

if [[ -z "${VOYAGE_API_KEY:-}" ]]; then
  echo "ERROR: VOYAGE_API_KEY not set. Run: source .env && export VOYAGE_API_KEY"
  exit 1
fi

for repo in mini-redis hyperfine cobra httpx hono zod; do
  echo "--- $repo ---"
  rm -rf "benchmarks/repos/$repo/.skelesearch"
  RUST_LOG=skelesearch=warn "$BINARY" index "benchmarks/repos/$repo" --provider voyage
  echo ""
done

echo "Done. Run ./benchmarks/scripts/eval.sh to measure."
