#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INSTALL_ROOT="${INSTALL_ROOT:-$HOME/.local}"
BIN_DIR="$INSTALL_ROOT/bin"
BIN_PATH="$BIN_DIR/skelesearch-mcp"
GLOBAL_MCP_JSON="$HOME/.omp/agent/mcp.json"

mkdir -p "$BIN_DIR"

echo "Installing skelesearch-mcp to $BIN_PATH"
cargo install --path "$ROOT/crates/mcp" --root "$INSTALL_ROOT" --force --features storage-rocksdb

echo
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
  echo "PATH note: $BIN_DIR is not on PATH for this shell."
  echo "Add this to your shell profile if you want to run skelesearch-mcp directly:"
  echo "  export PATH=\"$BIN_DIR:\$PATH\""
  echo
fi

echo "Global OMP MCP config file: $GLOBAL_MCP_JSON"
cat <<EOF

Recommended ~/.omp/agent/mcp.json entry:

{
  "mcpServers": {
    "skelesearch": {
      "command": "$BIN_PATH",
      "env": {
        "VOYAGE_API_KEY": "<set-me>",
        "SKELESEARCH_RERANKER": "local",
        "SKELESEARCH_RERANKER_MODEL_DIR": "$HOME/.cache/skelesearch/reranker/gte-modernbert-base",
        "RUST_LOG": "skelesearch=info"
      }
    }
  }
}

After updating the config, restart OMP.
EOF
