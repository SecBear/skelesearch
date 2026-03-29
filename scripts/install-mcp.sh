#!/usr/bin/env bash
# Prerequisite: Nix with flakes enabled
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INSTALL_ROOT="${INSTALL_ROOT:-$HOME/.local}"
BIN_DIR="$INSTALL_ROOT/bin"
STATE_DIR="$INSTALL_ROOT/share/skelesearch"
GCROOT_DIR="$STATE_DIR/nix"
MCP_BIN_PATH="$BIN_DIR/skelesearch-mcp"
DAEMON_BIN_PATH="$BIN_DIR/skelesearchd"
MCP_ROOT_LINK="$GCROOT_DIR/skelesearch-mcp"
DAEMON_ROOT_LINK="$GCROOT_DIR/skelesearch-daemon"
ORT_ROOT_LINK="$GCROOT_DIR/onnxruntime-lib"
GLOBAL_MCP_JSON="$HOME/.omp/agent/mcp.json"

if ! command -v nix >/dev/null 2>&1; then
  echo "install-mcp.sh requires Nix with flakes enabled" >&2
  exit 2
fi

mkdir -p "$BIN_DIR" "$GCROOT_DIR"

NIX_FLAGS=(--extra-experimental-features 'nix-command flakes')

echo "Building reproducible skelesearch packages via Nix flake outputs"
nix "${NIX_FLAGS[@]}" build --out-link "$MCP_ROOT_LINK" "$ROOT#skelesearch-mcp" >/dev/null
nix "${NIX_FLAGS[@]}" build --out-link "$DAEMON_ROOT_LINK" "$ROOT#skelesearch-daemon" >/dev/null
nix "${NIX_FLAGS[@]}" build --out-link "$ORT_ROOT_LINK" "$ROOT#onnxruntime-lib" >/dev/null

MCP_OUT="$MCP_ROOT_LINK"
DAEMON_OUT="$DAEMON_ROOT_LINK"
ORT_OUT="$ORT_ROOT_LINK"

write_wrapper() {
  local target="$1"
  local exe="$2"
  local extra_env="${3:-}"
  cat > "$target" <<EOF
#!/usr/bin/env bash
export DYLD_LIBRARY_PATH="$ORT_OUT/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="$ORT_OUT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
${extra_env}
exec "$exe" "\$@"
EOF
  chmod +x "$target"
}

write_wrapper \
  "$MCP_BIN_PATH" \
  "$MCP_OUT/bin/skelesearch-mcp" \
  "export SKELESEARCH_DAEMON_PROGRAM=\"$DAEMON_BIN_PATH\""
write_wrapper "$DAEMON_BIN_PATH" "$DAEMON_OUT/bin/skelesearchd"

if [[ ! -x "$MCP_BIN_PATH" || ! -x "$DAEMON_BIN_PATH" ]]; then
  echo "failed to install skelesearch binaries into $BIN_DIR" >&2
  exit 1
fi

echo "Installed wrapper skelesearch-mcp -> $MCP_OUT/bin/skelesearch-mcp"
echo "Installed wrapper skelesearchd   -> $DAEMON_OUT/bin/skelesearchd"
echo "Wrapped ONNX Runtime library dir -> $ORT_OUT/lib"
echo "Pinned GC roots under            -> $GCROOT_DIR"
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
      "command": "$MCP_BIN_PATH",
      "env": {
        "VOYAGE_API_KEY": "<set-me>",
        "RUST_LOG": "skelesearch=info"
      }
    }
  }
}

After updating the config, restart OMP.

Note: 'skelesearch-mcp' auto-starts the sibling 'skelesearchd' binary when daemon-backed
tools such as 'index' and 'get_index_status' are called. Installing both binaries into the
same bin directory keeps that path working.
EOF
