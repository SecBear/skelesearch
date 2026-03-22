#!/usr/bin/env bash
set -euo pipefail

# install-pre-commit.sh — Install quick-check as a git pre-commit hook.
#
# The hook runs 2 cases per repo (~15s) on every commit.  If the binary or
# benchmark repos are absent the hook exits 0 silently — it only fires when
# the full toolchain is available locally.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HOOK_PATH="${PROJECT_ROOT}/.git/hooks/pre-commit"

if [[ ! -d "${PROJECT_ROOT}/.git" ]]; then
  echo "ERROR: Not a git repository: ${PROJECT_ROOT}"
  exit 1
fi

cat > "${HOOK_PATH}" << 'HOOK'
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"

# Skip silently when the toolchain isn't fully set up.
[[ -x "${REPO_ROOT}/target/release/skelesearch" ]] || exit 0
[[ -d "${REPO_ROOT}/benchmarks/repos/mini-redis" ]] || exit 0

"${REPO_ROOT}/benchmarks/scripts/quick-check.sh" --per-repo 2 || {
  echo ""
  echo "QUICK CHECK FAILED — retrieval quality regression detected."
  echo "Run ./benchmarks/scripts/quick-check.sh for details."
  exit 1
}
HOOK

chmod +x "${HOOK_PATH}"
echo "Pre-commit hook installed: ${HOOK_PATH}"
echo "Runs 2 cases/repo (~15s) on each commit."
echo "Remove with: rm ${HOOK_PATH}"
