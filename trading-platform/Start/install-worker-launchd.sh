#!/usr/bin/env bash
# Install strategy worker as a launchd user agent.
# Usage:
#   ./Start/install-worker-launchd.sh          # install + load
#   ./Start/install-worker-launchd.sh uninstall  # unload + remove
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PLIST_SRC="$SCRIPT_DIR/com.tqtrader.strategy-worker.plist"
PLIST_DST="$HOME/Library/LaunchAgents/com.tqtrader.strategy-worker.plist"
LABEL="com.tqtrader.strategy-worker"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"

if [[ "${1:-}" == "uninstall" ]]; then
    launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
    rm -f "$PLIST_DST"
    echo "Uninstalled $LABEL"
    exit 0
fi

mkdir -p "$HOME/Library/LaunchAgents" "$PROJECT_ROOT/logs"
sed \
    -e "s|__PROJECT_ROOT__|$PROJECT_ROOT|g" \
    -e "s|__PYTHON_BIN__|$PYTHON_BIN|g" \
    -e "s|__HOME_DIR__|$HOME|g" \
    "$PLIST_SRC" > "$PLIST_DST"

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST_DST"
launchctl enable "gui/$(id -u)/$LABEL"
launchctl kickstart -k "gui/$(id -u)/$LABEL"

echo "Installed and started $LABEL"
echo "  plist: $PLIST_DST"
echo "  logs:  $PROJECT_ROOT/logs/strategy-worker-*.log"
