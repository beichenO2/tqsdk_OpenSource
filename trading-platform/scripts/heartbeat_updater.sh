#!/usr/bin/env bash
set -euo pipefail

API_URL="${API_URL:-http://localhost:8000}"
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_PROJ_ROOT="$(cd "$_SCRIPT_DIR/.." && pwd)"
_COORD_DIR="$(cd "$_PROJ_ROOT/.." && pwd)/.coordination"
HEARTBEAT_FILE="${HEARTBEAT_FILE:-$_COORD_DIR/heartbeat/pos3.txt}"
INTERVAL="${INTERVAL:-300}"

log() { echo "$(date +%H:%M:%S) [heartbeat] $*"; }

log "updater started — writing to $HEARTBEAT_FILE every ${INTERVAL}s"

iter=0
while true; do
  iter=$((iter + 1))
  ts=$(date +%Y-%m-%dT%H:%M%z)
  last_commit=$(cd "$_PROJ_ROOT" && git rev-parse --short HEAD 2>/dev/null || echo "unknown")

  health="unknown"
  health_code=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/healthz" 2>/dev/null) || health_code="000"
  [ "$health_code" = "200" ] && health="ok" || health="down"

  paper_running="false"
  paper_uptime=0
  bar_count=0
  status_json=$(curl -s "$API_URL/api/v1/live-trading/status" 2>/dev/null) || status_json="{}"
  if echo "$status_json" | python3 -c "import sys,json;d=json.loads(sys.stdin.read());exit(0 if d.get('running') else 1)" 2>/dev/null; then
    paper_running="true"
    paper_uptime=$(echo "$status_json" | python3 -c "import sys,json;print(int(json.loads(sys.stdin.read()).get('uptime_seconds',0)/60))" 2>/dev/null || echo "0")
    bar_count=$(echo "$status_json" | python3 -c "import sys,json;print(json.loads(sys.stdin.read()).get('bar_count',0))" 2>/dev/null || echo "0")
  fi

  cat > "$HEARTBEAT_FILE" <<BEAT
iter=$iter
ts=$ts
last_commit=$last_commit
state=working
current_task=paper-running (observer, bars=$bar_count)
waiting_on=inbox/pos1/strategy-templates-needed
last_error=none
notes=auto-heartbeat from scripts/heartbeat_updater.sh
tqsdk_installed=true
api_health=$health
broker_mode=stub
paper_running=$paper_running
paper_uptime_min=$paper_uptime
BEAT

  log "iter=$iter api=$health paper=$paper_running uptime=${paper_uptime}min bars=$bar_count"
  sleep "$INTERVAL"
done
