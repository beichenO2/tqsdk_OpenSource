#!/usr/bin/env bash
set -euo pipefail

API_URL="${API_URL:-http://localhost:8000}"
POLL_INTERVAL="${POLL_INTERVAL:-300}"
COORDINATION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.coordination" 2>/dev/null && pwd)" \
  || COORDINATION_DIR=""

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [watchdog] $*"; }

check_healthz() {
  local code
  code=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/healthz" 2>/dev/null) || code="000"
  echo "$code"
}

check_live_trading_status() {
  curl -s "$API_URL/api/v1/live-trading/status" 2>/dev/null || echo '{"error":"unreachable"}'
}

write_bus_event() {
  local event="$1" summary="$2"
  if [ -n "$COORDINATION_DIR" ] && [ -f "$COORDINATION_DIR/bus.md" ]; then
    local ts
    ts=$(date +%Y-%m-%dT%H:%M%z)
    echo "| $ts | pos3 | $event | watchdog | $summary |" >> "$COORDINATION_DIR/bus.md"
  fi
}

log "starting — polling $API_URL every ${POLL_INTERVAL}s"

consecutive_failures=0

while true; do
  health_code=$(check_healthz)

  if [ "$health_code" = "200" ]; then
    if [ "$consecutive_failures" -gt 0 ]; then
      log "recovered after $consecutive_failures failures"
      write_bus_event "ack" "API recovered after $consecutive_failures poll failures"
    fi
    consecutive_failures=0
    log "healthz=$health_code ok"
  else
    consecutive_failures=$((consecutive_failures + 1))
    log "healthz=$health_code FAIL (consecutive=$consecutive_failures)"
    if [ "$consecutive_failures" -ge 3 ]; then
      write_bus_event "blocker" "API /healthz failed $consecutive_failures consecutive times (last=$health_code)"
      log "wrote blocker to bus.md"
    fi
  fi

  sleep "$POLL_INTERVAL"
done
