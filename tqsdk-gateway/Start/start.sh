#!/usr/bin/env bash
# TqSdk Gateway lifecycle — credentials isolated in this process only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PID_FILE="$SCRIPT_DIR/.pid"
SERVICE_NAME="tqsdk-gateway"
PROJECT="tqsdk"
PREFERRED_PORT=12891
LOG_FILE="$SCRIPT_DIR/tqsdk-gateway.log"

source "$PROJECT_DIR/../Agent_core/scripts/port-claim.sh"
PORT=$(claim_port "$SERVICE_NAME" "$PROJECT" "$PREFERRED_PORT")
HEALTH_URL="http://127.0.0.1:${PORT}/health"

cd "$PROJECT_DIR"

do_start() {
  OCCUPANT_PID=$(lsof -iTCP:"$PORT" -sTCP:LISTEN -P -n -t 2>/dev/null | head -1 || true)
  if [ -n "$OCCUPANT_PID" ]; then
    echo "Already running pid=$OCCUPANT_PID port=$PORT"
    exit 0
  fi

  export TQSDK_GATEWAY_HOST=127.0.0.1
  export TQSDK_GATEWAY_PORT="$PORT"

  nohup python3 main.py >> "$LOG_FILE" 2>&1 &
  DAEMON_PID=$!
  echo "$DAEMON_PID" > "$PID_FILE"

  for _ in $(seq 1 30); do
    if curl -sf "$HEALTH_URL" >/dev/null 2>&1; then
      echo "Started pid=$DAEMON_PID port=$PORT"
      exit 0
    fi
    if ! kill -0 "$DAEMON_PID" 2>/dev/null; then
      echo "Process exited immediately" >&2
      tail -20 "$LOG_FILE" >&2 || true
      rm -f "$PID_FILE"
      exit 1
    fi
    sleep 1
  done

  echo "Timed out waiting for $HEALTH_URL" >&2
  rm -f "$PID_FILE"
  exit 1
}

do_stop() {
  local pids=""
  if [ -f "$PID_FILE" ]; then pids="$(cat "$PID_FILE" 2>/dev/null || true)"; fi
  pids="$pids $(lsof -iTCP:"$PORT" -sTCP:LISTEN -P -n -t 2>/dev/null || true)"
  pids=$(printf '%s\n' $pids | grep -E '^[0-9]+$' | sort -u || true)
  if [ -z "$pids" ]; then
    echo "Not running"
    rm -f "$PID_FILE"
    exit 0
  fi
  for p in $pids; do kill "$p" 2>/dev/null || true; done
  rm -f "$PID_FILE"
  echo "Stopped"
}

do_status() {
  local occ
  occ=$(lsof -iTCP:"$PORT" -sTCP:LISTEN -P -n -t 2>/dev/null | head -1 || true)
  if [ -n "$occ" ]; then
    echo "Running pid=$occ port=$PORT"
    exit 0
  fi
  echo "Not running"
  exit 1
}

case "${1:-start}" in
  start) do_start ;;
  stop) do_stop ;;
  restart) do_stop; do_start ;;
  status) do_status ;;
  *) echo "Usage: bash Start/start.sh [start|stop|restart|status]" >&2; exit 1 ;;
esac
