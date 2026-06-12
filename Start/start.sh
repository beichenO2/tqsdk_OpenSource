#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
# ── Dynamic port allocation via PolarPort ────────────
source "$PROJECT_DIR/../Agent_core/scripts/port-claim.sh"
PORT=$(claim_port "tqsdk-collector" "tqsdk" "18900")
PID_FILE="$SCRIPT_DIR/.pid"

cd "$PROJECT_DIR"

usage() {
    echo "Usage: bash Start/start.sh [start|stop|restart|status]"
}

is_running() {
    if [ -f "$PID_FILE" ]; then
        local OLD_PID
        OLD_PID=$(cat "$PID_FILE" 2>/dev/null || true)
        if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
            return 0
        fi
        rm -f "$PID_FILE"
    fi
    return 1
}

do_start() {
    # Idempotent: if already running on the expected port, report and exit
    OCCUPANT_PID=$(lsof -iTCP:"$PORT" -sTCP:LISTEN -P -n -t 2>/dev/null | head -1 || true)
    if [ -n "$OCCUPANT_PID" ]; then
        echo "pid=$OCCUPANT_PID"
        echo "port=$PORT"
        exit 0
    fi

    # Also check PID file
    if is_running; then
        OLD_PID=$(cat "$PID_FILE")
        echo "pid=$OLD_PID"
        echo "port=$PORT"
        exit 0
    fi

    # Install Python deps if needed
    if [ ! -d "trading-platform/.venv" ] && [ -z "${VIRTUAL_ENV:-}" ]; then
        echo "[tqsdk] Creating virtual environment..."
        python3 -m venv trading-platform/.venv
    fi

    # Activate virtual environment
    if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "trading-platform/.venv/bin/activate" ]; then
        # shellcheck disable=SC1091
        source trading-platform/.venv/bin/activate
    fi

    echo "[tqsdk] Installing dependencies..."
    pip install -r trading-platform/requirements.txt -q 2>&1 || { echo "Dependency installation failed" >&2; exit 1; }

    # Start daemon in background
    mkdir -p trading-platform/data
    nohup python trading-platform/apps/api/main.py > trading-platform/data/tqsdk.log 2>&1 &
    DAEMON_PID=$!
    echo "$DAEMON_PID" > "$PID_FILE"

    # Wait for port to become available (max 30s)
    for i in $(seq 1 30); do
        if lsof -iTCP:"$PORT" -sTCP:LISTEN -P -n -t >/dev/null 2>&1; then
            ACTUAL_PID=$(lsof -iTCP:"$PORT" -sTCP:LISTEN -P -n -t 2>/dev/null | head -1 || echo "$DAEMON_PID")
            echo "pid=$ACTUAL_PID"
            echo "port=$PORT"
            exit 0
        fi
        sleep 1
    done

    echo "Timed out waiting for port $PORT" >&2
    rm -f "$PID_FILE"
    exit 1
}

do_stop() {
    if ! is_running; then
        echo "Not running"
        # Also try to free the port if something is lingering
        OCCUPANT_PID=$(lsof -iTCP:"$PORT" -sTCP:LISTEN -P -n -t 2>/dev/null | head -1 || true)
        if [ -n "$OCCUPANT_PID" ]; then
            echo "Killing lingering process on port $PORT (pid=$OCCUPANT_PID)"
            kill -TERM "$OCCUPANT_PID" 2>/dev/null || true
            for i in $(seq 1 10); do
                if ! kill -0 "$OCCUPANT_PID" 2>/dev/null; then
                    break
                fi
                sleep 1
            done
            kill -9 "$OCCUPANT_PID" 2>/dev/null || true
        fi
        exit 0
    fi

    OLD_PID=$(cat "$PID_FILE")
    echo "Stopping pid=$OLD_PID..."
    kill -TERM "$OLD_PID" 2>/dev/null || true

    # Wait for process to exit (max 10s)
    for i in $(seq 1 10); do
        if ! kill -0 "$OLD_PID" 2>/dev/null; then
            break
        fi
        sleep 1
    done

    # Force kill if still alive
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Force killing pid=$OLD_PID"
        kill -9 "$OLD_PID" 2>/dev/null || true
    fi

    rm -f "$PID_FILE"
    echo "Stopped"
}

do_restart() {
    do_stop
    do_start
}

do_status() {
    if is_running; then
        OLD_PID=$(cat "$PID_FILE")
        echo "Running (pid=$OLD_PID, port=$PORT)"
        exit 0
    fi

    # Check if something is on the port anyway
    OCCUPANT_PID=$(lsof -iTCP:"$PORT" -sTCP:LISTEN -P -n -t 2>/dev/null | head -1 || true)
    if [ -n "$OCCUPANT_PID" ]; then
        echo "Running on port $PORT (pid=$OCCUPANT_PID) — PID file out of sync"
        exit 0
    fi

    echo "Not running"
    exit 1
}

COMMAND="${1:-start}"

case "$COMMAND" in
    start)   do_start   ;;
    stop)    do_stop    ;;
    restart) do_restart ;;
    status)  do_status  ;;
    *)       usage; exit 1 ;;
esac
