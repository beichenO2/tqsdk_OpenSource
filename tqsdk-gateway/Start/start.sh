#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
GATEWAY_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
POLARPORT_URL=${POLARPORT_URL:-http://127.0.0.1:11050}
POLARPROCESS_URL=${POLARPROCESS_URL:-http://127.0.0.1:11055}
SERVICE_ID=tqsdk-gateway
PREFERRED_PORT=12890
PYTHON_BIN=${TQSDK_PYTHON_BIN:-/opt/homebrew/Caskroom/miniforge/base/bin/python}

if [ "$#" -ne 0 ]; then
  echo "tqsdk gateway lifecycle is managed by PolarProcess; do not pass arguments" >&2
  exit 2
fi
if [ ! -x "$PYTHON_BIN" ]; then
  echo "tqsdk Python executable missing: $PYTHON_BIN" >&2
  exit 1
fi
if ! curl -fsS --max-time 3 "$POLARPORT_URL/api/health" >/dev/null; then
  echo "PolarPort is unavailable; refusing preferred-port fallback" >&2
  exit 1
fi
if ! curl -fsS --max-time 3 "$POLARPROCESS_URL/api/health" >/dev/null; then
  echo "PolarProcess is unavailable; refusing unmanaged service start" >&2
  exit 1
fi

service_status=$(curl -fsS --max-time 3 "$POLARPROCESS_URL/api/services/$SERVICE_ID" | jq -r '.status')
case "$service_status" in
  starting|running) ;;
  *)
    echo "$SERVICE_ID may only be launched by its exact PolarProcess start action" >&2
    exit 1
    ;;
esac

source "$HOME/Polarisor/Agent_core/scripts/port-claim.sh"
PORT=$(claim_port "tqsdk-gateway" "tqsdk" 12890)
if [ "$PORT" -ne "$PREFERRED_PORT" ]; then
  release_port "$PORT"
  echo "PolarPort returned $PORT, but tqsdk-gateway requires $PREFERRED_PORT" >&2
  exit 1
fi

cd "$GATEWAY_DIR"
export PORT
export TQSDK_GATEWAY_PORT=$PORT
export POLAR_RUNTIME_MANAGED=1
exec "$PYTHON_BIN" main.py
