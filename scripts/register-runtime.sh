#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
POLARPORT_URL=${POLARPORT_URL:-http://127.0.0.1:11050}
POLARPROCESS_URL=${POLARPROCESS_URL:-http://127.0.0.1:11055}

curl -fsS --max-time 3 "$POLARPORT_URL/api/health" >/dev/null
curl -fsS --max-time 3 "$POLARPROCESS_URL/api/health" >/dev/null

# Replace only this project's legacy collector identity. The gateway identity is
# already canonical and remains reserved without being started.
curl -fsS -X DELETE "$POLARPORT_URL/api/ports/reserve/tqsdk-collector/tqsdk" >/dev/null

reserve() {
  local service_id=$1 preferred_port=$2
  curl -fsS -X POST "$POLARPORT_URL/api/ports/reserve" \
    -H 'Content-Type: application/json' \
    -d "{\"service_name\":\"$service_id\",\"project\":\"tqsdk\",\"preferred_port\":$preferred_port}" >/dev/null
}

register() {
  local service_id=$1 name=$2 command=$3 port=$4 health_url=$5 auto_start=$6 max_restarts=$7
  local payload
  payload=$(jq -n \
    --arg id "$service_id" \
    --arg name "$name" \
    --arg command "$command" \
    --arg work_dir "$PROJECT_DIR" \
    --arg health_url "$health_url" \
    --argjson port "$port" \
    --argjson auto_start "$auto_start" \
    --argjson max_restarts "$max_restarts" \
    '{
      id: $id,
      name: $name,
      command: $command,
      work_dir: $work_dir,
      device_id: "any",
      auto_start: $auto_start,
      restart_on_failure: true,
      max_restarts: $max_restarts,
      port: $port,
      health_check_url: $health_url,
      start_script_dir: "-"
    }')
  curl -fsS -X POST "$POLARPROCESS_URL/api/services/register" \
    -H 'Content-Type: application/json' \
    -d "$payload" >/dev/null
}

reserve tqsdk-data-collector 18900
reserve tqsdk-gateway 12890
register tqsdk-data-collector "TqSdk Data Collector" "bash Start/start.sh" 18900 "http://127.0.0.1:18900/health" true 10
register tqsdk-gateway "TqSdk Credential Gateway" "bash tqsdk-gateway/Start/start.sh" 12890 "http://127.0.0.1:12890/health" false 10

echo "Registered tqsdk-data-collector (auto-start) and tqsdk-gateway (stopped policy); no lifecycle action was requested"
