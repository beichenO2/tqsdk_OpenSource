#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)

fail() {
  printf 'FAIL: %s\n' "$*" >&2
  exit 1
}

assert_contains() {
  local file=$1 text=$2
  grep -Fq "$text" "$file" || fail "$file does not contain $text"
}

assert_not_contains() {
  local file=$1 pattern=$2
  if grep -En "$pattern" "$file"; then
    fail "$file contains forbidden runtime behavior"
  fi
}

collector="$ROOT/Start/start.sh"
gateway="$ROOT/tqsdk-gateway/Start/start.sh"

for launcher in "$collector" "$gateway"; do
  [ -x "$launcher" ] || fail "$launcher must exist and be executable"
  assert_contains "$launcher" '127.0.0.1:11050'
  assert_contains "$launcher" '127.0.0.1:11055'
  assert_contains "$launcher" '/api/health'
  assert_contains "$launcher" 'port-claim.sh'
  assert_contains "$launcher" 'claim_port'
  assert_contains "$launcher" 'release_port'
  assert_contains "$launcher" 'exec '
  assert_contains "$launcher" 'starting|running)'
  assert_not_contains "$launcher" '(^|[[:space:]])(nohup|disown|pkill|killall|kill|lsof)([[:space:]]|$)|PID_FILE|[^&]&[[:space:]]*$'
done

assert_contains "$collector" 'tqsdk-data-collector'
assert_contains "$collector" 'claim_port "tqsdk-data-collector" "tqsdk" 18900'
assert_contains "$collector" 'TQSDK_COLLECTOR_PORT=$PORT'
assert_contains "$collector" 'data-collector'
assert_contains "$collector" 'main.py'

assert_contains "$gateway" 'tqsdk-gateway'
assert_contains "$gateway" 'claim_port "tqsdk-gateway" "tqsdk" 12890'
assert_contains "$gateway" 'TQSDK_GATEWAY_PORT=$PORT'
assert_contains "$gateway" 'main.py'

register="$ROOT/scripts/register-runtime.sh"
assert_contains "$register" 'start_script_dir: "-"'
assert_contains "$register" '/api/ports/reserve'
assert_contains "$register" '/api/ports/reserve/tqsdk-collector/tqsdk'
assert_contains "$register" 'tqsdk-data-collector'
assert_contains "$register" 'tqsdk-gateway'
assert_not_contains "$register" 'api/services/.*/(start|stop|restart)'
assert_not_contains "$register" 'command:.*--port'

jq -e '
  .service_management.service_id == "tqsdk-data-collector" and
  .service_management.start_command == "bash Start/start.sh" and
  .service_management.auto_start == true and
  (.service_management.services | length) == 2 and
  ([.service_management.services[] | .service_id] | sort) == ["tqsdk-data-collector", "tqsdk-gateway"] and
  ([.service_management.services[] | .preferred_port] | sort) == [12890, 18900] and
  ([.service_management.services[] | .auto_start] | sort) == [false, true]
' "$ROOT/polaris.json" >/dev/null || fail "polaris.json does not declare both governed services"

jq -e '
  .requirements[]
  | select(.id == "R12")
  | .features[]
  | select(.name == "runtime_governance")
  | .status == "in-progress" or .status == "tested" or .status == "done"
' "$ROOT/polaris.json" >/dev/null || fail "runtime_governance SSoT is missing"

skill="$ROOT/PolarSkills/tqsdk-ops/SKILL.md"
assert_contains "$skill" 'name: tqsdk-ops'
assert_contains "$skill" 'Use when'
assert_contains "$skill" 'PolarProcess'
assert_contains "$skill" 'PolarPort'
assert_contains "$skill" 'tqsdk-data-collector'
assert_contains "$skill" 'tqsdk-gateway'

for doc in \
  "$ROOT/README.md" \
  "$ROOT/PolarSkills/tqsdk-ops/SKILL.md" \
  "$ROOT/PolarSkills/tqsdk-ops/DEPLOY.md" \
  "$ROOT/PolarSkills/tqsdk-ops/TROUBLESHOOT.md"; do
  assert_contains "$doc" '127.0.0.1:11055'
  assert_not_contains "$doc" '^[[:space:]]*(python(3)? (collector|main)\.py|pgrep|pkill|killall|nohup)|Start/start\.sh (start|stop|restart)'
done

printf 'tqsdk runtime governance contract passed\n'
