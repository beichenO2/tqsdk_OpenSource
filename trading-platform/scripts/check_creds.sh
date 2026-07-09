#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PRIVPORTAL_URL="${PRIVPORTAL_URL:-http://127.0.0.1:12790}"

log() { echo "$(date +%H:%M:%S) $*"; }

log "=== PolarPrivate 凭证健康检查 ==="
log "URL: $PRIVPORTAL_URL"
echo ""

vault_status=$(curl -s "$PRIVPORTAL_URL/api/vault/status" 2>/dev/null) || {
  log "FAIL: PolarPrivate 不可达 ($PRIVPORTAL_URL)"
  echo ""
  log "降级方案: 设置以下环境变量"
  log "  TQ_AUTH_EMAIL=<快期账号>"
  log "  TQ_AUTH_PASSWORD=<快期密码>"
  exit 1
}

locked=$(echo "$vault_status" | python3 -c "import sys,json; print(json.loads(sys.stdin.read()).get('locked', True))" 2>/dev/null)

if [ "$locked" = "True" ]; then
  log "WARN: Vault 已锁定"
  echo ""
  log "请解锁: curl -X POST $PRIVPORTAL_URL/api/vault/unlock -H 'Content-Type: application/json' -d '{\"master_password\":\"...\"}'"
  exit 1
fi

log "Vault 状态: 已解锁 ✓"

check_key() {
  local key="$1" label="$2"
  local resp
  resp=$(curl -s "$PRIVPORTAL_URL/api/secrets?q=$key&limit=5" 2>/dev/null) || { log "  $label: 查询失败"; return 1; }
  local count
  count=$(echo "$resp" | python3 -c "import sys,json; items=json.loads(sys.stdin.read()).get('items',[]); print(sum(1 for i in items if i['key']=='$key'))" 2>/dev/null)
  if [ "$count" = "1" ]; then
    log "  $label: ✓"
    return 0
  else
    log "  $label: MISSING"
    return 1
  fi
}

echo ""
log "--- TqSdk 快期凭证 ---"
ok=0
check_key "exchange.tqsdk.auth_user" "快期账号" && ok=$((ok+1))
check_key "exchange.tqsdk.auth_password" "快期密码" && ok=$((ok+1))

log "--- TqSdk 期货公司凭证 ---"
check_key "exchange.tqsdk.broker" "期货公司" && ok=$((ok+1))
check_key "exchange.tqsdk.account" "期货账号" && ok=$((ok+1))
check_key "exchange.tqsdk.password" "期货密码" && ok=$((ok+1))

echo ""
log "凭证检查完成: $ok/5 项通过"
