# 夜盘全链路 QA 设计（TDD 驱动）

> 场景：国内期货夜盘开市（21:00+），模拟盘账户连接实盘行情数据  
> 策略：先定义验收用例（RED），再修代码（GREEN），最后回归（REFACTOR）  
> 日期：2026-07-09

## 1. 验收目标

开市期间，用户从 Web UI 到 TqSdk Gateway 的整条链路应满足：

1. **实时行情**：`message=ok`，非 `closed_market_cache`
2. **账户快照**：`stale=false`，权益曲线持续追加
3. **风控前置**：RiskGate 可探测、可拦截
4. **系统健康**：gateway / execution / risk_gate 全绿
5. **无锁饥饿**：gateway 并发 quote 不返回 `session busy`

## 2. 分层测试矩阵

| 层级 | 组件 | 端口 | 用例 ID | 验收标准 | 测试文件 |
|------|------|------|---------|----------|----------|
| L0 | Gateway 锁模型 | 12890 | G-01 | `wait_update` 不阻塞 `get_quote` | `tqsdk-gateway/tests/test_session_lock.py` |
| L1 | Gateway 健康 | 12890 | G-02 | `/health` connected=true | `tests/e2e/test_live_chain.py` |
| L1 | Gateway 行情 | 12890 | G-03 | quote 200，无 session busy | 同上 |
| L1 | Gateway 账户 | 12890 | G-04 | account 200，balance>0 | 同上 |
| L2 | API 健康 | 8600 | A-01 | `/healthz` + `/readyz` ok | 同上 |
| L2 | API 系统聚合 | 8600 | A-02 | `/api/v1/system/health` status=ok | 同上 |
| L2 | API 行情代理 | 8600 | A-03 | quote `message=ok`，last_price 实时 | 同上 |
| L2 | API 账户代理 | 8600 | A-04 | account `stale=false` | 同上 |
| L2 | API 权益历史 | 8600 | A-05 | pnl-history 有新点（开市后） | 同上 |
| L3 | 风控链路 | 8600 | R-01 | risk-probe 合法单 allowed=true | 同上 |
| L3 | 风控链路 | 8600 | R-02 | risk-probe 超限单 allowed=false | 同上 |
| L3 | Paper 下单 | 8600 | R-03 | paper order ACCEPTED_PAPER | 同上 |
| L4 | 研究链路 | 8600 | F-01 | factors list 非空 | 同上 |
| L4 | 研究链路 | 8600 | F-02 | analyze-cs 返回 ic_mean | 同上 |
| L5 | WebSocket | 8600 | W-01 | `/ws` 连接 + pong | 同上 |

## 3. TDD 执行顺序

```
① RED   — 写 test_session_lock + test_live_chain（当前应失败：G-03/G-04/A-03/A-04）
② GREEN — 修 gateway session：update_loop 释放锁
③ VERIFY— 重启 gateway → pytest tests/e2e -m live
④ REFACTOR — 若 account stale 仍 true，检查 snapshot_loop 超时
```

## 4. 关键失败判定

| 症状 | 根因假设 | 对应用例 |
|------|----------|----------|
| `session busy` | wait_update 占锁 | G-01, G-03 |
| `closed_market_cache` 开市时 | gateway 超时回退 | A-03 |
| `stale: true` | gateway account 503 | A-04 |
| pnl 不增长 | snapshot_loop 被 skip | A-05 |
| risk 422 | 限价/交割月拦截 | R-02（预期失败） |

## 5. 运行命令

```bash
# 单元：gateway 锁（无需服务）
cd tqsdk/tqsdk-gateway && python -m pytest tests/test_session_lock.py -v

# E2E：需 gateway + API 运行
cd tqsdk/trading-platform
pytest tests/e2e/test_live_chain.py -m live -v

# 全量回归（跳过 live）
pytest tests/ -m "not live" -q
```

## 6. 非目标（本轮不测）

- 真金白银 live 下单（`LIVE_TRADING_ENABLED=false`）
- 加密 WEEX 链路
- data-collector 18900（当前未启动）
- 前端 Playwright UI 自动化（后续轮次）
