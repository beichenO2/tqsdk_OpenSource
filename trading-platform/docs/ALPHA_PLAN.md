# AlphaGate 验收方案

## 背景：1h 因子挖掘失败教训

上一轮 BTC 1h MCTS 因子挖掘产出若干"Surviving"信号，但策略回测暴露结构性缺陷：

| 问题 | 观测 |
|------|------|
| 换手过高 | 年化 54–146 倍 |
| 成本吞噬 | 成本占初始名义 2.7–7.3% |
| 单笔期望不足 | 平均单笔净收益 < 往返成本 |
| OOS 环境不利 | 单段测试期恰逢下跌，未做多段 walk-forward |

**结论**：不能再以 IC/IR 或单段 OOS 夏普作为 alpha 判据。必须先过验收闸门，再投入参数优化与 paper trading。

## AlphaGate 五道门

输入：因果化 `position ∈ [-1,1]`（滚动统计一律 `shift(1)`）、OHLCV、单边成本（默认 5bp）。

收益：`ret_net[t] = position[t-1] × Δclose[t] - |Δposition[t]| × one_way_cost`

| 门 | 判据 | 阈值 |
|----|------|------|
| G1 Walk-forward | 评估区间等分 4 段，各段独立算净收益 | ≥3/4 段净收益 > 0 |
| G2 成本敏感性 | 单边成本 2/5/10bp 重算总净收益 | 10bp 档总净收益 > 0 |
| G3 单笔期望 | 按仓位变化切分 trade，平均净收益 | ≥ 2×往返成本（bp） |
| G4 换手与成本 | 年化换手（4h 折算）+ 成本占比 | 换手 ≤ 100 且成本占比 ≤ 2% |
| G5 基准对比 | 策略净总收益 | > buy&hold 且 > supertrend(10,3) |

**Verdict**：5/5 → PASS；4/5 → MARGINAL；否则 REJECT。

## 四个假设因子（4h 主频）

| ID | 名称 | 经济逻辑 | 构造要点 |
|----|------|----------|----------|
| H1 | funding_contrarian | 永续资金费率极端时拥挤方向易反转 | funding zscore(30d) 取反；无真实 funding 数据则 SKIPPED |
| H2 | vol_adj_momentum | 7 天风险调整动量，高 vol 时降杠杆 | roc(42)/ts_std(ret,42) → 因果 zscore(180) → clip |
| H3 | daily_mean_reversion | 日线级别均值回归，4h 执行降频 | 1d 上 -zscore(close,30) → ffill 到 4h |
| H4 | short_high_momentum_4h | 24h 高动量反转（上轮主题降频复验） | -sign(roc(6)) × min(|zscore(roc,180)|, 1) |

## 后续阶段

**若 H1–H4 全 REJECT**：
1. 换信息源：真实资金费率历史、基差、订单流、持仓量
2. 放宽频率至 1d 纯方向，或 vol-bar 自适应采样
3. 组合层 dynamic_combine 替代单因子

**若有 PASS / MARGINAL**：
1. 参数邻域稳定性（window ±20%，cost ±5bp）
2. 更长历史 walk-forward（6+ 段）
3. 小资金 paper trading + 实时 funding 监控
4. 与 supertrend / buy&hold 组合做风险预算

## 运行

```bash
cd tqsdk/trading-platform
.venv/bin/python scripts/run_alpha_gate_btc.py --symbol BTCUSDT --cost-bps 5
```

输出：`output/research/alpha_gate_report_<ts>.md` + 同名 JSON。
