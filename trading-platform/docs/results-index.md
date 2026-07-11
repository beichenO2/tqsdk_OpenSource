# Results 文件索引

> **维护规则：** 每次新增或修改训练/优化脚本的输出文件时，必须在此文档中更新对应条目。  
> **Agent 查询原则：** 询问"最新结果"时，优先读取 `*_latest.json`，而非历史快照文件。

---

## 为什么需要这份文档

项目中存在多个平行的结果存储体系（`results/`、`eternal-optimizer/results*/`、`models/`、`apps/web/public/data/`），
历史上曾多次出现 Agent 查询时读到旧文件、把优化阶段结果当成最终结论的问题。

**根本原因是**：没有单一权威"最新"指针，多个同名相近的 JSON 没有任何元数据说明谁更新、谁是正式输出。

本文档建立了明确的文件层级规范：**`*_latest.json` = 权威最新，其他 = 历史快照或中间产物**。

> **2026-07-11 清理：** gate 全 reject / 回测全负的策略与产物已移除，见 git 历史。

---

## 一、`results/` 目录（主要查询入口）

| 文件 | 类型 | 由哪个脚本写入 | 说明 |
|------|------|--------------|------|
| `futures_backtest_report.json/csv` | 定期快照 | `run_futures_backtest.py` | 多策略期货回测汇总。 |
| `futures_backtest_test.json/csv` | 测试快照 | 同上（测试集） | 小规模验证用。 |
| `backtest_log.txt` | 运行日志 | 各脚本 stderr | ~1.2 GB 大文件，仅在排查问题时查阅。 |

---

## 二、`eternal-optimizer/results*/` 目录（永续优化器专属）

| 目录 | 内容 | 如何查最新 |
|------|------|-----------|
| `results/` | 1h 版本轮次结果，命名 `round_NNNN_YYYYMMDD_*.json` | 读 `eternal-optimizer/STATUS.json` 中对应 variant 的 `latest_round` |
| `results-5min/` | 5min 版本 | 同上 |
| `results-volbar/` | Volbar 版本 | 同上 |
| `results-futures/` | Futures 版本 | 同上；注意 `futures` variant 的 `timestamp` 字段曾为空（bug） |

> `eternal-optimizer/STATUS.json` 是永续优化器各 variant 的唯一 SSoT，优先读它。

---

## 三、`models/` 目录（策略参数快照）

| 文件模式 | 说明 |
|---------|------|
| `optuna_v4_results.json` | v4 策略 Optuna 100-trial 结果 |
| `optuna_v4_500trial.json` | v4 策略 500-trial 更充分搜索 |
| `optuna_v4_blend.json` | v4 融合参数 |
| `optuna_scalp_results.json` | Scalp 策略 Optuna |
| `strategy_*_best.json` | 各策略 eternal optimizer 当前最优参数 |
| `trade_analysis_report.md` | 交易分析叙述性报告 |

**注意：** `models/` 下无 latest 机制，查询时须按文件名中的版本/trial 数量判断新旧。

---

## 四、`apps/web/public/data/` 目录（前端静态快照）

| 文件 | 说明 |
|------|------|
| `optimizer.json` | 前端展示用，**非实时数据**，需手动同步 |
| `backtests.json` | 前端回测展示，同上 |

> 这两个文件是前端静态资产，不能当作"最新训练结果"来读。

---

## 五、查询路由速查

```
我想知道 eternal optimizer 最新状态 → eternal-optimizer/STATUS.json
我想知道某个 eternal 轮次详情     → eternal-optimizer/STATUS.json 查 latest_round，再读对应 results*/round_*.json
我想知道加密策略最新 OOS 结果     → .planning/STATE.md（人工维护的里程碑表）
```

---

## 六、维护协议

1. **脚本维护者**：每次修改训练/优化脚本的输出路径，必须同步更新本表。
2. **Agent 维护者**：若新增结果目录或文件，在本文档新增一行并注明 latest 机制。
3. **禁止行为**：不得将 `*_latest.json` 手动编辑为旧数据；latest 文件只能由脚本写入。
4. **过时文件处理**：历史快照保留（勿删），但在文件的 `_meta.superseded_by` 字段中注明被哪个文件替代（如适用）。

---

*本文档创建于 2026-04-28，用于解决"Agent 每次查询 tqsdk 训练结果都得到旧数据"的根因问题。*
