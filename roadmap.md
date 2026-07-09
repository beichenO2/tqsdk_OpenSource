# tqsdk Roadmap

> 进度视图：当前阶段、完成情况、下一步。事实源是 `polaris.json`，本文件只做进度摘要。

## 当前状态

| 维度 | 状态 |
| --- | --- |
| 版本 | 3.0 |
| 项目状态 | active |

## Requirement 完成情况

| ID | 名称 | 完成度 | 说明 |
| --- | --- | --- | --- |
| R1 | 双市场量化交易基础设施 | 100% | 5/5 done |
| R2 | ML/DL/RL 研究平台 | 100% | 4/4 done（RL PPO 2026-05-05 验证 Return=1.53%） |
| R3 | v4.0 RL PPO 改进 | 100% | 4/4 done（MambaFormer + DSR + 课程学习） |
| R4 | 策略库持续进化 | 100% | 5/5 done（含永续优化器 LLM 404 降级 + 冠军保存） |
| R5 | 实盘量化交易能力 | 100% | 8/8 done（含 2026-07-09 TqSdk Gateway 凭证隔离） |
| R6 | 庄家识别技术路线 | 100% | 3/3 done |
| R7 | 攻防建模 | 100% | 2/2 done |
| R8 | 自然语言策略研究工作台 | 100% | 4/4 done |
| R9 | 网站与策略对接 | 100% | 4/4 done（原重复编号 R5，2026-07-09 更正为 R9） |
| R10 | 策略/因子开发平台 | 0% | planned — 见 `trading-platform/docs/prd/web-v2-and-quant-platform.md` |
| R11 | Web v2 + 实盘贯通 | 100% | RiskGate + Live 下单 + 四工作区 IA + ⌘K done |

## 已知阻塞项

无（与 `polaris.json` 一致）。

## 2026-07-09 赛马合并批次（gnhf）

| 步骤 | 内容 | Commit |
| --- | --- | --- |
| 1 | TqSdk Gateway 凭证隔离（gateway 全套 + bootstrap） | `52ddf0e` |
| 2 | broker/采集全面经 gateway，移除 binance/okx | `56098c1` |
| 3 | merge `gnhf/pos1-init`（赛马超集：pos2/3/4 已被 pos1 合并） | `49ee89f` |
| 4 | 抢救 pos2 未提交：主力合约过滤 + whale_detector 结果 + 排行榜 | `a60475e` |

**赛马结论**：pos1 = 超集（含 intraday_reversal A/B/C、复利 blow-up 修复、11 个复活策略、
packages/research + MCP router、ResearchRuns 前端页、26 个集成测试）。
**worktrees 已删除**：`tqsdk-gnhf-worktrees/` 四个 worktree 已 `git worktree remove`，
未跟踪产物归档 `~/Desktop/ClawBin/tqsdk/2026-07-09-gnhf-worktrees/`。
分支 ref `gnhf/pos{1-4}-init`（本地+远端）按 Proto-C 保留。

## 2026-05-25 批次完成清单

| 步骤 | 内容 | Commit |
| --- | --- | --- |
| 1 | SSoT 全量同步：roadmap/PolarSoul/PROJECT_OVERVIEW 对齐 polaris.json | `2f0179a` |
| 2 | 过时文档归档 ClawBin（5 项）+ 引用清理 | `aa344c8` |
| 3 | tqsdk pin 3.9.2 + smoke 入口 + optimizer import 修复 | `164298d` |
| 4 | smoke 扩展至 API/RL 路径（5/5 通过） | `9b2a7ee` |

**归档位置**：`~/Desktop/ClawBin/tqsdk/2026-05-25-outdated-docs/`

**现行文档入口**：`README.md` → `polaris.json` → `roadmap.md` → `PolarSoul.md`

## 下一步

按 PRD `trading-platform/docs/prd/web-v2-and-quant-platform.md` 5 阶段推进：

1. ~~Phase 0 合并与清理~~（2026-07-09 完成）
2. Phase 1 因子子系统（R10：packages/factor + IC 分析 + /factors API）
3. Phase 2 统一研究工作流（R10：pipeline + promote）
4. ~~Phase 3 Web v2 信息架构~~（2026-07-09：四工作区 + ⌘K + 旧路由兼容）
5. ~~Phase 4 前端↔实盘贯通~~（2026-07-09：RiskGate + live confirm + Live 下单页）
6. Phase 5 LLM 因子进化（R10 增强，可选）

另：polaris.json 中 `test_status: not_tested` 的 feature 逐项补集成测试。

## 更新记录

| 日期 | 更新内容 |
| --- | --- |
| 2026-07-09 | Phase 3 Web v2 IA：四工作区导航 + ⌘K CommandPalette + 市场上下文过滤 + ResearchRuns 路由
| 2026-07-09 | Phase 4 RiskGate：期货专项闸 + LIVE 双闸 + /order|/risk-probe + Live 页下单；23 unit tests passed
| 2026-07-09 | gnhf 赛马合并：pos1 超集并入 main + Gateway 凭证隔离入库 + worktrees 删除归档；新增 R10/R11 规划（因子平台 + Web v2）；修正重复 R5→R9 |
| 2026-05-26 | WF 过拟合门控修复：多资产 common-era 对齐（2020-09+）+ deploy 硬门控 + wf_robust_search 回退 |
| 2026-05-26 | Phase B/C：策略多态切换 + Idea→Factor→Eval agentic loop + deploy 后自动过拟合验证 |
| 2026-05-25 | Phase A 盈利赛马：volbar 冠军部署 + paper probe SOL +100%/BTC +67% |
| 2026-04-29 | R4 冠军保存体系完成：champion_archive.py + 历史最佳恢复 + 三变体集成 |
| 2026-04-29 | 初始创建：从 polaris.json 提取进度信息 |
