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
| R5 | 实盘量化交易能力 | 100% | 7/7 done |
| R6 | 庄家识别技术路线 | 100% | 3/3 done |
| R7 | 攻防建模 | 100% | 2/2 done |
| R8 | 自然语言策略研究工作台 | 100% | 4/4 done |

## 已知阻塞项

无（与 `polaris.json` 一致）。

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

roadmap 待办已全部完成。后续可按 polaris.json 中 `test_status: not_tested` 的 feature 逐项补集成测试。

## 更新记录

| 日期 | 更新内容 |
| --- | --- |
| 2026-05-26 | WF 过拟合门控修复：多资产 common-era 对齐（2020-09+）+ deploy 硬门控 + wf_robust_search 回退 |
| 2026-05-26 | Phase B/C：策略多态切换 + Idea→Factor→Eval agentic loop + deploy 后自动过拟合验证 |
| 2026-05-25 | Phase A 盈利赛马：volbar 冠军部署 + paper probe SOL +100%/BTC +67% |
| 2026-04-29 | R4 冠军保存体系完成：champion_archive.py + 历史最佳恢复 + 三变体集成 |
| 2026-04-29 | 初始创建：从 polaris.json 提取进度信息 |
