# tqsdk — PolarSoul

## 设计哲学

tqsdk 是 Polarisor 生态中的量化策略研究沙箱，覆盖国内期货 + 加密双市场，从数据采集、回测、ML/DL/RL 研究、永续自进化优化到模拟/实盘交易。

- **策略沙箱**: 每个策略在隔离环境中运行，互不干扰
- **回测优先**: 策略开发以历史数据回测为起点，验证逻辑正确后才进入模拟/实盘
- **凭证隔离**: 期货账户凭证通过 PolarPrivate 代理管理，策略代码不接触明文密钥

## 功能介绍

- **生态位**: domain 层量化研究与交易平台
- **承担功能**（详见 [polaris.json](polaris.json)）:

| 编号 | 功能域 | 说明 |
|---|---|---|
| R1 | 双市场量化交易基础设施 | 期货/加密采集、事件驱动回测、Web UI、风控 |
| R2 | ML/DL/RL 研究平台 | LightGBM/XGBoost、LSTM/Transformer、PPO + 实验管理 |
| R3 | v4.0 RL PPO 改进 | DSR 奖励、增强观测、MambaFormer、课程学习 |
| R4 | 策略库持续进化 | 102+ 策略、永续自进化优化器、冠军保存、Lobster SDK |
| R5 | 实盘量化交易能力 | LiveScheduler、WebSocket 推送、SQLite 持久化、参数部署 |
| R6 | 庄家识别技术路线 | 5 种操纵模式检测 + 跟庄策略 |
| R7 | 攻防建模 | Lotka-Volterra 攻防力量估计与交易信号 |
| R8 | 自然语言策略研究工作台 | ResearchRun、artifact 导出、技能库、Swarm 审计委员会 |

## 与其他项目的关系

- **依赖 PolarPrivate**: 期货账户与 LLM 代理
- **依赖 PolarPort/sdk**: 服务发现与端口分配
- **依赖 PolarClaw/polarclaw-project-sdk**: Lobster SDK 事件写入
- **被依赖**: 无（生态叶子节点）

## 关键设计决策

### Why 独立项目而非 PolarClaw Skill

**问题**: 量化交易可以作为 PolarClaw 的一个 Skill 接入。

**决策**: 量化交易有独立的技术栈（Python + TqSdk + PyTorch），策略代码量大且频繁迭代，作为独立 domain 项目更清晰。

**不可妥协**: 交易密钥永远通过 PolarPrivate 代理，不在策略代码中硬编码。

## 依赖与被依赖

### 依赖

| 依赖项 | 说明 |
|---|---|
| PolarPrivate | 期货账户凭证 + LLM 代理 |
| PolarPort/sdk | 端口与服务注册 |
| PolarClaw/polarclaw-project-sdk | Lobster 事件适配 |

### 被依赖

无下游项目依赖。

---

## 详情入口

- [SSoT](polaris.json)
- [Roadmap](roadmap.md)
