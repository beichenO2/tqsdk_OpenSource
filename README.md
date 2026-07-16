# tqsdk

国内期货 + 加密双市场量化交易与研究平台，Polarisor 生态的 domain 层项目。

## 定位

tqsdk 是 Polarisor 生态中的 **量化策略研究沙箱**，提供从数据采集、策略回测、ML/DL/RL 研究、永续自进化优化到实盘交易的全链路能力。

## 核心能力

| 能力 | 说明 |
|------|------|
| **双市场数据采集** | TqSdk 期货 29 品种 5min/daily + Binance WebSocket 10 币种实时 tick |
| **事件驱动回测** | Walk-Forward / Monte Carlo / Optuna 超参搜索，期货与加密双市场 |
| **ML/DL 研究** | LightGBM / XGBoost + LSTM / Transformer 特征工程管线 |
| **RL PPO 训练** | MambaFormer 特征提取 + DSR 奖励 + 4 阶段课程学习 |
| **永续自进化优化** | 6 变体优化器 + LLM 引导进化 + overfit 检测门 + 冠军保存体系 |
| **模拟/实盘交易** | paper/live 一键切换；期货经 TqSdk gateway，加密经 WEEX（PolarPrivate sign） |
| **特色策略** | 攻防建模 (Lotka-Volterra)、庄家识别 (5 种操纵模式检测) |
| **自然语言研究** | ResearchRun 工作台，策略生成→诊断→验证→artifact 导出 |

## 技术栈

- **Runtime**: Python 3.13, Node 20
- **Trading**: TqSdk gateway + WEEX（PolarPrivate B-class sign）
- **ML**: PyTorch 2.11, LightGBM, SB3 (PPO), mamba-ssm
- **Optimization**: Optuna
- **Backend**: FastAPI (port 8000)
- **Frontend**: React 18 + Recharts + Vite
- **Database**: SQLite (WAL mode)

## 运行时治理

持久服务统一由 PolarPort（端口）和 PolarProcess（生命周期）管理。注册契约本身不启动服务：

```bash
bash scripts/register-runtime.sh
curl -fsS http://127.0.0.1:11055/api/services/tqsdk-data-collector
curl -fsS http://127.0.0.1:11055/api/services/tqsdk-gateway
```

禁止直接运行 Python/Uvicorn/Vite 服务、后台任务、PID 文件、直接信号或 launchd。生命周期动作必须命中精确 service ID。

## 服务操作

### TqSdk 网关（凭证隔离）

```bash
# 仅在明确需要期货交易/行情时启动；平时保持 stopped
curl -fsS -X POST http://127.0.0.1:11055/api/services/tqsdk-gateway/start
# → http://127.0.0.1:12890/health

# 验证（本进程不接触期货密码）
cd trading-platform && python3 scripts/test_tqsdk_login.py
```

trading-platform / data-collector 通过 `TQSDK_GATEWAY_URL` 访问网关，**不再** D-class 取明文。

### 数据采集服务

```bash
curl -fsS -X POST http://127.0.0.1:11055/api/services/tqsdk-data-collector/restart
curl -fsS http://127.0.0.1:18900/health
```

### 交易平台 API

交易平台 API 尚未作为本轮在线服务注册。需要持久运行前，必须先在 `polaris.json` 声明 health、preferred port、前台 launcher，并注册到 PolarProcess；不得直接启动。

### 前端

前端开发服务器同样必须先以独立 service ID 完成 PolarPort/PolarProcess onboarding；安装与 build 可以作为一次性命令直接运行。

### 冒烟测试

```bash
cd trading-platform
python scripts/run_smoke.py
```

### 永续优化器

优化器属于持久 worker，运行前必须以独立 service ID 注册到 PolarProcess；不得从 shell 留下后台进程。

## 项目结构

```
tqsdk/
├── tqsdk-gateway/           # TqSdk 凭证隔离网关（唯一持有明文，D-class 仅在此进程）
├── data-collector/          # 数据采集（经 gateway HTTP，无本地凭证）
├── trading-platform/
│   ├── apps/
│   │   ├── api/             # FastAPI 后端入口
│   │   ├── web/             # React 前端
│   │   └── skills/          # Agent 技能定义
│   ├── eternal-optimizer/   # 永续自进化优化器
│   ├── packages/            # 15 个领域包
│   │   ├── backtest/        # 回测引擎
│   │   ├── broker_tqsdk/    # TqSdk 经纪商适配器
│   │   ├── broker_crypto/   # WEEX 加密经纪商适配
│   │   ├── core/            # 事件总线、持久化、类型定义
│   │   ├── crypto/          # 加密策略 (52 个)
│   │   ├── datahub/         # 数据加载与清洗
│   │   ├── execution/       # 订单与持仓管理
│   │   ├── experiment/      # Optuna 超参搜索与验证门
│   │   ├── features/        # 特征工程
│   │   ├── ml/              # ML/DL 模型
│   │   ├── rl/              # 强化学习 (PPO + MambaFormer)
│   │   ├── risk/            # 风控引擎
│   │   ├── sim_live/        # 模拟/实盘调度
│   │   └── strategy/        # 策略基类与注册表
│   └── scripts/             # 工具脚本
├── lobster/                 # Lobster SDK 事件日志
└── contracts/               # Research artifact JSON Schema
```

## 生态依赖

| 依赖 | 用途 |
|------|------|
| **PolarPrivate** | WEEX 签名（B-class）；期货凭证仅经 TqSdk gateway（D-class） |
| **PolarPort/sdk** | 服务发现与端口分配 |
| **PolarClaw/polarclaw-project-sdk** | Lobster SDK 事件写入 |

## 文档

| 文档 | 用途 |
|------|------|
| [polaris.json](polaris.json) | SSoT — 需求与 feature 事实源 |
| [roadmap.md](roadmap.md) | 进度摘要 + 批次完成清单 |
| [PolarSoul.md](PolarSoul.md) | 项目灵魂与设计决策 |
| [capabilities.json](capabilities.json) | 生态能力注册表 |
| [contracts/](contracts/) | Research artifact JSON Schema |
| [trading-platform/docs/results-index.md](trading-platform/docs/results-index.md) | 训练/优化结果文件索引 |
