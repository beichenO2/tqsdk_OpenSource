# tqsdk Web v2 + 策略/因子开发平台 设计文档

> 状态：草案 v0.1（2026-07-09）
> 作者：架构师 Agent（hub-agent-4）
> 事实源：本文件是**设计提案**，落地后需回写 `polaris.json`（新增 R9/R10）。
> 定位：在现有 `trading-platform`（React18 + FastAPI + 15 领域包）之上做 Web 层重构 +
> 打通实盘链路 + 新增"策略/因子开发平台"，对标 Qlib / RD-Agent(Q) / vn.py 4.x / QuantConnect LEAN。

---

## 0. 为什么要做（问题诊断）

当前 Web（`apps/web`）已有 15 个页面（仪表盘/交易/策略/回测/回测对比/风控/模拟实盘/实盘交易/加密四页/策略详情/设置），
底层能力其实很厚（R1–R8 全部 done：回测引擎、ML/DL/RL、永续优化器、庄家/攻防策略、
ResearchRun 工作台、LiveScheduler 实盘通道）。真正的缺口不在"功能有没有"，而在：

| 缺口 | 现状 | 影响 |
|------|------|------|
| **能力散落、无统一研究工作流** | 回测、优化、ML 训练各有入口，靠人肉串联 | 无法"想法→因子→回测→验证→部署"闭环 |
| **因子层缺失** | 策略直接吃 K 线，因子没有一等公民地位 | 无法复用/组合/去重因子，无 IC 分析 |
| **前端与实盘是"两张皮"** | LiveTrading 页有了，但下单链路、风控前置、实时回报未端到端贯通 | 不敢开实盘 |
| **前端信息架构陈旧** | 侧边栏 7 项平铺，双市场用 store 切换，研究/实盘/监控混在一起 | 心智负担高，扩展困难 |
| **无"研究即产物"沉淀** | ResearchRun 有 API（R8）但前端未接 | 研究结果不可追溯、不可复现 |

对标 SOTA 后的结论（见 §5）：**Qlib 的分层工作流 + RD-Agent 的 LLM 因子-模型联合进化 +
vn.py 的事件驱动实盘/风控前置 + LEAN 的 research→backtest→live 一键晋升**，正是我们要补的形状。

---

## 1. 目标与非目标

### 1.1 目标（本设计覆盖）
1. **Web v2**：重构信息架构与设计语言，把"研究 / 交易 / 监控 / 平台"分区，双市场从全局切换降级为上下文标签。
2. **前端↔实盘贯通**：定义端到端下单链路（前端 → API → 风控前置 → ExecutionService → BrokerAdapter → TqSdk Gateway / WEEX），实时回报经 WebSocket 回流。
3. **策略/因子开发平台**：新增"因子"一等公民 + 统一研究工作流（Idea→Factor→Model→Backtest→Validate→Deploy），对接已有 ResearchRun / eternal-optimizer / experiment gate。

### 1.2 非目标（本轮不做）
- 不重写回测/优化/RL 内核（R1–R7 已 done，只做编排与暴露）。
- 不引入新交易所（维持 TqSdk 期货 + WEEX 加密）。
- 不做多租户/账户体系（本地单用户沙箱）。
- 实盘"真金白银"开关默认关闭，本设计只做"可安全开实盘"的工程准备。

---

## 2. Web v2 设计

### 2.1 信息架构（IA）重构

从"7 项平铺 + 全局市场切换"改为**四大工作区**（顶层），每个工作区内再分页：

```
PolarTrade v2
├── 研究 Research              # 想法 → 产物 的主战场
│   ├── 工作台 Workbench        # ResearchRun：自然语言/表单发起研究，看进度+诊断
│   ├── 因子 Factors            # 【新】因子库、IC 分析、因子构造器
│   ├── 策略 Strategies         # 策略库（80+ 已注册），详情/回测入口
│   ├── 回测 Backtest           # 单跑 + 对比 + 参数热力图
│   └── 优化 Optimizer          # eternal-optimizer 冠军榜、gate 报告、部署
├── 交易 Trading               # 模拟 + 实盘
│   ├── 模拟实盘 Paper          # 排行榜、账户、净值
│   ├── 实盘 Live               # 下单、持仓、回报（默认只读，需二次确认开实盘）
│   └── 手动交易 Manual         # 手工下单（期货/加密）
├── 监控 Monitor               # 风控 + 系统健康
│   ├── 风控 Risk               # 限额、风险指标、告警
│   ├── 实时事件 Events         # WebSocket 6 类事件流（position/trade/account/strategy/signal/risk）
│   └── 系统 Health             # gateway/API/optimizer/data 健康（复用 lobster get_health）
└── 平台 Platform
    ├── 数据 Data               # 采集状态、品种覆盖、数据质量
    ├── 技能 Skills             # docs/skills 的策略生成/诊断 SOP
    └── 设置 Settings           # 凭证（经 PolarPrivate）、端口、主题
```

双市场（期货/加密）**不再是全局开关**，而是每个页面内的**上下文过滤器**（chips），
因为研究/因子大多是单市场的，全局切换会误伤。

### 2.2 设计语言（可全部推翻旧样式）

- **主题**：默认深色（交易场景护眼），保留浅色切换。品牌色沿用 `brand`，盈亏用 `profit`/`loss` 语义色。
- **布局**：左侧一级导航（四工作区图标）+ 二级导航（工作区内页），顶部为面包屑 + 上下文过滤器 + 全局搜索（⌘K 命令面板，可跳页/发起研究/查策略）。
- **组件基线**：沿用现有 `components/ui`（Button/Dialog/Table/Tabs/Toast/Skeleton…），补充：
  - `CommandPalette`（⌘K）
  - `MetricPill` / `ICBadge`（因子/策略指标一眼可读）
  - `PipelineStepper`（研究工作流可视化：Idea→Factor→Model→Backtest→Validate→Deploy）
  - `RealtimeFeed`（WebSocket 事件流虚拟列表）
- **图表**：沿用 Recharts；新增 **IC 时间序列图**、**因子相关性热力图**、**权益/回撤双轴**。
- **数据获取**：沿用 `@tanstack/react-query`；实时数据走 WebSocket hook（已存在 `useWebSocket.ts`）。

### 2.3 技术栈（保留 + 强化）
- React 18 + Vite + TS + Tailwind + react-router + react-query（现状，保留）。
- 状态：`marketStore` 降级为上下文过滤；新增 `researchStore`（当前 run）、`realtimeStore`（WS 订阅）。
- 不引入重型 UI 框架，保持现有轻量组件体系。

---

## 3. 前端 ↔ 实盘 贯通（为实盘做准备）

### 3.1 端到端下单链路（目标态）

```
[Web Live 页]
   │  POST /live-trading/order  (symbol, side, qty, type, price, strategy_id?)
   ▼
[FastAPI live_trading router]
   │  1) 身份/模式校验（paper|live，live 需 explicit confirm token）
   ▼
[RiskGate 风控前置]  ← 【新】下单前拦截：限额/保证金/涨跌停/交割月/单笔上限
   │  pass ↓          reject → 4xx + 风控事件（WS: risk）
   ▼
[ExecutionService]  →  [BrokerAdapter]
   │                       ├── 期货 → TqSdk Gateway (127.0.0.1:12890, D-class 隔离)
   │                       └── 加密 → WEEX (PolarPrivate B-class sign)
   ▼
[回报] → EventBus → WebSocket 推送 (trade/position/account) → 前端实时更新
        └→ SQLite 持久化 (orders/fills/equity_snapshots)
```

现状：LiveScheduler、ExecutionService、BrokerAdapter、EventBus、WS、SQLite 持久化都已存在（R5 done）。
**要补的是**：
1. **RiskGate 下单前置**（把 `packages/risk` 从"事后监控"提升为"事前拦截"，挂到下单入口）。
2. **实盘二次确认**：live 模式下单需 `X-Live-Confirm` token（防误触），前端弹 `ConfirmDialog`。
3. **国内期货专项校验**（复用 R8 诊断 SOP）：涨跌停、夜盘时段、交割月、保证金、手续费。
4. **前端 Live 页**：从"只读状态"升级为"下单 + 持仓 + 回报 + 一键平仓"，默认 paper。

### 3.2 安全隔离（不可退让）
- 期货凭证**只**经 `tqsdk-gateway/`（D-class），trading-platform 进程不碰明文（现状已重构，未提交）。
- 加密走 WEEX B-class sign（PolarPrivate）。
- 实盘开关：环境变量 `LIVE_TRADING_ENABLED=false` 默认关；前端显式确认 + 后端 token 双闸。

### 3.3 里程碑验收（实盘就绪定义）
- [ ] paper 模式端到端：前端下单 → 回报 → 净值刷新（无真钱）。
- [ ] RiskGate 拦截演示：超限单被拒 + risk 事件到前端。
- [ ] 期货专项校验：涨跌停/交割月/夜盘用例通过。
- [ ] live 模式灰度：1 手最小单 + 二次确认 + 立即可平。

---

## 4. 策略 / 因子开发平台（核心增量）

### 4.1 核心理念：因子成为一等公民 + 统一研究工作流

借鉴 Qlib（分层工作流 + Recorder 产物追踪）、RD-Agent(Q)（LLM 因子-模型联合进化 +
IC 去重 + bandit 调度）、vn.py alpha（多因子 ML 一站式）、LEAN（research→backtest→live 晋升）。

统一工作流（一条主线，每步都有产物且可追溯）：

```
① Idea        自然语言/模板  →  假设 (hypothesis)          [已有 ResearchRun POST]
② Factor      因子实现+计算   →  因子值 + 元数据            【新 packages/factor】
③ Validate-F  因子体检        →  IC / IR / 去重(相关性>0.99剔除) / 衰减  【新】
④ Model       特征→模型       →  预测信号 (alpha)          [复用 packages/ml, rl]
⑤ Backtest    信号→回测       →  权益/交易/指标            [复用 packages/backtest]
⑥ Validate-S  策略门控        →  OOS/WF/MC-CI/X-Asset      [复用 experiment gate]
⑦ Deploy      冠军→部署       →  paper/live 参数           [复用 eternal-optimizer + deploy API]
⑧ Record      全程产物        →  research_artifact.json    [复用 contracts + R8 artifact]
```

### 4.2 因子子系统（全新 `packages/factor`）

```
packages/factor/
├── base.py            # Factor 基类：name, category, compute(bars)->Series, meta
├── registry.py        # @factor_register 装饰器 + FactorRegistry（对齐 StrategyRegistry）
├── library/           # 内置因子库（动量/反转/波动/量价/OI/微观结构/攻防/庄家 …）
├── analysis.py        # IC/RankIC/IR、IC 衰减、因子相关性矩阵、去重（IC_max≥0.99 剔除）
├── combine.py         # 因子合成（等权/IC 加权/正交化）
└── export.py          # 因子值 → parquet；元数据 → factor_meta.json
```

- **数据契约**：因子吃 `datahub` 的标准 bars，吐 `pd.Series`（index=时间，对齐主力合约——正好复用
  pos2 未提交的 `_filter_main_contract` 主力合约过滤补丁，见 §6 抢救项）。
- **与策略对接**：策略从"直接算指标"改为"声明依赖因子"，因子层统一计算+缓存，避免重复+可组合。
- **与 ML/RL 对接**：因子矩阵直接喂 `packages/ml` 的 FuturesDataLoader / RL 观测空间。

### 4.3 因子/策略 API（新增，挂在现有 FastAPI）

```
# 因子
GET  /factors                      # 因子库列表（category/IC 概览）
GET  /factors/{name}               # 因子详情（定义/IC 曲线/衰减/相关性）
POST /factors/compute              # 计算因子值（symbol, timeframe, factor_names[]）
POST /factors/analyze              # IC/IR/去重分析
POST /factors/combine              # 因子合成

# 研究工作流（扩展现有 R8 ResearchRun）
POST /research/runs                # 发起（已有）→ 扩展支持 factor/model 阶段
GET  /research/runs/{id}/pipeline  # 【新】返回 8 步 pipeline 状态（供 PipelineStepper）
POST /research/runs/{id}/promote   # 【新】research→backtest→paper→live 晋升
```

### 4.4 前端：研究工作区（对接上述 API）
- **Workbench**：⌘K 或表单发起研究 → `PipelineStepper` 实时展示 8 步 → 每步产物可点开。
- **Factors 页**：因子库表格（IC/IR/category 过滤）+ 因子详情（IC 时序、衰减、相关性热力图）+ 因子构造器（选数据源→选算子→预览 IC）。
- **Optimizer 页**：eternal-optimizer 冠军榜 + gate 拒绝报告（rejected_by/severity）+ 一键晋升。

### 4.5 LLM 因子进化（可选增强，对标 RD-Agent(Q)）
- 复用现有 `eternal-optimizer` 的"LLM 引导进化 + 404 降级"，把进化目标从"策略参数"扩展到"因子表达式"。
- bandit 调度：在 因子挖掘 / 模型优化 之间自适应分配预算（RD-Agent 的关键设计）。
- 硬约束（沿用现有 gate）：新因子须过 IC 去重 + OOS/WF/MC，才能进冠军库→paper。**不自动开 live**。

---

## 5. SOTA 对标结论（借鉴映射）

| SOTA 项目 | 借鉴点 | 落到本平台 |
|-----------|--------|-----------|
| **Microsoft Qlib** | 分层工作流（data→model→eval→record）、Recorder 产物追踪、Alpha158/360 因子集、qrun 声明式配置 | §4.1 统一工作流 + §4.8 research_artifact 产物；因子库参考 Alpha158 结构 |
| **RD-Agent(Q)** | LLM 因子-模型联合进化、IC≥0.99 去重、bandit 调度、"2× 收益/70% 更少因子" | §4.5 LLM 因子进化 + §4.2 去重；扩展现有 eternal-optimizer |
| **vn.py 4.x** | 事件驱动引擎、风控前置（下单前拦截）、web_trader B-S 架构、alpha 多因子 ML 模块 | §3.1 RiskGate 前置 + EventBus（已有）；因子 ML 一站式 |
| **QuantConnect LEAN** | research(Jupyter)→backtest→live 一键晋升、handler 可插拔、事件驱动防未来函数 | §4.3 `/research/runs/{id}/promote` 晋升链；回测引擎已事件驱动 |

---

## 6. 与"赛马合并"的衔接（重要：抢救未提交产物）

赛马结论：**`gnhf/pos1-init` 是超集**（pos2/pos3/pos4 已被 pos1 显式 merge，git 已验证三者均为 pos1 祖先）。
合并 pos1→main 时，以下**未提交**改动与本设计强相关，需一并抢救进 main：

| 来源 | 未提交产物 | 与本设计关系 |
|------|-----------|-------------|
| 主库工作区 | `tqsdk-gateway/` 全套 + `broker_tqsdk/gateway_client.py` + `gateway_market_adapter.py` + `sim_live/live_feed.py:TqGatewayLiveFeed` + `data-collector` gateway 采集 | §3.2 凭证隔离、§3.1 实盘链路的**基础** |
| 主库工作区 | `privportal.py`（get_tqsdk_keys 标记 DEPRECATED，仅 gateway 用） | §3.2 安全隔离 |
| pos2 worktree | `validate_gate.py`/`validate_quick_sweep.py` 的 `_filter_main_contract` 主力合约过滤 + `whale_detector_*.json` 结果 + `.coordination/leaderboard.md` | §4.2 因子对齐主力合约、策略验证 |
| pos1 已提交 | `intraday_reversal` A/B/C 变体、复利 blow-up 修复、11 个复活策略模板、ResearchRuns 前端页 | §4.4 研究工作区已有雏形 |

> ⚠️ 合并顺序建议：先 commit 主库 gateway 改动（分 2 批：gateway 隔离 / broker+采集适配），
> 再 merge pos1（解 4 个冲突文件：`main.py` / `test_tqsdk_login.py` / `research.py` / `Layout.tsx` / `api.ts`），
> 最后抢救 pos2 的 validate 补丁与 leaderboard。详见执行计划 §7。

---

## 7. 实施路线图（分阶段，落 polaris.json R9/R10）

### Phase 0 — 合并与清理（前置，需 shell）
- 主库 gateway 改动分批 commit（py_compile 已全绿）。
- merge `gnhf/pos1-init` → main，解冲突。
- 抢救 pos2 validate 补丁 + leaderboard。
- P13 六维检查 + ClawBin 归档 `tqsdk-gnhf-worktrees/`，`git worktree remove ×4` + 删目录。

### Phase 1 — 因子子系统（R9 核心）
- `packages/factor` 骨架 + 内置因子库（先迁移现有指标为因子）。
- IC/IR/去重分析 + `/factors` API。
- 前端 Factors 页（库表格 + IC 图 + 相关性热力图）。

### Phase 2 — 统一研究工作流（R9）
- 扩展 ResearchRun 支持 factor/model 阶段 + `/pipeline` + `/promote`。
- 前端 Workbench + PipelineStepper + ⌘K 命令面板。

### Phase 3 — Web v2 信息架构（R10）
- 四工作区导航重构 + 上下文过滤（市场降级）。
- Optimizer 页（冠军榜 + gate 报告 + 晋升）。

### Phase 4 — 前端↔实盘贯通（R10）
- RiskGate 下单前置 + 实盘二次确认 + 期货专项校验。
- Live 页升级（下单/持仓/回报/一键平仓），paper 默认。

### Phase 5 — LLM 因子进化（R9 增强，可选）
- eternal-optimizer 进化目标扩展到因子表达式 + bandit 调度。

---

## 8. 验收标准（Definition of Done）

- **因子平台**：≥20 内置因子可计算，IC 分析可视化，去重生效（相关性>0.99 剔除），因子可喂 ML/回测。
- **研究工作流**：一次 run 走完 8 步且产物可追溯（research_artifact.json），前端可视化。
- **Web v2**：四工作区导航上线，⌘K 可用，双市场降级为上下文，无回归。
- **实盘就绪**：paper 端到端下单→回报→净值；RiskGate 拦截演示；期货专项校验用例过。
- **SSoT**：`polaris.json` 新增 R9（因子/研究平台）、R10（Web v2 + 实盘贯通），roadmap 同步。

---

## 附录 A：现有资产复用清单（避免重复造轮子）

| 需求 | 已有资产 | 复用方式 |
|------|---------|---------|
| 回测 | `packages/backtest`（Walk-Forward/MC/Optuna） | 因子信号→回测直接调 |
| 门控 | `packages/experiment`（OOS/WF/MC-CI/X-Asset gate） | 策略验证 §4.1 ⑥ |
| 优化 | `eternal-optimizer`（6 变体 + LLM + 冠军库） | §4.5 因子进化外壳 |
| 实盘 | `packages/sim_live` + `execution` + EventBus + SQLite（R5） | §3.1 链路 |
| 研究 | R8 ResearchRun API + contracts schema | §4.3 扩展 |
| 前端 | `apps/web` 15 页 + `components/ui` + WS/LiveTrading hooks | §2 v2 重构基线 |
| 技能 | `docs/skills`（策略生成/诊断 SOP）+ `docs/swarm-presets` | §4 工作流内嵌 SOP |
