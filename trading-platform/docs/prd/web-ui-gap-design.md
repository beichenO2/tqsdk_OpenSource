# Web UI 缺口设计：deploy / ml / explain / 研究产物 / 设置

> 状态：设计已定稿，待实现（M8）
> 前提：约 40% 后端路由无前端消费。本文档为这些能力定义页面归属、布局与组件，
> 实现时不得偏离信息架构决策；细节（间距、文案）可由实现者裁量。
> 技术约束：沿用现有栈（React 19 + TS + Tailwind v4 + shadcn 风格 + TanStack Query + Recharts + lightweight-charts）。

## 0. 信息架构决策

不新增顶级工作区。所有缺口能力挂入现有四工作区：

| 能力 | 归属 | 形态 |
|---|---|---|
| deploy（参数部署/回滚） | 研究 → `/research/deploy` | 新页面 |
| ml（训练/模型/预测） | 研究 → `/research/ml` | 新页面 |
| explain（交易证据链） | 监控 → `/monitor/explain` | 新页面 |
| 研究产物追溯 | 研究 → ResearchRuns 页内 | Run 详情抽屉（Drawer），不新开页面 |
| 设置持久化 | 平台 → `/platform/settings` | 改造现有静态页 |
| 因子分析展示增强（相关性热力图、进化历史可视化） | — | **进 roadmap 搁置**（用户决策 2026-07-10） |

## 1. `/research/deploy` — 部署控制台

消费 API：`deploy.py` 全部（`GET /deploy/params`、`GET /deploy/optuna-best`、`POST /deploy/apply`、`POST /deploy/rollback`、`GET /deploy/history`）。

布局（左右两栏）：
- **左栏：当前生效参数卡片**。展示 `active_deployment` 内容（策略名、mode 徽章 paper/live、参数键值表、生效时间）。
- **右栏：候选参数**。来自 optuna-best，与当前参数做 diff 高亮（新值绿、旧值删除线）。底部主按钮「应用到 Paper」；若 mode=live 需复用 Live 页现有二次确认对话框（X-Live-Confirm 机制）。
- **下方：部署历史时间线**。每条含时间、操作人来源、参数摘要、状态；每条右侧「回滚到此版本」按钮，二次确认后调 rollback。

组件：`DeployDiffTable`（参数 diff）、复用 `ui/Dialog` 做确认。

## 2. `/research/ml` — 模型工作台

消费 API：`ml.py` 全部（train / models / predict）。

布局（上下三段）：
- **模型列表表格**：模型名、类型、训练时间、训练数据范围、指标（AUC/IC 等，后端有什么展示什么）。行点击展开预测面板。
- **发起训练表单**：折叠卡片，选品种 + 特征集（复用 factors 列表 API 做多选）+ 目标周期，提交后行内显示训练状态（轮询 models 列表刷新，不做 SSE）。
- **预测面板**（选中模型后）：输入 symbol → 调 predict → 展示预测值 + 方向徽章 + 最近 N 次预测的小型折线（Recharts）。

原则：这是研究工具页，功能可用优先，不做训练过程实时曲线（roadmap）。

## 3. `/monitor/explain` — 交易证据链

消费 API：`explain.py` 全部（timeline / factors / graph）。

布局：
- **顶部筛选**：策略、品种、时间范围。
- **主体：决策时间线**（垂直 timeline 组件）。每个节点 = 一次信号/下单决策，展开显示：触发因子快照（因子名+当时值+z-score）、风控检查结果、订单结果。
- **右侧上下文面板**：选中节点时显示该时刻 K 线小图（复用 `KLineChart`，标注决策点垂直线）+ 因子贡献条形图（Recharts 横向 bar）。
- graph 端点若返回因果图结构，第一版用缩进列表渲染，不引入图可视化库（roadmap）。

价值定位：这是"为什么下了这单"的审计页，实盘出问题时的第一排查入口。

## 4. ResearchRuns 产物追溯（页内增强）

消费 API：`research.py` 未接端点（diagnostics / iterations / validation / factor-snapshot / artifact / artifact/markdown / events / PATCH / DELETE）。

设计：Run 列表行点击 → 右侧滑出 **Drawer（约 60% 宽）**，内部 Tabs：
- **概览**：8 步 pipeline 用新组件 `PipelineStepper`（水平步骤条，状态色：灰=未跑/蓝=进行/绿=过/红=fail），替代现有徽章堆。
- **产物**：artifact 列表，markdown 类产物直接渲染（引入轻量 md 渲染，如 `marked` + sanitize），其余给下载链接。
- **验证**：validation + gate 报告表格。
- **迭代**：iterations 列表 + 每轮指标小折线。
- **事件**：events 流水（复用 EventsPage 的行样式）。
Drawer 头部放操作：重命名（PATCH）、删除（DELETE，二次确认）、晋升（复用现有 promote）。

## 5. `/platform/settings` 改造

消费 API：`settings.py` 全部（GET / PUT / PATCH / reset）。

设计：
- 进入页面 GET 拉真实配置，按后端返回的分组渲染（风控参数 / 交易开关 / 数据源）。
- 每个分组独立「保存」（PATCH 该分组），顶部全局「重置为默认」（reset，二次确认）。
- 敏感项（凭证类）只显示"已配置/未配置"状态 + 指引文案（凭证走 PolarPrivate，不在 Web 明文编辑）。
- 保存成功/失败必须 toast，禁止静默。

## 6. 横切规范

1. 所有新页面注册进 `nav/workspaces.ts` 与命令面板（⌘K）。
2. 所有 mutation 失败必须 toast 展示后端 detail，禁止静默 catch（与 F1 修复保持一致）。
3. 新组件放 `src/components/`，页面级放 `src/pages/`；API 封装统一加进 `src/services/api.ts`。
4. K 线相关一律复用 `KLineChart`（默认对数价格轴）。

## 7. 实现优先级

P0：settings 改造（影响可配置性）、ResearchRuns 产物 Drawer（研究闭环）
P1：deploy 页（优化→部署晋升闭环）
P2：explain 页、ml 页

## Roadmap（本轮搁置）

- 因子相关性热力图、因子进化 bandit 历史可视化
- explain 因果图可视化（图布局库）
- ML 训练过程实时曲线（SSE）
- crypto 死代码（`pages/crypto/*`、web-btc CI 步骤）清理
