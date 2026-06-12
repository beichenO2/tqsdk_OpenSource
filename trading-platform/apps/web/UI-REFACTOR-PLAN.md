# PolarTrade UI 重构方案

> 2026-04-19 · 由 TQSDK UI 审查与改进 Agent 编写

## 现状诊断

两个独立 React 应用（web: React 19/Tailwind v4, web-btc: React 18/Tailwind v3），共 43 个 TSX 文件，大量重复代码，web-btc 半成品状态（Tab 不工作、K 线图占位符）。整体可用性差，无 Error Boundary，高危金融操作无确认。

## 目标

合并为**单一 React 应用**，统一设计系统，覆盖期货+加密货币双市场，交付专业级量化交易 UI。

---

## 技术栈（锁定）

| 层 | 选型 | 理由 |
|----|------|------|
| 框架 | React 19 + TypeScript 6 | 已有基础 |
| 构建 | Vite 8 | 已有基础 |
| 路由 | react-router-dom v7 | web 已用 |
| 样式 | Tailwind CSS v4 (`@theme`) | web 已用，token 体系保留 |
| 服务端状态 | TanStack Query v5 | web-btc 已引入，替代手动 polling |
| 客户端状态 | Zustand v5 | web-btc 已引入，管理 UI 全局状态 |
| 图表-交易 | lightweight-charts v4 | K 线图专用 |
| 图表-统计 | Recharts v3 | 回测/仪表盘统计图 |
| 实时通信 | 原生 WebSocket + TanStack Query 集成 | 替代 setInterval 轮询 |
| 图标 | lucide-react | 已有 |
| 无头 UI | 自建（参考 Radix 模式） | Dialog/Tabs/Dropdown 等交互组件 |

---

## 设计系统：PolarUI

### 设计 Token（index.css @theme 扩展）

```css
@theme {
  /* 现有 brand/surface/text/semantic 保留 */

  /* 新增 —— 交互态 */
  --color-focus-ring: #f0780e80;
  --color-overlay: #00000080;

  /* 新增 —— 信息色（补充单调橙色问题） */
  --color-info: #3b82f6;
  --color-info-light: #60a5fa;

  /* 间距系统 */
  --spacing-page: 1.5rem;
  --spacing-card: 1rem;
  --spacing-section: 1.5rem;

  /* 圆角 */
  --radius-sm: 0.375rem;
  --radius-md: 0.5rem;
  --radius-lg: 0.75rem;
  --radius-xl: 1rem;

  /* 过渡 */
  --transition-fast: 150ms cubic-bezier(0.4, 0, 0.2, 1);
  --transition-normal: 200ms cubic-bezier(0.4, 0, 0.2, 1);

  /* 阴影 */
  --shadow-modal: 0 25px 50px -12px rgb(0 0 0 / 0.5);
  --shadow-dropdown: 0 10px 15px -3px rgb(0 0 0 / 0.4);
}
```

### 基础组件清单（22 个）

| 组件 | 职责 | 当前状态 |
|------|------|----------|
| `Button` | 主要/次要/危险/幽灵 变体，loading态 | **新建** |
| `IconButton` | 图标按钮，统一尺寸和 a11y | **新建** |
| `Input` | 文本/数字输入，前后缀，错误态 | **新建**（现为裸 input） |
| `Select` | 下拉选择，支持搜索 | **新建** |
| `Dialog` | 模态框，focus trap + Escape + aria | **新建**（替代内联 div） |
| `ConfirmDialog` | 确认对话框（危险操作专用） | **新建** |
| `Tabs` | 受控 Tab 组件，键盘导航 | **新建**（替代 web-btc 死 Tab） |
| `Table` | 排序 + 键盘行选 + 交替色 + 空态 | **新建**（替代裸 table） |
| `Card` | 保留现有，扩展 loading skeleton | **重构** |
| `StatCard` | 统一为一个版本（合并 PaperTrading 变体） | **重构** |
| `Badge` / `StatusBadge` | 保留，加 aria-label | **增强** |
| `Toast` | 右下角 toast 通知 | **新建** |
| `Skeleton` | 骨架屏加载态 | **新建** |
| `EmptyState` | 空状态插画 + 引导文案 | **新建** |
| `Tooltip` | 信息提示 | **新建** |
| `Dropdown` | 下拉菜单 | **新建** |
| `SearchInput` | 带搜索图标的过滤输入 | **新建** |
| `KLineChart` | lightweight-charts 封装 | **新建** |
| `EquityCurveChart` | 保留，改用 CSS 变量色值 | **重构** |
| `DrawdownChart` | 保留，改用 CSS 变量色值 | **重构** |
| `TradeMarkerChart` | 保留，改用 CSS 变量色值 | **重构** |
| `ErrorBoundary` | 路由级 + 组件级双层 | **新建** |

---

## 目录结构

```
apps/web/src/
├── app/
│   ├── App.tsx                    # BrowserRouter + providers
│   ├── routes.tsx                 # lazy route 定义
│   └── providers.tsx              # QueryClient + Zustand + Toast
├── components/
│   ├── ui/                        # PolarUI 基础组件
│   │   ├── Button.tsx
│   │   ├── Dialog.tsx
│   │   ├── ConfirmDialog.tsx
│   │   ├── Tabs.tsx
│   │   ├── Table.tsx
│   │   ├── Toast.tsx
│   │   ├── Skeleton.tsx
│   │   └── ...
│   ├── layout/
│   │   ├── Sidebar.tsx            # 重构后侧边栏（加宽+折叠）
│   │   ├── Header.tsx             # 顶部状态栏（连接状态+通知）
│   │   └── PageLayout.tsx
│   ├── charts/
│   │   ├── KLineChart.tsx
│   │   ├── EquityCurveChart.tsx
│   │   ├── DrawdownChart.tsx
│   │   ├── TradeMarkerChart.tsx
│   │   └── ParameterHeatmap.tsx
│   └── domain/                    # 业务组件
│       ├── OrderForm.tsx          # 统一下单面板（期货+加密共用）
│       ├── PositionTable.tsx
│       ├── OrderTable.tsx
│       ├── StrategyCard.tsx
│       ├── RiskGauge.tsx
│       ├── BacktestForm.tsx
│       └── MarketSwitcher.tsx     # 市场切换组件
├── pages/
│   ├── Dashboard.tsx              # 统一仪表盘（市场可切换）
│   ├── Markets.tsx                # 行情中心（搜索+关注+分组）
│   ├── Trading.tsx                # 交易（K线+订单簿+下单）
│   ├── Strategies.tsx
│   ├── Backtest.tsx
│   ├── BacktestCompare.tsx
│   ├── Risk.tsx
│   ├── PaperTrading.tsx
│   └── Settings.tsx
├── hooks/
│   ├── useWebSocket.ts           # WS 连接管理
│   ├── useMarketData.ts          # 行情数据 (WS)
│   ├── useAccount.ts             # 账户查询 (TanStack Query)
│   ├── usePositions.ts
│   ├── useOrders.ts
│   ├── useStrategies.ts
│   ├── useBacktest.ts
│   ├── useConfirm.ts             # 确认弹窗 hook
│   └── useToast.ts               # toast hook
├── stores/
│   ├── marketStore.ts            # 当前市场 (futures/crypto)
│   ├── layoutStore.ts            # 侧边栏折叠等 UI 状态
│   └── watchlistStore.ts         # 自选列表
├── services/
│   ├── api.ts                    # HTTP 请求（TanStack Query 的 queryFn）
│   ├── ws.ts                     # WebSocket 管理器
│   └── export.ts                 # 数据导出工具
├── lib/
│   ├── format.ts                 # fmt() 统一定义（消除4处重复）
│   ├── cn.ts                     # clsx + twMerge 封装
│   └── constants.ts
├── types/
│   └── index.ts                  # 统一类型（合并 web + web-btc 的 types）
├── index.css
└── main.tsx
```

---

## 导航重构

### 现状
- 64px 超窄侧边栏，10px 标签难以阅读
- 期货 / 加密分两组 nav item
- web-btc 独立应用有自己的导航

### 目标

侧边栏双态：**折叠（64px 图标态）+ 展开（200px 标签态）**，用户可切换。

导航不再按市场分组，改为功能分组 + **顶部市场切换器**：

```
┌──────────────────┐
│  PolarTrade  [▼] │  ← 品牌 + 市场选择器（期货 / BTC / 全部）
├──────────────────┤
│  📊  仪表盘      │
│  📈  行情        │  ← 新增（行情中心）
│  💱  交易        │
│  🤖  策略        │
│  🧪  回测        │
│  ⚖️  风控        │
│  🏆  模拟实盘    │
│──────────────────│
│  ⚙️  设置        │
└──────────────────┘
```

当切换市场时，所有页面数据自动切换到对应市场，而不是跳转到另一个 nav 区域。

---

## 核心页面重构细节

### 1. Dashboard（合并期货+加密仪表盘）

**改变：**
- 移除 `Math.random()` 假数据，用 Skeleton 占位
- 使用 TanStack Query 管理数据获取（自动 refetch、stale-while-revalidate）
- "一键平仓"、"暂停策略" 改用 `ConfirmDialog`
- 市场切换器决定展示哪个市场的数据
- 图表色值改用 CSS 变量引用

```tsx
// 确认弹窗示例
const confirm = useConfirm();

const handleCloseAll = async () => {
  const ok = await confirm({
    title: '确认一键平仓',
    description: '这将关闭所有持仓，该操作不可撤回。',
    variant: 'destructive',
    confirmText: '确认平仓',
  });
  if (!ok) return;
  await closeAllPositions.mutateAsync();
  toast.success('已提交平仓指令');
};
```

### 2. Markets（新增行情中心）

**现状：** 行情表格嵌在 Trading 页面内，无搜索。
**改变：** 独立页面，包含：
- 搜索框（按合约代码/名称实时过滤）
- 分组视图（按交易所 / 策略类型 / 自选）
- 自选列表（Zustand 持久化到 localStorage）
- 点击合约 → 跳转交易页或弹出快速下单面板

### 3. Trading（K 线 + 深度重构）

**改变：**
- 集成 lightweight-charts K 线图（替代 "等待集成" 占位符）
- 订单簿面板（从 web-btc 的 OrderBookPanel 迁入）
- 统一 OrderForm（支持期货的"开仓/平仓"和加密的"做多/做空"）
- 底部 Tabs 用真正的 `Tabs` 组件（持仓/挂单/成交历史 可切换）
- 布局可拖拽调整比例（可选 Phase 2）

### 4. Strategies

**改变：**
- "新建策略" 按钮加 `type="button"`
- "启动/停止" 操作加确认弹窗
- "编辑" 按钮连接到实际编辑模态框
- 策略卡片显示更多统计（最大回撤、胜率）

### 5. Backtest（增强，基础已扎实）

**改变：**
- 模态框加 focus trap + Escape 关闭 + `aria-labelledby`
- 回测列表行加 `tabIndex` + Enter/Space 键盘选择
- 导出按钮（CSV/JSON）
- 运行中的回测显示进度条

### 6. Risk（增强）

**改变：**
- 风控告警支持 Toast 推送（新告警自动弹 toast）
- 风险仪表盘加动画过渡
- 告警可标记为已读/已解决

### 7. Settings（功能化）

**改变：**
- 表单状态用 Zustand 管理
- 保存按钮连接到后端 API
- 保存成功/失败有 Toast 反馈
- 新增"数据管理"区域（清除缓存、导出配置）
- 新增"API 连接测试"按钮

---

## 数据层重构

### WebSocket 替代轮询

```
现状: setInterval(loadData, 5000)  → 延迟 0-5s，浪费请求
目标: WebSocket 推送              → 延迟 <100ms，按需更新
```

WebSocket 管理器：
- 自动重连（指数退避）
- 心跳保活
- 与 TanStack Query 缓存集成（WS 推送 → 更新 query cache → 组件自动刷新）

### TanStack Query 替代手动 useState

```
现状: useState + useEffect + setInterval（每个页面重复写）
目标: useQuery/useMutation 统一管理

好处:
- 自动缓存 + stale-while-revalidate
- loading/error 状态自动管理
- 页面切换返回时瞬间显示缓存数据
- Suspense 集成
- 请求去重
```

---

## 可访问性（A11y）清单

| 项 | 实现 |
|----|------|
| 所有按钮 `type="button"` | Button 组件默认设置 |
| Dialog focus trap + Escape | Dialog 组件内置 |
| Dialog aria-labelledby | Dialog 自动绑定 title |
| Table 行键盘导航 | Table 组件 `tabIndex` + `onKeyDown` |
| Tabs 键盘左右切换 | Tabs 组件 ArrowLeft/ArrowRight |
| 颜色对比度 ≥ 4.5:1 | Token 设计时验证 |
| 图表替代文本 | 每个图表下方提供数据摘要 |
| Toast 用 `role="alert"` | Toast 组件内置 |
| 最小触摸目标 44x44 | 侧边栏加宽后满足 |

---

## 执行分 Phase

### Phase 1：基础设施（预计 8h）
1. 删除 web-btc 目录（其有价值的代码迁入 web）
2. 安装 TanStack Query + Zustand
3. 创建 `components/ui/` 目录，实现基础组件
   - Button, Input, Dialog, ConfirmDialog, Tabs, Toast, Skeleton, ErrorBoundary
4. 提取 `lib/format.ts`（消除 fmt 重复）
5. 提取 `lib/cn.ts`（clsx + twMerge）
6. 设置 App providers（QueryClientProvider, ToastProvider）
7. 路由 lazy loading + Suspense
8. 扩展 @theme token

### Phase 2：导航 + 布局（预计 4h）
1. 重构 Sidebar（折叠/展开双态）
2. 实现 MarketSwitcher
3. 创建 Header 组件（连接状态 + 通知铃铛）
4. PageLayout 包装统一间距

### Phase 3：数据层（预计 6h）
1. WebSocket 管理器
2. 迁移所有 API 调用到 TanStack Query hooks
3. WS 推送 → Query cache 更新集成
4. 统一类型定义（合并两个 app 的 types）
5. Zustand stores（market, layout, watchlist）

### Phase 4：页面重构（预计 12h）
1. Dashboard — 合并期货+加密，Skeleton，确认弹窗
2. Markets — 新增行情中心
3. Trading — K 线图集成，统一 OrderForm，Tabs 修复
4. Strategies — 编辑模态框，确认弹窗
5. Backtest — Modal a11y，键盘导航，导出
6. Risk — Toast 通知推送
7. PaperTrading — 统一 StatCard
8. Settings — 功能化

### Phase 5：打磨（预计 4h）
1. 所有图表色值改 CSS 变量
2. 表格交替色 + 排序功能
3. 移动端适配
4. 性能优化（React.memo, useMemo）
5. 全面 A11y 审查

---

## 预计总工作量

| Phase | 工时 |
|-------|------|
| Phase 1 基础设施 | 8h |
| Phase 2 导航布局 | 4h |
| Phase 3 数据层 | 6h |
| Phase 4 页面重构 | 12h |
| Phase 5 打磨 | 4h |
| **总计** | **~34h** |

---

## 成功标准

1. **单一应用**：web-btc 目录删除，所有功能在 web 内
2. **零 Math.random()**：无假数据，所有数据来自 API 或 Skeleton
3. **零无确认的危险操作**：平仓、停策略、删回测全有确认
4. **全键盘可操作**：Tab 导航覆盖所有交互元素
5. **WebSocket 实时**：行情延迟 < 100ms
6. **首屏 < 1s**：lazy loading + Skeleton
7. **零 console.error**：Error Boundary 兜底

---

## 不做的事（边界）

- 不换框架（不迁 Next.js / Remix）
- 不引入 CSS-in-JS（保持 Tailwind）
- 不引入 Storybook（项目规模不需要）
- 不做 i18n（目前中文单语言）
- 不做深色/浅色切换（保持深色主题）
