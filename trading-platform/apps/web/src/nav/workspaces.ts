/** Web v2 navigation config — four workspaces with secondary pages. */

export type WorkspaceId = 'research' | 'trading' | 'monitor' | 'platform';

export interface NavPage {
  to: string;
  label: string;
  keywords?: string[];
}

export interface Workspace {
  id: WorkspaceId;
  label: string;
  to: string;
  pages: NavPage[];
}

export const WORKSPACES: Workspace[] = [
  {
    id: 'research',
    label: '研究',
    to: '/research',
    pages: [
      { to: '/research', label: '工作台', keywords: ['research', 'run', '工作台'] },
      { to: '/research/factors', label: '因子', keywords: ['factor', 'ic', '因子'] },
      { to: '/research/strategies', label: '策略', keywords: ['strategy', '策略'] },
      { to: '/research/backtest', label: '回测', keywords: ['backtest', '回测'] },
      { to: '/research/optimizer', label: '优化', keywords: ['optimizer', 'optuna', '冠军'] },
      { to: '/research/deploy', label: '部署', keywords: ['deploy', '部署', '参数'] },
      { to: '/research/ml', label: '模型', keywords: ['ml', '模型', '训练', 'xgboost'] },
    ],
  },
  {
    id: 'trading',
    label: '交易',
    to: '/trading/paper',
    pages: [
      { to: '/trading/paper', label: '模拟实盘', keywords: ['paper', '模拟'] },
      { to: '/trading/live', label: '实盘', keywords: ['live', '实盘'] },
      { to: '/trading/manual', label: '手动交易', keywords: ['manual', '下单'] },
    ],
  },
  {
    id: 'monitor',
    label: '监控',
    to: '/monitor/risk',
    pages: [
      { to: '/monitor/risk', label: '风控', keywords: ['risk', '风控'] },
      { to: '/monitor/events', label: '事件流', keywords: ['events', 'websocket'] },
      { to: '/monitor/explain', label: '证据链', keywords: ['explain', '证据', '审计'] },
      { to: '/monitor/health', label: '系统健康', keywords: ['health', 'gateway'] },
    ],
  },
  {
    id: 'platform',
    label: '平台',
    to: '/platform/settings',
    pages: [
      { to: '/platform/data', label: '数据', keywords: ['data', '采集'] },
      { to: '/platform/skills', label: '技能', keywords: ['skills', 'sop'] },
      { to: '/platform/settings', label: '设置', keywords: ['settings', '设置'] },
    ],
  },
];

export function workspaceForPath(pathname: string): WorkspaceId {
  if (pathname.startsWith('/trading') || pathname.startsWith('/crypto')) return 'trading';
  if (pathname.startsWith('/monitor') || pathname.startsWith('/risk')) return 'monitor';
  if (pathname.startsWith('/platform') || pathname.startsWith('/settings')) return 'platform';
  if (
    pathname.startsWith('/research') ||
    pathname.startsWith('/strategies') ||
    pathname.startsWith('/backtest') ||
    pathname.startsWith('/strategy')
  ) {
    return 'research';
  }
  return 'research';
}

export function allCommandItems(): { to: string; label: string; group: string; keywords: string[] }[] {
  const items = WORKSPACES.flatMap((ws) =>
    ws.pages.map((p) => ({
      to: p.to,
      label: p.label,
      group: ws.label,
      keywords: [p.label, ws.label, ...(p.keywords || [])],
    })),
  );
  items.push(
    { to: '/', label: '仪表盘', group: '总览', keywords: ['dashboard', '首页', '仪表盘'] },
    { to: '/research/backtest/compare', label: '回测对比', group: '研究', keywords: ['compare', '对比'] },
  );
  return items;
}
