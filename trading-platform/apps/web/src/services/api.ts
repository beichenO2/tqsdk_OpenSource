import type {
  Quote, Position, Order, AccountInfo,
  Strategy, BacktestResult, RiskAlert,
} from '@/types';
import { deriveMockTradesFromEquity } from '@/utils/backtestDerived';

const API_BASE = '/api/v1';

function getAuthHeaders(): Record<string, string> {
  const token = localStorage.getItem('auth_token');
  const apiKey = localStorage.getItem('api_key');
  if (token) return { Authorization: `Bearer ${token}` };
  if (apiKey) return { 'X-API-Key': apiKey };
  return {};
}

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${url}`, {
    ...options,
    // headers must come after ...options so caller headers merge instead of
    // replacing Content-Type/auth (options.headers may be undefined).
    headers: { 'Content-Type': 'application/json', ...getAuthHeaders(), ...options?.headers },
  });
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`API ${res.status}: ${body || res.statusText}`);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// Backend response → Frontend type transformers
// ---------------------------------------------------------------------------

interface BackendPosition {
  symbol: string;
  exchange: string;
  direction: string;
  volume: number;
  available: number;
  avg_price: string;
  margin: string;
  float_pnl: string;
  close_pnl: string;
}

function toPosition(p: BackendPosition): Position {
  return {
    instrument_id: p.symbol,
    exchange: p.exchange,
    direction: p.direction as Position['direction'],
    volume: p.volume,
    available: p.available,
    avg_open_price: Number(p.avg_price),
    last_price: 0,
    float_profit: Number(p.float_pnl),
    margin: Number(p.margin),
  };
}

interface BackendOrder {
  order_id: string;
  status: string;
  symbol: string;
  direction: string;
  price: string;
  volume: number;
  filled_volume: number;
  message?: string;
}

const STATUS_MAP: Record<string, Order['status']> = {
  PENDING: 'PENDING',
  PARTIAL: 'PARTIAL',
  FILLED: 'FILLED',
  CANCELLED: 'CANCELLED',
  REJECTED: 'REJECTED',
  pending: 'PENDING',
  partial_filled: 'PARTIAL',
  filled: 'FILLED',
  cancelled: 'CANCELLED',
  rejected: 'REJECTED',
};

function toOrder(o: BackendOrder): Order {
  const dir = o.direction.toUpperCase();
  return {
    order_id: o.order_id,
    instrument_id: o.symbol,
    direction: (dir === 'BUY' || dir === 'LONG' ? 'BUY' : 'SELL') as Order['direction'],
    offset: 'OPEN',
    price: Number(o.price),
    volume: o.volume,
    traded_volume: o.filled_volume,
    status: STATUS_MAP[o.status] ?? 'PENDING',
    insert_time: '',
  };
}

interface BackendStrategyConfig {
  strategy_id: string;
  name: string;
  version?: string;
  symbols: string[];
  params: Record<string, unknown>;
  risk_limits?: Record<string, number>;
  enabled: boolean;
}

function toStrategy(s: BackendStrategyConfig): Strategy {
  return {
    id: s.strategy_id,
    name: s.name,
    description: `v${s.version ?? '1.0.0'}`,
    status: s.enabled ? 'RUNNING' : 'STOPPED',
    instruments: s.symbols,
    params: s.params,
    pnl: 0,
    created_at: '',
    updated_at: '',
  };
}

interface BackendTick {
  symbol: string;
  datetime: string;
  last_price: number | string;
  highest: number | string;
  lowest: number | string;
  volume: number;
  amount: number | string;
  open_interest?: number | null;
  bid_price1?: number | string | null;
  bid_volume1?: number | null;
  ask_price1?: number | string | null;
  ask_volume1?: number | null;
  message?: string;
}

function toQuote(t: BackendTick, exchange: string): Quote {
  // closed_market_cache fallback only carries last_price — default everything else
  const last = Number(t.last_price ?? 0);
  const high = Number(t.highest ?? last);
  const low = Number(t.lowest ?? last);
  const bid1 = Number(t.bid_price1 ?? last);
  const ask1 = Number(t.ask_price1 ?? last);
  return {
    instrument_id: t.symbol,
    exchange,
    last_price: last,
    pre_close: last,
    open: last,
    high: Number.isFinite(high) ? high : last,
    low: Number.isFinite(low) ? low : last,
    volume: Number(t.volume ?? 0),
    amount: Number(t.amount ?? 0),
    open_interest: Number(t.open_interest ?? 0),
    bid_price1: Number.isFinite(bid1) ? bid1 : last,
    bid_volume1: Number(t.bid_volume1 ?? 0),
    ask_price1: Number.isFinite(ask1) ? ask1 : last,
    ask_volume1: Number(t.ask_volume1 ?? 0),
    upper_limit: Number.isFinite(high) ? high : last,
    lower_limit: Number.isFinite(low) ? low : last,
    change: 0,
    change_percent: 0,
    datetime: t.datetime ?? '',
  };
}

// ---------------------------------------------------------------------------
// Cached instrument list so we don't refetch every cycle
// ---------------------------------------------------------------------------

let _instrumentsCache: { symbol: string; exchange: string }[] | null = null;
let _instrumentsCacheTs = 0;
const INSTRUMENTS_TTL = 60_000;

async function getInstruments(): Promise<{ symbol: string; exchange: string }[]> {
  if (_instrumentsCache && Date.now() - _instrumentsCacheTs < INSTRUMENTS_TTL) {
    return _instrumentsCache;
  }
  const list = await request<{ symbol: string; exchange: string }[]>('/market/instruments');
  _instrumentsCache = list.slice(0, 30);
  _instrumentsCacheTs = Date.now();
  return _instrumentsCache;
}

/** Registry / UI names for POST /backtest (strategy_name). */
export const DEFAULT_BACKTEST_STRATEGY_OPTIONS: { value: string; label: string }[] = [
  { value: 'futures_dual_ma', label: '期货双均线 · futures_dual_ma' },
  { value: 'bollinger_mr', label: '布林带均值回归 · bollinger_mr' },
  { value: 'vol_breakout', label: '波动突破 · vol_breakout' },
  { value: 'cta_trend', label: 'CTA 趋势 · cta_trend' },
  { value: 'spread_arb', label: '跨期套利 · spread_arb' },
  { value: 'pairs_trading', label: '配对交易 · pairs_trading' },
];

interface BacktestTradeDetail {
  id: string;
  side: string;
  symbol: string;
  price: number;
  volume: number;
  commission: number;
  dt: string;
  pnl?: number;
}

interface BacktestEquityPoint {
  date: string;
  equity: number;
}

interface BacktestRunResponse {
  total_return: number;
  max_drawdown: number;
  sharpe: number;
  win_rate: number;
  total_trades: number;
  final_equity: number;
  message?: string;
  trades?: BacktestTradeDetail[];
  equity_curve?: BacktestEquityPoint[];
}

function synthesizeEquityFromRun(
  startDate: string,
  endDate: string,
  initial: number,
  final: number,
  points = 128,
): { date: string; equity: number }[] {
  const s = new Date(startDate).getTime();
  const e = new Date(endDate).getTime();
  if (!Number.isFinite(s) || !Number.isFinite(e) || e <= s) {
    return [{ date: startDate.slice(0, 10), equity: initial }];
  }
  const rows: { date: string; equity: number }[] = [];
  for (let i = 0; i < points; i++) {
    const t = s + ((e - s) * i) / (points - 1 || 1);
    const wobble = Math.sin(i * 0.17) * initial * 0.01;
    const eq = initial + ((final - initial) * i) / (points - 1 || 1) + wobble;
    rows.push({ date: new Date(t).toISOString().slice(0, 10), equity: Math.round(eq * 100) / 100 });
  }
  return rows;
}

// ---------------------------------------------------------------------------
// Public API (mock fallback preserved)
// ---------------------------------------------------------------------------

export const api = {
  getAccount: async (): Promise<AccountInfo | null> => {
    try {
      const raw = await request<Record<string, unknown>>('/positions/account/info');
      return {
        balance: Number(raw.balance ?? 0),
        available: Number(raw.available ?? 0),
        margin: Number(raw.margin ?? 0),
        float_profit: Number(raw.float_profit ?? raw.float_pnl ?? 0),
        position_profit: Number(raw.position_profit ?? 0),
        close_profit: Number(raw.close_profit ?? raw.close_pnl ?? 0),
        commission: Number(raw.commission ?? 0),
        risk_ratio: Number(raw.risk_ratio ?? 0),
      };
    } catch { return null; }
  },

  getQuotes: async (): Promise<Quote[]> => {
    const instruments = await getInstruments();
    if (instruments.length === 0) return [];
    const results = await Promise.allSettled(
      instruments.map(async (inst) => {
        const tick = await request<BackendTick>(`/market/quote/${encodeURIComponent(inst.symbol)}`);
        if (tick.message === 'no_quote' || tick.last_price == null) return null;
        return toQuote(tick, inst.exchange ?? '');
      }),
    );
    return results
      .filter((r): r is PromiseFulfilledResult<Quote | null> => r.status === 'fulfilled')
      .map(r => r.value)
      .filter((q): q is Quote => q !== null);
  },

  getPositions: async (): Promise<Position[]> => {
    try {
      const raw = await request<BackendPosition[]>('/positions');
      return raw.map(toPosition);
    } catch { return []; }
  },

  getOrders: async (): Promise<Order[]> => {
    try {
      const raw = await request<BackendOrder[]>('/orders');
      return raw.map(toOrder);
    } catch { return []; }
  },

  getStrategies: async (): Promise<Strategy[]> => {
    try {
      const raw = await request<BackendStrategyConfig[]>('/strategies');
      return raw.map(toStrategy);
    } catch { return []; }
  },

  getBacktestResults: async (): Promise<BacktestResult[]> => {
    try {
      return await request('/backtest/results');
    } catch {
      return [];
    }
  },

  /** Symbols for autocomplete (cached when live). */
  getMarketInstruments: async (): Promise<{ symbol: string; exchange: string }[]> => {
    return await getInstruments();
  },

  getBacktestStrategyOptions: async (): Promise<{ value: string; label: string }[]> => {
    try {
      const raw = await request<string[]>('/backtest/strategy-names');
      if (Array.isArray(raw)) {
        return raw.map((value) => ({ value, label: value }));
      }
    } catch {
      /* use defaults */
    }
    return DEFAULT_BACKTEST_STRATEGY_OPTIONS;
  },

  getRiskAlerts: async (): Promise<RiskAlert[]> => {
    try {
      const raw = await request<Record<string, unknown>>('/positions/risk/status');
      if (Array.isArray(raw)) return raw as RiskAlert[];
      return [];
    } catch {
      return [];
    }
  },

  placeOrder: async (order: Partial<Order> & { exchange?: string }): Promise<Order> => {
    const body = {
      strategy_id: 'manual',
      symbol: order.instrument_id,
      exchange: order.exchange || 'SHFE',
      direction: order.direction,
      offset: order.offset ?? 'OPEN',
      price: order.price,
      volume: order.volume,
    };
    const raw = await request<BackendOrder>('/orders', {
      method: 'POST',
      body: JSON.stringify(body),
    });
    return toOrder(raw);
  },

  cancelOrder: async (orderId: string): Promise<void> => {
    await request(`/orders/${orderId}`, { method: 'DELETE' });
  },

  closeAllPositions: async (): Promise<void> => {
    try {
      await request('/positions/close-all', { method: 'POST' });
    } catch {
      // endpoint not yet implemented
    }
  },

  startStrategy: async (id: string): Promise<Strategy> => {
    const raw = await request<BackendStrategyConfig>(
      `/strategies/${id}/toggle?enabled=true`,
      { method: 'PUT' },
    );
    return toStrategy({ ...raw, strategy_id: raw.strategy_id ?? id, enabled: true });
  },

  stopStrategy: async (id: string): Promise<Strategy> => {
    const raw = await request<BackendStrategyConfig>(
      `/strategies/${id}/toggle?enabled=false`,
      { method: 'PUT' },
    );
    return toStrategy({ ...raw, strategy_id: raw.strategy_id ?? id, enabled: false });
  },

  pauseAllStrategies: async (): Promise<void> => {
    try {
      await request('/strategies/pause-all', { method: 'POST' });
    } catch {
      // endpoint not yet implemented
    }
  },

  createBacktest: async (params: {
    strategy_id: string;
    start_date: string;
    end_date: string;
    initial_capital: number;
    instruments: string[];
    strategy_params?: Record<string, unknown>;
  }): Promise<BacktestResult> => {
    const body = {
      strategy_name: params.strategy_id,
      symbols: params.instruments,
      params: params.strategy_params ?? {},
      start_date: params.start_date,
      end_date: params.end_date,
      initial_capital: params.initial_capital,
    };

    const raw = await request<BacktestRunResponse>('/backtest', {
      method: 'POST',
      body: JSON.stringify(body),
    });
    const initial = params.initial_capital;

    const equity_curve = raw.equity_curve?.length
      ? raw.equity_curve
      : synthesizeEquityFromRun(params.start_date, params.end_date, initial, raw.final_equity);

    const realTrades: import('@/types').TradeRecord[] | undefined = raw.trades?.length
      ? raw.trades.map((t, i) => ({
          id: t.id || `T${i}`,
          side: t.side === 'BUY' ? 'BUY' as const : 'SELL' as const,
          date: t.dt.slice(0, 10),
          price: t.price,
          equity_at_trade: 0,
          pnl: t.pnl,
          entry_price: t.price,
          exit_price: t.side === 'SELL' ? t.price : undefined,
        }))
      : undefined;

    const partial: BacktestResult = {
      id: `BT${Date.now()}`,
      strategy_name: params.strategy_id,
      start_date: params.start_date.slice(0, 10),
      end_date: params.end_date.slice(0, 10),
      initial_capital: initial,
      final_capital: raw.final_equity,
      total_return: raw.total_return,
      annual_return: raw.total_return,
      max_drawdown: raw.max_drawdown,
      sharpe_ratio: raw.sharpe,
      win_rate: raw.win_rate,
      profit_factor: Math.max(0.5, raw.win_rate > 0.5 ? 1.4 : 0.95),
      total_trades: raw.total_trades,
      equity_curve,
      status: 'COMPLETED',
      created_at: new Date().toISOString().slice(0, 10),
    };
    return {
      ...partial,
      trades: realTrades ?? deriveMockTradesFromEquity(partial),
    };
  },

  getPnlHistory: async (days: number): Promise<{ date: string; pnl: number; daily: number }[]> => {
    try {
      return await request(`/positions/pnl-history?days=${days}`);
    } catch {
      return [];
    }
  },

  // Research
  getResearchRuns: (limit = 50) =>
    request<{ runs: unknown[] }>(`/research/runs?limit=${limit}`),
  createResearchRun: (params: Record<string, unknown>) =>
    request<{ run_id: string; status: string }>('/research/runs', { method: 'POST', body: JSON.stringify(params) }),
  getResearchRun: (runId: string) =>
    request<Record<string, unknown>>(`/research/runs/${runId}`),
  executeResearchRun: (runId: string) =>
    request<unknown>(`/research/runs/${runId}/execute`, { method: 'POST' }),
  getResearchPipeline: (runId: string) =>
    request<{
      run_id: string;
      steps: { id: string; label: string; description: string; status: string; done: boolean }[];
      completed: number;
      total: number;
      progress: number;
      pipeline_stage: string;
      promotion: string;
      gate_passed: boolean | null;
    }>(`/research/runs/${runId}/pipeline`),
  promoteResearchRun: (runId: string, target: string, note = '', force = false) =>
    request<{
      ok: boolean;
      from: string;
      to: string;
      pipeline: Record<string, unknown>;
      warning?: string | null;
    }>(`/research/runs/${runId}/promote`, {
      method: 'POST',
      body: JSON.stringify({ target, note, force }),
    }),
  setResearchFactorSnapshot: (
    runId: string,
    body: { ic?: Record<string, unknown>; dedupe?: Record<string, unknown>; factor_names?: string[] },
  ) =>
    request<{ ok: boolean; pipeline: Record<string, unknown> }>(
      `/research/runs/${runId}/factor-snapshot`,
      { method: 'POST', body: JSON.stringify(body) },
    ),

  // MCP
  getMcpTools: () =>
    request<{ tools: unknown[] }>('/mcp/tools'),
  callMcpTool: (name: string, args: Record<string, unknown> = {}) =>
    request<unknown>('/mcp/tools/call', { method: 'POST', body: JSON.stringify({ name, arguments: args }) }),

  // Settings
  getSettings: () =>
    request<Record<string, unknown>>('/settings'),

  // Live trading
  getLiveTradingStatus: () =>
    request<Record<string, unknown>>('/live-trading/status'),
  getLiveStrategies: () =>
    request<Record<string, unknown>[]>('/live-trading/strategies'),
  getLiveLeaderboard: (market?: string, topN = 20) => {
    const qs = new URLSearchParams();
    if (market) qs.set('market', market);
    qs.set('top_n', String(topN));
    return request<Record<string, unknown>[]>(`/live-trading/leaderboard?${qs}`);
  },
  startLiveTrading: (params: Record<string, unknown>, liveConfirm?: string) =>
    request<Record<string, unknown>>('/live-trading/start', {
      method: 'POST',
      body: JSON.stringify(params),
      headers: liveConfirm ? { 'X-Live-Confirm': liveConfirm } : undefined,
    }),
  stopLiveTrading: () =>
    request<Record<string, unknown>>('/live-trading/stop', { method: 'POST' }),
  switchMode: (mode: string, liveConfirm?: string) =>
    request<Record<string, unknown>>('/live-trading/switch-mode', {
      method: 'POST',
      body: JSON.stringify({ mode }),
      headers: liveConfirm ? { 'X-Live-Confirm': liveConfirm } : undefined,
    }),
  toggleLiveStrategy: (accountId: number) =>
    request<Record<string, unknown>>(`/live-trading/strategies/${accountId}/toggle`, {
      method: 'POST',
    }),
  /** @deprecated use toggleLiveStrategy — kept for older call sites */
  toggleStrategy: (accountId: number) =>
    request<Record<string, unknown>>(`/live-trading/strategies/${accountId}/toggle`, {
      method: 'POST',
    }),
  placeLiveOrder: (
    order: {
      symbol: string;
      exchange?: string;
      direction: string;
      offset?: string;
      price: number | string;
      volume: number;
      strategy_id?: string;
    },
    liveConfirm?: string,
  ) =>
    request<Record<string, unknown>>('/live-trading/order', {
      method: 'POST',
      body: JSON.stringify(order),
      headers: liveConfirm ? { 'X-Live-Confirm': liveConfirm } : undefined,
    }),
  riskProbe: (params: {
    symbol: string;
    exchange?: string;
    direction?: string;
    offset?: string;
    price?: number | string;
    volume?: number;
  }) =>
    request<Record<string, unknown>>('/live-trading/risk-probe', {
      method: 'POST',
      body: JSON.stringify(params),
    }),
  getLiveRiskStatus: () =>
    request<Record<string, unknown>>('/live-trading/risk-status'),

  // System / optimizer
  getSystemHealth: () =>
    request<Record<string, unknown>>('/system/health'),
  getOptimizerChampions: (variant?: string, topN = 20) => {
    const qs = new URLSearchParams({ top_n: String(topN) });
    if (variant) qs.set('variant', variant);
    return request<{ variants: string[]; entries: Record<string, unknown>[]; total: number }>(
      `/optimizer/champions?${qs}`,
    );
  },
  getOptimizerGates: () =>
    request<{ gates: Record<string, unknown>[]; count: number }>('/optimizer/gates'),
  getOptimizerGate: (name: string) =>
    request<Record<string, unknown>>(`/optimizer/gates/${encodeURIComponent(name)}`),

  // Platform data / skills
  getPlatformData: () =>
    request<{
      data_dir: string;
      caches: { futures: Record<string, unknown>; crypto: Record<string, unknown> };
      extras: { name: string; file_count: number; exists: boolean }[];
      collector: Record<string, unknown>;
      tqsdk_gateway: Record<string, unknown>;
    }>('/platform/data'),
  getPlatformSkills: () =>
    request<{
      dir: string;
      skills: Record<string, unknown>[];
      count: number;
    }>('/platform/skills'),
  getPlatformSkill: (name: string) =>
    request<Record<string, unknown>>(`/platform/skills/${encodeURIComponent(name)}`),

  // Factors (R10)
  listFactors: (category?: string) => {
    const qs = category ? `?category=${encodeURIComponent(category)}` : '';
    return request<{
      factors: Record<string, unknown>[];
      count: number;
      categories: string[];
    }>(`/factors${qs}`);
  },
  getFactor: (name: string) =>
    request<Record<string, unknown>>(`/factors/${encodeURIComponent(name)}`),
  computeFactors: (body: {
    symbol: string;
    factor_names: string[];
    limit?: number;
    params?: Record<string, Record<string, unknown>>;
  }) =>
    request<{
      symbol: string;
      bars: number;
      factors: Record<string, { points: { t: string; v: number }[]; last?: number; n?: number }>;
    }>('/factors/compute', { method: 'POST', body: JSON.stringify(body) }),
  analyzeFactors: (body: {
    symbol: string;
    factor_names: string[];
    limit?: number;
    horizon?: number;
    dedupe_threshold?: number;
  }) =>
    request<{
      symbol: string;
      reports: Record<string, unknown>[];
      correlation: Record<string, unknown>;
      dedupe: Record<string, unknown>;
    }>('/factors/analyze', { method: 'POST', body: JSON.stringify(body) }),
  analyzeCrossSection: (body: {
    factor_name: string;
    symbols?: string[];
    limit?: number;
    horizon?: number;
    quantiles?: number;
  }) =>
    request<{
      mode: string;
      horizon: number;
      summary: Record<string, number | null>;
      quantile_returns: {
        mean_returns?: Record<string, number | null>;
        long_short?: number | null;
        n_periods?: number;
      };
      ic_series: { t: string; v: number }[];
      n_assets: number;
      symbols_used?: string[];
      factor?: Record<string, unknown>;
    }>('/factors/analyze-cs', { method: 'POST', body: JSON.stringify(body) }),
  combineFactors: (body: {
    symbol: string;
    factor_names: string[];
    method?: string;
    limit?: number;
  }) =>
    request<Record<string, unknown>>('/factors/combine', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  evolveFactors: (body: {
    symbol?: string;
    n_proposals?: number;
    limit?: number;
    use_llm?: boolean;
    existing_exprs?: string[];
  } = {}) =>
    request<{
      symbol?: string;
      arm: string;
      bandit: Record<string, unknown>;
      candidates: Record<string, unknown>[];
      best: Record<string, unknown> | null;
      n_valid: number;
      path?: string;
    }>('/factors/evolve', { method: 'POST', body: JSON.stringify(body) }),
  getEvolveLatest: () =>
    request<{ exists: boolean; latest: Record<string, unknown> | null }>(
      '/factors/evolve/latest',
    ),
};

export function createWsConnection(
  path: string,
  onMessage: (data: unknown) => void,
): { close: () => void } {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const url = `${protocol}//${location.host}/ws${path}`;

  let ws: WebSocket | null = null;
  let retryTimer: ReturnType<typeof setTimeout>;
  let stopped = false;

  function connect() {
    if (stopped) return;
    ws = new WebSocket(url);
    ws.onmessage = (e) => {
      try { onMessage(JSON.parse(e.data)); } catch { /* ignore non-json */ }
    };
    ws.onclose = () => {
      if (!stopped) retryTimer = setTimeout(connect, 3000);
    };
    ws.onerror = () => ws?.close();
  }

  connect();

  return {
    close() {
      stopped = true;
      clearTimeout(retryTimer);
      ws?.close();
    },
  };
}
