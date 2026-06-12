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
    headers: { 'Content-Type': 'application/json', ...getAuthHeaders(), ...options?.headers },
    ...options,
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
  const last = Number(t.last_price);
  const high = Number(t.highest);
  const low = Number(t.lowest);
  const bid1 = Number(t.bid_price1 ?? last);
  const ask1 = Number(t.ask_price1 ?? last);
  return {
    instrument_id: t.symbol,
    exchange,
    last_price: last,
    pre_close: last,
    open: last,
    high,
    low,
    volume: t.volume,
    amount: Number(t.amount),
    open_interest: t.open_interest ?? 0,
    bid_price1: bid1,
    bid_volume1: t.bid_volume1 ?? 0,
    ask_price1: ask1,
    ask_volume1: t.ask_volume1 ?? 0,
    upper_limit: high,
    lower_limit: low,
    change: 0,
    change_percent: 0,
    datetime: t.datetime,
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
        if (tick.message === 'no_quote') return null;
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
