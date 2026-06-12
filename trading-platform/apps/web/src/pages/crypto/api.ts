import type {
  CryptoQuote, CryptoPosition, CryptoOrder,
  CryptoAccount, CryptoStrategy, CryptoBacktestResult,
} from './types';

const API_BASE = '/api/v1/btc';

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${url}`, {
    headers: { 'Content-Type': 'application/json', ...options?.headers },
    ...options,
  });
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`API ${res.status}: ${body || res.statusText}`);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

const WATCHED_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT'];

export const cryptoApi = {
  getAccount: async (): Promise<CryptoAccount | null> => {
    try {
      const balances = await request<any[]>('/account/balances');
      if (Array.isArray(balances) && balances.length === 0) return null;
      return balances as unknown as CryptoAccount;
    } catch { return null; }
  },

  getQuotes: async (): Promise<CryptoQuote[]> => {
    const results = await Promise.allSettled(
      WATCHED_SYMBOLS.map(s => request<CryptoQuote>(`/market/ticker/${s}`)),
    );
    return results.filter((r): r is PromiseFulfilledResult<CryptoQuote> => r.status === "fulfilled").map(r => r.value);
  },

  getPositions: async (): Promise<CryptoPosition[]> => {
    try {
      const data = await request<any>('/account/positions');
      return Array.isArray(data) ? data : data?.positions ?? [];
    } catch { return []; }
  },

  getOrders: async (): Promise<CryptoOrder[]> => {
    try {
      const data = await request<any>('/orders');
      return Array.isArray(data) ? data : data?.orders ?? [];
    } catch { return []; }
  },

  getStrategies: async (): Promise<CryptoStrategy[]> => {
    try {
      const data = await request<any>('/strategies');
      return Array.isArray(data) ? data : data?.strategies ?? [];
    } catch { return []; }
  },

  getBacktestResults: async (): Promise<CryptoBacktestResult[]> => {
    try {
      const data = await request<any>('/backtest');
      return Array.isArray(data) ? data : data?.backtests ?? [];
    } catch { return []; }
  },

  placeOrder: (order: {
    symbol: string;
    side: 'BUY' | 'SELL';
    type: 'LIMIT' | 'MARKET';
    price?: number;
    qty: number;
    leverage?: number;
    reduce_only?: boolean;
  }): Promise<CryptoOrder> =>
    request('/orders', {
      method: 'POST',
      body: JSON.stringify({
        exchange: 'binance',
        symbol: order.symbol,
        side: order.side.toLowerCase(),
        order_type: order.type.toLowerCase(),
        quantity: order.qty,
        price: order.price,
      }),
    }),

  cancelOrder: (orderId: string, symbol = 'BTCUSDT'): Promise<void> =>
    request(`/orders/${orderId}?exchange=binance&symbol=${symbol}`, { method: 'DELETE' }),

  closePosition: (symbol: string): Promise<void> =>
    request('/orders', {
      method: 'POST',
      body: JSON.stringify({
        exchange: 'binance',
        symbol,
        side: 'sell',
        order_type: 'market',
        quantity: 0,
      }),
    }).then(() => undefined),

  closeAllPositions: (): Promise<void> =>
    cryptoApi.getPositions().then(positions =>
      Promise.all(positions.map(p => cryptoApi.closePosition(p.symbol))),
    ).then(() => undefined),

  runBacktest: (params: {
    strategy_name: string;
    symbol?: string;
    start_date: string;
    end_date: string;
    initial_capital?: number;
    params?: Record<string, unknown>;
  }): Promise<{ backtest_id: string }> =>
    request('/backtest', {
      method: 'POST',
      body: JSON.stringify({
        strategy_name: params.strategy_name,
        symbol: params.symbol ?? 'BTCUSDT',
        exchange: 'binance',
        interval: '1h',
        start_date: params.start_date,
        end_date: params.end_date,
        initial_capital: params.initial_capital ?? 10000,
        params: params.params ?? {},
      }),
    }),

  getExchanges: async (): Promise<{ id: string; name: string; connected: boolean }[]> => {
    try {
      const data = await request<any>('/exchanges');
      return data?.exchanges ?? (Array.isArray(data) ? data : []);
    } catch { return []; }
  },
};
