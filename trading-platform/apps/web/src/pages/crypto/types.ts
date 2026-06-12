export interface CryptoQuote {
  symbol: string;
  exchange: string;
  last_price: number;
  price_change_24h: number;
  price_change_percent_24h: number;
  high_24h: number;
  low_24h: number;
  volume_24h: number;
  turnover_24h: number;
  bid_price: number;
  bid_qty: number;
  ask_price: number;
  ask_qty: number;
  funding_rate: number;
  next_funding_time: string;
  open_interest: number;
  datetime: string;
}

export interface CryptoPosition {
  symbol: string;
  exchange: string;
  side: 'LONG' | 'SHORT';
  size: number;
  entry_price: number;
  mark_price: number;
  liq_price: number;
  unrealized_pnl: number;
  leverage: number;
  margin: number;
  margin_mode: 'cross' | 'isolated';
}

export interface CryptoOrder {
  order_id: string;
  symbol: string;
  side: 'BUY' | 'SELL';
  type: 'LIMIT' | 'MARKET' | 'STOP_LIMIT' | 'STOP_MARKET';
  price: number;
  qty: number;
  filled_qty: number;
  status: 'NEW' | 'PARTIALLY_FILLED' | 'FILLED' | 'CANCELLED' | 'REJECTED';
  reduce_only: boolean;
  created_at: string;
}

export interface CryptoAccount {
  total_equity: number;
  available_balance: number;
  total_margin: number;
  unrealized_pnl: number;
  realized_pnl_24h: number;
  margin_ratio: number;
  btc_equivalent: number;
}

export interface CryptoStrategy {
  id: string;
  name: string;
  type: 'grid' | 'momentum' | 'mean_reversion' | 'arbitrage';
  status: 'RUNNING' | 'STOPPED' | 'ERROR' | 'PAUSED';
  symbols: string[];
  params: Record<string, unknown>;
  pnl_24h: number;
  total_pnl: number;
  created_at: string;
  updated_at: string;
}

export interface CryptoBacktestResult {
  id: string;
  strategy_name: string;
  strategy_type: string;
  symbol: string;
  start_date: string;
  end_date: string;
  initial_capital: number;
  final_capital: number;
  total_return: number;
  annual_return: number;
  max_drawdown: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  win_rate: number;
  profit_factor: number;
  total_trades: number;
  avg_holding_hours: number;
  equity_curve: { date: string; equity: number }[];
  status: 'RUNNING' | 'COMPLETED' | 'FAILED';
  created_at: string;
}
