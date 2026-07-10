export interface Quote {
  instrument_id: string;
  exchange: string;
  last_price: number;
  pre_close: number;
  open: number;
  high: number;
  low: number;
  volume: number;
  amount: number;
  open_interest: number;
  bid_price1: number;
  bid_volume1: number;
  ask_price1: number;
  ask_volume1: number;
  upper_limit: number;
  lower_limit: number;
  change: number;
  change_percent: number;
  datetime: string;
}

export interface KlineBar {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface Position {
  instrument_id: string;
  exchange: string;
  direction: 'LONG' | 'SHORT';
  volume: number;
  available: number;
  avg_open_price: number;
  last_price: number;
  float_profit: number;
  margin: number;
}

export interface Order {
  order_id: string;
  instrument_id: string;
  direction: 'BUY' | 'SELL';
  offset: 'OPEN' | 'CLOSE' | 'CLOSETODAY';
  price: number;
  volume: number;
  traded_volume: number;
  status: 'PENDING' | 'PARTIAL' | 'FILLED' | 'CANCELLED' | 'REJECTED';
  insert_time: string;
}

export interface Trade {
  trade_id: string;
  order_id: string;
  instrument_id: string;
  direction: 'BUY' | 'SELL';
  offset: 'OPEN' | 'CLOSE' | 'CLOSETODAY';
  price: number;
  volume: number;
  trade_time: string;
  commission: number;
}

export interface AccountInfo {
  balance: number;
  available: number;
  margin: number;
  float_profit: number;
  position_profit: number;
  close_profit: number;
  commission: number;
  risk_ratio: number;
}

export interface Strategy {
  id: string;
  name: string;
  description: string;
  status: 'RUNNING' | 'STOPPED' | 'ERROR' | 'PAUSED';
  instruments: string[];
  params: Record<string, unknown>;
  pnl: number;
  created_at: string;
  updated_at: string;
}

/** Single trade marker for backtest visualization (may be synthesized in mock). */
export interface TradeRecord {
  id: string;
  side: 'BUY' | 'SELL';
  date: string;
  price: number;
  equity_at_trade?: number;
  /** Round-trip P&L (e.g. on exit leg). */
  pnl?: number;
  holding_period_days?: number;
  entry_price?: number;
  exit_price?: number;
}

/** Daily underwater drawdown as percentage (typically <= 0). */
export interface DrawdownData {
  date: string;
  drawdown_pct: number;
}

/** One cell in a parameter optimization grid. */
export interface ParameterSweepResult {
  id: string;
  param1_value: number | string;
  param2_value: number | string;
  param1_label?: string;
  param2_label?: string;
  sharpe: number;
  total_return: number;
  max_drawdown: number;
}

/** Aligned multi-series equity for strategy comparison charts. */
export interface BacktestCompareData {
  results: BacktestResult[];
  /** Rows: date + one numeric key per backtest id. */
  aligned_equity: { date: string; [seriesKey: string]: number | string }[];
}

export interface BacktestResult {
  id: string;
  strategy_name: string;
  start_date: string;
  end_date: string;
  initial_capital: number;
  final_capital: number;
  total_return: number;
  annual_return: number;
  max_drawdown: number;
  sharpe_ratio: number;
  win_rate: number;
  profit_factor: number;
  total_trades: number;
  equity_curve: { date: string; equity: number }[];
  status: 'RUNNING' | 'COMPLETED' | 'FAILED';
  created_at: string;
  /** Optional trade log for markers / tooltips. */
  trades?: TradeRecord[];
  /** Optional parameter optimization grid. */
  parameter_sweep?: ParameterSweepResult[];
}

export interface RiskAlert {
  id: string;
  level: 'INFO' | 'WARNING' | 'CRITICAL';
  type: string;
  message: string;
  instrument_id?: string;
  strategy_id?: string;
  created_at: string;
  resolved: boolean;
}

export interface KLine {
  datetime: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}
