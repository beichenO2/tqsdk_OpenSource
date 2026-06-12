import { useQuery } from '@tanstack/react-query';

export interface BacktestEntry {
  id: string;
  source_file: string;
  symbol: string;
  strategy_name: string;
  market: 'crypto' | 'futures';
  is_split: boolean;
  split_type: string | null;
  initial_capital: number;
  final_capital: number;
  total_return: number;
  max_drawdown: number;
  sharpe_ratio: number;
  calmar_ratio: number | null;
  total_trades: number;
  win_rate: number;
  profit_factor: number;
  avg_win: number;
  avg_loss: number;
  payoff_ratio: number;
  total_commission: number;
  bars_tested: number;
  date_range: string;
  round_trips: number;
}

interface OptimizerRound {
  round: number;
  file: string;
  best_score: number | null;
  sharpe: number | null;
  total_return: number | null;
  max_drawdown: number | null;
  params: Record<string, number> | null;
}

interface BacktestsData {
  backtests: BacktestEntry[];
  optuna: { file: string; data: Record<string, unknown> }[];
}

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch ${url}`);
  return res.json();
}

export function useRealBacktests() {
  return useQuery<BacktestsData>({
    queryKey: ['static-backtests'],
    queryFn: () => fetchJson('/data/backtests.json'),
    staleTime: Infinity,
  });
}

export function useOptimizerRounds() {
  return useQuery<OptimizerRound[]>({
    queryKey: ['static-optimizer'],
    queryFn: () => fetchJson('/data/optimizer.json'),
    staleTime: Infinity,
  });
}
