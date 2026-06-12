import type { BacktestResult, DrawdownData, TradeRecord } from '@/types';

/** Underwater series: non-positive drawdown % from running peak. */
export function equityToDrawdown(equity: { date: string; equity: number }[]): DrawdownData[] {
  let peak = equity[0]?.equity ?? 0;
  return equity.map((row) => {
    if (row.equity > peak) peak = row.equity;
    const dd = peak > 0 ? ((row.equity - peak) / peak) * 100 : 0;
    return { date: row.date, drawdown_pct: dd };
  });
}

export function maxDrawdownIndex(dd: DrawdownData[]): number {
  if (dd.length === 0) return -1;
  let min = 0;
  let idx = 0;
  dd.forEach((d, i) => {
    if (d.drawdown_pct < min) {
      min = d.drawdown_pct;
      idx = i;
    }
  });
  return idx;
}

/** Simple passive curve: smooth growth to a fraction of strategy terminal return. */
export function syntheticBuyHoldCurve(
  equity: { date: string; equity: number }[],
  initial: number,
  strategyTotalReturnPct: number,
  passiveFactor = 0.55,
): { date: string; benchmark: number }[] {
  if (equity.length === 0) return [];
  const passiveEnd = initial * (1 + (strategyTotalReturnPct / 100) * passiveFactor);
  const n = equity.length;
  return equity.map((row, i) => {
    const t = n <= 1 ? 1 : i / (n - 1);
    const bench = initial + (passiveEnd - initial) * t;
    return { date: row.date, benchmark: bench };
  });
}

export function mergeEquityAndBenchmark(
  equity: { date: string; equity: number }[],
  benchmark: { date: string; benchmark: number }[],
): { date: string; equity: number; benchmark: number; drawdown_pct: number }[] {
  const dd = equityToDrawdown(equity);
  const ddByDate = new Map(dd.map((d) => [d.date, d.drawdown_pct]));
  return equity.map((row, i) => ({
    date: row.date,
    equity: row.equity,
    benchmark: benchmark[i]?.benchmark ?? row.equity,
    drawdown_pct: ddByDate.get(row.date) ?? 0,
  }));
}

/** Deterministic mock trades for demos without backend fills. */
export function deriveMockTradesFromEquity(result: BacktestResult): TradeRecord[] {
  const { equity_curve: eq, initial_capital: initial } = result;
  if (eq.length < 20) return [];
  const trades: TradeRecord[] = [];
  const chunks = 6;
  const span = Math.floor((eq.length - 15) / chunks);
  for (let c = 0; c < chunks; c++) {
    const buyIdx = 5 + c * span;
    const sellIdx = Math.min(buyIdx + Math.max(3, Math.floor(span * 0.4)), eq.length - 1);
    const buyRow = eq[buyIdx];
    const sellRow = eq[sellIdx];
    if (!buyRow || !sellRow) continue;
    const base = 3000 + c * 37;
    const drift = (sellRow.equity - buyRow.equity) / initial;
    const buyPrice = base;
    const sellPrice = base * (1 + drift * 2.5 + (c % 2 === 0 ? 0.008 : -0.004));
    const holding = Math.max(1, sellIdx - buyIdx);
    trades.push({
      id: `${result.id}-B${c}`,
      side: 'BUY',
      date: buyRow.date,
      price: Math.round(buyPrice * 100) / 100,
      equity_at_trade: buyRow.equity,
      entry_price: buyPrice,
    });
    trades.push({
      id: `${result.id}-S${c}`,
      side: 'SELL',
      date: sellRow.date,
      price: Math.round(sellPrice * 100) / 100,
      equity_at_trade: sellRow.equity,
      pnl: Math.round((sellPrice - buyPrice) * 10) / 10,
      holding_period_days: holding,
      entry_price: buyPrice,
      exit_price: sellPrice,
    });
  }
  return trades;
}

/** Mark dates where drawdown recovers from a deep leg (for chart labels). */
export function recoveryAnnotations(dd: DrawdownData[]): { date: string; label: string }[] {
  const out: { date: string; label: string }[] = [];
  let wasDeep = false;
  for (let i = 0; i < dd.length; i++) {
    const v = dd[i]!.drawdown_pct;
    const prev = i > 0 ? dd[i - 1]!.drawdown_pct : 0;
    if (v < -1.5) wasDeep = true;
    if (wasDeep && v >= -0.05 && prev < -1.5) {
      out.push({ date: dd[i]!.date, label: '恢复' });
      wasDeep = false;
    }
  }
  const maxIdx = maxDrawdownIndex(dd);
  if (maxIdx >= 0) {
    for (let j = maxIdx + 1; j < dd.length; j++) {
      if (dd[j]!.drawdown_pct >= -0.05) {
        if (!out.some((x) => x.date === dd[j]!.date)) {
          out.unshift({ date: dd[j]!.date, label: '主要恢复' });
        }
        break;
      }
    }
  }
  return out.slice(0, 6);
}
