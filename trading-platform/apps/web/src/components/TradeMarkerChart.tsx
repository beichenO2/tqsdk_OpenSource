import { useMemo, useState } from 'react';
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type { TradeRecord } from '@/types';
import { Filter } from 'lucide-react';

type Row = { date: string; equity: number; buy: number | null; sell: number | null };

function buildRows(
  equity: { date: string; equity: number }[],
  trades: TradeRecord[],
  filter: 'all' | 'win' | 'loss',
): Row[] {
  const sells = trades.filter((t) => t.side === 'SELL');
  const winSellIds = new Set(sells.filter((t) => (t.pnl ?? 0) >= 0).map((t) => t.id));
  const lossSellIds = new Set(sells.filter((t) => (t.pnl ?? 0) < 0).map((t) => t.id));

  const includeSell = (t: TradeRecord) => {
    if (t.side !== 'SELL') return true;
    if (filter === 'all') return true;
    if (filter === 'win') return winSellIds.has(t.id);
    return lossSellIds.has(t.id);
  };

  const includeBuy = (t: TradeRecord) => {
    if (t.side !== 'BUY') return true;
    if (filter === 'all') return true;
    const exit = sells.find((s) => s.date >= t.date && Math.abs((s.entry_price ?? 0) - (t.entry_price ?? t.price)) < 1e-6);
    if (!exit) return false;
    if (filter === 'win') return winSellIds.has(exit.id);
    return lossSellIds.has(exit.id);
  };

  const byDate = new Map<string, TradeRecord[]>();
  for (const t of trades) {
    if (t.side === 'SELL' && !includeSell(t)) continue;
    if (t.side === 'BUY' && !includeBuy(t)) continue;
    const arr = byDate.get(t.date) ?? [];
    arr.push(t);
    byDate.set(t.date, arr);
  }

  return equity.map((e) => {
    const list = byDate.get(e.date) ?? [];
    let buy: number | null = null;
    let sell: number | null = null;
    for (const t of list) {
      if (t.side === 'BUY') buy = e.equity;
      if (t.side === 'SELL') sell = e.equity;
    }
    return { date: e.date, equity: e.equity, buy, sell };
  });
}

const tooltipBox =
  'max-w-xs rounded-lg border border-border bg-surface-secondary px-3 py-2 text-xs shadow-lg';

export interface TradeMarkerChartProps {
  equity: { date: string; equity: number }[];
  trades: TradeRecord[];
  height?: number;
}

export default function TradeMarkerChart({ equity, trades, height = 380 }: TradeMarkerChartProps) {
  const [filter, setFilter] = useState<'all' | 'win' | 'loss'>('all');
  const data = useMemo(() => buildRows(equity, trades, filter), [equity, trades, filter]);

  const CustomTooltip = ({
    active,
    payload,
  }: {
    active?: boolean;
    payload?: { dataKey: string; payload: Row }[];
  }) => {
    if (!active || !payload?.length) return null;
    const row = payload[0]!.payload;
    const dk = payload[0]!.dataKey as string;
    if (dk === 'equity') {
      return (
        <div className={tooltipBox}>
          <p className="text-text-muted">{row.date}</p>
          <p className="text-text-primary tabular-nums">权益 {row.equity.toLocaleString('zh-CN')}</p>
        </div>
      );
    }
    const side = dk === 'buy' ? 'BUY' : 'SELL';
    const t = trades.find((x) => x.date === row.date && x.side === side);
    return (
      <div className={tooltipBox}>
        <p className="text-text-muted mb-1">{row.date}</p>
        <p className="text-text-primary font-medium">{side === 'BUY' ? '买入' : '卖出'}</p>
        {t && (
          <>
            <p className="tabular-nums text-text-secondary">成交价 {t.price}</p>
            {t.pnl != null && side === 'SELL' && (
              <p className={t.pnl >= 0 ? 'text-profit tabular-nums' : 'text-loss tabular-nums'}>
                盈亏 {t.pnl >= 0 ? '+' : ''}
                {t.pnl}
              </p>
            )}
            {t.holding_period_days != null && side === 'SELL' && (
              <p className="text-text-muted">持有约 {t.holding_period_days} 个交易日</p>
            )}
          </>
        )}
      </div>
    );
  };

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="inline-flex items-center gap-1 text-xs text-text-muted">
          <Filter className="h-3.5 w-3.5" />
          交易筛选
        </span>
        {(['all', 'win', 'loss'] as const).map((k) => (
          <button
            key={k}
            type="button"
            onClick={() => setFilter(k)}
            className={`rounded-md px-2.5 py-1 text-xs font-medium transition-colors ${
              filter === k
                ? 'bg-brand/20 text-brand-light'
                : 'bg-surface-tertiary text-text-secondary hover:text-text-primary'
            }`}
          >
            {k === 'all' ? '全部' : k === 'win' ? '盈利平仓' : '亏损平仓'}
          </button>
        ))}
      </div>

      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="tradeEqFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="var(--color-profit)" stopOpacity={0.28} />
              <stop offset="95%" stopColor="var(--color-profit)" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
          <XAxis
            dataKey="date"
            tick={{ fill: 'var(--color-text-muted)', fontSize: 10 }}
            tickFormatter={(v) => String(v).slice(5)}
            minTickGap={28}
          />
          <YAxis
            tick={{ fill: 'var(--color-text-muted)', fontSize: 10 }}
            tickFormatter={(v) => `${(Number(v) / 1e6).toFixed(2)}M`}
            width={52}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Area
            type="monotone"
            dataKey="equity"
            name="权益"
            stroke="var(--color-profit)"
            fill="url(#tradeEqFill)"
            strokeWidth={1.2}
            isAnimationActive={false}
          />
          <Scatter name="买入" dataKey="buy" fill="var(--color-profit)" isAnimationActive={false} />
          <Scatter name="卖出" dataKey="sell" fill="var(--color-loss)" isAnimationActive={false} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
