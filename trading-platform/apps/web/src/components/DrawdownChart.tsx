import { useMemo } from 'react';
import {
  Area,
  AreaChart,
  CartesianGrid,
  Label,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type { DrawdownData } from '@/types';
import { equityToDrawdown, maxDrawdownIndex, recoveryAnnotations } from '@/utils/backtestDerived';

const tooltipBox =
  'rounded-lg border border-border bg-surface-secondary px-3 py-2 text-xs shadow-lg';

export interface DrawdownChartProps {
  equity: { date: string; equity: number }[];
  height?: number;
}

export default function DrawdownChart({ equity, height = 320 }: DrawdownChartProps) {
  const dd: DrawdownData[] = useMemo(() => equityToDrawdown(equity), [equity]);
  const maxIdx = useMemo(() => maxDrawdownIndex(dd), [dd]);
  const maxRow = maxIdx >= 0 ? dd[maxIdx] : null;
  const recoveries = useMemo(() => recoveryAnnotations(dd).slice(0, 3), [dd]);

  const CustomTooltip = ({
    active,
    payload,
  }: {
    active?: boolean;
    payload?: { payload: DrawdownData }[];
  }) => {
    if (!active || !payload?.length) return null;
    const row = payload[0]!.payload;
    return (
      <div className={tooltipBox}>
        <p className="text-text-muted">{row.date}</p>
        <p className="text-loss tabular-nums font-medium">{row.drawdown_pct.toFixed(2)}%</p>
      </div>
    );
  };

  return (
    <div className="space-y-2">
      {maxRow && (
        <p className="text-xs text-text-muted">
          最大回撤 <span className="text-loss font-semibold tabular-nums">{maxRow.drawdown_pct.toFixed(2)}%</span>
          {' · '}
          <span className="text-text-secondary">{maxRow.date}</span>
        </p>
      )}
      <ResponsiveContainer width="100%" height={height}>
        <AreaChart data={dd} margin={{ top: 16, right: 12, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="ddFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--color-loss)" stopOpacity={0.45} />
              <stop offset="100%" stopColor="var(--color-loss)" stopOpacity={0.05} />
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
            tickFormatter={(v) => `${Number(v).toFixed(1)}%`}
            domain={['dataMin', 0.5]}
            width={48}
          />
          <Tooltip content={<CustomTooltip />} />
          {maxRow ? (
            <ReferenceLine
              x={maxRow.date}
              stroke="var(--color-warning)"
              strokeDasharray="4 3"
              strokeWidth={1.5}
            >
              <Label
                value="最大回撤"
                position="top"
                fill="var(--color-text-muted)"
                fontSize={10}
              />
            </ReferenceLine>
          ) : null}
          {maxRow ? (
            <ReferenceLine
              y={maxRow.drawdown_pct}
              stroke="var(--color-loss)"
              strokeOpacity={0.6}
              strokeDasharray="6 4"
            />
          ) : null}
          {recoveries.map((r) => (
            <ReferenceLine
              key={r.date}
              x={r.date}
              stroke="var(--color-profit)"
              strokeOpacity={0.35}
              strokeDasharray="2 4"
            >
              <Label value={r.label} position="insideTopRight" fill="var(--color-profit)" fontSize={9} />
            </ReferenceLine>
          ))}
          <Area
            type="monotone"
            dataKey="drawdown_pct"
            name="回撤 %"
            stroke="var(--color-loss)"
            fill="url(#ddFill)"
            strokeWidth={1.2}
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
