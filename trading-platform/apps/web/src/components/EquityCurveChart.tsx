import { useCallback, useMemo, useState } from 'react';
import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  ReferenceArea,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { GitCompareArrows } from 'lucide-react';
import { mergeEquityAndBenchmark, syntheticBuyHoldCurve } from '@/utils/backtestDerived';

export interface EquityCurveChartProps {
  equity: { date: string; equity: number }[];
  initialCapital: number;
  totalReturnPct: number;
  height?: number;
  showBenchmarkToggle?: boolean;
  /** When false, still renders chart but hides toggle UI (e.g. overview tab). */
  benchmarkEnabledDefault?: boolean;
}

type ChartRow = { date: string; equity: number; benchmark: number; drawdown_pct: number };

function fmtMoney(n: number) {
  return n.toLocaleString('zh-CN', { minimumFractionDigits: 0, maximumFractionDigits: 0 });
}

const tooltipBox =
  'rounded-lg border border-border bg-surface-secondary px-3 py-2 text-xs shadow-lg';

export default function EquityCurveChart({
  equity,
  initialCapital,
  totalReturnPct,
  height = 360,
  showBenchmarkToggle = true,
  benchmarkEnabledDefault = true,
}: EquityCurveChartProps) {
  const [benchmarkOn, setBenchmarkOn] = useState(benchmarkEnabledDefault);
  const [refAreaLeft, setRefAreaLeft] = useState<string | null>(null);
  const [refAreaRight, setRefAreaRight] = useState<string | null>(null);
  const [left, setLeft] = useState<string | null>(null);
  const [right, setRight] = useState<string | null>(null);

  const fullData: ChartRow[] = useMemo(() => {
    const bench = syntheticBuyHoldCurve(equity, initialCapital, totalReturnPct);
    return mergeEquityAndBenchmark(equity, bench);
  }, [equity, initialCapital, totalReturnPct]);

  const data = useMemo(() => {
    if (!left || !right) return fullData;
    const li = fullData.findIndex((d) => d.date === left);
    const ri = fullData.findIndex((d) => d.date === right);
    if (li < 0 || ri < 0) return fullData;
    const [a, b] = li <= ri ? [li, ri] : [ri, li];
    return fullData.slice(a, b + 1);
  }, [fullData, left, right]);

  const zoom = useCallback(() => {
    if (!refAreaLeft || !refAreaRight || refAreaLeft === refAreaRight) {
      setRefAreaLeft(null);
      setRefAreaRight(null);
      return;
    }
    const [l, r] = refAreaLeft <= refAreaRight ? [refAreaLeft, refAreaRight] : [refAreaRight, refAreaLeft];
    setLeft(l);
    setRight(r);
    setRefAreaLeft(null);
    setRefAreaRight(null);
  }, [refAreaLeft, refAreaRight]);

  const resetZoom = useCallback(() => {
    setLeft(null);
    setRight(null);
  }, []);

  const CustomTooltip = ({
    active,
    payload,
    label,
  }: {
    active?: boolean;
    payload?: { payload: ChartRow }[];
    label?: string;
  }) => {
    if (!active || !payload?.length) return null;
    const row = payload[0]!.payload;
    return (
      <div className={tooltipBox}>
        <p className="text-text-muted mb-1">{label}</p>
        <p className="text-text-primary tabular-nums">权益 ¥{fmtMoney(row.equity)}</p>
        <p className="text-loss tabular-nums">回撤 {row.drawdown_pct.toFixed(2)}%</p>
        {benchmarkOn && (
          <p className="text-text-secondary tabular-nums mt-0.5">
            基准 ¥{fmtMoney(row.benchmark)}
          </p>
        )}
      </div>
    );
  };

  return (
    <div className="space-y-2">
      {showBenchmarkToggle && (
        <div className="flex flex-wrap items-center justify-between gap-2">
          <button
            type="button"
            onClick={() => setBenchmarkOn((v) => !v)}
            className={`inline-flex items-center gap-2 rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors ${
              benchmarkOn
                ? 'border-brand/50 bg-brand/15 text-brand-light'
                : 'border-border bg-surface-tertiary text-text-secondary hover:text-text-primary'
            }`}
          >
            <GitCompareArrows className="h-3.5 w-3.5" />
            买入持有基准
          </button>
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-text-muted">拖拽图表区域缩放</span>
            {(left && right) || refAreaLeft ? (
              <button
                type="button"
                onClick={resetZoom}
                className="text-xs text-brand-light hover:underline"
              >
                重置缩放
              </button>
            ) : null}
          </div>
        </div>
      )}

      <ResponsiveContainer width="100%" height={height}>
        <AreaChart
          data={data}
          margin={{ top: 8, right: 12, left: 0, bottom: 0 }}
          onMouseDown={(e) => {
            if (e?.activeLabel) setRefAreaLeft(String(e.activeLabel));
          }}
          onMouseMove={(e) => {
            if (refAreaLeft && e?.activeLabel) setRefAreaRight(String(e.activeLabel));
          }}
          onMouseUp={zoom}
          onMouseLeave={() => {
            if (refAreaLeft) zoom();
          }}
        >
          <defs>
            <linearGradient id="equityFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="var(--color-profit)" stopOpacity={0.35} />
              <stop offset="95%" stopColor="var(--color-profit)" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="benchFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="var(--color-brand-light)" stopOpacity={0.2} />
              <stop offset="95%" stopColor="var(--color-brand-light)" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
          <XAxis
            dataKey="date"
            tick={{ fill: 'var(--color-text-muted)', fontSize: 10 }}
            tickFormatter={(v) => String(v).slice(5)}
            minTickGap={24}
          />
          <YAxis
            tick={{ fill: 'var(--color-text-muted)', fontSize: 10 }}
            tickFormatter={(v) => `${(Number(v) / 1e6).toFixed(2)}M`}
            width={52}
          />
          <Tooltip content={<CustomTooltip />} />
          {benchmarkOn && <Legend wrapperStyle={{ fontSize: 11 }} />}
          <Area
            type="monotone"
            dataKey="equity"
            name="策略权益"
            stroke="var(--color-profit)"
            fill="url(#equityFill)"
            strokeWidth={1.5}
            isAnimationActive={false}
          />
          {benchmarkOn && (
            <Area
              type="monotone"
              dataKey="benchmark"
              name="买入持有"
              stroke="var(--color-brand-light)"
              fill="url(#benchFill)"
              strokeWidth={1}
              strokeDasharray="4 3"
              isAnimationActive={false}
            />
          )}
          {refAreaLeft && refAreaRight ? (
            <ReferenceArea x1={refAreaLeft} x2={refAreaRight} strokeOpacity={0.4} fill="var(--color-brand)" fillOpacity={0.12} />
          ) : null}
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
