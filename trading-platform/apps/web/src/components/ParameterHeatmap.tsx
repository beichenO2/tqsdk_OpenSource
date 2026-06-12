import { Fragment, useMemo, useState } from 'react';
import type { ParameterSweepResult } from '@/types';
import { SlidersHorizontal, X } from 'lucide-react';

export type HeatmapMetric = 'sharpe' | 'total_return' | 'max_drawdown';

export interface ParameterHeatmapProps {
  results: ParameterSweepResult[];
  param1AxisLabel?: string;
  param2AxisLabel?: string;
}

function metricValue(r: ParameterSweepResult, m: HeatmapMetric): number {
  if (m === 'sharpe') return r.sharpe;
  if (m === 'total_return') return r.total_return;
  return r.max_drawdown;
}

/** Lower max DD is better — invert for color scale. */
function normalizedScore(v: number, m: HeatmapMetric, minV: number, maxV: number): number {
  if (maxV <= minV) return 0.5;
  const t = (v - minV) / (maxV - minV);
  if (m === 'max_drawdown') return 1 - t;
  return t;
}

function cellColor(score: number): string {
  const r = Math.round(255 * (1 - score));
  const g = Math.round(200 * score + 40);
  const b = Math.round(80 * (1 - Math.abs(score - 0.5)));
  return `rgb(${r},${g},${b})`;
}

export default function ParameterHeatmap({
  results,
  param1AxisLabel = '参数 1',
  param2AxisLabel = '参数 2',
}: ParameterHeatmapProps) {
  const [metric, setMetric] = useState<HeatmapMetric>('sharpe');
  const [selected, setSelected] = useState<ParameterSweepResult | null>(null);

  const { xs, ys, cellMap, minV, maxV } = useMemo(() => {
    const xSet = new Set<string>();
    const ySet = new Set<string>();
    for (const r of results) {
      xSet.add(String(r.param1_value));
      ySet.add(String(r.param2_value));
    }
    const xs = Array.from(xSet).sort((a, b) => Number(a) - Number(b));
    const ys = Array.from(ySet).sort((a, b) => Number(a) - Number(b));
    const cellMap = new Map<string, ParameterSweepResult>();
    for (const r of results) {
      cellMap.set(`${String(r.param1_value)}\0${String(r.param2_value)}`, r);
    }
    const vals = results.map((r) => metricValue(r, metric));
    const minV = Math.min(...vals);
    const maxV = Math.max(...vals);
    return { xs, ys, cellMap, minV, maxV };
  }, [results, metric]);

  if (results.length === 0) {
    return (
      <div className="rounded-xl border border-border border-dashed bg-surface-secondary/50 p-8 text-center text-sm text-text-muted">
        暂无参数扫描数据
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2 text-sm text-text-secondary">
          <SlidersHorizontal className="h-4 w-4 text-text-muted" />
          <span>指标</span>
          <select
            value={metric}
            onChange={(e) => setMetric(e.target.value as HeatmapMetric)}
            className="rounded-lg border border-border bg-surface-tertiary px-2 py-1.5 text-xs text-text-primary focus:outline-none focus:ring-2 focus:ring-brand/40"
          >
            <option value="sharpe">夏普</option>
            <option value="total_return">收益率 %</option>
            <option value="max_drawdown">最大回撤 %（低更好）</option>
          </select>
        </div>
        <div className="flex items-center gap-2 text-[10px] text-text-muted">
          <span className="h-3 w-16 rounded bg-gradient-to-r from-red-600 to-emerald-500" />
          <span>差 → 好</span>
        </div>
      </div>

      <div className="overflow-x-auto rounded-xl border border-border bg-surface-secondary p-3">
        <div
          className="inline-grid gap-1"
          style={{
            gridTemplateColumns: `auto repeat(${xs.length}, minmax(3rem,1fr))`,
          }}
        >
          <div />
          {xs.map((x) => (
            <div key={x} className="px-1 text-center text-[10px] font-medium text-text-muted">
              {x}
            </div>
          ))}
          {ys.map((y) => (
            <Fragment key={y}>
              <div className="flex items-center pr-2 text-[10px] font-medium text-text-muted">
                {y}
              </div>
              {xs.map((x) => {
                const cell = cellMap.get(`${x}\0${y}`);
                if (!cell) {
                  return (
                    <div
                      key={`${x}-${y}`}
                      className="h-10 rounded-md bg-surface-tertiary/80"
                    />
                  );
                }
                const v = metricValue(cell, metric);
                const score = normalizedScore(v, metric, minV, maxV);
                return (
                  <button
                    key={cell.id}
                    type="button"
                    title={`${param1AxisLabel}=${x}, ${param2AxisLabel}=${y}`}
                    onClick={() => setSelected(cell)}
                    className="h-10 min-w-[2.75rem] rounded-md border border-border/60 text-[11px] font-semibold tabular-nums text-white shadow-sm transition-transform hover:scale-[1.03] hover:ring-2 hover:ring-brand/40 focus:outline-none"
                    style={{ backgroundColor: cellColor(score) }}
                  >
                    {metric === 'sharpe' ? v.toFixed(2) : `${v.toFixed(1)}`}
                  </button>
                );
              })}
            </Fragment>
          ))}
        </div>
      </div>

      <div className="flex justify-between text-[10px] text-text-muted px-1">
        <span>{param2AxisLabel}（行）</span>
        <span>{param1AxisLabel}（列）</span>
      </div>

      {selected && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
          role="dialog"
          aria-modal="true"
          onClick={() => setSelected(null)}
        >
          <div
            className="max-h-[90vh] w-full max-w-md overflow-y-auto rounded-xl border border-border bg-surface-secondary p-5 shadow-xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="mb-4 flex items-start justify-between gap-2">
              <h4 className="text-sm font-semibold text-text-primary">参数组合详情</h4>
              <button
                type="button"
                onClick={() => setSelected(null)}
                className="rounded-lg p-1 text-text-muted hover:bg-surface-tertiary hover:text-text-primary"
                aria-label="关闭"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
            <dl className="space-y-2 text-sm">
              <div className="flex justify-between gap-4">
                <dt className="text-text-muted">{param1AxisLabel}</dt>
                <dd className="tabular-nums text-text-primary">{String(selected.param1_value)}</dd>
              </div>
              <div className="flex justify-between gap-4">
                <dt className="text-text-muted">{param2AxisLabel}</dt>
                <dd className="tabular-nums text-text-primary">{String(selected.param2_value)}</dd>
              </div>
              <div className="flex justify-between gap-4">
                <dt className="text-text-muted">夏普</dt>
                <dd className="tabular-nums text-text-primary">{selected.sharpe.toFixed(3)}</dd>
              </div>
              <div className="flex justify-between gap-4">
                <dt className="text-text-muted">总收益 %</dt>
                <dd className="tabular-nums text-profit">{selected.total_return.toFixed(2)}</dd>
              </div>
              <div className="flex justify-between gap-4">
                <dt className="text-text-muted">最大回撤 %</dt>
                <dd className="tabular-nums text-loss">{selected.max_drawdown.toFixed(2)}</dd>
              </div>
              <div className="flex justify-between gap-4">
                <dt className="text-text-muted">ID</dt>
                <dd className="break-all text-xs text-text-secondary">{selected.id}</dd>
              </div>
            </dl>
          </div>
        </div>
      )}
    </div>
  );
}
