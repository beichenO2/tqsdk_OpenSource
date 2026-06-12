import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { ArrowLeft } from 'lucide-react';
import { api } from '@/services/api';
import Card from '@/components/Card';
import type { BacktestCompareData, BacktestResult } from '@/types';

const SERIES_COLORS = [
  'var(--color-profit)',
  'var(--color-brand-light)',
  'var(--color-warning)',
  '#a78bfa',
  '#38bdf8',
  '#f472b6',
];

function buildCompareData(results: BacktestResult[]): BacktestCompareData {
  if (results.length === 0) return { results: [], aligned_equity: [] };
  const maps = results.map((r) => new Map(r.equity_curve.map((e) => [e.date, e.equity])));
  let common = new Set(maps[0]!.keys());
  for (let i = 1; i < maps.length; i++) {
    common = new Set([...common].filter((d) => maps[i]!.has(d)));
  }
  const dates = [...common].sort();
  const aligned_equity = dates.map((date) => {
    const row: { date: string; [k: string]: number | string } = { date };
    results.forEach((r, i) => {
      row[`eq_${r.id}`] = maps[i]!.get(date)!;
    });
    return row;
  });
  return { results, aligned_equity };
}

function fmt(n: number, digits = 2) {
  return n.toLocaleString('zh-CN', { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export default function BacktestCompare() {
  const [pool, setPool] = useState<BacktestResult[]>([]);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());

  useEffect(() => {
    api.getBacktestResults().then((r) => {
      const done = r.filter((x) => x.status === 'COMPLETED');
      setPool(done);
      setSelectedIds(new Set(done.slice(0, 2).map((x) => x.id)));
    });
  }, []);

  const selected = useMemo(
    () => pool.filter((r) => selectedIds.has(r.id)),
    [pool, selectedIds],
  );

  const compare = useMemo(() => buildCompareData(selected), [selected]);

  function toggle(id: string) {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  const tooltipStyle = {
    background: 'var(--color-surface-secondary)',
    border: '1px solid var(--color-border)',
    borderRadius: 8,
    fontSize: 12,
  };

  return (
    <div className="space-y-6 p-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <Link
            to="/backtest"
            className="inline-flex items-center gap-1 text-sm text-text-muted hover:text-brand-light"
          >
            <ArrowLeft className="h-4 w-4" />
            返回回测
          </Link>
          <h1 className="text-xl font-semibold text-text-primary">策略对比</h1>
        </div>
        <p className="text-xs text-text-muted">至少选择 2 个已完成回测</p>
      </div>

      <Card title="选择回测" noPadding>
        <div className="max-h-56 overflow-y-auto divide-y divide-border">
          {pool.map((r) => (
            <label
              key={r.id}
              className="flex cursor-pointer items-center gap-3 px-4 py-2.5 text-sm hover:bg-surface-tertiary/50"
            >
              <input
                type="checkbox"
                checked={selectedIds.has(r.id)}
                onChange={() => toggle(r.id)}
                className="rounded border-border text-brand focus:ring-brand/40"
              />
              <span className="font-medium text-text-primary">{r.strategy_name}</span>
              <span className="text-xs text-text-muted">
                {r.start_date} ~ {r.end_date}
              </span>
            </label>
          ))}
          {pool.length === 0 && (
            <p className="px-4 py-8 text-center text-sm text-text-muted">没有可对比的已完成回测</p>
          )}
        </div>
      </Card>

      {selected.length >= 2 && (
        <>
          <Card title="指标对比">
            <div className="overflow-x-auto">
              <table className="w-full min-w-[640px] text-sm">
                <thead>
                  <tr className="border-b border-border text-left text-xs text-text-muted">
                    <th className="px-3 py-2 font-medium">策略</th>
                    {selected.map((r) => (
                      <th key={r.id} className="px-3 py-2 font-medium">
                        {r.strategy_name}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="tabular-nums">
                  {(
                    [
                      { label: '总收益 %', value: (r: BacktestResult) => `${r.total_return.toFixed(2)}` },
                      { label: '年化 %', value: (r: BacktestResult) => `${r.annual_return.toFixed(2)}` },
                      { label: '最大回撤 %', value: (r: BacktestResult) => `-${r.max_drawdown.toFixed(2)}` },
                      { label: '夏普', value: (r: BacktestResult) => r.sharpe_ratio.toFixed(2) },
                      { label: '胜率', value: (r: BacktestResult) => `${(r.win_rate * 100).toFixed(1)}%` },
                      { label: '盈亏比', value: (r: BacktestResult) => r.profit_factor.toFixed(2) },
                      { label: '交易次数', value: (r: BacktestResult) => String(r.total_trades) },
                      { label: '最终权益', value: (r: BacktestResult) => `¥${fmt(r.final_capital)}` },
                    ] as const
                  ).map(({ label, value }) => (
                    <tr key={label} className="border-b border-border/60">
                      <td className="px-3 py-2 text-text-secondary">{label}</td>
                      {selected.map((r) => (
                        <td key={r.id} className="px-3 py-2 text-text-primary">
                          {value(r)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="权益曲线叠加">
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={compare.aligned_equity} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                <XAxis
                  dataKey="date"
                  tick={{ fill: 'var(--color-text-muted)', fontSize: 10 }}
                  tickFormatter={(v) => String(v).slice(5)}
                  minTickGap={20}
                />
                <YAxis
                  tick={{ fill: 'var(--color-text-muted)', fontSize: 10 }}
                  tickFormatter={(v) => `${(Number(v) / 1e6).toFixed(2)}M`}
                  width={52}
                />
                <Tooltip contentStyle={tooltipStyle} labelStyle={{ color: 'var(--color-text-muted)' }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                {selected.map((r, i) => (
                  <Line
                    key={r.id}
                    type="monotone"
                    dataKey={`eq_${r.id}`}
                    name={r.strategy_name}
                    stroke={SERIES_COLORS[i % SERIES_COLORS.length]}
                    dot={false}
                    strokeWidth={1.8}
                    isAnimationActive={false}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </>
      )}

      {selected.length < 2 && pool.length > 0 && (
        <p className="text-center text-sm text-text-muted">请勾选至少两个回测以查看对比表与叠加曲线。</p>
      )}
    </div>
  );
}
