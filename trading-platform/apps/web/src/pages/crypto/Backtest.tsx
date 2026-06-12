import { useEffect, useState } from 'react';
import {
  AreaChart, Area, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid,
} from 'recharts';
import { Plus, Trophy, BarChart3, Bitcoin } from 'lucide-react';
import { cryptoApi } from './api';
import Card from '@/components/Card';
import StatCard from '@/components/StatCard';
import StatusBadge from '@/components/StatusBadge';
import type { CryptoBacktestResult } from './types';

function fmt(n: number, digits = 2) {
  return n.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

function fmtUsd(n: number) {
  return `$${fmt(n)}`;
}

const STRATEGY_TYPE_LABEL: Record<string, string> = {
  grid: '网格',
  momentum: '动量',
  mean_reversion: '均值回归',
  arbitrage: '套利',
};

export default function CryptoBacktest() {
  const [results, setResults] = useState<CryptoBacktestResult[]>([]);
  const [selected, setSelected] = useState<CryptoBacktestResult | null>(null);

  useEffect(() => {
    cryptoApi.getBacktestResults().then(r => {
      setResults(r);
      if (r.length > 0) setSelected(r[0]);
    });
  }, []);

  const completedResults = results.filter(r => r.status === 'COMPLETED');
  const bestSharpe = completedResults.length > 0
    ? completedResults.reduce((best, r) => r.sharpe_ratio > best.sharpe_ratio ? r : best)
    : null;

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Bitcoin className="w-6 h-6 text-[#f7931a]" />
          <h1 className="text-xl font-semibold text-text-primary">BTC 回测</h1>
        </div>
        <button className="flex items-center gap-1.5 px-4 py-2 bg-brand text-white rounded-lg text-sm font-medium hover:bg-brand-dark transition-colors">
          <Plus className="w-4 h-4" />
          新建回测
        </button>
      </div>

      {/* Overview Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          label="回测总数"
          value={String(results.length)}
          sub={`${completedResults.length} 已完成`}
          trend="neutral"
        />
        <StatCard
          label="最佳夏普"
          value={bestSharpe ? bestSharpe.sharpe_ratio.toFixed(2) : '-'}
          sub={bestSharpe?.strategy_name ?? ''}
          trend={bestSharpe && bestSharpe.sharpe_ratio >= 1.5 ? 'up' : 'down'}
        />
        <StatCard
          label="平均年化"
          value={completedResults.length > 0
            ? `${(completedResults.reduce((s, r) => s + r.annual_return, 0) / completedResults.length).toFixed(2)}%`
            : '-'}
          trend={completedResults.length > 0 && completedResults.reduce((s, r) => s + r.annual_return, 0) / completedResults.length > 0 ? 'up' : 'down'}
        />
        <StatCard
          label="平均最大回撤"
          value={completedResults.length > 0
            ? `${(completedResults.reduce((s, r) => s + r.max_drawdown, 0) / completedResults.length).toFixed(2)}%`
            : '-'}
          trend="down"
        />
      </div>

      {/* Results Table */}
      <Card
        title="回测列表"
        extra={
          <div className="flex items-center gap-1.5 text-xs text-text-muted">
            <BarChart3 className="w-3.5 h-3.5" />
            <span>点击行查看详情</span>
          </div>
        }
        noPadding
      >
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border text-text-muted text-xs">
                <th className="text-left px-4 py-2.5 font-medium">策略</th>
                <th className="text-left px-4 py-2.5 font-medium">类型</th>
                <th className="text-left px-4 py-2.5 font-medium">币对</th>
                <th className="text-left px-4 py-2.5 font-medium">区间</th>
                <th className="text-right px-4 py-2.5 font-medium">总收益</th>
                <th className="text-right px-4 py-2.5 font-medium">年化</th>
                <th className="text-right px-4 py-2.5 font-medium">最大回撤</th>
                <th className="text-right px-4 py-2.5 font-medium">夏普</th>
                <th className="text-right px-4 py-2.5 font-medium">胜率</th>
                <th className="text-left px-4 py-2.5 font-medium">状态</th>
              </tr>
            </thead>
            <tbody>
              {results.map(r => (
                <tr
                  key={r.id}
                  onClick={() => setSelected(r)}
                  className={`border-b border-border/50 cursor-pointer transition-colors ${
                    selected?.id === r.id ? 'bg-brand/10' : 'hover:bg-surface-tertiary/50'
                  }`}
                >
                  <td className="px-4 py-2.5 text-text-primary font-medium">
                    <div className="flex items-center gap-1.5">
                      {bestSharpe?.id === r.id && <Trophy className="w-3.5 h-3.5 text-warning" />}
                      {r.strategy_name}
                    </div>
                  </td>
                  <td className="px-4 py-2.5 text-text-secondary text-xs">
                    {STRATEGY_TYPE_LABEL[r.strategy_type] ?? r.strategy_type}
                  </td>
                  <td className="px-4 py-2.5 font-mono text-text-secondary text-xs">{r.symbol}</td>
                  <td className="px-4 py-2.5 text-text-secondary text-xs">{r.start_date} ~ {r.end_date}</td>
                  <td className={`px-4 py-2.5 text-right tabular-nums ${r.total_return >= 0 ? 'text-profit' : 'text-loss'}`}>
                    {r.total_return >= 0 ? '+' : ''}{r.total_return.toFixed(2)}%
                  </td>
                  <td className={`px-4 py-2.5 text-right tabular-nums ${r.annual_return >= 0 ? 'text-profit' : 'text-loss'}`}>
                    {r.annual_return >= 0 ? '+' : ''}{r.annual_return.toFixed(2)}%
                  </td>
                  <td className="px-4 py-2.5 text-right tabular-nums text-loss">
                    -{r.max_drawdown.toFixed(2)}%
                  </td>
                  <td className={`px-4 py-2.5 text-right tabular-nums ${r.sharpe_ratio >= 1.5 ? 'text-profit' : r.sharpe_ratio >= 1 ? 'text-text-primary' : 'text-loss'}`}>
                    {r.sharpe_ratio.toFixed(2)}
                  </td>
                  <td className="px-4 py-2.5 text-right tabular-nums text-text-primary">
                    {(r.win_rate * 100).toFixed(1)}%
                  </td>
                  <td className="px-4 py-2.5">
                    <StatusBadge
                      variant={r.status === 'COMPLETED' ? 'success' : r.status === 'RUNNING' ? 'info' : 'error'}
                      label={r.status === 'COMPLETED' ? '完成' : r.status === 'RUNNING' ? '运行中' : '失败'}
                      pulse={r.status === 'RUNNING'}
                    />
                  </td>
                </tr>
              ))}
              {results.length === 0 && (
                <tr>
                  <td colSpan={10} className="px-4 py-12 text-center text-text-muted text-sm">
                    暂无回测结果，点击右上角新建回测
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Detail Panel */}
      {selected && (
        <>
          <div className="grid grid-cols-2 lg:grid-cols-4 xl:grid-cols-8 gap-4">
            <StatCard label="总收益率" value={`${selected.total_return.toFixed(2)}%`} trend={selected.total_return >= 0 ? 'up' : 'down'} />
            <StatCard label="年化收益" value={`${selected.annual_return.toFixed(2)}%`} trend={selected.annual_return >= 0 ? 'up' : 'down'} />
            <StatCard label="最大回撤" value={`${selected.max_drawdown.toFixed(2)}%`} trend="down" />
            <StatCard label="夏普比率" value={selected.sharpe_ratio.toFixed(2)} trend={selected.sharpe_ratio >= 1 ? 'up' : 'down'} />
            <StatCard label="索提诺" value={selected.sortino_ratio.toFixed(2)} trend={selected.sortino_ratio >= 1.5 ? 'up' : 'down'} />
            <StatCard label="胜率" value={`${(selected.win_rate * 100).toFixed(1)}%`} trend={selected.win_rate >= 0.5 ? 'up' : 'down'} />
            <StatCard label="盈亏比" value={selected.profit_factor.toFixed(2)} trend={selected.profit_factor >= 1 ? 'up' : 'down'} />
            <StatCard label="平均持仓" value={`${selected.avg_holding_hours.toFixed(1)}h`} trend="neutral" />
          </div>

          <Card title={`资金曲线 - ${selected.strategy_name} (${selected.symbol})`}>
            <ResponsiveContainer width="100%" height={320}>
              <AreaChart data={selected.equity_curve}>
                <defs>
                  <linearGradient id="cryptoEqGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#f7931a" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#f7931a" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis
                  dataKey="date"
                  tick={{ fill: '#64748b', fontSize: 11 }}
                  tickFormatter={v => v.slice(5)}
                />
                <YAxis
                  tick={{ fill: '#64748b', fontSize: 11 }}
                  tickFormatter={v => `$${(v / 1000).toFixed(1)}K`}
                />
                <Tooltip
                  contentStyle={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8, fontSize: 12 }}
                  labelStyle={{ color: '#94a3b8' }}
                  formatter={(v) => [fmtUsd(Number(v)), '权益']}
                />
                <Area type="monotone" dataKey="equity" stroke="#f7931a" fill="url(#cryptoEqGrad)" />
              </AreaChart>
            </ResponsiveContainer>
          </Card>

          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <StatCard label="初始资金" value={fmtUsd(selected.initial_capital)} />
            <StatCard label="最终资金" value={fmtUsd(selected.final_capital)} trend={selected.final_capital > selected.initial_capital ? 'up' : 'down'} />
            <StatCard label="总交易次数" value={selected.total_trades.toLocaleString()} />
            <StatCard label="回测区间" value={`${selected.start_date} ~ ${selected.end_date}`} />
          </div>
        </>
      )}
    </div>
  );
}
