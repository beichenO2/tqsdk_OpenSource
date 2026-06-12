import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { GitCompareArrows, Plus, Trophy, BarChart3, LayoutGrid, LineChart, ArrowDownRight, Layers, Grid3x3 } from 'lucide-react';
import { api } from '@/services/api';
import Card from '@/components/Card';
import StatCard from '@/components/StatCard';
import StatusBadge from '@/components/StatusBadge';
import EquityCurveChart from '@/components/EquityCurveChart';
import TradeMarkerChart from '@/components/TradeMarkerChart';
import ParameterHeatmap from '@/components/ParameterHeatmap';
import DrawdownChart from '@/components/DrawdownChart';
import BacktestForm from '@/components/BacktestForm';
import type { BacktestResult } from '@/types';
import { deriveMockTradesFromEquity } from '@/utils/backtestDerived';
import clsx from 'clsx';

function fmt(n: number, digits = 2) {
  return n.toLocaleString('zh-CN', { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

type TabId = 'overview' | 'equity' | 'trades' | 'drawdown' | 'parameters';

const tabs: { id: TabId; label: string; icon: typeof LayoutGrid }[] = [
  { id: 'overview', label: '概览', icon: LayoutGrid },
  { id: 'equity', label: '权益', icon: LineChart },
  { id: 'trades', label: '成交', icon: BarChart3 },
  { id: 'drawdown', label: '回撤', icon: ArrowDownRight },
  { id: 'parameters', label: '参数', icon: Grid3x3 },
];

export default function Backtest() {
  const [results, setResults] = useState<BacktestResult[]>([]);
  const [selected, setSelected] = useState<BacktestResult | null>(null);
  const [tab, setTab] = useState<TabId>('overview');
  const [formOpen, setFormOpen] = useState(false);

  useEffect(() => {
    api.getBacktestResults().then((r) => {
      setResults(r);
      if (r.length > 0) setSelected(r[0]!);
    });
  }, []);

  const completedResults = results.filter((r) => r.status === 'COMPLETED');
  const bestSharpe = completedResults.length > 0
    ? completedResults.reduce((best, r) => (r.sharpe_ratio > best.sharpe_ratio ? r : best))
    : null;

  const tradesForSelected = useMemo(() => {
    if (!selected) return [];
    return selected.trades?.length ? selected.trades : deriveMockTradesFromEquity(selected);
  }, [selected]);

  function onNewResult(r: BacktestResult) {
    setResults((prev) => [r, ...prev]);
    setSelected(r);
    setFormOpen(false);
    setTab('overview');
  }

  return (
    <div className="space-y-6 p-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-xl font-semibold text-text-primary">回测研究</h1>
        <div className="flex flex-wrap items-center gap-2">
          <Link
            to="/backtest/compare"
            className="inline-flex items-center gap-1.5 rounded-lg border border-border px-3 py-2 text-sm font-medium text-text-secondary transition-colors hover:bg-surface-tertiary hover:text-text-primary"
          >
            <GitCompareArrows className="h-4 w-4" />
            对比
          </Link>
          <button
            type="button"
            onClick={() => setFormOpen(true)}
            className="inline-flex items-center gap-1.5 rounded-lg bg-brand px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-brand-dark"
          >
            <Plus className="h-4 w-4" />
            新建回测
          </button>
        </div>
      </div>

      {formOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
          role="dialog"
          aria-modal="true"
          onClick={() => setFormOpen(false)}
        >
          <div
            className="w-full max-w-lg rounded-xl border border-border bg-surface-secondary p-5 shadow-xl"
            onClick={(e) => e.stopPropagation()}
          >
            <h2 className="mb-4 text-lg font-semibold text-text-primary">新建回测</h2>
            <BacktestForm onSubmitted={onNewResult} onCancel={() => setFormOpen(false)} />
          </div>
        </div>
      )}

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
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
          trend={bestSharpe && bestSharpe.sharpe_ratio >= 1 ? 'up' : 'down'}
        />
        <StatCard
          label="平均年化"
          value={
            completedResults.length > 0
              ? `${(completedResults.reduce((s, r) => s + r.annual_return, 0) / completedResults.length).toFixed(2)}%`
              : '-'
          }
          trend={
            completedResults.length > 0 &&
            completedResults.reduce((s, r) => s + r.annual_return, 0) / completedResults.length > 0
              ? 'up'
              : 'down'
          }
        />
        <StatCard
          label="平均最大回撤"
          value={
            completedResults.length > 0
              ? `${(completedResults.reduce((s, r) => s + r.max_drawdown, 0) / completedResults.length).toFixed(2)}%`
              : '-'
          }
          trend="down"
        />
      </div>

      <Card
        title="回测列表"
        extra={
          <div className="flex items-center gap-1.5 text-xs text-text-muted">
            <BarChart3 className="h-3.5 w-3.5" />
            <span>点击行查看详情</span>
          </div>
        }
        noPadding
      >
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border text-xs text-text-muted">
                <th className="px-4 py-2.5 text-left font-medium">策略</th>
                <th className="px-4 py-2.5 text-left font-medium">区间</th>
                <th className="px-4 py-2.5 text-right font-medium">总收益</th>
                <th className="px-4 py-2.5 text-right font-medium">年化</th>
                <th className="px-4 py-2.5 text-right font-medium">最大回撤</th>
                <th className="px-4 py-2.5 text-right font-medium">夏普</th>
                <th className="px-4 py-2.5 text-right font-medium">胜率</th>
                <th className="px-4 py-2.5 text-left font-medium">状态</th>
              </tr>
            </thead>
            <tbody>
              {results.map((r) => (
                <tr
                  key={r.id}
                  onClick={() => {
                    setSelected(r);
                    setTab('overview');
                  }}
                  className={clsx(
                    'cursor-pointer border-b border-border/50 transition-colors',
                    selected?.id === r.id ? 'bg-brand/10' : 'hover:bg-surface-tertiary/50',
                  )}
                >
                  <td className="px-4 py-2.5 font-medium text-text-primary">
                    <div className="flex items-center gap-1.5">
                      {bestSharpe?.id === r.id && <Trophy className="h-3.5 w-3.5 text-warning" />}
                      {r.strategy_name}
                    </div>
                  </td>
                  <td className="px-4 py-2.5 text-xs text-text-secondary">
                    {r.start_date} ~ {r.end_date}
                  </td>
                  <td
                    className={clsx(
                      'px-4 py-2.5 text-right tabular-nums',
                      r.total_return >= 0 ? 'text-profit' : 'text-loss',
                    )}
                  >
                    {r.total_return >= 0 ? '+' : ''}
                    {r.total_return.toFixed(2)}%
                  </td>
                  <td
                    className={clsx(
                      'px-4 py-2.5 text-right tabular-nums',
                      r.annual_return >= 0 ? 'text-profit' : 'text-loss',
                    )}
                  >
                    {r.annual_return >= 0 ? '+' : ''}
                    {r.annual_return.toFixed(2)}%
                  </td>
                  <td className="px-4 py-2.5 text-right tabular-nums text-loss">-{r.max_drawdown.toFixed(2)}%</td>
                  <td
                    className={clsx(
                      'px-4 py-2.5 text-right tabular-nums',
                      r.sharpe_ratio >= 1.5 ? 'text-profit' : r.sharpe_ratio >= 1 ? 'text-text-primary' : 'text-loss',
                    )}
                  >
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
                  <td colSpan={8} className="px-4 py-12 text-center text-sm text-text-muted">
                    暂无回测结果，点击「新建回测」开始
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </Card>

      {selected && (
        <>
          <div className="flex flex-wrap gap-1 border-b border-border">
            {tabs.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                type="button"
                onClick={() => setTab(id)}
                className={clsx(
                  'inline-flex items-center gap-1.5 border-b-2 px-3 py-2 text-sm font-medium transition-colors',
                  tab === id
                    ? 'border-brand text-brand-light'
                    : 'border-transparent text-text-muted hover:text-text-secondary',
                )}
              >
                <Icon className="h-3.5 w-3.5" />
                {label}
              </button>
            ))}
          </div>

          {tab === 'overview' && (
            <div className="space-y-6">
              <div className="grid grid-cols-2 gap-4 xl:grid-cols-6 lg:grid-cols-3">
                <StatCard
                  label="总收益率"
                  value={`${selected.total_return.toFixed(2)}%`}
                  trend={selected.total_return >= 0 ? 'up' : 'down'}
                />
                <StatCard
                  label="年化收益"
                  value={`${selected.annual_return.toFixed(2)}%`}
                  trend={selected.annual_return >= 0 ? 'up' : 'down'}
                />
                <StatCard label="最大回撤" value={`${selected.max_drawdown.toFixed(2)}%`} trend="down" />
                <StatCard
                  label="夏普比率"
                  value={selected.sharpe_ratio.toFixed(2)}
                  trend={selected.sharpe_ratio >= 1 ? 'up' : 'down'}
                />
                <StatCard
                  label="胜率"
                  value={`${(selected.win_rate * 100).toFixed(1)}%`}
                  trend={selected.win_rate >= 0.5 ? 'up' : 'down'}
                />
                <StatCard
                  label="盈亏比"
                  value={selected.profit_factor.toFixed(2)}
                  trend={selected.profit_factor >= 1 ? 'up' : 'down'}
                />
              </div>

              <Card title={`资金曲线 · ${selected.strategy_name}`}>
                <EquityCurveChart
                  equity={selected.equity_curve}
                  initialCapital={selected.initial_capital}
                  totalReturnPct={selected.total_return}
                  height={280}
                  showBenchmarkToggle
                  benchmarkEnabledDefault={false}
                />
              </Card>

              <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
                <StatCard label="初始资金" value={`¥${fmt(selected.initial_capital)}`} />
                <StatCard
                  label="最终资金"
                  value={`¥${fmt(selected.final_capital)}`}
                  trend={selected.final_capital > selected.initial_capital ? 'up' : 'down'}
                />
                <StatCard label="总交易次数" value={selected.total_trades.toString()} />
                <StatCard label="回测区间" value={`${selected.start_date} ~ ${selected.end_date}`} />
              </div>
            </div>
          )}

          {tab === 'equity' && (
            <Card title={`权益曲线 · ${selected.strategy_name}`}>
              <EquityCurveChart
                equity={selected.equity_curve}
                initialCapital={selected.initial_capital}
                totalReturnPct={selected.total_return}
                height={480}
                showBenchmarkToggle
                benchmarkEnabledDefault
              />
            </Card>
          )}

          {tab === 'trades' && (
            <Card title={`成交标记 · ${selected.strategy_name}`}>
              <TradeMarkerChart equity={selected.equity_curve} trades={tradesForSelected} height={440} />
            </Card>
          )}

          {tab === 'drawdown' && (
            <Card title={`回撤（水下曲线）· ${selected.strategy_name}`}>
              <DrawdownChart equity={selected.equity_curve} height={380} />
            </Card>
          )}

          {tab === 'parameters' && (
            <Card
              title={`参数热力图 · ${selected.strategy_name}`}
              extra={
                selected.parameter_sweep?.length ? (
                  <span className="text-xs text-text-muted">{selected.parameter_sweep.length} 组</span>
                ) : null
              }
            >
              {selected.parameter_sweep?.length ? (
                <ParameterHeatmap
                  results={selected.parameter_sweep}
                  param1AxisLabel={selected.parameter_sweep[0]?.param1_label ?? '参数 1'}
                  param2AxisLabel={selected.parameter_sweep[0]?.param2_label ?? '参数 2'}
                />
              ) : (
                <div className="flex flex-col items-center justify-center gap-2 py-12 text-center text-sm text-text-muted">
                  <Layers className="h-8 w-8 opacity-40" />
                  <p>当前回测无参数扫描数据。</p>
                  <p className="text-xs">选择含参数优化的回测记录（如示例「双均线趋势跟踪」）查看热力图。</p>
                </div>
              )}
            </Card>
          )}
        </>
      )}
    </div>
  );
}
