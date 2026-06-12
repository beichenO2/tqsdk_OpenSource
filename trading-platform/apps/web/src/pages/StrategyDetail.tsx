import { useMemo, useState } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid, RadarChart,
  PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar,
} from 'recharts';
import { ArrowLeft, ChevronLeft, ChevronRight, BarChart3, Target, Activity, AlertTriangle } from 'lucide-react';
import { useRealBacktests } from '@/hooks/useStaticData';
import { Button } from '@/components/ui/Button';
import { Tabs } from '@/components/ui/Tabs';
import Card from '@/components/Card';
import StatCard from '@/components/StatCard';
import StatusBadge from '@/components/StatusBadge';
import { fmtCny, fmtPercent, fmtCompact } from '@/lib/format';
import { cn } from '@/lib/cn';

export default function StrategyDetail() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { data: btData } = useRealBacktests();
  const [tab, setTab] = useState('metrics');

  const backtests = btData?.backtests ?? [];
  const entry = backtests.find((b) => b.id === id);

  const relatedStrategies = useMemo(() => {
    if (!entry) return [];
    return backtests
      .filter((b) => b.strategy_name === entry.strategy_name && b.id !== entry.id && !b.is_split)
      .sort((a, b) => b.sharpe_ratio - a.sharpe_ratio);
  }, [backtests, entry]);

  const allSorted = useMemo(() => {
    return [...backtests].filter((b) => !b.is_split).sort((a, b) => b.sharpe_ratio - a.sharpe_ratio);
  }, [backtests]);

  const currentIndex = allSorted.findIndex((b) => b.id === id);
  const prevEntry = currentIndex > 0 ? allSorted[currentIndex - 1] : null;
  const nextEntry = currentIndex < allSorted.length - 1 ? allSorted[currentIndex + 1] : null;

  if (!entry) {
    return (
      <div className="p-6">
        <Link to="/" className="text-sm text-brand hover:underline">&larr; 返回仪表盘</Link>
        <div className="mt-8 text-center text-text-muted">策略不存在</div>
      </div>
    );
  }

  const rank = currentIndex + 1;

  return (
    <div className="p-6 space-y-6">
      {/* Navigation header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Button variant="ghost" size="sm" onClick={() => navigate(-1)}>
            <ArrowLeft className="w-4 h-4" />
          </Button>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-semibold text-text-primary">{entry.strategy_name}</h1>
              <span className={cn('text-xs px-1.5 py-0.5 rounded', entry.market === 'crypto' ? 'bg-[#f7931a]/15 text-[#f7931a]' : 'bg-brand/15 text-brand')}>
                {entry.market === 'crypto' ? '加密' : '期货'}
              </span>
              <span className="text-xs text-text-muted font-mono">{entry.symbol}</span>
            </div>
            <p className="text-xs text-text-muted mt-0.5">
              排名 #{rank} / {allSorted.length} · {entry.date_range || '无日期范围'} · {entry.source_file}
            </p>
          </div>
        </div>

        {/* Prev/Next navigation */}
        <div className="flex items-center gap-2">
          <Button
            variant="secondary"
            size="sm"
            disabled={!prevEntry}
            onClick={() => prevEntry && navigate(`/strategy/${prevEntry.id}`)}
          >
            <ChevronLeft className="w-4 h-4" />
            上一个
          </Button>
          <span className="text-xs text-text-muted tabular-nums">{rank}/{allSorted.length}</span>
          <Button
            variant="secondary"
            size="sm"
            disabled={!nextEntry}
            onClick={() => nextEntry && navigate(`/strategy/${nextEntry.id}`)}
          >
            下一个
            <ChevronRight className="w-4 h-4" />
          </Button>
        </div>
      </div>

      {/* Key metrics row */}
      <div className="grid grid-cols-2 lg:grid-cols-6 gap-3">
        <StatCard label="总收益率" value={fmtPercent(entry.total_return)} trend={entry.total_return >= 0 ? 'up' : 'down'} />
        <StatCard label="最大回撤" value={`-${entry.max_drawdown.toFixed(2)}%`} trend="down" />
        <StatCard label="夏普比率" value={entry.sharpe_ratio.toFixed(2)} trend={entry.sharpe_ratio >= 1 ? 'up' : 'down'} />
        <StatCard label="胜率" value={`${entry.win_rate.toFixed(1)}%`} trend={entry.win_rate >= 50 ? 'up' : 'down'} />
        <StatCard label="盈亏比" value={entry.profit_factor.toFixed(2)} trend={entry.profit_factor >= 1 ? 'up' : 'down'} />
        <StatCard label="总交易" value={String(entry.total_trades)} sub={`${entry.round_trips} 往返`} />
      </div>

      {/* Tabs */}
      <Tabs
        tabs={[
          { id: 'metrics', label: '详细指标', icon: <BarChart3 className="w-3.5 h-3.5" /> },
          { id: 'trades', label: '买卖点审核', icon: <Target className="w-3.5 h-3.5" /> },
          { id: 'compare', label: '同策略对比', icon: <Activity className="w-3.5 h-3.5" /> },
        ]}
        active={tab}
        onChange={setTab}
      />

      {tab === 'metrics' && (
        <div className="space-y-4">
          {/* Radar chart — strategy quality dimensions */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card title="策略质量雷达图">
              <ResponsiveContainer width="100%" height={280}>
                <RadarChart
                  data={[
                    { dim: '收益率', value: Math.min(Math.max(entry.total_return / 50, 0), 100), fullMark: 100 },
                    { dim: '夏普', value: Math.min(Math.max(entry.sharpe_ratio / 3 * 100, 0), 100), fullMark: 100 },
                    { dim: '胜率', value: entry.win_rate, fullMark: 100 },
                    { dim: '盈亏比', value: Math.min(entry.profit_factor / 3 * 100, 100), fullMark: 100 },
                    { dim: '低回撤', value: Math.max(100 - entry.max_drawdown * 2, 0), fullMark: 100 },
                    { dim: '交易量', value: Math.min(entry.total_trades / 5, 100), fullMark: 100 },
                  ]}
                >
                  <PolarGrid stroke="var(--color-border)" />
                  <PolarAngleAxis dataKey="dim" tick={{ fill: 'var(--color-text-secondary)', fontSize: 12 }} />
                  <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                  <Radar dataKey="value" stroke="var(--color-brand)" fill="var(--color-brand)" fillOpacity={0.15} strokeWidth={2} />
                </RadarChart>
              </ResponsiveContainer>
            </Card>

            {relatedStrategies.length > 0 && (
              <Card title={`${entry.strategy_name} 跨品种收益对比`}>
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart
                    data={[entry, ...relatedStrategies].map((b) => ({
                      symbol: b.symbol.replace(/_\d+[mhd]$/, ''),
                      return: b.total_return,
                      sharpe: b.sharpe_ratio,
                    }))}
                    layout="vertical"
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--color-chart-grid)" />
                    <XAxis type="number" tick={{ fill: 'var(--color-chart-axis)', fontSize: 10 }} />
                    <YAxis type="category" dataKey="symbol" tick={{ fill: 'var(--color-text-secondary)', fontSize: 11 }} width={70} />
                    <Tooltip
                      contentStyle={{ background: 'var(--color-chart-tooltip-bg)', border: '1px solid var(--color-chart-tooltip-border)', borderRadius: 8, fontSize: 12 }}
                      formatter={(v, name) => [name === 'return' ? `${Number(v).toFixed(2)}%` : Number(v).toFixed(2), name === 'return' ? '收益率' : '夏普']}
                    />
                    <Bar dataKey="return" fill="var(--color-brand)" radius={[0, 4, 4, 0]} name="收益率" />
                  </BarChart>
                </ResponsiveContainer>
              </Card>
            )}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Card title="收益与风险">
            <div className="space-y-3 text-sm">
              {[
                { label: '初始资金', value: fmtCny(entry.initial_capital), color: '' },
                { label: '最终资金', value: fmtCny(entry.final_capital), color: entry.final_capital >= entry.initial_capital ? 'text-profit' : 'text-loss' },
                { label: '总收益率', value: fmtPercent(entry.total_return), color: entry.total_return >= 0 ? 'text-profit' : 'text-loss' },
                { label: '最大回撤', value: `-${entry.max_drawdown.toFixed(2)}%`, color: 'text-loss' },
                { label: '夏普比率', value: entry.sharpe_ratio.toFixed(3), color: entry.sharpe_ratio >= 1 ? 'text-profit' : '' },
                { label: 'Calmar 比率', value: entry.calmar_ratio?.toFixed(3) ?? '—', color: '' },
              ].map(({ label, value, color }) => (
                <div key={label} className="flex justify-between">
                  <span className="text-text-muted">{label}</span>
                  <span className={cn('tabular-nums font-medium', color || 'text-text-primary')}>{value}</span>
                </div>
              ))}
            </div>
          </Card>

          <Card title="交易统计">
            <div className="space-y-3 text-sm">
              {[
                { label: '总交易次数', value: String(entry.total_trades) },
                { label: '往返交易', value: String(entry.round_trips) },
                { label: '胜率', value: `${entry.win_rate.toFixed(1)}%` },
                { label: '盈亏比', value: entry.profit_factor >= 999 ? '∞' : entry.profit_factor.toFixed(2) },
                { label: '平均盈利', value: fmtCny(entry.avg_win) },
                { label: '平均亏损', value: fmtCny(entry.avg_loss) },
                { label: '赔率', value: entry.payoff_ratio.toFixed(2) },
                { label: '总手续费', value: fmtCny(entry.total_commission) },
                { label: '测试 K 线数', value: fmtCompact(entry.bars_tested) },
              ].map(({ label, value }) => (
                <div key={label} className="flex justify-between">
                  <span className="text-text-muted">{label}</span>
                  <span className="tabular-nums text-text-primary">{value}</span>
                </div>
              ))}
            </div>
          </Card>
        </div>
        </div>
      )}

      {tab === 'trades' && (
        <Card title="买卖点审核 · 证据链">
          <div className="flex flex-col items-center justify-center gap-4 py-16 text-center">
            <div className="rounded-full bg-warning/15 p-4">
              <AlertTriangle className="h-8 w-8 text-warning" />
            </div>
            <div>
              <h3 className="text-base font-semibold text-text-primary">暂无交易明细数据</h3>
              <p className="mt-2 text-sm text-text-muted max-w-md">
                当前回测结果文件仅包含汇总指标，不包含每笔交易的入场/离场时间和原因。
              </p>
              <p className="mt-2 text-sm text-text-muted max-w-md">
                需要修改回测引擎，在运行时将 <code className="bg-surface-tertiary px-1 py-0.5 rounded text-xs">EvidenceChain</code> 序列化到 JSON 文件。
              </p>
            </div>
            <div className="bg-surface-tertiary rounded-lg p-4 text-left text-xs text-text-secondary max-w-md w-full">
              <p className="font-medium text-text-primary mb-2">预期数据格式（每笔交易）：</p>
              <pre className="overflow-x-auto whitespace-pre font-mono text-[11px] text-text-muted">{`{
  "trade_id": "T001",
  "entry_time": "2024-03-15 14:30:00",
  "exit_time": "2024-03-16 10:15:00",
  "direction": "LONG",
  "entry_price": 4250.5,
  "exit_price": 4312.0,
  "pnl": 615.0,
  "entry_reason": "布林下轨突破 + RSI超卖 + 体制=ranging",
  "exit_reason": "ATR止盈触发 (2.5x ATR)",
  "evidence": [
    {"type": "signal", "data": {"rsi": 28.3, "bb_lower": 4240}},
    {"type": "risk_check", "data": {"position_size": 1, "margin_ok": true}}
  ]
}`}</pre>
            </div>
          </div>
        </Card>
      )}

      {tab === 'compare' && (
        <Card title={`同策略「${entry.strategy_name}」在不同品种的表现`} noPadding>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-xs text-text-muted">
                  <th className="text-left px-4 py-2 font-medium">品种</th>
                  <th className="text-left px-4 py-2 font-medium">市场</th>
                  <th className="text-right px-4 py-2 font-medium">收益率</th>
                  <th className="text-right px-4 py-2 font-medium">夏普</th>
                  <th className="text-right px-4 py-2 font-medium">最大回撤</th>
                  <th className="text-right px-4 py-2 font-medium">胜率</th>
                  <th className="text-right px-4 py-2 font-medium">交易数</th>
                  <th className="px-4 py-2 font-medium"></th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-brand/20 bg-brand/5">
                  <td className="px-4 py-2.5 font-mono text-text-primary font-medium">{entry.symbol}</td>
                  <td className="px-4 py-2.5">
                    <StatusBadge variant="info" label="当前" />
                  </td>
                  <td className={cn('px-4 py-2.5 text-right tabular-nums', entry.total_return >= 0 ? 'text-profit' : 'text-loss')}>
                    {fmtPercent(entry.total_return)}
                  </td>
                  <td className="px-4 py-2.5 text-right tabular-nums font-medium">{entry.sharpe_ratio.toFixed(2)}</td>
                  <td className="px-4 py-2.5 text-right tabular-nums text-loss">-{entry.max_drawdown.toFixed(2)}%</td>
                  <td className="px-4 py-2.5 text-right tabular-nums">{entry.win_rate.toFixed(1)}%</td>
                  <td className="px-4 py-2.5 text-right tabular-nums">{entry.total_trades}</td>
                  <td className="px-4 py-2.5"></td>
                </tr>
                {relatedStrategies.map((b) => (
                  <tr
                    key={b.id}
                    tabIndex={0}
                    role="button"
                    onClick={() => navigate(`/strategy/${b.id}`)}
                    onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && navigate(`/strategy/${b.id}`)}
                    className="border-b border-border/50 cursor-pointer hover:bg-surface-tertiary/50 focus:outline-none focus-visible:ring-1 focus-visible:ring-focus-ring"
                  >
                    <td className="px-4 py-2.5 font-mono text-text-primary">{b.symbol}</td>
                    <td className="px-4 py-2.5">
                      <span className={cn('text-xs px-1.5 py-0.5 rounded', b.market === 'crypto' ? 'bg-[#f7931a]/15 text-[#f7931a]' : 'bg-brand/15 text-brand')}>
                        {b.market === 'crypto' ? '加密' : '期货'}
                      </span>
                    </td>
                    <td className={cn('px-4 py-2.5 text-right tabular-nums', b.total_return >= 0 ? 'text-profit' : 'text-loss')}>
                      {fmtPercent(b.total_return)}
                    </td>
                    <td className="px-4 py-2.5 text-right tabular-nums font-medium">{b.sharpe_ratio.toFixed(2)}</td>
                    <td className="px-4 py-2.5 text-right tabular-nums text-loss">-{b.max_drawdown.toFixed(2)}%</td>
                    <td className="px-4 py-2.5 text-right tabular-nums">{b.win_rate.toFixed(1)}%</td>
                    <td className="px-4 py-2.5 text-right tabular-nums">{b.total_trades}</td>
                    <td className="px-4 py-2.5 text-text-muted text-xs">查看 →</td>
                  </tr>
                ))}
                {relatedStrategies.length === 0 && (
                  <tr>
                    <td colSpan={8} className="px-4 py-8 text-center text-text-muted text-sm">
                      该策略仅在当前品种有回测记录
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
}
