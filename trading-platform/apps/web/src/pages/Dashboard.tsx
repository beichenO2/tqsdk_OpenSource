import { useMemo } from 'react';
import {
  AreaChart, Area, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid,
} from 'recharts';
import { useNavigate } from 'react-router-dom';
import { RefreshCw, XCircle, PauseCircle, Trophy, TrendingUp, BarChart3, Radio, Play, Square } from 'lucide-react';
import { useQueryClient } from '@tanstack/react-query';
import { useAccount } from '@/hooks/useAccount';
import { usePositions, useCloseAllPositions } from '@/hooks/usePositions';
import { useStrategies, usePauseAllStrategies } from '@/hooks/useStrategies';
import { useRiskAlerts, usePnlHistory } from '@/hooks/useMarketData';
import { useRealBacktests, useOptimizerRounds } from '@/hooks/useStaticData';
import { useLiveTradingStatus, useStartLiveTrading, useStopLiveTrading, useLiveEvents } from '@/hooks/useLiveTrading';
import { useConfirm } from '@/components/ui/ConfirmDialog';
import { useToast } from '@/components/ui/Toast';
import { Button } from '@/components/ui/Button';
import { StatCardSkeleton, ChartSkeleton } from '@/components/ui/Skeleton';
import StatCard from '@/components/StatCard';
import Card from '@/components/Card';
import StatusBadge from '@/components/StatusBadge';
import MarketWatch from '@/components/MarketWatch';
import { fmtCny, fmtPercent } from '@/lib/format';
import { cn } from '@/lib/cn';

export default function Dashboard() {
  const qc = useQueryClient();
  const confirm = useConfirm();
  const toast = useToast();
  const navigate = useNavigate();

  const { data: account, isError: acctError } = useAccount();
  const { data: positions = [] } = usePositions();
  const { data: strategies = [] } = useStrategies();
  const { data: alerts = [] } = useRiskAlerts();
  const { data: pnlData = [] } = usePnlHistory(30);
  const { data: btData, isLoading: btLoading } = useRealBacktests();
  const { data: optRounds = [] } = useOptimizerRounds();

  const closeAll = useCloseAllPositions();
  const pauseAll = usePauseAllStrategies();
  const { data: liveStatus } = useLiveTradingStatus() as { data: Record<string, unknown> | undefined };
  const startLive = useStartLiveTrading();
  const stopLive = useStopLiveTrading();
  const { events: liveEvents, connected: wsConnected } = useLiveEvents();
  const liveRunning = !!(liveStatus as Record<string, unknown>)?.running;
  const liveMode = String((liveStatus as Record<string, unknown>)?.mode || 'paper');

  const backtests = btData?.backtests ?? [];

  const stats = useMemo(() => {
    if (!backtests.length) return null;
    const valid = backtests.filter((b) => !b.is_split && b.total_trades > 0 && b.sharpe_ratio > -5);
    const crypto = valid.filter((b) => b.market === 'crypto');
    const futures = valid.filter((b) => b.market === 'futures');
    const best = [...valid].sort((a, b) => b.sharpe_ratio - a.sharpe_ratio).slice(0, 5);
    const profitable = valid.filter((b) => b.sharpe_ratio > 0);
    const medianSharpe = (() => {
      const sorted = [...valid].map((b) => b.sharpe_ratio).sort((a, b) => a - b);
      const mid = Math.floor(sorted.length / 2);
      return sorted.length % 2 === 0
        ? (sorted[mid - 1]! + sorted[mid]!) / 2
        : sorted[mid]!;
    })();
    return {
      crypto: crypto.length,
      futures: futures.length,
      total: valid.length,
      totalRaw: backtests.length,
      filtered: backtests.length - valid.length,
      best,
      medianSharpe,
      profitableCount: profitable.length,
    };
  }, [backtests]);

  const optimizerChart = useMemo(() => {
    return optRounds
      .filter((r) => r.best_score != null)
      .map((r) => ({ round: r.round, score: r.best_score }));
  }, [optRounds]);

  // 闭市时 pnl-history 返回上一交易区间的静止权益快照，曲线自然冻结
  const lastSnapshotLabel = useMemo(() => {
    if (!pnlData.length) return '';
    const last = pnlData[pnlData.length - 1] as { date?: string };
    return last?.date ? `截至 ${last.date}` : '';
  }, [pnlData]);

  const handleCloseAll = async () => {
    const ok = await confirm({
      title: '确认一键平仓',
      description: '这将关闭所有持仓，该操作不可撤回。',
      variant: 'destructive',
      confirmText: '确认平仓',
    });
    if (!ok) return;
    try { await closeAll.mutateAsync(); toast.success('已提交平仓指令'); }
    catch { toast.error('平仓失败'); }
  };

  const handlePauseAll = async () => {
    const ok = await confirm({
      title: '确认暂停所有策略',
      description: '所有运行中的策略将被暂停。',
      variant: 'warning',
      confirmText: '暂停全部',
    });
    if (!ok) return;
    try { await pauseAll.mutateAsync(); toast.success('所有策略已暂停'); }
    catch { toast.error('暂停失败'); }
  };

  const runningStrategies = strategies.filter((s) => s.status === 'RUNNING').length;
  const activeAlerts = alerts.filter((a) => !a.resolved);

  return (
    <div className="px-3 py-3 space-y-3 max-w-[110rem] mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="font-mono text-base font-semibold tracking-widest text-text-primary uppercase">
          Overview<span className="text-brand">_</span>
        </h1>
        <div className="flex items-center gap-2">
          <Button variant="destructive" size="sm" onClick={handleCloseAll} loading={closeAll.isPending}>
            <XCircle className="w-3.5 h-3.5" />
            一键平仓
          </Button>
          <Button variant="secondary" size="sm" onClick={handlePauseAll} loading={pauseAll.isPending}>
            <PauseCircle className="w-3.5 h-3.5" />
            暂停策略
          </Button>
          <div className="h-5 w-px bg-border" />
          <Button variant="ghost" size="sm" onClick={() => { qc.invalidateQueries(); toast.info('已刷新'); }}>
            <RefreshCw className="w-4 h-4" />
          </Button>
          <div className="flex items-center gap-1.5">
            <span className={`led ${acctError ? 'text-loss bg-loss' : 'text-profit bg-profit'}`} />
            <span className="text-[11px] font-mono text-text-muted">{acctError ? 'OFFLINE' : 'ONLINE'}</span>
          </div>
        </div>
      </div>

      {/* Live Trading Status Bar */}
      <div className={cn(
        'flex items-center justify-between rounded-xl border px-5 py-3',
        liveRunning
          ? liveMode === 'live' ? 'bg-loss/5 border-loss/20' : 'bg-profit/5 border-profit/20'
          : 'bg-surface-secondary border-border',
      )}>
        <div className="flex items-center gap-3">
          <Radio className={cn('w-4 h-4', liveRunning ? (liveMode === 'live' ? 'text-loss' : 'text-profit') : 'text-text-muted')} />
          <span className="text-sm font-medium text-text-primary">
            {liveRunning
              ? `${liveMode === 'live' ? '实盘' : '模拟'}交易运行中`
              : '交易未启动'}
          </span>
          {liveRunning && (
            <>
              <span className="text-xs text-text-muted">
                {Number((liveStatus as Record<string, unknown>)?.strategy_count || 0)} 策略 ·
                {' '}{Number((liveStatus as Record<string, unknown>)?.bar_count || 0)} bars
              </span>
              <div className="flex items-center gap-1">
                <span className={cn('w-1.5 h-1.5 rounded-full', wsConnected ? 'bg-profit animate-pulse' : 'bg-loss')} />
                <span className="text-[10px] text-text-muted">WS</span>
              </div>
            </>
          )}
        </div>
        <div className="flex items-center gap-2">
          {liveRunning ? (
            <Button variant="destructive" size="sm" onClick={async () => {
              const ok = await confirm({ title: '确认停止', description: '停止所有策略运行', variant: 'destructive', confirmText: '停止' });
              if (ok) { try { await stopLive.mutateAsync(); toast.success('已停止'); } catch { toast.error('停止失败'); } }
            }} loading={stopLive.isPending}>
              <Square className="w-3 h-3" />
              停止
            </Button>
          ) : (
            <Button variant="primary" size="sm" onClick={async () => {
              try { await startLive.mutateAsync({ mode: 'paper', market: 'crypto' }); toast.success('模拟交易已启动'); }
              catch (e) { toast.error(e instanceof Error ? e.message : '启动失败'); }
            }} loading={startLive.isPending}>
              <Play className="w-3 h-3" />
              快速启动模拟
            </Button>
          )}
          <Button variant="ghost" size="sm" onClick={() => navigate('/live-trading')}>
            详情 →
          </Button>
        </div>
      </div>

      {/* Real Data KPI Row */}
      {btLoading ? (
        <div className="grid grid-cols-2 lg:grid-cols-5 gap-3">
          {Array.from({ length: 5 }).map((_, i) => <StatCardSkeleton key={i} />)}
        </div>
      ) : (
        <div className="grid grid-cols-2 lg:grid-cols-5 gap-3">
          <StatCard label="有效回测" value={String(stats?.total ?? 0)} sub={`${stats?.crypto ?? 0} 加密 + ${stats?.futures ?? 0} 期货${stats?.filtered ? ` (${stats.filtered} 无效已过滤)` : ''}`} trend="neutral" />
          <StatCard
            label="最佳夏普"
            value={stats?.best?.[0]?.sharpe_ratio?.toFixed(2) ?? '—'}
            sub={stats?.best?.[0]?.strategy_name ?? ''}
            trend={(stats?.best?.[0]?.sharpe_ratio ?? 0) >= 1 ? 'up' : 'down'}
          />
          <StatCard
            label="盈利策略"
            value={`${stats?.profitableCount ?? 0} / ${stats?.total ?? 0}`}
            sub={`中位夏普 ${stats?.medianSharpe?.toFixed(2) ?? '—'}`}
            trend={(stats?.profitableCount ?? 0) > (stats?.total ?? 1) / 2 ? 'up' : 'down'}
          />
          <StatCard
            label="优化器轮次"
            value={String(optRounds.length > 0 ? optRounds[optRounds.length - 1]!.round : 0)}
            sub={`${optRounds.length} 轮已采样`}
            trend="neutral"
          />
          <StatCard
            label="运行策略"
            value={`${runningStrategies} / ${strategies.length}`}
            sub={activeAlerts.length > 0 ? `${activeAlerts.length} 条告警` : '无告警'}
            trend={activeAlerts.some((a) => a.level === 'CRITICAL') ? 'down' : 'up'}
          />
        </div>
      )}

      {/* Terminal 3-column: MarketWatch | Equity+Optimizer | Account+Risk */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-3">
        {/* Left: market watchlist with tick flash */}
        <Card title="行情看板" className="xl:col-span-3 min-h-[300px]" noPadding
          extra={<span className="text-[10px] font-mono text-text-muted">3s poll</span>}>
          <div className="max-h-[430px] overflow-y-auto">
            <MarketWatch maxRows={16} />
          </div>
        </Card>

        {/* Center: equity + optimizer stacked */}
        <div className="xl:col-span-6 flex flex-col gap-3 min-w-0">
          <Card
            title="实盘权益曲线"
            extra={
              lastSnapshotLabel ? (
                <span className="text-[10px] font-mono text-text-muted">{lastSnapshotLabel}</span>
              ) : undefined
            }
          >
            {pnlData.length > 0 ? (
              <div className="h-[190px] min-h-[190px]">
                <ResponsiveContainer width="100%" height="100%" minWidth={200} minHeight={160}>
                  <AreaChart data={pnlData}>
                    <defs>
                      <linearGradient id="pnlGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="var(--color-chart-brand)" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="var(--color-chart-brand)" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--color-chart-grid)" />
                    <XAxis dataKey="date" tick={{ fill: 'var(--color-chart-axis)', fontSize: 10 }} />
                    <YAxis
                      domain={['auto', 'auto']}
                      tick={{ fill: 'var(--color-chart-axis)', fontSize: 10 }}
                      tickFormatter={(v) => `${(v / 1000).toFixed(0)}k`}
                    />
                    <Tooltip
                      contentStyle={{ background: 'var(--color-chart-tooltip-bg)', border: '1px solid var(--color-chart-tooltip-border)', borderRadius: 4, fontSize: 12 }}
                      formatter={(v) => [fmtCny(Number(v)), '权益']}
                    />
                    <Area type="monotone" dataKey="pnl" stroke="var(--color-chart-brand)" fill="url(#pnlGrad)" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <div className="flex items-center justify-center h-[190px] text-sm text-text-muted">
                <div className="text-center">
                  <BarChart3 className="w-8 h-8 mx-auto mb-2 opacity-30" />
                  <p>暂无权益快照 — 开市后每分钟自动记录，闭市显示上一交易区间</p>
                </div>
              </div>
            )}
          </Card>

          <Card title="永续优化器收敛曲线" extra={<span className="text-[10px] font-mono text-text-muted">{optRounds.length > 0 ? `${optRounds[optRounds.length - 1]!.round}+ RND` : ''}</span>}>
            {optimizerChart.length > 0 ? (
              <div className="h-[170px] min-h-[170px]">
                <ResponsiveContainer width="100%" height="100%" minWidth={200} minHeight={140}>
                  <AreaChart data={optimizerChart}>
                    <defs>
                      <linearGradient id="optGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="var(--color-chart-brand)" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="var(--color-chart-brand)" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--color-chart-grid)" />
                    <XAxis dataKey="round" tick={{ fill: 'var(--color-chart-axis)', fontSize: 10 }} />
                    <YAxis tick={{ fill: 'var(--color-chart-axis)', fontSize: 10 }} />
                    <Tooltip
                      contentStyle={{ background: 'var(--color-chart-tooltip-bg)', border: '1px solid var(--color-chart-tooltip-border)', borderRadius: 4, fontSize: 12 }}
                      formatter={(v) => [Number(v).toFixed(3), 'Best Score']}
                      labelFormatter={(l) => `Round ${l}`}
                    />
                    <Area type="monotone" dataKey="score" stroke="var(--color-chart-brand)" fill="url(#optGrad)" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <ChartSkeleton height={170} />
            )}
          </Card>
        </div>

        {/* Right: account + risk */}
        <div className="xl:col-span-3 flex flex-col gap-3 min-w-0">
          <Card title="账户概览">
            {account ? (
              <div className="space-y-2.5 text-[13px]">
                <div className="flex justify-between">
                  <span className="text-text-muted">权益</span>
                  <span className="text-text-primary tabular-nums font-medium">{fmtCny(account.balance)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-text-muted">可用</span>
                  <span className="text-text-primary tabular-nums">{fmtCny(account.available)}</span>
                </div>
                <div className="h-px bg-border/50" />
                <div className="flex justify-between">
                  <span className="text-text-muted">浮盈</span>
                  <span className={cn('tabular-nums font-medium', account.float_profit >= 0 ? 'text-profit' : 'text-loss')}>
                    {fmtCny(account.float_profit)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-text-muted">风险度</span>
                  <span className={cn('tabular-nums', account.risk_ratio > 0.5 ? 'text-loss' : 'text-profit')}>
                    {(account.risk_ratio * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-text-muted">持仓</span>
                  <span className="text-text-primary tabular-nums">{positions.length} 品种</span>
                </div>
              </div>
            ) : (
              <div className="flex items-center justify-center py-8 text-sm text-text-muted">
                <div className="text-center">
                  <TrendingUp className="w-6 h-6 mx-auto mb-2 opacity-30" />
                  <p>启动后端 API 后显示</p>
                </div>
              </div>
            )}
          </Card>

          <Card title="风控提醒" extra={
            activeAlerts.length > 0
              ? <StatusBadge variant={activeAlerts.some((a) => a.level === 'CRITICAL') ? 'error' : 'warning'} label={`${activeAlerts.length}`} />
              : <StatusBadge variant="success" label="正常" />
          } noPadding>
            <div className="divide-y divide-border/50 max-h-[220px] overflow-y-auto">
              {activeAlerts.length === 0 && (
                <div className="px-4 py-6 text-center text-text-muted text-sm">暂无风控告警</div>
              )}
              {alerts.filter((a) => !a.resolved).map((a) => (
                <div key={a.id} className="flex items-start gap-2.5 px-3 py-2.5">
                  <StatusBadge variant={a.level === 'CRITICAL' ? 'error' : a.level === 'WARNING' ? 'warning' : 'info'} label={a.level} />
                  <div className="min-w-0">
                    <p className="text-[12.5px] text-text-primary truncate">{a.message}</p>
                    <p className="text-[10.5px] font-mono text-text-muted mt-0.5">{a.created_at.replace('T', ' ')}</p>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>
      </div>

      {/* Top 5 Strategies + Positions */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        <Card title="Top 5 策略（按夏普排序）" extra={<Trophy className="w-4 h-4 text-warning" />} noPadding>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-xs text-text-muted">
                  <th className="text-left px-4 py-2 font-medium">#</th>
                  <th className="text-left px-4 py-2 font-medium">策略</th>
                  <th className="text-left px-4 py-2 font-medium">品种</th>
                  <th className="text-right px-4 py-2 font-medium">收益率</th>
                  <th className="text-right px-4 py-2 font-medium">夏普</th>
                  <th className="text-right px-4 py-2 font-medium">最大回撤</th>
                  <th className="text-right px-4 py-2 font-medium">胜率</th>
                </tr>
              </thead>
              <tbody>
                {(stats?.best ?? []).map((b, i) => (
                  <tr
                    key={b.id}
                    tabIndex={0}
                    role="button"
                    onClick={() => navigate(`/strategy/${b.id}`)}
                    onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && navigate(`/strategy/${b.id}`)}
                    className="border-b border-border/50 hover:bg-surface-tertiary/50 cursor-pointer focus:outline-none focus-visible:ring-1 focus-visible:ring-focus-ring"
                  >
                    <td className="px-4 py-2.5 text-text-muted">{i + 1}</td>
                    <td className="px-4 py-2.5 text-text-primary font-medium">{b.strategy_name}</td>
                    <td className="px-4 py-2.5">
                      <div className="flex items-center gap-1.5">
                        <span className={cn('text-xs px-1.5 py-0.5 rounded', b.market === 'crypto' ? 'bg-[#f7931a]/15 text-[#f7931a]' : 'bg-brand/15 text-brand')}>
                          {b.market === 'crypto' ? '加密' : '期货'}
                        </span>
                        <span className="text-xs text-text-muted font-mono">{b.symbol}</span>
                      </div>
                    </td>
                    <td className={cn('px-4 py-2.5 text-right tabular-nums', b.total_return >= 0 ? 'text-profit' : 'text-loss')}>
                      {fmtPercent(b.total_return)}
                    </td>
                    <td className={cn('px-4 py-2.5 text-right tabular-nums font-medium', b.sharpe_ratio >= 1.5 ? 'text-profit' : b.sharpe_ratio >= 1 ? 'text-text-primary' : 'text-loss')}>
                      {b.sharpe_ratio.toFixed(2)}
                    </td>
                    <td className="px-4 py-2.5 text-right tabular-nums text-loss">-{b.max_drawdown.toFixed(2)}%</td>
                    <td className="px-4 py-2.5 text-right tabular-nums text-text-primary">{b.win_rate.toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <Card title="持仓概览" extra={<span className="text-xs text-text-muted tabular-nums">{positions.length} 个品种</span>} noPadding>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-text-muted text-xs">
                  <th className="text-left px-4 py-2.5 font-medium">合约</th>
                  <th className="text-left px-4 py-2.5 font-medium">方向</th>
                  <th className="text-right px-4 py-2.5 font-medium">数量</th>
                  <th className="text-right px-4 py-2.5 font-medium">浮盈</th>
                  <th className="text-right px-4 py-2.5 font-medium">保证金</th>
                </tr>
              </thead>
              <tbody>
                {positions.map((p) => (
                  <tr key={`${p.instrument_id}-${p.direction}`} className="border-b border-border/50 hover:bg-surface-tertiary/50">
                    <td className="px-4 py-2.5 font-mono text-text-primary">{p.instrument_id}</td>
                    <td className="px-4 py-2.5">
                      <StatusBadge variant={p.direction === 'LONG' ? 'success' : 'error'} label={p.direction === 'LONG' ? '多' : '空'} />
                    </td>
                    <td className="px-4 py-2.5 text-right tabular-nums">{p.volume}</td>
                    <td className={cn('px-4 py-2.5 text-right tabular-nums font-medium', p.float_profit >= 0 ? 'text-profit' : 'text-loss')}>
                      {fmtCny(p.float_profit)}
                    </td>
                    <td className="px-4 py-2.5 text-right tabular-nums text-text-secondary">{fmtCny(p.margin)}</td>
                  </tr>
                ))}
                {positions.length === 0 && (
                  <tr><td colSpan={5} className="px-4 py-8 text-center text-text-muted text-sm">暂无持仓 — 启动后端后显示</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    </div>
  );
}
