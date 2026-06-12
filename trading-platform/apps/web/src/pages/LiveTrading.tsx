import { useState, useMemo } from 'react';
import {
  Play, Square, RefreshCw, Zap, Shield,
  ArrowUpRight, ArrowDownRight, Activity, Radio,
} from 'lucide-react';
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid,
} from 'recharts';
import {
  useLiveTradingStatus, useLiveStrategies, useLiveLeaderboard,
  useStartLiveTrading, useStopLiveTrading, useSwitchMode,
  useToggleStrategy, useLiveEvents,
} from '@/hooks/useLiveTrading';
import { useConfirm } from '@/components/ui/ConfirmDialog';
import { useToast } from '@/components/ui/Toast';
import { Button } from '@/components/ui/Button';
import { Tabs } from '@/components/ui/Tabs';
import Card from '@/components/Card';
import StatusBadge from '@/components/StatusBadge';
import { cn } from '@/lib/cn';

function ReturnBadge({ value }: { value: number }) {
  const positive = value >= 0;
  return (
    <span className={cn('inline-flex items-center gap-0.5 text-sm font-medium tabular-nums', positive ? 'text-profit' : 'text-loss')}>
      {positive ? <ArrowUpRight className="w-3.5 h-3.5" /> : <ArrowDownRight className="w-3.5 h-3.5" />}
      {value >= 0 ? '+' : ''}{value.toFixed(2)}%
    </span>
  );
}

export default function LiveTrading() {
  const confirm = useConfirm();
  const toast = useToast();

  const { data: status } = useLiveTradingStatus() as { data: Record<string, unknown> | undefined };
  const { data: strategies = [] } = useLiveStrategies() as { data: Record<string, unknown>[] };
  const { data: leaderboard = [] } = useLiveLeaderboard() as { data: Record<string, unknown>[] };
  const startTrading = useStartLiveTrading();
  const stopTrading = useStopLiveTrading();
  const switchMode = useSwitchMode();
  const toggleStrategy = useToggleStrategy();
  const { events, connected } = useLiveEvents();

  const [activeTab, setActiveTab] = useState('strategies');
  const [startMode, setStartMode] = useState<'paper' | 'live'>('paper');
  const [startMarket, setStartMarket] = useState<'crypto' | 'futures' | 'both'>('crypto');

  const isRunning = !!(status as Record<string, unknown>)?.running;
  const currentMode = String((status as Record<string, unknown>)?.mode || 'paper');
  const barCount = Number((status as Record<string, unknown>)?.bar_count || 0);
  const strategyCount = Number((status as Record<string, unknown>)?.strategy_count || 0);
  const summary = (status as Record<string, unknown>)?.accounts_summary as Record<string, Record<string, number>> | undefined;

  const recentFills = useMemo(
    () => events.filter((e) => e.type === 'trade_fill').slice(-20),
    [events],
  );

  const handleStart = async () => {
    if (startMode === 'live') {
      const ok = await confirm({
        title: '确认启动实盘交易',
        description: '实盘模式会向真实交易所发送订单，请确保已配置正确的凭证和风控参数。',
        variant: 'destructive',
        confirmText: '确认启动实盘',
      });
      if (!ok) return;
    }
    try {
      await startTrading.mutateAsync({ mode: startMode, market: startMarket });
      toast.success(`${startMode === 'live' ? '实盘' : '模拟'}交易已启动`);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : '启动失败');
    }
  };

  const handleStop = async () => {
    const ok = await confirm({
      title: '确认停止交易',
      description: '这将停止所有运行中的策略。',
      variant: 'warning',
      confirmText: '确认停止',
    });
    if (!ok) return;
    try {
      await stopTrading.mutateAsync();
      toast.success('交易已停止');
    } catch (e) {
      toast.error(e instanceof Error ? e.message : '停止失败');
    }
  };

  const handleSwitchMode = async () => {
    const targetMode = currentMode === 'paper' ? 'live' : 'paper';
    if (targetMode === 'live') {
      const ok = await confirm({
        title: '切换到实盘模式',
        description: '切换后，策略信号将发送到真实交易所。请确认！',
        variant: 'destructive',
        confirmText: '确认切换到实盘',
      });
      if (!ok) return;
    }
    try {
      await switchMode.mutateAsync(targetMode);
      toast.success(`已切换到${targetMode === 'live' ? '实盘' : '模拟'}模式`);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : '切换失败');
    }
  };

  return (
    <div className="px-[3%] py-[2%] space-y-[clamp(1rem,2vw,1.5rem)] max-w-[96rem] mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h1 className="text-xl font-semibold text-text-primary">实盘交易</h1>
          <div className="flex items-center gap-1.5">
            <span className={cn('w-2 h-2 rounded-full', connected ? 'bg-profit animate-pulse' : 'bg-loss')} />
            <span className="text-xs text-text-muted">{connected ? 'WebSocket 已连接' : '未连接'}</span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {isRunning && (
            <>
              <StatusBadge
                variant={currentMode === 'live' ? 'error' : 'success'}
                label={currentMode === 'live' ? '实盘' : '模拟'}
              />
              <Button variant="secondary" size="sm" onClick={handleSwitchMode} loading={switchMode.isPending}>
                <RefreshCw className="w-3.5 h-3.5" />
                切换{currentMode === 'paper' ? '实盘' : '模拟'}
              </Button>
              <Button variant="destructive" size="sm" onClick={handleStop} loading={stopTrading.isPending}>
                <Square className="w-3.5 h-3.5" />
                停止
              </Button>
            </>
          )}
        </div>
      </div>

      {/* Control Panel - when not running */}
      {!isRunning && (
        <Card title="启动交易" extra={<Shield className="w-4 h-4 text-text-muted" />}>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div>
              <label className="block text-sm text-text-secondary mb-2">交易模式</label>
              <div className="flex gap-2">
                {(['paper', 'live'] as const).map((m) => (
                  <button
                    key={m}
                    onClick={() => setStartMode(m)}
                    className={cn(
                      'flex-1 py-2.5 rounded-lg text-sm font-medium transition-colors border',
                      startMode === m
                        ? m === 'live' ? 'bg-loss/10 border-loss text-loss' : 'bg-profit/10 border-profit text-profit'
                        : 'bg-surface-tertiary border-border text-text-muted hover:text-text-primary',
                    )}
                  >
                    {m === 'paper' ? '模拟' : '实盘'}
                  </button>
                ))}
              </div>
            </div>
            <div>
              <label className="block text-sm text-text-secondary mb-2">市场</label>
              <div className="flex gap-2">
                {(['crypto', 'futures', 'both'] as const).map((m) => (
                  <button
                    key={m}
                    onClick={() => setStartMarket(m)}
                    className={cn(
                      'flex-1 py-2.5 rounded-lg text-sm font-medium transition-colors border',
                      startMarket === m
                        ? 'bg-brand/10 border-brand text-brand'
                        : 'bg-surface-tertiary border-border text-text-muted hover:text-text-primary',
                    )}
                  >
                    {m === 'crypto' ? '加密' : m === 'futures' ? '期货' : '双市场'}
                  </button>
                ))}
              </div>
            </div>
            <div className="flex items-end">
              <Button
                variant={startMode === 'live' ? 'destructive' : 'primary'}
                size="lg"
                className="w-full"
                onClick={handleStart}
                loading={startTrading.isPending}
              >
                <Play className="w-4 h-4" />
                启动{startMode === 'live' ? '实盘' : '模拟'}交易
              </Button>
            </div>
          </div>
          {startMode === 'live' && (
            <div className="mt-4 p-3 bg-loss/5 border border-loss/20 rounded-lg flex items-start gap-2">
              <Zap className="w-4 h-4 text-loss mt-0.5 shrink-0" />
              <p className="text-xs text-loss">
                实盘模式将向真实交易所发送订单，产生实际盈亏。请确保已配置正确的 API Key 和风控参数。
              </p>
            </div>
          )}
        </Card>
      )}

      {/* Running KPIs */}
      {isRunning && (
        <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
          <div className="bg-surface-secondary rounded-xl border border-border p-4">
            <div className="text-xs text-text-muted mb-1">运行模式</div>
            <div className={cn('text-lg font-bold', currentMode === 'live' ? 'text-loss' : 'text-profit')}>
              {currentMode === 'live' ? '实盘' : '模拟'}
            </div>
          </div>
          <div className="bg-surface-secondary rounded-xl border border-border p-4">
            <div className="text-xs text-text-muted mb-1">Bar 计数</div>
            <div className="text-lg font-bold text-text-primary tabular-nums">{barCount}</div>
          </div>
          <div className="bg-surface-secondary rounded-xl border border-border p-4">
            <div className="text-xs text-text-muted mb-1">策略数</div>
            <div className="text-lg font-bold text-text-primary">{strategyCount}</div>
          </div>
          <div className="bg-surface-secondary rounded-xl border border-border p-4">
            <div className="text-xs text-text-muted mb-1">加密平均收益</div>
            <div className={cn('text-lg font-bold tabular-nums', (summary?.crypto?.avg_return ?? 0) >= 0 ? 'text-profit' : 'text-loss')}>
              {(summary?.crypto?.avg_return ?? 0).toFixed(2)}%
            </div>
          </div>
          <div className="bg-surface-secondary rounded-xl border border-border p-4">
            <div className="text-xs text-text-muted mb-1">实时事件</div>
            <div className="text-lg font-bold text-text-primary tabular-nums">{events.length}</div>
            <div className="text-[10px] text-text-muted">{recentFills.length} 笔成交</div>
          </div>
        </div>
      )}

      {/* Tabs: Strategies / Leaderboard / Events */}
      {isRunning && (
        <>
          <Tabs
            tabs={[
              { id: 'strategies', label: `策略 (${strategies.length})` },
              { id: 'leaderboard', label: '排行榜' },
              { id: 'events', label: `事件流 (${events.length})` },
            ]}
            active={activeTab}
            onChange={setActiveTab}
          />

          {activeTab === 'strategies' && (
            <Card noPadding>
              <div className="overflow-auto max-h-[60vh]">
                <table className="w-full text-sm">
                  <thead className="sticky top-0 bg-surface-secondary z-10">
                    <tr className="border-b border-border text-xs text-text-muted">
                      <th className="text-left px-4 py-2.5 font-medium">#</th>
                      <th className="text-left px-3 py-2.5 font-medium">策略</th>
                      <th className="text-left px-3 py-2.5 font-medium">状态</th>
                      <th className="text-left px-3 py-2.5 font-medium">品种</th>
                      <th className="text-right px-3 py-2.5 font-medium">收益率</th>
                      <th className="text-right px-3 py-2.5 font-medium">交易次数</th>
                      <th className="px-4 py-2.5 font-medium text-right">操作</th>
                    </tr>
                  </thead>
                  <tbody>
                    {strategies.map((s) => (
                      <tr key={Number(s.account_id)} className="border-b border-border/50 hover:bg-surface-tertiary/50">
                        <td className="px-4 py-2.5 text-text-muted">{Number(s.account_id)}</td>
                        <td className="px-3 py-2.5 font-medium text-text-primary">{String(s.name)}</td>
                        <td className="px-3 py-2.5">
                          <StatusBadge
                            variant={s.enabled ? 'success' : 'neutral'}
                            label={s.enabled ? String(s.state) : '已禁用'}
                          />
                        </td>
                        <td className="px-3 py-2.5 text-text-muted text-xs font-mono">
                          {(s.symbols as string[])?.slice(0, 3).join(', ')}
                          {((s.symbols as string[])?.length ?? 0) > 3 && '...'}
                        </td>
                        <td className="px-3 py-2.5 text-right">
                          <ReturnBadge value={Number(s.return_pct)} />
                        </td>
                        <td className="px-3 py-2.5 text-right text-text-secondary tabular-nums">
                          {Number(s.total_trades)}
                        </td>
                        <td className="px-4 py-2.5 text-right">
                          <Button
                            variant={s.enabled ? 'secondary' : 'primary'}
                            size="sm"
                            onClick={() => toggleStrategy.mutate(Number(s.account_id))}
                          >
                            {s.enabled ? '禁用' : '启用'}
                          </Button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}

          {activeTab === 'leaderboard' && (
            <Card noPadding>
              <div className="overflow-auto max-h-[60vh]">
                <table className="w-full text-sm">
                  <thead className="sticky top-0 bg-surface-secondary z-10">
                    <tr className="border-b border-border text-xs text-text-muted">
                      <th className="text-left px-4 py-2.5 font-medium">#</th>
                      <th className="text-left px-3 py-2.5 font-medium">策略</th>
                      <th className="text-left px-3 py-2.5 font-medium">市场</th>
                      <th className="text-right px-3 py-2.5 font-medium">收益率</th>
                      <th className="text-right px-3 py-2.5 font-medium">交易次数</th>
                      <th className="text-right px-4 py-2.5 font-medium">胜率</th>
                    </tr>
                  </thead>
                  <tbody>
                    {leaderboard.map((a, i) => (
                      <tr key={Number(a.account_id)} className="border-b border-border/50 hover:bg-surface-tertiary/50">
                        <td className="px-4 py-2.5 text-text-muted">{i + 1}</td>
                        <td className="px-3 py-2.5 font-medium text-text-primary">{String(a.strategy_name)}</td>
                        <td className="px-3 py-2.5">
                          <span className={cn('text-xs px-1.5 py-0.5 rounded', a.market === 'crypto' ? 'bg-[#f7931a]/15 text-[#f7931a]' : 'bg-brand/15 text-brand')}>
                            {a.market === 'crypto' ? '加密' : '期货'}
                          </span>
                        </td>
                        <td className="px-3 py-2.5 text-right">
                          <ReturnBadge value={Number(a.total_return_pct)} />
                        </td>
                        <td className="px-3 py-2.5 text-right text-text-secondary tabular-nums">{Number(a.total_trades)}</td>
                        <td className="px-4 py-2.5 text-right text-text-secondary tabular-nums">{Number(a.win_rate).toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}

          {activeTab === 'events' && (
            <Card title="实时事件流" extra={<Activity className="w-4 h-4 text-profit animate-pulse" />}>
              <div className="space-y-1 max-h-[50vh] overflow-auto font-mono text-xs">
                {events.length === 0 && (
                  <p className="text-text-muted text-center py-8">等待事件...</p>
                )}
                {[...events].reverse().map((e, i) => (
                  <div key={i} className="flex items-start gap-3 py-1.5 border-b border-border/30">
                    <span className="text-text-muted shrink-0 w-20">{e.timestamp?.slice(11, 19)}</span>
                    <StatusBadge
                      variant={
                        e.type === 'trade_fill' ? 'success' :
                        e.type === 'risk_alert' ? 'error' :
                        e.type === 'position_update' ? 'warning' : 'info'
                      }
                      label={e.type.replace('_', ' ')}
                    />
                    <span className="text-text-secondary truncate">
                      {JSON.stringify(e.data).slice(0, 120)}
                    </span>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </>
      )}
    </div>
  );
}
