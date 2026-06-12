import { useEffect, useState, useCallback, useRef } from 'react';
import {
  AreaChart, Area, XAxis, YAxis, Tooltip,
  ResponsiveContainer, BarChart, Bar, CartesianGrid,
} from 'recharts';
import { RefreshCw, XCircle, PauseCircle, Bitcoin } from 'lucide-react';
import { cryptoApi } from './api';
import StatCard from '@/components/StatCard';
import Card from '@/components/Card';
import StatusBadge from '@/components/StatusBadge';
import { Button } from '@/components/ui/Button';
import { useConfirm } from '@/components/ui/ConfirmDialog';
import { useToast } from '@/components/ui/Toast';
import { fmtUsd, fmt } from '@/lib/format';
import type { CryptoAccount, CryptoPosition, CryptoStrategy } from './types';

const STRATEGY_TYPE_LABEL: Record<string, string> = {
  grid: '网格',
  momentum: '动量',
  mean_reversion: '均值回归',
  arbitrage: '套利',
};

export default function CryptoDashboard() {
  const confirm = useConfirm();
  const toast = useToast();
  const [account, setAccount] = useState<CryptoAccount | null>(null);
  const [positions, setPositions] = useState<CryptoPosition[]>([]);
  const [strategies, setStrategies] = useState<CryptoStrategy[]>([]);
  const [pnlData] = useState<{ time: string; pnl: number; hourly: number }[]>([]);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());
  const [refreshing, setRefreshing] = useState(false);
  const timerRef = useRef<ReturnType<typeof setInterval> | undefined>(undefined);

  const loadData = useCallback(async () => {
    try {
      const [a, p, s] = await Promise.all([
        cryptoApi.getAccount(),
        cryptoApi.getPositions(),
        cryptoApi.getStrategies(),
      ]);
      setAccount(a);
      setPositions(p);
      setStrategies(s);
      setLastUpdate(new Date());
    } catch { /* retry next tick */ }
  }, []);

  useEffect(() => {
    loadData();
    timerRef.current = setInterval(loadData, 5_000);
    return () => clearInterval(timerRef.current);
  }, [loadData]);

  const handleRefresh = async () => {
    setRefreshing(true);
    await loadData();
    setRefreshing(false);
  };

  const handleCloseAll = async () => {
    const ok = await confirm({
      title: '确认一键平仓（加密）',
      description: '这将关闭所有加密货币持仓，该操作不可撤回。',
      variant: 'destructive',
      confirmText: '确认平仓',
    });
    if (!ok) return;
    try {
      await cryptoApi.closeAllPositions();
      await loadData();
      toast.success('已提交平仓指令');
    } catch {
      toast.error('平仓失败');
    }
  };

  const EMPTY_ACCT: CryptoAccount = { total_equity: 0, available_balance: 0, unrealized_pnl: 0, realized_pnl_24h: 0, margin_ratio: 0, btc_equivalent: 0, total_margin: 0 };
  const acct: CryptoAccount = account
    ? { ...EMPTY_ACCT, ...account }
    : EMPTY_ACCT;

  const safeStrategies = strategies ?? [];
  const runningStrategies = safeStrategies.filter(s => s.status === 'RUNNING').length;
  const totalPnl24h = safeStrategies.reduce((sum, s) => sum + s.pnl_24h, 0);

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Bitcoin className="w-6 h-6 text-[#f7931a]" />
          <h1 className="text-xl font-semibold text-text-primary">BTC 仪表盘</h1>
        </div>
        <div className="flex items-center gap-3">
          <Button variant="destructive" size="sm" onClick={handleCloseAll}>
            <XCircle className="w-3.5 h-3.5" />
            一键平仓
          </Button>
          <Button variant="secondary" size="sm">
            <PauseCircle className="w-3.5 h-3.5" />
            暂停策略
          </Button>
          <div className="h-5 w-px bg-border" />
          <Button variant="ghost" size="sm" onClick={handleRefresh} title="刷新数据">
            <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
          </Button>
          <div className="flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-profit animate-pulse" />
            <span className="text-xs text-text-muted">
              {lastUpdate.toLocaleTimeString('zh-CN')} · 24/7
            </span>
          </div>
        </div>
      </div>

      {/* KPI Row */}
      <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
        <StatCard
          label="总权益"
          value={fmtUsd(acct.total_equity)}
          sub={`≈ ${acct.btc_equivalent.toFixed(4)} BTC`}
          trend="neutral"
        />
        <StatCard
          label="可用余额"
          value={fmtUsd(acct.available_balance)}
          trend="neutral"
        />
        <StatCard
          label="未实现盈亏"
          value={fmtUsd(acct.unrealized_pnl)}
          trend={acct.unrealized_pnl >= 0 ? 'up' : 'down'}
        />
        <StatCard
          label="24h 已实现"
          value={fmtUsd(acct.realized_pnl_24h)}
          sub={`${totalPnl24h >= 0 ? '+' : ''}${fmt(totalPnl24h)} 策略`}
          trend={acct.realized_pnl_24h >= 0 ? 'up' : 'down'}
        />
        <StatCard
          label="保证金率"
          value={`${(acct.margin_ratio * 100).toFixed(1)}%`}
          sub={acct.margin_ratio > 0.8 ? '风险偏高' : '风险可控'}
          trend={acct.margin_ratio > 0.8 ? 'down' : 'up'}
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <Card title="72h 累计盈亏" className="lg:col-span-2">
          <ResponsiveContainer width="100%" height={240}>
            <AreaChart data={pnlData}>
              <defs>
                <linearGradient id="cryptoPnlGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="var(--color-chart-btc)" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="var(--color-chart-btc)" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-chart-grid)" />
              <XAxis
                dataKey="time"
                tick={{ fill: 'var(--color-chart-axis)', fontSize: 11 }}
                tickFormatter={v => v.slice(11, 16)}
              />
              <YAxis
                tick={{ fill: 'var(--color-chart-axis)', fontSize: 11 }}
                tickFormatter={v => `$${v}`}
              />
              <Tooltip
                contentStyle={{ background: 'var(--color-chart-tooltip-bg)', border: '1px solid var(--color-chart-tooltip-border)', borderRadius: 8, fontSize: 12 }}
                labelStyle={{ color: 'var(--color-text-secondary)' }}
                formatter={(v) => [fmtUsd(Number(v)), '累计盈亏']}
              />
              <Area type="monotone" dataKey="pnl" stroke="var(--color-chart-btc)" fill="url(#cryptoPnlGrad)" />
            </AreaChart>
          </ResponsiveContainer>
        </Card>

        <Card title="每小时盈亏">
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={pnlData.slice(-24)}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-chart-grid)" />
              <XAxis
                dataKey="time"
                tick={{ fill: 'var(--color-chart-axis)', fontSize: 10 }}
                tickFormatter={v => v.slice(11, 16)}
              />
              <YAxis tick={{ fill: 'var(--color-chart-axis)', fontSize: 10 }} />
              <Tooltip
                contentStyle={{ background: 'var(--color-chart-tooltip-bg)', border: '1px solid var(--color-chart-tooltip-border)', borderRadius: 8, fontSize: 12 }}
                formatter={(v) => [fmtUsd(Number(v)), '小时盈亏']}
              />
              <Bar dataKey="hourly" radius={[3, 3, 0, 0]} fill="var(--color-chart-btc)" />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>

      {/* Positions & Strategies */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Card title="持仓概览" extra={<span className="text-xs text-text-muted tabular-nums">{positions.length} 个币种</span>} noPadding>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-text-muted text-xs">
                  <th className="text-left px-4 py-2.5 font-medium">币对</th>
                  <th className="text-left px-4 py-2.5 font-medium">方向</th>
                  <th className="text-right px-4 py-2.5 font-medium">数量</th>
                  <th className="text-right px-4 py-2.5 font-medium">开仓价</th>
                  <th className="text-right px-4 py-2.5 font-medium">标记价</th>
                  <th className="text-right px-4 py-2.5 font-medium">未实现盈亏</th>
                  <th className="text-right px-4 py-2.5 font-medium">杠杆</th>
                </tr>
              </thead>
              <tbody>
                {positions.map(p => (
                  <tr key={`${p.symbol}-${p.side}`} className="border-b border-border/50 hover:bg-surface-tertiary/50">
                    <td className="px-4 py-2.5 font-mono text-text-primary">{p.symbol}</td>
                    <td className="px-4 py-2.5">
                      <StatusBadge
                        variant={p.side === 'LONG' ? 'success' : 'error'}
                        label={p.side === 'LONG' ? '多' : '空'}
                      />
                    </td>
                    <td className="px-4 py-2.5 text-right tabular-nums">{p.size}</td>
                    <td className="px-4 py-2.5 text-right tabular-nums">{fmtUsd(p.entry_price)}</td>
                    <td className="px-4 py-2.5 text-right tabular-nums">{fmtUsd(p.mark_price)}</td>
                    <td className={`px-4 py-2.5 text-right tabular-nums font-medium ${p.unrealized_pnl >= 0 ? 'text-profit' : 'text-loss'}`}>
                      {fmtUsd(p.unrealized_pnl)}
                    </td>
                    <td className="px-4 py-2.5 text-right">
                      <span className="inline-flex items-center px-1.5 py-0.5 rounded bg-brand/10 text-brand-light text-xs tabular-nums">
                        {p.leverage}x
                      </span>
                    </td>
                  </tr>
                ))}
                {positions.length === 0 && (
                  <tr><td colSpan={7} className="px-4 py-8 text-center text-text-muted text-sm">暂无持仓</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </Card>

        <div className="space-y-4">
          <Card title="策略状态" extra={<span className="text-xs text-text-muted tabular-nums">{runningStrategies} 个运行中</span>} noPadding>
            <div className="divide-y divide-border/50">
              {strategies.map(s => (
                <div key={s.id} className="flex items-center justify-between px-4 py-3 hover:bg-surface-tertiary/50">
                  <div>
                    <p className="text-sm text-text-primary font-medium">{s.name}</p>
                    <p className="text-xs text-text-muted mt-0.5">
                      {STRATEGY_TYPE_LABEL[s.type] ?? s.type} · {s.symbols.join(', ')}
                    </p>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="text-right">
                      <span className={`text-sm tabular-nums font-medium ${s.total_pnl >= 0 ? 'text-profit' : 'text-loss'}`}>
                        {fmtUsd(s.total_pnl)}
                      </span>
                      <p className={`text-xs tabular-nums ${s.pnl_24h >= 0 ? 'text-profit' : 'text-loss'}`}>
                        24h {s.pnl_24h >= 0 ? '+' : ''}{fmtUsd(s.pnl_24h)}
                      </p>
                    </div>
                    <StatusBadge
                      variant={
                        s.status === 'RUNNING' ? 'success'
                        : s.status === 'ERROR' ? 'error'
                        : s.status === 'PAUSED' ? 'warning'
                        : 'neutral'
                      }
                      label={
                        s.status === 'RUNNING' ? '运行中'
                        : s.status === 'STOPPED' ? '已停止'
                        : s.status === 'PAUSED' ? '暂停'
                        : '异常'
                      }
                      pulse={s.status === 'RUNNING'}
                    />
                  </div>
                </div>
              ))}
              {strategies.length === 0 && (
                <div className="px-4 py-8 text-center text-text-muted text-sm">暂无策略</div>
              )}
            </div>
          </Card>

          <Card title="风险指标">
            <div className="space-y-3">
              <div className="flex justify-between text-sm">
                <span className="text-text-muted">保证金率</span>
                <span className={`tabular-nums font-medium ${acct.margin_ratio > 0.8 ? 'text-loss' : acct.margin_ratio > 0.5 ? 'text-warning' : 'text-profit'}`}>
                  {(acct.margin_ratio * 100).toFixed(1)}%
                </span>
              </div>
              <div className="w-full bg-surface-tertiary rounded-full h-2">
                <div
                  className={`h-2 rounded-full transition-all ${acct.margin_ratio > 0.8 ? 'bg-loss' : acct.margin_ratio > 0.5 ? 'bg-warning' : 'bg-profit'}`}
                  style={{ width: `${Math.min(acct.margin_ratio * 100, 100)}%` }}
                />
              </div>
              <div className="h-px bg-border/50" />
              <div className="flex justify-between text-sm">
                <span className="text-text-muted">已用保证金</span>
                <span className="tabular-nums text-text-primary">{fmtUsd(acct.total_margin)}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-text-muted">BTC 等值</span>
                <span className="tabular-nums text-text-primary">{acct.btc_equivalent.toFixed(4)} BTC</span>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
