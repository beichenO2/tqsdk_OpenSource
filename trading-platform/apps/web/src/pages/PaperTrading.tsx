import { useEffect, useState, useMemo } from 'react';
import {
  Trophy, TrendingUp, Users, BarChart3,
  ArrowUpRight, ArrowDownRight, Filter,
  Play, Square, Radio,
} from 'lucide-react';
import { useLiveTradingStatus, useStartLiveTrading, useStopLiveTrading, useSwitchMode } from '@/hooks/useLiveTrading';
import { useConfirm } from '@/components/ui/ConfirmDialog';
import { useToast } from '@/components/ui/Toast';
import { Button } from '@/components/ui/Button';
import { cn } from '@/lib/cn';
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer,
} from 'recharts';

const API_BASE = '/api/v1/paper-trading';

interface ISummary {
  crypto: {
    count: number;
    avg_return: number;
    best: number;
    worst: number;
    profitable: number;
  };
  futures: {
    count: number;
    avg_return: number;
    best: number;
    worst: number;
    profitable: number;
  };
}

interface IAccount {
  account_id: number;
  market: string;
  strategy_name: string;
  initial_capital: number;
  capital: number;
  total_equity: number;
  total_return_pct: number;
  total_trades: number;
  win_rate: number;
  is_active: boolean;
  positions: Record<string, unknown>;
}

interface ICategoryStats {
  [family: string]: {
    count: number;
    avg_return: number;
    best: number;
    worst: number;
    profitable: number;
  };
}

async function fetchJson<T>(url: string): Promise<T | null> {
  try {
    const res = await fetch(`${API_BASE}${url}`);
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

function StatCard({
  label, value, sub, icon: Icon, color,
}: {
  label: string;
  value: string;
  sub?: string;
  icon: typeof TrendingUp;
  color: string;
}) {
  return (
    <div className="bg-surface-secondary rounded-xl border border-border p-5">
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm text-text-secondary">{label}</span>
        <div className={`w-9 h-9 rounded-lg flex items-center justify-center ${color}`}>
          <Icon className="w-[18px] h-[18px]" />
        </div>
      </div>
      <p className="text-2xl font-bold text-text-primary">{value}</p>
      {sub && <p className="text-xs text-text-muted mt-1">{sub}</p>}
    </div>
  );
}

function ReturnBadge({ value }: { value: number }) {
  const positive = value >= 0;
  return (
    <span className={`inline-flex items-center gap-0.5 text-sm font-medium ${
      positive ? 'text-profit' : 'text-loss'
    }`}>
      {positive ? <ArrowUpRight className="w-3.5 h-3.5" /> : <ArrowDownRight className="w-3.5 h-3.5" />}
      {value >= 0 ? '+' : ''}{value.toFixed(2)}%
    </span>
  );
}

export default function PaperTrading() {
  const confirm = useConfirm();
  const toast = useToast();
  const [summary, setSummary] = useState<ISummary | null>(null);
  const [leaderboard, setLeaderboard] = useState<IAccount[]>([]);
  const [categoryStats, setCategoryStats] = useState<ICategoryStats>({});
  const [selectedAccount, setSelectedAccount] = useState<number | null>(null);
  const [equityCurve, setEquityCurve] = useState<[string, number][]>([]);
  const [marketFilter, setMarketFilter] = useState<string>('all');
  const [loading, setLoading] = useState(true);
  const { data: liveStatus } = useLiveTradingStatus() as { data: Record<string, unknown> | undefined };
  const startLive = useStartLiveTrading();
  const stopLive = useStopLiveTrading();
  const switchMode = useSwitchMode();
  const liveRunning = !!(liveStatus as Record<string, unknown>)?.running;
  const liveMode = String((liveStatus as Record<string, unknown>)?.mode || 'paper');

  useEffect(() => {
    async function load() {
      setLoading(true);
      const [sum, lb, cats] = await Promise.all([
        fetchJson<ISummary>('/summary'),
        fetchJson<IAccount[]>(
          `/leaderboard?top_n=50${marketFilter !== 'all' ? `&market=${marketFilter}` : ''}`
        ),
        fetchJson<ICategoryStats>('/category-stats'),
      ]);
      if (sum) setSummary(sum);
      if (lb) setLeaderboard(lb);
      if (cats) setCategoryStats(cats);
      setLoading(false);
    }
    load();
  }, [marketFilter]);

  useEffect(() => {
    if (selectedAccount === null) return;
    fetchJson<[string, number][]>(`/equity/${selectedAccount}`).then((data) => {
      if (data) setEquityCurve(data);
    });
  }, [selectedAccount]);

  const chartData = useMemo(
    () => equityCurve.map(([ts, eq]) => ({
      time: ts.slice(0, 16),
      equity: Math.round(eq * 100) / 100,
    })),
    [equityCurve],
  );

  const sortedFamilies = useMemo(
    () => Object.entries(categoryStats).sort(([, a], [, b]) => b.avg_return - a.avg_return),
    [categoryStats],
  );

  if (loading && !summary) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="animate-spin w-8 h-8 border-2 border-brand border-t-transparent rounded-full" />
      </div>
    );
  }

  if (!summary) {
    return (
      <div className="p-8 text-center">
        <p className="text-text-secondary text-lg">模拟实盘数据未就绪</p>
        <p className="text-text-muted text-sm mt-2">
          请先运行 <code className="bg-surface-tertiary px-1.5 py-0.5 rounded">
          python scripts/run_paper_trading.py</code>
        </p>
      </div>
    );
  }

  return (
    <div className="px-[3%] py-[2%] space-y-[clamp(1rem,2vw,1.5rem)]">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-text-primary">模拟实盘</h1>
          <p className="text-text-secondary text-sm mt-1">
            200 个 SOTA 策略 · 100 加密货币 + 100 期货
          </p>
        </div>
        <div className="flex items-center gap-3">
          {/* Live Trading Quick Controls */}
          <div className="flex items-center gap-2 mr-4">
            {liveRunning ? (
              <>
                <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-profit/10 border border-profit/20">
                  <Radio className="w-3.5 h-3.5 text-profit animate-pulse" />
                  <span className="text-xs font-medium text-profit">{liveMode === 'live' ? '实盘' : '模拟'}中</span>
                </div>
                <Button variant="secondary" size="sm" onClick={async () => {
                  const target = liveMode === 'paper' ? 'live' : 'paper';
                  if (target === 'live') {
                    const ok = await confirm({ title: '切换到实盘', description: '策略信号将发送到真实交易所！', variant: 'destructive', confirmText: '确认' });
                    if (!ok) return;
                  }
                  try { await switchMode.mutateAsync(target); toast.success(`已切换到${target === 'live' ? '实盘' : '模拟'}`); }
                  catch { toast.error('切换失败'); }
                }}>
                  切换{liveMode === 'paper' ? '实盘' : '模拟'}
                </Button>
                <Button variant="destructive" size="sm" onClick={async () => {
                  try { await stopLive.mutateAsync(); toast.success('已停止'); } catch { toast.error('停止失败'); }
                }} loading={stopLive.isPending}>
                  <Square className="w-3 h-3" />
                </Button>
              </>
            ) : (
              <Button variant="primary" size="sm" onClick={async () => {
                try { await startLive.mutateAsync({ mode: 'paper', market: marketFilter === 'all' ? 'crypto' : marketFilter }); toast.success('已启动模拟交易'); }
                catch (e) { toast.error(e instanceof Error ? e.message : '启动失败'); }
              }} loading={startLive.isPending}>
                <Play className="w-3 h-3" />
                启动实时模拟
              </Button>
            )}
          </div>
          <div className="h-5 w-px bg-border" />
          {['all', 'crypto', 'futures'].map((m) => (
            <button
              key={m}
              onClick={() => setMarketFilter(m)}
              className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                marketFilter === m
                  ? 'bg-brand text-white'
                  : 'bg-surface-tertiary text-text-secondary hover:text-text-primary'
              }`}
            >
              {m === 'all' ? '全部' : m === 'crypto' ? '加密货币' : '期货'}
            </button>
          ))}
        </div>
      </div>

      {/* 统计卡片 */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          label="加密货币平均收益"
          value={`${summary.crypto.avg_return >= 0 ? '+' : ''}${summary.crypto.avg_return.toFixed(2)}%`}
          sub={`${summary.crypto.profitable}/${summary.crypto.count} 盈利`}
          icon={TrendingUp}
          color="bg-[#f7931a]/15 text-[#f7931a]"
        />
        <StatCard
          label="期货平均收益"
          value={`${summary.futures.avg_return >= 0 ? '+' : ''}${summary.futures.avg_return.toFixed(2)}%`}
          sub={`${summary.futures.profitable}/${summary.futures.count} 盈利`}
          icon={BarChart3}
          color="bg-brand/15 text-brand"
        />
        <StatCard
          label="最佳策略"
          value={`+${Math.max(summary.crypto.best, summary.futures.best).toFixed(2)}%`}
          sub={summary.crypto.best > summary.futures.best ? '加密货币' : '期货'}
          icon={Trophy}
          color="bg-profit/15 text-profit"
        />
        <StatCard
          label="总账号数"
          value={`${summary.crypto.count + summary.futures.count}`}
          sub={`${summary.crypto.count} 加密 + ${summary.futures.count} 期货`}
          icon={Users}
          color="bg-text-secondary/15 text-text-secondary"
        />
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* 排行榜 */}
        <div className="xl:col-span-2 bg-surface-secondary rounded-xl border border-border">
          <div className="p-4 border-b border-border flex items-center gap-2">
            <Trophy className="w-5 h-5 text-[#f7931a]" />
            <h2 className="font-semibold text-text-primary">策略排行榜</h2>
            <span className="text-xs text-text-muted ml-auto">Top 50</span>
          </div>
          <div className="overflow-auto max-h-[60vh]">
            <table className="w-full">
              <thead className="sticky top-0 bg-surface-secondary z-10">
                <tr className="text-xs text-text-muted border-b border-border">
                  <th className="text-left px-4 py-2.5 font-medium">#</th>
                  <th className="text-left px-3 py-2.5 font-medium">策略</th>
                  <th className="text-left px-3 py-2.5 font-medium">市场</th>
                  <th className="text-right px-3 py-2.5 font-medium">收益率</th>
                  <th className="text-right px-3 py-2.5 font-medium">交易</th>
                  <th className="text-right px-4 py-2.5 font-medium">胜率</th>
                </tr>
              </thead>
              <tbody>
                {leaderboard.map((acct, i) => (
                  <tr
                    key={acct.account_id}
                    onClick={() => setSelectedAccount(acct.account_id)}
                    className={`border-b border-border/50 hover:bg-surface-tertiary cursor-pointer transition-colors ${
                      selectedAccount === acct.account_id ? 'bg-brand/5' : ''
                    }`}
                  >
                    <td className="px-4 py-2.5 text-sm text-text-muted">{i + 1}</td>
                    <td className="px-3 py-2.5">
                      <div className="text-sm font-medium text-text-primary">
                        {acct.strategy_name}
                      </div>
                      <div className="text-[10px] text-text-muted">#{acct.account_id}</div>
                    </td>
                    <td className="px-3 py-2.5">
                      <span className={`text-xs px-1.5 py-0.5 rounded ${
                        acct.market === 'crypto'
                          ? 'bg-[#f7931a]/15 text-[#f7931a]'
                          : 'bg-brand/15 text-brand'
                      }`}>
                        {acct.market === 'crypto' ? '加密' : '期货'}
                      </span>
                    </td>
                    <td className="px-3 py-2.5 text-right">
                      <ReturnBadge value={acct.total_return_pct} />
                    </td>
                    <td className="px-3 py-2.5 text-right text-sm text-text-secondary">
                      {acct.total_trades}
                    </td>
                    <td className="px-4 py-2.5 text-right text-sm text-text-secondary">
                      {acct.win_rate.toFixed(1)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* 策略家族统计 */}
        <div className="bg-surface-secondary rounded-xl border border-border">
          <div className="p-4 border-b border-border flex items-center gap-2">
            <Filter className="w-5 h-5 text-brand" />
            <h2 className="font-semibold text-text-primary">策略家族</h2>
          </div>
          <div className="overflow-auto max-h-[60vh] p-2">
            {sortedFamilies.map(([family, stats]) => (
              <div
                key={family}
                className="flex items-center justify-between px-3 py-2.5 rounded-lg hover:bg-surface-tertiary"
              >
                <div>
                  <span className="text-sm font-medium text-text-primary">{family}</span>
                  <span className="text-[10px] text-text-muted ml-1.5">×{stats.count}</span>
                </div>
                <div className="text-right">
                  <ReturnBadge value={stats.avg_return} />
                  <div className="text-[10px] text-text-muted">
                    {stats.profitable}/{stats.count} 盈利
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* 净值曲线 */}
      {selectedAccount !== null && chartData.length > 0 && (
        <div className="bg-surface-secondary rounded-xl border border-border p-5">
          <h2 className="font-semibold text-text-primary mb-4">
            #{selectedAccount} 净值曲线
            <span className="text-sm font-normal text-text-secondary ml-2">
              {leaderboard.find(a => a.account_id === selectedAccount)?.strategy_name}
            </span>
          </h2>
          <div className="aspect-[3/1] min-h-[12rem]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="eqGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#f0780e" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#f0780e" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis
                  dataKey="time"
                  tick={{ fontSize: 10, fill: '#94a3b8' }}
                  tickLine={false}
                  axisLine={false}
                />
                <YAxis
                  tick={{ fontSize: 10, fill: '#94a3b8' }}
                  tickLine={false}
                  axisLine={false}
                  width={60}
                  tickFormatter={(v: number) => v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v.toFixed(0)}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1e293b',
                    border: '1px solid #334155',
                    borderRadius: '8px',
                    fontSize: '12px',
                  }}
                  formatter={(v) => [`$${Number(v).toLocaleString()}`, '净值']}
                />
                <Area
                  type="monotone"
                  dataKey="equity"
                  stroke="#f0780e"
                  strokeWidth={2}
                  fill="url(#eqGrad)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}
