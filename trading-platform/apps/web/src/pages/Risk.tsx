import { useAccount } from '@/hooks/useAccount';
import { useRiskAlerts } from '@/hooks/useMarketData';
import { usePositions } from '@/hooks/usePositions';
import { StatCardSkeleton } from '@/components/ui/Skeleton';
import Card from '@/components/Card';
import StatCard from '@/components/StatCard';
import StatusBadge from '@/components/StatusBadge';
import { fmtCny } from '@/lib/format';

function RiskGauge({ value, max, label, unit }: { value: number; max: number; label: string; unit: string }) {
  const pct = Math.min(value / max, 1) * 100;
  const color = pct > 80 ? 'bg-loss' : pct > 50 ? 'bg-warning' : 'bg-profit';

  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs text-text-muted">{label}</span>
        <span className="text-xs text-text-primary tabular-nums">{value.toFixed(1)}{unit}</span>
      </div>
      <div className="h-2 bg-surface-tertiary rounded-full overflow-hidden">
        <div className={`h-full rounded-full transition-all duration-500 ${color}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export default function Risk() {
  const { data: account, isLoading } = useAccount();
  const { data: alerts = [] } = useRiskAlerts();
  const { data: positions = [] } = usePositions();

  if (isLoading) {
    return (
      <div className="p-6 space-y-6">
        <h1 className="text-xl font-semibold text-text-primary">风控监控</h1>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          {Array.from({ length: 4 }).map((_, i) => <StatCardSkeleton key={i} />)}
        </div>
      </div>
    );
  }

  if (!account) return <div className="p-6 text-text-muted">无法加载账户数据</div>;

  const totalMargin = positions.reduce((sum, p) => sum + p.margin, 0);
  const activeAlerts = alerts.filter((a) => !a.resolved);
  const criticalCount = activeAlerts.filter((a) => a.level === 'CRITICAL').length;
  const warningCount = activeAlerts.filter((a) => a.level === 'WARNING').length;

  return (
    <div className="p-6 space-y-6">
      <h1 className="text-xl font-semibold text-text-primary">风控监控</h1>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          label="风险度"
          value={`${(account.risk_ratio * 100).toFixed(1)}%`}
          sub="预警线 30% / 强平线 80%"
          trend={account.risk_ratio > 0.3 ? 'down' : 'up'}
        />
        <StatCard label="占用保证金" value={fmtCny(account.margin)} sub={`可用 ${fmtCny(account.available)}`} trend="neutral" />
        <StatCard
          label="活跃风控提醒"
          value={String(activeAlerts.length)}
          sub={criticalCount > 0 ? `${criticalCount} 严重` : warningCount > 0 ? `${warningCount} 警告` : '无异常'}
          trend={criticalCount > 0 ? 'down' : warningCount > 0 ? 'down' : 'up'}
        />
        <StatCard
          label="持仓品种"
          value={String(positions.length)}
          sub="建议 ≥3 个品种分散风险"
          trend={positions.length >= 3 ? 'up' : 'down'}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Card title="风险指标仪表">
          <div className="space-y-5">
            <RiskGauge value={account.risk_ratio * 100} max={100} label="资金风险度" unit="%" />
            <RiskGauge
              value={totalMargin > 0 && positions[0] ? (positions[0].margin / totalMargin * 100) : 0}
              max={100} label="最大品种集中度" unit="%"
            />
            <RiskGauge value={positions.length} max={10} label="持仓品种数" unit="个" />
            <RiskGauge
              value={Math.abs(account.float_profit / (account.balance || 1) * 100)}
              max={10} label="浮盈比" unit="%"
            />
          </div>
        </Card>

        <Card title="持仓保证金分布" noPadding>
          <div className="p-4 space-y-3">
            {positions.map((p) => {
              const pct = totalMargin > 0 ? (p.margin / totalMargin * 100) : 0;
              return (
                <div key={`${p.instrument_id}-${p.direction}`}>
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-mono text-text-primary">{p.instrument_id}</span>
                      <StatusBadge variant={p.direction === 'LONG' ? 'success' : 'error'} label={p.direction === 'LONG' ? '多' : '空'} />
                    </div>
                    <span className="text-xs text-text-secondary tabular-nums">
                      {fmtCny(p.margin)} ({pct.toFixed(1)}%)
                    </span>
                  </div>
                  <div className="h-2 bg-surface-tertiary rounded-full overflow-hidden">
                    <div className="h-full bg-brand rounded-full" style={{ width: `${pct}%` }} />
                  </div>
                </div>
              );
            })}
            {positions.length === 0 && (
              <p className="text-sm text-text-muted text-center py-4">暂无持仓</p>
            )}
          </div>
        </Card>
      </div>

      <Card title="风控日志" noPadding>
        <div className="divide-y divide-border/50">
          {alerts.map((a) => (
            <div key={a.id} className={`flex items-start gap-3 px-4 py-3 ${a.resolved ? 'opacity-50' : ''}`}>
              <StatusBadge
                variant={a.level === 'CRITICAL' ? 'error' : a.level === 'WARNING' ? 'warning' : 'info'}
                label={a.level}
              />
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-medium text-text-primary">{a.type}</span>
                  {a.resolved && <StatusBadge variant="neutral" label="已解决" />}
                </div>
                <p className="text-sm text-text-secondary mt-0.5">{a.message}</p>
                <p className="text-xs text-text-muted mt-1">{a.created_at.replace('T', ' ')}</p>
              </div>
            </div>
          ))}
          {alerts.length === 0 && (
            <div className="px-4 py-8 text-center text-text-muted text-sm">暂无风控记录</div>
          )}
        </div>
      </Card>
    </div>
  );
}
