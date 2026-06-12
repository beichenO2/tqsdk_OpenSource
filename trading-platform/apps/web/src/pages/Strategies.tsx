import { Play, Square, Pencil, Plus, TrendingUp, TrendingDown, Clock } from 'lucide-react';
import { useStrategies, useToggleStrategy } from '@/hooks/useStrategies';
import { useConfirm } from '@/components/ui/ConfirmDialog';
import { useToast } from '@/components/ui/Toast';
import { Button } from '@/components/ui/Button';
import { StatCardSkeleton } from '@/components/ui/Skeleton';
import Card from '@/components/Card';
import StatusBadge from '@/components/StatusBadge';
import StatCard from '@/components/StatCard';
import { fmtCny } from '@/lib/format';
import type { Strategy } from '@/types';

function statusVariant(s: Strategy['status']) {
  return s === 'RUNNING' ? 'success' as const
    : s === 'ERROR' ? 'error' as const
    : s === 'PAUSED' ? 'warning' as const
    : 'neutral' as const;
}

function statusLabel(s: Strategy['status']) {
  return s === 'RUNNING' ? '运行中'
    : s === 'STOPPED' ? '已停止'
    : s === 'PAUSED' ? '暂停'
    : '异常';
}

export default function Strategies() {
  const { data: strategies = [], isLoading } = useStrategies();
  const toggleStrategy = useToggleStrategy();
  const confirm = useConfirm();
  const toast = useToast();

  const handleToggle = async (s: Strategy) => {
    const action = s.status === 'RUNNING' ? '停止' : '启动';
    const ok = await confirm({
      title: `确认${action}策略`,
      description: `${action}策略「${s.name}」？${s.status === 'RUNNING' ? '停止后将不再产生新交易信号。' : '启动后策略将开始运行。'}`,
      variant: s.status === 'RUNNING' ? 'warning' : 'default',
      confirmText: `确认${action}`,
    });
    if (!ok) return;
    try {
      await toggleStrategy.mutateAsync({ id: s.id, running: s.status === 'RUNNING' });
      toast.success(`策略已${action}`);
    } catch {
      toast.error(`${action}失败`);
    }
  };

  const running = strategies.filter((s) => s.status === 'RUNNING');
  const totalPnl = strategies.reduce((sum, s) => sum + s.pnl, 0);
  const profitable = strategies.filter((s) => s.pnl > 0).length;

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold text-text-primary">策略管理</h1>
        <Button>
          <Plus className="w-4 h-4" />
          新建策略
        </Button>
      </div>

      {isLoading ? (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          {Array.from({ length: 4 }).map((_, i) => <StatCardSkeleton key={i} />)}
        </div>
      ) : (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <StatCard label="策略总数" value={String(strategies.length)} sub={`${running.length} 个运行中`} trend="neutral" />
          <StatCard label="策略总盈亏" value={fmtCny(totalPnl)} trend={totalPnl >= 0 ? 'up' : 'down'} />
          <StatCard
            label="盈利策略"
            value={`${profitable} / ${strategies.length}`}
            sub={strategies.length > 0 ? `${(profitable / strategies.length * 100).toFixed(0)}% 盈利` : '暂无'}
            trend={profitable > strategies.length / 2 ? 'up' : 'down'}
          />
          <StatCard
            label="覆盖品种"
            value={String(new Set(strategies.flatMap((s) => s.instruments)).size)}
            sub="个合约"
            trend="neutral"
          />
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {strategies.map((s) => (
          <Card key={s.id}>
            <div className="space-y-4">
              <div className="flex items-start justify-between">
                <div>
                  <div className="flex items-center gap-2">
                    <h3 className="text-base font-semibold text-text-primary">{s.name}</h3>
                    <StatusBadge variant={statusVariant(s.status)} label={statusLabel(s.status)} pulse={s.status === 'RUNNING'} />
                  </div>
                  <p className="text-xs text-text-muted mt-1">{s.description}</p>
                </div>
                <div className="text-right">
                  <span className={`text-lg font-semibold tabular-nums ${s.pnl >= 0 ? 'text-profit' : 'text-loss'}`}>
                    {fmtCny(s.pnl)}
                  </span>
                  <div className="flex items-center justify-end gap-1 mt-0.5">
                    {s.pnl >= 0 ? <TrendingUp className="w-3 h-3 text-profit" /> : <TrendingDown className="w-3 h-3 text-loss" />}
                    <span className={`text-[10px] tabular-nums ${s.pnl >= 0 ? 'text-profit' : 'text-loss'}`}>
                      {s.pnl >= 0 ? '+' : ''}{((s.pnl / 100000) * 100).toFixed(2)}%
                    </span>
                  </div>
                </div>
              </div>

              <div className="flex flex-wrap gap-1.5">
                {s.instruments.map((i) => (
                  <span key={i} className="px-2 py-0.5 bg-surface-tertiary rounded text-xs font-mono text-text-secondary">{i}</span>
                ))}
              </div>

              <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs bg-surface-tertiary/50 rounded-lg p-3">
                {Object.entries(s.params).map(([k, v]) => (
                  <div key={k} className="flex justify-between">
                    <span className="text-text-muted">{k}</span>
                    <span className="text-text-secondary tabular-nums">{String(v)}</span>
                  </div>
                ))}
              </div>

              <div className="flex items-center justify-between pt-2 border-t border-border/50">
                <div className="flex items-center gap-1.5 text-xs text-text-muted">
                  <Clock className="w-3 h-3" />
                  <span>更新于 {s.updated_at}</span>
                </div>
                <div className="flex gap-2">
                  <Button variant="secondary" size="sm">
                    <Pencil className="w-3 h-3" />
                    编辑
                  </Button>
                  <Button
                    variant={s.status === 'RUNNING' ? 'destructive' : 'profit'}
                    size="sm"
                    onClick={() => handleToggle(s)}
                    loading={toggleStrategy.isPending}
                  >
                    {s.status === 'RUNNING' ? <><Square className="w-3 h-3" />停止</> : <><Play className="w-3 h-3" />启动</>}
                  </Button>
                </div>
              </div>
            </div>
          </Card>
        ))}

        {strategies.length === 0 && !isLoading && (
          <Card className="lg:col-span-2">
            <div className="py-12 text-center">
              <p className="text-text-muted text-sm">暂无策略，点击右上角创建第一个策略</p>
            </div>
          </Card>
        )}
      </div>
    </div>
  );
}
