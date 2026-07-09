import { useMemo, useState } from 'react';
import { Activity, Filter } from 'lucide-react';
import { useLiveEvents } from '@/hooks/useLiveTrading';
import Card from '@/components/Card';
import StatusBadge from '@/components/StatusBadge';
import { cn } from '@/lib/cn';

const CHANNELS = [
  'position_update',
  'trade_fill',
  'account_update',
  'strategy_status',
  'signal',
  'risk_alert',
] as const;

function badgeVariant(type: string) {
  if (type === 'trade_fill') return 'success' as const;
  if (type === 'risk_alert') return 'error' as const;
  if (type === 'position_update') return 'warning' as const;
  if (type === 'signal') return 'info' as const;
  return 'neutral' as const;
}

export default function EventsPage() {
  const { events, connected } = useLiveEvents();
  const [filter, setFilter] = useState<string>('all');

  const filtered = useMemo(() => {
    if (filter === 'all') return events;
    return events.filter((e) => e.type === filter);
  }, [events, filter]);

  const counts = useMemo(() => {
    const m: Record<string, number> = {};
    for (const e of events) m[e.type] = (m[e.type] || 0) + 1;
    return m;
  }, [events]);

  return (
    <div className="px-[3%] py-[2%] space-y-4 max-w-[96rem] mx-auto">
      <div className="flex items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold text-text-primary">实时事件流</h1>
          <p className="text-sm text-text-secondary mt-0.5">
            WebSocket 六类事件：持仓 / 成交 / 账户 / 策略 / 信号 / 风控
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <span className={cn('w-2 h-2 rounded-full', connected ? 'bg-profit animate-pulse' : 'bg-loss')} />
          <span className="text-text-muted">{connected ? '已连接' : '未连接'}</span>
          <span className="text-text-muted">· {events.length} 条</span>
        </div>
      </div>

      <div className="flex flex-wrap gap-2 items-center">
        <Filter className="w-3.5 h-3.5 text-text-muted" />
        <button
          type="button"
          onClick={() => setFilter('all')}
          className={cn(
            'rounded-lg px-2.5 py-1 text-xs border transition-colors',
            filter === 'all' ? 'bg-brand/10 border-brand text-brand' : 'border-border text-text-muted',
          )}
        >
          全部 ({events.length})
        </button>
        {CHANNELS.map((ch) => (
          <button
            key={ch}
            type="button"
            onClick={() => setFilter(ch)}
            className={cn(
              'rounded-lg px-2.5 py-1 text-xs border transition-colors font-mono',
              filter === ch ? 'bg-brand/10 border-brand text-brand' : 'border-border text-text-muted',
            )}
          >
            {ch} ({counts[ch] || 0})
          </button>
        ))}
      </div>

      <Card title="事件" extra={<Activity className="w-4 h-4 text-profit" />}>
        <div className="space-y-1 max-h-[70vh] overflow-auto font-mono text-xs">
          {filtered.length === 0 && (
            <p className="text-text-muted text-center py-12">
              {connected ? '等待事件…（可在实盘页下单或启动策略产生事件）' : 'WebSocket 未连接'}
            </p>
          )}
          {[...filtered].reverse().map((e, i) => (
            <div key={`${e.timestamp}-${i}`} className="flex items-start gap-3 py-1.5 border-b border-border/30">
              <span className="text-text-muted shrink-0 w-20">{e.timestamp?.slice(11, 19) || '--:--:--'}</span>
              <StatusBadge variant={badgeVariant(e.type)} label={e.type.replace(/_/g, ' ')} />
              <span className="text-text-secondary break-all">
                {JSON.stringify(e.data ?? e).slice(0, 200)}
              </span>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
