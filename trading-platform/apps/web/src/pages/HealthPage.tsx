import { useQuery } from '@tanstack/react-query';
import { Activity, Server, Shield, Wifi } from 'lucide-react';
import { api } from '@/services/api';
import Card from '@/components/Card';
import StatusBadge from '@/components/StatusBadge';
import { cn } from '@/lib/cn';

function StatusDot({ ok }: { ok: boolean }) {
  return <span className={cn('inline-block w-2.5 h-2.5 rounded-full', ok ? 'bg-profit' : 'bg-loss')} />;
}

export default function HealthPage() {
  const { data, isLoading, error, refetch, isFetching } = useQuery({
    queryKey: ['system-health'],
    queryFn: () => api.getSystemHealth(),
    refetchInterval: 8_000,
  });

  const components = (data?.components || {}) as Record<string, Record<string, unknown>>;
  const overallOk = data?.status === 'ok';

  return (
    <div className="px-[3%] py-[2%] space-y-4 max-w-[96rem] mx-auto">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold text-text-primary">系统健康</h1>
          <p className="text-sm text-text-secondary mt-0.5">
            API · Execution · RiskGate · TqSdk Gateway
          </p>
        </div>
        <div className="flex items-center gap-3">
          <StatusBadge
            variant={overallOk ? 'success' : 'error'}
            label={overallOk ? '全部正常' : '降级'}
          />
          <button
            type="button"
            onClick={() => refetch()}
            className="text-xs text-brand hover:underline"
          >
            {isFetching ? '刷新中…' : '刷新'}
          </button>
        </div>
      </div>

      {isLoading && <p className="text-sm text-text-muted">加载中…</p>}
      {error && (
        <p className="text-sm text-loss">
          无法获取健康状态：{error instanceof Error ? error.message : String(error)}
        </p>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card title="API" extra={<Server className="w-4 h-4 text-text-muted" />}>
          <div className="flex items-center gap-2 text-sm">
            <StatusDot ok={!!components.api?.ok} />
            <span>{String(components.api?.status || '—')}</span>
          </div>
        </Card>

        <Card title="ExecutionService" extra={<Activity className="w-4 h-4 text-text-muted" />}>
          <div className="flex items-center gap-2 text-sm">
            <StatusDot ok={!!components.execution?.ok} />
            <span>{String(components.execution?.status || '—')}</span>
          </div>
        </Card>

        <Card title="RiskGate" extra={<Shield className="w-4 h-4 text-text-muted" />}>
          <div className="space-y-2 text-sm">
            <div className="flex items-center gap-2">
              <StatusDot ok={!!components.risk_gate?.ok} />
              <span>
                live_enabled={String(components.risk_gate?.live_enabled ?? data?.live_enabled ?? false)}
              </span>
            </div>
            {Array.isArray(components.risk_gate?.limits) && (
              <p className="text-xs text-text-muted font-mono">
                闸：{(components.risk_gate.limits as string[]).join(' · ') || '无'}
              </p>
            )}
            {'reject_count' in (components.risk_gate || {}) && (
              <p className="text-xs text-text-muted">
                累计拒绝：{String(components.risk_gate.reject_count)}
              </p>
            )}
          </div>
        </Card>

        <Card title="TqSdk Gateway" extra={<Wifi className="w-4 h-4 text-text-muted" />}>
          <div className="space-y-2 text-sm">
            <div className="flex items-center gap-2">
              <StatusDot ok={!!components.tqsdk_gateway?.ok} />
              <span className="font-mono text-xs">
                {String(components.tqsdk_gateway?.url || '—')}
              </span>
            </div>
            {components.tqsdk_gateway?.body && (
              <pre className="text-[11px] text-text-muted bg-surface-tertiary rounded-lg p-2 overflow-auto max-h-24">
                {String(components.tqsdk_gateway.body)}
              </pre>
            )}
            {components.tqsdk_gateway?.error && (
              <p className="text-xs text-loss">{String(components.tqsdk_gateway.error)}</p>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
}
