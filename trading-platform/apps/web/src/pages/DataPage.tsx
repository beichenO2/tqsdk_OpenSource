import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Database, HardDrive, Radio, Wifi } from 'lucide-react';
import { api } from '@/services/api';
import Card from '@/components/Card';
import StatusBadge from '@/components/StatusBadge';
import { Tabs } from '@/components/ui/Tabs';
import { cn } from '@/lib/cn';

type CacheInfo = {
  kind: string;
  path: string;
  exists: boolean;
  file_count: number;
  total_bytes: number;
  total_human?: string;
  symbol_count?: number;
  symbols?: string[];
  newest_mtime?: string | null;
  oldest_mtime?: string | null;
  files?: { path: string; bytes: number; mtime?: string }[];
};

function fmtTime(iso?: string | null) {
  if (!iso) return '—';
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

function CachePanel({ cache }: { cache: CacheInfo }) {
  const symbols = cache.symbols || [];
  return (
    <Card
      title={`${cache.kind === 'futures' ? '期货' : '加密'}缓存`}
      extra={<HardDrive className="w-4 h-4 text-text-muted" />}
    >
      <div className="space-y-3 text-sm">
        <div className="flex flex-wrap gap-x-4 gap-y-1 text-text-secondary">
          <span>
            文件 <span className="text-text-primary font-mono">{cache.file_count}</span>
          </span>
          <span>
            体积 <span className="text-text-primary font-mono">{cache.total_human || '—'}</span>
          </span>
          <span>
            品种 <span className="text-text-primary font-mono">{cache.symbol_count ?? symbols.length}</span>
          </span>
        </div>
        <div className="text-xs text-text-muted font-mono">{cache.path}</div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
          <div>
            <span className="text-text-muted">最新 </span>
            {fmtTime(cache.newest_mtime)}
          </div>
          <div>
            <span className="text-text-muted">最早 </span>
            {fmtTime(cache.oldest_mtime)}
          </div>
        </div>
        {symbols.length > 0 && (
          <div className="flex flex-wrap gap-1.5 max-h-28 overflow-auto">
            {symbols.map((s) => (
              <span
                key={s}
                className="rounded border border-border px-1.5 py-0.5 text-[11px] font-mono text-text-secondary"
              >
                {s}
              </span>
            ))}
          </div>
        )}
        {(cache.files || []).length > 0 && (
          <div className="overflow-auto max-h-48 border-t border-border/60 pt-2">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-text-muted">
                  <th className="text-left py-1">文件</th>
                  <th className="text-right py-1">大小</th>
                  <th className="text-right py-1">mtime</th>
                </tr>
              </thead>
              <tbody>
                {(cache.files || []).slice(0, 15).map((f) => (
                  <tr key={f.path} className="border-t border-border/40">
                    <td className="py-1 font-mono truncate max-w-[14rem]">{f.path}</td>
                    <td className="py-1 text-right font-mono">{(f.bytes / 1024).toFixed(0)}K</td>
                    <td className="py-1 text-right text-text-muted">{fmtTime(f.mtime)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </Card>
  );
}

export default function DataPage() {
  const [tab, setTab] = useState('overview');
  const { data, isLoading, error, refetch, isFetching } = useQuery({
    queryKey: ['platform-data'],
    queryFn: () => api.getPlatformData(),
    refetchInterval: 15_000,
  });

  const futures = (data?.caches?.futures || {}) as CacheInfo;
  const crypto = (data?.caches?.crypto || {}) as CacheInfo;
  const collector = (data?.collector || {}) as Record<string, unknown>;
  const gateway = (data?.tqsdk_gateway || {}) as Record<string, unknown>;
  const extras = (data?.extras || []) as { name: string; file_count: number }[];

  const collectorOk = !!collector.ok;
  const gatewayOk = !!gateway.ok;

  const statusLabel = useMemo(() => {
    if (!data) return '加载中';
    if (collectorOk && gatewayOk) return '链路可用';
    if (gatewayOk) return 'Gateway 在线 · Collector 离线';
    return '降级';
  }, [data, collectorOk, gatewayOk]);

  return (
    <div className="px-[3%] py-[2%] space-y-4 max-w-[96rem] mx-auto">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold text-text-primary">数据</h1>
          <p className="text-sm text-text-secondary mt-0.5">
            缓存覆盖 · freshness · data-collector / gateway
          </p>
        </div>
        <div className="flex items-center gap-3">
          <StatusBadge
            variant={collectorOk && gatewayOk ? 'success' : gatewayOk ? 'warning' : 'error'}
            label={statusLabel}
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
          {error instanceof Error ? error.message : String(error)}
        </p>
      )}

      <Tabs
        tabs={[
          { id: 'overview', label: '概览' },
          { id: 'futures', label: `期货 (${futures.file_count || 0})` },
          { id: 'crypto', label: `加密 (${crypto.file_count || 0})` },
        ]}
        active={tab}
        onChange={setTab}
      />

      {tab === 'overview' && data && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card title="Data Collector" extra={<Radio className="w-4 h-4 text-text-muted" />}>
            <div className="space-y-2 text-sm">
              <div className="flex items-center gap-2">
                <span
                  className={cn(
                    'inline-block w-2.5 h-2.5 rounded-full',
                    collectorOk ? 'bg-profit' : 'bg-loss',
                  )}
                />
                <span className="font-mono text-xs">{String(collector.url || '—')}</span>
              </div>
              {collector.status != null && (
                <pre className="text-xs bg-surface-tertiary rounded-lg p-2 overflow-auto max-h-32">
                  {JSON.stringify(collector.status, null, 2)}
                </pre>
              )}
              {typeof collector.error === 'string' && (
                <p className="text-xs text-loss">{collector.error}</p>
              )}
            </div>
          </Card>

          <Card title="TqSdk Gateway" extra={<Wifi className="w-4 h-4 text-text-muted" />}>
            <div className="space-y-2 text-sm">
              <div className="flex items-center gap-2">
                <span
                  className={cn(
                    'inline-block w-2.5 h-2.5 rounded-full',
                    gatewayOk ? 'bg-profit' : 'bg-loss',
                  )}
                />
                <span className="font-mono text-xs">{String(gateway.url || '—')}</span>
              </div>
              {typeof gateway.status_code === 'number' && (
                <p className="text-xs text-text-muted">HTTP {gateway.status_code}</p>
              )}
              {typeof gateway.error === 'string' && (
                <p className="text-xs text-loss">{gateway.error}</p>
              )}
            </div>
          </Card>

          <Card title="缓存摘要" extra={<Database className="w-4 h-4 text-text-muted" />}>
            <ul className="text-sm space-y-2">
              <li>
                期货：{futures.file_count || 0} 文件 · {futures.total_human || '—'} ·{' '}
                {futures.symbol_count ?? 0} 品种
              </li>
              <li>
                加密：{crypto.file_count || 0} 文件 · {crypto.total_human || '—'} ·{' '}
                {crypto.symbol_count ?? 0} 品种
              </li>
            </ul>
          </Card>

          {extras.length > 0 && (
            <Card title="其它 data/">
              <ul className="text-sm space-y-1.5">
                {extras.map((e) => (
                  <li key={e.name} className="flex justify-between font-mono text-xs">
                    <span>{e.name}</span>
                    <span className="text-text-muted">{e.file_count} files</span>
                  </li>
                ))}
              </ul>
            </Card>
          )}
        </div>
      )}

      {tab === 'futures' && futures.exists !== false && <CachePanel cache={futures} />}
      {tab === 'crypto' && crypto.exists !== false && <CachePanel cache={crypto} />}
    </div>
  );
}
