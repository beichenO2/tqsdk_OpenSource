import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Trophy, ShieldAlert } from 'lucide-react';
import { api } from '@/services/api';
import Card from '@/components/Card';
import StatusBadge from '@/components/StatusBadge';
import { Tabs } from '@/components/ui/Tabs';
import { cn } from '@/lib/cn';

export default function OptimizerPage() {
  const [tab, setTab] = useState('champions');
  const [variant, setVariant] = useState<string | undefined>(undefined);

  const champions = useQuery({
    queryKey: ['optimizer-champions', variant],
    queryFn: () => api.getOptimizerChampions(variant, 50),
  });

  const gates = useQuery({
    queryKey: ['optimizer-gates'],
    queryFn: () => api.getOptimizerGates(),
  });

  const variants = champions.data?.variants || [];
  const entries = champions.data?.entries || [];
  const gateRows = gates.data?.gates || [];

  const passCount = useMemo(
    () => gateRows.filter((g) => g.outcome === 'pass' || g.outcome === 'PASS').length,
    [gateRows],
  );

  return (
    <div className="px-[3%] py-[2%] space-y-4 max-w-[96rem] mx-auto">
      <div>
        <h1 className="text-xl font-semibold text-text-primary">永续优化器</h1>
        <p className="text-sm text-text-secondary mt-0.5">
          冠军榜（champions/）与 gate 验证报告（results/*_gate.json）
        </p>
      </div>

      <Tabs
        tabs={[
          { id: 'champions', label: `冠军榜 (${entries.length})` },
          { id: 'gates', label: `Gate 报告 (${gateRows.length})` },
        ]}
        active={tab}
        onChange={setTab}
      />

      {tab === 'champions' && (
        <>
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() => setVariant(undefined)}
              className={cn(
                'rounded-lg px-2.5 py-1 text-xs border',
                !variant ? 'bg-brand/10 border-brand text-brand' : 'border-border text-text-muted',
              )}
            >
              全部
            </button>
            {variants.map((v) => (
              <button
                key={v}
                type="button"
                onClick={() => setVariant(v)}
                className={cn(
                  'rounded-lg px-2.5 py-1 text-xs border font-mono',
                  variant === v ? 'bg-brand/10 border-brand text-brand' : 'border-border text-text-muted',
                )}
              >
                {v}
              </button>
            ))}
          </div>

          <Card title="Champions" extra={<Trophy className="w-4 h-4 text-brand" />} noPadding>
            {champions.isLoading && <p className="p-4 text-sm text-text-muted">加载中…</p>}
            {champions.error && (
              <p className="p-4 text-sm text-loss">
                {champions.error instanceof Error ? champions.error.message : '加载失败'}
              </p>
            )}
            <div className="overflow-auto max-h-[65vh]">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-surface-secondary z-10">
                  <tr className="border-b border-border text-xs text-text-muted">
                    <th className="text-left px-4 py-2.5">#</th>
                    <th className="text-left px-3 py-2.5">变体</th>
                    <th className="text-left px-3 py-2.5">快照</th>
                    <th className="text-right px-3 py-2.5">Score</th>
                    <th className="text-right px-3 py-2.5">Round</th>
                    <th className="text-right px-4 py-2.5">保存时间</th>
                  </tr>
                </thead>
                <tbody>
                  {entries.map((e, i) => (
                    <tr key={`${e.variant}-${e.snapshot}-${i}`} className="border-b border-border/50">
                      <td className="px-4 py-2 text-text-muted">{i + 1}</td>
                      <td className="px-3 py-2 font-mono text-xs text-brand">{String(e.variant)}</td>
                      <td className="px-3 py-2 font-mono text-xs text-text-secondary">{String(e.snapshot)}</td>
                      <td className="px-3 py-2 text-right tabular-nums font-medium">
                        {Number(e.score).toFixed(4)}
                      </td>
                      <td className="px-3 py-2 text-right tabular-nums text-text-muted">{String(e.round)}</td>
                      <td className="px-4 py-2 text-right text-xs text-text-muted">
                        {String(e.saved_at || '').slice(0, 19)}
                      </td>
                    </tr>
                  ))}
                  {!champions.isLoading && entries.length === 0 && (
                    <tr>
                      <td colSpan={6} className="px-4 py-8 text-center text-text-muted">
                        暂无冠军数据
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {tab === 'gates' && (
        <Card
          title="Gate 验证"
          extra={
            <div className="flex items-center gap-2 text-xs text-text-muted">
              <ShieldAlert className="w-4 h-4" />
              pass {passCount}/{gateRows.length}
            </div>
          }
          noPadding
        >
          {gates.isLoading && <p className="p-4 text-sm text-text-muted">加载中…</p>}
          <div className="overflow-auto max-h-[65vh]">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-surface-secondary z-10">
                <tr className="border-b border-border text-xs text-text-muted">
                  <th className="text-left px-4 py-2.5">策略</th>
                  <th className="text-left px-3 py-2.5">结果</th>
                  <th className="text-right px-3 py-2.5">通过</th>
                  <th className="text-left px-3 py-2.5">主品种</th>
                  <th className="text-left px-4 py-2.5">文件</th>
                </tr>
              </thead>
              <tbody>
                {gateRows.map((g) => {
                  const outcome = String(g.outcome || 'unknown');
                  const ok = outcome.toLowerCase() === 'pass';
                  return (
                    <tr key={String(g.file)} className="border-b border-border/50">
                      <td className="px-4 py-2 font-medium">{String(g.name)}</td>
                      <td className="px-3 py-2">
                        <StatusBadge variant={ok ? 'success' : 'error'} label={outcome} />
                      </td>
                      <td className="px-3 py-2 text-right tabular-nums">
                        {Number(g.passed)}/{Number(g.total)}
                      </td>
                      <td className="px-3 py-2 font-mono text-xs">{String(g.primary_symbol || '—')}</td>
                      <td className="px-4 py-2 font-mono text-[11px] text-text-muted truncate max-w-[20rem]">
                        {String(g.file)}
                      </td>
                    </tr>
                  );
                })}
                {!gates.isLoading && gateRows.length === 0 && (
                  <tr>
                    <td colSpan={5} className="px-4 py-8 text-center text-text-muted">
                      暂无 gate 报告
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
