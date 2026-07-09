import { useMemo, useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { FlaskConical, LineChart } from 'lucide-react';
import {
  Line,
  LineChart as ReLineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { api } from '@/services/api';
import Card from '@/components/Card';
import StatusBadge from '@/components/StatusBadge';
import { cn } from '@/lib/cn';

type FactorMeta = {
  name: string;
  category: string;
  description: string;
  params?: Record<string, unknown>;
  output_columns?: string[];
};

export default function FactorsPage() {
  const [category, setCategory] = useState<string | undefined>();
  const [selected, setSelected] = useState<string | null>(null);
  const [symbol, setSymbol] = useState('rb');
  const [picked, setPicked] = useState<string[]>([]);

  const list = useQuery({
    queryKey: ['factors', category],
    queryFn: () => api.listFactors(category),
  });

  const factors = (list.data?.factors || []) as FactorMeta[];
  const categories = (list.data?.categories || []) as string[];

  const filtered = useMemo(
    () => (category ? factors.filter((f) => f.category === category) : factors),
    [factors, category],
  );

  const analyze = useMutation({
    mutationFn: () =>
      api.analyzeFactors({
        symbol,
        factor_names: picked.length ? picked : selected ? [selected] : [],
        limit: 400,
        horizon: 1,
      }),
  });

  const csAnalyze = useMutation({
    mutationFn: () =>
      api.analyzeCrossSection({
        factor_name: selected!,
        limit: 250,
        horizon: 1,
        quantiles: 5,
      }),
  });

  const evolve = useMutation({
    mutationFn: () =>
      api.evolveFactors({
        symbol,
        n_proposals: 5,
        limit: 300,
        use_llm: true,
      }),
  });

  const compute = useMutation({
    mutationFn: (names: string[]) =>
      api.computeFactors({ symbol, factor_names: names, limit: 300 }),
  });

  const report = analyze.data?.reports?.[0] as
    | {
        name: string;
        summary?: Record<string, number | null>;
        decay?: { horizon: number; ic_mean?: number | null; ir?: number | null }[];
        ic_series?: { t: string; v: number }[];
      }
    | undefined;

  const csSummary = csAnalyze.data?.summary;
  const qret = csAnalyze.data?.quantile_returns;
  const csIcChart = (csAnalyze.data?.ic_series || []).map((p) => ({
    t: p.t.slice(5, 16),
    ic: Number(p.v.toFixed(4)),
  }));

  const icChart = (report?.ic_series || []).map((p) => ({
    t: p.t.slice(5, 16),
    ic: Number(p.v.toFixed(4)),
  }));

  function togglePick(name: string) {
    setPicked((prev) =>
      prev.includes(name) ? prev.filter((x) => x !== name) : [...prev, name].slice(0, 8),
    );
  }

  return (
    <div className="px-[3%] py-[2%] space-y-4 max-w-[96rem] mx-auto">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold text-text-primary">因子库</h1>
          <p className="text-sm text-text-secondary mt-0.5">
            Alpha158/WQ101 + 单品种 IC + Alphalens 截面 IC/分位收益
          </p>
        </div>
        <div className="flex items-center gap-2">
          <label className="text-xs text-text-muted">
            品种
            <input
              value={symbol}
              onChange={(e) => setSymbol(e.target.value.trim())}
              className="ml-2 rounded-lg border border-border bg-surface-tertiary px-2 py-1 text-sm font-mono w-28"
            />
          </label>
          <button
            type="button"
            disabled={!selected && picked.length === 0}
            onClick={() => analyze.mutate()}
            className="rounded-lg bg-brand/90 text-white text-xs px-3 py-1.5 disabled:opacity-40"
          >
            {analyze.isPending ? '分析中…' : '单品种 IC'}
          </button>
          <button
            type="button"
            disabled={!selected}
            onClick={() => csAnalyze.mutate()}
            className="rounded-lg border border-brand text-brand text-xs px-3 py-1.5 disabled:opacity-40"
          >
            {csAnalyze.isPending ? '截面中…' : '截面 IC'}
          </button>
          <button
            type="button"
            onClick={() => evolve.mutate()}
            className="rounded-lg border border-border text-text-secondary text-xs px-3 py-1.5"
          >
            {evolve.isPending ? '进化中…' : 'LLM 进化一轮'}
          </button>
        </div>
      </div>

      <div className="flex flex-wrap gap-2">
        <button
          type="button"
          onClick={() => setCategory(undefined)}
          className={cn(
            'rounded-lg px-2.5 py-1 text-xs border',
            !category ? 'bg-brand/10 border-brand text-brand' : 'border-border text-text-muted',
          )}
        >
          全部 ({list.data?.count ?? '…'})
        </button>
        {categories.map((c) => (
          <button
            key={c}
            type="button"
            onClick={() => setCategory(c)}
            className={cn(
              'rounded-lg px-2.5 py-1 text-xs border',
              category === c ? 'bg-brand/10 border-brand text-brand' : 'border-border text-text-muted',
            )}
          >
            {c}
          </button>
        ))}
      </div>

      {list.isLoading && <p className="text-sm text-text-muted">加载因子库…</p>}
      {list.error && (
        <p className="text-sm text-loss">
          {list.error instanceof Error ? list.error.message : '加载失败'}
        </p>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-[1.2fr_1fr] gap-4">
        <Card title="因子列表" extra={<FlaskConical className="w-4 h-4 text-text-muted" />} noPadding>
          <div className="overflow-auto max-h-[62vh]">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-surface-secondary z-10">
                <tr className="border-b border-border text-xs text-text-muted">
                  <th className="text-left px-3 py-2">选</th>
                  <th className="text-left px-2 py-2">名称</th>
                  <th className="text-left px-2 py-2">类别</th>
                  <th className="text-left px-3 py-2">说明</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((f) => (
                  <tr
                    key={f.name}
                    className={cn(
                      'border-b border-border/40 cursor-pointer',
                      selected === f.name && 'bg-brand/5',
                    )}
                    onClick={() => {
                      setSelected(f.name);
                      if (!picked.includes(f.name)) setPicked([f.name]);
                    }}
                  >
                    <td className="px-3 py-2" onClick={(e) => e.stopPropagation()}>
                      <input
                        type="checkbox"
                        checked={picked.includes(f.name)}
                        onChange={() => togglePick(f.name)}
                      />
                    </td>
                    <td className="px-2 py-2 font-mono text-xs text-brand">{f.name}</td>
                    <td className="px-2 py-2 text-xs text-text-muted">{f.category}</td>
                    <td className="px-3 py-2 text-xs text-text-secondary truncate max-w-[20rem]">
                      {f.description || '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <div className="space-y-4">
          <Card title={selected ? `详情 · ${selected}` : '详情'}>
            {!selected && <p className="text-sm text-text-muted">选择因子查看元数据与 IC。</p>}
            {selected && (
              <div className="space-y-3 text-sm">
                {(() => {
                  const f = factors.find((x) => x.name === selected);
                  if (!f) return null;
                  return (
                    <>
                      <p className="text-text-secondary">{f.description || '无描述'}</p>
                      <p className="text-xs font-mono text-text-muted">
                        outputs: {(f.output_columns || []).join(', ') || '—'}
                      </p>
                      <p className="text-xs font-mono text-text-muted">
                        params: {JSON.stringify(f.params || {})}
                      </p>
                      <button
                        type="button"
                        className="text-xs text-brand hover:underline"
                        onClick={() => compute.mutate([selected])}
                      >
                        {compute.isPending ? '计算中…' : '预览因子值（尾部）'}
                      </button>
                      {compute.data?.factors?.[selected] && (
                        <p className="text-xs font-mono">
                          last={String((compute.data.factors[selected] as { last?: number }).last)} ·
                          n=
                          {String((compute.data.factors[selected] as { n?: number }).n)}
                        </p>
                      )}
                    </>
                  );
                })()}
              </div>
            )}
          </Card>

          <Card title="IC / IR（单品种）" extra={<LineChart className="w-4 h-4 text-text-muted" />}>
            {analyze.isError && (
              <p className="text-sm text-loss">
                {analyze.error instanceof Error ? analyze.error.message : '分析失败'}
              </p>
            )}
            {!report && !analyze.isPending && (
              <p className="text-sm text-text-muted">勾选因子后点「单品种 IC」。</p>
            )}
            {report?.summary && (
              <div className="space-y-3">
                <div className="flex flex-wrap gap-2">
                  <StatusBadge
                    variant="info"
                    label={`IC ${report.summary.ic_mean?.toFixed?.(4) ?? '—'}`}
                  />
                  <StatusBadge
                    variant="neutral"
                    label={`IR ${report.summary.ir?.toFixed?.(3) ?? '—'}`}
                  />
                  <StatusBadge
                    variant="neutral"
                    label={`n=${report.summary.n ?? 0}`}
                  />
                </div>
                {icChart.length > 0 && (
                  <div className="h-40">
                    <ResponsiveContainer width="100%" height="100%">
                      <ReLineChart data={icChart}>
                        <XAxis dataKey="t" hide />
                        <YAxis domain={[-1, 1]} width={32} tick={{ fontSize: 10 }} />
                        <Tooltip />
                        <Line type="monotone" dataKey="ic" stroke="var(--color-brand, #3b82f6)" dot={false} />
                      </ReLineChart>
                    </ResponsiveContainer>
                  </div>
                )}
                {report.decay && (
                  <div className="overflow-auto">
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="text-text-muted">
                          <th className="text-left py-1">H</th>
                          <th className="text-right py-1">IC</th>
                          <th className="text-right py-1">IR</th>
                        </tr>
                      </thead>
                      <tbody>
                        {report.decay.map((d) => (
                          <tr key={d.horizon} className="border-t border-border/40 font-mono">
                            <td className="py-1">{d.horizon}</td>
                            <td className="py-1 text-right">{d.ic_mean?.toFixed?.(4) ?? '—'}</td>
                            <td className="py-1 text-right">{d.ir?.toFixed?.(3) ?? '—'}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
                {analyze.data?.dedupe && (
                  <p className="text-xs text-text-muted">
                    去重保留 {(analyze.data.dedupe as { kept?: string[] }).kept?.join(', ') || '—'}
                    {(analyze.data.dedupe as { dropped?: string[] }).dropped?.length
                      ? ` · 剔除 ${(analyze.data.dedupe as { dropped: string[] }).dropped.join(', ')}`
                      : ''}
                  </p>
                )}
              </div>
            )}
          </Card>

          <Card title="截面 IC / 分位收益（Alphalens）">
            {csAnalyze.isError && (
              <p className="text-sm text-loss">
                {csAnalyze.error instanceof Error ? csAnalyze.error.message : '截面分析失败'}
              </p>
            )}
            {!csSummary && !csAnalyze.isPending && (
              <p className="text-sm text-text-muted">选中因子后点「截面 IC」（多品种面板）。</p>
            )}
            {csSummary && (
              <div className="space-y-3">
                <div className="flex flex-wrap gap-2">
                  <StatusBadge
                    variant="info"
                    label={`CS-IC ${Number(csSummary.ic_mean ?? 0).toFixed(4)}`}
                  />
                  <StatusBadge
                    variant="neutral"
                    label={`IR ${csSummary.ir != null ? Number(csSummary.ir).toFixed(3) : '—'}`}
                  />
                  <StatusBadge
                    variant="neutral"
                    label={`assets=${csAnalyze.data?.n_assets ?? 0}`}
                  />
                </div>
                {csIcChart.length > 0 && (
                  <div className="h-36">
                    <ResponsiveContainer width="100%" height="100%">
                      <ReLineChart data={csIcChart}>
                        <XAxis dataKey="t" hide />
                        <YAxis domain={[-1, 1]} width={32} tick={{ fontSize: 10 }} />
                        <Tooltip />
                        <Line type="monotone" dataKey="ic" stroke="#22c55e" dot={false} />
                      </ReLineChart>
                    </ResponsiveContainer>
                  </div>
                )}
                {qret?.mean_returns && (
                  <table className="w-full text-xs font-mono">
                    <thead>
                      <tr className="text-text-muted">
                        <th className="text-left py-1">分位</th>
                        <th className="text-right py-1">均值收益</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(qret.mean_returns).map(([k, v]) => (
                        <tr key={k} className="border-t border-border/40">
                          <td className="py-1">{k}</td>
                          <td className="py-1 text-right">
                            {v == null ? '—' : Number(v).toFixed(6)}
                          </td>
                        </tr>
                      ))}
                      <tr className="border-t border-border">
                        <td className="py-1 text-brand">Long-Short</td>
                        <td className="py-1 text-right text-brand">
                          {qret.long_short == null ? '—' : Number(qret.long_short).toFixed(6)}
                        </td>
                      </tr>
                    </tbody>
                  </table>
                )}
                {csAnalyze.data?.symbols_used && (
                  <p className="text-[11px] text-text-muted font-mono truncate">
                    {csAnalyze.data.symbols_used.join(', ')}
                  </p>
                )}
              </div>
            )}
          </Card>

          <Card title="因子进化（bandit）">
            {evolve.isError && (
              <p className="text-sm text-loss">
                {evolve.error instanceof Error ? evolve.error.message : '进化失败'}
              </p>
            )}
            {!evolve.data && !evolve.isPending && (
              <p className="text-sm text-text-muted">
                点「LLM 进化一轮」：Thompson bandit 分配挖因子/调参预算；LLM 不可用时模板变异。
              </p>
            )}
            {evolve.data && (
              <div className="space-y-2 text-sm">
                <div className="flex flex-wrap gap-2">
                  <StatusBadge variant="info" label={`arm=${evolve.data.arm}`} />
                  <StatusBadge
                    variant="neutral"
                    label={`valid=${evolve.data.n_valid}`}
                  />
                </div>
                {evolve.data.best && (
                  <div className="font-mono text-xs space-y-1">
                    <p className="text-brand break-all">
                      best: {String((evolve.data.best as { expr?: string }).expr || '')}
                    </p>
                    <p className="text-text-muted">
                      score=
                      {String((evolve.data.best as { score?: number }).score ?? '—')}
                      {' · '}IC=
                      {String((evolve.data.best as { ic_mean?: number }).ic_mean ?? '—')}
                      {' · '}src=
                      {String((evolve.data.best as { source?: string }).source ?? '')}
                    </p>
                  </div>
                )}
                <ul className="text-[11px] font-mono text-text-muted max-h-28 overflow-auto space-y-0.5">
                  {(evolve.data.candidates || []).slice(0, 8).map((c, i) => (
                    <li key={i} className="truncate">
                      {(c as { source?: string }).source}:{' '}
                      {(c as { expr?: string }).expr}
                      {(c as { error?: string }).error
                        ? ` ✗ ${(c as { error: string }).error}`
                        : ` → ${String((c as { score?: number }).score ?? '—')}`}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}
