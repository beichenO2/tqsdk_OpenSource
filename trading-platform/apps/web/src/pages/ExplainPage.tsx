import { useMemo, useState, type ReactNode } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
} from 'recharts';
import { Search, GitBranch } from 'lucide-react';
import Card from '@/components/Card';
import { Button } from '@/components/ui/Button';
import StatusBadge from '@/components/StatusBadge';
import KLineChart from '@/components/charts/KLineChart';
import { useToast } from '@/components/ui/Toast';
import { api } from '@/services/api';
import { parseApiError } from '@/lib/apiError';
import { cn } from '@/lib/cn';

interface TimelineEntry {
  timestamp: string;
  event_type: string;
  data: Record<string, unknown>;
  metadata: Record<string, unknown>;
}

interface TimelineView {
  trade_id: string;
  symbol: string;
  entries: TimelineEntry[];
}

interface DecisionNode {
  id: string;
  label: string;
  event_type: string;
  children: DecisionNode[];
}

function renderGraphNode(node: DecisionNode, depth = 0): ReactNode {
  return (
    <div key={node.id} style={{ paddingLeft: depth * 16 }} className="py-0.5">
      <span className="text-xs font-mono text-text-secondary">{node.label}</span>
      {node.children?.map((child) => renderGraphNode(child, depth + 1))}
    </div>
  );
}

export default function ExplainPage() {
  const toast = useToast();
  const [symbol, setSymbol] = useState('rb');
  const [start, setStart] = useState(() => {
    const d = new Date();
    d.setDate(d.getDate() - 7);
    return d.toISOString().slice(0, 16);
  });
  const [end, setEnd] = useState(() => new Date().toISOString().slice(0, 16));
  const [searchKey, setSearchKey] = useState('');
  const [selectedTradeId, setSelectedTradeId] = useState<string | null>(null);
  const [selectedEntry, setSelectedEntry] = useState<TimelineEntry | null>(null);
  const [expandedEntry, setExpandedEntry] = useState<string | null>(null);

  const { data: timelines = [], isLoading, isError, error, refetch } = useQuery({
    queryKey: ['explain-timelines', symbol, start, end],
    queryFn: async () => {
      const startIso = new Date(start).toISOString();
      const endIso = new Date(end).toISOString();
      return api.listExplainTimelines(symbol, startIso, endIso) as Promise<TimelineView[]>;
    },
    enabled: !!symbol && !!start && !!end,
    retry: false,
  });

  const { data: factors } = useQuery({
    queryKey: ['explain-factors', selectedTradeId],
    queryFn: () => api.getExplainFactors(selectedTradeId!),
    enabled: !!selectedTradeId,
  });

  const { data: graph } = useQuery({
    queryKey: ['explain-graph', selectedTradeId],
    queryFn: () => api.getExplainGraph(selectedTradeId!) as Promise<{
      trade_id: string;
      symbol: string;
      root: DecisionNode;
    }>,
    enabled: !!selectedTradeId,
  });

  const activeTimeline = useMemo(() => {
    if (!timelines.length) return null;
    if (selectedTradeId) {
      return timelines.find((t) => t.trade_id === selectedTradeId) ?? timelines[0];
    }
    return timelines[0];
  }, [timelines, selectedTradeId]);

  const highlightTime = useMemo(() => {
    if (!selectedEntry?.timestamp) return undefined;
    return Math.floor(new Date(selectedEntry.timestamp).getTime() / 1000);
  }, [selectedEntry]);

  const factorBars = useMemo(() => {
    return (factors?.factors ?? []).map((f) => ({
      name: f.factor,
      weight: Math.abs(f.weight),
      raw: f.weight,
      source: f.source,
    }));
  }, [factors]);

  const handleSearch = () => {
    setSearchKey(`${symbol}-${start}-${end}`);
    void refetch().catch((e) => toast.error(parseApiError(e, '查询失败')));
  };

  return (
    <div className="px-[3%] py-[2%] space-y-4 max-w-[96rem] mx-auto">
      <div>
        <h1 className="text-xl font-semibold text-text-primary">交易证据链</h1>
        <p className="text-sm text-text-muted mt-0.5">为什么下了这单 — 决策时间线 + 因子贡献</p>
      </div>

      <Card>
        <div className="flex flex-wrap items-end gap-3">
          <div>
            <label className="block text-xs text-text-muted mb-1">品种</label>
            <input
              value={symbol}
              onChange={(e) => setSymbol(e.target.value)}
              className="bg-surface-tertiary border border-border rounded-lg px-3 py-1.5 text-sm font-mono w-32"
            />
          </div>
          <div>
            <label className="block text-xs text-text-muted mb-1">开始</label>
            <input
              type="datetime-local"
              value={start}
              onChange={(e) => setStart(e.target.value)}
              className="bg-surface-tertiary border border-border rounded-lg px-3 py-1.5 text-sm"
            />
          </div>
          <div>
            <label className="block text-xs text-text-muted mb-1">结束</label>
            <input
              type="datetime-local"
              value={end}
              onChange={(e) => setEnd(e.target.value)}
              className="bg-surface-tertiary border border-border rounded-lg px-3 py-1.5 text-sm"
            />
          </div>
          <Button size="sm" onClick={handleSearch}>
            <Search className="w-3.5 h-3.5 mr-1" />
            查询
          </Button>
        </div>
      </Card>

      {isError && (
        <Card>
          <p className="text-sm text-loss">{parseApiError(error, '无证据链数据')}</p>
          <p className="text-xs text-text-muted mt-1">数据库中可能尚无该品种/时间段的证据链记录</p>
        </Card>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
        <div className="xl:col-span-2 space-y-4">
          <Card title="决策时间线">
            {isLoading ? (
              <div className="h-48 animate-pulse bg-surface-tertiary rounded-lg" />
            ) : !activeTimeline || activeTimeline.entries.length === 0 ? (
              <p className="text-sm text-text-muted text-center py-12">
                {searchKey ? '该条件下无交易证据链' : '设置筛选条件后点击查询'}
              </p>
            ) : (
              <div className="space-y-0">
                {timelines.length > 1 && (
                  <div className="flex flex-wrap gap-1 mb-3">
                    {timelines.map((t) => (
                      <button
                        key={t.trade_id}
                        type="button"
                        onClick={() => {
                          setSelectedTradeId(t.trade_id);
                          setSelectedEntry(null);
                        }}
                        className={cn(
                          'text-xs font-mono px-2 py-1 rounded border',
                          selectedTradeId === t.trade_id || (!selectedTradeId && t === timelines[0])
                            ? 'border-brand bg-brand/10 text-brand'
                            : 'border-border text-text-muted',
                        )}
                      >
                        {t.trade_id.slice(0, 8)}
                      </button>
                    ))}
                  </div>
                )}
                <div className="relative pl-4 border-l-2 border-border space-y-3">
                  {activeTimeline.entries.map((entry, i) => {
                    const key = `${entry.timestamp}-${entry.event_type}-${i}`;
                    const isExpanded = expandedEntry === key;
                    return (
                      <div key={key} className="relative">
                        <span className="absolute -left-[21px] top-1.5 w-2.5 h-2.5 rounded-full bg-brand border-2 border-surface-secondary" />
                        <button
                          type="button"
                          onClick={() => {
                            setExpandedEntry(isExpanded ? null : key);
                            setSelectedEntry(entry);
                            setSelectedTradeId(activeTimeline.trade_id);
                          }}
                          className={cn(
                            'w-full text-left rounded-lg border px-3 py-2 transition-colors',
                            selectedEntry === entry
                              ? 'border-brand bg-brand/5'
                              : 'border-border hover:border-brand/40',
                          )}
                        >
                          <div className="flex items-center gap-2">
                            <span className="text-[10px] font-mono text-text-muted">
                              {entry.timestamp?.slice(11, 19)}
                            </span>
                            <StatusBadge variant="info" label={entry.event_type} />
                          </div>
                          {isExpanded && (
                            <div className="mt-2 text-xs text-text-secondary space-y-1">
                              {entry.event_type === 'signal' && (
                                <div>
                                  <p className="text-text-muted mb-1">因子快照</p>
                                  {Object.entries(entry.data).map(([k, v]) => (
                                    <div key={k} className="font-mono">{k}: {String(v)}</div>
                                  ))}
                                </div>
                              )}
                              {entry.event_type.includes('risk') && (
                                <p>风控: {JSON.stringify(entry.data)}</p>
                              )}
                              {entry.event_type.includes('order') && (
                                <p>订单: {JSON.stringify(entry.data)}</p>
                              )}
                              {!['signal'].includes(entry.event_type) && (
                                <pre className="text-[10px] overflow-auto">{JSON.stringify(entry.data, null, 2)}</pre>
                              )}
                            </div>
                          )}
                        </button>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </Card>

          {graph?.root && (
            <Card title="因果图（缩进列表）" extra={<GitBranch className="w-4 h-4 text-text-muted" />}>
              <div className="max-h-48 overflow-auto">
                {renderGraphNode(graph.root)}
              </div>
            </Card>
          )}
        </div>

        <div className="space-y-4">
          {activeTimeline && (
            <KLineChart
              symbol={activeTimeline.symbol || symbol}
              height={220}
              limit={120}
              highlightTime={highlightTime}
            />
          )}

          <Card title="因子贡献">
            {factorBars.length === 0 ? (
              <p className="text-sm text-text-muted text-center py-8">
                选中时间线节点查看因子贡献
              </p>
            ) : (
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={factorBars} layout="vertical" margin={{ left: 8, right: 8 }}>
                    <XAxis type="number" tick={{ fontSize: 9 }} />
                    <YAxis type="category" dataKey="name" width={72} tick={{ fontSize: 9 }} />
                    <Tooltip
                      formatter={(v, _n, item) => {
                        const payload = item?.payload as { raw?: number; source?: string } | undefined;
                        const raw = payload?.raw ?? Number(v);
                        return [`${raw.toFixed(4)} (${payload?.source ?? ''})`, 'weight'];
                      }}
                    />
                    <Bar dataKey="weight" radius={[0, 4, 4, 0]}>
                      {factorBars.map((entry, i) => (
                        <Cell key={i} fill={entry.raw >= 0 ? 'var(--profit)' : 'var(--loss)'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}
