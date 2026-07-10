import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  ArrowUpRight, Pencil, Trash2, Download, FileText, Activity,
} from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import {
  Sheet, SheetContent, SheetHeader, SheetTitle, SheetDescription,
} from '@/components/shadcn/sheet';
import { Tabs } from '@/components/ui/Tabs';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Dialog } from '@/components/ui/Dialog';
import StatusBadge from '@/components/StatusBadge';
import PipelineStepper from '@/components/PipelineStepper';
import MarkdownContent from '@/components/MarkdownContent';
import { useConfirm } from '@/components/ui/ConfirmDialog';
import { useToast } from '@/components/ui/Toast';
import { api } from '@/services/api';
import { parseApiError } from '@/lib/apiError';
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
} from 'recharts';

interface ResearchRun {
  run_id: string;
  prompt: string;
  strategy_name: string;
  symbols: string[];
  timeframe: string;
  status: string;
  metrics: Record<string, number>;
  tags: string[];
  notes?: string;
  promotion?: string;
  diagnostics?: { category: string; code: string; message: string; severity: string; timestamp?: number }[];
  validation?: { gate: string; passed: boolean; metrics: Record<string, number>; thresholds: Record<string, number> }[];
  iterations?: {
    iteration: number;
    prompt: string;
    changes: string;
    metrics_before: Record<string, number>;
    metrics_after: Record<string, number>;
    timestamp?: number;
  }[];
  artifact?: Record<string, unknown>;
  created_at?: number | string;
  updated_at?: number | string;
}

type Pipeline = {
  steps: { id: string; label: string; description: string; status: string; done: boolean }[];
  completed: number;
  total: number;
  progress: number;
  pipeline_stage: string;
  promotion: string;
  gate_passed: boolean | null;
  status?: string;
};

interface ResearchRunDrawerProps {
  run: ResearchRun | null;
  open: boolean;
  onClose: () => void;
  onRefresh: () => void;
}

function eventBadgeVariant(type: string) {
  if (type.includes('error') || type.includes('fail')) return 'error' as const;
  if (type.includes('validation') || type.includes('complete')) return 'success' as const;
  if (type.includes('progress') || type.includes('running')) return 'info' as const;
  return 'neutral' as const;
}

function toEpochSec(ts: number | string | undefined): number | null {
  if (ts == null) return null;
  if (typeof ts === 'number') return ts;
  const n = Number(ts);
  if (Number.isFinite(n)) return n;
  const d = new Date(ts).getTime();
  return Number.isFinite(d) ? d / 1000 : null;
}

function nextPromote(promo?: string) {
  if (!promo || promo === 'research') return 'backtest';
  if (promo === 'backtest') return 'paper';
  if (promo === 'paper') return 'live';
  return null;
}

export default function ResearchRunDrawer({ run, open, onClose, onRefresh }: ResearchRunDrawerProps) {
  const toast = useToast();
  const confirm = useConfirm();
  const [activeTab, setActiveTab] = useState('overview');
  const [renameOpen, setRenameOpen] = useState(false);
  const [renameValue, setRenameValue] = useState('');
  const [promoting, setPromoting] = useState(false);
  const [markdown, setMarkdown] = useState<string | null>(null);
  const [mdLoading, setMdLoading] = useState(false);

  const runId = run?.run_id ?? '';

  const { data: detail } = useQuery({
    queryKey: ['research-run', runId],
    queryFn: () => api.getResearchRun(runId) as unknown as Promise<ResearchRun>,
    enabled: open && !!runId,
  });

  const { data: pipeline } = useQuery({
    queryKey: ['research-pipeline', runId],
    queryFn: () => api.getResearchPipeline(runId) as Promise<Pipeline>,
    enabled: open && !!runId,
  });

  const fullRun = detail ?? run;

  useEffect(() => {
    if (!open) {
      setActiveTab('overview');
      setMarkdown(null);
    }
  }, [open]);

  useEffect(() => {
    if (renameOpen && fullRun) {
      setRenameValue(fullRun.strategy_name || fullRun.notes || '');
    }
  }, [renameOpen, fullRun]);

  const loadMarkdown = useCallback(async () => {
    if (!runId || fullRun?.status !== 'completed') return;
    try {
      setMdLoading(true);
      const md = await api.getResearchArtifactMarkdown(runId);
      setMarkdown(md);
    } catch (e) {
      toast.error(parseApiError(e, '加载 Markdown 产物失败'));
      setMarkdown(null);
    } finally {
      setMdLoading(false);
    }
  }, [runId, fullRun?.status, toast]);

  useEffect(() => {
    if (activeTab === 'artifacts' && fullRun?.status === 'completed' && !markdown) {
      void loadMarkdown();
    }
  }, [activeTab, fullRun?.status, markdown, loadMarkdown]);

  const synthesizedEvents = useMemo(() => {
    if (!fullRun) return [];
    const events: { timestamp: string; type: string; data: unknown }[] = [];

    if (toEpochSec(fullRun.created_at)) {
      events.push({
        timestamp: new Date(toEpochSec(fullRun.created_at)! * 1000).toISOString(),
        type: 'run_created',
        data: { run_id: fullRun.run_id, strategy: fullRun.strategy_name },
      });
    }

    for (const d of fullRun.diagnostics ?? []) {
      events.push({
        timestamp: d.timestamp
          ? new Date(d.timestamp * 1000).toISOString()
          : new Date().toISOString(),
        type: `diagnostic_${d.severity}`,
        data: { category: d.category, code: d.code, message: d.message },
      });
    }

    for (const v of fullRun.validation ?? []) {
      events.push({
        timestamp: new Date().toISOString(),
        type: v.passed ? 'validation_pass' : 'validation_fail',
        data: { gate: v.gate, metrics: v.metrics },
      });
    }

    if (fullRun.status) {
      events.push({
        timestamp: toEpochSec(fullRun.updated_at)
          ? new Date(toEpochSec(fullRun.updated_at)! * 1000).toISOString()
          : new Date().toISOString(),
        type: `status_${fullRun.status}`,
        data: { status: fullRun.status },
      });
    }

    return events.sort((a, b) => b.timestamp.localeCompare(a.timestamp));
  }, [fullRun]);

  const handleRename = async () => {
    if (!runId || !renameValue.trim()) return;
    try {
      await api.updateResearchRun(runId, { notes: renameValue.trim() });
      toast.success('已重命名');
      setRenameOpen(false);
      onRefresh();
    } catch (e) {
      toast.error(parseApiError(e, '重命名失败'));
    }
  };

  const handleDelete = async () => {
    if (!runId) return;
    const ok = await confirm({
      title: '删除研究 Run',
      description: `确认删除 ${runId}？此操作不可恢复。`,
      variant: 'destructive',
      confirmText: '确认删除',
    });
    if (!ok) return;
    try {
      await api.deleteResearchRun(runId);
      toast.success('已删除');
      onClose();
      onRefresh();
    } catch (e) {
      toast.error(parseApiError(e, '删除失败'));
    }
  };

  const handlePromote = async (target: string) => {
    if (!runId) return;
    try {
      setPromoting(true);
      const res = await api.promoteResearchRun(runId, target);
      if (res.warning) toast.warning(res.warning);
      else toast.success(`已晋升至 ${target}`);
      onRefresh();
    } catch (e) {
      toast.error(parseApiError(e, '晋升失败'));
    } finally {
      setPromoting(false);
    }
  };

  const iterationCharts = (fullRun?.iterations ?? []).map((it) => {
    const keys = new Set([
      ...Object.keys(it.metrics_before ?? {}),
      ...Object.keys(it.metrics_after ?? {}),
    ]);
    return {
      iteration: it.iteration,
      data: [...keys].map((k) => ({
        metric: k,
        before: it.metrics_before?.[k] ?? 0,
        after: it.metrics_after?.[k] ?? 0,
      })),
    };
  });

  if (!fullRun) return null;

  const promoteTarget = pipeline ? nextPromote(pipeline.promotion) : null;

  return (
    <>
      <Sheet open={open} onOpenChange={(v) => !v && onClose()}>
        <SheetContent
          side="right"
          className="w-[60vw] sm:max-w-none overflow-y-auto bg-surface-secondary border-border"
        >
          <SheetHeader className="border-b border-border pb-4">
            <div className="flex items-start justify-between gap-3 pr-8">
              <div>
                <SheetTitle className="text-text-primary">
                  {fullRun.strategy_name || fullRun.notes || 'Unnamed Run'}
                </SheetTitle>
                <SheetDescription className="font-mono text-xs mt-1">
                  {fullRun.run_id} · {fullRun.status}
                </SheetDescription>
              </div>
              <div className="flex gap-1.5 shrink-0">
                <Button size="sm" variant="secondary" onClick={() => setRenameOpen(true)}>
                  <Pencil className="w-3 h-3" />
                </Button>
                {promoteTarget && (
                  <Button
                    size="sm"
                    variant="secondary"
                    disabled={promoting}
                    onClick={() => void handlePromote(promoteTarget)}
                  >
                    <ArrowUpRight className="w-3 h-3 mr-1" />
                    → {promoteTarget}
                  </Button>
                )}
                <Button size="sm" variant="destructive" onClick={() => void handleDelete()}>
                  <Trash2 className="w-3 h-3" />
                </Button>
              </div>
            </div>
          </SheetHeader>

          <div className="py-4">
            <Tabs
              tabs={[
                { id: 'overview', label: '概览' },
                { id: 'artifacts', label: '产物' },
                { id: 'validation', label: '验证' },
                { id: 'iterations', label: '迭代' },
                { id: 'events', label: '事件' },
              ]}
              active={activeTab}
              onChange={setActiveTab}
            />

            <div className="mt-4">
              {activeTab === 'overview' && pipeline && (
                <div className="space-y-4">
                  <PipelineStepper
                    steps={pipeline.steps}
                    runFailed={fullRun.status === 'failed'}
                  />
                  <div className="flex gap-4 text-xs text-text-muted">
                    <span>进度 {pipeline.completed}/{pipeline.total}</span>
                    <span>阶段 {pipeline.pipeline_stage}</span>
                    <span>晋升 {pipeline.promotion}</span>
                    {pipeline.gate_passed !== null && (
                      <span>门控 {pipeline.gate_passed ? '通过' : '未通过'}</span>
                    )}
                  </div>
                  {fullRun.metrics && Object.keys(fullRun.metrics).length > 0 && (
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                      {Object.entries(fullRun.metrics).map(([k, v]) => (
                        <div key={k} className="rounded-lg border border-border p-2">
                          <div className="text-[10px] text-text-muted">{k}</div>
                          <div className="font-mono text-sm">{typeof v === 'number' ? v.toFixed(4) : String(v)}</div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {activeTab === 'artifacts' && (
                <div className="space-y-4">
                  {fullRun.status !== 'completed' ? (
                    <p className="text-sm text-text-muted text-center py-8">
                      Run 未完成，暂无产物可导出
                    </p>
                  ) : (
                    <>
                      <div className="flex gap-2">
                        <a
                          href={`/api/v1/research/runs/${runId}/artifact/markdown`}
                          target="_blank"
                          rel="noreferrer"
                          className="inline-flex items-center gap-1 text-xs text-brand hover:underline"
                        >
                          <Download className="w-3 h-3" /> 下载 Markdown
                        </a>
                        <button
                          type="button"
                          onClick={() => void loadMarkdown()}
                          className="inline-flex items-center gap-1 text-xs text-text-muted hover:text-text-primary"
                        >
                          <FileText className="w-3 h-3" /> 刷新预览
                        </button>
                      </div>
                      {mdLoading ? (
                        <div className="h-32 animate-pulse bg-surface-tertiary rounded-lg" />
                      ) : markdown ? (
                        <div className="prose prose-sm max-w-none text-text-primary rounded-lg border border-border p-4 bg-surface-tertiary overflow-auto max-h-[50vh]">
                          <MarkdownContent content={markdown} />
                        </div>
                      ) : (
                        <p className="text-sm text-text-muted">暂无 Markdown 产物</p>
                      )}
                      {fullRun.artifact && Object.keys(fullRun.artifact).length > 0 && (
                        <div className="rounded-lg border border-border p-3">
                          <p className="text-xs text-text-muted mb-2">Artifact JSON</p>
                          <pre className="text-[10px] font-mono overflow-auto max-h-40 text-text-secondary">
                            {JSON.stringify(fullRun.artifact, null, 2)}
                          </pre>
                        </div>
                      )}
                    </>
                  )}
                </div>
              )}

              {activeTab === 'validation' && (
                <div className="overflow-auto">
                  {(fullRun.validation ?? []).length === 0 ? (
                    <p className="text-sm text-text-muted text-center py-8">暂无验证记录</p>
                  ) : (
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="border-b border-border text-text-muted">
                          <th className="text-left py-2 px-2">Gate</th>
                          <th className="text-left py-2 px-2">结果</th>
                          <th className="text-left py-2 px-2">指标</th>
                          <th className="text-left py-2 px-2">阈值</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(fullRun.validation ?? []).map((v) => (
                          <tr key={v.gate} className="border-b border-border/50">
                            <td className="py-2 px-2 font-mono">{v.gate}</td>
                            <td className="py-2 px-2">
                              <StatusBadge
                                variant={v.passed ? 'success' : 'error'}
                                label={v.passed ? 'PASS' : 'FAIL'}
                              />
                            </td>
                            <td className="py-2 px-2 font-mono text-text-secondary">
                              {Object.entries(v.metrics ?? {}).map(([k, val]) => (
                                <div key={k}>{k}: {typeof val === 'number' ? val.toFixed(4) : val}</div>
                              ))}
                            </td>
                            <td className="py-2 px-2 font-mono text-text-muted">
                              {Object.entries(v.thresholds ?? {}).map(([k, val]) => (
                                <div key={k}>{k}: {val}</div>
                              ))}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </div>
              )}

              {activeTab === 'iterations' && (
                <div className="space-y-4">
                  {(fullRun.iterations ?? []).length === 0 ? (
                    <p className="text-sm text-text-muted text-center py-8">暂无迭代记录</p>
                  ) : (
                    iterationCharts.map((it) => (
                      <div key={it.iteration} className="rounded-lg border border-border p-3">
                        <p className="text-sm font-medium mb-2">Iteration {it.iteration}</p>
                        {it.data.length > 0 && (
                          <div className="h-24">
                            <ResponsiveContainer width="100%" height="100%">
                              <LineChart data={it.data}>
                                <XAxis dataKey="metric" tick={{ fontSize: 9 }} />
                                <YAxis tick={{ fontSize: 9 }} width={40} />
                                <Tooltip />
                                <Line type="monotone" dataKey="before" stroke="#888" dot={false} strokeWidth={1.5} />
                                <Line type="monotone" dataKey="after" stroke="var(--brand)" dot={false} strokeWidth={1.5} />
                              </LineChart>
                            </ResponsiveContainer>
                          </div>
                        )}
                      </div>
                    ))
                  )}
                </div>
              )}

              {activeTab === 'events' && (
                <div className="space-y-1 max-h-[50vh] overflow-auto font-mono text-xs">
                  {synthesizedEvents.length === 0 ? (
                    <p className="text-text-muted text-center py-8">
                      暂无事件记录（后端仅提供 SSE 流，无 REST 事件列表）
                    </p>
                  ) : (
                    synthesizedEvents.map((e, i) => (
                      <div key={`${e.timestamp}-${i}`} className="flex items-start gap-3 py-1.5 border-b border-border/30">
                        <span className="text-text-muted shrink-0 w-20">
                          {e.timestamp?.slice(11, 19) || '--:--:--'}
                        </span>
                        <StatusBadge variant={eventBadgeVariant(e.type)} label={e.type.replace(/_/g, ' ')} />
                        <span className="text-text-secondary break-all">
                          {JSON.stringify(e.data).slice(0, 200)}
                        </span>
                      </div>
                    ))
                  )}
                  <p className="text-[10px] text-text-muted pt-2 flex items-center gap-1">
                    <Activity className="w-3 h-3" />
                    事件由 diagnostics / validation / status 合成；实时 SSE 见 /research/runs/{'{id}'}/events
                  </p>
                </div>
              )}
            </div>
          </div>
        </SheetContent>
      </Sheet>

      <Dialog open={renameOpen} onClose={() => setRenameOpen(false)} title="重命名 Run">
        <div className="space-y-4">
          <Input
            label="显示名称（保存至 notes）"
            value={renameValue}
            onChange={(e) => setRenameValue(e.target.value)}
          />
          <Button onClick={() => void handleRename()} disabled={!renameValue.trim()}>
            保存
          </Button>
        </div>
      </Dialog>
    </>
  );
}
