import { useState, useEffect, useCallback } from 'react';
import { Play, RefreshCw, FileText, Activity, CheckCircle, XCircle, Clock, Plus } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Dialog } from '@/components/ui/Dialog';
import { Input } from '@/components/ui/Input';
import Card from '@/components/Card';
import StatusBadge from '@/components/StatusBadge';
import ResearchRunDrawer from '@/components/ResearchRunDrawer';
import { api } from '@/services/api';
import { parseApiError } from '@/lib/apiError';
import { useToast } from '@/components/ui/Toast';

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
  created_at?: number | string;
  promotion?: string;
}

const statusIcon = (s: string) => {
  switch (s) {
    case 'completed': return <CheckCircle className="w-4 h-4 text-green-500" />;
    case 'failed': return <XCircle className="w-4 h-4 text-red-500" />;
    case 'running': return <Activity className="w-4 h-4 text-blue-500 animate-pulse" />;
    default: return <Clock className="w-4 h-4 text-gray-400" />;
  }
};

function statusVariant(s: string): 'success' | 'error' | 'info' | 'neutral' | 'warning' {
  if (s === 'completed') return 'success';
  if (s === 'failed') return 'error';
  if (s === 'running') return 'info';
  return 'neutral';
}

export default function ResearchRuns() {
  const toast = useToast();
  const [runs, setRuns] = useState<ResearchRun[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedRun, setSelectedRun] = useState<ResearchRun | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [showCreate, setShowCreate] = useState(false);
  const [creating, setCreating] = useState(false);
  const [form, setForm] = useState({
    prompt: '',
    strategy_name: '',
    symbols: 'rb',
    timeframe: '5m',
  });

  const fetchRuns = useCallback(async () => {
    try {
      setLoading(true);
      const data = await api.getResearchRuns();
      setRuns((data.runs || []) as ResearchRun[]);
    } catch (e: unknown) {
      toast.error(parseApiError(e, '加载失败'));
    } finally {
      setLoading(false);
    }
  }, [toast]);

  useEffect(() => { fetchRuns(); }, [fetchRuns]);

  const openDrawer = (run: ResearchRun) => {
    setSelectedRun(run);
    setDrawerOpen(true);
  };

  const handleExecute = async (runId: string) => {
    try {
      await api.executeResearchRun(runId);
      toast.success('回测已启动');
      await fetchRuns();
    } catch (e: unknown) {
      toast.error(parseApiError(e, '执行失败'));
    }
  };

  const handleCreate = async () => {
    if (!form.prompt.trim()) return;
    try {
      setCreating(true);
      await api.createResearchRun({
        prompt: form.prompt,
        strategy_name: form.strategy_name,
        symbols: form.symbols.split(',').map(s => s.trim()).filter(Boolean),
        timeframe: form.timeframe,
      });
      setShowCreate(false);
      setForm({ prompt: '', strategy_name: '', symbols: 'rb', timeframe: '5m' });
      toast.success('Run 已创建');
      await fetchRuns();
    } catch (e: unknown) {
      toast.error(parseApiError(e, '创建失败'));
    } finally {
      setCreating(false);
    }
  };

  return (
    <div className="space-y-6 px-[3%] py-[2%]">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Research Runs</h1>
          <p className="text-gray-500 mt-1">
            8 步 pipeline · research→backtest→paper→live 晋升
          </p>
        </div>
        <div className="flex gap-2">
          <Button onClick={() => setShowCreate(true)} size="sm">
            <Plus className="w-4 h-4 mr-2" />
            New Run
          </Button>
          <Button onClick={fetchRuns} variant="secondary" size="sm">
            <RefreshCw className="w-4 h-4 mr-2" />
            Refresh
          </Button>
        </div>
      </div>

      <Dialog open={showCreate} onClose={() => setShowCreate(false)} title="Create Research Run">
        <div className="space-y-4">
          <Input
            label="Research Prompt"
            placeholder="Describe what you want to research..."
            value={form.prompt}
            onChange={e => setForm(f => ({ ...f, prompt: e.target.value }))}
          />
          <Input
            label="Strategy Name"
            placeholder="e.g. attack_defense, dual_ma"
            value={form.strategy_name}
            onChange={e => setForm(f => ({ ...f, strategy_name: e.target.value }))}
          />
          <Input
            label="Symbols"
            placeholder="rb, au, cu"
            value={form.symbols}
            onChange={e => setForm(f => ({ ...f, symbols: e.target.value }))}
          />
          <Input
            label="Timeframe"
            placeholder="5m"
            value={form.timeframe}
            onChange={e => setForm(f => ({ ...f, timeframe: e.target.value }))}
          />
          <Button onClick={handleCreate} disabled={creating || !form.prompt.trim()}>
            {creating ? 'Creating…' : 'Create'}
          </Button>
        </div>
      </Dialog>

      <ResearchRunDrawer
        run={selectedRun}
        open={drawerOpen}
        onClose={() => { setDrawerOpen(false); setSelectedRun(null); }}
        onRefresh={fetchRuns}
      />

      {loading ? (
        <div className="grid gap-4">
          {[1, 2, 3].map(i => (
            <div key={i} className="h-24 bg-gray-100 rounded-lg animate-pulse" />
          ))}
        </div>
      ) : runs.length === 0 ? (
        <Card>
          <div className="text-center py-12 text-gray-500">
            <FileText className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p className="text-lg font-medium">No research runs yet</p>
            <p className="mt-1">Create a run to start the Idea→Factor→…→Record pipeline</p>
          </div>
        </Card>
      ) : (
        <div className="grid gap-4">
          {runs.map(run => (
            <div
              key={run.run_id}
              className="cursor-pointer"
              onClick={() => openDrawer(run)}
            >
            <Card className="hover:border-blue-300 transition-colors">
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-3">
                  {statusIcon(run.status)}
                  <div>
                    <div className="font-medium">
                      {run.strategy_name || run.notes || 'Unnamed'}
                      <span className="text-gray-400 text-sm ml-2 font-mono">
                        {run.run_id.slice(0, 8)}
                      </span>
                    </div>
                    <div className="text-sm text-gray-500 mt-0.5">
                      {run.prompt.slice(0, 100)}{run.prompt.length > 100 ? '...' : ''}
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <StatusBadge variant={statusVariant(run.status)} label={run.status} />
                  {run.status === 'pending' && (
                    <Button
                      size="sm"
                      onClick={(e) => { e.stopPropagation(); handleExecute(run.run_id); }}
                    >
                      <Play className="w-3 h-3 mr-1" /> Run
                    </Button>
                  )}
                </div>
              </div>

              <div className="flex gap-4 mt-3 text-xs text-gray-500">
                <span>Symbols: {(run.symbols || []).join(', ')}</span>
                <span>Timeframe: {run.timeframe}</span>
                {(run.tags || []).length > 0 && (
                  <span>Tags: {run.tags.join(', ')}</span>
                )}
              </div>
            </Card>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
