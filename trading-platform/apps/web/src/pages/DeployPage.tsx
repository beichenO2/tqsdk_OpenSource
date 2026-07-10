import { useMemo, useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Rocket, RotateCcw, History, ChevronRight } from 'lucide-react';
import Card from '@/components/Card';
import { Button } from '@/components/ui/Button';
import StatusBadge from '@/components/StatusBadge';
import DeployDiffTable from '@/components/DeployDiffTable';
import { useConfirm } from '@/components/ui/ConfirmDialog';
import { useToast } from '@/components/ui/Toast';
import { api } from '@/services/api';
import { parseApiError } from '@/lib/apiError';

interface OptunaEntry {
  file: string;
  strategy: string;
  market: string;
  best_score?: number;
  sharpe?: number;
  total_return?: number;
  best_params: Record<string, unknown>;
  n_trials: number;
}

interface DeployRecord {
  strategy_name: string;
  source: string;
  note: string;
  old_params: Record<string, unknown>;
  new_params: Record<string, unknown>;
  deployed_at: string;
}

export default function DeployPage() {
  const toast = useToast();
  const confirm = useConfirm();
  const qc = useQueryClient();

  const [strategyName, setStrategyName] = useState('futures_dual_ma');
  const [selectedOptuna, setSelectedOptuna] = useState<OptunaEntry | null>(null);

  const { data: currentParams, isLoading: paramsLoading } = useQuery({
    queryKey: ['deploy-params', strategyName],
    queryFn: () => api.getDeployParams(strategyName),
    enabled: !!strategyName,
  });

  const { data: optunaList = [], isLoading: optunaLoading } = useQuery({
    queryKey: ['deploy-optuna'],
    queryFn: () => api.getOptunaBest(undefined, 20) as unknown as Promise<OptunaEntry[]>,
  });

  const { data: history = [], isLoading: historyLoading } = useQuery({
    queryKey: ['deploy-history'],
    queryFn: () => api.getDeployHistory(30) as unknown as Promise<DeployRecord[]>,
  });

  const applyMutation = useMutation({
    mutationFn: (studyName: string) => {
      const name = studyName.replace(/^optuna_/, '').replace(/\.json$/, '');
      return api.applyOptunaResult(name, strategyName);
    },
    onSuccess: () => {
      toast.success('参数已应用');
      void qc.invalidateQueries({ queryKey: ['deploy-params'] });
      void qc.invalidateQueries({ queryKey: ['deploy-history'] });
    },
    onError: (e) => toast.error(parseApiError(e, '应用失败')),
  });

  const rollbackMutation = useMutation({
    mutationFn: (name: string) => api.rollbackDeploy(name),
    onSuccess: () => {
      toast.success('已回滚到上一版本');
      void qc.invalidateQueries({ queryKey: ['deploy-params'] });
      void qc.invalidateQueries({ queryKey: ['deploy-history'] });
    },
    onError: (e) => toast.error(parseApiError(e, '回滚失败')),
  });

  const candidate = selectedOptuna?.best_params ?? {};
  const activeParams = currentParams?.params ?? {};

  const filteredOptuna = useMemo(() => {
    if (!strategyName) return optunaList;
    return optunaList.filter(
      (o) => !o.strategy || o.strategy === strategyName || strategyName.includes(o.strategy),
    );
  }, [optunaList, strategyName]);

  const handleApply = async () => {
    if (!selectedOptuna) {
      toast.warning('请先选择 Optuna 候选参数');
      return;
    }
    const ok = await confirm({
      title: '应用参数到策略',
      description: `将 ${selectedOptuna.file} 的最优参数部署到 ${strategyName}。`,
      confirmText: '确认应用',
    });
    if (!ok) return;
    const studyName = selectedOptuna.file.replace(/^optuna_/, '').replace(/\.json$/, '');
    applyMutation.mutate(studyName);
  };

  const handleRollback = async (record: DeployRecord) => {
    const ok = await confirm({
      title: '回滚到此版本',
      description: `将 ${record.strategy_name} 回滚到 ${record.deployed_at} 之前的参数。`,
      variant: 'warning',
      confirmText: '确认回滚',
    });
    if (!ok) return;
    rollbackMutation.mutate(record.strategy_name);
  };

  return (
    <div className="px-[3%] py-[2%] space-y-6 max-w-[96rem] mx-auto">
      <div>
        <h1 className="text-xl font-semibold text-text-primary">部署控制台</h1>
        <p className="text-sm text-text-muted mt-0.5">
          Optuna 最优参数 → 策略部署 · 历史回滚
        </p>
      </div>

      <div className="flex items-center gap-3">
        <label className="text-xs text-text-muted">策略名</label>
        <input
          value={strategyName}
          onChange={(e) => setStrategyName(e.target.value)}
          className="bg-surface-tertiary border border-border rounded-lg px-3 py-1.5 text-sm font-mono focus:outline-none focus:border-brand w-64"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Card title="当前生效参数">
          {paramsLoading ? (
            <div className="h-32 animate-pulse bg-surface-tertiary rounded-lg" />
          ) : (
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <span className="text-sm font-mono text-text-primary">{strategyName}</span>
                <StatusBadge
                  variant={currentParams?.deployed ? 'success' : 'neutral'}
                  label={currentParams?.deployed ? '已部署' : '未部署'}
                />
              </div>
              {Object.keys(activeParams).length === 0 ? (
                <p className="text-sm text-text-muted">暂无部署参数</p>
              ) : (
                <div className="rounded-lg border border-border overflow-auto max-h-64">
                  <table className="w-full text-xs">
                    <tbody>
                      {Object.entries(activeParams).map(([k, v]) => (
                        <tr key={k} className="border-b border-border/40">
                          <td className="py-1.5 px-2 font-mono text-text-muted">{k}</td>
                          <td className="py-1.5 px-2 font-mono text-text-primary">{String(v)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </Card>

        <Card title="候选参数（Optuna）">
          {optunaLoading ? (
            <div className="h-32 animate-pulse bg-surface-tertiary rounded-lg" />
          ) : filteredOptuna.length === 0 ? (
            <p className="text-sm text-text-muted">models/ 目录下暂无 Optuna 结果</p>
          ) : (
            <div className="space-y-3">
              <select
                value={selectedOptuna?.file ?? ''}
                onChange={(e) => {
                  const entry = filteredOptuna.find((o) => o.file === e.target.value) ?? null;
                  setSelectedOptuna(entry);
                }}
                className="w-full bg-surface-tertiary border border-border rounded-lg px-3 py-2 text-sm font-mono focus:outline-none focus:border-brand"
              >
                <option value="">选择 Optuna 结果…</option>
                {filteredOptuna.map((o) => (
                  <option key={o.file} value={o.file}>
                    {o.file} · score={o.best_score ?? 'N/A'} · {o.market}
                  </option>
                ))}
              </select>

              {selectedOptuna && (
                <>
                  <DeployDiffTable current={activeParams} candidate={candidate} />
                  <Button onClick={() => void handleApply()} loading={applyMutation.isPending}>
                    <Rocket className="w-3.5 h-3.5 mr-1.5" />
                    应用 Optuna 参数
                  </Button>
                </>
              )}
            </div>
          )}
        </Card>
      </div>

      <Card title="部署历史" extra={<History className="w-4 h-4 text-text-muted" />}>
        {historyLoading ? (
          <div className="h-24 animate-pulse bg-surface-tertiary rounded-lg" />
        ) : history.length === 0 ? (
          <p className="text-sm text-text-muted text-center py-8">暂无部署记录</p>
        ) : (
          <div className="space-y-2 max-h-[50vh] overflow-auto">
            {history.map((record, i) => (
              <div
                key={`${record.deployed_at}-${i}`}
                className="flex items-start justify-between gap-3 rounded-lg border border-border px-3 py-2"
              >
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2 text-sm">
                    <span className="font-mono text-text-primary">{record.strategy_name}</span>
                    <StatusBadge variant="info" label={record.source} />
                  </div>
                  <p className="text-xs text-text-muted mt-0.5">
                    {record.deployed_at?.slice(0, 19).replace('T', ' ')}
                    {record.note && ` · ${record.note}`}
                  </p>
                  <p className="text-[10px] font-mono text-text-secondary mt-1 truncate">
                    {Object.keys(record.new_params ?? {}).slice(0, 4).map((k) => `${k}=${record.new_params[k]}`).join(', ')}
                  </p>
                </div>
                <Button
                  size="sm"
                  variant="secondary"
                  onClick={() => void handleRollback(record)}
                  loading={rollbackMutation.isPending}
                >
                  <RotateCcw className="w-3 h-3 mr-1" />
                  回滚
                </Button>
              </div>
            ))}
          </div>
        )}
        <p className="text-[10px] text-text-muted mt-3 flex items-center gap-1">
          <ChevronRight className="w-3 h-3" />
          回滚恢复上一版本 old_params（后端无按条目精确回滚）
        </p>
      </Card>
    </div>
  );
}
