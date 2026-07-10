import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
} from 'recharts';
import { Brain, Play, ChevronDown, ChevronUp, TrendingUp, TrendingDown } from 'lucide-react';
import Card from '@/components/Card';
import { Button } from '@/components/ui/Button';
import StatusBadge from '@/components/StatusBadge';
import { useToast } from '@/components/ui/Toast';
import { api } from '@/services/api';
import { parseApiError } from '@/lib/apiError';
import { cn } from '@/lib/cn';

const FEATURE_COLUMNS = [
  'open', 'high', 'low', 'close', 'volume',
  'returns', 'volatility', 'volume_ratio',
];

interface MlModel {
  model_id: string;
  model_path: string;
  report: Record<string, unknown> | null;
}

export default function MLPage() {
  const toast = useToast();
  const qc = useQueryClient();
  const [trainOpen, setTrainOpen] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [predictions, setPredictions] = useState<{
    prediction: number;
    probability_up: number;
    probability_down: number;
    ts: string;
  }[]>([]);

  const [trainForm, setTrainForm] = useState({
    n_bars: 2000,
    max_depth: 6,
    n_estimators: 100,
    learning_rate: 0.1,
  });

  const [featureForm, setFeatureForm] = useState<Record<string, string>>(
    Object.fromEntries(FEATURE_COLUMNS.map((c) => [c, '0'])),
  );

  const { data: models = [], isLoading, refetch } = useQuery({
    queryKey: ['ml-models'],
    queryFn: () => api.listMlModels() as Promise<MlModel[]>,
    refetchInterval: (query) => {
      const training = query.state.data?.some((m) => !m.report);
      return training ? 3000 : false;
    },
  });

  const trainMutation = useMutation({
    mutationFn: () => api.trainMlModel(trainForm),
    onSuccess: (res) => {
      toast.success(`训练完成: ${String(res.model_id)}`);
      void qc.invalidateQueries({ queryKey: ['ml-models'] });
      setSelectedModel(String(res.model_id));
    },
    onError: (e) => toast.error(parseApiError(e, '训练失败')),
  });

  const predictMutation = useMutation({
    mutationFn: () => {
      const features: Record<string, number> = {};
      for (const col of FEATURE_COLUMNS) {
        features[col] = Number(featureForm[col]) || 0;
      }
      return api.predictMl({ model_id: selectedModel!, features });
    },
    onSuccess: (res) => {
      setPredictions((prev) => [
        ...prev.slice(-9),
        {
          prediction: res.prediction,
          probability_up: res.probability_up,
          probability_down: res.probability_down,
          ts: new Date().toISOString().slice(11, 19),
        },
      ]);
      toast.success('预测完成');
    },
    onError: (e) => toast.error(parseApiError(e, '预测失败')),
  });

  const selectedReport = models.find((m) => m.model_id === selectedModel)?.report;

  return (
    <div className="px-[3%] py-[2%] space-y-6 max-w-[96rem] mx-auto">
      <div>
        <h1 className="text-xl font-semibold text-text-primary">模型工作台</h1>
        <p className="text-sm text-text-muted mt-0.5">XGBoost 训练 · 模型列表 · 方向预测</p>
      </div>

      <Card title="模型列表">
        {isLoading ? (
          <div className="h-24 animate-pulse bg-surface-tertiary rounded-lg" />
        ) : models.length === 0 ? (
          <p className="text-sm text-text-muted text-center py-8">
            暂无模型。展开下方训练表单发起首次训练。
          </p>
        ) : (
          <div className="overflow-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border text-text-muted">
                  <th className="text-left py-2 px-2">模型 ID</th>
                  <th className="text-left py-2 px-2">训练时间</th>
                  <th className="text-left py-2 px-2">数据范围</th>
                  <th className="text-left py-2 px-2">指标</th>
                </tr>
              </thead>
              <tbody>
                {models.map((m) => {
                  const report = m.report as Record<string, unknown> | null;
                  const testMetrics = report?.test_metrics as Record<string, number> | undefined;
                  const dataInfo = report?.data_info as Record<string, unknown> | undefined;
                  return (
                    <tr
                      key={m.model_id}
                      onClick={() => setSelectedModel(m.model_id)}
                      className={cn(
                        'border-b border-border/40 cursor-pointer hover:bg-surface-tertiary',
                        selectedModel === m.model_id && 'bg-brand/5',
                      )}
                    >
                      <td className="py-2 px-2 font-mono text-text-primary">{m.model_id}</td>
                      <td className="py-2 px-2 text-text-muted">
                        {String(report?.trained_at ?? '—').slice(0, 19)}
                      </td>
                      <td className="py-2 px-2 text-text-secondary">
                        {String(dataInfo?.source ?? '—')}
                      </td>
                      <td className="py-2 px-2 font-mono">
                        {testMetrics
                          ? Object.entries(testMetrics).slice(0, 3).map(([k, v]) => (
                              <span key={k} className="mr-2">{k}:{typeof v === 'number' ? v.toFixed(3) : v}</span>
                            ))
                          : '—'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      <Card
        title="发起训练"
        extra={
          <button type="button" onClick={() => setTrainOpen((v) => !v)} className="text-text-muted">
            {trainOpen ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>
        }
      >
        {trainOpen && (
          <div className="space-y-3">
            <p className="text-xs text-text-muted">
              后端从本地 parquet 自动加载 OHLCV 数据训练（无品种/特征集选择接口）
            </p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {(['n_bars', 'max_depth', 'n_estimators', 'learning_rate'] as const).map((key) => (
                <div key={key}>
                  <label className="block text-xs text-text-muted mb-1">{key}</label>
                  <input
                    type="number"
                    step={key === 'learning_rate' ? 0.01 : 1}
                    value={trainForm[key]}
                    onChange={(e) =>
                      setTrainForm((f) => ({ ...f, [key]: Number(e.target.value) }))
                    }
                    className="w-full bg-surface-tertiary border border-border rounded-lg px-2 py-1.5 text-sm font-mono"
                  />
                </div>
              ))}
            </div>
            <Button onClick={() => trainMutation.mutate()} loading={trainMutation.isPending}>
              <Play className="w-3.5 h-3.5 mr-1.5" />
              开始训练
            </Button>
          </div>
        )}
        {!trainOpen && (
          <p className="text-xs text-text-muted">点击展开训练表单</p>
        )}
      </Card>

      {selectedModel && (
        <Card title={`预测面板 · ${selectedModel}`} extra={<Brain className="w-4 h-4 text-brand" />}>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div>
              <p className="text-xs text-text-muted mb-2">输入特征向量</p>
              <div className="grid grid-cols-2 gap-2">
                {FEATURE_COLUMNS.map((col) => (
                  <div key={col}>
                    <label className="block text-[10px] text-text-muted">{col}</label>
                    <input
                      type="number"
                      step="any"
                      value={featureForm[col]}
                      onChange={(e) =>
                        setFeatureForm((f) => ({ ...f, [col]: e.target.value }))
                      }
                      className="w-full bg-surface-tertiary border border-border rounded px-2 py-1 text-xs font-mono"
                    />
                  </div>
                ))}
              </div>
              <Button
                className="mt-3"
                size="sm"
                onClick={() => predictMutation.mutate()}
                loading={predictMutation.isPending}
              >
                预测
              </Button>

              {predictions.length > 0 && (
                <div className="mt-4 flex items-center gap-3">
                  {predictions[predictions.length - 1]!.prediction === 1 ? (
                    <StatusBadge variant="success" label="看涨" />
                  ) : (
                    <StatusBadge variant="error" label="看跌" />
                  )}
                  <span className="text-sm font-mono">
                    P(up)={predictions[predictions.length - 1]!.probability_up.toFixed(3)}
                  </span>
                  {predictions[predictions.length - 1]!.prediction === 1 ? (
                    <TrendingUp className="w-4 h-4 text-profit" />
                  ) : (
                    <TrendingDown className="w-4 h-4 text-loss" />
                  )}
                </div>
              )}
            </div>

            <div>
              <p className="text-xs text-text-muted mb-2">最近预测</p>
              {predictions.length === 0 ? (
                <p className="text-sm text-text-muted py-8 text-center">暂无预测记录</p>
              ) : (
                <div className="h-32">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={predictions}>
                      <XAxis dataKey="ts" tick={{ fontSize: 9 }} />
                      <YAxis domain={[0, 1]} tick={{ fontSize: 9 }} width={32} />
                      <Tooltip />
                      <Line type="monotone" dataKey="probability_up" stroke="var(--profit)" dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
              {selectedReport && (
                <div className="mt-3 text-[10px] font-mono text-text-muted">
                  train_acc: {String((selectedReport.train_result as Record<string, unknown>)?.train_score ?? '—')}
                </div>
              )}
            </div>
          </div>
        </Card>
      )}

      <Button variant="secondary" size="sm" onClick={() => void refetch()}>
        刷新模型列表
      </Button>
    </div>
  );
}
