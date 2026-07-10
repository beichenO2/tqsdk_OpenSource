import { useCallback, useEffect, useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { RotateCcw, Save, Shield } from 'lucide-react';
import Card from '@/components/Card';
import { Button } from '@/components/ui/Button';
import { useConfirm } from '@/components/ui/ConfirmDialog';
import { useToast } from '@/components/ui/Toast';
import { api } from '@/services/api';
import { parseApiError } from '@/lib/apiError';
import { cn } from '@/lib/cn';

const SECTION_META: Record<string, { title: string; description: string }> = {
  research: { title: '研究配置', description: '默认品种、并发数、验证门控' },
  mcp: { title: 'MCP 工具', description: '暴露给外部 Agent 的工具列表' },
  upload: { title: '数据上传', description: '文件大小与允许扩展名' },
  notifications: { title: '通知', description: 'SSE 心跳与事件推送' },
};

function FieldEditor({
  label,
  value,
  onChange,
  type = 'text',
}: {
  label: string;
  value: unknown;
  onChange: (v: unknown) => void;
  type?: 'text' | 'number' | 'boolean' | 'json' | 'tags';
}) {
  if (type === 'boolean') {
    return (
      <label className="flex items-center justify-between cursor-pointer py-1">
        <span className="text-sm text-text-secondary">{label}</span>
        <input
          type="checkbox"
          checked={!!value}
          onChange={(e) => onChange(e.target.checked)}
          className="w-4 h-4 rounded border-border text-brand focus:ring-brand bg-surface-tertiary"
        />
      </label>
    );
  }

  if (type === 'tags' || (Array.isArray(value) && type === 'json')) {
    const arr = Array.isArray(value) ? value : [];
    return (
      <div>
        <label className="block text-xs text-text-muted mb-1">{label}</label>
        <input
          type="text"
          value={arr.join(', ')}
          onChange={(e) =>
            onChange(
              e.target.value
                .split(',')
                .map((s) => s.trim())
                .filter(Boolean),
            )
          }
          className="w-full bg-surface-tertiary border border-border rounded-lg px-3 py-2 text-sm text-text-primary font-mono focus:outline-none focus:border-brand"
        />
      </div>
    );
  }

  if (typeof value === 'object' && value !== null) {
    return (
      <div>
        <label className="block text-xs text-text-muted mb-1">{label}</label>
        <textarea
          rows={4}
          value={JSON.stringify(value, null, 2)}
          onChange={(e) => {
            try {
              onChange(JSON.parse(e.target.value));
            } catch {
              /* ignore invalid json while typing */
            }
          }}
          className="w-full bg-surface-tertiary border border-border rounded-lg px-3 py-2 text-xs text-text-primary font-mono focus:outline-none focus:border-brand"
        />
      </div>
    );
  }

  return (
    <div>
      <label className="block text-xs text-text-muted mb-1">{label}</label>
      <input
        type={type === 'number' ? 'number' : 'text'}
        value={String(value ?? '')}
        onChange={(e) =>
          onChange(type === 'number' ? Number(e.target.value) : e.target.value)
        }
        className="w-full bg-surface-tertiary border border-border rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:border-brand"
      />
    </div>
  );
}

function inferFieldType(key: string, value: unknown): 'text' | 'number' | 'boolean' | 'json' | 'tags' {
  if (typeof value === 'boolean') return 'boolean';
  if (typeof value === 'number') return 'number';
  if (Array.isArray(value)) {
    if (key.includes('extensions') || key.includes('symbols') || key.includes('gates') || key.includes('tools')) {
      return 'tags';
    }
    return 'json';
  }
  if (typeof value === 'object' && value !== null) return 'json';
  return 'text';
}

function SectionCard({
  sectionKey,
  data,
  onSave,
  saving,
}: {
  sectionKey: string;
  data: Record<string, unknown>;
  onSave: (section: string, body: Record<string, unknown>) => void;
  saving: boolean;
}) {
  const [draft, setDraft] = useState(data);
  const meta = SECTION_META[sectionKey] ?? { title: sectionKey, description: '' };

  useEffect(() => {
    setDraft(data);
  }, [data]);

  const updateField = (key: string, value: unknown) => {
    setDraft((prev) => ({ ...prev, [key]: value }));
  };

  return (
    <Card title={meta.title}>
      <p className="text-xs text-text-muted mb-4">{meta.description}</p>
      <div className="space-y-3">
        {Object.entries(draft).map(([key, value]) => (
          <FieldEditor
            key={key}
            label={key}
            value={value}
            type={inferFieldType(key, value)}
            onChange={(v) => updateField(key, v)}
          />
        ))}
      </div>
      <Button
        className="mt-4"
        size="sm"
        onClick={() => onSave(sectionKey, draft)}
        loading={saving}
      >
        <Save className="w-3.5 h-3.5 mr-1.5" />
        保存此分组
      </Button>
    </Card>
  );
}

export default function SettingsPage() {
  const toast = useToast();
  const confirm = useConfirm();
  const qc = useQueryClient();

  const { data: settings, isLoading, isError, error } = useQuery({
    queryKey: ['settings'],
    queryFn: () => api.getSettings(),
  });

  const saveMutation = useMutation({
    mutationFn: ({ section, body }: { section: string; body: Record<string, unknown> }) =>
      api.replaceSettingsSection(section, body),
    onSuccess: (_, { section }) => {
      toast.success(`${SECTION_META[section]?.title ?? section} 已保存`);
      void qc.invalidateQueries({ queryKey: ['settings'] });
    },
    onError: (e) => toast.error(parseApiError(e, '保存失败')),
  });

  const resetMutation = useMutation({
    mutationFn: () => api.resetSettings(),
    onSuccess: () => {
      toast.success('已重置为默认配置');
      void qc.invalidateQueries({ queryKey: ['settings'] });
    },
    onError: (e) => toast.error(parseApiError(e, '重置失败')),
  });

  const handleReset = useCallback(async () => {
    const ok = await confirm({
      title: '重置为默认配置',
      description: '所有设置分组将恢复为系统默认值，此操作不可撤销。',
      variant: 'destructive',
      confirmText: '确认重置',
    });
    if (!ok) return;
    resetMutation.mutate();
  }, [confirm, resetMutation]);

  const sections = settings ? Object.keys(settings).filter((k) => typeof settings[k] === 'object') : [];

  return (
    <div className="p-6 space-y-6 max-w-5xl">
      <div className="flex items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold text-text-primary">系统设置</h1>
          <p className="text-sm text-text-muted mt-0.5">
            研究平台本地配置（仅 localhost 可修改）
          </p>
        </div>
        <Button
          variant="secondary"
          size="sm"
          onClick={() => void handleReset()}
          loading={resetMutation.isPending}
        >
          <RotateCcw className="w-3.5 h-3.5 mr-1.5" />
          重置为默认
        </Button>
      </div>

      {isLoading && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="h-48 bg-surface-secondary rounded-xl border border-border animate-pulse" />
          ))}
        </div>
      )}

      {isError && (
        <Card>
          <p className="text-sm text-loss">{parseApiError(error, '加载设置失败')}</p>
          <p className="text-xs text-text-muted mt-2">
            设置 API 仅允许 localhost 访问。请通过本地开发服务器打开此页面。
          </p>
        </Card>
      )}

      {!isLoading && !isError && settings && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Card title="交易凭证">
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <Shield className="w-4 h-4 text-text-muted" />
                <span className="text-sm text-text-secondary">TqSdk / 期货公司凭证</span>
              </div>
              <div className="flex items-center justify-between rounded-lg border border-border px-3 py-2">
                <span className="text-xs text-text-muted">配置状态</span>
                <span className={cn(
                  'text-xs font-medium px-2 py-0.5 rounded',
                  'bg-warning/10 text-warning border border-warning/30',
                )}>
                  通过 PolarPrivate 管理
                </span>
              </div>
              <p className="text-xs text-text-muted leading-relaxed">
                敏感凭证不在 Web 端明文编辑。请使用 PolarPrivate 或环境变量配置 TqSdk 账号、密码与期货公司信息。
              </p>
            </div>
          </Card>

          {sections.map((key) => (
            <SectionCard
              key={key}
              sectionKey={key}
              data={settings[key] as Record<string, unknown>}
              onSave={(section, body) => saveMutation.mutate({ section, body })}
              saving={saveMutation.isPending}
            />
          ))}
        </div>
      )}
    </div>
  );
}
