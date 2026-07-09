import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { BookOpen, ChevronRight } from 'lucide-react';
import { api } from '@/services/api';
import Card from '@/components/Card';
import { cn } from '@/lib/cn';

type SkillSummary = {
  name: string;
  file: string;
  description: string;
  category: string;
  bindings?: string[];
  inputs?: { name?: string; type?: string; description?: string }[];
  outputs?: { name?: string; type?: string; description?: string }[];
  steps?: string[];
};

const CATEGORY_LABEL: Record<string, string> = {
  research: '研究',
  export: '导出',
  unknown: '其它',
  error: '错误',
};

export default function SkillsPage() {
  const [selected, setSelected] = useState<string | null>(null);
  const [category, setCategory] = useState<string | undefined>(undefined);

  const list = useQuery({
    queryKey: ['platform-skills'],
    queryFn: () => api.getPlatformSkills(),
  });

  const detail = useQuery({
    queryKey: ['platform-skill', selected],
    queryFn: () => api.getPlatformSkill(selected!),
    enabled: !!selected,
  });

  const skills = (list.data?.skills || []) as SkillSummary[];
  const categories = useMemo(
    () => Array.from(new Set(skills.map((s) => s.category))).sort(),
    [skills],
  );
  const filtered = useMemo(
    () => (category ? skills.filter((s) => s.category === category) : skills),
    [skills, category],
  );

  return (
    <div className="px-[3%] py-[2%] space-y-4 max-w-[96rem] mx-auto">
      <div>
        <h1 className="text-xl font-semibold text-text-primary">技能</h1>
        <p className="text-sm text-text-secondary mt-0.5">
          docs/skills — 策略生成 / 回测诊断 / 因子风险 / 导出 SOP
          {list.data?.dir ? (
            <span className="ml-2 font-mono text-xs text-text-muted">{list.data.dir}</span>
          ) : null}
        </p>
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
          全部 ({skills.length})
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
            {CATEGORY_LABEL[c] || c}
          </button>
        ))}
      </div>

      {list.isLoading && <p className="text-sm text-text-muted">加载中…</p>}
      {list.error && (
        <p className="text-sm text-loss">
          {list.error instanceof Error ? list.error.message : '加载失败'}
        </p>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="space-y-2">
          {filtered.map((s) => (
            <button
              key={s.name}
              type="button"
              onClick={() => setSelected(s.name)}
              className={cn(
                'w-full text-left rounded-xl border px-4 py-3 transition-colors',
                selected === s.name
                  ? 'border-brand bg-brand/5'
                  : 'border-border hover:border-brand/40 bg-surface-secondary',
              )}
            >
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    <BookOpen className="w-3.5 h-3.5 text-brand shrink-0" />
                    <span className="font-mono text-sm text-text-primary">{s.name}</span>
                    <span className="text-[10px] uppercase tracking-wide text-text-muted border border-border rounded px-1">
                      {CATEGORY_LABEL[s.category] || s.category}
                    </span>
                  </div>
                  <p className="text-xs text-text-secondary mt-1 line-clamp-2">{s.description}</p>
                  {(s.steps || []).length > 0 && (
                    <p className="text-[11px] text-text-muted mt-1.5 font-mono truncate">
                      steps: {(s.steps || []).join(' → ')}
                    </p>
                  )}
                </div>
                <ChevronRight className="w-4 h-4 text-text-muted shrink-0 mt-0.5" />
              </div>
            </button>
          ))}
          {!list.isLoading && filtered.length === 0 && (
            <p className="text-sm text-text-muted">暂无技能定义</p>
          )}
        </div>

        <Card
          title={selected ? `详情 · ${selected}` : '详情'}
          extra={<span className="text-xs text-text-muted">YAML</span>}
        >
          {!selected && (
            <p className="text-sm text-text-muted">选择左侧技能查看输入/输出与绑定。</p>
          )}
          {selected && detail.isLoading && (
            <p className="text-sm text-text-muted">加载详情…</p>
          )}
          {selected && detail.error && (
            <p className="text-sm text-loss">
              {detail.error instanceof Error ? detail.error.message : '加载失败'}
            </p>
          )}
          {selected && detail.data && (
            <div className="space-y-4 text-sm">
              <p className="text-text-secondary">{String(detail.data.description || '')}</p>

              <div>
                <h3 className="text-xs text-text-muted mb-1.5">输入</h3>
                <ul className="space-y-1">
                  {((detail.data.inputs as SkillSummary['inputs']) || []).map((inp, i) => (
                    <li key={i} className="font-mono text-xs">
                      <span className="text-brand">{inp.name}</span>
                      {inp.type ? <span className="text-text-muted"> : {inp.type}</span> : null}
                      {inp.description ? (
                        <span className="text-text-secondary"> — {inp.description}</span>
                      ) : null}
                    </li>
                  ))}
                  {!(detail.data.inputs as unknown[])?.length && (
                    <li className="text-xs text-text-muted">—</li>
                  )}
                </ul>
              </div>

              <div>
                <h3 className="text-xs text-text-muted mb-1.5">输出</h3>
                <ul className="space-y-1">
                  {((detail.data.outputs as SkillSummary['outputs']) || []).map((out, i) => (
                    <li key={i} className="font-mono text-xs">
                      <span className="text-brand">{out.name}</span>
                      {out.type ? <span className="text-text-muted"> : {out.type}</span> : null}
                    </li>
                  ))}
                </ul>
              </div>

              {Array.isArray(detail.data.bindings) && detail.data.bindings.length > 0 && (
                <div>
                  <h3 className="text-xs text-text-muted mb-1.5">tqsdk_bindings</h3>
                  <ul className="space-y-1">
                    {(detail.data.bindings as string[]).map((b) => (
                      <li key={b} className="font-mono text-xs text-text-secondary">
                        {b}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {typeof detail.data.content === 'string' && (
                <pre className="text-[11px] font-mono bg-surface-tertiary rounded-lg p-3 overflow-auto max-h-[40vh] whitespace-pre-wrap">
                  {detail.data.content}
                </pre>
              )}
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
