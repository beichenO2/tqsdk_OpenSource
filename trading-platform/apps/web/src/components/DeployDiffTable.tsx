import { cn } from '@/lib/cn';

interface DeployDiffTableProps {
  current: Record<string, unknown>;
  candidate: Record<string, unknown>;
  className?: string;
}

export default function DeployDiffTable({ current, candidate, className }: DeployDiffTableProps) {
  const allKeys = [...new Set([...Object.keys(current), ...Object.keys(candidate)])].sort();

  if (allKeys.length === 0) {
    return <p className="text-sm text-text-muted">无参数可对比</p>;
  }

  return (
    <div className={cn('overflow-auto rounded-lg border border-border', className)}>
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-border bg-surface-tertiary text-text-muted">
            <th className="text-left py-2 px-3 font-medium">参数</th>
            <th className="text-left py-2 px-3 font-medium">当前值</th>
            <th className="text-left py-2 px-3 font-medium">候选值</th>
          </tr>
        </thead>
        <tbody>
          {allKeys.map((key) => {
            const oldVal = current[key];
            const newVal = candidate[key];
            const changed = JSON.stringify(oldVal) !== JSON.stringify(newVal);
            const isNew = !(key in current) && key in candidate;
            return (
              <tr key={key} className="border-b border-border/40">
                <td className="py-2 px-3 font-mono text-text-secondary">{key}</td>
                <td className="py-2 px-3 font-mono">
                  {key in current ? (
                    <span className={cn(changed && 'line-through text-text-muted')}>
                      {formatVal(oldVal)}
                    </span>
                  ) : (
                    <span className="text-text-muted">—</span>
                  )}
                </td>
                <td className="py-2 px-3 font-mono">
                  {key in candidate ? (
                    <span className={cn(isNew || changed ? 'text-profit font-medium' : 'text-text-primary')}>
                      {formatVal(newVal)}
                    </span>
                  ) : (
                    <span className="text-text-muted">—</span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function formatVal(v: unknown): string {
  if (v == null) return '—';
  if (typeof v === 'object') return JSON.stringify(v);
  return String(v);
}
