import { cn } from '@/lib/cn';

interface SkeletonProps {
  className?: string;
  style?: React.CSSProperties;
}

export function Skeleton({ className, style }: SkeletonProps) {
  return (
    <div className={cn('animate-pulse rounded-lg bg-surface-tertiary', className)} style={style} />
  );
}

export function StatCardSkeleton() {
  return (
    <div className="bg-surface-secondary border border-border rounded-xl p-4 space-y-2">
      <Skeleton className="h-3 w-16" />
      <Skeleton className="h-7 w-24" />
      <Skeleton className="h-3 w-20" />
    </div>
  );
}

export function TableRowSkeleton({ cols = 6 }: { cols?: number }) {
  return (
    <tr className="border-b border-border/50">
      {Array.from({ length: cols }).map((_, i) => (
        <td key={i} className="px-4 py-2.5">
          <Skeleton className="h-4 w-full" />
        </td>
      ))}
    </tr>
  );
}

export function ChartSkeleton({ height = 240 }: { height?: number }) {
  return (
    <div className="flex items-end gap-1 px-4" style={{ height }}>
      {Array.from({ length: 20 }).map((_, i) => (
        <Skeleton
          key={i}
          className="flex-1 rounded-t"
          style={{ height: `${30 + Math.random() * 60}%` }}
        />
      ))}
    </div>
  );
}
