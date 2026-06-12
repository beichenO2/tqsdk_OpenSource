import type { ReactNode } from 'react';
import { cn } from '@/lib/cn';

interface EmptyStateProps {
  icon?: ReactNode;
  title: string;
  description?: string;
  action?: ReactNode;
  className?: string;
}

export function EmptyState({ icon, title, description, action, className }: EmptyStateProps) {
  return (
    <div className={cn('flex flex-col items-center justify-center gap-3 py-12 text-center', className)}>
      {icon && <div className="text-text-muted opacity-40">{icon}</div>}
      <div>
        <p className="text-sm font-medium text-text-secondary">{title}</p>
        {description && <p className="mt-1 text-xs text-text-muted max-w-sm">{description}</p>}
      </div>
      {action}
    </div>
  );
}
