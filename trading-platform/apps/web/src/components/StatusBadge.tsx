import { Badge } from '@/components/shadcn/badge';
import { cn } from '@/lib/utils';

type Variant = 'success' | 'error' | 'warning' | 'info' | 'neutral';

const variantClass: Record<Variant, string> = {
  success: 'border-profit/40 bg-profit/10 text-profit',
  error: 'border-loss/40 bg-loss/10 text-loss',
  warning: 'border-warning/40 bg-warning/10 text-warning',
  info: 'border-primary/40 bg-primary/10 text-primary',
  neutral: 'border-border bg-muted text-muted-foreground',
};

interface StatusBadgeProps {
  variant: Variant;
  label: string;
  pulse?: boolean;
}

/** Legacy-API wrapper over shadcn/ui Badge. */
export default function StatusBadge({ variant, label, pulse }: StatusBadgeProps) {
  return (
    <Badge variant="outline" className={cn('font-mono tabular-nums', variantClass[variant])}>
      {pulse && <span className="size-1.5 rounded-full bg-current animate-pulse" />}
      {label}
    </Badge>
  );
}
