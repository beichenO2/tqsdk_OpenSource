import clsx from 'clsx';

type Variant = 'success' | 'error' | 'warning' | 'info' | 'neutral';

const variants: Record<Variant, string> = {
  success: 'bg-profit/15 text-profit',
  error: 'bg-loss/15 text-loss',
  warning: 'bg-warning/15 text-warning',
  info: 'bg-brand/15 text-brand-light',
  neutral: 'bg-border text-text-secondary',
};

interface StatusBadgeProps {
  variant: Variant;
  label: string;
  pulse?: boolean;
}

export default function StatusBadge({ variant, label, pulse }: StatusBadgeProps) {
  return (
    <span
      className={clsx(
        'inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium',
        variants[variant],
      )}
    >
      {pulse && (
        <span
          className={clsx(
            'w-1.5 h-1.5 rounded-full animate-pulse',
            variant === 'success' && 'bg-profit',
            variant === 'error' && 'bg-loss',
            variant === 'warning' && 'bg-warning',
            variant === 'info' && 'bg-brand-light',
          )}
        />
      )}
      {label}
    </span>
  );
}
