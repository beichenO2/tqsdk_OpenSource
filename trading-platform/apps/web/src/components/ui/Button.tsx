import { forwardRef, type ButtonHTMLAttributes } from 'react';
import { cn } from '@/lib/cn';
import { Loader2 } from 'lucide-react';

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'destructive' | 'ghost' | 'profit' | 'loss';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
}

const variantStyles: Record<NonNullable<ButtonProps['variant']>, string> = {
  primary: 'bg-brand text-white hover:bg-brand-dark active:bg-brand-dark/90',
  secondary: 'bg-surface-tertiary text-text-secondary border border-border hover:text-text-primary hover:bg-surface-700',
  destructive: 'bg-loss/15 text-loss hover:bg-loss/25 active:bg-loss/30',
  ghost: 'text-text-muted hover:text-text-primary hover:bg-surface-tertiary',
  profit: 'bg-profit text-white hover:bg-profit/80',
  loss: 'bg-loss text-white hover:bg-loss/80',
};

const sizeStyles: Record<NonNullable<ButtonProps['size']>, string> = {
  sm: 'px-2.5 py-1 text-xs gap-1',
  md: 'px-3.5 py-2 text-sm gap-1.5',
  lg: 'px-5 py-2.5 text-sm gap-2',
};

const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = 'primary', size = 'md', loading, disabled, children, ...props }, ref) => (
    <button
      ref={ref}
      type="button"
      disabled={disabled || loading}
      className={cn(
        'inline-flex items-center justify-center rounded-lg font-medium transition-colors',
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus-ring',
        'disabled:opacity-50 disabled:pointer-events-none',
        variantStyles[variant],
        sizeStyles[size],
        className,
      )}
      {...props}
    >
      {loading && <Loader2 className="h-4 w-4 animate-spin" />}
      {children}
    </button>
  ),
);

Button.displayName = 'Button';
export { Button };
