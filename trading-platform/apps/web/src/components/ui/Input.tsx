import { forwardRef, type InputHTMLAttributes, type ReactNode } from 'react';
import { cn } from '@/lib/cn';

interface InputProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'prefix'> {
  label?: string;
  error?: string;
  prefix?: ReactNode;
  suffix?: ReactNode;
}

const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ className, label, error, prefix, suffix, id, ...props }, ref) => {
    const inputId = id || (label ? `input-${label.replace(/\s/g, '-')}` : undefined);
    return (
      <div className="space-y-1">
        {label && (
          <label htmlFor={inputId} className="block text-xs text-text-muted">
            {label}
          </label>
        )}
        <div className="relative">
          {prefix && (
            <div className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3 text-text-muted">
              {prefix}
            </div>
          )}
          <input
            ref={ref}
            id={inputId}
            className={cn(
              'w-full rounded-lg border bg-surface-tertiary px-3 py-2 text-sm text-text-primary tabular-nums',
              'transition-colors focus:outline-none focus:border-brand focus:ring-1 focus:ring-focus-ring',
              error ? 'border-loss' : 'border-border',
              prefix && 'pl-9',
              suffix && 'pr-9',
              className,
            )}
            {...props}
          />
          {suffix && (
            <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center pr-3 text-text-muted">
              {suffix}
            </div>
          )}
        </div>
        {error && <p className="text-xs text-loss">{error}</p>}
      </div>
    );
  },
);

Input.displayName = 'Input';
export { Input };
