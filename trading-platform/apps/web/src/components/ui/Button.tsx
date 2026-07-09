import { forwardRef, type ButtonHTMLAttributes } from 'react';
import { Loader2 } from 'lucide-react';
import { Button as ShadButton } from '@/components/shadcn/button';
import { cn } from '@/lib/utils';

/** Legacy-API wrapper over shadcn/ui Button. */
export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'destructive' | 'ghost' | 'profit' | 'loss';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
}

const variantMap: Record<
  NonNullable<ButtonProps['variant']>,
  { shad: 'default' | 'secondary' | 'destructive' | 'ghost' | 'outline'; extra?: string }
> = {
  primary: { shad: 'default' },
  secondary: { shad: 'outline' },
  destructive: { shad: 'destructive' },
  ghost: { shad: 'ghost' },
  profit: {
    shad: 'outline',
    extra: 'border-profit/50 text-profit hover:bg-profit/10 hover:text-profit',
  },
  loss: {
    shad: 'outline',
    extra: 'border-loss/50 text-loss hover:bg-loss/10 hover:text-loss',
  },
};

const sizeMap: Record<NonNullable<ButtonProps['size']>, 'sm' | 'default' | 'lg'> = {
  sm: 'sm',
  md: 'default',
  lg: 'lg',
};

const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = 'primary', size = 'md', loading, disabled, children, ...props }, ref) => {
    const m = variantMap[variant];
    return (
      <ShadButton
        ref={ref}
        type="button"
        variant={m.shad}
        size={sizeMap[size]}
        disabled={disabled || loading}
        className={cn(m.extra, className)}
        {...props}
      >
        {loading && <Loader2 className="h-4 w-4 animate-spin" />}
        {children}
      </ShadButton>
    );
  },
);

Button.displayName = 'Button';
export { Button };
