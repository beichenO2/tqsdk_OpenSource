import type { ReactNode } from 'react';
import clsx from 'clsx';

interface CardProps {
  title?: string;
  extra?: ReactNode;
  children: ReactNode;
  className?: string;
  noPadding?: boolean;
}

export default function Card({ title, extra, children, className, noPadding }: CardProps) {
  return (
    <div className={clsx('bg-surface-secondary border border-border rounded-2xl shadow-sm transition-colors', className)}>
      {title && (
        <div className="flex items-center justify-between px-6 py-4 border-b border-border">
          <h3 className="text-[15px] font-semibold text-text-primary tracking-tight">{title}</h3>
          {extra}
        </div>
      )}
      <div className={noPadding ? '' : 'p-6'}>{children}</div>
    </div>
  );
}
