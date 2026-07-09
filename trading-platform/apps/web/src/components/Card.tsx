import type { ReactNode } from 'react';
import {
  Card as ShadCard,
  CardContent,
  CardHeader,
  CardTitle,
  CardAction,
} from '@/components/shadcn/card';
import { cn } from '@/lib/utils';

interface CardProps {
  title?: string;
  extra?: ReactNode;
  children: ReactNode;
  className?: string;
  noPadding?: boolean;
}

/** Legacy-API wrapper over shadcn/ui Card. */
export default function Card({ title, extra, children, className, noPadding }: CardProps) {
  return (
    <ShadCard className={cn('gap-0 py-0 overflow-hidden', className)}>
      {title && (
        <CardHeader className="border-b px-4 !py-2.5 min-h-11 flex-row items-center [.border-b]:pb-2.5">
          <CardTitle className="panel-label">{title}</CardTitle>
          {extra && <CardAction className="self-center row-start-1">{extra}</CardAction>}
        </CardHeader>
      )}
      <CardContent className={noPadding ? 'p-0' : 'p-4'}>{children}</CardContent>
    </ShadCard>
  );
}
