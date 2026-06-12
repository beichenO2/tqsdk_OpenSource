import clsx from 'clsx';

interface StatCardProps {
  label: string;
  value: string;
  sub?: string;
  trend?: 'up' | 'down' | 'neutral';
}

export default function StatCard({ label, value, sub, trend }: StatCardProps) {
  return (
    <div className="bg-surface-secondary border border-border rounded-2xl p-6 shadow-sm transition-colors hover:border-brand/30">
      <p className="text-[13px] text-text-secondary mb-2 tracking-tight">{label}</p>
      <p
        className={clsx(
          'text-2xl font-semibold tabular-nums tracking-tight',
          trend === 'up' && 'text-profit',
          trend === 'down' && 'text-loss',
          (!trend || trend === 'neutral') && 'text-text-primary',
        )}
      >
        {value}
      </p>
      {sub && (
        <p
          className={clsx(
            'text-[13px] mt-1.5 tabular-nums',
            trend === 'up' && 'text-profit/70',
            trend === 'down' && 'text-loss/70',
            (!trend || trend === 'neutral') && 'text-text-muted',
          )}
        >
          {sub}
        </p>
      )}
    </div>
  );
}
