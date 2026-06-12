import clsx from 'clsx';

interface IPriceDisplayProps {
  price: number;
  change?: number;
  changePercent?: number;
  size?: 'sm' | 'md' | 'lg';
  showSign?: boolean;
}

const sizeClasses = {
  sm: 'text-sm',
  md: 'text-lg',
  lg: 'text-3xl font-bold',
};

function formatPrice(price: number): string {
  if (Math.abs(price) >= 1) {
    return price.toLocaleString('en-US', {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    });
  }
  return price.toFixed(8);
}

export default function PriceDisplay({
  price,
  change,
  changePercent,
  size = 'md',
  showSign = true,
}: IPriceDisplayProps) {
  const isPositive = (change ?? 0) >= 0;

  return (
    <div className="flex items-baseline gap-3">
      <span className={clsx(sizeClasses[size], 'font-mono tabular-nums')}>
        {formatPrice(price)}
      </span>
      {change !== undefined && (
        <span
          className={clsx(
            'font-mono text-sm tabular-nums',
            isPositive ? 'text-profit' : 'text-loss',
          )}
        >
          {showSign && (isPositive ? '+' : '')}
          {formatPrice(change)}
        </span>
      )}
      {changePercent !== undefined && (
        <span
          className={clsx(
            'rounded px-1.5 py-0.5 text-xs font-medium tabular-nums',
            isPositive ? 'bg-profit/15 text-profit' : 'bg-loss/15 text-loss',
          )}
        >
          {isPositive ? '+' : ''}
          {changePercent.toFixed(2)}%
        </span>
      )}
    </div>
  );
}
