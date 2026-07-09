import { useEffect, useRef, useState } from 'react';
import { useQuotes } from '@/hooks/useMarketData';
import { cn } from '@/lib/cn';
import type { Quote } from '@/types';

type FlashDir = 'up' | 'down' | null;

/** Terminal market watchlist — rows tick-flash on price change. */
export default function MarketWatch({ maxRows = 14 }: { maxRows?: number }) {
  const { data: quotes = [] } = useQuotes();
  const prevPrices = useRef<Map<string, number>>(new Map());
  const [flash, setFlash] = useState<Map<string, FlashDir>>(new Map());

  useEffect(() => {
    if (!quotes.length) return;
    const next = new Map<string, FlashDir>();
    for (const q of quotes) {
      const prev = prevPrices.current.get(q.instrument_id);
      if (prev !== undefined && prev !== q.last_price) {
        next.set(q.instrument_id, q.last_price > prev ? 'up' : 'down');
      }
      prevPrices.current.set(q.instrument_id, q.last_price);
    }
    if (next.size) {
      setFlash(next);
      const t = setTimeout(() => setFlash(new Map()), 650);
      return () => clearTimeout(t);
    }
  }, [quotes]);

  const rows: Quote[] = quotes.slice(0, maxRows);

  return (
    <div className="overflow-y-auto">
      <table className="w-full text-[12px]">
        <thead className="sticky top-0 bg-surface-secondary z-10">
          <tr className="border-b border-border text-[10px] text-text-muted font-mono uppercase tracking-wider">
            <th className="text-left px-3 py-1.5">合约</th>
            <th className="text-right px-2 py-1.5">最新</th>
            <th className="text-right px-3 py-1.5">仓量</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((q) => {
            const dir = flash.get(q.instrument_id);
            const sym = q.instrument_id.split('.').pop() || q.instrument_id;
            return (
              <tr
                key={q.instrument_id}
                className={cn(
                  'border-b border-border/40',
                  dir && 'tick-flash',
                )}
              >
                <td className="px-3 py-[5px] font-mono text-text-secondary">
                  <span className="text-text-primary">{sym}</span>
                  <span className="ml-1.5 text-[9.5px] text-text-muted">{q.exchange}</span>
                </td>
                <td
                  className={cn(
                    'px-2 py-[5px] text-right font-mono font-medium',
                    dir === 'up' && 'text-profit',
                    dir === 'down' && 'text-loss',
                    !dir && 'text-text-primary',
                  )}
                >
                  {q.last_price.toLocaleString()}
                  {dir === 'up' && <span className="ml-0.5 text-[9px]">▲</span>}
                  {dir === 'down' && <span className="ml-0.5 text-[9px]">▼</span>}
                </td>
                <td className="px-3 py-[5px] text-right font-mono text-text-muted">
                  {q.open_interest > 0 ? (q.open_interest / 1000).toFixed(0) + 'k' : '—'}
                </td>
              </tr>
            );
          })}
          {rows.length === 0 && (
            <tr>
              <td colSpan={3} className="px-3 py-8 text-center text-text-muted text-xs">
                行情加载中…（闭市显示最近缓存价）
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
