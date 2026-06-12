import { useEffect, useMemo, useState } from 'react';
import { Loader2, Play } from 'lucide-react';
import { api } from '@/services/api';
import type { BacktestResult } from '@/types';

export interface BacktestFormProps {
  onSubmitted: (result: BacktestResult) => void;
  onCancel?: () => void;
}

export default function BacktestForm({ onSubmitted, onCancel }: BacktestFormProps) {
  const [strategies, setStrategies] = useState<{ value: string; label: string }[]>([]);
  const [instruments, setInstruments] = useState<{ symbol: string; exchange: string }[]>([]);
  const [strategyId, setStrategyId] = useState('');
  const [symbolInput, setSymbolInput] = useState('');
  const [startDate, setStartDate] = useState('2025-01-01');
  const [endDate, setEndDate] = useState('2025-12-31');
  const [capital, setCapital] = useState('1000000');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showSuggestions, setShowSuggestions] = useState(false);

  useEffect(() => {
    api.getBacktestStrategyOptions().then((opts) => {
      setStrategies(opts);
      if (opts[0]) setStrategyId(opts[0].value);
    });
    api.getMarketInstruments().then(setInstruments);
  }, []);

  const suggestions = useMemo(() => {
    const q = symbolInput.trim().toUpperCase();
    if (!q) return instruments.slice(0, 12);
    return instruments.filter((i) => i.symbol.toUpperCase().includes(q)).slice(0, 12);
  }, [symbolInput, instruments]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    const sym = symbolInput.trim();
    if (!sym) {
      setError('请填写合约 / 标的');
      return;
    }
    const cap = Number(capital.replace(/,/g, ''));
    if (!Number.isFinite(cap) || cap <= 0) {
      setError('请输入有效资金');
      return;
    }
    if (!strategyId) {
      setError('请选择策略');
      return;
    }
    setLoading(true);
    try {
      const result = await api.createBacktest({
        strategy_id: strategyId,
        start_date: startDate,
        end_date: endDate,
        initial_capital: cap,
        instruments: [sym],
      });
      onSubmitted(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : '提交失败');
    } finally {
      setLoading(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="grid gap-4 sm:grid-cols-2">
        <div className="sm:col-span-2">
          <label className="mb-1 block text-xs font-medium text-text-muted">策略</label>
          <select
            value={strategyId}
            onChange={(e) => setStrategyId(e.target.value)}
            className="w-full rounded-lg border border-border bg-surface-tertiary px-3 py-2 text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-brand/40"
          >
            {strategies.map((s) => (
              <option key={s.value} value={s.value}>
                {s.label}
              </option>
            ))}
          </select>
        </div>

        <div className="relative sm:col-span-2">
          <label className="mb-1 block text-xs font-medium text-text-muted">合约代码</label>
          <input
            value={symbolInput}
            onChange={(e) => {
              setSymbolInput(e.target.value);
              setShowSuggestions(true);
            }}
            onFocus={() => setShowSuggestions(true)}
            onBlur={() => setTimeout(() => setShowSuggestions(false), 150)}
            placeholder="如 SHFE.rb2501 或 IF2504"
            className="w-full rounded-lg border border-border bg-surface-tertiary px-3 py-2 text-sm text-text-primary placeholder:text-text-muted focus:outline-none focus:ring-2 focus:ring-brand/40"
            autoComplete="off"
          />
          {showSuggestions && suggestions.length > 0 && (
            <ul className="absolute z-20 mt-1 max-h-48 w-full overflow-auto rounded-lg border border-border bg-surface-secondary py-1 text-sm shadow-lg">
              {suggestions.map((i) => (
                <li key={i.symbol}>
                  <button
                    type="button"
                    className="flex w-full items-center justify-between px-3 py-2 text-left text-text-primary hover:bg-surface-tertiary"
                    onMouseDown={(ev) => {
                      ev.preventDefault();
                      setSymbolInput(i.symbol);
                      setShowSuggestions(false);
                    }}
                  >
                    <span className="font-medium">{i.symbol}</span>
                    <span className="text-xs text-text-muted">{i.exchange}</span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        <div>
          <label className="mb-1 block text-xs font-medium text-text-muted">开始日期</label>
          <input
            type="date"
            value={startDate}
            onChange={(e) => setStartDate(e.target.value)}
            className="w-full rounded-lg border border-border bg-surface-tertiary px-3 py-2 text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-brand/40"
          />
        </div>
        <div>
          <label className="mb-1 block text-xs font-medium text-text-muted">结束日期</label>
          <input
            type="date"
            value={endDate}
            onChange={(e) => setEndDate(e.target.value)}
            className="w-full rounded-lg border border-border bg-surface-tertiary px-3 py-2 text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-brand/40"
          />
        </div>

        <div className="sm:col-span-2">
          <label className="mb-1 block text-xs font-medium text-text-muted">初始资金</label>
          <input
            value={capital}
            onChange={(e) => setCapital(e.target.value)}
            inputMode="decimal"
            className="w-full rounded-lg border border-border bg-surface-tertiary px-3 py-2 text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-brand/40"
          />
        </div>
      </div>

      {error && <p className="text-xs text-loss">{error}</p>}

      <div className="flex flex-wrap gap-2">
        <button
          type="submit"
          disabled={loading}
          className="inline-flex items-center gap-2 rounded-lg bg-brand px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-brand-dark disabled:opacity-60"
        >
          {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
          运行回测
        </button>
        {onCancel && (
          <button
            type="button"
            onClick={onCancel}
            className="rounded-lg border border-border px-4 py-2 text-sm text-text-secondary hover:bg-surface-tertiary hover:text-text-primary"
          >
            取消
          </button>
        )}
      </div>
    </form>
  );
}
