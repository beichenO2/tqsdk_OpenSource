import { useEffect, useState, useCallback, useRef } from 'react';
import { AlertTriangle, Bitcoin } from 'lucide-react';
import { cryptoApi } from './api';
import Card from '@/components/Card';
import StatusBadge from '@/components/StatusBadge';
import type { CryptoQuote, CryptoPosition, CryptoOrder } from './types';

const QUOTE_REFRESH = 3_000;

function fmt(n: number, digits = 2) {
  return n.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

function fmtUsd(n: number) {
  return `$${fmt(n)}`;
}

function QuotesPanel({ quotes, onSelect, selected }: {
  quotes: CryptoQuote[];
  onSelect: (q: CryptoQuote) => void;
  selected: string;
}) {
  return (
    <Card title="行情" extra={<span className="text-xs text-text-muted tabular-nums">{quotes.length} 个合约</span>} noPadding>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-border text-text-muted">
              <th className="text-left px-3 py-2 font-medium">币对</th>
              <th className="text-left px-3 py-2 font-medium">交易所</th>
              <th className="text-right px-3 py-2 font-medium">最新价</th>
              <th className="text-right px-3 py-2 font-medium">24h 涨跌</th>
              <th className="text-right px-3 py-2 font-medium">24h 量</th>
              <th className="text-right px-3 py-2 font-medium">资金费率</th>
            </tr>
          </thead>
          <tbody>
            {quotes.map(q => {
              const key = `${q.symbol}-${q.exchange}`;
              return (
                <tr
                  key={key}
                  onClick={() => onSelect(q)}
                  className={`border-b border-border/50 cursor-pointer transition-colors ${
                    selected === key ? 'bg-brand/10' : 'hover:bg-surface-tertiary/50'
                  }`}
                >
                  <td className="px-3 py-2 font-mono text-text-primary font-medium">{q.symbol}</td>
                  <td className="px-3 py-2 text-text-muted">{q.exchange}</td>
                  <td className={`px-3 py-2 text-right tabular-nums font-medium ${
                    q.price_change_percent_24h >= 0 ? 'text-profit' : 'text-loss'
                  }`}>
                    {fmtUsd(q.last_price)}
                  </td>
                  <td className={`px-3 py-2 text-right tabular-nums ${
                    q.price_change_percent_24h >= 0 ? 'text-profit' : 'text-loss'
                  }`}>
                    {q.price_change_percent_24h >= 0 ? '+' : ''}{q.price_change_percent_24h.toFixed(2)}%
                  </td>
                  <td className="px-3 py-2 text-right tabular-nums text-text-secondary">
                    {q.volume_24h >= 1000 ? `${(q.volume_24h / 1000).toFixed(1)}K` : fmt(q.volume_24h)}
                  </td>
                  <td className={`px-3 py-2 text-right tabular-nums ${
                    q.funding_rate > 0.0001 ? 'text-profit' : q.funding_rate < -0.0001 ? 'text-loss' : 'text-text-secondary'
                  }`}>
                    {(q.funding_rate * 100).toFixed(4)}%
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

function OrderPanel({ quote, onOrderPlaced }: { quote: CryptoQuote | null; onOrderPlaced: () => void }) {
  const [side, setSide] = useState<'BUY' | 'SELL'>('BUY');
  const [orderType, setOrderType] = useState<'LIMIT' | 'MARKET'>('LIMIT');
  const [price, setPrice] = useState('');
  const [qty, setQty] = useState('');
  const [leverage, setLeverage] = useState('10');
  const [reduceOnly, setReduceOnly] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    if (quote) setPrice(String(quote.last_price));
  }, [quote]);

  const handleSubmit = async () => {
    if (!quote) return;
    setSubmitting(true);
    setError('');
    try {
      await cryptoApi.placeOrder({
        symbol: quote.symbol,
        side,
        type: orderType,
        price: orderType === 'LIMIT' ? Number(price) : undefined,
        qty: Number(qty),
        leverage: Number(leverage),
        reduce_only: reduceOnly,
      });
      onOrderPlaced();
    } catch (e) {
      setError(e instanceof Error ? e.message : '下单失败');
    } finally {
      setSubmitting(false);
    }
  };

  if (!quote) {
    return (
      <Card title="下单">
        <p className="text-sm text-text-muted py-8 text-center">请从左侧选择交易对</p>
      </Card>
    );
  }

  return (
    <Card title={`下单 - ${quote.symbol}`}>
      <div className="space-y-4">
        {/* Bid/Ask */}
        <div className="grid grid-cols-2 gap-2">
          <div className="flex items-center gap-1 text-xs text-text-muted">
            <span>买一</span>
            <span className="text-profit tabular-nums">{fmtUsd(quote.bid_price)}</span>
            <span className="text-text-muted tabular-nums">×{quote.bid_qty}</span>
          </div>
          <div className="flex items-center gap-1 text-xs text-text-muted justify-end">
            <span>卖一</span>
            <span className="text-loss tabular-nums">{fmtUsd(quote.ask_price)}</span>
            <span className="text-text-muted tabular-nums">×{quote.ask_qty}</span>
          </div>
        </div>

        {/* Buy/Sell */}
        <div className="grid grid-cols-2 gap-2">
          <button
            onClick={() => setSide('BUY')}
            className={`py-2 rounded-lg text-sm font-medium transition-colors ${
              side === 'BUY'
                ? 'bg-profit text-white'
                : 'bg-surface-tertiary text-text-secondary hover:text-text-primary'
            }`}
          >
            做多
          </button>
          <button
            onClick={() => setSide('SELL')}
            className={`py-2 rounded-lg text-sm font-medium transition-colors ${
              side === 'SELL'
                ? 'bg-loss text-white'
                : 'bg-surface-tertiary text-text-secondary hover:text-text-primary'
            }`}
          >
            做空
          </button>
        </div>

        {/* Order Type */}
        <div className="grid grid-cols-2 gap-2">
          <button
            onClick={() => setOrderType('LIMIT')}
            className={`py-1.5 rounded-lg text-xs font-medium transition-colors ${
              orderType === 'LIMIT'
                ? 'bg-brand/20 text-brand-light border border-brand/40'
                : 'bg-surface-tertiary text-text-secondary'
            }`}
          >
            限价
          </button>
          <button
            onClick={() => setOrderType('MARKET')}
            className={`py-1.5 rounded-lg text-xs font-medium transition-colors ${
              orderType === 'MARKET'
                ? 'bg-brand/20 text-brand-light border border-brand/40'
                : 'bg-surface-tertiary text-text-secondary'
            }`}
          >
            市价
          </button>
        </div>

        {/* Leverage */}
        <div>
          <label className="block text-xs text-text-muted mb-1">杠杆倍数</label>
          <div className="flex gap-1.5">
            {['1', '5', '10', '20', '50'].map(lev => (
              <button
                key={lev}
                onClick={() => setLeverage(lev)}
                className={`flex-1 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                  leverage === lev
                    ? 'bg-brand/20 text-brand-light border border-brand/40'
                    : 'bg-surface-tertiary text-text-secondary hover:text-text-primary'
                }`}
              >
                {lev}x
              </button>
            ))}
          </div>
        </div>

        {/* Price */}
        {orderType === 'LIMIT' && (
          <div>
            <label className="block text-xs text-text-muted mb-1">价格 (USDT)</label>
            <input
              type="number"
              value={price}
              onChange={e => setPrice(e.target.value)}
              className="w-full bg-surface-tertiary border border-border rounded-lg px-3 py-2 text-sm text-text-primary tabular-nums focus:outline-none focus:border-brand"
            />
            <div className="flex gap-2 mt-1.5">
              <button
                onClick={() => setPrice(String(quote.bid_price))}
                className="text-[10px] text-text-muted hover:text-profit transition-colors"
              >
                买一价
              </button>
              <button
                onClick={() => setPrice(String(quote.ask_price))}
                className="text-[10px] text-text-muted hover:text-loss transition-colors"
              >
                卖一价
              </button>
              <button
                onClick={() => setPrice(String(quote.last_price))}
                className="text-[10px] text-text-muted hover:text-text-primary transition-colors"
              >
                最新价
              </button>
            </div>
          </div>
        )}

        {/* Quantity */}
        <div>
          <label className="block text-xs text-text-muted mb-1">数量</label>
          <input
            type="number"
            value={qty}
            onChange={e => setQty(e.target.value)}
            step="0.001"
            min="0"
            placeholder={quote.symbol.replace('USDT', '')}
            className="w-full bg-surface-tertiary border border-border rounded-lg px-3 py-2 text-sm text-text-primary tabular-nums focus:outline-none focus:border-brand"
          />
        </div>

        {/* Reduce Only */}
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={reduceOnly}
            onChange={e => setReduceOnly(e.target.checked)}
            className="w-3.5 h-3.5 rounded border-border bg-surface-tertiary text-brand focus:ring-brand"
          />
          <span className="text-xs text-text-muted">只减仓</span>
        </label>

        {error && (
          <div className="flex items-center gap-2 p-2 bg-loss/10 rounded-lg">
            <AlertTriangle className="w-3.5 h-3.5 text-loss shrink-0" />
            <span className="text-xs text-loss">{error}</span>
          </div>
        )}

        <button
          onClick={handleSubmit}
          disabled={submitting || !qty}
          className={`w-full py-2.5 rounded-lg text-sm font-semibold text-white transition-colors disabled:opacity-50 ${
            side === 'BUY'
              ? 'bg-profit hover:bg-profit/80'
              : 'bg-loss hover:bg-loss/80'
          }`}
        >
          {submitting ? '提交中...' : `${side === 'BUY' ? '做多' : '做空'} ${quote.symbol}`}
        </button>
      </div>
    </Card>
  );
}

export default function CryptoTrading() {
  const [quotes, setQuotes] = useState<CryptoQuote[]>([]);
  const [positions, setPositions] = useState<CryptoPosition[]>([]);
  const [orders, setOrders] = useState<CryptoOrder[]>([]);
  const [selectedQuote, setSelectedQuote] = useState<CryptoQuote | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | undefined>(undefined);

  const loadData = useCallback(async () => {
    try {
      const [q, p, o] = await Promise.all([
        cryptoApi.getQuotes(), cryptoApi.getPositions(), cryptoApi.getOrders(),
      ]);
      setQuotes(q);
      setPositions(p);
      setOrders(o);
      if (selectedQuote) {
        const key = `${selectedQuote.symbol}-${selectedQuote.exchange}`;
        const updated = q.find(x => `${x.symbol}-${x.exchange}` === key);
        if (updated) setSelectedQuote(updated);
      }
    } catch { /* retry next tick */ }
  }, [selectedQuote]);

  useEffect(() => {
    loadData();
    timerRef.current = setInterval(loadData, QUOTE_REFRESH);
    return () => clearInterval(timerRef.current);
  }, [loadData]);

  const handleCancelOrder = async (orderId: string) => {
    await cryptoApi.cancelOrder(orderId);
    await loadData();
  };

  const selectedKey = selectedQuote ? `${selectedQuote.symbol}-${selectedQuote.exchange}` : '';

  return (
    <div className="p-6 space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Bitcoin className="w-6 h-6 text-[#f7931a]" />
          <h1 className="text-xl font-semibold text-text-primary">BTC 交易</h1>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-profit animate-pulse" />
          <span className="text-xs text-text-muted">行情实时刷新 · 24/7</span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        <div className="lg:col-span-2">
          <QuotesPanel quotes={quotes} onSelect={setSelectedQuote} selected={selectedKey} />
        </div>
        <div className="lg:col-span-1">
          <OrderPanel quote={selectedQuote} onOrderPlaced={loadData} />
        </div>
        <div className="lg:col-span-1 space-y-4">
          <Card title="合约信息">
            {selectedQuote ? (
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-text-muted">24h 最高</span>
                  <span className="text-profit tabular-nums">{fmtUsd(selectedQuote.high_24h)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-text-muted">24h 最低</span>
                  <span className="text-loss tabular-nums">{fmtUsd(selectedQuote.low_24h)}</span>
                </div>
                <div className="h-px bg-border/50 my-1" />
                <div className="flex justify-between">
                  <span className="text-text-muted">24h 成交量</span>
                  <span className="tabular-nums text-text-primary">
                    {fmt(selectedQuote.volume_24h, 2)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-text-muted">24h 成交额</span>
                  <span className="tabular-nums text-text-primary">
                    ${(selectedQuote.turnover_24h / 1e9).toFixed(2)}B
                  </span>
                </div>
                <div className="h-px bg-border/50 my-1" />
                <div className="flex justify-between">
                  <span className="text-text-muted">资金费率</span>
                  <span className={`tabular-nums ${
                    selectedQuote.funding_rate > 0 ? 'text-profit' : 'text-loss'
                  }`}>
                    {(selectedQuote.funding_rate * 100).toFixed(4)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-text-muted">下次结算</span>
                  <span className="tabular-nums text-text-primary text-xs">
                    {selectedQuote.next_funding_time.replace('T', ' ').slice(0, 16)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-text-muted">持仓量</span>
                  <span className="tabular-nums text-text-primary">
                    {fmt(selectedQuote.open_interest, 1)}
                  </span>
                </div>
              </div>
            ) : (
              <p className="text-sm text-text-muted text-center py-4">选择交易对查看</p>
            )}
          </Card>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Card title="持仓" extra={<span className="text-xs text-text-muted tabular-nums">{positions.length} 个</span>} noPadding>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border text-text-muted">
                  <th className="text-left px-3 py-2 font-medium">币对</th>
                  <th className="text-left px-3 py-2 font-medium">方向</th>
                  <th className="text-right px-3 py-2 font-medium">数量</th>
                  <th className="text-right px-3 py-2 font-medium">开仓价</th>
                  <th className="text-right px-3 py-2 font-medium">强平价</th>
                  <th className="text-right px-3 py-2 font-medium">盈亏</th>
                  <th className="text-right px-3 py-2 font-medium">杠杆</th>
                </tr>
              </thead>
              <tbody>
                {positions.map(p => (
                  <tr key={`${p.symbol}-${p.side}`} className="border-b border-border/50 hover:bg-surface-tertiary/50">
                    <td className="px-3 py-2 font-mono text-text-primary">{p.symbol}</td>
                    <td className="px-3 py-2">
                      <StatusBadge
                        variant={p.side === 'LONG' ? 'success' : 'error'}
                        label={p.side === 'LONG' ? '多' : '空'}
                      />
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums">{p.size}</td>
                    <td className="px-3 py-2 text-right tabular-nums">{fmtUsd(p.entry_price)}</td>
                    <td className="px-3 py-2 text-right tabular-nums text-warning">{fmtUsd(p.liq_price)}</td>
                    <td className={`px-3 py-2 text-right tabular-nums font-medium ${p.unrealized_pnl >= 0 ? 'text-profit' : 'text-loss'}`}>
                      {fmtUsd(p.unrealized_pnl)}
                    </td>
                    <td className="px-3 py-2 text-right">
                      <span className="inline-flex items-center px-1.5 py-0.5 rounded bg-brand/10 text-brand-light text-xs tabular-nums">
                        {p.leverage}x
                      </span>
                    </td>
                  </tr>
                ))}
                {positions.length === 0 && (
                  <tr><td colSpan={7} className="px-3 py-6 text-center text-text-muted">暂无持仓</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </Card>

        <Card title="委托" extra={
          <span className="text-xs text-text-muted tabular-nums">
            {orders.filter(o => o.status === 'NEW' || o.status === 'PARTIALLY_FILLED').length} 挂单
          </span>
        } noPadding>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border text-text-muted">
                  <th className="text-left px-3 py-2 font-medium">币对</th>
                  <th className="text-left px-3 py-2 font-medium">方向</th>
                  <th className="text-left px-3 py-2 font-medium">类型</th>
                  <th className="text-right px-3 py-2 font-medium">价格</th>
                  <th className="text-right px-3 py-2 font-medium">数量</th>
                  <th className="text-left px-3 py-2 font-medium">状态</th>
                  <th className="px-3 py-2 font-medium"></th>
                </tr>
              </thead>
              <tbody>
                {orders.map(o => (
                  <tr key={o.order_id} className="border-b border-border/50 hover:bg-surface-tertiary/50">
                    <td className="px-3 py-2 font-mono text-text-primary">{o.symbol}</td>
                    <td className="px-3 py-2">
                      <StatusBadge
                        variant={o.side === 'BUY' ? 'success' : 'error'}
                        label={o.side === 'BUY' ? '买' : '卖'}
                      />
                    </td>
                    <td className="px-3 py-2 text-text-secondary">
                      {o.type === 'LIMIT' ? '限价' : o.type === 'MARKET' ? '市价' : o.type === 'STOP_LIMIT' ? '止损限价' : '止损市价'}
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums">{fmtUsd(o.price)}</td>
                    <td className="px-3 py-2 text-right tabular-nums">{o.filled_qty}/{o.qty}</td>
                    <td className="px-3 py-2">
                      <StatusBadge
                        variant={
                          o.status === 'FILLED' ? 'success'
                          : o.status === 'NEW' ? 'warning'
                          : o.status === 'PARTIALLY_FILLED' ? 'info'
                          : o.status === 'CANCELLED' ? 'neutral'
                          : 'error'
                        }
                        label={
                          o.status === 'FILLED' ? '已成'
                          : o.status === 'NEW' ? '挂单'
                          : o.status === 'PARTIALLY_FILLED' ? '部成'
                          : o.status === 'CANCELLED' ? '已撤'
                          : '拒绝'
                        }
                      />
                    </td>
                    <td className="px-3 py-2">
                      {(o.status === 'NEW' || o.status === 'PARTIALLY_FILLED') && (
                        <button
                          onClick={() => handleCancelOrder(o.order_id)}
                          className="text-xs text-loss hover:underline"
                        >
                          撤单
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
                {orders.length === 0 && (
                  <tr><td colSpan={7} className="px-3 py-6 text-center text-text-muted">暂无委托</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    </div>
  );
}
