import { useState, useMemo } from 'react';
import { AlertTriangle, Search } from 'lucide-react';
import { useQuotes } from '@/hooks/useMarketData';
import { usePositions } from '@/hooks/usePositions';
import { useOrders, usePlaceOrder, useCancelOrder } from '@/hooks/useOrders';
import { useConfirm } from '@/components/ui/ConfirmDialog';
import { useToast } from '@/components/ui/Toast';
import { Button } from '@/components/ui/Button';
import { Tabs } from '@/components/ui/Tabs';
import { SearchInput } from '@/components/ui/SearchInput';
import Card from '@/components/Card';
import StatusBadge from '@/components/StatusBadge';
import KLineChart from '@/components/charts/KLineChart';
import { fmt, fmtCny } from '@/lib/format';
import { cn } from '@/lib/cn';
import type { Quote, Order } from '@/types';

function OrderPanel({ quote, onOrderPlaced }: { quote: Quote | null; onOrderPlaced: () => void }) {
  const [direction, setDirection] = useState<'BUY' | 'SELL'>('BUY');
  const [offset, setOffset] = useState<'OPEN' | 'CLOSE'>('OPEN');
  const [price, setPrice] = useState('');
  const [volume, setVolume] = useState('1');
  const placeOrder = usePlaceOrder();
  const confirm = useConfirm();
  const toast = useToast();

  const syncPrice = (q: Quote | null) => {
    if (q) setPrice(String(q.last_price));
  };
  if (quote && !price) syncPrice(quote);

  const handleSubmit = async () => {
    if (!quote) return;
    const ok = await confirm({
      title: `确认${direction === 'BUY' ? '买入' : '卖出'}${offset === 'OPEN' ? '开仓' : '平仓'}`,
      description: `${quote.instrument_id} · ${direction === 'BUY' ? '买' : '卖'} ${volume} 手 @ ¥${price}`,
      variant: direction === 'SELL' && offset === 'CLOSE' ? 'destructive' : 'default',
      confirmText: '确认下单',
    });
    if (!ok) return;

    try {
      await placeOrder.mutateAsync({
        instrument_id: quote.instrument_id,
        direction,
        offset,
        price: Number(price),
        volume: Number(volume),
      });
      toast.success('下单成功');
      onOrderPlaced();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : '下单失败');
    }
  };

  if (!quote) {
    return (
      <Card title="下单">
        <p className="text-sm text-text-muted py-8 text-center">请从左侧选择合约</p>
      </Card>
    );
  }

  return (
    <Card title={`下单 - ${quote.instrument_id}`}>
      <div className="space-y-4">
        <div className="grid grid-cols-2 gap-2">
          <div className="flex items-center gap-1 text-xs text-text-muted">
            <span>买一</span>
            <span className="text-profit tabular-nums">{fmt(quote.bid_price1)}</span>
            <span className="text-text-muted tabular-nums">×{quote.bid_volume1}</span>
          </div>
          <div className="flex items-center gap-1 text-xs text-text-muted justify-end">
            <span>卖一</span>
            <span className="text-loss tabular-nums">{fmt(quote.ask_price1)}</span>
            <span className="text-text-muted tabular-nums">×{quote.ask_volume1}</span>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2">
          <Button
            variant={direction === 'BUY' ? 'profit' : 'secondary'}
            onClick={() => setDirection('BUY')}
            className="py-2"
          >
            买入
          </Button>
          <Button
            variant={direction === 'SELL' ? 'loss' : 'secondary'}
            onClick={() => setDirection('SELL')}
            className="py-2"
          >
            卖出
          </Button>
        </div>

        <div className="grid grid-cols-2 gap-2">
          <Button
            variant={offset === 'OPEN' ? 'primary' : 'secondary'}
            size="sm"
            onClick={() => setOffset('OPEN')}
          >
            开仓
          </Button>
          <Button
            variant={offset === 'CLOSE' ? 'primary' : 'secondary'}
            size="sm"
            onClick={() => setOffset('CLOSE')}
          >
            平仓
          </Button>
        </div>

        <div>
          <label className="block text-xs text-text-muted mb-1">价格</label>
          <input
            type="number"
            value={price}
            onChange={(e) => setPrice(e.target.value)}
            className="w-full bg-surface-tertiary border border-border rounded-lg px-3 py-2 text-sm text-text-primary tabular-nums focus:outline-none focus:border-brand focus:ring-1 focus:ring-focus-ring"
          />
          <div className="flex gap-2 mt-1.5">
            {[
              { label: '买一价', val: quote.bid_price1 },
              { label: '卖一价', val: quote.ask_price1 },
              { label: '最新价', val: quote.last_price },
            ].map(({ label, val }) => (
              <button
                key={label}
                type="button"
                onClick={() => setPrice(String(val))}
                className="text-[10px] text-text-muted hover:text-text-primary transition-colors"
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        <div>
          <label className="block text-xs text-text-muted mb-1">数量</label>
          <input
            type="number"
            value={volume}
            onChange={(e) => setVolume(e.target.value)}
            min="1"
            className="w-full bg-surface-tertiary border border-border rounded-lg px-3 py-2 text-sm text-text-primary tabular-nums focus:outline-none focus:border-brand focus:ring-1 focus:ring-focus-ring"
          />
        </div>

        {placeOrder.error && (
          <div className="flex items-center gap-2 p-2 bg-loss/10 rounded-lg">
            <AlertTriangle className="w-3.5 h-3.5 text-loss shrink-0" />
            <span className="text-xs text-loss">{placeOrder.error.message}</span>
          </div>
        )}

        <Button
          variant={direction === 'BUY' ? 'profit' : 'loss'}
          size="lg"
          className="w-full"
          onClick={handleSubmit}
          loading={placeOrder.isPending}
        >
          {`${direction === 'BUY' ? '买入' : '卖出'}${offset === 'OPEN' ? '开仓' : '平仓'}`}
        </Button>
      </div>
    </Card>
  );
}

export default function Trading() {
  const { data: quotes = [] } = useQuotes();
  const { data: positions = [] } = usePositions();
  const { data: orders = [] } = useOrders();
  const cancelOrder = useCancelOrder();
  const confirm = useConfirm();
  const toast = useToast();
  const [selectedQuote, setSelectedQuote] = useState<Quote | null>(null);
  const [search, setSearch] = useState('');
  const [bottomTab, setBottomTab] = useState('positions');

  const filteredQuotes = useMemo(() => {
    if (!search) return quotes;
    const q = search.toUpperCase();
    return quotes.filter((x) => x.instrument_id.toUpperCase().includes(q) || x.exchange.toUpperCase().includes(q));
  }, [quotes, search]);

  const handleCancelOrder = async (o: Order) => {
    const ok = await confirm({
      title: '确认撤单',
      description: `撤回 ${o.instrument_id} ${o.direction === 'BUY' ? '买' : '卖'} ${o.volume} 手 @ ¥${fmt(o.price)}`,
      variant: 'warning',
      confirmText: '确认撤单',
    });
    if (!ok) return;
    try {
      await cancelOrder.mutateAsync(o.order_id);
      toast.success('撤单成功');
    } catch {
      toast.error('撤单失败');
    }
  };

  return (
    <div className="px-[3%] py-[2%] space-y-[clamp(0.75rem,1.5vw,1rem)]">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold text-text-primary">交易</h1>
        <div className="flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-profit animate-pulse" />
          <span className="text-xs text-text-muted">行情实时刷新</span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        {/* Quotes panel with search */}
        <div className="lg:col-span-2">
          <Card title="行情" extra={<span className="text-xs text-text-muted tabular-nums">{filteredQuotes.length} 个合约</span>} noPadding>
            <div className="px-3 pt-3">
              <SearchInput
                placeholder="搜索合约..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                onClear={() => setSearch('')}
              />
            </div>
            <div className="overflow-x-auto mt-2">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-border text-text-muted">
                    <th className="text-left px-3 py-2 font-medium">合约</th>
                    <th className="text-left px-3 py-2 font-medium">交易所</th>
                    <th className="text-right px-3 py-2 font-medium">最新</th>
                    <th className="text-right px-3 py-2 font-medium">涨跌幅</th>
                    <th className="text-right px-3 py-2 font-medium">成交量</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredQuotes.map((q) => (
                    <tr
                      key={q.instrument_id}
                      tabIndex={0}
                      role="button"
                      onClick={() => setSelectedQuote(q)}
                      onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && setSelectedQuote(q)}
                      className={cn(
                        'border-b border-border/50 cursor-pointer transition-colors',
                        'focus:outline-none focus-visible:ring-1 focus-visible:ring-focus-ring',
                        selectedQuote?.instrument_id === q.instrument_id ? 'bg-brand/10' : 'hover:bg-surface-tertiary/50',
                      )}
                    >
                      <td className="px-3 py-2 font-mono text-text-primary font-medium">{q.instrument_id}</td>
                      <td className="px-3 py-2 text-text-muted">{q.exchange}</td>
                      <td className={`px-3 py-2 text-right tabular-nums font-medium ${q.change >= 0 ? 'text-profit' : 'text-loss'}`}>
                        {fmt(q.last_price)}
                      </td>
                      <td className={`px-3 py-2 text-right tabular-nums ${q.change_percent >= 0 ? 'text-profit' : 'text-loss'}`}>
                        {q.change_percent >= 0 ? '+' : ''}{q.change_percent.toFixed(2)}%
                      </td>
                      <td className="px-3 py-2 text-right tabular-nums text-text-secondary">{q.volume.toLocaleString()}</td>
                    </tr>
                  ))}
                  {filteredQuotes.length === 0 && (
                    <tr>
                      <td colSpan={5} className="px-3 py-8 text-center text-text-muted">
                        {search ? (
                          <div className="flex items-center justify-center gap-2">
                            <Search className="w-4 h-4" />
                            <span>未找到匹配的合约</span>
                          </div>
                        ) : '暂无行情数据'}
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </Card>
        </div>

        <div className="lg:col-span-1">
          <OrderPanel quote={selectedQuote} onOrderPlaced={() => {}} />
        </div>

        <div className="lg:col-span-1 space-y-4">
          <Card title="盘口数据">
            {selectedQuote ? (
              <div className="space-y-2 text-sm">
                {[
                  { label: '涨停', value: fmt(selectedQuote.upper_limit), color: 'text-profit' },
                  { label: '跌停', value: fmt(selectedQuote.lower_limit), color: 'text-loss' },
                  null,
                  { label: '开盘', value: fmt(selectedQuote.open), color: 'text-text-primary' },
                  { label: '最高', value: fmt(selectedQuote.high), color: 'text-profit' },
                  { label: '最低', value: fmt(selectedQuote.low), color: 'text-loss' },
                  { label: '昨收', value: fmt(selectedQuote.pre_close), color: 'text-text-primary' },
                  null,
                  { label: '持仓量', value: selectedQuote.open_interest.toLocaleString(), color: 'text-text-primary' },
                  { label: '成交额', value: `${(selectedQuote.amount / 1e8).toFixed(2)}亿`, color: 'text-text-primary' },
                ].map((item, i) =>
                  item === null ? (
                    <div key={`sep-${i}`} className="h-px bg-border/50 my-1" />
                  ) : (
                    <div key={item.label} className="flex justify-between">
                      <span className="text-text-muted">{item.label}</span>
                      <span className={`tabular-nums ${item.color}`}>{item.value}</span>
                    </div>
                  ),
                )}
              </div>
            ) : (
              <p className="text-sm text-text-muted text-center py-4">选择合约查看</p>
            )}
          </Card>
        </div>
      </div>

      {selectedQuote && (
        <KLineChart symbol={selectedQuote.instrument_id} height={380} />
      )}

      {/* Bottom: Positions / Orders with working tabs */}
      <div>
        <Tabs
          tabs={[
            { id: 'positions', label: `持仓 (${positions.length})` },
            { id: 'orders', label: `委托 (${orders.filter((o) => o.status === 'PENDING').length} 挂单)` },
          ]}
          active={bottomTab}
          onChange={setBottomTab}
        />

        <div className="mt-2">
          {bottomTab === 'positions' && (
            <Card noPadding>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-border text-text-muted">
                      <th className="text-left px-3 py-2 font-medium">合约</th>
                      <th className="text-left px-3 py-2 font-medium">方向</th>
                      <th className="text-right px-3 py-2 font-medium">持仓</th>
                      <th className="text-right px-3 py-2 font-medium">可用</th>
                      <th className="text-right px-3 py-2 font-medium">均价</th>
                      <th className="text-right px-3 py-2 font-medium">浮盈</th>
                    </tr>
                  </thead>
                  <tbody>
                    {positions.map((p) => (
                      <tr key={`${p.instrument_id}-${p.direction}`} className="border-b border-border/50 hover:bg-surface-tertiary/50">
                        <td className="px-3 py-2 font-mono text-text-primary">{p.instrument_id}</td>
                        <td className="px-3 py-2">
                          <StatusBadge variant={p.direction === 'LONG' ? 'success' : 'error'} label={p.direction === 'LONG' ? '多' : '空'} />
                        </td>
                        <td className="px-3 py-2 text-right tabular-nums">{p.volume}</td>
                        <td className="px-3 py-2 text-right tabular-nums">{p.available}</td>
                        <td className="px-3 py-2 text-right tabular-nums">{fmt(p.avg_open_price)}</td>
                        <td className={`px-3 py-2 text-right tabular-nums font-medium ${p.float_profit >= 0 ? 'text-profit' : 'text-loss'}`}>
                          {fmtCny(p.float_profit)}
                        </td>
                      </tr>
                    ))}
                    {positions.length === 0 && (
                      <tr><td colSpan={6} className="px-3 py-6 text-center text-text-muted">暂无持仓</td></tr>
                    )}
                  </tbody>
                </table>
              </div>
            </Card>
          )}

          {bottomTab === 'orders' && (
            <Card noPadding>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-border text-text-muted">
                      <th className="text-left px-3 py-2 font-medium">合约</th>
                      <th className="text-left px-3 py-2 font-medium">方向</th>
                      <th className="text-left px-3 py-2 font-medium">开平</th>
                      <th className="text-right px-3 py-2 font-medium">价格</th>
                      <th className="text-right px-3 py-2 font-medium">数量</th>
                      <th className="text-left px-3 py-2 font-medium">状态</th>
                      <th className="px-3 py-2 font-medium"></th>
                    </tr>
                  </thead>
                  <tbody>
                    {orders.map((o) => (
                      <tr key={o.order_id} className="border-b border-border/50 hover:bg-surface-tertiary/50">
                        <td className="px-3 py-2 font-mono text-text-primary">{o.instrument_id}</td>
                        <td className="px-3 py-2">
                          <StatusBadge variant={o.direction === 'BUY' ? 'success' : 'error'} label={o.direction === 'BUY' ? '买' : '卖'} />
                        </td>
                        <td className="px-3 py-2 text-text-secondary">{o.offset === 'OPEN' ? '开' : '平'}</td>
                        <td className="px-3 py-2 text-right tabular-nums">{fmt(o.price)}</td>
                        <td className="px-3 py-2 text-right tabular-nums">{o.traded_volume}/{o.volume}</td>
                        <td className="px-3 py-2">
                          <StatusBadge
                            variant={o.status === 'FILLED' ? 'success' : o.status === 'PENDING' ? 'warning' : o.status === 'CANCELLED' ? 'neutral' : 'error'}
                            label={o.status === 'FILLED' ? '已成' : o.status === 'PENDING' ? '挂单' : o.status === 'PARTIAL' ? '部成' : o.status === 'CANCELLED' ? '已撤' : '拒绝'}
                          />
                        </td>
                        <td className="px-3 py-2">
                          {o.status === 'PENDING' && (
                            <Button variant="destructive" size="sm" onClick={() => handleCancelOrder(o)}>
                              撤单
                            </Button>
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
          )}
        </div>
      </div>
    </div>
  );
}
