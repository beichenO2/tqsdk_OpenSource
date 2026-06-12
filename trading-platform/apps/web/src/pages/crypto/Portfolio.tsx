import { useEffect, useState, useCallback } from 'react';
import { RefreshCw, Wallet } from 'lucide-react';
import { cryptoApi } from './api';
import Card from '@/components/Card';
import StatCard from '@/components/StatCard';
import type { CryptoAccount, CryptoPosition } from './types';

function fmt(n: number, digits = 2) {
  return n.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

function fmtUsd(n: number) {
  return `$${fmt(n)}`;
}

interface IAssetRow {
  asset: string;
  free: number;
  locked: number;
  total: number;
  usdValue: number;
}

function AssetRowView({ asset, free, locked, total, usdValue }: IAssetRow) {
  const decimals = asset === 'USDT' ? 2 : 4;
  return (
    <tr className="border-b border-border/50 hover:bg-surface-tertiary/50">
      <td className="px-4 py-3 font-mono font-medium text-text-primary">{asset}</td>
      <td className="px-4 py-3 text-right font-mono tabular-nums text-text-primary">{free.toFixed(decimals)}</td>
      <td className="px-4 py-3 text-right font-mono tabular-nums text-text-muted">{locked.toFixed(decimals)}</td>
      <td className="px-4 py-3 text-right font-mono tabular-nums text-text-primary">{total.toFixed(decimals)}</td>
      <td className="px-4 py-3 text-right font-mono tabular-nums text-text-primary">{fmtUsd(usdValue)}</td>
    </tr>
  );
}

export default function CryptoPortfolio() {
  const [account, setAccount] = useState<CryptoAccount | null>(null);
  const [positions, setPositions] = useState<CryptoPosition[]>([]);
  const [refreshing, setRefreshing] = useState(false);

  const loadData = useCallback(async () => {
    try {
      const [a, p] = await Promise.all([
        cryptoApi.getAccount(),
        cryptoApi.getPositions(),
      ]);
      setAccount(a);
      setPositions(p);
    } catch { /* retry manually */ }
  }, []);

  useEffect(() => { loadData(); }, [loadData]);

  const handleRefresh = async () => {
    setRefreshing(true);
    await loadData();
    setRefreshing(false);
  };

  const acct = account ?? {
    total_equity: 0, available_balance: 0, unrealized_pnl: 0,
    realized_pnl_24h: 0, margin_ratio: 0, btc_equivalent: 0, total_margin: 0,
  } as CryptoAccount;

  const assets: IAssetRow[] = [
    { asset: 'USDT', free: acct.available_balance, locked: acct.total_margin, total: acct.total_equity, usdValue: acct.total_equity },
    { asset: 'BTC', free: acct.btc_equivalent * 0.7, locked: acct.btc_equivalent * 0.3, total: acct.btc_equivalent, usdValue: acct.btc_equivalent * 67234 },
  ];

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Wallet className="w-6 h-6 text-brand" />
          <h1 className="text-xl font-semibold text-text-primary">资产管理</h1>
        </div>
        <button
          onClick={handleRefresh}
          className="p-1.5 rounded-lg text-text-muted hover:text-text-primary hover:bg-surface-tertiary transition-colors"
          title="刷新数据"
        >
          <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
        </button>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard label="总资产 (USDT)" value={fmtUsd(acct.total_equity)} sub={`≈ ${acct.btc_equivalent.toFixed(4)} BTC`} trend="neutral" />
        <StatCard label="可用余额" value={fmtUsd(acct.available_balance)} trend="neutral" />
        <StatCard label="保证金占用" value={fmtUsd(acct.total_margin)} trend="neutral" />
        <StatCard label="未实现盈亏" value={fmtUsd(acct.unrealized_pnl)} trend={acct.unrealized_pnl >= 0 ? 'up' : 'down'} />
      </div>

      <Card title="资产明细" extra={<span className="text-xs text-text-muted">{assets.length} 个币种</span>} noPadding>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border text-text-muted text-xs">
                <th className="text-left px-4 py-2.5 font-medium">币种</th>
                <th className="text-right px-4 py-2.5 font-medium">可用</th>
                <th className="text-right px-4 py-2.5 font-medium">冻结</th>
                <th className="text-right px-4 py-2.5 font-medium">总计</th>
                <th className="text-right px-4 py-2.5 font-medium">估值 (USDT)</th>
              </tr>
            </thead>
            <tbody>
              {assets.map(a => <AssetRowView key={a.asset} {...a} />)}
            </tbody>
          </table>
        </div>
      </Card>

      {positions.length > 0 && (
        <Card title="持仓资产" extra={<span className="text-xs text-text-muted tabular-nums">{positions.length} 个</span>} noPadding>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-text-muted text-xs">
                  <th className="text-left px-4 py-2.5 font-medium">币对</th>
                  <th className="text-left px-4 py-2.5 font-medium">方向</th>
                  <th className="text-right px-4 py-2.5 font-medium">数量</th>
                  <th className="text-right px-4 py-2.5 font-medium">开仓价</th>
                  <th className="text-right px-4 py-2.5 font-medium">标记价</th>
                  <th className="text-right px-4 py-2.5 font-medium">未实现盈亏</th>
                  <th className="text-right px-4 py-2.5 font-medium">杠杆</th>
                </tr>
              </thead>
              <tbody>
                {positions.map(p => (
                  <tr key={`${p.symbol}-${p.side}`} className="border-b border-border/50 hover:bg-surface-tertiary/50">
                    <td className="px-4 py-2.5 font-mono text-text-primary">{p.symbol}</td>
                    <td className="px-4 py-2.5">
                      <span className={`inline-flex items-center px-1.5 py-0.5 rounded-full text-xs font-medium ${
                        p.side === 'LONG' ? 'bg-profit/15 text-profit' : 'bg-loss/15 text-loss'
                      }`}>
                        {p.side === 'LONG' ? '多' : '空'}
                      </span>
                    </td>
                    <td className="px-4 py-2.5 text-right tabular-nums text-text-primary">{p.size}</td>
                    <td className="px-4 py-2.5 text-right tabular-nums text-text-primary">{fmtUsd(p.entry_price)}</td>
                    <td className="px-4 py-2.5 text-right tabular-nums text-text-primary">{fmtUsd(p.mark_price)}</td>
                    <td className={`px-4 py-2.5 text-right tabular-nums font-medium ${p.unrealized_pnl >= 0 ? 'text-profit' : 'text-loss'}`}>
                      {fmtUsd(p.unrealized_pnl)}
                    </td>
                    <td className="px-4 py-2.5 text-right">
                      <span className="inline-flex items-center px-1.5 py-0.5 rounded bg-brand/10 text-brand-light text-xs tabular-nums">
                        {p.leverage}x
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
}
