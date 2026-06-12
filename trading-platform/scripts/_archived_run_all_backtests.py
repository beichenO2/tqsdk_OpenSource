"""Run backtests on ALL strategies (crypto + futures) with real data.

Usage:
    python scripts/run_all_backtests.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from datahub.crypto_loader import CryptoDataLoader
from strategy.base import Signal, SignalType, StrategyConfig, Position, OrderSide
from strategy.btc.trend_following import BTCTrendFollowingStrategy
from strategy.btc.momentum import BTCMomentumStrategy
from strategy.btc.mean_reversion import BTCMeanReversionStrategy
from strategy.btc.grid import BTCGridStrategy
from strategy.btc.multifactor_strategy import BTCMultiFactorStrategy
from strategy.btc.ensemble_strategy import EnsembleStrategy
from strategy.btc.funding_rate_arb import FundingRateArbitrage
from strategy.btc.regime_detector import MarketRegimeDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CRYPTO_DATA_DIR = Path.home() / "Downloads" / "crypto_data"


class SimpleBacktester:
    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        position_size_pct: float = 0.1,
    ) -> None:
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.position_size_pct = position_size_pct

    async def run(self, strategy: Any, bars: pd.DataFrame, symbol: str = "BTCUSDT") -> dict[str, Any]:
        capital = self.initial_capital
        position_qty = 0.0
        position_side = None
        entry_price = 0.0
        trades: list[dict] = []
        equity_curve = [capital]
        peak = capital
        total_commission = 0.0
        regime = MarketRegimeDetector()

        for _, row in bars.iterrows():
            bar = {k: row[k] for k in ["open", "high", "low", "close", "volume"]}
            bar["timestamp"] = row.name if isinstance(row.name, datetime) else datetime.now(timezone.utc)
            price = float(bar["close"])

            regime.update(float(bar["high"]), float(bar["low"]), price)
            pos = Position(symbol=symbol, side=OrderSide.BUY, quantity=abs(position_qty),
                           entry_price=entry_price) if position_qty != 0 else None

            try:
                sig = await strategy.generate_signal(symbol, bar) if asyncio.iscoroutinefunction(
                    strategy.generate_signal) else strategy.generate_signal(symbol, bar)
            except Exception:
                sig = Signal(signal_type=SignalType.HOLD, strength=0, strategy_id="backtest", symbol=symbol)

            if sig.signal_type in (SignalType.LONG_ENTRY,) and position_qty == 0:
                qty = (capital * self.position_size_pct) / price
                cost = qty * price * self.commission_pct
                capital -= cost
                total_commission += cost
                fill = price * (1 + self.slippage_pct)
                position_qty = qty
                position_side = "long"
                entry_price = fill
            elif sig.signal_type in (SignalType.SHORT_ENTRY,) and position_qty == 0:
                qty = (capital * self.position_size_pct) / price
                cost = qty * price * self.commission_pct
                capital -= cost
                total_commission += cost
                fill = price * (1 - self.slippage_pct)
                position_qty = -qty
                position_side = "short"
                entry_price = fill
            elif sig.signal_type in (SignalType.LONG_EXIT, SignalType.SHORT_EXIT) and position_qty != 0:
                fill = price * (1 - self.slippage_pct) if position_side == "long" else price * (1 + self.slippage_pct)
                if position_side == "long":
                    pnl = abs(position_qty) * (fill - entry_price)
                else:
                    pnl = abs(position_qty) * (entry_price - fill)
                cost = abs(position_qty) * price * self.commission_pct
                capital += pnl - cost
                total_commission += cost
                trades.append({"pnl": pnl - cost, "side": position_side, "entry": entry_price, "exit": fill})
                position_qty = 0.0
                position_side = None

            mark = capital + (position_qty * (price - entry_price) if position_qty != 0 else 0)
            equity_curve.append(mark)
            peak = max(peak, mark)

        final = equity_curve[-1]
        total_return = (final - self.initial_capital) / self.initial_capital
        max_dd = self._calc_max_dd(equity_curve)
        sharpe = self._calc_sharpe(equity_curve)
        wins = [t for t in trades if t["pnl"] > 0]
        win_rate = len(wins) / len(trades) if trades else 0
        avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
        losses = [t for t in trades if t["pnl"] <= 0]
        avg_loss = abs(np.mean([t["pnl"] for t in losses])) if losses else 1
        profit_factor = avg_win / avg_loss if avg_loss > 0 and wins else 0

        return {
            "total_return_pct": round(total_return * 100, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "sharpe_ratio": round(sharpe, 3),
            "total_trades": len(trades),
            "win_rate_pct": round(win_rate * 100, 1),
            "profit_factor": round(profit_factor, 3),
            "total_commission": round(total_commission, 2),
            "final_equity": round(final, 2),
        }

    def _calc_max_dd(self, curve: list[float]) -> float:
        arr = np.array(curve)
        peaks = np.maximum.accumulate(arr)
        dd = (peaks - arr) / np.where(peaks > 0, peaks, 1)
        return float(np.max(dd))

    def _calc_sharpe(self, curve: list[float], rf: float = 0.03, periods: int = 252) -> float:
        arr = np.array(curve)
        rets = np.diff(arr) / arr[:-1]
        if len(rets) < 2 or np.std(rets) == 0:
            return 0.0
        excess = np.mean(rets) - rf / periods
        return float(excess / np.std(rets) * np.sqrt(periods))


def build_crypto_strategies(symbol: str) -> list[tuple[str, Any]]:
    cfg = lambda name, **kw: StrategyConfig(name=name, symbols=[symbol], params=kw)
    strategies = []
    strategies.append(("trend_following", BTCTrendFollowingStrategy(cfg("trend_following"))))
    strategies.append(("momentum", BTCMomentumStrategy(cfg("momentum"))))
    strategies.append(("mean_reversion", BTCMeanReversionStrategy(cfg("mean_reversion"))))
    strategies.append(("grid", BTCGridStrategy(cfg("grid"))))
    strategies.append(("multifactor", BTCMultiFactorStrategy(cfg("multifactor"))))
    try:
        strategies.append(("ensemble", EnsembleStrategy(cfg("ensemble"))))
    except Exception as e:
        logger.warning("Skipping ensemble: %s", e)
    try:
        strategies.append(("funding_rate_arb", FundingRateArbitrage(cfg("funding_rate_arb"))))
    except Exception as e:
        logger.warning("Skipping funding_rate_arb: %s", e)
    return strategies


async def run_crypto_backtests():
    loader = CryptoDataLoader(str(CRYPTO_DATA_DIR))
    symbol = "BTCUSDT"
    timeframe = "4h"

    logger.info("Loading %s %s data...", symbol, timeframe)
    try:
        bars = loader.load(symbol.lower(), timeframe)
    except Exception as e:
        logger.error("Failed to load data: %s", e)
        return

    if bars is None or len(bars) == 0:
        logger.error("No data loaded")
        return

    logger.info("Loaded %d bars for %s %s (%s ~ %s)", len(bars), symbol, timeframe,
                bars.index[0], bars.index[-1])

    bt = SimpleBacktester()
    strategies = build_crypto_strategies(symbol)
    results = []

    for name, strat in strategies:
        logger.info("Running %s...", name)
        start = time.monotonic()
        try:
            result = await bt.run(strat, bars, symbol)
            elapsed = time.monotonic() - start
            result["strategy"] = name
            result["elapsed_s"] = round(elapsed, 1)
            results.append(result)
            logger.info("  %s: return=%.1f%% dd=%.1f%% sharpe=%.3f trades=%d win=%.1f%% (%.1fs)",
                        name, result["total_return_pct"], result["max_drawdown_pct"],
                        result["sharpe_ratio"], result["total_trades"],
                        result["win_rate_pct"], elapsed)
        except Exception as e:
            logger.error("  %s FAILED: %s", name, e)

    return results


def print_results_table(results: list[dict]):
    if not results:
        print("\nNo results to display.")
        return

    header = f"{'Strategy':<20} {'Return%':>8} {'MaxDD%':>8} {'Sharpe':>8} {'Trades':>7} {'WinRate':>8} {'PF':>6}"
    print(f"\n{'='*75}")
    print("CRYPTO STRATEGY BACKTEST RESULTS (BTC 4h)")
    print(f"{'='*75}")
    print(header)
    print("-" * 75)

    for r in sorted(results, key=lambda x: x.get("sharpe_ratio", 0), reverse=True):
        print(f"{r['strategy']:<20} {r['total_return_pct']:>7.1f}% {r['max_drawdown_pct']:>7.1f}% "
              f"{r['sharpe_ratio']:>8.3f} {r['total_trades']:>7d} {r['win_rate_pct']:>7.1f}% "
              f"{r['profit_factor']:>5.2f}")

    print(f"{'='*75}")
    best = max(results, key=lambda x: x.get("sharpe_ratio", 0))
    print(f"Best strategy: {best['strategy']} (Sharpe={best['sharpe_ratio']:.3f})")


async def main():
    print("=" * 75)
    print("TRADING PLATFORM — FULL STRATEGY BACKTEST")
    print(f"Time: {datetime.now()}")
    print("=" * 75)

    results = await run_crypto_backtests()
    if results:
        print_results_table(results)


if __name__ == "__main__":
    asyncio.run(main())
