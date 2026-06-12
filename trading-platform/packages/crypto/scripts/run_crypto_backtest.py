"""Run backtests on all crypto strategies using downloaded historical data.

Usage:
    python scripts/run_crypto_backtest.py [--symbol BTCUSDT] [--timeframe 1h] [--year 2024]
"""

from __future__ import annotations

import asyncio
import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from datahub.crypto_loader import CryptoDataLoader
from strategy.base import Signal, SignalType, StrategyConfig, Position, OrderSide
from strategy.btc.funding_rate_alpha import FundingRateAlphaStrategy
from strategy.btc.cross_sectional_momentum import TimeSeriesMomentumStrategy
from strategy.btc.funding_meta_ensemble import FundingMetaEnsembleStrategy
from strategy.btc.regime_detector import MarketRegimeDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class KellyPositionSizer:
    """Rolling Kelly Criterion position sizing.

    f* = W - (1-W)/R  where W=win_rate, R=avg_win/avg_loss
    Applies fractional Kelly (default half-Kelly) with floor/ceiling bounds.
    Uses a warm-up period with a conservative fixed fraction before enough
    trade data exists to estimate Kelly reliably.
    """

    def __init__(
        self,
        fraction: float = 0.5,
        min_pct: float = 0.02,
        max_pct: float = 0.25,
        warmup_trades: int = 10,
        warmup_pct: float = 0.05,
    ) -> None:
        self.fraction = fraction
        self.min_pct = min_pct
        self.max_pct = max_pct
        self.warmup_trades = warmup_trades
        self.warmup_pct = warmup_pct
        self._wins: list[float] = []
        self._losses: list[float] = []

    def record_trade(self, pnl: float) -> None:
        if pnl > 0:
            self._wins.append(pnl)
        elif pnl < 0:
            self._losses.append(abs(pnl))

    @property
    def total_trades(self) -> int:
        return len(self._wins) + len(self._losses)

    @property
    def kelly_fraction(self) -> float:
        if self.total_trades < self.warmup_trades:
            return self.warmup_pct
        n = self.total_trades
        w = len(self._wins) / n
        avg_win = sum(self._wins) / len(self._wins) if self._wins else 0.0
        avg_loss = sum(self._losses) / len(self._losses) if self._losses else 1.0
        if avg_loss <= 0:
            return self.max_pct
        r = avg_win / avg_loss
        kelly = w - (1 - w) / r
        sized = kelly * self.fraction
        return max(self.min_pct, min(sized, self.max_pct))

    def summary(self) -> dict[str, Any]:
        n = self.total_trades
        w = len(self._wins) / n if n > 0 else 0.0
        avg_win = sum(self._wins) / len(self._wins) if self._wins else 0.0
        avg_loss = sum(self._losses) / len(self._losses) if self._losses else 0.0
        r = avg_win / avg_loss if avg_loss > 0 else 0.0
        raw_kelly = w - (1 - w) / r if r > 0 else 0.0
        return {
            "total_trades": n,
            "win_rate": round(w, 4),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "payoff_ratio": round(r, 4),
            "raw_kelly": round(raw_kelly, 4),
            "fractional_kelly": round(raw_kelly * self.fraction, 4),
            "applied_pct": round(self.kelly_fraction, 4),
        }


class SimpleBacktester:
    """Lightweight bar-by-bar backtester for evaluating crypto strategies."""

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        kelly_fraction: float = 0.5,
        kelly_max_pct: float = 0.25,
    ) -> None:
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.kelly_fraction = kelly_fraction
        self.kelly_max_pct = kelly_max_pct

    async def run(
        self,
        strategy: Any,
        bars: pd.DataFrame,
        symbol: str = "BTCUSDT",
    ) -> dict[str, Any]:
        capital = self.initial_capital
        position_qty = 0.0
        position_side = None  # "long" or "short"
        entry_price = 0.0

        trades: list[dict[str, Any]] = []
        equity_curve: list[float] = [capital]
        drawdown_curve: list[float] = [0.0]
        peak_equity = capital
        regime_history: list[str] = []

        regime_detector = MarketRegimeDetector()
        total_commission = 0.0
        sizer = KellyPositionSizer(
            fraction=self.kelly_fraction,
            max_pct=self.kelly_max_pct,
        )

        _CORE_COLS = {"open", "high", "low", "close", "volume", "taker_buy_volume", "open_time", "close_time"}
        pending_signals: list = []

        for idx, row in bars.iterrows():
            bar = {
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row.get("volume", 0),
                "taker_buy_volume": row.get("taker_buy_volume", row.get("volume", 0) * 0.5),
            }
            for col in row.index:
                if col not in _CORE_COLS and pd.notna(row[col]):
                    bar[col] = row[col]

            regime = regime_detector.update(bar["high"], bar["low"], bar["close"])
            regime_history.append(regime.value)

            exec_price = bar["open"]
            for sig in pending_signals:
                pos_pct = sizer.kelly_fraction
                trade_value = capital * pos_pct

                if sig.signal_type == SignalType.LONG_ENTRY and position_qty == 0:
                    slip_price = exec_price * (1 + self.slippage_pct)
                    qty = trade_value / slip_price
                    commission = trade_value * self.commission_pct
                    capital -= commission
                    total_commission += commission
                    position_qty = qty
                    position_side = "long"
                    entry_price = slip_price
                    strategy.update_position(Position(
                        symbol=symbol, side=OrderSide.BUY,
                        qty=qty, avg_price=slip_price,
                    ))
                    trades.append({
                        "type": "long_entry", "price": slip_price,
                        "qty": qty, "capital_after": capital,
                        "time": str(row.get("open_time", idx)),
                        "regime": regime.value,
                        "reason": sig.reason,
                    })

                elif sig.signal_type == SignalType.SHORT_ENTRY and position_qty == 0:
                    slip_price = exec_price * (1 - self.slippage_pct)
                    qty = trade_value / slip_price
                    commission = trade_value * self.commission_pct
                    capital -= commission
                    total_commission += commission
                    position_qty = qty
                    position_side = "short"
                    entry_price = slip_price
                    strategy.update_position(Position(
                        symbol=symbol, side=OrderSide.SELL,
                        qty=qty, avg_price=slip_price,
                    ))
                    trades.append({
                        "type": "short_entry", "price": slip_price,
                        "qty": qty, "capital_after": capital,
                        "time": str(row.get("open_time", idx)),
                        "regime": regime.value,
                        "reason": sig.reason,
                    })

                elif sig.signal_type == SignalType.LONG_EXIT and position_side == "long":
                    slip_price = exec_price * (1 - self.slippage_pct)
                    pnl = (slip_price - entry_price) * position_qty
                    exit_notional = slip_price * position_qty
                    commission = exit_notional * self.commission_pct
                    net_pnl = pnl - commission
                    capital += net_pnl
                    total_commission += commission
                    sizer.record_trade(net_pnl)
                    strategy.remove_position(symbol)
                    trades.append({
                        "type": "long_exit", "price": slip_price,
                        "pnl": net_pnl, "capital_after": capital,
                        "time": str(row.get("open_time", idx)),
                        "regime": regime.value,
                        "reason": sig.reason,
                        "kelly_pct": round(pos_pct, 4),
                    })
                    position_qty = 0
                    position_side = None

                elif sig.signal_type == SignalType.SHORT_EXIT and position_side == "short":
                    slip_price = exec_price * (1 + self.slippage_pct)
                    pnl = (entry_price - slip_price) * position_qty
                    exit_notional = slip_price * position_qty
                    commission = exit_notional * self.commission_pct
                    net_pnl = pnl - commission
                    capital += net_pnl
                    total_commission += commission
                    sizer.record_trade(net_pnl)
                    strategy.remove_position(symbol)
                    trades.append({
                        "type": "short_exit", "price": slip_price,
                        "pnl": net_pnl, "capital_after": capital,
                        "time": str(row.get("open_time", idx)),
                        "regime": regime.value,
                        "reason": sig.reason,
                        "kelly_pct": round(pos_pct, 4),
                    })
                    position_qty = 0
                    position_side = None

            pending_signals.clear()

            try:
                signals = await strategy.on_bar(symbol, bar)
            except Exception as e:
                logger.debug("Strategy error on bar %s: %s", idx, e)
                signals = []
            pending_signals.extend(signals)

            unrealized = 0.0
            if position_side == "long":
                unrealized = (bar["close"] - entry_price) * position_qty
            elif position_side == "short":
                unrealized = (entry_price - bar["close"]) * position_qty

            total_equity = capital + unrealized
            equity_curve.append(total_equity)
            peak_equity = max(peak_equity, total_equity)
            dd = (peak_equity - total_equity) / peak_equity if peak_equity > 0 else 0
            drawdown_curve.append(dd)

        if position_side:
            final_price = bars.iloc[-1]["close"]
            slip = final_price * self.slippage_pct
            if position_side == "long":
                exit_p = final_price - slip
                pnl = (exit_p - entry_price) * position_qty
            else:
                exit_p = final_price + slip
                pnl = (entry_price - exit_p) * position_qty
            exit_notional = abs(exit_p * position_qty)
            commission = exit_notional * self.commission_pct
            capital += pnl - commission
            total_commission += commission

        return self._compute_metrics(
            capital, equity_curve, drawdown_curve, trades,
            total_commission, regime_history, bars, sizer,
        )

    def _compute_metrics(
        self,
        final_capital: float,
        equity_curve: list[float],
        drawdown_curve: list[float],
        trades: list[dict],
        total_commission: float,
        regime_history: list[str],
        bars: pd.DataFrame,
        sizer: KellyPositionSizer | None = None,
    ) -> dict[str, Any]:
        total_return = (final_capital - self.initial_capital) / self.initial_capital
        max_drawdown = max(drawdown_curve) if drawdown_curve else 0

        exit_trades = [t for t in trades if "pnl" in t]
        winning = [t for t in exit_trades if t["pnl"] > 0]
        losing = [t for t in exit_trades if t["pnl"] <= 0]
        win_rate = len(winning) / len(exit_trades) if exit_trades else 0

        avg_win = np.mean([t["pnl"] for t in winning]) if winning else 0
        avg_loss = abs(np.mean([t["pnl"] for t in losing])) if losing else 1
        profit_factor = (sum(t["pnl"] for t in winning) / abs(sum(t["pnl"] for t in losing))) if losing and sum(t["pnl"] for t in losing) != 0 else float("inf")

        eq = np.array(equity_curve)
        bar_returns = np.diff(eq) / eq[:-1]
        bar_returns = bar_returns[np.isfinite(bar_returns)]

        bars_per_day = {"1m": 1440, "5m": 288, "15m": 96, "30m": 48,
                        "1h": 24, "2h": 12, "4h": 6, "1d": 1}
        bpd = bars_per_day.get(bars.attrs.get("timeframe", ""), 0)
        if bpd == 0:
            bpd = max(len(bars) / max((bars.iloc[-1].get("open_time", pd.Timestamp.now()) - bars.iloc[0].get("open_time", pd.Timestamp.now())).days, 1), 1)
        annualize_factor = np.sqrt(252 * bpd)
        sharpe = (np.mean(bar_returns) / np.std(bar_returns, ddof=1) * annualize_factor) if len(bar_returns) > 1 and np.std(bar_returns, ddof=1) > 0 else 0

        calmar = total_return / max_drawdown if max_drawdown > 0 else 0

        regime_dist = {}
        for r in regime_history:
            regime_dist[r] = regime_dist.get(r, 0) + 1

        n_bars = len(bars)
        date_range = ""
        if n_bars > 0 and "open_time" in bars.columns:
            date_range = f"{bars['open_time'].iloc[0].strftime('%Y-%m-%d')} → {bars['open_time'].iloc[-1].strftime('%Y-%m-%d')}"

        result = {
            "initial_capital": self.initial_capital,
            "final_capital": round(final_capital, 2),
            "total_return": round(total_return * 100, 2),
            "max_drawdown": round(max_drawdown * 100, 2),
            "sharpe_ratio": round(sharpe, 3),
            "calmar_ratio": round(calmar, 3),
            "total_trades": len(trades),
            "round_trips": len(exit_trades),
            "win_rate": round(win_rate * 100, 1),
            "profit_factor": round(profit_factor, 3) if profit_factor != float("inf") else "inf",
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "payoff_ratio": round(avg_win / avg_loss, 3) if avg_loss > 0 else "inf",
            "total_commission": round(total_commission, 2),
            "bars_tested": n_bars,
            "date_range": date_range,
            "regime_distribution": regime_dist,
        }
        if sizer is not None:
            result["kelly"] = sizer.summary()
        result["_equity_curve"] = equity_curve
        return result


def create_strategies(
    symbol: str, bars: pd.DataFrame, all_symbols: list[str] | None = None,
) -> dict[str, Any]:
    """Create SOTA strategy instances for backtesting."""
    base_config = lambda name, params=None, syms=None: StrategyConfig(
        name=name, symbols=syms or [symbol], params=params or {}
    )

    strats: dict[str, Any] = {
        "funding_rate": FundingRateAlphaStrategy(base_config("FundingRateAlpha")),
        "ts_momentum": TimeSeriesMomentumStrategy(base_config("TSMomentum")),
        "fund_meta": FundingMetaEnsembleStrategy(base_config("FundMetaEnsemble")),
    }

    return strats


def _print_summary_table(results: dict[str, dict[str, Any]], label: str = "") -> None:
    if label:
        logger.info("\n%s", label)
    header = f"{'Strategy':<20} {'Return%':>10} {'MaxDD%':>10} {'Sharpe':>8} {'WinRate%':>10} {'PF':>8} {'Trades':>8} {'Kelly%':>8}"
    logger.info(header)
    logger.info("-" * len(header))
    for name, r in sorted(results.items(), key=lambda x: x[1].get("total_return", -9999), reverse=True):
        if "error" in r:
            logger.info("%s  ERROR: %s", name.ljust(20), r["error"])
            continue
        kelly_pct = r.get("kelly", {}).get("applied_pct", 0) * 100
        logger.info(
            "%s %10s %10s %8s %10s %8s %8d %7.1f%%",
            name.ljust(20),
            f"{r['total_return']}%",
            f"{r['max_drawdown']}%",
            r["sharpe_ratio"],
            f"{r['win_rate']}%",
            r["profit_factor"],
            r["round_trips"],
            kelly_pct,
        )


def _print_portfolio_combo(
    train_results: dict[str, dict[str, Any]],
    test_results: dict[str, dict[str, Any]],
    initial_capital: float,
) -> None:
    """Compute and display HRP portfolio combination from individual strategy results."""
    from strategy.portfolio.hrp_allocator import HRPAllocator, RiskParityAllocator

    valid_train = {k: v for k, v in train_results.items() if "error" not in v and "_equity_curve" in v}
    valid_test = {k: v for k, v in test_results.items() if "error" not in v and "_equity_curve" in v}

    if len(valid_train) < 2:
        return

    train_returns: dict[str, list[float]] = {}
    for name, res in valid_train.items():
        eq = np.array(res["_equity_curve"])
        rets = list(np.diff(eq) / eq[:-1])
        train_returns[name] = rets

    for allocator_name, allocator in [("HRP", HRPAllocator()), ("Risk Parity", RiskParityAllocator())]:
        weights = allocator.allocate(train_returns)

        if not valid_test:
            continue

        combined_eq = np.zeros(0)
        for name, w in weights.items():
            if name in valid_test and "_equity_curve" in valid_test[name]:
                eq = np.array(valid_test[name]["_equity_curve"])
                strategy_returns = np.diff(eq) / eq[:-1]
                if len(combined_eq) == 0:
                    combined_eq = np.zeros(len(strategy_returns))
                min_len = min(len(combined_eq), len(strategy_returns))
                combined_eq[:min_len] += w * strategy_returns[:min_len]

        if len(combined_eq) == 0:
            continue

        port_equity = [initial_capital]
        for r in combined_eq:
            port_equity.append(port_equity[-1] * (1 + r))

        port_eq = np.array(port_equity)
        port_return = (port_eq[-1] - initial_capital) / initial_capital
        port_dd = np.max(1 - port_eq / np.maximum.accumulate(port_eq))
        port_bar_rets = combined_eq[np.isfinite(combined_eq)]
        port_sharpe = (np.mean(port_bar_rets) / np.std(port_bar_rets, ddof=1) * np.sqrt(252 * 6)) if len(port_bar_rets) > 1 and np.std(port_bar_rets, ddof=1) > 0 else 0

        weight_str = ", ".join(f"{k}={v:.1%}" for k, v in sorted(weights.items(), key=lambda x: -x[1]))
        logger.info(
            "\n%s PORTFOLIO (OOS): Return=%.2f%%, MaxDD=%.2f%%, Sharpe=%.3f | Weights: %s",
            allocator_name, port_return * 100, port_dd * 100, port_sharpe, weight_str,
        )


async def _run_strategies(
    backtester: SimpleBacktester,
    strategies: dict[str, Any],
    bars: pd.DataFrame,
    symbol: str,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for name, strategy in strategies.items():
        logger.info("--- Running: %s ---", name)
        try:
            result = await backtester.run(strategy, bars, symbol)
            results[name] = result
        except Exception as e:
            logger.exception("Backtest failed for %s: %s", name, e)
            results[name] = {"error": str(e)}
    return results


async def main() -> None:
    parser = argparse.ArgumentParser(description="Crypto strategy backtester")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--capital", type=float, default=100_000)
    parser.add_argument("--strategies", nargs="*", default=None, help="Strategy names to test")
    parser.add_argument("--split", type=float, default=0.0,
                        help="Train/test split ratio (e.g. 0.7 = 70%% train, 30%% test). 0=no split.")
    args = parser.parse_args()

    loader = CryptoDataLoader()
    available = loader.available_symbols()
    if not available:
        logger.error("No data available in %s. Run download_crypto_data.py first.", loader.data_dir)
        return

    logger.info("Available symbols: %s", available)
    logger.info("Available timeframes for %s: %s", args.symbol, loader.available_timeframes(args.symbol))

    bars = loader.load_with_funding(args.symbol, args.timeframe, args.start, args.end)
    if bars.empty:
        logger.error("No data for %s %s", args.symbol, args.timeframe)
        return
    bars.attrs["timeframe"] = args.timeframe

    all_symbols = available

    logger.info("=" * 70)
    logger.info("CRYPTO BACKTEST — %s %s", args.symbol, args.timeframe)
    logger.info("Bars: %d | Capital: $%s", len(bars), f"{args.capital:,.0f}")

    if args.split > 0:
        split_idx = int(len(bars) * args.split)
        train_bars = bars.iloc[:split_idx].copy()
        test_bars = bars.iloc[split_idx:].copy()
        train_bars.attrs = bars.attrs.copy()
        test_bars.attrs = bars.attrs.copy()

        train_range = ""
        test_range = ""
        if "open_time" in bars.columns:
            train_range = f"{train_bars['open_time'].iloc[0].strftime('%Y-%m-%d')} → {train_bars['open_time'].iloc[-1].strftime('%Y-%m-%d')}"
            test_range = f"{test_bars['open_time'].iloc[0].strftime('%Y-%m-%d')} → {test_bars['open_time'].iloc[-1].strftime('%Y-%m-%d')}"

        logger.info("Train: %d bars (%s)", len(train_bars), train_range)
        logger.info("Test:  %d bars (%s)", len(test_bars), test_range)
        logger.info("=" * 70)

        backtester = SimpleBacktester(initial_capital=args.capital, kelly_fraction=0.5, kelly_max_pct=0.25)

        train_strats = create_strategies(args.symbol, train_bars, all_symbols)
        if args.strategies:
            train_strats = {k: v for k, v in train_strats.items() if k in args.strategies}
        train_results = await _run_strategies(backtester, train_strats, train_bars, args.symbol)

        test_strats = create_strategies(args.symbol, test_bars, all_symbols)
        if args.strategies:
            test_strats = {k: v for k, v in test_strats.items() if k in args.strategies}
        test_results = await _run_strategies(backtester, test_strats, test_bars, args.symbol)

        _print_summary_table(train_results, f"IN-SAMPLE (train {int(args.split*100)}%)")
        _print_summary_table(test_results, f"OUT-OF-SAMPLE (test {int((1-args.split)*100)}%)")

        _print_portfolio_combo(train_results, test_results, args.capital)

        def _strip_internal(res: dict) -> dict:
            return {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} if isinstance(v, dict) else v for k, v in res.items()}

        combined = {
            "train": _strip_internal(train_results),
            "test": _strip_internal(test_results),
            "split_ratio": args.split,
        }
        output_dir = Path("models")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"backtest_{args.symbol}_{args.timeframe}_split.json"
        with open(output_path, "w") as f:
            json.dump(combined, f, indent=2, default=str)
        logger.info("\nResults saved to %s", output_path)

    else:
        logger.info("=" * 70)
        backtester = SimpleBacktester(initial_capital=args.capital, kelly_fraction=0.5, kelly_max_pct=0.25)
        strategies = create_strategies(args.symbol, bars, all_symbols)
        if args.strategies:
            strategies = {k: v for k, v in strategies.items() if k in args.strategies}
        results = await _run_strategies(backtester, strategies, bars, args.symbol)
        _print_summary_table(results, "RESULTS SUMMARY")

        _print_portfolio_combo(results, results, args.capital)

        output_dir = Path("models")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"backtest_{args.symbol}_{args.timeframe}.json"
        serializable = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} if isinstance(v, dict) else v for k, v in results.items()}
        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        logger.info("\nResults saved to %s", output_path)


if __name__ == "__main__":
    asyncio.run(main())
