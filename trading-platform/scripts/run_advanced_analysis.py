#!/usr/bin/env python3
"""Walk-Forward validation + Monte Carlo simulation on best strategies.

Usage:
    python scripts/run_advanced_analysis.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys

import os as _os
import sys as _sys

try:
    from polarisor_port_sdk import submit_task as _sdk_submit, complete_task as _sdk_complete
except ImportError:
    _sdk_submit = _sdk_complete = None

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from datahub.crypto_loader import CryptoDataLoader
from strategy.base import Signal, SignalType, StrategyConfig, Position, OrderSide
from strategy.btc.momentum import BTCMomentumStrategy
from strategy.btc.trend_following import BTCTrendFollowingStrategy
from strategy.btc.regime_detector import MarketRegimeDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


async def simple_backtest(
    strategy: Any,
    bars: pd.DataFrame,
    symbol: str = "BTCUSDT",
    initial_capital: float = 100_000.0,
) -> dict[str, Any]:
    """Lightweight async backtester returning metrics + trade list."""
    capital = initial_capital
    pos_qty = 0.0
    pos_side: str | None = None
    entry_price = 0.0
    equity_curve = [capital]
    peak = capital
    trades: list[dict[str, Any]] = []
    regime = MarketRegimeDetector()

    for _, row in bars.iterrows():
        bar = {
            "open": row["open"], "high": row["high"], "low": row["low"],
            "close": row["close"], "volume": row.get("volume", 0),
            "taker_buy_volume": row.get("taker_buy_volume", row.get("volume", 0) * 0.5),
        }
        regime.update(bar["high"], bar["low"], bar["close"])

        try:
            sigs = await strategy.on_bar(symbol, bar)
        except Exception:
            sigs = []

        price = float(bar["close"])
        tv = capital * 0.1

        for sig in sigs:
            if sig.signal_type == SignalType.LONG_ENTRY and pos_qty == 0:
                sp = price * 1.0005
                pos_qty = tv / sp
                pos_side = "long"
                entry_price = sp
                capital -= tv * 0.001
                strategy.update_position(Position(symbol=symbol, side=OrderSide.BUY, qty=pos_qty, avg_price=sp))
            elif sig.signal_type == SignalType.SHORT_ENTRY and pos_qty == 0:
                sp = price * 0.9995
                pos_qty = tv / sp
                pos_side = "short"
                entry_price = sp
                capital -= tv * 0.001
                strategy.update_position(Position(symbol=symbol, side=OrderSide.SELL, qty=pos_qty, avg_price=sp))
            elif sig.signal_type == SignalType.LONG_EXIT and pos_side == "long":
                sp = price * 0.9995
                pnl = (sp - entry_price) * pos_qty
                commission = abs(pnl) * 0.001
                capital += pnl - commission
                trades.append({"pnl": pnl - commission, "side": "long"})
                pos_qty = 0
                pos_side = None
                strategy.remove_position(symbol)
            elif sig.signal_type == SignalType.SHORT_EXIT and pos_side == "short":
                sp = price * 1.0005
                pnl = (entry_price - sp) * pos_qty
                commission = abs(pnl) * 0.001
                capital += pnl - commission
                trades.append({"pnl": pnl - commission, "side": "short"})
                pos_qty = 0
                pos_side = None
                strategy.remove_position(symbol)

        unr = 0
        if pos_side == "long":
            unr = (price - entry_price) * pos_qty
        elif pos_side == "short":
            unr = (entry_price - price) * pos_qty
        eq = capital + unr
        equity_curve.append(eq)
        peak = max(peak, eq)

    if pos_side:
        fp = float(bars.iloc[-1]["close"])
        pnl = ((fp - entry_price) if pos_side == "long" else (entry_price - fp)) * pos_qty
        capital += pnl
        trades.append({"pnl": pnl, "side": pos_side})

    total_return = (capital - initial_capital) / initial_capital
    eq_arr = np.array(equity_curve)
    rets = np.diff(eq_arr) / eq_arr[:-1]
    rets = rets[np.isfinite(rets)]
    sharpe = float(np.mean(rets) / np.std(rets) * np.sqrt(252)) if len(rets) > 1 and np.std(rets) > 0 else 0
    maxdd = max((peak - e) / peak for e in equity_curve) if equity_curve else 0
    winning = [t for t in trades if t["pnl"] > 0]
    wr = len(winning) / len(trades) if trades else 0

    return {
        "total_return_pct": round(total_return * 100, 2),
        "max_drawdown_pct": round(maxdd * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "win_rate_pct": round(wr * 100, 1),
        "round_trips": len(trades),
        "final_capital": round(capital, 2),
        "trades": trades,
        "equity_curve": equity_curve,
    }


def monte_carlo_from_trades(
    trades: list[dict[str, Any]],
    initial_capital: float = 100_000.0,
    n_sims: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Monte Carlo by shuffling trade PnLs."""
    pnls = np.array([t["pnl"] for t in trades], dtype=np.float64)
    if pnls.size == 0:
        return {"error": "no trades"}

    rng = np.random.default_rng(seed)
    order = np.argsort(rng.random((n_sims, pnls.size)), axis=1)
    shuffled = pnls[order]
    cumulative = np.cumsum(shuffled, axis=1)
    equity = initial_capital + np.hstack([np.zeros((n_sims, 1)), cumulative])
    peaks = np.maximum.accumulate(equity, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd_pct = np.where(peaks > 0, (peaks - equity) / peaks, 0.0)
    max_dd_pct = dd_pct.max(axis=1)
    final_eq = equity[:, -1]
    final_ret = (final_eq - initial_capital) / initial_capital

    return {
        "n_simulations": n_sims,
        "n_trades": int(pnls.size),
        "final_return_mean_pct": round(float(np.mean(final_ret)) * 100, 2),
        "final_return_std_pct": round(float(np.std(final_ret)) * 100, 2),
        "var_95_pct": round(float(np.percentile(final_ret, 5)) * 100, 2),
        "var_99_pct": round(float(np.percentile(final_ret, 1)) * 100, 2),
        "expected_max_dd_pct": round(float(np.mean(max_dd_pct)) * 100, 2),
        "worst_max_dd_pct": round(float(np.max(max_dd_pct)) * 100, 2),
        "ci_95_pct": [
            round(float(np.percentile(final_ret, 2.5)) * 100, 2),
            round(float(np.percentile(final_ret, 97.5)) * 100, 2),
        ],
        "probability_profit_pct": round(float(np.mean(final_ret > 0)) * 100, 1),
        "probability_loss_gt10_pct": round(float(np.mean(final_ret < -0.1)) * 100, 1),
    }


async def walk_forward(
    strategy_cls: type,
    bars: pd.DataFrame,
    symbol: str,
    n_windows: int = 5,
    train_ratio: float = 0.7,
) -> dict[str, Any]:
    """Rolling walk-forward: split data into windows, train on first part, test on second."""
    total = len(bars)
    window_size = total // n_windows
    results = []

    for i in range(n_windows):
        start = i * window_size
        end = min(start + window_size, total)
        window_bars = bars.iloc[start:end].copy()
        split = int(len(window_bars) * train_ratio)

        train_bars = window_bars.iloc[:split]
        test_bars = window_bars.iloc[split:]

        if len(test_bars) < 30:
            continue

        strategy = strategy_cls(StrategyConfig(name=f"WF_{i}", symbols=[symbol], params={}))
        test_result = await simple_backtest(strategy, test_bars, symbol)

        date_col = "open_time" if "open_time" in window_bars.columns else window_bars.index.name or "index"
        window_info = {
            "window": i,
            "train_bars": len(train_bars),
            "test_bars": len(test_bars),
            "test_return_pct": test_result["total_return_pct"],
            "test_sharpe": test_result["sharpe_ratio"],
            "test_maxdd_pct": test_result["max_drawdown_pct"],
            "test_win_rate_pct": test_result["win_rate_pct"],
            "test_trades": test_result["round_trips"],
        }
        if "open_time" in window_bars.columns:
            window_info["period"] = f"{train_bars['open_time'].iloc[0].strftime('%Y-%m')} → {test_bars['open_time'].iloc[-1].strftime('%Y-%m')}"
        results.append(window_info)

    test_returns = [r["test_return_pct"] for r in results]
    test_sharpes = [r["test_sharpe"] for r in results]
    profitable_windows = sum(1 for r in test_returns if r > 0)

    return {
        "n_windows": len(results),
        "windows": results,
        "summary": {
            "avg_test_return_pct": round(np.mean(test_returns), 2) if test_returns else 0,
            "std_test_return_pct": round(np.std(test_returns), 2) if test_returns else 0,
            "avg_test_sharpe": round(np.mean(test_sharpes), 3) if test_sharpes else 0,
            "profitable_windows": profitable_windows,
            "win_consistency_pct": round(profitable_windows / len(results) * 100, 1) if results else 0,
        },
    }


async def main() -> None:

    _task_id = None
    if _sdk_submit:
        try:
            _tr = _sdk_submit(task_type="backtest", command="run_advanced_analysis.py", requester="run-advanced-analysis", estimated_duration_sec=1200)
            _task_id = _tr.get("task_id")
        except Exception:
            pass
    loader = CryptoDataLoader()
    bars = loader.load("BTCUSDT", "4h")
    logger.info("Loaded BTC 4h: %d bars [%s → %s]",
                len(bars), bars["open_time"].iloc[0], bars["open_time"].iloc[-1])

    strategies = {
        "momentum": BTCMomentumStrategy,
        "trend_following": BTCTrendFollowingStrategy,
    }

    all_results: dict[str, Any] = {}

    for name, cls in strategies.items():
        logger.info("\n" + "=" * 60)
        logger.info("ANALYZING: %s", name.upper())
        logger.info("=" * 60)

        strat = cls(StrategyConfig(name=name, symbols=["BTCUSDT"], params={}))
        bt = await simple_backtest(strat, bars, "BTCUSDT")
        logger.info("Baseline: Return=%.1f%% Sharpe=%.3f MaxDD=%.1f%% Trades=%d",
                     bt["total_return_pct"], bt["sharpe_ratio"], bt["max_drawdown_pct"], bt["round_trips"])

        logger.info("\n--- Walk-Forward Validation (5 windows) ---")
        wf = await walk_forward(cls, bars, "BTCUSDT", n_windows=5)
        for w in wf["windows"]:
            logger.info("  Window %d [%s]: Return=%.1f%% Sharpe=%.3f DD=%.1f%% Trades=%d",
                         w["window"], w.get("period", ""), w["test_return_pct"],
                         w["test_sharpe"], w["test_maxdd_pct"], w["test_trades"])
        s = wf["summary"]
        logger.info("  Summary: Avg Return=%.1f%% ± %.1f%% | Avg Sharpe=%.3f | Win Consistency=%s%%",
                     s["avg_test_return_pct"], s["std_test_return_pct"],
                     s["avg_test_sharpe"], s["win_consistency_pct"])

        logger.info("\n--- Monte Carlo Simulation (1000 paths) ---")
        mc = monte_carlo_from_trades(bt["trades"])
        if "error" not in mc:
            logger.info("  Mean Return: %.1f%% ± %.1f%%", mc["final_return_mean_pct"], mc["final_return_std_pct"])
            logger.info("  VaR 95%%: %.1f%% | VaR 99%%: %.1f%%", mc["var_95_pct"], mc["var_99_pct"])
            logger.info("  Expected Max DD: %.1f%% | Worst Max DD: %.1f%%",
                         mc["expected_max_dd_pct"], mc["worst_max_dd_pct"])
            logger.info("  95%% CI: [%.1f%%, %.1f%%]", mc["ci_95_pct"][0], mc["ci_95_pct"][1])
            logger.info("  P(profit): %.1f%% | P(loss>10%%): %.1f%%",
                         mc["probability_profit_pct"], mc["probability_loss_gt10_pct"])
        else:
            logger.info("  No trades for Monte Carlo")

        all_results[name] = {
            "baseline": {k: v for k, v in bt.items() if k not in ("trades", "equity_curve")},
            "walk_forward": wf,
            "monte_carlo": mc if "error" not in mc else None,
        }

    output = Path("models") / "advanced_analysis.json"
    output.parent.mkdir(exist_ok=True)
    with open(output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("\nAll results saved to %s", output)


    if _task_id and _sdk_complete:
        try:
            _sdk_complete(_task_id)
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
