#!/usr/bin/env python3
"""Walk-Forward Efficiency (WFE) Analysis — rolling window validation.

WFE = mean(OOS_Sharpe) / mean(IS_Sharpe)
A WFE > 0.5 indicates the strategy retains most of its in-sample edge
out-of-sample and is unlikely to be overfit.

Usage:
    python scripts/run_wfe_analysis.py --instrument rb --strategies adaptive_bb regime_mom
"""

from __future__ import annotations

import argparse
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

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages"))

from datahub.futures_loader import FuturesDataLoader
from strategy.base import Signal, SignalType, StrategyConfig, Position, OrderSide
from strategy.futures.adaptive_bollinger import AdaptiveBollingerStrategy
from strategy.futures.bollinger_mr import BollingerMRStrategy
from strategy.futures.regime_momentum import RegimeMomentumStrategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "futures_cache"

CONTRACT_SPEC: dict[str, dict[str, float]] = {
    "rb": {"mult": 10, "tick": 1.0, "commission": 5.0, "margin_rate": 0.10},
    "IF": {"mult": 300, "tick": 0.2, "commission": 30.0, "margin_rate": 0.12},
    "m":  {"mult": 10, "tick": 1.0, "commission": 3.0, "margin_rate": 0.08},
    "cu": {"mult": 5, "tick": 10.0, "commission": 10.0, "margin_rate": 0.10},
    "au": {"mult": 1000, "tick": 0.02, "commission": 10.0, "margin_rate": 0.08},
    "AP": {"mult": 10, "tick": 1.0, "commission": 5.0, "margin_rate": 0.10},
}


def make_strategy(name: str, symbol: str) -> Any:
    cfg = StrategyConfig(name=name, symbols=[symbol])
    if name == "adaptive_bb":
        return AdaptiveBollingerStrategy(cfg)
    if name == "bollinger_mr":
        return BollingerMRStrategy(cfg)
    if name == "regime_mom":
        return RegimeMomentumStrategy(cfg)
    raise ValueError(f"Unknown strategy: {name}")


async def run_window(
    strategy: Any, bars: pd.DataFrame, symbol: str,
    mult: float, tick: float, comm: float, capital: float, max_lots: int,
) -> dict[str, Any]:
    """Run a single backtest window and return metrics."""
    pos_qty = 0.0
    pos_side: str | None = None
    entry_price = 0.0
    equity = capital
    equity_curve = [capital]
    peak = capital
    max_dd = 0.0

    for _, row in bars.iterrows():
        bar = {
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row.get("volume", 0)),
        }
        if "datetime" in row:
            bar["datetime"] = row["datetime"]

        try:
            signals = await strategy.on_bar(symbol, bar)
        except Exception:
            signals = []

        for sig in signals:
            if not isinstance(sig, Signal):
                continue
            price = bar["close"]
            margin_per_lot = price * mult * 0.10
            lots = min(max_lots, max(1, int(equity * 0.05 / margin_per_lot))) if margin_per_lot > 0 else 1

            if sig.signal_type == SignalType.LONG_ENTRY and pos_qty == 0:
                slip = price + tick
                equity -= lots * comm
                pos_qty, pos_side, entry_price = float(lots), "long", slip
                strategy.update_position(Position(symbol=symbol, side=OrderSide.BUY, qty=lots, avg_price=slip))
            elif sig.signal_type == SignalType.SHORT_ENTRY and pos_qty == 0:
                slip = price - tick
                equity -= lots * comm
                pos_qty, pos_side, entry_price = float(lots), "short", slip
                strategy.update_position(Position(symbol=symbol, side=OrderSide.SELL, qty=lots, avg_price=slip))
            elif sig.signal_type == SignalType.LONG_EXIT and pos_side == "long":
                slip = price - tick
                pnl = (slip - entry_price) * pos_qty * mult
                equity += pnl - pos_qty * comm
                strategy.remove_position(symbol)
                pos_qty, pos_side = 0.0, None
            elif sig.signal_type == SignalType.SHORT_EXIT and pos_side == "short":
                slip = price + tick
                pnl = (entry_price - slip) * pos_qty * mult
                equity += pnl - pos_qty * comm
                strategy.remove_position(symbol)
                pos_qty, pos_side = 0.0, None

        unrealized = 0.0
        if pos_side == "long":
            unrealized = (bar["close"] - entry_price) * pos_qty * mult
        elif pos_side == "short":
            unrealized = (entry_price - bar["close"]) * pos_qty * mult
        total_eq = equity + unrealized
        equity_curve.append(total_eq)
        peak = max(peak, total_eq)
        dd = (peak - total_eq) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    if pos_side:
        final = float(bars.iloc[-1]["close"])
        pnl = (final - entry_price) * pos_qty * mult if pos_side == "long" else (entry_price - final) * pos_qty * mult
        equity += pnl

    total_ret = (equity - capital) / capital
    eq = np.array(equity_curve)
    step_rets = np.diff(eq) / eq[:-1]
    step_rets = step_rets[np.isfinite(step_rets)]
    sharpe = float(np.mean(step_rets) / np.std(step_rets) * np.sqrt(252 * 48)) if len(step_rets) > 1 and np.std(step_rets) > 0 else 0.0

    return {
        "return_pct": round(total_ret * 100, 3),
        "max_dd_pct": round(max_dd * 100, 3),
        "sharpe": round(sharpe, 4),
        "bars": len(bars),
    }


async def walk_forward(
    strat_name: str, instrument: str, timeframe: str,
    start: str, end: str, train_days: int, test_days: int,
    capital: float,
) -> dict[str, Any]:
    """Run rolling walk-forward analysis and compute WFE."""
    loader = FuturesDataLoader()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    bars = loader.load_main_contract_bars(
        instrument, timeframe, start, end, cache_dir=str(CACHE_DIR),
    )
    if bars.empty:
        return {"error": f"No data for {instrument}"}

    if "datetime" not in bars.columns:
        return {"error": "No datetime column in bars"}

    bars["datetime"] = pd.to_datetime(bars["datetime"])
    bars = bars.sort_values("datetime").reset_index(drop=True)

    inst = instrument.lower()
    spec = CONTRACT_SPEC.get(inst, CONTRACT_SPEC.get(instrument, {}))
    mult = spec.get("mult", 10)
    tick = spec.get("tick", 1.0)
    comm = spec.get("commission", 5.0)
    margin_rate = spec.get("margin_rate", 0.10)
    avg_price = bars["close"].mean()
    margin_per_lot = avg_price * mult * margin_rate
    max_lots = max(1, int(capital * 0.3 / margin_per_lot))

    sym = instrument.upper()

    bars_per_day = len(bars) / max(1, (bars["datetime"].iloc[-1] - bars["datetime"].iloc[0]).days)
    train_bars = int(train_days * bars_per_day)
    test_bars = int(test_days * bars_per_day)
    total_needed = train_bars + test_bars

    if len(bars) < total_needed:
        return {"error": f"Not enough bars: {len(bars)} < {total_needed}"}

    windows = []
    cursor = 0
    while cursor + total_needed <= len(bars):
        train_df = bars.iloc[cursor:cursor + train_bars]
        test_df = bars.iloc[cursor + train_bars:cursor + total_needed]

        train_strat = make_strategy(strat_name, sym)
        train_result = await run_window(train_strat, train_df, sym, mult, tick, comm, capital, max_lots)

        oos_strat = make_strategy(strat_name, sym)
        oos_result = await run_window(oos_strat, test_df, sym, mult, tick, comm, capital, max_lots)

        windows.append({
            "window": len(windows),
            "train": {
                "start": str(train_df["datetime"].iloc[0]),
                "end": str(train_df["datetime"].iloc[-1]),
                **train_result,
            },
            "test": {
                "start": str(test_df["datetime"].iloc[0]),
                "end": str(test_df["datetime"].iloc[-1]),
                **oos_result,
            },
        })

        cursor += test_bars

    is_sharpes = [w["train"]["sharpe"] for w in windows]
    oos_sharpes = [w["test"]["sharpe"] for w in windows]
    is_returns = [w["train"]["return_pct"] for w in windows]
    oos_returns = [w["test"]["return_pct"] for w in windows]

    avg_is_sharpe = np.mean(is_sharpes) if is_sharpes else 0
    avg_oos_sharpe = np.mean(oos_sharpes) if oos_sharpes else 0
    wfe = avg_oos_sharpe / avg_is_sharpe if avg_is_sharpe != 0 else 0

    return {
        "strategy": strat_name,
        "instrument": instrument,
        "train_days": train_days,
        "test_days": test_days,
        "total_windows": len(windows),
        "avg_is_sharpe": round(float(avg_is_sharpe), 4),
        "avg_oos_sharpe": round(float(avg_oos_sharpe), 4),
        "WFE": round(float(wfe), 4),
        "is_sharpes": [round(s, 4) for s in is_sharpes],
        "oos_sharpes": [round(s, 4) for s in oos_sharpes],
        "avg_is_return": round(float(np.mean(is_returns)), 3),
        "avg_oos_return": round(float(np.mean(oos_returns)), 3),
        "windows": windows,
    }


async def main() -> None:

    _task_id = None
    if _sdk_submit:
        try:
            _tr = _sdk_submit(task_type="backtest", command="run_wfe_analysis.py", requester="run-wfe-analysis", estimated_duration_sec=1200)
            _task_id = _tr.get("task_id")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="Walk-Forward Efficiency analysis")
    parser.add_argument("--instrument", default="rb")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2024-03-31")
    parser.add_argument("--train-days", type=int, default=30)
    parser.add_argument("--test-days", type=int, default=15)
    parser.add_argument("--capital", type=float, default=500_000)
    parser.add_argument("--strategies", nargs="*", default=["adaptive_bb", "bollinger_mr", "regime_mom"])
    args = parser.parse_args()

    all_results = {}
    for strat in args.strategies:
        logger.info("WFE Analysis: %s on %s %s", strat, args.instrument.upper(), args.timeframe)
        result = await walk_forward(
            strat, args.instrument, args.timeframe,
            args.start, args.end, args.train_days, args.test_days,
            args.capital,
        )
        all_results[strat] = result

        if "error" in result:
            logger.error("  ERROR: %s", result["error"])
            continue

        logger.info(
            "  Windows: %d | IS Sharpe: %.3f | OOS Sharpe: %.3f | WFE: %.3f",
            result["total_windows"], result["avg_is_sharpe"],
            result["avg_oos_sharpe"], result["WFE"],
        )
        for w in result["windows"]:
            logger.info(
                "    W%d: IS %.3f%% (Sharpe %.2f) → OOS %.3f%% (Sharpe %.2f)",
                w["window"],
                w["train"]["return_pct"], w["train"]["sharpe"],
                w["test"]["return_pct"], w["test"]["sharpe"],
            )

    logger.info("\n" + "=" * 70)
    logger.info("WFE SUMMARY — %s %s", args.instrument.upper(), args.timeframe)
    logger.info("=" * 70)
    header = f"{'Strategy':<16} {'IS_Sharpe':>10} {'OOS_Sharpe':>11} {'WFE':>8} {'IS_Ret%':>8} {'OOS_Ret%':>9} {'Windows':>8}"
    logger.info(header)
    logger.info("-" * len(header))

    for name, r in all_results.items():
        if "error" in r:
            logger.info("%s  ERROR: %s", name.ljust(16), r["error"])
            continue
        wfe_indicator = "OK" if abs(r["WFE"]) > 0.5 else "WARN"
        logger.info(
            "%s %10.3f %11.3f %8.3f %8.3f %9.3f %8d  %s",
            name.ljust(16),
            r["avg_is_sharpe"], r["avg_oos_sharpe"], r["WFE"],
            r["avg_is_return"], r["avg_oos_return"],
            r["total_windows"], wfe_indicator,
        )

    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"wfe_{args.instrument}_{args.timeframe}.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("\nSaved to %s", output_path)


    if _task_id and _sdk_complete:
        try:
            _sdk_complete(_task_id)
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
