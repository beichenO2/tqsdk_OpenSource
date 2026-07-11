#!/usr/bin/env python3
"""2号位工具：快速扫描所有策略的信号质量和方向性。

用法:
  cd ~/Polarisor/tqsdk-gnhf-worktrees/pos2/trading-platform
  .venv/bin/python3 scripts/validate_quick_sweep.py --symbol rb --max-bars 5000
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJ_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_DIR / "packages"))

from datahub.futures_loader import FuturesDataLoader
from strategy.base import BaseStrategy, Signal, SignalType, StrategyConfig
from strategy.registry import StrategyRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("quick-sweep")

CACHE_DIR = str(PROJ_DIR / "data" / "futures_cache")


def sweep_strategy(
    name: str,
    bars: pd.DataFrame,
    initial_capital: float = 100_000.0,
) -> dict | None:
    cls = StrategyRegistry.get(name)
    if cls is None:
        return None

    config = StrategyConfig(name=name, strategy_id=f"{name}_sweep")
    strat = cls(config)
    loop = asyncio.new_event_loop()

    capital = initial_capital
    position = None
    entry_price = 0.0
    signal_counts = {"LONG_ENTRY": 0, "SHORT_ENTRY": 0, "LONG_EXIT": 0, "SHORT_EXIT": 0}
    trades = 0
    wins = 0

    for _, row in bars.iterrows():
        bar = {
            "datetime": row.get("datetime"),
            "open": float(row.get("open", 0)),
            "high": float(row.get("high", 0)),
            "low": float(row.get("low", 0)),
            "close": float(row.get("close", 0)),
            "volume": float(row.get("volume", 0)),
            "open_interest": float(row.get("open_interest", 0)) if "open_interest" in row.index else 0.0,
        }
        sym = str(row.get("instrument", "unknown"))

        try:
            signals = loop.run_until_complete(strat.on_bar(sym, bar))
        except Exception:
            signals = []

        close = bar["close"]
        if close <= 0:
            continue

        for sig in signals:
            st = sig.signal_type.name
            if st in signal_counts:
                signal_counts[st] += 1

            if sig.signal_type == SignalType.LONG_ENTRY and position is None:
                position = "long"
                entry_price = close
            elif sig.signal_type == SignalType.SHORT_ENTRY and position is None:
                position = "short"
                entry_price = close
            elif sig.signal_type == SignalType.LONG_EXIT and position == "long":
                pnl = (close - entry_price) / entry_price
                capital += capital * max(pnl, -0.99)
                capital = max(capital, 0.01)
                trades += 1
                if pnl > 0:
                    wins += 1
                position = None
            elif sig.signal_type == SignalType.SHORT_EXIT and position == "short":
                pnl = (entry_price - close) / entry_price
                capital += capital * max(pnl, -0.99)
                capital = max(capital, 0.01)
                trades += 1
                if pnl > 0:
                    wins += 1
                position = None

    loop.close()

    total_entry = signal_counts["LONG_ENTRY"] + signal_counts["SHORT_ENTRY"]
    long_ratio = signal_counts["LONG_ENTRY"] / total_entry if total_entry > 0 else 0.5
    balance = min(long_ratio, 1 - long_ratio)

    return {
        "name": name,
        "return": round((capital - initial_capital) / initial_capital, 4),
        "trades": trades,
        "win_rate": round(wins / trades, 2) if trades > 0 else 0,
        "long_entries": signal_counts["LONG_ENTRY"],
        "short_entries": signal_counts["SHORT_ENTRY"],
        "balance": round(balance, 2),
        "capital": round(capital, 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick strategy signal sweep")
    parser.add_argument("--symbol", default="rb")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--max-bars", type=int, default=5000)
    args = parser.parse_args()

    import strategy.futures

    registered = StrategyRegistry.list_registered()
    logger.info("Registered strategies: %d", len(registered))

    loader = FuturesDataLoader()
    bars = loader.load_bars(args.symbol, args.timeframe, cache_dir=CACHE_DIR)
    if "instrument" in bars.columns and bars["instrument"].nunique() > 1:
        vol_by_inst = bars.groupby("instrument")["volume"].sum()
        main_inst = vol_by_inst.idxmax()
        bars = bars[bars["instrument"] == main_inst].reset_index(drop=True)
        logger.info("Filtered to main contract %s", main_inst)
    if args.max_bars and len(bars) > args.max_bars:
        bars = bars.tail(args.max_bars).reset_index(drop=True)
    logger.info("Loaded %d bars for %s", len(bars), args.symbol)

    results = []
    for name in sorted(registered):
        t0 = time.time()
        r = sweep_strategy(name, bars)
        elapsed = time.time() - t0
        if r is None:
            continue
        r["elapsed_s"] = round(elapsed, 1)
        results.append(r)
        logger.info(
            "  %-30s ret=%+.4f trades=%3d wr=%.2f L/S=%d/%d bal=%.2f (%.1fs)",
            name, r["return"], r["trades"], r["win_rate"],
            r["long_entries"], r["short_entries"], r["balance"], elapsed,
        )

    results.sort(key=lambda x: -x["return"])
    logger.info("\n" + "=" * 90)
    logger.info("TOP 10 BY RETURN (on %s, %d bars):", args.symbol, len(bars))
    logger.info("%-30s %8s %6s %5s %5s %5s %5s", "strategy", "return", "trades", "wr", "L", "S", "bal")
    logger.info("-" * 90)
    for r in results[:10]:
        logger.info(
            "%-30s %+8.4f %6d %5.2f %5d %5d %5.2f",
            r["name"], r["return"], r["trades"], r["win_rate"],
            r["long_entries"], r["short_entries"], r["balance"],
        )

    positive = [r for r in results if r["return"] > 0]
    balanced = [r for r in results if r["balance"] >= 0.2]
    logger.info("\nPositive return: %d/%d", len(positive), len(results))
    logger.info("Balanced signals (>=0.2): %d/%d", len(balanced), len(results))


if __name__ == "__main__":
    main()
