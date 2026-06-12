#!/usr/bin/env python3
"""全量期货策略回测脚本 — 所有策略 × 所有品种 × 真实 tick 数据。

用法:
  cd ~/Polarisor/tqsdk/trading-platform
  .venv/bin/python3 scripts/run_futures_backtest.py
  .venv/bin/python3 scripts/run_futures_backtest.py --symbols rb cu --strategies cta_trend rbreaker
  .venv/bin/python3 scripts/run_futures_backtest.py --timeframe 5m --output results/futures_backtest.json

输出: JSON + CSV 报告，按策略×品种的 Sharpe/收益/回撤/胜率矩阵
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

import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJ_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_DIR / "packages"))

from datahub.futures_loader import FuturesDataLoader
from strategy.base import BaseStrategy, Signal, SignalType, StrategyConfig
from strategy.registry import StrategyRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("futures-backtest")

CACHE_DIR = str(PROJ_DIR / "data" / "futures_cache")

DEFAULT_SYMBOLS = ["rb", "cu", "ag", "m", "i", "IF"]
DEFAULT_STRATEGIES = [
    "cta_trend", "rbreaker", "bollinger_mr", "vol_breakout", "volume_price",
    "adaptive_bollinger", "regime_momentum", "spread_arb", "dual_ma",
    "chan_theory", "orderflow_imbalance", "intraday_reversal",
    "kalman_trend", "har_volatility",
]


def backtest_strategy_on_bars(
    strategy: BaseStrategy,
    bars: pd.DataFrame,
    initial_capital: float = 100000.0,
    commission_rate: float = 0.00005,
    slippage_pct: float = 0.0001,
) -> dict:
    """Run a strategy through historical bars and compute metrics."""
    capital = initial_capital
    position = None  # "long" or "short"
    entry_price = 0.0
    trades = []
    equity_curve = [capital]

    for idx, row in bars.iterrows():
        bar = {
            "datetime": row.get("datetime"),
            "open": float(row.get("open", 0)),
            "high": float(row.get("high", 0)),
            "low": float(row.get("low", 0)),
            "close": float(row.get("close", 0)),
            "volume": float(row.get("volume", 0)),
            "open_interest": float(row.get("open_interest", 0)) if "open_interest" in row.index else 0.0,
        }

        symbol = str(row.get("instrument", "unknown"))
        signals = asyncio.get_event_loop().run_until_complete(
            strategy.on_bar(symbol, bar)
        )

        for sig in signals:
            close_price = bar["close"]

            if sig.signal_type == SignalType.LONG_ENTRY and position is None:
                position = "long"
                entry_price = close_price
                capital *= (1 - commission_rate - slippage_pct)

            elif sig.signal_type == SignalType.SHORT_ENTRY and position is None:
                position = "short"
                entry_price = close_price
                capital *= (1 - commission_rate - slippage_pct)

            elif sig.signal_type == SignalType.LONG_EXIT and position == "long":
                pnl_pct = (close_price - entry_price) / entry_price
                capital *= (1 + pnl_pct) * (1 - commission_rate - slippage_pct)
                trades.append({"pnl_pct": pnl_pct, "side": "long"})
                position = None

            elif sig.signal_type == SignalType.SHORT_EXIT and position == "short":
                pnl_pct = (entry_price - close_price) / entry_price
                capital *= (1 + pnl_pct) * (1 - commission_rate - slippage_pct)
                trades.append({"pnl_pct": pnl_pct, "side": "short"})
                position = None

        equity_curve.append(capital)

    if position and len(bars) > 0:
        last_close = float(bars.iloc[-1]["close"])
        if position == "long":
            pnl_pct = (last_close - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - last_close) / entry_price
        capital *= (1 + pnl_pct) * (1 - commission_rate - slippage_pct)
        trades.append({"pnl_pct": pnl_pct, "side": position})

    total_return = (capital - initial_capital) / initial_capital
    n_trades = len(trades)
    wins = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]

    win_rate = len(wins) / n_trades if n_trades > 0 else 0.0
    avg_win = np.mean([t["pnl_pct"] for t in wins]) if wins else 0.0
    avg_loss = np.mean([abs(t["pnl_pct"]) for t in losses]) if losses else 0.0
    profit_factor = sum(t["pnl_pct"] for t in wins) / max(sum(abs(t["pnl_pct"]) for t in losses), 1e-10) if losses else (10.0 if wins else 0.0)

    equity = np.array(equity_curve)
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    max_dd = float(np.max(drawdown))

    weekly_returns = []
    step = max(1, len(equity) // 52)
    for i in range(step, len(equity), step):
        r = (equity[i] - equity[i - step]) / equity[i - step] if equity[i - step] > 0 else 0.0
        weekly_returns.append(r)

    if len(weekly_returns) >= 2:
        sharpe = float(np.mean(weekly_returns) / max(np.std(weekly_returns), 1e-10) * np.sqrt(52))
    else:
        sharpe = 0.0

    return {
        "total_return": round(total_return, 6),
        "sharpe": round(sharpe, 4),
        "max_dd": round(max_dd, 6),
        "trades": n_trades,
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4),
        "avg_win": round(avg_win, 6),
        "avg_loss": round(avg_loss, 6),
        "final_capital": round(capital, 2),
    }


def main():

    _task_id = None
    if _sdk_submit:
        try:
            _tr = _sdk_submit(task_type="backtest", command="run_futures_backtest.py", requester="run-futures-backtest", estimated_duration_sec=1800)
            _task_id = _tr.get("task_id")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="Full futures strategy backtester")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--strategies", nargs="+", default=DEFAULT_STRATEGIES)
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--output", default="results/futures_backtest_report.json")
    args = parser.parse_args()

    import strategy.futures  # trigger @auto_register

    loader = FuturesDataLoader()
    registry = StrategyRegistry()

    available_strategies = registry.list_registered()
    strategies_to_run = [s for s in args.strategies if s in available_strategies]

    if not strategies_to_run:
        logger.error("No matching strategies found. Available: %s", available_strategies)
        return

    logger.info("=" * 80)
    logger.info("FUTURES BACKTEST — %d strategies × %d symbols", len(strategies_to_run), len(args.symbols))
    logger.info("Strategies: %s", strategies_to_run)
    logger.info("Symbols: %s", args.symbols)
    logger.info("Timeframe: %s", args.timeframe)
    logger.info("=" * 80)

    all_results = []

    for sym in args.symbols:
        logger.info("\nLoading data for %s...", sym)
        bars = loader.load_bars(sym, args.timeframe, cache_dir=CACHE_DIR)

        if bars.empty:
            bars = loader.load_main_contract_bars(sym, args.timeframe, cache_dir=CACHE_DIR)

        if bars.empty:
            logger.warning("No data for %s, skipping", sym)
            continue

        logger.info("Loaded %d bars for %s", len(bars), sym)

        for strat_name in strategies_to_run:
            logger.info("  Running %s on %s (%d bars)...", strat_name, sym, len(bars))
            t0 = time.time()

            try:
                config = StrategyConfig(name=strat_name, strategy_id=strat_name)
                strategy_cls = registry.get(strat_name)
                if strategy_cls is None:
                    logger.warning("  Strategy %s not found in registry", strat_name)
                    continue

                strategy_instance = strategy_cls(config)
                result = backtest_strategy_on_bars(strategy_instance, bars)
                result["strategy"] = strat_name
                result["symbol"] = sym
                result["bars"] = len(bars)
                result["duration_s"] = round(time.time() - t0, 1)
                all_results.append(result)

                logger.info(
                    "    %s/%s: Return=%.2f%% Sharpe=%.3f MaxDD=%.2f%% Trades=%d WR=%.1f%% PF=%.2f (%.1fs)",
                    strat_name, sym,
                    result["total_return"] * 100,
                    result["sharpe"],
                    result["max_dd"] * 100,
                    result["trades"],
                    result["win_rate"] * 100,
                    result["profit_factor"],
                    result["duration_s"],
                )

            except Exception as e:
                logger.error("    %s/%s FAILED: %s", strat_name, sym, e)
                all_results.append({
                    "strategy": strat_name,
                    "symbol": sym,
                    "error": str(e),
                })

    output_path = PROJ_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "symbols": args.symbols,
            "strategies": strategies_to_run,
            "timeframe": args.timeframe,
        },
        "results": all_results,
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("\nReport saved to %s", output_path)

    csv_path = output_path.with_suffix(".csv")
    valid_results = [r for r in all_results if "error" not in r]
    if valid_results:
        df = pd.DataFrame(valid_results)
        df.to_csv(csv_path, index=False)
        logger.info("CSV saved to %s", csv_path)

        logger.info("\n" + "=" * 80)
        logger.info("SUMMARY — Best strategies by Sharpe:")
        logger.info("=" * 80)
        summary = df.groupby("strategy").agg({
            "sharpe": "mean",
            "total_return": "mean",
            "max_dd": "mean",
            "trades": "sum",
            "win_rate": "mean",
        }).sort_values("sharpe", ascending=False)
        for strat, row in summary.iterrows():
            logger.info(
                "  %-25s Sharpe=%.3f Return=%.2f%% MaxDD=%.2f%% Trades=%d WR=%.1f%%",
                strat, row["sharpe"], row["total_return"] * 100,
                row["max_dd"] * 100, int(row["trades"]), row["win_rate"] * 100,
            )


    if _task_id and _sdk_complete:
        try:
            _sdk_complete(_task_id)
        except Exception:
            pass


if __name__ == "__main__":
    main()
