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
import json
import logging
import sys
import time
from datetime import datetime
from decimal import Decimal
from pathlib import Path

try:
    from polarisor_port_sdk import submit_task as _sdk_submit, complete_task as _sdk_complete
except ImportError:
    _sdk_submit = _sdk_complete = None

import pandas as pd

PROJ_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_DIR / "packages"))

from backtest.futures_matrix import (  # noqa: E402
    DEFAULT_COMMISSION_RATE,
    DEFAULT_CONTRACT_MULTIPLIER,
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_SLIPPAGE_TICKS,
    DEFAULT_TICK_SIZE,
    run_futures_matrix,
)
from datahub.futures_loader import FuturesDataLoader  # noqa: E402
from strategy.registry import StrategyRegistry  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("futures-backtest")

CACHE_DIR = str(PROJ_DIR / "data" / "futures_cache")

DEFAULT_SYMBOLS = ["rb", "cu", "ag", "m", "i", "IF"]
DEFAULT_STRATEGIES = [
    "cta_trend", "rbreaker", "bollinger_mr", "vol_breakout", "volume_price",
    "adaptive_bollinger", "regime_momentum", "spread_arb", "futures_dual_ma",
    "chan_theory", "orderflow_imbalance", "intraday_reversal",
    "kalman_trend", "har_volatility",
]


def main() -> None:
    _task_id = None
    if _sdk_submit:
        try:
            _tr = _sdk_submit(
                task_type="backtest",
                command="run_futures_backtest.py",
                requester="run-futures-backtest",
                estimated_duration_sec=1800,
            )
            _task_id = _tr.get("task_id")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Full futures strategy backtester")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--strategies", nargs="+", default=DEFAULT_STRATEGIES)
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--output", default="results/futures_backtest_report.json")
    args = parser.parse_args()

    import strategy.futures  # noqa: F401 — trigger @auto_register

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

    symbol_bars: dict[str, pd.DataFrame] = {}
    for sym in args.symbols:
        logger.info("\nLoading data for %s...", sym)
        bars = loader.load_bars(sym, args.timeframe, cache_dir=CACHE_DIR)
        if bars.empty:
            bars = loader.load_main_contract_bars(sym, args.timeframe, cache_dir=CACHE_DIR)
        if bars.empty:
            logger.warning("No data for %s, skipping", sym)
            continue
        logger.info("Loaded %d bars for %s", len(bars), sym)
        symbol_bars[sym] = bars

    t_matrix = time.time()
    all_results = run_futures_matrix(
        symbol_bars,
        strategies_to_run,
        registry,
        initial_capital=DEFAULT_INITIAL_CAPITAL,
        commission_rate=DEFAULT_COMMISSION_RATE,
        slippage_ticks=DEFAULT_SLIPPAGE_TICKS,
        tick_size=DEFAULT_TICK_SIZE,
        contract_multiplier=DEFAULT_CONTRACT_MULTIPLIER,
    )

    for result in all_results:
        if "error" in result:
            continue
        logger.info(
            "    %s/%s: Return=%.2f%% Sharpe=%.3f MaxDD=%.2f%% Trades=%d WR=%.1f%% PF=%.2f (%.1fs)",
            result["strategy"],
            result["symbol"],
            result["total_return"] * 100,
            result["sharpe"],
            result["max_dd"] * 100,
            result["trades"],
            result["win_rate"] * 100,
            result["profit_factor"],
            result["duration_s"],
        )

    logger.info("Matrix completed in %.1fs", time.time() - t_matrix)

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
                strat,
                row["sharpe"],
                row["total_return"] * 100,
                row["max_dd"] * 100,
                int(row["trades"]),
                row["win_rate"] * 100,
            )

    if _task_id and _sdk_complete:
        try:
            _sdk_complete(_task_id)
        except Exception:
            pass


if __name__ == "__main__":
    main()
