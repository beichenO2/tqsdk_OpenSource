#!/usr/bin/env python3
"""Optimize BTC momentum / trend / multifactor parameters with OptunaHyperSearch."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path


try:
    from polarisor_port_sdk import submit_task as _sdk_submit, complete_task as _sdk_complete
except ImportError:
    _sdk_submit = _sdk_complete = None
from typing import Any

import numpy as np
import pandas as pd  # noqa: TC002

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages"))

from datahub.crypto_loader import CryptoDataLoader  # noqa: E402
from experiment.optuna_search import OptunaHyperSearch  # noqa: E402
from strategy.base import OrderSide, Position, SignalType, StrategyConfig  # noqa: E402
from strategy.btc.momentum import BTCMomentumStrategy  # noqa: E402
from strategy.btc.multifactor_strategy import BTCMultiFactorStrategy  # noqa: E402
from strategy.btc.trend_following import BTCTrendFollowingStrategy  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SYMBOL = "BTCUSDT"
TIMEFRAME = "4h"
START_DATE = "2021-01-01"
END_DATE = "2025-12-31"
INITIAL_CAPITAL = 100_000.0
COMMISSION = 0.001
SLIPPAGE = 0.0005
POSITION_SIZE = 0.1
MIN_TRADES_PENALTY = -10.0


async def quick_backtest(strategy: Any, bars: pd.DataFrame, symbol: str) -> dict[str, float]:
    """Lightweight single-symbol backtest (Sharpe on equity step returns)."""
    capital = INITIAL_CAPITAL
    pos_qty = 0.0
    pos_side: str | None = None
    entry_price = 0.0
    equity = [capital]
    peak = capital
    max_dd = 0.0
    trades = 0
    wins = 0

    for _, row in bars.iterrows():
        bar = {
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
            "volume": row.get("volume", 0),
            "taker_buy_volume": row.get("taker_buy_volume", row.get("volume", 0) * 0.5),
        }

        try:
            signals = await strategy.on_bar(symbol, bar)
        except Exception:
            signals = []

        price = float(bar["close"])
        trade_val = capital * POSITION_SIZE

        for sig in signals:
            if sig.signal_type == SignalType.LONG_ENTRY and pos_qty == 0:
                sp = price * (1 + SLIPPAGE)
                pos_qty = trade_val / sp
                pos_side = "long"
                entry_price = sp
                capital -= trade_val * COMMISSION
                strategy.update_position(
                    Position(symbol=symbol, side=OrderSide.BUY, qty=pos_qty, avg_price=sp)
                )

            elif sig.signal_type == SignalType.SHORT_ENTRY and pos_qty == 0:
                sp = price * (1 - SLIPPAGE)
                pos_qty = trade_val / sp
                pos_side = "short"
                entry_price = sp
                capital -= trade_val * COMMISSION
                strategy.update_position(
                    Position(symbol=symbol, side=OrderSide.SELL, qty=pos_qty, avg_price=sp)
                )

            elif sig.signal_type == SignalType.LONG_EXIT and pos_side == "long":
                sp = price * (1 - SLIPPAGE)
                pnl = (sp - entry_price) * pos_qty
                capital += pnl - abs(pnl) * COMMISSION
                trades += 1
                if pnl > 0:
                    wins += 1
                pos_qty = 0.0
                pos_side = None
                strategy.remove_position(symbol)

            elif sig.signal_type == SignalType.SHORT_EXIT and pos_side == "short":
                sp = price * (1 + SLIPPAGE)
                pnl = (entry_price - sp) * pos_qty
                capital += pnl - abs(pnl) * COMMISSION
                trades += 1
                if pnl > 0:
                    wins += 1
                pos_qty = 0.0
                pos_side = None
                strategy.remove_position(symbol)

        unreal = 0.0
        if pos_side == "long":
            unreal = (price - entry_price) * pos_qty
        elif pos_side == "short":
            unreal = (entry_price - price) * pos_qty
        eq = capital + unreal
        equity.append(eq)
        peak = max(peak, eq)
        dd = (peak - eq) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    if pos_side:
        fp = float(bars.iloc[-1]["close"])
        pnl = (fp - entry_price) * pos_qty if pos_side == "long" else (entry_price - fp) * pos_qty
        capital += pnl

    total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
    eq_arr = np.array(equity)
    rets = np.diff(eq_arr) / eq_arr[:-1]
    rets = rets[np.isfinite(rets)]
    sharpe = (
        float(np.mean(rets) / np.std(rets) * np.sqrt(252))
        if len(rets) > 1 and np.std(rets) > 0
        else 0.0
    )
    win_rate = wins / trades if trades > 0 else 0.0

    return {
        "sharpe": sharpe,
        "return": total_return,
        "max_dd": max_dd,
        "trades": trades,
        "win_rate": win_rate,
    }


def _run_bt_sync(strategy: Any, bars: pd.DataFrame, symbol: str) -> dict[str, float]:
    return asyncio.run(quick_backtest(strategy, bars, symbol))


def load_bars(loader: CryptoDataLoader) -> pd.DataFrame:
    df = loader.load(SYMBOL, TIMEFRAME, START_DATE, END_DATE)
    if df.empty:
        logger.warning("No data for %s %s — optimization will return zeros.", SYMBOL, TIMEFRAME)
    return df


def objective_momentum_factory(bars: pd.DataFrame):
    def objective(params: dict[str, Any]) -> float:
        cfg = StrategyConfig(name="OptMom", symbols=[SYMBOL], params=params)
        strat = BTCMomentumStrategy(cfg)
        m = _run_bt_sync(strat, bars, SYMBOL)
        if m["trades"] < 5:
            return MIN_TRADES_PENALTY
        return m["sharpe"]

    return objective


def objective_trend_factory(bars: pd.DataFrame):
    def objective(params: dict[str, Any]) -> float:
        cfg = StrategyConfig(name="OptTrend", symbols=[SYMBOL], params=params)
        strat = BTCTrendFollowingStrategy(cfg)
        m = _run_bt_sync(strat, bars, SYMBOL)
        if m["trades"] < 5:
            return MIN_TRADES_PENALTY
        return m["sharpe"]

    return objective


def objective_multifactor_factory(bars: pd.DataFrame):
    def objective(params: dict[str, Any]) -> float:
        cfg = StrategyConfig(name="OptMulti", symbols=[SYMBOL], params=params)
        strat = BTCMultiFactorStrategy(cfg)
        m = _run_bt_sync(strat, bars, SYMBOL)
        if m["trades"] < 5:
            return MIN_TRADES_PENALTY
        return m["sharpe"]

    return objective


MOMENTUM_PARAM_SPACE: dict[str, tuple[Any, ...]] = {
    "fast_period": (4, 12, "int"),
    "slow_period": (15, 40, "int"),
    "volume_ma_period": (10, 30, "int"),
    "momentum_threshold": (0.005, 0.04),
    "volume_surge_ratio": (1.0, 2.5),
    "atr_period": (10, 20, "int"),
    "trailing_stop_atr_mult": (1.5, 4.0),
}

TREND_PARAM_SPACE: dict[str, tuple[Any, ...]] = {
    "ema_fast": (6, 18, "int"),
    "ema_slow": (18, 40, "int"),
    "ema_trend": (30, 80, "int"),
    "adx_period": (10, 20, "int"),
    "adx_threshold": (15.0, 35.0),
    "atr_period": (10, 20, "int"),
    "trailing_stop_atr_mult": (1.5, 4.0),
    "partial_take_profit_atr_mult": (2.0, 6.0),
    "partial_close_pct": (0.3, 0.7),
    "risk_per_trade_pct": (0.01, 0.05),
}

MULTIFACTOR_PARAM_SPACE: dict[str, tuple[Any, ...]] = {
    "vwap_period": (10, 40, "int"),
    "obv_ma_period": (10, 40, "int"),
    "fund_flow_period": (5, 20, "int"),
    "rsi_period": (7, 21, "int"),
    "macd_fast": (8, 16, "int"),
    "macd_slow": (20, 35, "int"),
    "macd_signal": (5, 12, "int"),
    "composite_entry_threshold": (0.15, 0.55),
    "composite_exit_threshold": (-0.3, 0.0),
    "atr_period": (10, 20, "int"),
    "stop_loss_atr_mult": (1.5, 4.0),
    "take_profit_atr_mult": (2.0, 6.0),
    "weight_vwap": (0.05, 0.35),
    "weight_obv": (0.05, 0.25),
    "weight_fund_flow": (0.05, 0.35),
    "weight_rsi": (0.05, 0.25),
    "weight_macd": (0.05, 0.25),
    "weight_vol_regime": (0.05, 0.25),
}


def baseline_sharpe(bars: pd.DataFrame, cls: type, name: str) -> float:
    cfg = StrategyConfig(name=f"Baseline{name}", symbols=[SYMBOL], params={})
    strat = cls(cfg)
    return _run_bt_sync(strat, bars, SYMBOL)["sharpe"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna hyper search for BTC crypto strategies.")
    parser.add_argument("--trials", type=int, default=25, help="Trials per strategy study")
    parser.add_argument("--data-dir", type=str, default=None, help="Override CryptoDataLoader dir")
    parser.add_argument(
        "--strategies",
        nargs="*",
        default=["momentum", "trend", "multifactor"],
        help="Which strategies to optimize (momentum, trend, multifactor)",
    )
    args = parser.parse_args()

    loader = CryptoDataLoader(args.data_dir) if args.data_dir else CryptoDataLoader()
    bars = load_bars(loader)
    if bars.empty:
        print("No parquet data available; exiting.")
        return

    print(f"Loaded {len(bars)} rows for {SYMBOL} {TIMEFRAME}\n")

    _task_id = None
    if _sdk_submit:
        try:
            _tr = _sdk_submit(task_type="optimization", command=f"optimize_crypto_params.py --trials {args.trials}", requester="optimize-crypto-params", estimated_duration_sec=1800)
            _task_id = _tr.get("task_id")
        except Exception:
            pass

    STRATEGIES = {
        "momentum": (BTCMomentumStrategy, objective_momentum_factory, MOMENTUM_PARAM_SPACE, "crypto_opt_momentum"),
        "trend": (BTCTrendFollowingStrategy, objective_trend_factory, TREND_PARAM_SPACE, "crypto_opt_trend"),
        "multifactor": (BTCMultiFactorStrategy, objective_multifactor_factory, MULTIFACTOR_PARAM_SPACE, "crypto_opt_multifactor"),
    }

    results_summary: list[dict[str, Any]] = []

    for strat_key in args.strategies:
        if strat_key not in STRATEGIES:
            print(f"Unknown strategy: {strat_key}")
            continue

        cls, factory, space, study_name = STRATEGIES[strat_key]
        base = baseline_sharpe(bars, cls, strat_key)
        print(f"Baseline Sharpe ({strat_key}): {base:.4f}")

        search = OptunaHyperSearch(
            objective_fn=factory(bars),
            param_space=space,
            direction="maximize",
            study_name=study_name,
        )
        res = search.run(n_trials=args.trials)
        improvement = res.best_value - base
        print(f"\n=== {cls.__name__} ===")
        print(f"Best Sharpe: {res.best_value:.4f} (Δ vs baseline: {improvement:+.4f})")
        print(f"Best params: {res.best_params}\n")

        results_summary.append({
            "strategy": strat_key,
            "baseline_sharpe": round(base, 4),
            "optimized_sharpe": round(res.best_value, 4),
            "improvement": round(improvement, 4),
            "best_params": res.best_params,
        })

    import json
    out = Path("models") / "optuna_results.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(results_summary, f, indent=2, default=str)
    print(f"\nResults saved to {out}")

    if _task_id and _sdk_complete:
        try:
            _sdk_complete(_task_id)
        except Exception:
            pass


if __name__ == "__main__":
    main()
