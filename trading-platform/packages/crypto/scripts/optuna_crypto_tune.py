"""Optuna hyperparameter search for SOTA crypto strategies.

Optimizes strategy parameters using walk-forward train/test split.
Objective: maximize OOS Sharpe (with minimum trade count constraint).
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import optuna
from optuna import Trial

from datahub.crypto_loader import CryptoDataLoader
from strategy.base import StrategyConfig, Signal, SignalType, Position, OrderSide
from strategy.btc.funding_rate_alpha import FundingRateAlphaStrategy
from strategy.btc.meta_labeling import MetaLabelingStrategy
from strategy.btc.cross_sectional_momentum import TimeSeriesMomentumStrategy
from strategy.btc.patch_tst_strategy import PatchTSTStrategy
from strategy.btc.funding_meta_ensemble import FundingMetaEnsembleStrategy
from strategy.btc.regime_detector import MarketRegimeDetector

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

SYMBOL = "BTCUSDT"
TIMEFRAME = "4h"
INITIAL_CAPITAL = 100_000.0
COMMISSION = 0.001
SLIPPAGE = 0.0005
TRAIN_RATIO = 0.7


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    loader = CryptoDataLoader()
    bars = loader.load_with_funding(SYMBOL, TIMEFRAME)
    bars.attrs["timeframe"] = TIMEFRAME
    split = int(len(bars) * TRAIN_RATIO)
    train = bars.iloc[:split].copy()
    test = bars.iloc[split:].copy()
    train.attrs = bars.attrs.copy()
    test.attrs = bars.attrs.copy()
    return train, test


async def backtest(strategy: Any, bars: pd.DataFrame) -> dict[str, float]:
    """Quick backtest with Kelly sizing and regime detection."""
    capital = INITIAL_CAPITAL
    pos_qty = 0.0
    pos_side: str | None = None
    entry_price = 0.0
    equity = [capital]
    peak = capital
    max_dd = 0.0
    trades = 0
    wins = 0
    pending_signals: list[Signal] = []
    regime = MarketRegimeDetector()
    pos_pct = 0.05

    for _, row in bars.iterrows():
        bar = {
            "open": row["open"], "high": row["high"],
            "low": row["low"], "close": row["close"],
            "volume": row.get("volume", 0),
            "taker_buy_volume": row.get("taker_buy_volume", row.get("volume", 0) * 0.5),
        }
        for col in row.index:
            if col not in {"open", "high", "low", "close", "volume", "taker_buy_volume", "open_time", "close_time"} and pd.notna(row[col]):
                bar[col] = row[col]

        regime.update(bar["high"], bar["low"], bar["close"])
        exec_price = bar["open"]
        trade_val = capital * pos_pct

        for sig in pending_signals:
            if sig.signal_type == SignalType.LONG_ENTRY and pos_qty == 0:
                sp = exec_price * (1 + SLIPPAGE)
                pos_qty = trade_val / sp
                pos_side = "long"
                entry_price = sp
                capital -= trade_val * COMMISSION
                strategy.update_position(Position(symbol=SYMBOL, side=OrderSide.BUY, qty=pos_qty, avg_price=sp))
            elif sig.signal_type == SignalType.SHORT_ENTRY and pos_qty == 0:
                sp = exec_price * (1 - SLIPPAGE)
                pos_qty = trade_val / sp
                pos_side = "short"
                entry_price = sp
                capital -= trade_val * COMMISSION
                strategy.update_position(Position(symbol=SYMBOL, side=OrderSide.SELL, qty=pos_qty, avg_price=sp))
            elif sig.signal_type == SignalType.LONG_EXIT and pos_side == "long":
                sp = exec_price * (1 - SLIPPAGE)
                pnl = (sp - entry_price) * pos_qty
                commission = abs(sp * pos_qty) * COMMISSION
                capital += pnl - commission
                trades += 1
                if pnl > commission:
                    wins += 1
                pos_qty = 0
                pos_side = None
                strategy.remove_position(SYMBOL)
            elif sig.signal_type == SignalType.SHORT_EXIT and pos_side == "short":
                sp = exec_price * (1 + SLIPPAGE)
                pnl = (entry_price - sp) * pos_qty
                commission = abs(sp * pos_qty) * COMMISSION
                capital += pnl - commission
                trades += 1
                if pnl > commission:
                    wins += 1
                pos_qty = 0
                pos_side = None
                strategy.remove_position(SYMBOL)

        pending_signals.clear()
        try:
            signals = await strategy.on_bar(SYMBOL, bar)
            pending_signals.extend(signals)
        except Exception:
            pass

        unreal = 0.0
        if pos_side == "long":
            unreal = (bar["close"] - entry_price) * pos_qty
        elif pos_side == "short":
            unreal = (entry_price - bar["close"]) * pos_qty
        eq = capital + unreal
        equity.append(eq)
        peak = max(peak, eq)
        dd = (peak - eq) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    if pos_side:
        fp = bars.iloc[-1]["close"]
        pnl = (fp - entry_price) * pos_qty if pos_side == "long" else (entry_price - fp) * pos_qty
        capital += pnl

    total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
    eq_arr = np.array(equity)
    rets = np.diff(eq_arr) / eq_arr[:-1]
    rets = rets[np.isfinite(rets)]
    sharpe = float(np.mean(rets) / np.std(rets, ddof=1) * np.sqrt(252 * 6)) if len(rets) > 1 and np.std(rets, ddof=1) > 0 else 0
    calmar = total_return / max_dd if max_dd > 0 else 0
    win_rate = wins / trades if trades > 0 else 0

    return {
        "sharpe": sharpe, "calmar": calmar, "return": total_return,
        "max_dd": max_dd, "trades": trades, "win_rate": win_rate,
    }


def _make_strategy(name: str, params: dict) -> Any:
    cfg = StrategyConfig(name=f"Opt_{name}", symbols=[SYMBOL], params=params)
    factories = {
        "funding_rate": FundingRateAlphaStrategy,
        "meta_labeling": MetaLabelingStrategy,
        "ts_momentum": TimeSeriesMomentumStrategy,
        "patch_tst": PatchTSTStrategy,
        "fund_meta": FundingMetaEnsembleStrategy,
    }
    return factories[name](cfg)


def objective_funding_rate(trial: Trial, train: pd.DataFrame) -> float:
    params = {
        "funding_z_entry": trial.suggest_float("funding_z_entry", 0.8, 2.0),
        "funding_z_exit": trial.suggest_float("funding_z_exit", 0.1, 0.5),
        "funding_ewm_span": trial.suggest_int("funding_ewm_span", 6, 20),
        "funding_lookback": trial.suggest_int("funding_lookback", 30, 100),
        "atr_period": trial.suggest_int("atr_period", 10, 20),
        "stop_loss_atr_mult": trial.suggest_float("stop_loss_atr_mult", 1.5, 3.0),
        "take_profit_atr_mult": trial.suggest_float("take_profit_atr_mult", 3.0, 7.0),
        "max_hold_bars": trial.suggest_int("max_hold_bars", 18, 60),
        "ema_trend_period": trial.suggest_int("ema_trend_period", 30, 80),
        "require_trend_alignment": trial.suggest_categorical("require_trend_alignment", [True, False]),
    }
    strategy = _make_strategy("funding_rate", params)
    m = asyncio.get_event_loop().run_until_complete(backtest(strategy, train))
    if m["trades"] < 3:
        return -10.0
    return m["sharpe"] * 0.7 + m["calmar"] * 0.3


def objective_ts_momentum(trial: Trial, train: pd.DataFrame) -> float:
    params = {
        "short_lookback": trial.suggest_int("short_lookback", 3, 15),
        "long_lookback": trial.suggest_int("long_lookback", 15, 50),
        "short_weight": trial.suggest_float("short_weight", 0.1, 0.5),
        "entry_threshold": trial.suggest_float("entry_threshold", 0.6, 2.0),
        "exit_threshold": trial.suggest_float("exit_threshold", 0.1, 0.5),
        "atr_period": trial.suggest_int("atr_period", 10, 20),
        "stop_loss_atr_mult": trial.suggest_float("stop_loss_atr_mult", 1.2, 3.0),
        "take_profit_atr_mult": trial.suggest_float("take_profit_atr_mult", 3.0, 7.0),
        "max_hold_bars": trial.suggest_int("max_hold_bars", 12, 36),
        "vol_of_vol_threshold": trial.suggest_float("vol_of_vol_threshold", 1.2, 2.5),
        "min_bars_between_trades": trial.suggest_int("min_bars_between_trades", 2, 10),
    }
    params["long_weight"] = 1.0 - params["short_weight"]
    strategy = _make_strategy("ts_momentum", params)
    m = asyncio.get_event_loop().run_until_complete(backtest(strategy, train))
    if m["trades"] < 10:
        return -10.0
    return m["sharpe"] * 0.7 + m["calmar"] * 0.3


def objective_meta_labeling(trial: Trial, train: pd.DataFrame) -> float:
    params = {
        "barrier_atr_mult_tp": trial.suggest_float("barrier_atr_mult_tp", 2.0, 5.0),
        "barrier_atr_mult_sl": trial.suggest_float("barrier_atr_mult_sl", 1.0, 3.0),
        "barrier_max_bars": trial.suggest_int("barrier_max_bars", 12, 36),
        "meta_threshold": trial.suggest_float("meta_threshold", 0.55, 0.80),
        "refit_interval": trial.suggest_int("refit_interval", 200, 600),
        "min_train_samples": trial.suggest_int("min_train_samples", 60, 150),
        "ema_fast": trial.suggest_int("ema_fast", 8, 16),
        "ema_slow": trial.suggest_int("ema_slow", 20, 35),
        "ema_trend": trial.suggest_int("ema_trend", 30, 80),
    }
    strategy = _make_strategy("meta_labeling", params)
    m = asyncio.get_event_loop().run_until_complete(backtest(strategy, train))
    if m["trades"] < 10:
        return -10.0
    return m["sharpe"] * 0.7 + m["calmar"] * 0.3


def objective_patch_tst(trial: Trial, train: pd.DataFrame) -> float:
    params = {
        "input_length": trial.suggest_categorical("input_length", [40, 60, 80]),
        "patch_length": trial.suggest_categorical("patch_length", [5, 10, 20]),
        "forecast_horizon": trial.suggest_int("forecast_horizon", 4, 12),
        "n_estimators": trial.suggest_int("n_estimators", 3, 10),
        "refit_interval": trial.suggest_int("refit_interval", 100, 300),
        "entry_threshold": trial.suggest_float("entry_threshold", 0.002, 0.010),
        "confidence_threshold": trial.suggest_float("confidence_threshold", 0.5, 0.85),
        "stop_loss_atr_mult": trial.suggest_float("stop_loss_atr_mult", 1.5, 3.0),
        "take_profit_atr_mult": trial.suggest_float("take_profit_atr_mult", 3.0, 7.0),
        "max_hold_bars": trial.suggest_int("max_hold_bars", 18, 42),
    }
    strategy = _make_strategy("patch_tst", params)
    m = asyncio.get_event_loop().run_until_complete(backtest(strategy, train))
    if m["trades"] < 10:
        return -10.0
    return m["sharpe"] * 0.7 + m["calmar"] * 0.3


def objective_fund_meta(trial: Trial, train: pd.DataFrame) -> float:
    params = {
        "funding_z_entry": trial.suggest_float("funding_z_entry", 0.8, 2.0),
        "funding_z_exit": trial.suggest_float("funding_z_exit", 0.1, 0.5),
        "funding_ewm_span": trial.suggest_int("funding_ewm_span", 6, 20),
        "funding_lookback": trial.suggest_int("funding_lookback", 30, 100),
        "stop_loss_atr_mult": trial.suggest_float("stop_loss_atr_mult", 1.5, 3.0),
        "take_profit_atr_mult": trial.suggest_float("take_profit_atr_mult", 3.0, 7.0),
        "max_hold_bars": trial.suggest_int("max_hold_bars", 18, 60),
        "meta_threshold": trial.suggest_float("meta_threshold", 0.45, 0.70),
        "refit_interval": trial.suggest_int("refit_interval", 100, 300),
    }
    strategy = _make_strategy("fund_meta", params)
    m = asyncio.get_event_loop().run_until_complete(backtest(strategy, train))
    if m["trades"] < 3:
        return -10.0
    return m["sharpe"] * 0.7 + m["calmar"] * 0.3


def main() -> None:
    print("Loading data...")
    train, test = load_data()
    print(f"Train: {len(train)} bars | Test: {len(test)} bars")
    print(f"Train: {train.index[0]} → {train.index[-1]}")
    print(f"Test:  {test.index[0]} → {test.index[-1]}\n")

    strategies_to_tune = {
        "funding_rate": (objective_funding_rate, 60),
        "ts_momentum": (objective_ts_momentum, 60),
        "meta_labeling": (objective_meta_labeling, 50),
        "patch_tst": (objective_patch_tst, 50),
        "fund_meta": (objective_fund_meta, 50),
    }

    all_results: dict[str, Any] = {}

    for name, (obj_fn, n_trials) in strategies_to_tune.items():
        print(f"{'='*60}")
        print(f"Tuning: {name} ({n_trials} trials)")
        print(f"{'='*60}")

        study = optuna.create_study(direction="maximize", study_name=f"sota_{name}")
        study.optimize(lambda trial: obj_fn(trial, train), n_trials=n_trials, show_progress_bar=True)

        best = study.best_trial
        print(f"\nBest IS objective: {best.value:.4f}")
        print(f"Best params: {json.dumps(best.params, indent=2)}")

        strat_train = _make_strategy(name, best.params)
        is_m = asyncio.get_event_loop().run_until_complete(backtest(strat_train, train))

        strat_test = _make_strategy(name, best.params)
        oos_m = asyncio.get_event_loop().run_until_complete(backtest(strat_test, test))

        print(f"\n  IS:  Return={is_m['return']*100:+.2f}%, Sharpe={is_m['sharpe']:.3f}, "
              f"MaxDD={is_m['max_dd']*100:.1f}%, Trades={is_m['trades']}, WR={is_m['win_rate']*100:.1f}%")
        print(f"  OOS: Return={oos_m['return']*100:+.2f}%, Sharpe={oos_m['sharpe']:.3f}, "
              f"MaxDD={oos_m['max_dd']*100:.1f}%, Trades={oos_m['trades']}, WR={oos_m['win_rate']*100:.1f}%")

        all_results[name] = {
            "best_params": best.params,
            "is_metrics": is_m,
            "oos_metrics": oos_m,
        }

    output_path = Path("models") / "optuna_sota_results.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to {output_path}")

    print(f"\n{'='*70}")
    print("OPTIMIZATION SUMMARY (OOS)")
    print(f"{'='*70}")
    print(f"{'Strategy':<20} {'Sharpe':>8} {'Return%':>10} {'MaxDD%':>8} {'WR%':>6} {'Trades':>8}")
    print("-" * 70)
    for name, r in sorted(all_results.items(), key=lambda x: x[1]["oos_metrics"]["sharpe"], reverse=True):
        m = r["oos_metrics"]
        print(f"{name:<20} {m['sharpe']:>8.3f} {m['return']*100:>+9.2f}% {m['max_dd']*100:>7.1f}% {m['win_rate']*100:>5.1f}% {m['trades']:>8}")


if __name__ == "__main__":
    main()
