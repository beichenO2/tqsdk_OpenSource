"""Optuna tuner for V4+Pullback dual-mode entry strategy.

Full parameter optimization: V4 base params (same ranges as optuna_v4_tune.py)
plus pullback-specific params. Train 70% / Test 30% completely isolated.

Usage:
    python scripts/optuna_v4_pullback_tune.py [--trials 200] [--leverage 8] [--timeframe 1h]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys


try:
    from polarisor_port_sdk import submit_task as _sdk_submit, complete_task as _sdk_complete
except ImportError:
    _sdk_submit = _sdk_complete = None
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from datahub.crypto_loader import CryptoDataLoader
from run_v4_pullback_fusion import backtest_v4_pullback

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def suggest_params(trial) -> dict:
    sl_atr = trial.suggest_float("sl_atr_mult", 0.8, 2.5)
    tp_atr = trial.suggest_float("tp_atr_mult", sl_atr * 1.5, 6.0)

    return {
        "lookback": trial.suggest_int("lookback", 12, 30),
        "sl_atr_mult": sl_atr,
        "tp_atr_mult": tp_atr,
        "max_hold_bars": trial.suggest_int("max_hold_bars", 36, 120),
        "cooldown_bars": trial.suggest_int("cooldown_bars", 1, 6),
        "vol_ma_period": trial.suggest_int("vol_ma_period", 8, 20),
        "vol_surge_mult": trial.suggest_float("vol_surge_mult", 1.0, 1.5),
        "tbr_long_min": trial.suggest_float("tbr_long_min", 0.51, 0.58),
        "tbr_short_max": trial.suggest_float("tbr_short_max", 0.42, 0.49),
        "nbz_threshold": trial.suggest_float("nbz_threshold", 0.3, 2.0),
        "ats_threshold": trial.suggest_float("ats_threshold", 1.0, 2.5),
        "atr_period": trial.suggest_int("atr_period", 10, 24),
        "adx_min": trial.suggest_float("adx_min", 15.0, 35.0),
        "vol_regime_period": trial.suggest_int("vol_regime_period", 24, 72),
        "vol_regime_max": trial.suggest_float("vol_regime_max", 1.5, 4.0),
        "trail_atr_mult": trial.suggest_float("trail_atr_mult", 0.0, 4.0),
        "trail_dist_atr": trial.suggest_float("trail_dist_atr", 0.5, 2.0),
        # Pullback-specific params
        "pullback_tolerance_atr": trial.suggest_float("pullback_tolerance_atr", 0.1, 1.5, step=0.05),
        "pullback_max_wait_bars": trial.suggest_int("pullback_max_wait_bars", 3, 24),
        "pullback_addon_frac": trial.suggest_float("pullback_addon_frac", 0.3, 0.8, step=0.1),
    }


def objective(trial, train_datasets, leverage):
    params = suggest_params(trial)

    scores = []
    for sym, bars in train_datasets:
        r = backtest_v4_pullback(bars, params, leverage=leverage)
        if r["max_dd"] > 0.50:
            return -10.0
        if r["trades"] < 5:
            return -5.0
        score = r["sharpe"] * 0.4 + r["profit_factor"] * 0.3 + r["calmar"] * 0.3
        scores.append(score)

    return float(np.mean(scores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--leverage", type=int, default=8)
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--weeks", type=int, default=80)
    args = parser.parse_args()

    loader = CryptoDataLoader()
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    train_datasets = []
    full_datasets = []
    test_datasets = []

    for sym in symbols:
        b = loader.load_with_funding(sym, args.timeframe)
        if b.empty:
            b = loader.load(sym, args.timeframe)
        if b.empty:
            logger.warning("No data for %s", sym)
            continue
        b.attrs["timeframe"] = args.timeframe
        if "open_time" in b.columns:
            cutoff = b["open_time"].iloc[-1] - pd.Timedelta(weeks=args.weeks)
            b = b[b["open_time"] >= cutoff].copy()
            b.attrs["timeframe"] = args.timeframe

        split = int(len(b) * 0.7)
        train_b = b.iloc[:split].copy()
        test_b = b.iloc[split:].copy()
        train_b.attrs = b.attrs.copy()
        test_b.attrs = b.attrs.copy()
        train_datasets.append((sym, train_b))
        test_datasets.append((sym, test_b))
        full_datasets.append((sym, b))

    if not train_datasets:
        logger.error("No training data")
        return

    logger.info("V4+Pullback Full Optuna | %d symbols | %d trials | %s %dx",
                len(train_datasets), args.trials, args.timeframe, args.leverage)

    _task_id = None
    if _sdk_submit:
        try:
            _tr = _sdk_submit(task_type="optimization", command=f"optuna_v4_pullback_tune.py --trials {args.trials}", requester="optuna-v4-pullback", estimated_duration_sec=3600)
            _task_id = _tr.get("task_id")
        except Exception:
            pass

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective(t, train_datasets, args.leverage), n_trials=args.trials)

    best_params = suggest_params(study.best_trial)
    logger.info("Best score: %.4f", study.best_value)
    logger.info("Best params: %s", json.dumps(best_params, indent=2))

    from optuna_v4_tune import backtest_v4

    v4_path = Path(__file__).parent.parent / "models" / "optuna_v4_results.json"
    with open(v4_path) as f:
        v4_base = json.load(f)["best_params"]

    logger.info("\n=== Comparison: V4 Only vs V4+Pullback (Optimized) ===")
    results_all = {}
    for sym, full_b in full_datasets:
        test_b = [tb for s, tb in test_datasets if s == sym][0]

        r_v4 = backtest_v4(full_b, v4_base, leverage=args.leverage)
        r_fusion = backtest_v4_pullback(full_b, best_params, leverage=args.leverage)
        r_oos = backtest_v4_pullback(test_b, best_params, leverage=args.leverage)

        results_all[sym] = {"v4": r_v4, "fusion": r_fusion, "oos": r_oos}

        logger.info("\n%s:", sym)
        logger.info("  V4 Only:        $%.2f (%+.1f%%) T=%d WR=%.0f%% PF=%.3f DD=%.1f%% Sharpe=%.2f",
                     r_v4["final_capital"], r_v4["total_return"] * 100,
                     r_v4["trades"], r_v4["win_rate"] * 100, r_v4["profit_factor"],
                     r_v4["max_dd"] * 100, r_v4["sharpe"])
        logger.info("  V4+PB Optimized: $%.2f (%+.1f%%) T=%d(BO=%d PB=%d) WR=%.0f%% PF=%.3f DD=%.1f%% Sharpe=%.2f",
                     r_fusion["final_capital"], r_fusion["total_return"] * 100,
                     r_fusion["trades"], r_fusion.get("breakout_entries", 0), r_fusion.get("pullback_entries", 0),
                     r_fusion["win_rate"] * 100, r_fusion["profit_factor"],
                     r_fusion["max_dd"] * 100, r_fusion["sharpe"])
        logger.info("  OOS (30%%):      $%.2f (%+.1f%%) T=%d WR=%.0f%% PF=%.3f DD=%.1f%% Sharpe=%.2f",
                     r_oos["final_capital"], r_oos["total_return"] * 100,
                     r_oos["trades"], r_oos["win_rate"] * 100, r_oos["profit_factor"],
                     r_oos["max_dd"] * 100, r_oos["sharpe"])

    outpath = Path("models") / "optuna_v4_pullback_results.json"
    outpath.parent.mkdir(exist_ok=True)
    with open(outpath, "w") as f:
        json.dump({
            "best_params": best_params,
            "best_score": study.best_value,
            "n_trials": args.trials,
            "leverage": args.leverage,
            "timeframe": args.timeframe,
        }, f, indent=2, default=str)
    logger.info("\nSaved to %s", outpath)

    if _task_id and _sdk_complete:
        try:
            _sdk_complete(_task_id)
        except Exception:
            pass


if __name__ == "__main__":
    main()
