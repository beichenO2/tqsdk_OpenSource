"""Optuna hyperparameter search for the Orderflow-Enhanced Breakout strategy.

Uses walk-forward: train on first 70% of data, evaluate on last 30%.
Objective: maximize leveraged Sharpe ratio with minimum trade count constraint.

Usage:
    python scripts/optuna_scalp_tune.py [--symbol BTCUSDT] [--trials 100] [--leverage 8]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys


try:
    from polarisor_port_sdk import submit_task as _sdk_submit, complete_task as _sdk_complete
except ImportError:
    _sdk_submit = _sdk_complete = None
from collections import deque
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from datahub.crypto_loader import CryptoDataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def backtest_breakout(
    bars: pd.DataFrame,
    params: dict,
    leverage: int = 8,
    initial_capital: float = 100.0,
    commission_pct: float = 0.0004,
    slippage_pct: float = 0.0003,
    position_fraction: float = 0.9,
) -> dict:
    """Run a single backtest with given params and return metrics."""
    closes = bars["close"].values
    highs = bars["high"].values
    lows = bars["low"].values
    volumes = bars["volume"].values
    tbvs = bars.get("taker_buy_volume", bars["volume"] * 0.5).values
    quote_vols = bars.get("quote_volume", bars["volume"] * bars["close"]).values
    trade_counts = bars.get("trades", pd.Series([1] * len(bars))).values

    lookback = params["lookback"]
    tp_pct = params["tp_pct"]
    sl_pct = params["sl_pct"]
    max_hold = params["max_hold_bars"]
    vol_ma_period = params["vol_ma_period"]
    vol_surge_mult = params["vol_surge_mult"]
    tbr_long_min = params["tbr_long_min"]
    tbr_short_max = params["tbr_short_max"]
    nbz_threshold = params["nbz_threshold"]
    ats_threshold = params["ats_threshold"]
    cooldown = params["cooldown_bars"]

    cost_per_side = commission_pct + slippage_pct

    capital = initial_capital
    holding = None
    hold_bars = 0
    cd = 0
    wins = 0
    losses_n = 0
    tw = 0.0
    tl = 0.0
    liqs = 0
    peak = capital
    max_dd = 0.0

    net_buys = deque(maxlen=200)
    avg_ts_buf = deque(maxlen=50)

    weekly_caps = [capital]
    bars_per_week = {"1m": 10080, "5m": 2016, "15m": 672, "1h": 168, "4h": 42}.get(
        bars.attrs.get("timeframe", "1h"), 168
    )

    warmup = max(lookback + 1, vol_ma_period + 1, 50)

    for i in range(len(closes)):
        tbr = tbvs[i] / max(volumes[i], 1)
        net_buy = (2 * tbr - 1) * volumes[i]
        net_buys.append(net_buy)
        tc = max(trade_counts[i], 1)
        avg_ts_buf.append(quote_vols[i] / tc)

        if (i - warmup) > 0 and (i - warmup) % bars_per_week == 0:
            weekly_caps.append(capital)

        if i < warmup:
            continue

        if cd > 0:
            cd -= 1

        if holding is not None:
            hold_bars += 1
            side = holding[1]

            if side == "long":
                if lows[i] <= holding[2]:
                    exit_p = holding[2]
                elif highs[i] >= holding[3]:
                    exit_p = holding[3]
                elif hold_bars >= max_hold:
                    exit_p = closes[i]
                else:
                    exit_p = None
            else:
                if highs[i] >= holding[2]:
                    exit_p = holding[2]
                elif lows[i] <= holding[3]:
                    exit_p = holding[3]
                elif hold_bars >= max_hold:
                    exit_p = closes[i]
                else:
                    exit_p = None

            if exit_p is not None:
                if side == "long":
                    pnl_pct = (exit_p - holding[0]) / holding[0]
                else:
                    pnl_pct = (holding[0] - exit_p) / holding[0]

                notional = holding[4]
                gross = notional * pnl_pct
                cost = notional * cost_per_side
                net = gross - cost
                capital += net
                capital = max(capital, 0.01)

                if net > 0:
                    wins += 1
                    tw += net
                else:
                    losses_n += 1
                    tl += abs(net)

                holding = None
                cd = cooldown
                peak = max(peak, capital)
                dd = (peak - capital) / peak
                max_dd = max(max_dd, dd)
                continue

            lev_pnl_pct = ((closes[i] - holding[0]) / holding[0] if side == "long"
                           else (holding[0] - closes[i]) / holding[0])
            if lev_pnl_pct * leverage * position_fraction <= -0.9:
                capital *= 0.1
                liqs += 1
                holding = None
                continue

        if holding is None and capital > 1.0 and cd <= 0:
            prev_high = max(highs[i - lookback:i])
            prev_low = min(lows[i - lookback:i])

            is_long = closes[i] > prev_high
            is_short = closes[i] < prev_low

            if is_long or is_short:
                vol_ma = np.mean(volumes[max(0, i - vol_ma_period):i])
                if vol_ma > 0 and volumes[i] >= vol_ma * vol_surge_mult:
                    tbr_v = tbvs[i] / max(volumes[i], 1)
                    tbr_ok = (is_long and tbr_v >= tbr_long_min) or (is_short and tbr_v <= tbr_short_max)

                    if tbr_ok and len(net_buys) >= 48:
                        nbs = list(net_buys)
                        nb_mean = sum(nbs[-48:]) / 48
                        nb_var = sum((x - nb_mean) ** 2 for x in nbs[-48:]) / 48
                        nb_std = nb_var ** 0.5 if nb_var > 0 else 1.0
                        nbz = (nbs[-1] - nb_mean) / nb_std if nb_std > 1e-10 else 0.0

                        nbz_ok = (is_long and nbz >= nbz_threshold) or (is_short and nbz <= -nbz_threshold)

                        if nbz_ok and len(avg_ts_buf) >= 20:
                            current_ats = avg_ts_buf[-1]
                            hist_ats = sum(list(avg_ts_buf)[-21:-1]) / 20
                            ats_ok = hist_ats > 0 and current_ats / hist_ats >= ats_threshold

                            if ats_ok:
                                entry_p = closes[i]
                                notional = capital * position_fraction * leverage
                                entry_cost = notional * cost_per_side
                                capital -= entry_cost

                                if is_long:
                                    sl_p = entry_p * (1 - sl_pct)
                                    tp_p = entry_p * (1 + tp_pct)
                                else:
                                    sl_p = entry_p * (1 + sl_pct)
                                    tp_p = entry_p * (1 - tp_pct)

                                holding = (entry_p, "long" if is_long else "short", sl_p, tp_p, notional)
                                hold_bars = 0

    weekly_caps.append(capital)

    total = wins + losses_n
    wr = wins / total if total > 0 else 0
    pf = tw / tl if tl > 0 else 0.0
    total_return = (capital - initial_capital) / initial_capital

    weekly_rets = []
    for j in range(1, len(weekly_caps)):
        if weekly_caps[j - 1] > 0:
            weekly_rets.append((weekly_caps[j] - weekly_caps[j - 1]) / weekly_caps[j - 1])

    if weekly_rets and np.std(weekly_rets) > 0:
        sharpe = np.mean(weekly_rets) / np.std(weekly_rets) * np.sqrt(52)
    else:
        sharpe = 0.0

    calmar = total_return / max_dd if max_dd > 0 else 0.0

    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "calmar": calmar,
        "max_dd": max_dd,
        "trades": total,
        "win_rate": wr,
        "profit_factor": pf,
        "liquidations": liqs,
        "final_capital": capital,
    }


def create_objective(train_datasets: list[pd.DataFrame], leverage: int, min_trades: int = 20):
    """Multi-asset joint objective: sum of scores across all assets.

    Enforces tp > sl (positive expectancy structure) and penalizes
    any asset with PF < 1.0.
    """

    def objective(trial: optuna.Trial) -> float:
        sl_pct = trial.suggest_float("sl_pct", 0.010, 0.025)
        tp_pct = trial.suggest_float("tp_pct", sl_pct * 1.2, 0.06)

        params = {
            "lookback": trial.suggest_int("lookback", 12, 30),
            "tp_pct": tp_pct,
            "sl_pct": sl_pct,
            "max_hold_bars": trial.suggest_int("max_hold_bars", 24, 96),
            "cooldown_bars": trial.suggest_int("cooldown_bars", 1, 5),
            "vol_ma_period": trial.suggest_int("vol_ma_period", 10, 25),
            "vol_surge_mult": trial.suggest_float("vol_surge_mult", 1.05, 1.5),
            "tbr_long_min": trial.suggest_float("tbr_long_min", 0.52, 0.58),
            "tbr_short_max": trial.suggest_float("tbr_short_max", 0.42, 0.48),
            "nbz_threshold": trial.suggest_float("nbz_threshold", 0.5, 2.0),
            "ats_threshold": trial.suggest_float("ats_threshold", 0.9, 1.5),
        }

        total_score = 0.0
        any_liq = False
        pfs = []

        for bars in train_datasets:
            result = backtest_breakout(bars, params, leverage=leverage)

            if result["liquidations"] > 0:
                any_liq = True
                break

            trades = result["trades"]
            if trades < 10:
                total_score += -3.0
                continue

            pf = result["profit_factor"]
            pfs.append(pf)
            sharpe = result["sharpe"]
            calmar = result["calmar"]

            asset_score = 0.35 * sharpe + 0.30 * min(calmar, 5.0) + 0.35 * min(pf, 4.0)

            if trades < min_trades:
                asset_score *= (trades / min_trades) ** 0.5

            if result["max_dd"] > 0.75:
                asset_score *= 0.3

            total_score += asset_score

        if any_liq:
            return -10.0

        if pfs and min(pfs) < 1.0:
            total_score *= 0.5

        return total_score / len(train_datasets)

    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--weeks", type=int, default=80)
    parser.add_argument("--trials", type=int, default=150)
    parser.add_argument("--leverage", type=int, default=8)
    args = parser.parse_args()

    loader = CryptoDataLoader()
    train_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    train_datasets: list[pd.DataFrame] = []
    test_datasets: dict[str, pd.DataFrame] = {}

    for sym in train_symbols:
        b = loader.load_with_funding(sym, args.timeframe)
        if b.empty:
            b = loader.load(sym, args.timeframe)
        if b.empty:
            logger.warning("No data for %s, skipping", sym)
            continue
        b.attrs["timeframe"] = args.timeframe
        if "open_time" in b.columns:
            cutoff = b["open_time"].iloc[-1] - pd.Timedelta(weeks=args.weeks)
            b = b[b["open_time"] >= cutoff].copy()
            b.attrs["timeframe"] = args.timeframe
        s_idx = int(len(b) * 0.7)
        tr = b.iloc[:s_idx].copy()
        te = b.iloc[s_idx:].copy()
        tr.attrs = b.attrs.copy()
        te.attrs = b.attrs.copy()
        train_datasets.append(tr)
        test_datasets[sym] = te
        logger.info("Loaded %s: %d train / %d test bars", sym, len(tr), len(te))

    if not train_datasets:
        logger.error("No training data available")
        return

    bars = loader.load_with_funding(args.symbol, args.timeframe)
    if bars.empty:
        bars = loader.load(args.symbol, args.timeframe)
    bars.attrs["timeframe"] = args.timeframe
    if "open_time" in bars.columns:
        cutoff = bars["open_time"].iloc[-1] - pd.Timedelta(weeks=args.weeks)
        bars = bars[bars["open_time"] >= cutoff].copy()
        bars.attrs["timeframe"] = args.timeframe
    split_idx = int(len(bars) * 0.7)
    train = bars.iloc[:split_idx].copy()
    test = bars.iloc[split_idx:].copy()
    train.attrs = bars.attrs.copy()
    test.attrs = bars.attrs.copy()

    logger.info("Multi-asset training on: %s", train_symbols)
    logger.info("Leverage: %dx | Trials: %d", args.leverage, args.trials)

    _task_id = None
    if _sdk_submit:
        try:
            _tr = _sdk_submit(task_type="optimization", command=f"optuna_scalp_tune.py --trials {args.trials}", requester="optuna-scalp-tune", estimated_duration_sec=3600)
            _task_id = _tr.get("task_id")
        except Exception:
            pass

    study = optuna.create_study(direction="maximize", study_name="scalp_breakout_multi")
    objective = create_objective(train_datasets, args.leverage)
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    best = study.best_trial
    logger.info("\nBest trial: #%d (score=%.4f)", best.number, best.value)
    logger.info("Best params: %s", json.dumps(best.params, indent=2))

    best_params = best.params

    train_result = backtest_breakout(train, best_params, leverage=args.leverage)
    test_result = backtest_breakout(test, best_params, leverage=args.leverage)

    logger.info("\nIN-SAMPLE (train):")
    logger.info("  Return: %.2f%% | Sharpe: %.3f | MaxDD: %.2f%% | Trades: %d | WR: %.1f%% | PF: %.3f",
                train_result["total_return"] * 100, train_result["sharpe"],
                train_result["max_dd"] * 100, train_result["trades"],
                train_result["win_rate"] * 100, train_result["profit_factor"])

    logger.info("\nOUT-OF-SAMPLE (test):")
    logger.info("  Return: %.2f%% | Sharpe: %.3f | MaxDD: %.2f%% | Trades: %d | WR: %.1f%% | PF: %.3f",
                test_result["total_return"] * 100, test_result["sharpe"],
                test_result["max_dd"] * 100, test_result["trades"],
                test_result["win_rate"] * 100, test_result["profit_factor"])
    logger.info("  Final capital: $%.2f (from $100)", test_result["final_capital"])

    for lev in [1, 3, 5, 8, 10, 15]:
        r = backtest_breakout(test, best_params, leverage=lev)
        logger.info("  %2dx: $%8.2f (%+.1f%%) PF=%.3f DD=%.1f%%",
                     lev, r["final_capital"], r["total_return"] * 100,
                     r["profit_factor"], r["max_dd"] * 100)

    logger.info("\nCross-asset validation (OOS, %dx):", args.leverage)
    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        b = loader.load_with_funding(sym, args.timeframe)
        if b.empty:
            b = loader.load(sym, args.timeframe)
        if b.empty:
            continue
        b.attrs["timeframe"] = args.timeframe
        if "open_time" in b.columns:
            cutoff2 = b["open_time"].iloc[-1] - pd.Timedelta(weeks=args.weeks)
            b = b[b["open_time"] >= cutoff2].copy()
            b.attrs["timeframe"] = args.timeframe
        s_idx = int(len(b) * 0.7)
        b_test = b.iloc[s_idx:].copy()
        b_test.attrs = b.attrs.copy()
        r = backtest_breakout(b_test, best_params, leverage=args.leverage)
        logger.info("  %s: $%.2f (%+.1f%%) trades=%d WR=%.1f%% PF=%.3f",
                     sym, r["final_capital"], r["total_return"] * 100,
                     r["trades"], r["win_rate"] * 100, r["profit_factor"])

    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    output = {
        "best_params": best_params,
        "best_score": best.value,
        "train_metrics": train_result,
        "test_metrics": test_result,
        "leverage": args.leverage,
        "symbol": args.symbol,
        "timeframe": args.timeframe,
    }
    outpath = output_dir / f"optuna_scalp_{args.symbol}_{args.timeframe}_{args.leverage}x.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("\nResults saved to %s", outpath)

    if _task_id and _sdk_complete:
        try:
            _sdk_complete(_task_id)
        except Exception:
            pass


if __name__ == "__main__":
    main()
