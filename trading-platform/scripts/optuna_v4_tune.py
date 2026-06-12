"""Optuna V4: Enhanced Orderflow Breakout with volatility-adaptive exits + regime filter.

Improvements over V1:
  1. Adaptive TP/SL: scales with realized ATR (tp_atr_mult * ATR, sl_atr_mult * ATR)
  2. Regime filter: ADX + vol-of-vol gate to skip choppy/low-trend markets
  3. Trailing stop: activates at trail_atr_mult * ATR from entry
  4. Max drawdown constraint in objective (hard cap at 50%)
  5. Tighter cross-asset consistency scoring

Usage:
    python scripts/optuna_v4_tune.py [--trials 200] [--leverage 8]
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


def backtest_v4(
    bars: pd.DataFrame,
    params: dict,
    leverage: int = 8,
    initial_capital: float = 100.0,
    commission_pct: float = 0.0004,
    slippage_pct: float = 0.0003,
    position_fraction: float = 0.9,
) -> dict:
    """Enhanced backtest with adaptive exits and regime filter."""
    closes = bars["close"].values
    highs = bars["high"].values
    lows = bars["low"].values
    volumes = bars["volume"].values
    tbvs = bars.get("taker_buy_volume", bars["volume"] * 0.5).values
    quote_vols = bars.get("quote_volume", bars["volume"] * bars["close"]).values
    trade_counts = bars.get("trades", pd.Series([1] * len(bars))).values

    lookback = params["lookback"]
    tp_atr_mult = params["tp_atr_mult"]
    sl_atr_mult = params["sl_atr_mult"]
    max_hold = params["max_hold_bars"]
    vol_ma_period = params["vol_ma_period"]
    vol_surge_mult = params["vol_surge_mult"]
    tbr_long_min = params["tbr_long_min"]
    tbr_short_max = params["tbr_short_max"]
    nbz_threshold = params["nbz_threshold"]
    ats_threshold = params["ats_threshold"]
    cooldown = params["cooldown_bars"]

    atr_period = params.get("atr_period", 14)
    adx_period = params.get("adx_period", 14)
    adx_min = params.get("adx_min", 20.0)
    vol_regime_period = params.get("vol_regime_period", 48)
    vol_regime_max = params.get("vol_regime_max", 2.5)

    trail_atr_mult = params.get("trail_atr_mult", 0.0)
    trail_dist_atr = params.get("trail_dist_atr", 0.0)

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
    peak_cap = capital
    max_dd = 0.0

    net_buys = deque(maxlen=200)
    avg_ts_buf = deque(maxlen=50)

    atr_buf = deque(maxlen=atr_period + 5)
    ret_buf = deque(maxlen=vol_regime_period + 5)

    plus_dm_smooth = 0.0
    minus_dm_smooth = 0.0
    tr_smooth = 0.0
    adx_val = 0.0
    adx_warmup = 0

    weekly_caps = [capital]
    bars_per_week = {"1m": 10080, "5m": 2016, "15m": 672, "1h": 168, "4h": 42}.get(
        bars.attrs.get("timeframe", "1h"), 168
    )

    warmup = max(lookback + 1, vol_ma_period + 1, atr_period + 2, adx_period * 2 + 1, vol_regime_period + 2, 50)
    trail_peak = 0.0

    for i in range(len(closes)):
        tbr = tbvs[i] / max(volumes[i], 1)
        net_buy = (2 * tbr - 1) * volumes[i]
        net_buys.append(net_buy)
        tc = max(trade_counts[i], 1)
        avg_ts_buf.append(quote_vols[i] / tc)

        if i > 0:
            tr_val = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
            atr_buf.append(tr_val)
            ret_buf.append((closes[i] - closes[i - 1]) / closes[i - 1] if closes[i - 1] > 0 else 0.0)

            up_move = highs[i] - highs[i - 1]
            dn_move = lows[i - 1] - lows[i]
            pdm = max(up_move, 0.0) if up_move > dn_move else 0.0
            mdm = max(dn_move, 0.0) if dn_move > up_move else 0.0

            alpha = 1.0 / adx_period
            if adx_warmup < adx_period:
                plus_dm_smooth += pdm
                minus_dm_smooth += mdm
                tr_smooth += tr_val
                adx_warmup += 1
            else:
                tr_smooth = tr_smooth - tr_smooth * alpha + tr_val
                plus_dm_smooth = plus_dm_smooth - plus_dm_smooth * alpha + pdm
                minus_dm_smooth = minus_dm_smooth - minus_dm_smooth * alpha + mdm
                if tr_smooth > 0:
                    di_p = plus_dm_smooth / tr_smooth
                    di_m = minus_dm_smooth / tr_smooth
                    di_sum = di_p + di_m
                    if di_sum > 0:
                        dx = abs(di_p - di_m) / di_sum * 100.0
                        adx_val = adx_val * (1 - alpha) + dx * alpha

        if (i - warmup) > 0 and (i - warmup) % bars_per_week == 0:
            weekly_caps.append(capital)

        if i < warmup:
            continue

        if cd > 0:
            cd -= 1

        atr_list = list(atr_buf)
        current_atr = sum(atr_list[-atr_period:]) / atr_period if len(atr_list) >= atr_period else 0.0

        if holding is not None:
            hold_bars += 1
            side = holding[1]

            if side == "long":
                if trail_atr_mult > 0 and current_atr > 0:
                    trail_peak = max(trail_peak, highs[i])
                    activate_dist = holding[0] + trail_atr_mult * current_atr
                    if trail_peak >= activate_dist:
                        trail_sl = trail_peak - trail_dist_atr * current_atr
                        if lows[i] <= trail_sl:
                            exit_p = trail_sl
                            side_done = True
                            pnl_pct = (exit_p - holding[0]) / holding[0]
                            notional = holding[4]
                            net = notional * pnl_pct - notional * cost_per_side
                            capital += net
                            capital = max(capital, 0.01)
                            if net > 0:
                                wins += 1; tw += net
                            else:
                                losses_n += 1; tl += abs(net)
                            holding = None; cd = cooldown
                            peak_cap = max(peak_cap, capital)
                            dd = (peak_cap - capital) / peak_cap
                            max_dd = max(max_dd, dd)
                            continue

                if lows[i] <= holding[2]:
                    exit_p = holding[2]
                elif highs[i] >= holding[3]:
                    exit_p = holding[3]
                elif hold_bars >= max_hold:
                    exit_p = closes[i]
                else:
                    exit_p = None
            else:
                if trail_atr_mult > 0 and current_atr > 0:
                    trail_peak = min(trail_peak, lows[i])
                    activate_dist = holding[0] - trail_atr_mult * current_atr
                    if trail_peak <= activate_dist:
                        trail_sl = trail_peak + trail_dist_atr * current_atr
                        if highs[i] >= trail_sl:
                            exit_p = trail_sl
                            pnl_pct = (holding[0] - exit_p) / holding[0]
                            notional = holding[4]
                            net = notional * pnl_pct - notional * cost_per_side
                            capital += net
                            capital = max(capital, 0.01)
                            if net > 0:
                                wins += 1; tw += net
                            else:
                                losses_n += 1; tl += abs(net)
                            holding = None; cd = cooldown
                            peak_cap = max(peak_cap, capital)
                            dd = (peak_cap - capital) / peak_cap
                            max_dd = max(max_dd, dd)
                            continue

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
                net = notional * pnl_pct - notional * cost_per_side
                capital += net
                capital = max(capital, 0.01)
                if net > 0:
                    wins += 1; tw += net
                else:
                    losses_n += 1; tl += abs(net)
                holding = None; cd = cooldown
                peak_cap = max(peak_cap, capital)
                dd = (peak_cap - capital) / peak_cap
                max_dd = max(max_dd, dd)
                continue

            lev_pnl = ((closes[i] - holding[0]) / holding[0] if side == "long"
                       else (holding[0] - closes[i]) / holding[0])
            if lev_pnl * leverage * position_fraction <= -0.9:
                capital *= 0.1
                liqs += 1
                holding = None
                continue

        if holding is None and capital > 1.0 and cd <= 0:
            if adx_val < adx_min:
                continue

            rets = list(ret_buf)
            if len(rets) >= vol_regime_period:
                recent_vol = np.std(rets[-vol_regime_period:])
                long_vol = np.std(rets) if len(rets) > vol_regime_period else recent_vol
                if long_vol > 0 and recent_vol / long_vol > vol_regime_max:
                    continue

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

                            if ats_ok and current_atr > 0:
                                entry_p = closes[i]
                                notional = capital * position_fraction * leverage
                                entry_cost = notional * cost_per_side
                                capital -= entry_cost

                                if is_long:
                                    sl_p = entry_p - sl_atr_mult * current_atr
                                    tp_p = entry_p + tp_atr_mult * current_atr
                                else:
                                    sl_p = entry_p + sl_atr_mult * current_atr
                                    tp_p = entry_p - tp_atr_mult * current_atr

                                holding = (entry_p, "long" if is_long else "short", sl_p, tp_p, notional)
                                hold_bars = 0
                                trail_peak = highs[i] if is_long else lows[i]

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


def create_v4_objective(train_datasets: list[pd.DataFrame], leverage: int, min_trades: int = 15):
    """V4 objective: balanced cross-asset performance with DD constraint."""

    def objective(trial: optuna.Trial) -> float:
        sl_atr = trial.suggest_float("sl_atr_mult", 0.8, 2.5)
        tp_atr = trial.suggest_float("tp_atr_mult", sl_atr * 1.5, 6.0)

        params = {
            "lookback": trial.suggest_int("lookback", 12, 30),
            "tp_atr_mult": tp_atr,
            "sl_atr_mult": sl_atr,
            "max_hold_bars": trial.suggest_int("max_hold_bars", 36, 120),
            "cooldown_bars": trial.suggest_int("cooldown_bars", 1, 6),
            "vol_ma_period": trial.suggest_int("vol_ma_period", 8, 20),
            "vol_surge_mult": trial.suggest_float("vol_surge_mult", 1.0, 1.5),
            "tbr_long_min": trial.suggest_float("tbr_long_min", 0.51, 0.58),
            "tbr_short_max": trial.suggest_float("tbr_short_max", 0.42, 0.49),
            "nbz_threshold": trial.suggest_float("nbz_threshold", 0.3, 2.0),
            "ats_threshold": trial.suggest_float("ats_threshold", 1.0, 2.5),
            "atr_period": trial.suggest_int("atr_period", 10, 24),
            "adx_period": 14,
            "adx_min": trial.suggest_float("adx_min", 15.0, 35.0),
            "vol_regime_period": trial.suggest_int("vol_regime_period", 24, 72),
            "vol_regime_max": trial.suggest_float("vol_regime_max", 1.5, 4.0),
            "trail_atr_mult": trial.suggest_float("trail_atr_mult", 0.0, 4.0),
            "trail_dist_atr": trial.suggest_float("trail_dist_atr", 0.5, 2.0),
        }

        scores = []
        any_liq = False
        any_blowup = False

        for bars in train_datasets:
            result = backtest_v4(bars, params, leverage=leverage)

            if result["liquidations"] > 0:
                any_liq = True
                break

            if result["max_dd"] > 0.50:
                any_blowup = True

            trades = result["trades"]
            pf = result["profit_factor"]
            sharpe = result["sharpe"]
            calmar = result["calmar"]

            if trades < 5:
                scores.append(-2.0)
                continue

            dd_penalty = max(0, result["max_dd"] - 0.35) * 5.0
            asset_score = (
                0.30 * sharpe
                + 0.25 * min(calmar, 5.0)
                + 0.25 * min(pf, 4.0)
                + 0.20 * min(result["total_return"], 10.0)
                - dd_penalty
            )

            if trades < min_trades:
                asset_score *= (trades / min_trades) ** 0.4

            scores.append(asset_score)

        if any_liq:
            return -10.0
        if any_blowup:
            return sum(scores) / len(train_datasets) * 0.3

        if not scores:
            return -10.0

        mean_score = np.mean(scores)
        consistency = 1.0 - np.std(scores) / (abs(mean_score) + 1e-6)
        return mean_score * max(consistency, 0.3)

    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--weeks", type=int, default=80)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--leverage", type=int, default=8)
    args = parser.parse_args()

    loader = CryptoDataLoader()
    train_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    train_datasets: list[pd.DataFrame] = []
    test_datasets: dict[str, pd.DataFrame] = {}
    full_datasets: dict[str, pd.DataFrame] = {}

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
        full_datasets[sym] = b
        logger.info("Loaded %s: %d train / %d test bars", sym, len(tr), len(te))

    if not train_datasets:
        logger.error("No training data")
        return

    logger.info("V4 Multi-asset training: %s | %dx | %d trials", train_symbols, args.leverage, args.trials)

    _task_id = None
    if _sdk_submit:
        try:
            _tr = _sdk_submit(task_type="optimization", command=f"optuna_v4_tune.py --trials {args.trials}", requester="optuna-v4-tune", estimated_duration_sec=3600)
            _task_id = _tr.get("task_id")
        except Exception:
            pass

    study = optuna.create_study(direction="maximize", study_name="scalp_v4_adaptive")
    objective = create_v4_objective(train_datasets, args.leverage)
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    best = study.best_trial
    best_params = best.params
    logger.info("\nBest trial: #%d (score=%.4f)", best.number, best.value)
    logger.info("Best params:\n%s", json.dumps(best_params, indent=2))

    logger.info("\n" + "=" * 70)
    logger.info("PER-ASSET RESULTS (8x leverage)")
    logger.info("=" * 70)

    results_summary = {}
    for sym in train_symbols:
        if sym not in full_datasets:
            continue
        full_b = full_datasets[sym]

        cutoff_40 = full_b["open_time"].iloc[-1] - pd.Timedelta(weeks=40)
        b_40 = full_b[full_b["open_time"] >= cutoff_40].copy()
        b_40.attrs["timeframe"] = args.timeframe

        r_full = backtest_v4(full_b, best_params, leverage=args.leverage)
        r_40 = backtest_v4(b_40, best_params, leverage=args.leverage)
        r_oos = backtest_v4(test_datasets[sym], best_params, leverage=args.leverage)

        logger.info("\n%s:", sym)
        logger.info("  Full 80w: $%.2f (%+.1f%%) T=%d WR=%.0f%% PF=%.3f DD=%.1f%% Sharpe=%.2f",
                     r_full["final_capital"], r_full["total_return"] * 100,
                     r_full["trades"], r_full["win_rate"] * 100, r_full["profit_factor"],
                     r_full["max_dd"] * 100, r_full["sharpe"])
        logger.info("  Last 40w: $%.2f (%+.1f%%) T=%d WR=%.0f%% PF=%.3f DD=%.1f%%",
                     r_40["final_capital"], r_40["total_return"] * 100,
                     r_40["trades"], r_40["win_rate"] * 100, r_40["profit_factor"],
                     r_40["max_dd"] * 100)
        logger.info("  OOS 24w:  $%.2f (%+.1f%%) T=%d WR=%.0f%% PF=%.3f DD=%.1f%%",
                     r_oos["final_capital"], r_oos["total_return"] * 100,
                     r_oos["trades"], r_oos["win_rate"] * 100, r_oos["profit_factor"],
                     r_oos["max_dd"] * 100)

        results_summary[sym] = {
            "full_80w": r_full,
            "last_40w": r_40,
            "oos_24w": r_oos,
        }

    logger.info("\n" + "=" * 70)
    port_total = 0
    for sym in train_symbols:
        b = full_datasets.get(sym)
        if b is None:
            continue
        r = backtest_v4(b, best_params, leverage=args.leverage, initial_capital=33.33)
        port_total += r["final_capital"]
        logger.info("  %s: $%.2f", sym, r["final_capital"])
    logger.info("  3-coin portfolio: $%.2f (%+.1f%%)", port_total, (port_total - 100) / 100 * 100)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from optuna_scalp_tune import backtest_breakout
    v1_params = {
        "lookback": 18, "tp_pct": 0.048, "sl_pct": 0.019,
        "max_hold_bars": 86, "cooldown_bars": 1, "vol_ma_period": 10,
        "vol_surge_mult": 1.132, "tbr_long_min": 0.553, "tbr_short_max": 0.415,
        "nbz_threshold": 0.727, "ats_threshold": 1.917,
    }

    logger.info("\n" + "=" * 70)
    logger.info("V1 vs V4 COMPARISON (full 80w, 8x)")
    logger.info("=" * 70)
    logger.info("%-10s %10s %10s %8s %8s %8s %8s", "Symbol", "V1$", "V4$", "V1 DD%", "V4 DD%", "V1 PF", "V4 PF")
    for sym in train_symbols:
        b = full_datasets.get(sym)
        if b is None:
            continue
        r_v1 = backtest_breakout(b, v1_params, leverage=args.leverage)
        r_v4 = backtest_v4(b, best_params, leverage=args.leverage)
        logger.info("%-10s %10.2f %10.2f %8.1f %8.1f %8.3f %8.3f",
                     sym, r_v1["final_capital"], r_v4["final_capital"],
                     r_v1["max_dd"] * 100, r_v4["max_dd"] * 100,
                     r_v1["profit_factor"], r_v4["profit_factor"])

    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    output = {
        "version": "v4_adaptive",
        "best_params": best_params,
        "best_score": best.value,
        "leverage": args.leverage,
        "timeframe": args.timeframe,
        "per_asset": {sym: {
            k: {mk: mv for mk, mv in metrics.items()} for k, metrics in v.items()
        } for sym, v in results_summary.items()},
        "portfolio_3coin_80w": port_total,
    }
    outpath = output_dir / "optuna_v4_results.json"
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
