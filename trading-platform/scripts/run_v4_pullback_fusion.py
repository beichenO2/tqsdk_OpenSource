"""V4 Breakout + Pullback Fusion Strategy — Dual-Mode Entry.

Two entry modes working together:
  Mode 1 (Breakout): Immediate full-position entry when V4 detects a breakout
  Mode 2 (Pullback): After breakout entry, if price retraces to the breakout
    level ± tolerance, add 50% more position (scale-in for better avg price)

Architecture:
  1. V4 Breakout Detection: Donchian channel breakout + volume surge + taker buy
     ratio + net-buy z-score + avg trade size → direction signal
  2. Immediate Entry: Full position on breakout (same as vanilla V4)
  3. Pullback Add-on: While holding, if price retraces to breakout_level ±
     pullback_tolerance_atr * ATR within pullback_max_wait_bars, add 50% notional
  4. Unified SL: Below breakout level for both entries
  5. Wide TP: tp_atr_mult * ATR from (blended) entry
  6. Regime Filter: ADX + vol-of-vol gate (inherited from V4)

Train/Test decoupling: This script takes params as input (from Optuna or manual)
and only runs backtest. No fitting/training happens inside.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import os as _os
import sys as _sys

try:
    from polarisor_port_sdk import submit_task as _sdk_submit, complete_task as _sdk_complete
except ImportError:
    _sdk_submit = _sdk_complete = None

from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from datahub.crypto_loader import CryptoDataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def backtest_v4_pullback(
    bars: pd.DataFrame,
    params: dict,
    leverage: int = 8,
    initial_capital: float = 100.0,
    commission_pct: float = 0.0004,
    slippage_pct: float = 0.0003,
    position_fraction: float = 0.9,
) -> dict:
    """Dual-mode: immediate breakout entry + optional pullback add-on at 50%."""
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

    pullback_tolerance = params.get("pullback_tolerance_atr", 0.5)
    pullback_max_wait = params.get("pullback_max_wait_bars", 12)
    pullback_addon_frac = params.get("pullback_addon_frac", 0.5)
    trail_atr_mult = params.get("trail_atr_mult", 0.0)
    trail_dist_atr = params.get("trail_dist_atr", 0.0)

    cost_per_side = commission_pct + slippage_pct

    capital = initial_capital
    holding = None  # (avg_entry, side, sl, tp, total_notional)
    hold_bars = 0
    cd = 0
    wins = 0
    losses_n = 0
    tw = 0.0
    tl = 0.0
    liqs = 0
    peak_cap = capital
    max_dd = 0.0

    pending_pullback = None  # (side, breakout_level, atr, bars_waited)

    net_buys: deque = deque(maxlen=200)
    avg_ts_buf: deque = deque(maxlen=50)
    atr_buf: deque = deque(maxlen=atr_period + 5)
    ret_buf: deque = deque(maxlen=vol_regime_period + 5)

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
    breakout_entries = 0
    pullback_entries = 0

    def _close_position(exit_price):
        nonlocal capital, holding, cd, wins, losses_n, tw, tl, peak_cap, max_dd, pending_pullback
        side = holding[1]
        if side == "long":
            pnl_pct = (exit_price - holding[0]) / holding[0]
        else:
            pnl_pct = (holding[0] - exit_price) / holding[0]
        notional = holding[4]
        net = notional * pnl_pct - notional * cost_per_side
        capital += net
        capital = max(capital, 0.01)
        if net > 0:
            wins += 1; tw += net
        else:
            losses_n += 1; tl += abs(net)
        holding = None
        pending_pullback = None
        cd = cooldown
        peak_cap = max(peak_cap, capital)
        max_dd = max(max_dd, (peak_cap - capital) / peak_cap)

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

        # --- Position management: exits + pullback add-on ---
        if holding is not None:
            hold_bars += 1
            side = holding[1]

            # Check pullback add-on opportunity (while already holding)
            if pending_pullback is not None:
                pb_side, pb_level, pb_atr, pb_waited = pending_pullback
                pb_waited += 1
                if pb_waited > pullback_max_wait:
                    pending_pullback = None
                elif pb_side == side:
                    tol = pullback_tolerance * pb_atr
                    triggered = False
                    if side == "long" and lows[i] <= pb_level + tol and closes[i] > pb_level - tol:
                        triggered = True
                        addon_entry = max(closes[i], pb_level)
                    elif side == "short" and highs[i] >= pb_level - tol and closes[i] < pb_level + tol:
                        triggered = True
                        addon_entry = min(closes[i], pb_level)

                    if triggered:
                        addon_notional = capital * pullback_addon_frac * position_fraction * leverage
                        addon_cost = addon_notional * cost_per_side
                        capital -= addon_cost
                        old_entry, _, old_sl, _, old_notional = holding
                        new_total = old_notional + addon_notional
                        blended_entry = (old_entry * old_notional + addon_entry * addon_notional) / new_total
                        if side == "long":
                            new_sl = pb_level - sl_atr_mult * pb_atr
                            new_tp = blended_entry + tp_atr_mult * pb_atr
                        else:
                            new_sl = pb_level + sl_atr_mult * pb_atr
                            new_tp = blended_entry - tp_atr_mult * pb_atr
                        holding = (blended_entry, side, new_sl, new_tp, new_total)
                        pullback_entries += 1
                        pending_pullback = None
                    else:
                        pending_pullback = (pb_side, pb_level, pb_atr, pb_waited)

            # Trailing stop
            if side == "long":
                if trail_atr_mult > 0 and current_atr > 0:
                    trail_peak = max(trail_peak, highs[i])
                    activate_dist = holding[0] + trail_atr_mult * current_atr
                    if trail_peak >= activate_dist:
                        trail_sl = trail_peak - trail_dist_atr * current_atr
                        if lows[i] <= trail_sl:
                            _close_position(trail_sl)
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
                            _close_position(trail_sl)
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
                _close_position(exit_p)
                continue

            lev_pnl = ((closes[i] - holding[0]) / holding[0] if side == "long"
                       else (holding[0] - closes[i]) / holding[0])
            if lev_pnl * leverage * position_fraction <= -0.9:
                capital *= 0.1
                liqs += 1
                holding = None
                pending_pullback = None
                continue

        if holding is not None:
            continue

        # --- New entry: immediate breakout + set pullback pending ---
        if capital > 1.0 and cd <= 0:
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
                                bo_side = "long" if is_long else "short"
                                bo_level = prev_high if is_long else prev_low
                                entry_p = closes[i]
                                notional = capital * position_fraction * leverage
                                entry_cost = notional * cost_per_side
                                capital -= entry_cost

                                if bo_side == "long":
                                    sl_p = bo_level - sl_atr_mult * current_atr
                                    tp_p = entry_p + tp_atr_mult * current_atr
                                    trail_peak = highs[i]
                                else:
                                    sl_p = bo_level + sl_atr_mult * current_atr
                                    tp_p = entry_p - tp_atr_mult * current_atr
                                    trail_peak = lows[i]

                                holding = (entry_p, bo_side, sl_p, tp_p, notional)
                                hold_bars = 0
                                breakout_entries += 1

                                pending_pullback = (bo_side, bo_level, current_atr, 0)

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
        "breakout_entries": breakout_entries,
        "pullback_entries": pullback_entries,
    }


def main():

    _task_id = None
    if _sdk_submit:
        try:
            _tr = _sdk_submit(task_type="backtest", command="run_v4_pullback_fusion.py", requester="run-v4-pullback-fusion", estimated_duration_sec=1800)
            _task_id = _tr.get("task_id")
        except Exception:
            pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--weeks", type=int, default=80)
    parser.add_argument("--leverage", type=int, default=8)
    parser.add_argument("--params-file", type=str, default=None,
                        help="JSON file with V4 params (default: load from models/optuna_v4_results.json)")
    args = parser.parse_args()

    params_file = args.params_file or str(Path(__file__).parent.parent / "models" / "optuna_v4_results.json")
    with open(params_file) as f:
        v4_data = json.load(f)
    base_params = v4_data["best_params"]

    base_params["pullback_tolerance_atr"] = 0.5
    base_params["pullback_max_wait_bars"] = 12

    loader = CryptoDataLoader()
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    logger.info("V4+Pullback Fusion | %s | %dx leverage | %d weeks", args.timeframe, args.leverage, args.weeks)
    logger.info("Params: %s", json.dumps(base_params, indent=2))

    from optuna_v4_tune import backtest_v4

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

        r_v4 = backtest_v4(b, base_params, leverage=args.leverage)
        r_fusion = backtest_v4_pullback(b, base_params, leverage=args.leverage)
        r_fusion_oos = backtest_v4_pullback(test_b, base_params, leverage=args.leverage)

        logger.info("\n%s:", sym)
        logger.info("  V4 Only:    $%.2f (%+.1f%%) T=%d WR=%.0f%% PF=%.3f DD=%.1f%% Sharpe=%.2f",
                     r_v4["final_capital"], r_v4["total_return"] * 100,
                     r_v4["trades"], r_v4["win_rate"] * 100, r_v4["profit_factor"],
                     r_v4["max_dd"] * 100, r_v4["sharpe"])
        logger.info("  V4+Pullback: $%.2f (%+.1f%%) T=%d(BO=%d PB=%d) WR=%.0f%% PF=%.3f DD=%.1f%% Sharpe=%.2f",
                     r_fusion["final_capital"], r_fusion["total_return"] * 100,
                     r_fusion["trades"], r_fusion.get("breakout_entries", 0), r_fusion.get("pullback_entries", 0),
                     r_fusion["win_rate"] * 100, r_fusion["profit_factor"],
                     r_fusion["max_dd"] * 100, r_fusion["sharpe"])
        logger.info("  OOS Only:   $%.2f (%+.1f%%) T=%d WR=%.0f%% PF=%.3f DD=%.1f%%",
                     r_fusion_oos["final_capital"], r_fusion_oos["total_return"] * 100,
                     r_fusion_oos["trades"], r_fusion_oos["win_rate"] * 100, r_fusion_oos["profit_factor"],
                     r_fusion_oos["max_dd"] * 100)

    results = {
        "strategy": "v4_pullback_fusion",
        "params": base_params,
        "leverage": args.leverage,
        "timeframe": args.timeframe,
    }
    outpath = Path("models") / "v4_pullback_fusion_results.json"
    outpath.parent.mkdir(exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("\nResults saved to %s", outpath)


    if _task_id and _sdk_complete:
        try:
            _sdk_complete(_task_id)
        except Exception:
            pass


if __name__ == "__main__":
    main()
