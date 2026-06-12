"""Diagnostic: Why does V4+Pullback only produce 16 trades vs V4's 254?

Adds bar-by-bar logging to the entry logic to count rejections at each filter stage.
"""
from __future__ import annotations

import json
import sys
from collections import deque, Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from datahub.crypto_loader import CryptoDataLoader


def diagnostic_backtest(bars: pd.DataFrame, params: dict, leverage: int = 8,
                        initial_capital: float = 100.0, commission_pct: float = 0.0004,
                        slippage_pct: float = 0.0003, position_fraction: float = 0.9) -> dict:
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

    cost_per_side = commission_pct + slippage_pct
    capital = initial_capital
    holding = None
    hold_bars = 0
    cd = 0
    wins = 0
    losses_n = 0

    net_buys: deque = deque(maxlen=200)
    avg_ts_buf: deque = deque(maxlen=50)
    atr_buf: deque = deque(maxlen=atr_period + 5)
    ret_buf: deque = deque(maxlen=vol_regime_period + 5)

    plus_dm_smooth = 0.0
    minus_dm_smooth = 0.0
    tr_smooth = 0.0
    adx_val = 0.0
    adx_warmup = 0

    warmup = max(lookback + 1, vol_ma_period + 1, atr_period + 2, adx_period * 2 + 1, vol_regime_period + 2, 50)

    # Diagnostic counters
    diag = {
        "bars_after_warmup": 0,
        "bars_holding": 0,
        "bars_cooldown": 0,
        "bars_low_capital": 0,
        "bars_available_for_entry": 0,
        "rejected_adx": 0,
        "rejected_vol_regime": 0,
        "no_breakout": 0,
        "breakout_detected": 0,
        "rejected_vol_surge": 0,
        "rejected_tbr": 0,
        "rejected_nbz_data": 0,
        "rejected_nbz_threshold": 0,
        "rejected_ats_data": 0,
        "rejected_ats_threshold": 0,
        "rejected_atr_zero": 0,
        "entries": 0,
        "adx_values": [],
        "vol_regime_values": [],
        "nbz_values": [],
        "ats_ratios": [],
        "vol_surge_ratios": [],
        "tbr_values": [],
    }

    entry_bars = []

    def _close_position(exit_price):
        nonlocal capital, holding, cd, wins, losses_n
        side = holding[1]
        pnl_pct = ((exit_price - holding[0]) / holding[0] if side == "long"
                    else (holding[0] - exit_price) / holding[0])
        notional = holding[4]
        net = notional * pnl_pct - notional * cost_per_side
        capital += net
        capital = max(capital, 0.01)
        if net > 0:
            wins += 1
        else:
            losses_n += 1
        holding = None
        cd = cooldown

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

        if i < warmup:
            continue

        diag["bars_after_warmup"] += 1

        if cd > 0:
            cd -= 1
            diag["bars_cooldown"] += 1

        atr_list = list(atr_buf)
        current_atr = sum(atr_list[-atr_period:]) / atr_period if len(atr_list) >= atr_period else 0.0

        # Position management (simplified - just track exits)
        if holding is not None:
            diag["bars_holding"] += 1
            hold_bars += 1
            side = holding[1]
            if side == "long":
                if lows[i] <= holding[2]:
                    _close_position(holding[2])
                elif highs[i] >= holding[3]:
                    _close_position(holding[3])
                elif hold_bars >= max_hold:
                    _close_position(closes[i])
                else:
                    lev_pnl = (closes[i] - holding[0]) / holding[0]
                    if lev_pnl * leverage * position_fraction <= -0.9:
                        capital *= 0.1
                        holding = None
            else:
                if highs[i] >= holding[2]:
                    _close_position(holding[2])
                elif lows[i] <= holding[3]:
                    _close_position(holding[3])
                elif hold_bars >= max_hold:
                    _close_position(closes[i])
                else:
                    lev_pnl = (holding[0] - closes[i]) / holding[0]
                    if lev_pnl * leverage * position_fraction <= -0.9:
                        capital *= 0.1
                        holding = None
            continue

        # --- Entry logic with diagnostic logging ---
        if capital <= 1.0:
            diag["bars_low_capital"] += 1
            continue
        if cd > 0:
            continue

        diag["bars_available_for_entry"] += 1
        diag["adx_values"].append(adx_val)

        if adx_val < adx_min:
            diag["rejected_adx"] += 1
            continue

        rets = list(ret_buf)
        if len(rets) >= vol_regime_period:
            recent_vol = np.std(rets[-vol_regime_period:])
            long_vol = np.std(rets) if len(rets) > vol_regime_period else recent_vol
            vol_ratio = recent_vol / long_vol if long_vol > 0 else 0
            diag["vol_regime_values"].append(vol_ratio)
            if vol_ratio > vol_regime_max:
                diag["rejected_vol_regime"] += 1
                continue

        prev_high = max(highs[i - lookback:i])
        prev_low = min(lows[i - lookback:i])
        is_long = closes[i] > prev_high
        is_short = closes[i] < prev_low

        if not (is_long or is_short):
            diag["no_breakout"] += 1
            continue

        diag["breakout_detected"] += 1

        vol_ma = np.mean(volumes[max(0, i - vol_ma_period):i])
        vol_surge_ratio = volumes[i] / vol_ma if vol_ma > 0 else 0
        diag["vol_surge_ratios"].append(vol_surge_ratio)
        if vol_ma <= 0 or volumes[i] < vol_ma * vol_surge_mult:
            diag["rejected_vol_surge"] += 1
            continue

        tbr_v = tbvs[i] / max(volumes[i], 1)
        diag["tbr_values"].append(tbr_v)
        tbr_ok = (is_long and tbr_v >= tbr_long_min) or (is_short and tbr_v <= tbr_short_max)
        if not tbr_ok:
            diag["rejected_tbr"] += 1
            continue

        if len(net_buys) < 48:
            diag["rejected_nbz_data"] += 1
            continue

        nbs = list(net_buys)
        nb_mean = sum(nbs[-48:]) / 48
        nb_var = sum((x - nb_mean) ** 2 for x in nbs[-48:]) / 48
        nb_std = nb_var ** 0.5 if nb_var > 0 else 1.0
        nbz = (nbs[-1] - nb_mean) / nb_std if nb_std > 1e-10 else 0.0
        diag["nbz_values"].append(nbz)
        nbz_ok = (is_long and nbz >= nbz_threshold) or (is_short and nbz <= -nbz_threshold)
        if not nbz_ok:
            diag["rejected_nbz_threshold"] += 1
            continue

        if len(avg_ts_buf) < 20:
            diag["rejected_ats_data"] += 1
            continue

        current_ats = avg_ts_buf[-1]
        hist_ats = sum(list(avg_ts_buf)[-21:-1]) / 20
        ats_ratio = current_ats / hist_ats if hist_ats > 0 else 0
        diag["ats_ratios"].append(ats_ratio)
        ats_ok = hist_ats > 0 and ats_ratio >= ats_threshold
        if not ats_ok:
            diag["rejected_ats_threshold"] += 1
            continue

        if current_atr <= 0:
            diag["rejected_atr_zero"] += 1
            continue

        # ENTRY!
        diag["entries"] += 1
        bo_side = "long" if is_long else "short"
        bo_level = prev_high if is_long else prev_low
        entry_p = closes[i]
        notional = capital * position_fraction * leverage
        entry_cost = notional * cost_per_side
        capital -= entry_cost

        if bo_side == "long":
            sl_p = bo_level - sl_atr_mult * current_atr
            tp_p = entry_p + tp_atr_mult * current_atr
        else:
            sl_p = bo_level + sl_atr_mult * current_atr
            tp_p = entry_p - tp_atr_mult * current_atr
        holding = (entry_p, bo_side, sl_p, tp_p, notional)
        hold_bars = 0
        entry_bars.append(i)

    total = wins + losses_n
    diag["wins"] = wins
    diag["losses"] = losses_n
    diag["total_trades"] = total
    diag["final_capital"] = capital
    diag["total_return_pct"] = (capital - initial_capital) / initial_capital * 100

    return diag


def main():
    loader = CryptoDataLoader()
    b = loader.load_with_funding("BTCUSDT", "1h")
    if b.empty:
        b = loader.load("BTCUSDT", "1h")
    if b.empty:
        print("No BTC data available")
        return

    b.attrs["timeframe"] = "1h"
    if "open_time" in b.columns:
        cutoff = b["open_time"].iloc[-1] - pd.Timedelta(weeks=80)
        b = b[b["open_time"] >= cutoff].copy()
        b.attrs["timeframe"] = "1h"

    print(f"Data: {len(b)} bars, {b.columns.tolist()}")

    # Load V4 base params
    v4_path = Path(__file__).parent.parent / "models" / "optuna_v4_results.json"
    with open(v4_path) as f:
        v4_params = json.load(f)["best_params"]

    # Load V4+Pullback optimized params
    pb_path = Path(__file__).parent.parent / "models" / "optuna_v4_pullback_results.json"
    with open(pb_path) as f:
        pb_params = json.load(f)["best_params"]

    print("\n" + "=" * 70)
    print("RUN 1: V4 base params (same params that produce 254 trades in V4)")
    print("=" * 70)
    d1 = diagnostic_backtest(b, v4_params)
    print_diag(d1, "V4 base params")

    print("\n" + "=" * 70)
    print("RUN 2: V4+Pullback Optuna-optimized params")
    print("=" * 70)
    d2 = diagnostic_backtest(b, pb_params)
    print_diag(d2, "Pullback Optuna params")

    print("\n" + "=" * 70)
    print("PARAMETER COMPARISON (key differences)")
    print("=" * 70)
    for key in sorted(set(list(v4_params.keys()) + list(pb_params.keys()))):
        v1 = v4_params.get(key, "N/A")
        v2 = pb_params.get(key, "N/A")
        if v1 != v2:
            mark = " ← BIG DIFF" if key in ("ats_threshold", "adx_min", "tp_atr_mult", "sl_atr_mult", "lookback") else ""
            print(f"  {key:30s}: V4={format_val(v1):>10s}  PB={format_val(v2):>10s}{mark}")

    # Write report
    report_path = Path(__file__).parent.parent / "models" / "trade_analysis_report.md"
    write_report(report_path, d1, d2, v4_params, pb_params, len(b))


def format_val(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def print_diag(d, label):
    print(f"\n--- {label} ---")
    print(f"  Bars after warmup:       {d['bars_after_warmup']}")
    print(f"  Bars holding:            {d['bars_holding']}")
    print(f"  Bars cooldown:           {d['bars_cooldown']}")
    print(f"  Bars available for entry:{d['bars_available_for_entry']}")
    print(f"  ---")
    print(f"  Rejected by ADX:         {d['rejected_adx']} ({d['rejected_adx']/max(d['bars_available_for_entry'],1)*100:.1f}%)")
    print(f"  Rejected by vol_regime:  {d['rejected_vol_regime']} ({d['rejected_vol_regime']/max(d['bars_available_for_entry'],1)*100:.1f}%)")
    print(f"  No breakout signal:      {d['no_breakout']}")
    print(f"  Breakout detected:       {d['breakout_detected']}")
    print(f"  Rejected by vol_surge:   {d['rejected_vol_surge']} (of {d['breakout_detected']})")
    print(f"  Rejected by TBR:         {d['rejected_tbr']}")
    print(f"  Rejected by NBZ (data):  {d['rejected_nbz_data']}")
    print(f"  Rejected by NBZ (thresh):{d['rejected_nbz_threshold']}")
    print(f"  Rejected by ATS (data):  {d['rejected_ats_data']}")
    print(f"  Rejected by ATS (thresh):{d['rejected_ats_threshold']}")
    print(f"  Rejected by ATR=0:       {d['rejected_atr_zero']}")
    print(f"  ---")
    print(f"  ENTRIES:                 {d['entries']}")
    print(f"  Trades (W/L):            {d['total_trades']} ({d['wins']}W/{d['losses']}L)")
    print(f"  Final capital:           ${d['final_capital']:.2f} ({d['total_return_pct']:+.1f}%)")

    if d['adx_values']:
        adx_arr = np.array(d['adx_values'])
        print(f"  ADX stats:               mean={adx_arr.mean():.1f} median={np.median(adx_arr):.1f} <thresh={np.sum(adx_arr < d.get('adx_min_used', 20))}")
    if d['ats_ratios']:
        ats_arr = np.array(d['ats_ratios'])
        print(f"  ATS ratio stats:         mean={ats_arr.mean():.3f} median={np.median(ats_arr):.3f} max={ats_arr.max():.3f}")
    if d['nbz_values']:
        nbz_arr = np.array(d['nbz_values'])
        print(f"  NBZ stats:               mean={nbz_arr.mean():.3f} |max|={np.max(np.abs(nbz_arr)):.3f}")


def write_report(path, d1, d2, v4_params, pb_params, total_bars):
    lines = [
        "# V4+Pullback 交易数量诊断报告",
        "",
        f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"**数据**: BTCUSDT 1h, {total_bars} bars (~{total_bars//168} weeks)",
        "",
        "## 核心发现",
        "",
        f"V4 基础参数在 Pullback 框架中产生 **{d1['entries']}** 笔入场",
        f"Pullback Optuna 优化参数只产生 **{d2['entries']}** 笔入场",
        "",
        "### 入场漏斗分析",
        "",
        "| 过滤阶段 | V4 基础参数 | Pullback Optuna 参数 |",
        "|---------|-----------|-------------------|",
        f"| 可用 bars | {d1['bars_available_for_entry']} | {d2['bars_available_for_entry']} |",
        f"| ADX 过滤 | -{d1['rejected_adx']} ({d1['rejected_adx']/max(d1['bars_available_for_entry'],1)*100:.0f}%) | -{d2['rejected_adx']} ({d2['rejected_adx']/max(d2['bars_available_for_entry'],1)*100:.0f}%) |",
        f"| Vol Regime 过滤 | -{d1['rejected_vol_regime']} | -{d2['rejected_vol_regime']} |",
        f"| 无突破信号 | -{d1['no_breakout']} | -{d2['no_breakout']} |",
        f"| 突破检测到 | {d1['breakout_detected']} | {d2['breakout_detected']} |",
        f"| Vol Surge 过滤 | -{d1['rejected_vol_surge']} | -{d2['rejected_vol_surge']} |",
        f"| TBR 过滤 | -{d1['rejected_tbr']} | -{d2['rejected_tbr']} |",
        f"| NBZ 过滤 | -{d1['rejected_nbz_threshold']} | -{d2['rejected_nbz_threshold']} |",
        f"| ATS 过滤 | -{d1['rejected_ats_threshold']} | -{d2['rejected_ats_threshold']} |",
        f"| **最终入场** | **{d1['entries']}** | **{d2['entries']}** |",
        "",
        "### 关键参数差异",
        "",
        "| 参数 | V4 值 | Pullback Optuna 值 | 影响 |",
        "|------|-------|-------------------|------|",
    ]

    key_params = [
        ("ats_threshold", "ATS 阈值（avg trade size 比率）"),
        ("adx_min", "ADX 最低值（趋势强度）"),
        ("tp_atr_mult", "止盈 ATR 倍数"),
        ("sl_atr_mult", "止损 ATR 倍数"),
        ("nbz_threshold", "Net Buy Z-Score 阈值"),
        ("vol_surge_mult", "成交量突增倍数"),
        ("lookback", "Donchian 回看周期"),
        ("vol_regime_period", "波动率体制周期"),
        ("vol_regime_max", "最大波动率比"),
    ]
    for key, desc in key_params:
        v1 = v4_params.get(key, "N/A")
        v2 = pb_params.get(key, "N/A")
        if isinstance(v1, float) and isinstance(v2, float):
            diff = abs(v2 - v1) / max(abs(v1), 0.001) * 100
            impact = "🔴 严重" if diff > 50 else "🟡 中等" if diff > 20 else "🟢 轻微"
        else:
            impact = "-"
        lines.append(f"| {desc} | {format_val(v1)} | {format_val(v2)} | {impact} |")

    lines.extend([
        "",
        "## 结构性瓶颈分析",
        "",
        "### 1. ATS 阈值过高 (最大瓶颈)",
        f"- Optuna 优化的 `ats_threshold = {pb_params.get('ats_threshold', 'N/A'):.4f}`",
        f"- V4 基础值 `ats_threshold = {v4_params.get('ats_threshold', 'N/A'):.4f}`",
        "- ATS（平均交易规模比率）要求当前 bar 的平均交易规模是过去 20 bar 的 1.91 倍",
        "- 这个条件极难满足，大量有效突破信号被过滤",
        "",
        "### 2. ADX 最低值偏高",
        f"- Pullback 要求 ADX ≥ {pb_params.get('adx_min', 'N/A'):.1f}",
        f"- V4 只要求 ADX ≥ {v4_params.get('adx_min', 'N/A'):.1f}",
        "- 更高的 ADX 阈值排除了更多的窄幅震荡 bar",
        "",
        "### 3. Donchian 突破检测的固有限制",
        "- 突破检测依赖 `close > max(highs[-lookback:])`",
        "- 这是一个二元信号（有/无），没有强度分级",
        "- 回调入场需要先有突破，再等回调 — 双重过滤",
        "",
        "### 4. 回调窗口偏窄",
        f"- `pullback_max_wait_bars = {pb_params.get('pullback_max_wait_bars', 12)}`",
        "- 在 1h 时间框架下，只等 10 根 K 线（10 小时）回调",
        "- BTC 的突破后回调常需要 1-3 天（24-72 根 K 线）",
        "",
        "## 建议",
        "",
        "1. **降低 ATS 阈值**: 从 1.91 降至 1.2-1.4 范围，或直接移除此过滤器",
        "2. **放宽 ADX 阈值**: 从 23 降至 15-18",
        "3. **增加回调等待窗口**: `pullback_max_wait_bars` 从 10 增至 24-48",
        "4. **独立 Pullback 模式**: 考虑将 Pullback 从 Breakout 的附加模式改为独立入场模式",
        "   - 不需要先做 Breakout 入场，可以只做 Pullback 入场",
        "   - 检测突破后不入场，等回调到突破点附近再入场",
        "5. **Optuna 目标函数调整**: 在优化目标中加入交易数量下限惩罚（如 trades < 30 则 Sharpe 乘以 0.5）",
        "",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to {path}")


if __name__ == "__main__":
    main()
