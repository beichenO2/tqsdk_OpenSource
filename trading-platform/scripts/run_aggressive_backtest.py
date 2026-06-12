"""Aggressive leveraged backtester for small-account growth.

Target: $100 initial → extreme growth over 40 weeks.
Supports leverage, full-position sizing, compound growth, multi-timeframe.

Usage:
    python scripts/run_aggressive_backtest.py [--symbol BTCUSDT] [--timeframe 15m] [--weeks 40]
"""

from __future__ import annotations

import asyncio
import argparse
import json
import logging
import math
import sys

import os as _os
import sys as _sys

try:
    from polarisor_port_sdk import submit_task as _sdk_submit, complete_task as _sdk_complete
except ImportError:
    _sdk_submit = _sdk_complete = None

from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from datahub.crypto_loader import CryptoDataLoader
from strategy.base import Signal, SignalType, StrategyConfig, Position, OrderSide
from strategy.indicators import calc_atr, ema_update, rsi, bollinger_bands
from strategy.btc.funding_rate_alpha import FundingRateAlphaStrategy
from strategy.btc.funding_meta_ensemble import FundingMetaEnsembleStrategy
from strategy.btc.cross_sectional_momentum import TimeSeriesMomentumStrategy
from strategy.btc.meta_labeling import MetaLabelingStrategy
from strategy.btc.patch_tst_strategy import PatchTSTStrategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class AggressiveScalpStrategy:
    """Orderflow-enhanced channel breakout with adaptive ATR exits + regime filter.

    V4-Blend: 1000-trial Optuna (200+500+300 rounds), 3-coin walk-forward on 1h 80w:
      BTC 80w: +6316% | Sharpe 5.18 | PF 5.67 | 252 trades | WR 69% | DD 13.1%
      ETH 80w: +320703% | Sharpe 6.33 | PF 13.14 | 242 trades | WR 76% | DD 17.3%
      SOL 80w: +71085% | Sharpe 4.93 | PF 3.90 | 225 trades | WR 73% | DD 19.1%

    OOS (unseen 24w): BTC +265% | ETH +527% (PF 16!) | SOL +145%

    3-coin portfolio 80w: +132,688% | 40w: +903%
    """

    V4_DEFAULTS = {
        "lookback": 21,
        "tp_atr_mult": 2.083,
        "sl_atr_mult": 0.931,
        "max_hold_bars": 68,
        "cooldown_bars": 5,
        "vol_ma_period": 19,
        "vol_surge_mult": 1.259,
        "tbr_long_min": 0.523,
        "tbr_short_max": 0.450,
        "nbz_threshold": 1.857,
        "ats_threshold": 1.036,
        "atr_period": 25,
        "adx_min": 15.19,
        "vol_regime_period": 55,
        "vol_regime_max": 2.111,
        "trail_atr_mult": 0.167,
        "trail_dist_atr": 0.411,
        "use_orderflow": True,
    }

    def __init__(self, params: dict | None = None):
        p = dict(self.V4_DEFAULTS)
        if params:
            p.update(params)
        self.params = p

        buf = 200
        self._closes: deque[float] = deque(maxlen=buf)
        self._highs: deque[float] = deque(maxlen=buf)
        self._lows: deque[float] = deque(maxlen=buf)
        self._volumes: deque[float] = deque(maxlen=buf)
        self._tbvs: deque[float] = deque(maxlen=buf)
        self._quote_vols: deque[float] = deque(maxlen=buf)
        self._trade_counts: deque[float] = deque(maxlen=buf)
        self._net_buys: deque[float] = deque(maxlen=buf)
        self._atr_buf: deque[float] = deque(maxlen=buf)
        self._ret_buf: deque[float] = deque(maxlen=buf)
        self._bar_count = 0
        self._cooldown = 0

        self._plus_dm_smooth = 0.0
        self._minus_dm_smooth = 0.0
        self._tr_smooth = 0.0
        self._adx_val = 0.0
        self._adx_warmup = 0

    def _current_atr(self) -> float:
        ap = self.params["atr_period"]
        buf = list(self._atr_buf)
        if len(buf) < ap:
            return 0.0
        return sum(buf[-ap:]) / ap

    def update(self, bar: dict[str, float]) -> list[dict]:
        c = bar["close"]
        h = bar["high"]
        lo = bar["low"]
        vol = bar.get("volume", 0)
        tbv = bar.get("taker_buy_volume", vol * 0.5)
        qv = bar.get("quote_volume", vol * c)
        tc = bar.get("trades", 1)

        self._closes.append(c)
        self._highs.append(h)
        self._lows.append(lo)
        self._volumes.append(vol)
        self._tbvs.append(tbv)
        self._quote_vols.append(qv)
        self._trade_counts.append(max(tc, 1))

        tbr_ratio = tbv / max(vol, 1)
        net_buy = (2 * tbr_ratio - 1) * vol
        self._net_buys.append(net_buy)
        self._bar_count += 1

        if len(self._closes) >= 2:
            prev_c = list(self._closes)[-2]
            tr_val = max(h - lo, abs(h - prev_c), abs(lo - prev_c))
            self._atr_buf.append(tr_val)
            self._ret_buf.append((c - prev_c) / prev_c if prev_c > 0 else 0.0)

            up_move = h - (list(self._highs)[-2] if len(self._highs) >= 2 else h)
            dn_move = (list(self._lows)[-2] if len(self._lows) >= 2 else lo) - lo
            pdm = max(up_move, 0.0) if up_move > dn_move else 0.0
            mdm = max(dn_move, 0.0) if dn_move > up_move else 0.0
            alpha = 1.0 / 14
            if self._adx_warmup < 14:
                self._plus_dm_smooth += pdm
                self._minus_dm_smooth += mdm
                self._tr_smooth += tr_val
                self._adx_warmup += 1
            else:
                self._tr_smooth = self._tr_smooth * (1 - alpha) + tr_val
                self._plus_dm_smooth = self._plus_dm_smooth * (1 - alpha) + pdm
                self._minus_dm_smooth = self._minus_dm_smooth * (1 - alpha) + mdm
                if self._tr_smooth > 0:
                    di_p = self._plus_dm_smooth / self._tr_smooth
                    di_m = self._minus_dm_smooth / self._tr_smooth
                    di_sum = di_p + di_m
                    if di_sum > 0:
                        dx = abs(di_p - di_m) / di_sum * 100.0
                        self._adx_val = self._adx_val * (1 - alpha) + dx * alpha

        if self._cooldown > 0:
            self._cooldown -= 1

        lb = self.params["lookback"]
        vol_p = self.params["vol_ma_period"]
        atr_p = self.params["atr_period"]
        vrp = self.params["vol_regime_period"]
        min_bars = max(lb + 1, vol_p + 1, atr_p + 2, vrp + 2, 50)
        if len(self._highs) < min_bars or self._cooldown > 0:
            return []

        if self._adx_val < self.params["adx_min"]:
            return []

        rets = list(self._ret_buf)
        if len(rets) >= vrp:
            import numpy as _np
            recent_vol = _np.std(rets[-vrp:])
            long_vol = _np.std(rets) if len(rets) > vrp else recent_vol
            if long_vol > 0 and recent_vol / long_vol > self.params["vol_regime_max"]:
                return []

        highs = list(self._highs)
        lows_l = list(self._lows)
        prev_high = max(highs[-(lb + 1):-1])
        prev_low = min(lows_l[-(lb + 1):-1])

        is_long = c > prev_high
        is_short = c < prev_low
        if not (is_long or is_short):
            return []

        vols = list(self._volumes)
        vol_ma = sum(vols[-vol_p - 1:-1]) / vol_p
        if vol_ma <= 0 or vol < vol_ma * self.params["vol_surge_mult"]:
            return []

        tbr = tbv / max(vol, 1)
        if is_long and tbr < self.params["tbr_long_min"]:
            return []
        if is_short and tbr > self.params["tbr_short_max"]:
            return []

        if self.params["use_orderflow"]:
            nbs = list(self._net_buys)
            if len(nbs) >= 48:
                nb_mean = sum(nbs[-48:]) / 48
                nb_var = sum((x - nb_mean) ** 2 for x in nbs[-48:]) / 48
                nb_std = nb_var ** 0.5 if nb_var > 0 else 1.0
                nbz = (nbs[-1] - nb_mean) / nb_std if nb_std > 1e-10 else 0.0
                if is_long and nbz < self.params["nbz_threshold"]:
                    return []
                if is_short and nbz > -self.params["nbz_threshold"]:
                    return []

            qvs = list(self._quote_vols)
            tcs = list(self._trade_counts)
            if len(qvs) >= 20:
                avg_ts = qvs[-1] / max(tcs[-1], 1)
                avg_ts_hist = sum(qvs[j] / max(tcs[j], 1) for j in range(-21, -1)) / 20
                if avg_ts_hist > 0 and avg_ts / avg_ts_hist < self.params["ats_threshold"]:
                    return []

        atr = self._current_atr()
        if atr <= 0:
            return []

        tp_atr = self.params["tp_atr_mult"]
        sl_atr = self.params["sl_atr_mult"]
        trail_atr = self.params.get("trail_atr_mult", 0)
        trail_dist = self.params.get("trail_dist_atr", 0)

        if is_long:
            return [{
                "side": "long",
                "type": "breakout",
                "sl": c - sl_atr * atr,
                "tp": c + tp_atr * atr,
                "trail_activate": c + trail_atr * atr if trail_atr > 0 else None,
                "trail_dist": trail_dist * atr if trail_dist > 0 else None,
                "reason": f"OFBv4_L({lb})",
            }]
        else:
            return [{
                "side": "short",
                "type": "breakout",
                "sl": c + sl_atr * atr,
                "tp": c - tp_atr * atr,
                "trail_activate": c - trail_atr * atr if trail_atr > 0 else None,
                "trail_dist": trail_dist * atr if trail_dist > 0 else None,
                "reason": f"OFBv4_S({lb})",
            }]

    def set_cooldown(self):
        self._cooldown = self.params["cooldown_bars"]


class LeveragedBacktester:
    """Backtester with leverage, liquidation, compound sizing, trailing stops."""

    def __init__(
        self,
        initial_capital: float = 100.0,
        leverage: int = 20,
        position_fraction: float = 0.9,
        commission_pct: float = 0.0004,
        slippage_pct: float = 0.0003,
        funding_rate_per_8h: float = 0.0001,
    ):
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.position_fraction = position_fraction
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.funding_rate_per_8h = funding_rate_per_8h

    async def run(
        self,
        strategy,
        bars: pd.DataFrame,
    ) -> dict[str, Any]:
        capital = self.initial_capital
        equity_curve = [capital]
        trades: list[dict] = []
        weekly_returns: list[float] = []

        is_async = getattr(strategy, '_is_async', False)

        pos = None
        total_commission = 0.0
        total_funding = 0.0
        liquidations = 0
        peak_equity = capital
        max_dd = 0.0

        bars_since_start = 0
        week_start_capital = capital
        bars_per_week = self._bars_per_week(bars)

        for idx, row in bars.iterrows():
            bar = {
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row.get("volume", 0),
            }
            for col in row.index:
                if col not in ("open", "high", "low", "close", "volume",
                               "open_time", "close_time") and pd.notna(row[col]):
                    bar[col] = row[col]

            close = bar["close"]
            bars_since_start += 1

            if bars_since_start % bars_per_week == 0:
                wk_ret = (capital - week_start_capital) / week_start_capital if week_start_capital > 0 else 0
                weekly_returns.append(wk_ret)
                week_start_capital = capital

            if pos is not None:
                pos["bars_held"] += 1

                funding_bars = self._funding_interval(bars)
                if funding_bars > 0 and bars_since_start % funding_bars == 0:
                    notional = pos["qty"] * close
                    funding_cost = notional * self.funding_rate_per_8h
                    capital -= funding_cost
                    total_funding += funding_cost

                liq_price = self._liquidation_price(pos, capital)
                if pos["side"] == "long" and bar["low"] <= liq_price:
                    capital = 0.0
                    liquidations += 1
                    trades.append({
                        "type": "liquidation", "side": pos["side"],
                        "entry": pos["entry_price"], "exit": liq_price,
                        "pnl": -capital, "capital": 0,
                        "reason": "LIQUIDATED",
                    })
                    pos = None
                    capital = self.initial_capital * 0.01
                    continue
                elif pos["side"] == "short" and bar["high"] >= liq_price:
                    capital = 0.0
                    liquidations += 1
                    trades.append({
                        "type": "liquidation", "side": pos["side"],
                        "entry": pos["entry_price"], "exit": liq_price,
                        "pnl": -capital, "capital": 0,
                        "reason": "LIQUIDATED",
                    })
                    pos = None
                    capital = self.initial_capital * 0.01
                    continue

                exit_price, exit_reason = self._check_exit(pos, bar, strategy)
                if exit_price is not None:
                    pnl = self._calc_pnl(pos, exit_price)
                    commission = pos["qty"] * exit_price * self.commission_pct
                    net_pnl = pnl - commission
                    capital += net_pnl
                    total_commission += commission
                    trades.append({
                        "type": "exit", "side": pos["side"],
                        "entry": pos["entry_price"], "exit": exit_price,
                        "pnl": round(net_pnl, 4), "capital": round(capital, 2),
                        "bars_held": pos["bars_held"],
                        "reason": exit_reason,
                    })
                    pos = None
                    strategy.set_cooldown()

            if is_async:
                signals = await strategy.async_update(bar)
            else:
                signals = strategy.update(bar)

            if pos is None and capital > 1.0:
                entry_sigs = [s for s in signals if s.get("side") in ("long", "short")]
                if entry_sigs:
                    sig = entry_sigs[0]
                    entry_price = close
                    if sig["side"] == "long":
                        entry_price *= (1 + self.slippage_pct)
                    else:
                        entry_price *= (1 - self.slippage_pct)

                    notional = capital * self.position_fraction * self.leverage
                    qty = notional / entry_price
                    commission = notional * self.commission_pct
                    capital -= commission
                    total_commission += commission

                    pos = {
                        "side": sig["side"],
                        "entry_price": entry_price,
                        "qty": qty,
                        "sl": sig["sl"],
                        "tp": sig["tp"],
                        "trail_activate": sig.get("trail_activate"),
                        "trail_dist": sig.get("trail_dist"),
                        "peak": entry_price,
                        "bars_held": 0,
                    }
                    trades.append({
                        "type": "entry", "side": sig["side"],
                        "price": entry_price, "qty": round(qty, 6),
                        "notional": round(notional, 2),
                        "capital": round(capital, 2),
                        "reason": sig["reason"],
                    })

            unrealized = 0.0
            if pos is not None:
                unrealized = self._calc_pnl(pos, close)

            equity = max(capital + unrealized, 0)
            equity_curve.append(equity)
            peak_equity = max(peak_equity, equity)
            dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
            max_dd = max(max_dd, dd)

        if pos is not None:
            final_price = bars.iloc[-1]["close"]
            pnl = self._calc_pnl(pos, final_price)
            commission = pos["qty"] * final_price * self.commission_pct
            capital += pnl - commission

        if week_start_capital > 0:
            wk_ret = (capital - week_start_capital) / week_start_capital
            weekly_returns.append(wk_ret)

        return self._metrics(
            capital, equity_curve, trades, weekly_returns,
            max_dd, total_commission, total_funding, liquidations, bars,
        )

    def _liquidation_price(self, pos: dict, capital: float) -> float:
        notional = pos["qty"] * pos["entry_price"]
        margin = notional / self.leverage
        maint_margin_rate = 0.005
        if pos["side"] == "long":
            return pos["entry_price"] * (1 - (1 / self.leverage) + maint_margin_rate)
        else:
            return pos["entry_price"] * (1 + (1 / self.leverage) - maint_margin_rate)

    def _check_exit(
        self, pos: dict, bar: dict, strategy: AggressiveScalpStrategy
    ) -> tuple[float | None, str]:
        close = bar["close"]
        high = bar["high"]
        low = bar["low"]

        if pos["side"] == "long":
            if low <= pos["sl"]:
                return pos["sl"], "STOP_LOSS"
            if pos["tp"] and high >= pos["tp"]:
                return pos["tp"], "TAKE_PROFIT"

            if pos["trail_activate"] and high >= pos["trail_activate"]:
                pos["peak"] = max(pos["peak"], high)
                trail_sl = pos["peak"] - pos["trail_dist"]
                if low <= trail_sl:
                    return trail_sl, "TRAILING_STOP"
                pos["sl"] = max(pos["sl"], trail_sl)

        else:
            if high >= pos["sl"]:
                return pos["sl"], "STOP_LOSS"
            if pos["tp"] and low <= pos["tp"]:
                return pos["tp"], "TAKE_PROFIT"

            if pos["trail_activate"] and low <= pos["trail_activate"]:
                pos["peak"] = min(pos["peak"], low)
                trail_sl = pos["peak"] + pos["trail_dist"]
                if high >= trail_sl:
                    return trail_sl, "TRAILING_STOP"
                pos["sl"] = min(pos["sl"], trail_sl)

        if pos["bars_held"] >= strategy.params["max_hold_bars"]:
            return close, "MAX_HOLD"

        return None, ""

    def _calc_pnl(self, pos: dict, exit_price: float) -> float:
        if pos["side"] == "long":
            return (exit_price - pos["entry_price"]) * pos["qty"]
        else:
            return (pos["entry_price"] - exit_price) * pos["qty"]

    def _bars_per_week(self, bars: pd.DataFrame) -> int:
        tf_map = {"1m": 10080, "5m": 2016, "15m": 672, "30m": 336,
                  "1h": 168, "4h": 42, "1d": 7}
        tf = bars.attrs.get("timeframe", "15m")
        return tf_map.get(tf, 672)

    def _funding_interval(self, bars: pd.DataFrame) -> int:
        tf_map = {"1m": 480, "5m": 96, "15m": 32, "30m": 16,
                  "1h": 8, "4h": 2, "1d": 0}
        tf = bars.attrs.get("timeframe", "15m")
        return tf_map.get(tf, 32)

    def _metrics(
        self, final_capital, equity_curve, trades, weekly_returns,
        max_dd, total_commission, total_funding, liquidations, bars,
    ) -> dict[str, Any]:
        total_return = (final_capital - self.initial_capital) / self.initial_capital

        exit_trades = [t for t in trades if t["type"] == "exit"]
        winning = [t for t in exit_trades if t["pnl"] > 0]
        losing = [t for t in exit_trades if t["pnl"] <= 0]
        win_rate = len(winning) / len(exit_trades) if exit_trades else 0

        avg_win = np.mean([t["pnl"] for t in winning]) if winning else 0
        avg_loss = abs(np.mean([t["pnl"] for t in losing])) if losing else 1
        total_wins = sum(t["pnl"] for t in winning) if winning else 0
        total_losses = abs(sum(t["pnl"] for t in losing)) if losing else 1
        pf = total_wins / total_losses if total_losses > 0 else float("inf")

        wr = np.array(weekly_returns) if weekly_returns else np.array([0.0])
        avg_weekly = np.mean(wr) * 100
        median_weekly = np.median(wr) * 100
        positive_weeks = (wr > 0).sum()
        negative_weeks = (wr <= 0).sum()

        date_range = ""
        if "open_time" in bars.columns:
            date_range = f"{bars['open_time'].iloc[0].strftime('%Y-%m-%d')} → {bars['open_time'].iloc[-1].strftime('%Y-%m-%d')}"

        n_weeks = len(weekly_returns)

        return {
            "initial_capital": self.initial_capital,
            "final_capital": round(final_capital, 2),
            "total_return_pct": round(total_return * 100, 2),
            "total_return_x": round(final_capital / self.initial_capital, 2),
            "leverage": self.leverage,
            "max_drawdown_pct": round(max_dd * 100, 2),
            "total_trades": len(exit_trades),
            "win_rate_pct": round(win_rate * 100, 1),
            "profit_factor": round(pf, 3) if pf != float("inf") else "inf",
            "avg_win": round(avg_win, 4),
            "avg_loss": round(avg_loss, 4),
            "liquidations": liquidations,
            "total_commission": round(total_commission, 2),
            "total_funding": round(total_funding, 4),
            "weeks": n_weeks,
            "avg_weekly_return_pct": round(avg_weekly, 2),
            "median_weekly_return_pct": round(median_weekly, 2),
            "positive_weeks": int(positive_weeks),
            "negative_weeks": int(negative_weeks),
            "weekly_win_rate_pct": round(positive_weeks / max(n_weeks, 1) * 100, 1),
            "date_range": date_range,
            "weekly_returns": [round(w * 100, 2) for w in weekly_returns],
            "_equity_curve": equity_curve,
        }


class ExistingStrategyAdapter:
    """Wraps an existing async strategy for use with LeveragedBacktester."""

    def __init__(self, strategy, symbol: str = "BTCUSDT"):
        self.strategy = strategy
        self.symbol = symbol
        self.params = {"cooldown_bars": 4, "max_hold_bars": 96}
        self._cooldown = 0
        self._is_async = True

    async def async_update(self, bar: dict[str, float]) -> list[dict]:
        try:
            signals = await self.strategy.on_bar(self.symbol, bar)
        except Exception:
            signals = []

        if self._cooldown > 0:
            self._cooldown -= 1
            return []

        result = []
        for sig in signals:
            close = sig.price or bar["close"]

            highs = list(getattr(self.strategy, '_high', {}).get(self.symbol, []))
            lows = list(getattr(self.strategy, '_low', {}).get(self.symbol, []))
            closes = list(getattr(self.strategy, '_close', {}).get(self.symbol, []))

            atr = calc_atr(highs or [close], lows or [close], closes or [close], 14)
            if atr is None or atr <= 0:
                atr = close * 0.015

            if sig.signal_type == SignalType.LONG_ENTRY:
                result.append({
                    "side": "long",
                    "type": "strategy",
                    "sl": close - atr * 1.5,
                    "tp": close + atr * 4.0,
                    "trail_activate": close + atr * 2.5,
                    "trail_dist": atr * 1.0,
                    "reason": sig.reason or "LONG",
                })
            elif sig.signal_type == SignalType.SHORT_ENTRY:
                result.append({
                    "side": "short",
                    "type": "strategy",
                    "sl": close + atr * 1.5,
                    "tp": close - atr * 4.0,
                    "trail_activate": close - atr * 2.5,
                    "trail_dist": atr * 1.0,
                    "reason": sig.reason or "SHORT",
                })
        return result

    def update(self, bar: dict[str, float]) -> list[dict]:
        raise RuntimeError("Use async_update for async strategies")

    def set_cooldown(self):
        self._cooldown = self.params["cooldown_bars"]


def print_results(results: dict[str, Any]) -> None:
    logger.info("=" * 70)
    logger.info("AGGRESSIVE BACKTEST RESULTS")
    logger.info("=" * 70)
    logger.info("Period: %s (%d weeks)", results["date_range"], results["weeks"])
    logger.info("Initial: $%.2f | Final: $%.2f | Return: %.2f%% (%.1fx)",
                results["initial_capital"], results["final_capital"],
                results["total_return_pct"], results["total_return_x"])
    logger.info("Leverage: %dx | Max Drawdown: %.2f%%",
                results["leverage"], results["max_drawdown_pct"])
    logger.info("-" * 70)
    logger.info("Trades: %d | Win Rate: %.1f%% | PF: %s",
                results["total_trades"], results["win_rate_pct"], results["profit_factor"])
    logger.info("Avg Win: $%.4f | Avg Loss: $%.4f",
                results["avg_win"], results["avg_loss"])
    logger.info("Liquidations: %d", results["liquidations"])
    logger.info("Commission: $%.2f | Funding: $%.4f",
                results["total_commission"], results["total_funding"])
    logger.info("-" * 70)
    logger.info("WEEKLY PERFORMANCE:")
    logger.info("  Avg: %.2f%% | Median: %.2f%%",
                results["avg_weekly_return_pct"], results["median_weekly_return_pct"])
    logger.info("  Positive weeks: %d/%d (%.1f%%)",
                results["positive_weeks"], results["weeks"], results["weekly_win_rate_pct"])

    wr = results["weekly_returns"]
    if wr:
        logger.info("  Best week: %.2f%% | Worst week: %.2f%%", max(wr), min(wr))
        above_100 = sum(1 for w in wr if w >= 100)
        logger.info("  Weeks >= 100%%: %d/%d", above_100, len(wr))

    logger.info("=" * 70)


STRATEGY_MAP = {
    "funding_rate": lambda sym: ExistingStrategyAdapter(
        FundingRateAlphaStrategy(StrategyConfig(name="FR", symbols=[sym])), sym
    ),
    "fund_meta": lambda sym: ExistingStrategyAdapter(
        FundingMetaEnsembleStrategy(StrategyConfig(name="FM", symbols=[sym])), sym
    ),
    "ts_momentum": lambda sym: ExistingStrategyAdapter(
        TimeSeriesMomentumStrategy(StrategyConfig(name="TM", symbols=[sym])), sym
    ),
    "meta_labeling": lambda sym: ExistingStrategyAdapter(
        MetaLabelingStrategy(StrategyConfig(name="ML", symbols=[sym])), sym
    ),
    "ensemble_scalp": lambda sym, **kw: AggressiveScalpStrategy(kw if kw else None),
}


async def main() -> None:

    _task_id = None
    if _sdk_submit:
        try:
            _tr = _sdk_submit(task_type="backtest", command="run_aggressive_backtest.py", requester="run-aggressive-backtest", estimated_duration_sec=1800)
            _task_id = _tr.get("task_id")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="Aggressive leveraged crypto backtester")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--weeks", type=int, default=40)
    parser.add_argument("--capital", type=float, default=100.0)
    parser.add_argument("--leverage", type=int, default=20)
    parser.add_argument("--position-frac", type=float, default=0.9)
    parser.add_argument("--strategy", default=None,
                        help="Strategy name: funding_rate, fund_meta, ts_momentum, "
                             "meta_labeling, ensemble_scalp. If not set, runs all.")
    parser.add_argument("--sl", type=float, default=None, help="Stop-loss %% (e.g. 0.01 = 1%%)")
    parser.add_argument("--tp", type=float, default=None, help="Take-profit %% (e.g. 0.02 = 2%%)")
    args = parser.parse_args()

    loader = CryptoDataLoader()
    full_bars = loader.load_with_funding(args.symbol, args.timeframe)
    if full_bars.empty:
        full_bars = loader.load(args.symbol, args.timeframe)
    if full_bars.empty:
        logger.error("No data for %s %s", args.symbol, args.timeframe)
        return

    full_bars.attrs["timeframe"] = args.timeframe

    if "open_time" in full_bars.columns:
        cutoff = full_bars["open_time"].iloc[-1] - pd.Timedelta(weeks=args.weeks)
        bars = full_bars[full_bars["open_time"] >= cutoff].copy()
        bars.attrs["timeframe"] = args.timeframe
    else:
        bars = full_bars
        bars.attrs["timeframe"] = args.timeframe

    logger.info("Data: %s %s | %d bars | %d weeks",
                args.symbol, args.timeframe, len(bars), args.weeks)

    strat_params = {}
    if hasattr(args, 'sl') and args.sl:
        strat_params["sl_pct"] = args.sl
    if hasattr(args, 'tp') and args.tp:
        strat_params["tp_pct"] = args.tp

    strat_names = [args.strategy] if args.strategy else list(STRATEGY_MAP.keys())
    all_results = {}

    for sname in strat_names:
        if sname not in STRATEGY_MAP:
            logger.warning("Unknown strategy: %s", sname)
            continue

        logger.info("\n--- Running: %s (leverage=%dx) ---", sname, args.leverage)
        factory = STRATEGY_MAP[sname]
        try:
            strategy = factory(args.symbol, **strat_params)
        except TypeError:
            strategy = factory(args.symbol)
        backtester = LeveragedBacktester(
            initial_capital=args.capital,
            leverage=args.leverage,
            position_fraction=args.position_frac,
        )

        results = await backtester.run(strategy, bars)
        print_results(results)
        all_results[sname] = results

    if len(all_results) > 1:
        logger.info("\n" + "=" * 70)
        logger.info("COMPARISON SUMMARY")
        logger.info("=" * 70)
        header = f"{'Strategy':<20} {'Final$':>10} {'Return%':>10} {'WR%':>8} {'PF':>8} {'Trades':>8} {'AvgWk%':>8} {'MaxDD%':>8}"
        logger.info(header)
        logger.info("-" * len(header))
        for name, r in sorted(all_results.items(), key=lambda x: x[1].get("total_return_pct", -9999), reverse=True):
            logger.info(
                "%-20s %10s %10s %8s %8s %8d %8s %8s",
                name,
                f"${r['final_capital']:.2f}",
                f"{r['total_return_pct']:.1f}%",
                f"{r['win_rate_pct']:.1f}%",
                f"{r['profit_factor']}",
                r["total_trades"],
                f"{r['avg_weekly_return_pct']:.1f}%",
                f"{r['max_drawdown_pct']:.1f}%",
            )

    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    for sname, results in all_results.items():
        save_data = {k: v for k, v in results.items() if not k.startswith("_")}
        output_path = output_dir / f"aggressive_{args.symbol}_{args.timeframe}_{args.leverage}x_{sname}.json"
        with open(output_path, "w") as f:
            json.dump(save_data, f, indent=2, default=str)
    logger.info("Results saved to models/")


    if _task_id and _sdk_complete:
        try:
            _sdk_complete(_task_id)
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
