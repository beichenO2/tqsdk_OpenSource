"""Hurst Exponent Regime-Switching Strategy.

Uses rolling Hurst exponent to classify the market into:
- Persistent (H > 0.55): trend-following mode
- Anti-persistent (H < 0.45): mean-reversion mode
- Random (0.45 < H < 0.55): no trade

When H > 0.55 → follow the trend (Donchian breakout)
When H < 0.45 → fade the move (Bollinger band reversion)

This is a meta-strategy that switches between two sub-strategies
based on the fractal character of the price series.

Research shows H values change over time, creating windows where
specific strategies have edge — this exploits those windows.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any

import numpy as np

from strategy.base import BaseStrategy, Signal, SignalType, StrategyConfig
from strategy.indicators import calc_atr, ema_update
from strategy.registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "hurst_window": 100,
    "hurst_persistent_threshold": 0.55,
    "hurst_antipersistent_threshold": 0.45,
    "trend_donchian_period": 40,
    "mr_bb_period": 20,
    "mr_bb_std": 2.0,
    "atr_period": 14,
    "trend_trail_mult": 2.5,
    "mr_sl_atr_mult": 1.5,
    "mr_tp_std_mult": 0.5,
    "max_hold_bars": 60,
    "cooldown_bars": 4,
    "ema_filter_period": 50,
}


def _compute_hurst(series: list[float], max_lag: int | None = None) -> float:
    """Rescaled Range (R/S) method for Hurst exponent estimation.

    Returns H in [0, 1]:
      H > 0.5 → persistent (trending)
      H < 0.5 → anti-persistent (mean-reverting)
      H ≈ 0.5 → random walk
    """
    n = len(series)
    if n < 20:
        return 0.5

    if max_lag is None:
        max_lag = min(n // 4, 50)

    lags = range(10, max_lag + 1, 5)
    rs_values = []
    lag_values = []

    for lag in lags:
        n_segments = n // lag
        if n_segments < 1:
            continue

        rs_list = []
        for seg in range(n_segments):
            start = seg * lag
            end = start + lag
            if end > n:
                break

            segment = series[start:end]
            mean_s = sum(segment) / len(segment)
            deviations = [x - mean_s for x in segment]
            cumulative = []
            cum = 0
            for d in deviations:
                cum += d
                cumulative.append(cum)

            r = max(cumulative) - min(cumulative)
            s = math.sqrt(sum(d ** 2 for d in deviations) / len(deviations))

            if s > 1e-10:
                rs_list.append(r / s)

        if rs_list:
            avg_rs = sum(rs_list) / len(rs_list)
            rs_values.append(math.log(avg_rs))
            lag_values.append(math.log(lag))

    if len(rs_values) < 3:
        return 0.5

    n_pts = len(lag_values)
    x_mean = sum(lag_values) / n_pts
    y_mean = sum(rs_values) / n_pts
    cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(lag_values, rs_values))
    var = sum((x - x_mean) ** 2 for x in lag_values)

    if var < 1e-10:
        return 0.5

    hurst = cov / var
    return max(0.0, min(1.0, hurst))


@auto_register("hurst_regime_switch")
class HurstRegimeSwitchStrategy(BaseStrategy):
    """Switches between trend-following and mean-reversion based on Hurst."""

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._c: dict[str, deque[float]] = {}
        self._h: dict[str, deque[float]] = {}
        self._l: dict[str, deque[float]] = {}
        self._ema: dict[str, float | None] = {}
        self._hold: dict[str, int] = {}
        self._cd: dict[str, int] = {}
        self._entry: dict[str, float] = {}
        self._peak: dict[str, float] = {}
        self._mode: dict[str, str] = {}
        self._buf = 300

    def _init(self, s: str) -> None:
        if s not in self._c:
            self._c[s] = deque(maxlen=self._buf)
            self._h[s] = deque(maxlen=self._buf)
            self._l[s] = deque(maxlen=self._buf)

    def _get_hurst(self, s: str) -> float:
        window = self.get_param("hurst_window")
        closes = list(self._c[s])
        if len(closes) < window:
            return 0.5
        returns = [(closes[i] - closes[i-1]) / closes[i-1]
                   for i in range(len(closes) - window, len(closes))
                   if closes[i-1] > 0]
        return _compute_hurst(returns)

    def _donchian(self, s: str, period: int) -> tuple[float, float] | None:
        highs = list(self._h[s])
        lows = list(self._l[s])
        if len(highs) < period:
            return None
        return max(highs[-period:]), min(lows[-period:])

    def _bollinger(self, s: str) -> tuple[float, float, float] | None:
        period = self.get_param("mr_bb_period")
        closes = list(self._c[s])
        if len(closes) < period:
            return None
        window = closes[-period:]
        mid = sum(window) / period
        std = math.sqrt(sum((x - mid) ** 2 for x in window) / period)
        mult = self.get_param("mr_bb_std")
        return mid, mid + std * mult, mid - std * mult

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._init(symbol)
        c, h, l = bar["close"], bar["high"], bar["low"]
        self._c[symbol].append(c)
        self._h[symbol].append(h)
        self._l[symbol].append(l)
        self._ema[symbol] = ema_update(self._ema.get(symbol), c, self.get_param("ema_filter_period"))

        hurst = self._get_hurst(symbol)
        atr = calc_atr(self._h[symbol], self._l[symbol], self._c[symbol], self.get_param("atr_period"))
        if atr is None or atr <= 0:
            return []

        persistent_t = self.get_param("hurst_persistent_threshold")
        antipersist_t = self.get_param("hurst_antipersistent_threshold")

        signals: list[Signal] = []
        pos = self.get_position(symbol)
        self._cd[symbol] = max(self._cd.get(symbol, 0) - 1, 0)

        if pos is None:
            if self._cd.get(symbol, 0) > 0:
                return signals

            if hurst > persistent_t:
                don = self._donchian(symbol, self.get_param("trend_donchian_period"))
                ema_val = self._ema[symbol]
                if don is None or ema_val is None:
                    return signals

                upper, lower = don
                if c > upper and c > ema_val:
                    signals.append(Signal(
                        strategy_id=self.strategy_id, symbol=symbol,
                        signal_type=SignalType.LONG_ENTRY, strength=0.8, price=c,
                        reason=f"HURST TREND LONG H={hurst:.2f}",
                        metadata={"hurst": hurst, "mode": "trend"},
                    ))
                    self._mode[symbol] = "trend"
                    self._hold[symbol] = 0
                    self._entry[symbol] = c
                    self._peak[symbol] = h

                elif c < lower and c < ema_val:
                    signals.append(Signal(
                        strategy_id=self.strategy_id, symbol=symbol,
                        signal_type=SignalType.SHORT_ENTRY, strength=0.8, price=c,
                        reason=f"HURST TREND SHORT H={hurst:.2f}",
                        metadata={"hurst": hurst, "mode": "trend"},
                    ))
                    self._mode[symbol] = "trend"
                    self._hold[symbol] = 0
                    self._entry[symbol] = c
                    self._peak[symbol] = l

            elif hurst < antipersist_t:
                bb = self._bollinger(symbol)
                if bb is None:
                    return signals

                mid, upper, lower = bb
                if c <= lower:
                    signals.append(Signal(
                        strategy_id=self.strategy_id, symbol=symbol,
                        signal_type=SignalType.LONG_ENTRY, strength=0.7, price=c,
                        reason=f"HURST MR LONG H={hurst:.2f} BB lower",
                        metadata={"hurst": hurst, "mode": "mean_reversion", "bb_mid": mid},
                    ))
                    self._mode[symbol] = "mr"
                    self._hold[symbol] = 0
                    self._entry[symbol] = c
                    self._peak[symbol] = mid

                elif c >= upper:
                    signals.append(Signal(
                        strategy_id=self.strategy_id, symbol=symbol,
                        signal_type=SignalType.SHORT_ENTRY, strength=0.7, price=c,
                        reason=f"HURST MR SHORT H={hurst:.2f} BB upper",
                        metadata={"hurst": hurst, "mode": "mean_reversion", "bb_mid": mid},
                    ))
                    self._mode[symbol] = "mr"
                    self._hold[symbol] = 0
                    self._entry[symbol] = c
                    self._peak[symbol] = mid

        else:
            self._hold[symbol] = self._hold.get(symbol, 0) + 1
            entry = self._entry.get(symbol, c)
            mode = self._mode.get(symbol, "trend")
            ex = False
            reason = ""

            if self._hold[symbol] >= self.get_param("max_hold_bars"):
                ex, reason = True, "timeout"
            elif mode == "trend":
                trail = atr * self.get_param("trend_trail_mult")
                if pos.side.value == "buy":
                    self._peak[symbol] = max(self._peak.get(symbol, c), h)
                    if l <= self._peak[symbol] - trail:
                        ex, reason = True, "trail SL"
                else:
                    self._peak[symbol] = min(self._peak.get(symbol, c), l)
                    if h >= self._peak[symbol] + trail:
                        ex, reason = True, "trail SL"
            elif mode == "mr":
                target = self._peak.get(symbol, c)
                if pos.side.value == "buy":
                    if c >= target:
                        ex, reason = True, "TP at BB mid"
                    elif c <= entry - atr * self.get_param("mr_sl_atr_mult"):
                        ex, reason = True, "SL"
                else:
                    if c <= target:
                        ex, reason = True, "TP at BB mid"
                    elif c >= entry + atr * self.get_param("mr_sl_atr_mult"):
                        ex, reason = True, "SL"

            if ex:
                et = SignalType.LONG_EXIT if pos.side.value == "buy" else SignalType.SHORT_EXIT
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=et, strength=0.9, price=c,
                    reason=f"HURST [{mode}]: {reason}",
                ))
                self._cd[symbol] = self.get_param("cooldown_bars")

        return signals

    async def generate_signals(self, md: dict[str, Any]) -> list[Signal]:
        out: list[Signal] = []
        for s in self.config.symbols:
            if s in md:
                out.extend(await self.on_bar(s, md[s]))
        return out
