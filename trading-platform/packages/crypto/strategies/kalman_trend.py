"""Kalman Filter Adaptive Trend Strategy.

Uses a Kalman filter to estimate the "true" price level and trend,
filtering out market noise. More adaptive than fixed-period EMAs
because it adjusts its smoothing based on recent prediction errors.

Kalman state: [price_level, price_velocity]
When velocity > 0 → uptrend → long
When velocity < 0 → downtrend → short
Velocity magnitude = trend strength

Research: Adaptive Kalman on crypto achieves Sharpe 1.74, Sortino 3.73.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any

import numpy as np

from strategy.base import BaseStrategy, Signal, SignalType, StrategyConfig
from strategy.indicators import calc_atr
from strategy.registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "process_noise": 0.01,
    "measurement_noise": 0.1,
    "velocity_threshold": 0.0005,
    "atr_period": 14,
    "trail_atr_mult": 2.5,
    "max_hold_bars": 96,
    "cooldown_bars": 4,
    "min_bars_warmup": 30,
    "use_adaptive_noise": True,
    "adaptive_window": 20,
}


class SimpleKalman:
    """1D Kalman filter with position and velocity state."""

    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 0.1) -> None:
        self.x = np.zeros(2)
        self.P = np.eye(2) * 1000
        self.F = np.array([[1, 1], [0, 1]], dtype=float)
        self.H = np.array([[1, 0]], dtype=float)
        self.Q = np.eye(2) * process_noise
        self.R = np.array([[measurement_noise]])
        self._initialized = False
        self._innovations: deque[float] = deque(maxlen=50)

    def update(self, measurement: float, adapt_noise: bool = False) -> tuple[float, float]:
        """Update with new measurement. Returns (filtered_price, velocity)."""
        if not self._initialized:
            self.x[0] = measurement
            self.x[1] = 0.0
            self._initialized = True
            return measurement, 0.0

        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        innovation = measurement - (self.H @ x_pred)[0]
        self._innovations.append(innovation)

        if adapt_noise and len(self._innovations) >= 10:
            innov_var = np.var(list(self._innovations))
            self.R[0, 0] = max(innov_var * 0.5, 0.001)

        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        self.x = x_pred + K.ravel() * innovation
        self.P = (np.eye(2) - K @ self.H) @ P_pred

        return float(self.x[0]), float(self.x[1])


@auto_register("kalman_trend")
class KalmanTrendStrategy(BaseStrategy):
    """Adaptive trend following using Kalman filter velocity."""

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._c: dict[str, deque[float]] = {}
        self._h: dict[str, deque[float]] = {}
        self._l: dict[str, deque[float]] = {}
        self._kf: dict[str, SimpleKalman] = {}
        self._bar_count: dict[str, int] = {}
        self._hold: dict[str, int] = {}
        self._cd: dict[str, int] = {}
        self._peak: dict[str, float] = {}
        self._entry: dict[str, float] = {}
        self._buf = 200

    def _init(self, s: str) -> None:
        if s not in self._c:
            self._c[s] = deque(maxlen=self._buf)
            self._h[s] = deque(maxlen=self._buf)
            self._l[s] = deque(maxlen=self._buf)
            self._kf[s] = SimpleKalman(
                self.get_param("process_noise"),
                self.get_param("measurement_noise"),
            )
            self._bar_count[s] = 0

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._init(symbol)
        c, h, l = bar["close"], bar["high"], bar["low"]
        self._c[symbol].append(c)
        self._h[symbol].append(h)
        self._l[symbol].append(l)
        self._bar_count[symbol] += 1

        adapt = self.get_param("use_adaptive_noise")
        filtered_price, velocity = self._kf[symbol].update(c, adapt)

        if self._bar_count[symbol] < self.get_param("min_bars_warmup"):
            return []

        atr = calc_atr(self._h[symbol], self._l[symbol], self._c[symbol], self.get_param("atr_period"))
        if atr is None or atr <= 0:
            return []

        vel_threshold = self.get_param("velocity_threshold")
        normalized_vel = velocity / c if c > 0 else 0

        signals: list[Signal] = []
        pos = self.get_position(symbol)
        self._cd[symbol] = max(self._cd.get(symbol, 0) - 1, 0)

        if pos is None:
            if self._cd.get(symbol, 0) > 0:
                return signals

            if normalized_vel > vel_threshold:
                strength = min(abs(normalized_vel) / (vel_threshold * 3), 1.0)
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY,
                    strength=round(strength, 3), price=c,
                    reason=f"KALMAN LONG vel={normalized_vel:.5f} filtered={filtered_price:.1f}",
                    metadata={"velocity": velocity, "filtered": filtered_price, "normalized_vel": normalized_vel},
                ))
                self._hold[symbol] = 0
                self._peak[symbol] = h
                self._entry[symbol] = c

            elif normalized_vel < -vel_threshold:
                strength = min(abs(normalized_vel) / (vel_threshold * 3), 1.0)
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY,
                    strength=round(strength, 3), price=c,
                    reason=f"KALMAN SHORT vel={normalized_vel:.5f} filtered={filtered_price:.1f}",
                    metadata={"velocity": velocity, "filtered": filtered_price, "normalized_vel": normalized_vel},
                ))
                self._hold[symbol] = 0
                self._peak[symbol] = l
                self._entry[symbol] = c

        else:
            self._hold[symbol] = self._hold.get(symbol, 0) + 1
            trail = atr * self.get_param("trail_atr_mult")
            ex = False
            reason = ""

            if self._hold[symbol] >= self.get_param("max_hold_bars"):
                ex, reason = True, "timeout"
            elif pos.side.value == "buy":
                self._peak[symbol] = max(self._peak.get(symbol, c), h)
                if l <= self._peak[symbol] - trail:
                    ex, reason = True, "trail SL"
                elif normalized_vel < -vel_threshold * 0.5:
                    ex, reason = True, f"Kalman reversal vel={normalized_vel:.5f}"
            elif pos.side.value == "sell":
                self._peak[symbol] = min(self._peak.get(symbol, c), l)
                if h >= self._peak[symbol] + trail:
                    ex, reason = True, "trail SL"
                elif normalized_vel > vel_threshold * 0.5:
                    ex, reason = True, f"Kalman reversal vel={normalized_vel:.5f}"

            if ex:
                et = SignalType.LONG_EXIT if pos.side.value == "buy" else SignalType.SHORT_EXIT
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=et, strength=0.9, price=c,
                    reason=f"KALMAN: {reason}",
                ))
                self._cd[symbol] = self.get_param("cooldown_bars")

        return signals

    async def generate_signals(self, md: dict[str, Any]) -> list[Signal]:
        out: list[Signal] = []
        for s in self.config.symbols:
            if s in md:
                out.extend(await self.on_bar(s, md[s]))
        return out
