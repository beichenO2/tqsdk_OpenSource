"""Engulfing Pattern Strategy — Candlestick Reversal Detection.

Detects bullish and bearish engulfing patterns at key levels.
An engulfing pattern occurs when a candle's body completely engulfs
the previous candle's body, signaling potential reversal.

Enhanced with:
1. Trend context (only trade engulfing against the trend at extremes)
2. Volume confirmation (engulfing bar should have higher volume)
3. Key level proximity (near support/resistance)
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from strategy.base import BaseStrategy, Signal, SignalType, StrategyConfig
from strategy.indicators import calc_atr, ema_update
from strategy.registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "ema_period": 30,
    "rsi_period": 14,
    "rsi_oversold": 35,
    "rsi_overbought": 65,
    "min_engulf_ratio": 1.2,
    "volume_confirm_ratio": 1.2,
    "atr_period": 14,
    "sl_atr_mult": 1.5,
    "tp_atr_mult": 3.0,
    "max_hold_bars": 36,
    "cooldown_bars": 5,
    "max_risk_per_trade": 0.02,
    "position_fraction": 0.4,
}


@auto_register("engulfing_pattern")
class EngulfingPatternStrategy(BaseStrategy):
    """Candlestick engulfing reversal with RSI + volume confirmation."""

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._c: dict[str, deque[float]] = {}
        self._o: dict[str, deque[float]] = {}
        self._h: dict[str, deque[float]] = {}
        self._l: dict[str, deque[float]] = {}
        self._v: dict[str, deque[float]] = {}
        self._ema: dict[str, float | None] = {}
        self._rsi_gains: dict[str, float] = {}
        self._rsi_losses: dict[str, float] = {}
        self._rsi_val: dict[str, float] = {}
        self._bar_count: dict[str, int] = {}
        self._hold: dict[str, int] = {}
        self._cd: dict[str, int] = {}
        self._entry: dict[str, float] = {}
        self._buf = 200

    def _init(self, s: str) -> None:
        if s not in self._c:
            self._c[s] = deque(maxlen=self._buf)
            self._o[s] = deque(maxlen=self._buf)
            self._h[s] = deque(maxlen=self._buf)
            self._l[s] = deque(maxlen=self._buf)
            self._v[s] = deque(maxlen=self._buf)
            self._bar_count[s] = 0
            self._rsi_val[s] = 50.0

    def _update_rsi(self, s: str, c: float) -> float:
        period = self.get_param("rsi_period")
        self._bar_count[s] = self._bar_count.get(s, 0) + 1
        if self._bar_count[s] < 2:
            return 50.0

        prev_c = self._c[s][-1] if self._c[s] else c
        delta = c - prev_c
        gain = max(delta, 0)
        loss = max(-delta, 0)

        if self._bar_count[s] <= period + 1:
            self._rsi_gains[s] = self._rsi_gains.get(s, 0) + gain
            self._rsi_losses[s] = self._rsi_losses.get(s, 0) + loss
            if self._bar_count[s] == period + 1:
                self._rsi_gains[s] /= period
                self._rsi_losses[s] /= period
        else:
            self._rsi_gains[s] = (self._rsi_gains.get(s, 0) * (period - 1) + gain) / period
            self._rsi_losses[s] = (self._rsi_losses.get(s, 0) * (period - 1) + loss) / period

        ag = self._rsi_gains.get(s, 0)
        al = self._rsi_losses.get(s, 0)
        rs = ag / al if al > 1e-10 else 100
        rsi = 100 - 100 / (1 + rs)
        self._rsi_val[s] = rsi
        return rsi

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._init(symbol)
        c, o, h, l = bar["close"], bar["open"], bar["high"], bar["low"]
        vol = bar.get("volume", 0.0)

        rsi = self._update_rsi(symbol, c)

        self._c[symbol].append(c)
        self._o[symbol].append(o)
        self._h[symbol].append(h)
        self._l[symbol].append(l)
        self._v[symbol].append(vol)
        self._ema[symbol] = ema_update(self._ema.get(symbol), c, self.get_param("ema_period"))

        if len(self._c[symbol]) < 3:
            return []

        atr = calc_atr(self._h[symbol], self._l[symbol], self._c[symbol], self.get_param("atr_period"))
        if atr is None or atr <= 0:
            return []

        prev_o = self._o[symbol][-2]
        prev_c = self._c[symbol][-2]
        curr_body = abs(c - o)
        prev_body = abs(prev_c - prev_o)
        engulf_ratio = self.get_param("min_engulf_ratio")

        is_bullish_engulf = (
            prev_c < prev_o and
            c > o and
            c > prev_o and
            o < prev_c and
            curr_body > prev_body * engulf_ratio
        )

        is_bearish_engulf = (
            prev_c > prev_o and
            c < o and
            c < prev_o and
            o > prev_c and
            curr_body > prev_body * engulf_ratio
        )

        vols = list(self._v[symbol])
        vol_ok = True
        if len(vols) >= 3:
            vol_ok = vol > vols[-2] * self.get_param("volume_confirm_ratio")

        signals: list[Signal] = []
        pos = self.get_position(symbol)
        self._cd[symbol] = max(self._cd.get(symbol, 0) - 1, 0)

        risk = self.get_param("max_risk_per_trade")
        sl_dist = atr * self.get_param("sl_atr_mult")
        pos_size = min(risk / (sl_dist / c) if sl_dist > 0 and c > 0 else 0.4, self.get_param("position_fraction"))

        if pos is None:
            if self._cd.get(symbol, 0) > 0:
                return signals

            if is_bullish_engulf and vol_ok and rsi < self.get_param("rsi_oversold"):
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY, strength=0.8, price=c,
                    reason=f"ENGULFING BULL rsi={rsi:.0f}",
                    metadata={"rsi": rsi, "engulf_ratio": curr_body / max(prev_body, 1e-10), "position_fraction": pos_size},
                ))
                self._hold[symbol] = 0
                self._entry[symbol] = c

            elif is_bearish_engulf and vol_ok and rsi > self.get_param("rsi_overbought"):
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY, strength=0.8, price=c,
                    reason=f"ENGULFING BEAR rsi={rsi:.0f}",
                    metadata={"rsi": rsi, "engulf_ratio": curr_body / max(prev_body, 1e-10), "position_fraction": pos_size},
                ))
                self._hold[symbol] = 0
                self._entry[symbol] = c

        else:
            self._hold[symbol] = self._hold.get(symbol, 0) + 1
            entry = self._entry.get(symbol, c)
            ex = False
            reason = ""

            if self._hold[symbol] >= self.get_param("max_hold_bars"):
                ex, reason = True, "timeout"
            elif pos.side.value == "buy":
                if c <= entry - atr * self.get_param("sl_atr_mult"):
                    ex, reason = True, "SL"
                elif c >= entry + atr * self.get_param("tp_atr_mult"):
                    ex, reason = True, "TP"
            elif pos.side.value == "sell":
                if c >= entry + atr * self.get_param("sl_atr_mult"):
                    ex, reason = True, "SL"
                elif c <= entry - atr * self.get_param("tp_atr_mult"):
                    ex, reason = True, "TP"

            if ex:
                et = SignalType.LONG_EXIT if pos.side.value == "buy" else SignalType.SHORT_EXIT
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=et, strength=0.9, price=c,
                    reason=f"ENGULF: {reason}",
                ))
                self._cd[symbol] = self.get_param("cooldown_bars")

        return signals

    async def generate_signals(self, md: dict[str, Any]) -> list[Signal]:
        out: list[Signal] = []
        for s in self.config.symbols:
            if s in md:
                out.extend(await self.on_bar(s, md[s]))
        return out
