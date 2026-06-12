"""Liquidation Cascade Reversal Strategy.

Research (Tigro Blanc 2026) shows cascade detection alone has
p=0.182 alpha (not significant) — but as a regime filter it's powerful.

Strategy: detect likely liquidation cascades via proxy signals, then
trade the reversal after the cascade exhausts itself.

Cascade proxy detection (no exchange liquidation feed needed):
1. Sudden volume spike > 3x average with large price move
2. OI declining sharply (forced position closures)
3. Funding rate extreme (over-leveraged positions)
4. Price wick > 2x ATR (stop hunt / liquidity sweep)

Entry: after cascade indicators fire AND price shows reversal candle
Exit: ATR-based with tight stop (cascade can continue)
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any

from strategy.base import BaseStrategy, Signal, SignalType, StrategyConfig
from strategy.indicators import calc_atr
from strategy.registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "atr_period": 14,
    "volume_spike_mult": 2.5,
    "wick_atr_mult": 1.8,
    "price_move_pct": 0.025,
    "oi_decline_threshold": -0.05,
    "funding_extreme": 0.001,
    "min_cascade_signals": 2,
    "reversal_candle_ratio": 0.6,
    "sl_atr_mult": 1.5,
    "tp_atr_mult": 3.0,
    "max_hold_bars": 24,
    "cooldown_bars": 12,
    "cascade_memory_bars": 3,
}


@auto_register("liquidation_reversal")
class LiquidationReversalStrategy(BaseStrategy):
    """Trade reversals after liquidation cascade exhaustion."""

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._c: dict[str, deque[float]] = {}
        self._h: dict[str, deque[float]] = {}
        self._l: dict[str, deque[float]] = {}
        self._o: dict[str, deque[float]] = {}
        self._v: dict[str, deque[float]] = {}
        self._oi: dict[str, deque[float]] = {}
        self._cascade_detected: dict[str, int] = {}
        self._cascade_direction: dict[str, str] = {}
        self._hold: dict[str, int] = {}
        self._cd: dict[str, int] = {}
        self._entry: dict[str, float] = {}
        self._buf = 200

    def _init(self, s: str) -> None:
        if s not in self._c:
            self._c[s] = deque(maxlen=self._buf)
            self._h[s] = deque(maxlen=self._buf)
            self._l[s] = deque(maxlen=self._buf)
            self._o[s] = deque(maxlen=self._buf)
            self._v[s] = deque(maxlen=self._buf)
            self._oi[s] = deque(maxlen=self._buf)

    def _detect_cascade(self, s: str, bar: dict[str, Any]) -> tuple[bool, str]:
        """Detect liquidation cascade using proxy signals."""
        c, h, l, o = bar["close"], bar["high"], bar["low"], bar["open"]
        vol = bar.get("volume", 0.0)
        funding = bar.get("funding_rate", 0.0)
        oi = bar.get("open_interest", 0.0)

        signals_fired = 0
        direction = "unknown"

        vols = list(self._v[s])
        if len(vols) >= 20:
            avg_vol = sum(vols[-20:]) / 20
            if vol > avg_vol * self.get_param("volume_spike_mult"):
                signals_fired += 1

        atr = calc_atr(self._h[s], self._l[s], self._c[s], self.get_param("atr_period"))
        if atr and atr > 0:
            upper_wick = h - max(o, c)
            lower_wick = min(o, c) - l
            if lower_wick > atr * self.get_param("wick_atr_mult"):
                signals_fired += 1
                direction = "down_cascade"
            elif upper_wick > atr * self.get_param("wick_atr_mult"):
                signals_fired += 1
                direction = "up_cascade"

        closes = list(self._c[s])
        if len(closes) >= 2:
            pct_move = abs(c - closes[-2]) / closes[-2] if closes[-2] > 0 else 0
            if pct_move > self.get_param("price_move_pct"):
                signals_fired += 1
                if c < closes[-2]:
                    direction = "down_cascade"
                else:
                    direction = "up_cascade"

        oi_list = list(self._oi[s])
        if len(oi_list) >= 5 and oi_list[-5] > 0:
            oi_change = (oi_list[-1] - oi_list[-5]) / oi_list[-5]
            if oi_change < self.get_param("oi_decline_threshold"):
                signals_fired += 1

        if abs(funding) > self.get_param("funding_extreme"):
            signals_fired += 1

        min_signals = self.get_param("min_cascade_signals")
        return signals_fired >= min_signals, direction

    def _is_reversal_candle(self, bar: dict[str, Any], cascade_dir: str) -> bool:
        """Check if current bar is a reversal candle after cascade."""
        o, c, h, l = bar["open"], bar["close"], bar["high"], bar["low"]
        body = abs(c - o)
        full_range = h - l
        if full_range <= 0:
            return False

        ratio = body / full_range
        if ratio < self.get_param("reversal_candle_ratio"):
            return False

        if cascade_dir == "down_cascade" and c > o:
            return True
        if cascade_dir == "up_cascade" and c < o:
            return True
        return False

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._init(symbol)
        c, h, l, o = bar["close"], bar["high"], bar["low"], bar["open"]
        vol = bar.get("volume", 0.0)
        oi = bar.get("open_interest", 0.0)

        self._c[symbol].append(c)
        self._h[symbol].append(h)
        self._l[symbol].append(l)
        self._o[symbol].append(o)
        self._v[symbol].append(vol)
        if oi > 0:
            self._oi[symbol].append(oi)

        is_cascade, direction = self._detect_cascade(symbol, bar)
        if is_cascade:
            self._cascade_detected[symbol] = self.get_param("cascade_memory_bars")
            self._cascade_direction[symbol] = direction

        cascade_memory = self._cascade_detected.get(symbol, 0)
        if cascade_memory > 0:
            self._cascade_detected[symbol] = cascade_memory - 1

        signals: list[Signal] = []
        pos = self.get_position(symbol)
        self._cd[symbol] = max(self._cd.get(symbol, 0) - 1, 0)
        atr = calc_atr(self._h[symbol], self._l[symbol], self._c[symbol], self.get_param("atr_period"))

        if pos is None:
            if self._cd.get(symbol, 0) > 0:
                return signals
            if atr is None or atr <= 0:
                return signals

            if cascade_memory > 0 and not is_cascade:
                cascade_dir = self._cascade_direction.get(symbol, "")
                if self._is_reversal_candle(bar, cascade_dir):
                    if cascade_dir == "down_cascade":
                        signals.append(Signal(
                            strategy_id=self.strategy_id, symbol=symbol,
                            signal_type=SignalType.LONG_ENTRY, strength=0.8, price=c,
                            reason=f"LIQ REVERSAL BUY after {cascade_dir}",
                            metadata={"cascade_dir": cascade_dir, "atr": atr},
                        ))
                    elif cascade_dir == "up_cascade":
                        signals.append(Signal(
                            strategy_id=self.strategy_id, symbol=symbol,
                            signal_type=SignalType.SHORT_ENTRY, strength=0.8, price=c,
                            reason=f"LIQ REVERSAL SELL after {cascade_dir}",
                            metadata={"cascade_dir": cascade_dir, "atr": atr},
                        ))

                    if signals:
                        self._hold[symbol] = 0
                        self._entry[symbol] = c
                        self._cascade_detected[symbol] = 0

        else:
            self._hold[symbol] = self._hold.get(symbol, 0) + 1
            entry = self._entry.get(symbol, c)
            ex = False
            reason = ""

            if self._hold[symbol] >= self.get_param("max_hold_bars"):
                ex, reason = True, "timeout"
            elif pos.side.value == "buy":
                if c <= entry - atr * self.get_param("sl_atr_mult"):
                    ex, reason = True, "SL (cascade continued)"
                elif c >= entry + atr * self.get_param("tp_atr_mult"):
                    ex, reason = True, "TP"
            elif pos.side.value == "sell":
                if c >= entry + atr * self.get_param("sl_atr_mult"):
                    ex, reason = True, "SL (cascade continued)"
                elif c <= entry - atr * self.get_param("tp_atr_mult"):
                    ex, reason = True, "TP"

            if ex:
                et = SignalType.LONG_EXIT if pos.side.value == "buy" else SignalType.SHORT_EXIT
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=et, strength=0.9, price=c,
                    reason=f"LIQ: {reason}",
                ))
                self._cd[symbol] = self.get_param("cooldown_bars")

        return signals

    async def generate_signals(self, md: dict[str, Any]) -> list[Signal]:
        out: list[Signal] = []
        for s in self.config.symbols:
            if s in md:
                out.extend(await self.on_bar(s, md[s]))
        return out
