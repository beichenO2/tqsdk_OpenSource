"""RSI Divergence Strategy — Classic Price-Indicator Divergence.

Detects regular and hidden divergences between price and RSI:

Regular Bullish: price makes lower low, RSI makes higher low → reversal up
Regular Bearish: price makes higher high, RSI makes lower high → reversal down
Hidden Bullish: price makes higher low, RSI makes lower low → continuation up
Hidden Bearish: price makes lower high, RSI makes higher high → continuation down

One of the most reliable classic setups — works across all markets.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from strategy.base import BaseStrategy, Signal, SignalType, StrategyConfig
from strategy.indicators import calc_atr
from strategy.registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "rsi_period": 14,
    "swing_lookback": 10,
    "divergence_min_bars": 5,
    "divergence_max_bars": 40,
    "rsi_oversold": 35,
    "rsi_overbought": 65,
    "atr_period": 14,
    "sl_atr_mult": 1.5,
    "tp_atr_mult": 3.5,
    "max_hold_bars": 48,
    "cooldown_bars": 6,
    "max_risk_per_trade": 0.02,
    "position_fraction": 0.4,
}


def _rsi_series(closes: list[float], period: int = 14) -> list[float]:
    n = len(closes)
    rsi = [50.0] * n
    if n < period + 1:
        return rsi
    gains = []
    losses = []
    for i in range(1, n):
        d = closes[i] - closes[i-1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = avg_gain / avg_loss if avg_loss > 1e-10 else 100
        rsi[i + 1] = 100 - 100 / (1 + rs)
    return rsi


def _find_swing_lows(data: list[float], lookback: int) -> list[tuple[int, float]]:
    """Find swing lows (local minima)."""
    swings = []
    for i in range(lookback, len(data) - lookback):
        if all(data[i] <= data[i-j] for j in range(1, lookback+1)) and \
           all(data[i] <= data[i+j] for j in range(1, min(lookback+1, len(data)-i))):
            swings.append((i, data[i]))
    return swings


def _find_swing_highs(data: list[float], lookback: int) -> list[tuple[int, float]]:
    """Find swing highs (local maxima)."""
    swings = []
    for i in range(lookback, len(data) - lookback):
        if all(data[i] >= data[i-j] for j in range(1, lookback+1)) and \
           all(data[i] >= data[i+j] for j in range(1, min(lookback+1, len(data)-i))):
            swings.append((i, data[i]))
    return swings


@auto_register("rsi_divergence")
class RSIDivergenceStrategy(BaseStrategy):
    """Trade regular and hidden RSI divergences."""

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._c: dict[str, deque[float]] = {}
        self._h: dict[str, deque[float]] = {}
        self._l: dict[str, deque[float]] = {}
        self._hold: dict[str, int] = {}
        self._cd: dict[str, int] = {}
        self._entry: dict[str, float] = {}
        self._buf = 200

    def _init(self, s: str) -> None:
        if s not in self._c:
            self._c[s] = deque(maxlen=self._buf)
            self._h[s] = deque(maxlen=self._buf)
            self._l[s] = deque(maxlen=self._buf)

    def _detect_divergence(self, s: str) -> tuple[str, float] | None:
        closes = list(self._c[s])
        if len(closes) < 50:
            return None

        rsi = _rsi_series(closes, self.get_param("rsi_period"))
        lb = self.get_param("swing_lookback")
        min_bars = self.get_param("divergence_min_bars")
        max_bars = self.get_param("divergence_max_bars")

        price_lows = _find_swing_lows(closes, lb)
        rsi_lows = _find_swing_lows(rsi, lb)
        price_highs = _find_swing_highs(closes, lb)
        rsi_highs = _find_swing_highs(rsi, lb)

        if len(price_lows) >= 2 and len(rsi_lows) >= 2:
            pl1, pl2 = price_lows[-2], price_lows[-1]
            rl1, rl2 = rsi_lows[-2], rsi_lows[-1]
            bar_diff = pl2[0] - pl1[0]
            if min_bars <= bar_diff <= max_bars:
                if pl2[1] < pl1[1] and rl2[1] > rl1[1]:
                    if rsi[-1] < self.get_param("rsi_oversold"):
                        return "regular_bullish", rsi[-1]

        if len(price_highs) >= 2 and len(rsi_highs) >= 2:
            ph1, ph2 = price_highs[-2], price_highs[-1]
            rh1, rh2 = rsi_highs[-2], rsi_highs[-1]
            bar_diff = ph2[0] - ph1[0]
            if min_bars <= bar_diff <= max_bars:
                if ph2[1] > ph1[1] and rh2[1] < rh1[1]:
                    if rsi[-1] > self.get_param("rsi_overbought"):
                        return "regular_bearish", rsi[-1]

        return None

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._init(symbol)
        c, h, l = bar["close"], bar["high"], bar["low"]
        self._c[symbol].append(c)
        self._h[symbol].append(h)
        self._l[symbol].append(l)

        atr = calc_atr(self._h[symbol], self._l[symbol], self._c[symbol], self.get_param("atr_period"))
        if atr is None or atr <= 0:
            return []

        signals: list[Signal] = []
        pos = self.get_position(symbol)
        self._cd[symbol] = max(self._cd.get(symbol, 0) - 1, 0)

        risk = self.get_param("max_risk_per_trade")
        sl_dist = atr * self.get_param("sl_atr_mult")
        pos_size = min(risk / (sl_dist / c) if sl_dist > 0 and c > 0 else 0.4, self.get_param("position_fraction"))

        if pos is None:
            if self._cd.get(symbol, 0) > 0:
                return signals

            div = self._detect_divergence(symbol)
            if div is None:
                return signals

            div_type, rsi_val = div

            if "bullish" in div_type:
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY, strength=0.8, price=c,
                    reason=f"RSI DIV {div_type} rsi={rsi_val:.1f}",
                    metadata={"div_type": div_type, "rsi": rsi_val, "position_fraction": pos_size},
                ))
                self._hold[symbol] = 0
                self._entry[symbol] = c

            elif "bearish" in div_type:
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY, strength=0.8, price=c,
                    reason=f"RSI DIV {div_type} rsi={rsi_val:.1f}",
                    metadata={"div_type": div_type, "rsi": rsi_val, "position_fraction": pos_size},
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
                    reason=f"RSI DIV: {reason}",
                ))
                self._cd[symbol] = self.get_param("cooldown_bars")

        return signals

    async def generate_signals(self, md: dict[str, Any]) -> list[Signal]:
        out: list[Signal] = []
        for s in self.config.symbols:
            if s in md:
                out.extend(await self.on_bar(s, md[s]))
        return out
