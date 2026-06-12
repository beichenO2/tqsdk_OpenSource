"""布林均值回归 — 布林带上下轨 + RSI 确认，回归中轨离场。"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

from ..base import BaseStrategy, Signal, SignalType, StrategyConfig
from ..registry import auto_register

DEFAULT_PARAMS = {
    "bb_period": 20,
    "bb_num_std": 2.0,
    "rsi_period": 14,
    "rsi_oversold": 32.0,
    "rsi_overbought": 68.0,
    "min_bars": 25,
}


@auto_register("bollinger_mr")
class BollingerMRStrategy(BaseStrategy):
    """价格触及布林下轨且 RSI 超卖做多；上轨且超买做空；持仓回到中轨附近平仓。"""

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._close: dict[str, deque[float]] = {}
        self._high: dict[str, deque[float]] = {}
        self._low: dict[str, deque[float]] = {}

    def _ensure(self, symbol: str) -> None:
        p = int(self.get_param("bb_period"))
        r = int(self.get_param("rsi_period"))
        mx = max(p, r) + 15
        if symbol not in self._close:
            self._close[symbol] = deque(maxlen=mx)
            self._high[symbol] = deque(maxlen=mx)
            self._low[symbol] = deque(maxlen=mx)

    @staticmethod
    def _mean(values: list[float]) -> float:
        return sum(values) / len(values)

    def _bbands(self, closes: list[float], n: int, k: float) -> tuple[float, float, float] | None:
        if len(closes) < n:
            return None
        w = closes[-n:]
        mid = self._mean(w)
        var = sum((x - mid) ** 2 for x in w) / max(len(w) - 1, 1)
        sd = math.sqrt(max(var, 0.0))
        upper = mid + k * sd
        lower = mid - k * sd
        return lower, mid, upper

    def _rsi(self, closes: list[float], n: int) -> float | None:
        if len(closes) < n + 1:
            return None
        gains = 0.0
        losses = 0.0
        for i in range(-n, 0):
            diff = closes[i] - closes[i - 1]
            if diff >= 0:
                gains += diff
            else:
                losses -= diff
        avg_gain = gains / n
        avg_loss = losses / n
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def _hold(self, symbol: str, price: float | None, reason: str) -> Signal:
        return Signal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            signal_type=SignalType.HOLD,
            strength=0.0,
            price=price,
            reason=reason,
        )

    def generate_signal(self, symbol: str, bar: dict[str, Any]) -> Signal:
        c, h, l = bar.get("close"), bar.get("high"), bar.get("low")
        if c is None or h is None or l is None:
            return self._hold(symbol, None, "incomplete bar")

        self._ensure(symbol)
        fc, fh, fl = float(c), float(h), float(l)
        self._close[symbol].append(fc)
        self._high[symbol].append(fh)
        self._low[symbol].append(fl)

        min_bars = int(self.get_param("min_bars"))
        if len(self._close[symbol]) < min_bars:
            return self._hold(symbol, fc, "warming up")

        closes = list(self._close[symbol])
        n = int(self.get_param("bb_period"))
        k = float(self.get_param("bb_num_std") or 2.0)
        bb = self._bbands(closes, n, k)
        if bb is None:
            return self._hold(symbol, fc, "bbands")
        lower, mid, upper = bb

        rsi_n = int(self.get_param("rsi_period"))
        rsi_val = self._rsi(closes, rsi_n)
        if rsi_val is None:
            return self._hold(symbol, fc, "rsi")

        rsi_os = float(self.get_param("rsi_oversold") or 30.0)
        rsi_ob = float(self.get_param("rsi_overbought") or 70.0)

        pos = self.get_position(symbol)

        # Mean reversion exit at middle band
        if pos is not None:
            if pos.side.value == "buy" and fc >= mid:
                sig = Signal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    signal_type=SignalType.LONG_EXIT,
                    strength=0.8,
                    price=fc,
                    reason=f"布林: 回归中轨 mid={mid:.4f}",
                    metadata={"mid": mid, "upper": upper, "lower": lower, "rsi": rsi_val},
                )
                self.record_signal(sig)
                return sig
            if pos.side.value == "sell" and fc <= mid:
                sig = Signal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    signal_type=SignalType.SHORT_EXIT,
                    strength=0.8,
                    price=fc,
                    reason=f"布林: 回归中轨 mid={mid:.4f}",
                    metadata={"mid": mid, "upper": upper, "lower": lower, "rsi": rsi_val},
                )
                self.record_signal(sig)
                return sig
            return self._hold(symbol, fc, "holding")

        touch_lower = fl <= lower or fc <= lower
        touch_upper = fh >= upper or fc >= upper

        if touch_lower and rsi_val < rsi_os:
            strength = min(0.5 + (rsi_os - rsi_val) / max(rsi_os, 1.0) * 0.5, 1.0)
            sig = Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                signal_type=SignalType.LONG_ENTRY,
                strength=round(strength, 4),
                price=fc,
                reason=f"布林下轨+RSI超卖 rsi={rsi_val:.1f}",
                metadata={"lower": lower, "mid": mid, "upper": upper, "rsi": rsi_val},
            )
            self.record_signal(sig)
            return sig

        if touch_upper and rsi_val > rsi_ob:
            strength = min(0.5 + (rsi_val - rsi_ob) / max(100.0 - rsi_ob, 1.0) * 0.5, 1.0)
            sig = Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                signal_type=SignalType.SHORT_ENTRY,
                strength=round(strength, 4),
                price=fc,
                reason=f"布林上轨+RSI超买 rsi={rsi_val:.1f}",
                metadata={"lower": lower, "mid": mid, "upper": upper, "rsi": rsi_val},
            )
            self.record_signal(sig)
            return sig

        return self._hold(symbol, fc, "no signal")

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        sig = self.generate_signal(symbol, bar)
        if sig.signal_type == SignalType.HOLD:
            return []
        return [sig]

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        out: list[Signal] = []
        for sym in self.config.symbols:
            b = market_data.get(sym)
            if b:
                out.extend(await self.on_bar(sym, b))
        return out
