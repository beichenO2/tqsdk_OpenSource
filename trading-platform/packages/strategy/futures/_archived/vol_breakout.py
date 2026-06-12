"""波动率突破 — ATR 低波动区间 + 放量突破 + ATR 追踪止损。"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from ..base import BaseStrategy, Signal, SignalType, StrategyConfig
from ..indicators import calc_atr
from ..registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "atr_period": 14,
    "atr_regime_period": 20,
    "narrow_atr_ratio": 0.85,
    "range_period": 10,
    "volume_ma_period": 20,
    "volume_rise_ratio": 1.15,
    "trailing_stop_atr_mult": 2.5,
    "min_bars": 25,
}


@auto_register("vol_breakout")
class VolBreakoutStrategy(BaseStrategy):
    """以 ATR 相对其均线判定窄幅；价格突破近端波动区间且成交量放大时入场；ATR 倍数追踪止损。"""

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._close_history: dict[str, deque[float]] = {}
        self._high_history: dict[str, deque[float]] = {}
        self._low_history: dict[str, deque[float]] = {}
        self._volume_history: dict[str, deque[float]] = {}
        self._atr_history: dict[str, deque[float]] = {}
        self._peak_price: dict[str, float] = {}
        self._trough_price: dict[str, float] = {}

    def _ensure_buffers(self, symbol: str) -> None:
        max_len = max(
            int(self.get_param("atr_period")),
            int(self.get_param("atr_regime_period")),
            int(self.get_param("range_period")),
            int(self.get_param("volume_ma_period")),
        ) + 25
        if symbol not in self._close_history:
            self._close_history[symbol] = deque(maxlen=max_len)
            self._high_history[symbol] = deque(maxlen=max_len)
            self._low_history[symbol] = deque(maxlen=max_len)
            self._volume_history[symbol] = deque(maxlen=max_len)
            self._atr_history[symbol] = deque(maxlen=int(self.get_param("atr_regime_period")) + 15)

    def _calc_atr(self, symbol: str) -> float | None:
        return calc_atr(
            self._high_history[symbol],
            self._low_history[symbol],
            self._close_history[symbol],
            int(self.get_param("atr_period")),
        )

    @staticmethod
    def _sma(values: list[float], n: int) -> float | None:
        if n <= 0 or len(values) < n:
            return None
        return sum(values[-n:]) / n

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
        close = bar.get("close")
        high = bar.get("high")
        low = bar.get("low")
        if close is None or high is None or low is None:
            return self._hold(symbol, None, "incomplete bar")

        self._ensure_buffers(symbol)
        fc, fh, fl = float(close), float(high), float(low)
        vol = float(bar.get("volume") or 0.0)

        self._close_history[symbol].append(fc)
        self._high_history[symbol].append(fh)
        self._low_history[symbol].append(fl)
        self._volume_history[symbol].append(vol)

        min_bars = int(self.get_param("min_bars"))
        if len(self._close_history[symbol]) < min_bars:
            return self._hold(symbol, fc, "warming up")

        atr = self._calc_atr(symbol)
        if atr is None or atr <= 0:
            return self._hold(symbol, fc, "no atr")

        self._atr_history[symbol].append(atr)
        atr_rp = int(self.get_param("atr_regime_period"))
        atr_list = list(self._atr_history[symbol])
        atr_mean = self._sma(atr_list, atr_rp)
        if atr_mean is None or atr_mean <= 0:
            return self._hold(symbol, fc, "atr regime warming")

        narrow_ratio = float(self.get_param("narrow_atr_ratio") or 0.85)
        narrow_regime = atr <= atr_mean * narrow_ratio

        rng_p = int(self.get_param("range_period"))
        highs = list(self._high_history[symbol])
        lows = list(self._low_history[symbol])
        if len(highs) < rng_p + 1:
            return self._hold(symbol, fc, "range window")

        range_high = max(highs[-(rng_p + 1) : -1])
        range_low = min(lows[-(rng_p + 1) : -1])

        vol_ma_p = int(self.get_param("volume_ma_period"))
        vols = list(self._volume_history[symbol])
        vol_ma = self._sma(vols, vol_ma_p)
        if vol_ma is None or vol_ma <= 0:
            return self._hold(symbol, fc, "volume ma")

        vol_ratio_req = float(self.get_param("volume_rise_ratio") or 1.0)
        volume_ok = vol >= vol_ma * vol_ratio_req

        pos = self.get_position(symbol)
        stop_mult = float(self.get_param("trailing_stop_atr_mult") or 2.5)

        if pos is not None and atr > 0:
            if pos.side.value == "buy":
                self._peak_price[symbol] = max(self._peak_price.get(symbol, fc), fc)
                trail = self._peak_price[symbol] - atr * stop_mult
                if fc < trail:
                    sig = Signal(
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        signal_type=SignalType.LONG_EXIT,
                        strength=0.9,
                        price=fc,
                        reason=f"ATR trailing stop trail={trail:.4f}",
                        metadata={"trail": trail, "atr": atr},
                    )
                    self.record_signal(sig)
                    self._peak_price.pop(symbol, None)
                    return sig
                return self._hold(symbol, fc, "long holding")

            if pos.side.value == "sell":
                self._trough_price[symbol] = min(self._trough_price.get(symbol, fc), fc)
                trail = self._trough_price[symbol] + atr * stop_mult
                if fc > trail:
                    sig = Signal(
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        signal_type=SignalType.SHORT_EXIT,
                        strength=0.9,
                        price=fc,
                        reason=f"ATR trailing stop trail={trail:.4f}",
                        metadata={"trail": trail, "atr": atr},
                    )
                    self.record_signal(sig)
                    self._trough_price.pop(symbol, None)
                    return sig
                return self._hold(symbol, fc, "short holding")

        if not narrow_regime:
            return self._hold(symbol, fc, "not narrow vol regime")
        if not volume_ok:
            return self._hold(symbol, fc, "volume not confirming")

        prev_close = self._close_history[symbol][-2]
        broke_up = fc > range_high and prev_close <= range_high
        broke_dn = fc < range_low and prev_close >= range_low

        if broke_up:
            strength = min((fc - range_high) / max(range_high, 1e-9) * 15 + 0.45, 1.0)
            sig = Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                signal_type=SignalType.LONG_ENTRY,
                strength=round(strength, 4),
                price=fc,
                reason=f"Vol breakout up range_high={range_high:.4f} ATR={atr:.4f}",
                metadata={
                    "range_high": range_high,
                    "range_low": range_low,
                    "atr": atr,
                    "atr_mean": atr_mean,
                    "vol_ratio": vol / vol_ma,
                },
            )
            self.record_signal(sig)
            self._peak_price[symbol] = fc
            return sig

        if broke_dn:
            strength = min((range_low - fc) / max(abs(range_low), 1e-9) * 15 + 0.45, 1.0)
            sig = Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                signal_type=SignalType.SHORT_ENTRY,
                strength=round(strength, 4),
                price=fc,
                reason=f"Vol breakout down range_low={range_low:.4f} ATR={atr:.4f}",
                metadata={
                    "range_high": range_high,
                    "range_low": range_low,
                    "atr": atr,
                    "atr_mean": atr_mean,
                    "vol_ratio": vol / vol_ma,
                },
            )
            self.record_signal(sig)
            self._trough_price[symbol] = fc
            return sig

        return self._hold(symbol, fc, "no breakout")

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
