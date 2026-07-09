"""自适应趋势策略 — Kaufman AMA / Hull MA / DEMA 等自适应均线系统。

SOTA 要点:
- Kaufman AMA: 根据市场噪声自动调节平滑系数
- Hull MA: 用加权组合消除均线延迟
- 双 DEMA 交叉: 低延迟趋势捕捉
- 支持通过参数切换不同的 MA 类型
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any

from ..base import BaseStrategy, Signal, SignalType, StrategyConfig
from ..indicators import calc_atr
from ..registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "ma_type": "kama",  # "kama", "hull", "dema", "tema"
    "fast_period": 10,
    "slow_period": 30,
    "er_period": 10,    # Kaufman efficiency ratio period
    "fast_sc": 2,       # KAMA fast smoothing constant
    "slow_sc": 30,      # KAMA slow smoothing constant
    "atr_period": 14,
    "trailing_stop_atr_mult": 2.5,
    "trend_strength_min": 0.3,  # 效率比最低阈值
    "max_hold_bars": 200,
}


@auto_register("adaptive_trend")
class AdaptiveTrendStrategy(BaseStrategy):
    """自适应均线趋势系统 — 支持 KAMA/Hull/DEMA/TEMA。"""

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._close_buf: dict[str, deque[float]] = {}
        self._high_buf: dict[str, deque[float]] = {}
        self._low_buf: dict[str, deque[float]] = {}
        self._kama: dict[str, float] = {}
        self._prev_ma_fast: dict[str, float | None] = {}
        self._prev_ma_slow: dict[str, float | None] = {}
        self._peak: dict[str, float] = {}
        self._trough: dict[str, float] = {}
        self._bars_in_pos: dict[str, int] = {}

    def _ensure(self, symbol: str) -> None:
        if symbol not in self._close_buf:
            max_len = max(int(self.get_param("slow_period")), int(self.get_param("atr_period"))) * 3 + 20
            self._close_buf[symbol] = deque(maxlen=max_len)
            self._high_buf[symbol] = deque(maxlen=max_len)
            self._low_buf[symbol] = deque(maxlen=max_len)

    @staticmethod
    def _sma(data: list[float], period: int) -> float | None:
        if len(data) < period:
            return None
        return sum(data[-period:]) / period

    @staticmethod
    def _ema(data: list[float], period: int) -> float | None:
        if len(data) < period:
            return None
        k = 2 / (period + 1)
        val = sum(data[:period]) / period
        for x in data[period:]:
            val = x * k + val * (1 - k)
        return val

    def _hull_ma(self, data: list[float], period: int) -> float | None:
        """Hull Moving Average = WMA(2*WMA(n/2) - WMA(n), sqrt(n))."""
        sqrt_p = max(int(math.sqrt(period)), 1)
        if len(data) < period + sqrt_p:
            return None
        half = max(period // 2, 1)
        combo: list[float] = []
        for i in range(sqrt_p):
            end = len(data) - sqrt_p + i + 1
            sub = data[:end]
            wh = self._wma(sub, half)
            wf = self._wma(sub, period)
            if wh is None or wf is None:
                return None
            combo.append(2 * wh - wf)
        return self._wma(combo, sqrt_p)

    @staticmethod
    def _wma(data: list[float], period: int) -> float | None:
        if len(data) < period:
            return None
        window = data[-period:]
        weights = range(1, period + 1)
        return sum(w * v for w, v in zip(weights, window)) / sum(weights)

    @staticmethod
    def _ema_series(data: list[float], period: int) -> list[float] | None:
        """Return the full EMA series for *data*."""
        if len(data) < period:
            return None
        k = 2 / (period + 1)
        val = sum(data[:period]) / period
        result = [val]
        for x in data[period:]:
            val = x * k + val * (1 - k)
            result.append(val)
        return result

    def _dema(self, data: list[float], period: int) -> float | None:
        """DEMA = 2*EMA(data) - EMA(EMA(data))."""
        ema1_series = self._ema_series(data, period)
        if ema1_series is None or len(ema1_series) < period:
            return None
        ema2 = self._ema(ema1_series, period)
        if ema2 is None:
            return None
        return 2 * ema1_series[-1] - ema2

    def _tema(self, data: list[float], period: int) -> float | None:
        """TEMA = 3*EMA1 - 3*EMA2 + EMA3."""
        s1 = self._ema_series(data, period)
        if s1 is None or len(s1) < period:
            return None
        s2 = self._ema_series(s1, period)
        if s2 is None or len(s2) < period:
            return None
        s3 = self._ema_series(s2, period)
        if s3 is None:
            return None
        return 3 * s1[-1] - 3 * s2[-1] + s3[-1]

    def _kama_value(self, symbol: str, er_period: int | None = None) -> float | None:
        """Kaufman Adaptive Moving Average, keyed by (symbol, er_period)."""
        buf = list(self._close_buf[symbol])
        if er_period is None:
            er_period = int(self.get_param("er_period"))
        if len(buf) < er_period + 1:
            return None

        direction = abs(buf[-1] - buf[-er_period - 1])
        volatility = sum(abs(buf[i] - buf[i - 1]) for i in range(-er_period, 0))
        er = direction / volatility if volatility > 0 else 0

        fast_sc = 2 / (int(self.get_param("fast_sc")) + 1)
        slow_sc = 2 / (int(self.get_param("slow_sc")) + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

        key = (symbol, er_period)
        prev = self._kama.get(key)
        if prev is None:
            self._kama[key] = buf[-1]
            return buf[-1]
        new_kama = prev + sc * (buf[-1] - prev)
        self._kama[key] = new_kama
        return new_kama

    def _get_ma(self, symbol: str, period: int) -> float | None:
        ma_type = self.get_param("ma_type")
        buf = list(self._close_buf[symbol])
        if ma_type == "kama":
            return self._kama_value(symbol, er_period=period)
        elif ma_type == "hull":
            return self._hull_ma(buf, period)
        elif ma_type == "dema":
            return self._dema(buf, period)
        elif ma_type == "tema":
            return self._tema(buf, period)
        return self._sma(buf, period)

    def _calc_atr(self, symbol: str) -> float | None:
        return calc_atr(
            self._high_buf[symbol], self._low_buf[symbol],
            self._close_buf[symbol], int(self.get_param("atr_period")),
        )

    def _efficiency_ratio(self, symbol: str) -> float:
        er_period = int(self.get_param("er_period"))
        buf = list(self._close_buf[symbol])
        if len(buf) < er_period + 1:
            return 0.0
        direction = abs(buf[-1] - buf[-er_period - 1])
        vol = sum(abs(buf[i] - buf[i - 1]) for i in range(-er_period, 0))
        return direction / vol if vol > 0 else 0.0

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._ensure(symbol)
        close = float(bar["close"])
        high = float(bar["high"])
        low = float(bar["low"])
        self._close_buf[symbol].append(close)
        self._high_buf[symbol].append(high)
        self._low_buf[symbol].append(low)

        fast_p = int(self.get_param("fast_period"))
        slow_p = int(self.get_param("slow_period"))
        ma_fast = self._get_ma(symbol, fast_p)
        ma_slow = self._get_ma(symbol, slow_p)
        atr = self._calc_atr(symbol)

        if ma_fast is None or ma_slow is None or atr is None:
            return []

        er = self._efficiency_ratio(symbol)
        min_er = float(self.get_param("trend_strength_min"))

        signals: list[Signal] = []
        pos = self.get_position(symbol)
        prev_fast = self._prev_ma_fast.get(symbol)
        prev_slow = self._prev_ma_slow.get(symbol)
        self._prev_ma_fast[symbol] = ma_fast
        self._prev_ma_slow[symbol] = ma_slow

        golden_cross = prev_fast is not None and prev_slow is not None and prev_fast <= prev_slow and ma_fast > ma_slow
        death_cross = prev_fast is not None and prev_slow is not None and prev_fast >= prev_slow and ma_fast < ma_slow

        if pos is None and er >= min_er:
            if golden_cross:
                strength = min(er, 1.0)
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY, strength=round(strength, 4), price=close,
                    reason=f"AMA金叉 er={er:.3f} fast={ma_fast:.2f} slow={ma_slow:.2f}",
                ))
                self._peak[symbol] = close
            elif death_cross:
                strength = min(er, 1.0)
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY, strength=round(strength, 4), price=close,
                    reason=f"AMA死叉 er={er:.3f} fast={ma_fast:.2f} slow={ma_slow:.2f}",
                ))
                self._trough[symbol] = close

        elif pos is not None:
            self._bars_in_pos[symbol] = self._bars_in_pos.get(symbol, 0) + 1
            stop_mult = float(self.get_param("trailing_stop_atr_mult"))
            max_hold = int(self.get_param("max_hold_bars"))

            if self._bars_in_pos.get(symbol, 0) >= max_hold:
                exit_t = SignalType.LONG_EXIT if pos.side.value == "buy" else SignalType.SHORT_EXIT
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=exit_t, strength=0.6, price=close, reason=f"超时{max_hold}",
                ))
                self._bars_in_pos[symbol] = 0
            elif pos.side.value == "buy":
                self._peak[symbol] = max(self._peak.get(symbol, close), close)
                trail = self._peak[symbol] - atr * stop_mult
                if close < trail or death_cross:
                    signals.append(Signal(
                        strategy_id=self.strategy_id, symbol=symbol,
                        signal_type=SignalType.LONG_EXIT, strength=0.85, price=close,
                        reason=f"AMA平多 trail={trail:.2f}",
                    ))
                    self._bars_in_pos[symbol] = 0
            elif pos.side.value == "sell":
                self._trough[symbol] = min(self._trough.get(symbol, close), close)
                trail = self._trough[symbol] + atr * stop_mult
                if close > trail or golden_cross:
                    signals.append(Signal(
                        strategy_id=self.strategy_id, symbol=symbol,
                        signal_type=SignalType.SHORT_EXIT, strength=0.85, price=close,
                        reason=f"AMA平空 trail={trail:.2f}",
                    ))
                    self._bars_in_pos[symbol] = 0
        else:
            self._bars_in_pos[symbol] = 0

        for s in signals:
            self.record_signal(s)
        return signals

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        all_sigs: list[Signal] = []
        for symbol in self.config.symbols:
            bar = market_data.get(symbol)
            if bar:
                all_sigs.extend(await self.on_bar(symbol, bar))
        return all_sigs
