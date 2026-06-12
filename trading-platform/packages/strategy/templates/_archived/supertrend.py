"""Supertrend 策略 — ATR 动态包络线的趋势跟踪。

SOTA 要点:
- Supertrend = 基于 ATR 的自适应通道，在趋势市中紧贴价格
- 当价格突破上轨做多、突破下轨做空
- 比固定均线系统更快适应波动率变化
- 可搭配 ADX 等趋势强度过滤器
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from ..base import BaseStrategy, Signal, SignalType, StrategyConfig
from ..indicators import calc_atr
from ..registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "atr_period": 10,
    "multiplier": 3.0,
    "adx_period": 14,
    "adx_threshold": 20.0,
    "trailing_stop_atr_mult": 2.0,
    "max_hold_bars": 150,
}


@auto_register("supertrend")
class SupertrendStrategy(BaseStrategy):
    """Supertrend 趋势策略。

    计算 upper_band = HL2 + multiplier * ATR
           lower_band = HL2 - multiplier * ATR
    趋势方向翻转时产生入场信号。
    """

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._high_buf: dict[str, deque[float]] = {}
        self._low_buf: dict[str, deque[float]] = {}
        self._close_buf: dict[str, deque[float]] = {}
        self._prev_upper: dict[str, float] = {}
        self._prev_lower: dict[str, float] = {}
        self._prev_trend: dict[str, int] = {}  # 1=up, -1=down
        self._bars_in_pos: dict[str, int] = {}
        self._peak: dict[str, float] = {}
        self._trough: dict[str, float] = {}

    def _ensure_buffers(self, symbol: str) -> None:
        if symbol not in self._high_buf:
            buf_len = max(int(self.get_param("atr_period")), int(self.get_param("adx_period"))) + 20
            self._high_buf[symbol] = deque(maxlen=buf_len)
            self._low_buf[symbol] = deque(maxlen=buf_len)
            self._close_buf[symbol] = deque(maxlen=buf_len)

    def _calc_atr(self, symbol: str) -> float | None:
        return calc_atr(self._high_buf[symbol], self._low_buf[symbol], self._close_buf[symbol], int(self.get_param("atr_period")))

    def _calc_adx(self, symbol: str) -> float | None:
        """简化 ADX：用 DI+/DI- 的差异比衡量趋势强度。"""
        period = int(self.get_param("adx_period"))
        h = list(self._high_buf[symbol])
        lo = list(self._low_buf[symbol])
        c = list(self._close_buf[symbol])
        if len(h) < period + 2:
            return None
        plus_dm_sum = 0.0
        minus_dm_sum = 0.0
        tr_sum = 0.0
        for i in range(-period, 0):
            up = h[i] - h[i - 1]
            down = lo[i - 1] - lo[i]
            plus_dm_sum += max(up, 0) if up > down else 0
            minus_dm_sum += max(down, 0) if down > up else 0
            tr = max(h[i] - lo[i], abs(h[i] - c[i - 1]), abs(lo[i] - c[i - 1]))
            tr_sum += tr
        if tr_sum == 0:
            return 0.0
        di_plus = plus_dm_sum / tr_sum * 100
        di_minus = minus_dm_sum / tr_sum * 100
        di_sum = di_plus + di_minus
        if di_sum == 0:
            return 0.0
        return abs(di_plus - di_minus) / di_sum * 100

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._ensure_buffers(symbol)
        close = float(bar["close"])
        high = float(bar["high"])
        low = float(bar["low"])
        self._high_buf[symbol].append(high)
        self._low_buf[symbol].append(low)
        self._close_buf[symbol].append(close)

        atr = self._calc_atr(symbol)
        if atr is None or atr <= 0:
            return []

        mult = float(self.get_param("multiplier"))
        hl2 = (high + low) / 2
        basic_upper = hl2 + mult * atr
        basic_lower = hl2 - mult * atr

        prev_upper = self._prev_upper.get(symbol, basic_upper)
        prev_lower = self._prev_lower.get(symbol, basic_lower)
        prev_close = list(self._close_buf[symbol])[-2] if len(self._close_buf[symbol]) >= 2 else close

        # Supertrend 规则：band 只在有利方向调整
        upper = min(basic_upper, prev_upper) if prev_close <= prev_upper else basic_upper
        lower = max(basic_lower, prev_lower) if prev_close >= prev_lower else basic_lower

        prev_trend = self._prev_trend.get(symbol, 1)
        if prev_trend == 1:
            trend = 1 if close >= lower else -1
        else:
            trend = -1 if close <= upper else 1

        self._prev_upper[symbol] = upper
        self._prev_lower[symbol] = lower

        signals: list[Signal] = []
        pos = self.get_position(symbol)

        # ADX 过滤
        adx = self._calc_adx(symbol)
        adx_ok = adx is not None and adx >= float(self.get_param("adx_threshold"))

        trend_changed = trend != prev_trend
        self._prev_trend[symbol] = trend

        if pos is None and trend_changed and adx_ok:
            if trend == 1:
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY, strength=0.75, price=close,
                    reason=f"Supertrend翻多 lower={lower:.2f} ADX={adx:.1f}",
                ))
                self._peak[symbol] = close
            else:
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY, strength=0.75, price=close,
                    reason=f"Supertrend翻空 upper={upper:.2f} ADX={adx:.1f}",
                ))
                self._trough[symbol] = close

        elif pos is not None:
            self._bars_in_pos[symbol] = self._bars_in_pos.get(symbol, 0) + 1
            max_hold = int(self.get_param("max_hold_bars"))
            stop_mult = float(self.get_param("trailing_stop_atr_mult"))

            if self._bars_in_pos.get(symbol, 0) >= max_hold:
                exit_t = SignalType.LONG_EXIT if pos.side.value == "buy" else SignalType.SHORT_EXIT
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=exit_t, strength=0.6, price=close,
                    reason=f"最大持仓{max_hold}bars",
                ))
                self._bars_in_pos[symbol] = 0
            elif pos.side.value == "buy":
                self._peak[symbol] = max(self._peak.get(symbol, close), close)
                trail = self._peak[symbol] - atr * stop_mult
                if close < trail or trend == -1:
                    signals.append(Signal(
                        strategy_id=self.strategy_id, symbol=symbol,
                        signal_type=SignalType.LONG_EXIT, strength=0.85, price=close,
                        reason=f"Supertrend平多 trend={trend} trail={trail:.2f}",
                    ))
                    self._bars_in_pos[symbol] = 0
            elif pos.side.value == "sell":
                self._trough[symbol] = min(self._trough.get(symbol, close), close)
                trail = self._trough[symbol] + atr * stop_mult
                if close > trail or trend == 1:
                    signals.append(Signal(
                        strategy_id=self.strategy_id, symbol=symbol,
                        signal_type=SignalType.SHORT_EXIT, strength=0.85, price=close,
                        reason=f"Supertrend平空 trend={trend} trail={trail:.2f}",
                    ))
                    self._bars_in_pos[symbol] = 0
        else:
            self._bars_in_pos[symbol] = 0

        for s in signals:
            self.record_signal(s)
        return signals

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        all_signals: list[Signal] = []
        for symbol in self.config.symbols:
            bar = market_data.get(symbol)
            if bar:
                all_signals.extend(await self.on_bar(symbol, bar))
        return all_signals
