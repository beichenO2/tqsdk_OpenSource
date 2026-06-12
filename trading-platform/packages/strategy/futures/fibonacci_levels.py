"""斐波那契回调策略 — 自动检测波段并在关键回调位交易。

经典 Fibonacci 回调水平：23.6%, 38.2%, 50%, 61.8%, 78.6%
  - 38.2% 和 61.8% 是最重要的回调位
  - 趋势行情中回调到 38.2% → 强趋势继续
  - 回调到 61.8% → 趋势可能反转
  - 超过 78.6% → 趋势大概率结束

自动化关键：波段识别
  1. 检测最近的 swing high 和 swing low
  2. 计算 Fibonacci 回调水平
  3. 价格触及回调位 + 出现反转 K 线 → 入场

Method: Fibonacci 数列 (Leonardo Fibonacci 1202, 经典数学)
应用于技术分析是经典方法。
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np

from ..base import BaseStrategy, Signal, SignalType, StrategyConfig
from ..indicators import calc_atr
from ..registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS: dict[str, Any] = {
    "swing_lookback": 30,
    "swing_threshold_atr": 2.0,
    "fib_levels": [0.236, 0.382, 0.500, 0.618, 0.786],
    "level_tolerance_atr": 0.3,
    "reversal_confirm_bars": 2,
    "atr_period": 14,
    "tp_atr_mult": 2.5,
    "sl_atr_mult": 1.0,
    "max_hold_bars": 35,
    "cooldown_bars": 3,
}


def _find_swing_high(highs: list[float], lookback: int) -> tuple[float, int] | None:
    if len(highs) < lookback:
        return None
    recent = highs[-lookback:]
    max_idx = int(np.argmax(recent))
    if max_idx == 0 or max_idx == len(recent) - 1:
        return None
    return recent[max_idx], len(highs) - lookback + max_idx


def _find_swing_low(lows: list[float], lookback: int) -> tuple[float, int] | None:
    if len(lows) < lookback:
        return None
    recent = lows[-lookback:]
    min_idx = int(np.argmin(recent))
    if min_idx == 0 or min_idx == len(recent) - 1:
        return None
    return recent[min_idx], len(lows) - lookback + min_idx


@auto_register("fibonacci_levels")
class FibonacciLevelsStrategy(BaseStrategy):
    """斐波那契回调策略 — 在关键回调位配合反转信号交易。"""

    def __init__(self, config: StrategyConfig) -> None:
        config = config.model_copy(
            update={"params": {**DEFAULT_PARAMS, **config.params}}
        )
        super().__init__(config)
        self._closes: deque[float] = deque(maxlen=200)
        self._highs: deque[float] = deque(maxlen=200)
        self._lows: deque[float] = deque(maxlen=200)
        self._bar_count = 0
        self._position_side: str | None = None
        self._entry_price = 0.0
        self._hold_bars = 0
        self._cd = 0
        self._reversal_count = 0
        self._reversal_direction = 0

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        h = float(bar.get("high", 0))
        l = float(bar.get("low", 0))
        c = float(bar.get("close", 0))
        o = float(bar.get("open", c))

        self._highs.append(h)
        self._lows.append(l)
        self._closes.append(c)
        self._bar_count += 1

        swing_lb = self.get_param("swing_lookback", 30)
        if self._bar_count < swing_lb + 5:
            return []

        atr = calc_atr(list(self._highs), list(self._lows), list(self._closes),
                       self.get_param("atr_period", 14))
        if atr is None or atr < 1e-10:
            return []

        signals = []

        if self._position_side:
            self._hold_bars += 1
            max_hold = self.get_param("max_hold_bars", 35)
            tp_mult = self.get_param("tp_atr_mult", 2.5)
            sl_mult = self.get_param("sl_atr_mult", 1.0)

            pnl = (c - self._entry_price) / self._entry_price if self._position_side == "long" else (self._entry_price - c) / self._entry_price
            if pnl >= tp_mult * atr / self._entry_price or pnl <= -sl_mult * atr / self._entry_price or self._hold_bars >= max_hold:
                exit_type = SignalType.LONG_EXIT if self._position_side == "long" else SignalType.SHORT_EXIT
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=exit_type, strength=0.8, price=c,
                    reason=f"fib_exit: hold={self._hold_bars}",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = None
                self._hold_bars = 0
                self._cd = self.get_param("cooldown_bars", 3)
                return signals

        if self._cd > 0:
            self._cd -= 1

        if not self._position_side and self._cd <= 0:
            high_list = list(self._highs)
            low_list = list(self._lows)

            sh = _find_swing_high(high_list, swing_lb)
            sl_found = _find_swing_low(low_list, swing_lb)

            if sh and sl_found:
                swing_high, sh_idx = sh
                swing_low, sl_idx = sl_found
                swing_range = swing_high - swing_low
                min_swing = self.get_param("swing_threshold_atr", 2.0) * atr

                if swing_range > min_swing:
                    fib_levels = self.get_param("fib_levels", [0.236, 0.382, 0.500, 0.618, 0.786])
                    tolerance = self.get_param("level_tolerance_atr", 0.3) * atr
                    confirm_bars = self.get_param("reversal_confirm_bars", 2)

                    is_bullish = c > o
                    is_bearish = c < o

                    if sh_idx > sl_idx:
                        for level in fib_levels:
                            fib_price = swing_high - swing_range * level
                            if abs(c - fib_price) < tolerance:
                                if is_bullish:
                                    self._reversal_count += 1
                                    self._reversal_direction = 1
                                else:
                                    self._reversal_count = 0
                                    self._reversal_direction = 0

                                if self._reversal_count >= confirm_bars and self._reversal_direction == 1:
                                    strength = 0.9 if level in (0.382, 0.618) else 0.6
                                    sig = Signal(
                                        strategy_id=self.strategy_id, symbol=symbol,
                                        signal_type=SignalType.LONG_ENTRY,
                                        strength=strength, price=c,
                                        reason=f"fib_buy: {level*100:.1f}% retracement at {fib_price:.1f}",
                                    )
                                    signals.append(sig)
                                    self.record_signal(sig)
                                    self._position_side = "long"
                                    self._entry_price = c
                                    self._hold_bars = 0
                                    self._reversal_count = 0
                                break

                    elif sl_idx > sh_idx:
                        for level in fib_levels:
                            fib_price = swing_low + swing_range * level
                            if abs(c - fib_price) < tolerance:
                                if is_bearish:
                                    self._reversal_count += 1
                                    self._reversal_direction = -1
                                else:
                                    self._reversal_count = 0
                                    self._reversal_direction = 0

                                if self._reversal_count >= confirm_bars and self._reversal_direction == -1:
                                    strength = 0.9 if level in (0.382, 0.618) else 0.6
                                    sig = Signal(
                                        strategy_id=self.strategy_id, symbol=symbol,
                                        signal_type=SignalType.SHORT_ENTRY,
                                        strength=strength, price=c,
                                        reason=f"fib_sell: {level*100:.1f}% retracement at {fib_price:.1f}",
                                    )
                                    signals.append(sig)
                                    self.record_signal(sig)
                                    self._position_side = "short"
                                    self._entry_price = c
                                    self._hold_bars = 0
                                    self._reversal_count = 0
                                break

        return signals

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        return []
