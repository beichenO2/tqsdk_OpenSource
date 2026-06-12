"""Parabolic SAR 策略 — 抛物线止损与反转系统。

Welles Wilder (1978, 经典，与 RSI/ATR 同一作者):
  SAR_{t+1} = SAR_t + AF * (EP - SAR_t)
  AF: 加速因子，从 0.02 开始，每次新高/低 +0.02，上限 0.20
  EP: 极值点（上升趋势的最高价，下降趋势的最低价）

特点：
  - 自带止损反转机制（SAR 被触及 = 平仓 + 反转）
  - 加速因子使得趋势后期止损越来越紧
  - 适合明确趋势行情，震荡行情会频繁反转

Method: Parabolic SAR (Wilder 1978, 经典趋势跟踪/止损工具)
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from ..base import BaseStrategy, Signal, SignalType, StrategyConfig
from ..registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS: dict[str, Any] = {
    "af_start": 0.02,
    "af_increment": 0.02,
    "af_max": 0.20,
    "max_hold_bars": 60,
    "cooldown_bars": 2,
}


@auto_register("parabolic_sar")
class ParabolicSARStrategy(BaseStrategy):
    """Parabolic SAR 策略 — 趋势跟踪 + 自动止损反转。"""

    def __init__(self, config: StrategyConfig) -> None:
        config = config.model_copy(update={"params": {**DEFAULT_PARAMS, **config.params}})
        super().__init__(config)
        self._highs: deque[float] = deque(maxlen=200)
        self._lows: deque[float] = deque(maxlen=200)
        self._closes: deque[float] = deque(maxlen=200)
        self._bar_count = 0
        self._position_side: str | None = None
        self._entry_price = 0.0
        self._hold_bars = 0
        self._cd = 0

        self._sar = 0.0
        self._af = 0.02
        self._ep = 0.0
        self._trend = 0  # 1=up, -1=down
        self._initialized = False

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        h = float(bar.get("high", 0))
        l = float(bar.get("low", 0))
        c = float(bar.get("close", 0))

        self._highs.append(h)
        self._lows.append(l)
        self._closes.append(c)
        self._bar_count += 1

        if self._bar_count < 3:
            return []

        af_start = self.get_param("af_start", 0.02)
        af_inc = self.get_param("af_increment", 0.02)
        af_max = self.get_param("af_max", 0.20)

        if not self._initialized:
            high_list = list(self._highs)
            low_list = list(self._lows)
            if c > list(self._closes)[-2]:
                self._trend = 1
                self._sar = min(low_list[-3:])
                self._ep = h
            else:
                self._trend = -1
                self._sar = max(high_list[-3:])
                self._ep = l
            self._af = af_start
            self._initialized = True
            return []

        prev_trend = self._trend
        new_sar = self._sar + self._af * (self._ep - self._sar)

        if self._trend == 1:
            new_sar = min(new_sar, list(self._lows)[-2], list(self._lows)[-1] if len(self._lows) > 1 else new_sar)
            if l < new_sar:
                self._trend = -1
                new_sar = self._ep
                self._ep = l
                self._af = af_start
            else:
                if h > self._ep:
                    self._ep = h
                    self._af = min(self._af + af_inc, af_max)
        else:
            new_sar = max(new_sar, list(self._highs)[-2], list(self._highs)[-1] if len(self._highs) > 1 else new_sar)
            if h > new_sar:
                self._trend = 1
                new_sar = self._ep
                self._ep = h
                self._af = af_start
            else:
                if l < self._ep:
                    self._ep = l
                    self._af = min(self._af + af_inc, af_max)

        self._sar = new_sar
        trend_flipped = self._trend != prev_trend

        signals = []

        if self._position_side:
            self._hold_bars += 1
            max_hold = self.get_param("max_hold_bars", 60)

            should_exit = self._hold_bars >= max_hold
            if self._position_side == "long" and self._trend == -1:
                should_exit = True
            elif self._position_side == "short" and self._trend == 1:
                should_exit = True

            if should_exit:
                exit_type = SignalType.LONG_EXIT if self._position_side == "long" else SignalType.SHORT_EXIT
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=exit_type, strength=0.8, price=c,
                    reason=f"psar_exit: SAR={self._sar:.1f} trend={self._trend} hold={self._hold_bars}",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = None
                self._hold_bars = 0
                self._cd = self.get_param("cooldown_bars", 2)

        if self._cd > 0:
            self._cd -= 1

        if not self._position_side and self._cd <= 0 and trend_flipped:
            if self._trend == 1:
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY,
                    strength=0.8, price=c,
                    reason=f"psar_buy: SAR={self._sar:.1f} flipped_up AF={self._af:.3f}",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = "long"
                self._entry_price = c
                self._hold_bars = 0

            elif self._trend == -1:
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY,
                    strength=0.8, price=c,
                    reason=f"psar_sell: SAR={self._sar:.1f} flipped_down AF={self._af:.3f}",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = "short"
                self._entry_price = c
                self._hold_bars = 0

        return signals

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        return []
