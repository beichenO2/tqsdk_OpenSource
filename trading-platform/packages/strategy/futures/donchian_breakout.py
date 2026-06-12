"""Donchian Channel 突破策略 — 海龟交易法核心。

Richard Donchian (1960s, 经典) → 海龟实验 (Richard Dennis 1983):
  Upper = max(highs, N)
  Lower = min(lows, N)
  Middle = (Upper + Lower) / 2

海龟规则的日内适配：
  - 20 周期突破入场
  - 10 周期突破退出
  - 2N ATR 止损
  - 单位仓位 = 账户 1% / N

这是所有趋势策略的"祖师爷"，与已有的 cta_trend 不同之处：
  - cta_trend 额外加了 ATR 波动过滤
  - 本策略更纯粹：纯粹的价格突破
  - 增加了动态出场通道（短周期 Donchian）

Method: Donchian Channel (1960s, 经典)
        Turtle Rules (Dennis 1983, 实验验证)
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
    "entry_period": 20,
    "exit_period": 10,
    "atr_period": 14,
    "atr_stop_mult": 2.0,
    "risk_per_trade_pct": 1.0,
    "max_hold_bars": 60,
    "cooldown_bars": 3,
}


@auto_register("donchian_breakout")
class DonchianBreakoutStrategy(BaseStrategy):
    """Donchian 通道突破策略 — 海龟法则日内版。"""

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

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        h = float(bar.get("high", 0))
        l = float(bar.get("low", 0))
        c = float(bar.get("close", 0))

        self._highs.append(h)
        self._lows.append(l)
        self._closes.append(c)
        self._bar_count += 1

        entry_period = self.get_param("entry_period", 20)
        if self._bar_count < entry_period + 3:
            return []

        atr = calc_atr(list(self._highs), list(self._lows), list(self._closes),
                       self.get_param("atr_period", 14))
        if atr is None or atr < 1e-10:
            return []

        high_list = list(self._highs)
        low_list = list(self._lows)

        entry_high = max(high_list[-entry_period - 1:-1])
        entry_low = min(low_list[-entry_period - 1:-1])

        exit_period = self.get_param("exit_period", 10)
        exit_high = max(high_list[-exit_period - 1:-1]) if len(high_list) > exit_period else entry_high
        exit_low = min(low_list[-exit_period - 1:-1]) if len(low_list) > exit_period else entry_low

        signals = []

        if self._position_side:
            self._hold_bars += 1
            max_hold = self.get_param("max_hold_bars", 60)
            atr_stop = self.get_param("atr_stop_mult", 2.0)

            if self._position_side == "long":
                stop_hit = c < self._entry_price - atr_stop * atr
                channel_exit = c < exit_low
            else:
                stop_hit = c > self._entry_price + atr_stop * atr
                channel_exit = c > exit_high

            if stop_hit or channel_exit or self._hold_bars >= max_hold:
                exit_type = SignalType.LONG_EXIT if self._position_side == "long" else SignalType.SHORT_EXIT
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=exit_type, strength=0.8, price=c,
                    reason=f"donchian_exit: hold={self._hold_bars} stop={stop_hit} channel={channel_exit}",
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
            if c > entry_high:
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY,
                    strength=0.8, price=c,
                    reason=f"donchian_break_up: c={c:.1f} > {entry_period}p_high={entry_high:.1f}",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = "long"
                self._entry_price = c
                self._hold_bars = 0

            elif c < entry_low:
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY,
                    strength=0.8, price=c,
                    reason=f"donchian_break_down: c={c:.1f} < {entry_period}p_low={entry_low:.1f}",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = "short"
                self._entry_price = c
                self._hold_bars = 0

        return signals

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        return []
