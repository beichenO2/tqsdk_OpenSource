"""ADX 趋势强度策略 — 用 ADX 值判断趋势强度并过滤入场。

Welles Wilder (1978, 经典):
  +DI = smoothed(+DM) / ATR * 100
  -DI = smoothed(-DM) / ATR * 100
  DX = |+DI - -DI| / (+DI + -DI) * 100
  ADX = smoothed(DX)

信号逻辑：
  ADX > 25 = 有趋势 → 趋势跟踪
  ADX < 20 = 无趋势 → 不交易或均值回归
  +DI > -DI = 多头趋势
  +DI < -DI = 空头趋势
  ADX 上升 = 趋势增强
  ADX 下降 = 趋势减弱

Method: ADX/DMI (Wilder 1978, 经典趋势强度指标)
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from ..base import BaseStrategy, Signal, SignalType, StrategyConfig
from ..indicators import calc_atr
from ..registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS: dict[str, Any] = {
    "adx_period": 14,
    "adx_trend_threshold": 25,
    "adx_strong_threshold": 35,
    "di_cross_confirm": 2,
    "tp_atr_mult": 2.5,
    "sl_atr_mult": 1.2,
    "max_hold_bars": 50,
    "cooldown_bars": 3,
}


@auto_register("adx_trend_strength")
class ADXTrendStrengthStrategy(BaseStrategy):
    """ADX 趋势强度策略 — DI 交叉 + ADX 过滤。"""

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

        self._plus_dm_smooth = 0.0
        self._minus_dm_smooth = 0.0
        self._tr_smooth = 0.0
        self._adx = 0.0
        self._prev_adx = 0.0
        self._plus_di = 0.0
        self._minus_di = 0.0
        self._warmup = 0

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        h = float(bar.get("high", 0))
        l = float(bar.get("low", 0))
        c = float(bar.get("close", 0))

        self._highs.append(h)
        self._lows.append(l)
        self._closes.append(c)
        self._bar_count += 1

        if self._bar_count < 2:
            return []

        period = self.get_param("adx_period", 14)
        high_list = list(self._highs)
        low_list = list(self._lows)

        tr = max(h - l, abs(h - list(self._closes)[-2]), abs(l - list(self._closes)[-2]))
        up_move = h - high_list[-2]
        dn_move = low_list[-2] - l
        pdm = max(up_move, 0.0) if up_move > dn_move else 0.0
        mdm = max(dn_move, 0.0) if dn_move > up_move else 0.0

        alpha = 1.0 / period

        if self._warmup < period:
            self._plus_dm_smooth += pdm
            self._minus_dm_smooth += mdm
            self._tr_smooth += tr
            self._warmup += 1
            return []

        self._tr_smooth = self._tr_smooth - self._tr_smooth * alpha + tr
        self._plus_dm_smooth = self._plus_dm_smooth - self._plus_dm_smooth * alpha + pdm
        self._minus_dm_smooth = self._minus_dm_smooth - self._minus_dm_smooth * alpha + mdm

        if self._tr_smooth > 0:
            self._plus_di = self._plus_dm_smooth / self._tr_smooth * 100
            self._minus_di = self._minus_dm_smooth / self._tr_smooth * 100

        di_sum = self._plus_di + self._minus_di
        dx = abs(self._plus_di - self._minus_di) / di_sum * 100 if di_sum > 0 else 0

        self._prev_adx = self._adx
        self._adx = self._adx + alpha * (dx - self._adx)

        if self._bar_count < period * 2 + 5:
            return []

        atr = calc_atr(list(self._highs), list(self._lows), list(self._closes), period)
        if atr is None or atr < 1e-10:
            return []

        signals = []
        trend_threshold = self.get_param("adx_trend_threshold", 25)
        has_trend = self._adx > trend_threshold
        adx_rising = self._adx > self._prev_adx

        if self._position_side:
            self._hold_bars += 1
            max_hold = self.get_param("max_hold_bars", 50)
            tp_mult = self.get_param("tp_atr_mult", 2.5)
            sl_mult = self.get_param("sl_atr_mult", 1.2)

            pnl = (c - self._entry_price) / self._entry_price if self._position_side == "long" else (self._entry_price - c) / self._entry_price
            di_reversed = (self._position_side == "long" and self._minus_di > self._plus_di) or \
                         (self._position_side == "short" and self._plus_di > self._minus_di)

            if pnl >= tp_mult * atr / self._entry_price or pnl <= -sl_mult * atr / self._entry_price or \
               self._hold_bars >= max_hold or (di_reversed and self._adx > trend_threshold):
                exit_type = SignalType.LONG_EXIT if self._position_side == "long" else SignalType.SHORT_EXIT
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=exit_type, strength=0.8, price=c,
                    reason=f"adx_exit: ADX={self._adx:.0f} +DI={self._plus_di:.0f} -DI={self._minus_di:.0f}",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = None
                self._hold_bars = 0
                self._cd = self.get_param("cooldown_bars", 3)
                return signals

        if self._cd > 0:
            self._cd -= 1

        if not self._position_side and self._cd <= 0 and has_trend and adx_rising:
            if self._plus_di > self._minus_di:
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY,
                    strength=min(self._adx / 50, 1.0), price=c,
                    reason=f"adx_buy: ADX={self._adx:.0f}↑ +DI={self._plus_di:.0f}>-DI={self._minus_di:.0f}",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = "long"
                self._entry_price = c
                self._hold_bars = 0

            elif self._minus_di > self._plus_di:
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY,
                    strength=min(self._adx / 50, 1.0), price=c,
                    reason=f"adx_sell: ADX={self._adx:.0f}↑ -DI={self._minus_di:.0f}>+DI={self._plus_di:.0f}",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = "short"
                self._entry_price = c
                self._hold_bars = 0

        return signals

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        return []
