"""收盘压力策略 — 尾盘清仓效应的反向交易。

国内期货 14:45-15:00 收盘效应：
  - 日内交易者集中平仓 → 产生单方向清仓压力
  - 清仓压力方向 = 当日持仓方向的反向
  - 多头集中平仓 → 尾盘下跌 → 次日开盘反弹概率高
  - 空头集中平仓 → 尾盘上涨 → 次日可能回落

信号构建（日内适配）：
  1. 14:30-14:45 期间的价格变化方向 = 清仓方向预判
  2. 14:45 后的加速移动 = 清仓压力确认
  3. 反向入场，目标为隔夜 gap 或次日开盘回弹
  4. 日内平仓版：13:30-14:30 累积趋势 → 14:30 后反向

Method: 日内交易理论 + 市场微结构（清仓效应），经典
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
    "afternoon_trend_period": 12,
    "trend_threshold_atr": 1.0,
    "reversal_confirm_atr": 0.3,
    "volume_surge_mult": 1.5,
    "atr_period": 14,
    "tp_atr_mult": 1.5,
    "sl_atr_mult": 0.8,
    "max_hold_bars": 15,
    "cooldown_bars": 2,
}


@auto_register("closing_pressure")
class ClosingPressureStrategy(BaseStrategy):
    """收盘压力策略 — 尾盘清仓效应的反向日内交易。"""

    def __init__(self, config: StrategyConfig) -> None:
        config = config.model_copy(
            update={"params": {**DEFAULT_PARAMS, **config.params}}
        )
        super().__init__(config)
        self._closes: deque[float] = deque(maxlen=200)
        self._highs: deque[float] = deque(maxlen=200)
        self._lows: deque[float] = deque(maxlen=200)
        self._volumes: deque[float] = deque(maxlen=200)
        self._bar_count = 0
        self._position_side: str | None = None
        self._entry_price = 0.0
        self._hold_bars = 0
        self._cd = 0
        self._in_closing_window = False
        self._afternoon_start_price: float | None = None

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        h = float(bar.get("high", 0))
        l = float(bar.get("low", 0))
        c = float(bar.get("close", 0))
        v = float(bar.get("volume", 0))

        self._highs.append(h)
        self._lows.append(l)
        self._closes.append(c)
        self._volumes.append(v)
        self._bar_count += 1

        dt = bar.get("datetime")
        if dt is not None:
            try:
                import pandas as pd
                ts = pd.Timestamp(dt)
                if ts.hour == 13 and ts.minute <= 35:
                    self._afternoon_start_price = c
                self._in_closing_window = (ts.hour == 14 and ts.minute >= 30) or ts.hour == 14 and ts.minute >= 30
            except Exception:
                pass

        if self._bar_count < 20:
            return []

        atr = calc_atr(list(self._highs), list(self._lows), list(self._closes),
                       self.get_param("atr_period", 14))
        if atr is None or atr < 1e-10:
            return []

        signals = []

        if self._position_side:
            self._hold_bars += 1
            max_hold = self.get_param("max_hold_bars", 15)
            tp_mult = self.get_param("tp_atr_mult", 1.5)
            sl_mult = self.get_param("sl_atr_mult", 0.8)

            pnl = (c - self._entry_price) / self._entry_price if self._position_side == "long" else (self._entry_price - c) / self._entry_price
            if pnl >= tp_mult * atr / self._entry_price or pnl <= -sl_mult * atr / self._entry_price or self._hold_bars >= max_hold:
                exit_type = SignalType.LONG_EXIT if self._position_side == "long" else SignalType.SHORT_EXIT
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=exit_type, strength=0.8, price=c,
                    reason=f"close_exit: hold={self._hold_bars}",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = None
                self._hold_bars = 0
                self._cd = self.get_param("cooldown_bars", 2)
                return signals

        if self._cd > 0:
            self._cd -= 1

        if not self._position_side and self._cd <= 0 and self._in_closing_window and self._afternoon_start_price is not None:
            close_list = list(self._closes)
            trend_period = self.get_param("afternoon_trend_period", 12)

            if len(close_list) >= trend_period:
                afternoon_trend = (c - close_list[-trend_period]) / atr
                trend_threshold = self.get_param("trend_threshold_atr", 1.0)

                vol_list = list(self._volumes)
                vol_avg = np.mean(vol_list[-20:]) if len(vol_list) >= 20 else v
                vol_surge = v > vol_avg * self.get_param("volume_surge_mult", 1.5)

                if afternoon_trend > trend_threshold and vol_surge:
                    sig = Signal(
                        strategy_id=self.strategy_id, symbol=symbol,
                        signal_type=SignalType.SHORT_ENTRY,
                        strength=min(abs(afternoon_trend) / 3.0, 1.0), price=c,
                        reason=f"close_fade_sell: trend={afternoon_trend:.2f}ATR vol_surge",
                    )
                    signals.append(sig)
                    self.record_signal(sig)
                    self._position_side = "short"
                    self._entry_price = c
                    self._hold_bars = 0

                elif afternoon_trend < -trend_threshold and vol_surge:
                    sig = Signal(
                        strategy_id=self.strategy_id, symbol=symbol,
                        signal_type=SignalType.LONG_ENTRY,
                        strength=min(abs(afternoon_trend) / 3.0, 1.0), price=c,
                        reason=f"close_fade_buy: trend={afternoon_trend:.2f}ATR vol_surge",
                    )
                    signals.append(sig)
                    self.record_signal(sig)
                    self._position_side = "long"
                    self._entry_price = c
                    self._hold_bars = 0

        return signals

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        return []
