"""夜盘外盘联动策略 — 利用夜盘开盘对外盘走势的反应。

国内期货夜盘独特现象：
  - 21:00 开盘时，LME/COMEX/NYMEX 已交易数小时
  - 外盘方向对国内品种有先导性影响
  - 品种联动关系：
    - cu/al/zn/ni ← LME 同品种
    - au/ag ← COMEX Gold/Silver
    - sc ← WTI/Brent Crude
  - 夜盘开盘的 gap 和前 15 分钟走势 = 市场对外盘的定价

信号构建：
  1. 日间收盘价 vs 夜盘开盘价的 gap 方向和大小
  2. 夜盘前 15 分钟的动量方向（市场对 gap 的确认/拒绝）
  3. 成交量异常（外盘重大事件 → 国内成交量放大）
  4. Gap 大小与历史分布的 Z-score（极端 gap = 趋势信号）

Method: 经典跨市场联动分析 + 时段效应
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
    "gap_zscore_threshold": 1.5,
    "gap_history_window": 30,
    "momentum_confirm_bars": 3,
    "momentum_threshold_atr": 0.5,
    "volume_surge_mult": 2.0,
    "atr_period": 14,
    "tp_atr_mult": 2.0,
    "sl_atr_mult": 1.2,
    "max_hold_bars": 30,
    "cooldown_bars": 3,
}


@auto_register("nighttime_linkage")
class NighttimeLinkageStrategy(BaseStrategy):
    """夜盘外盘联动策略 — 开盘 gap + 动量确认。"""

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

        self._last_day_close: float | None = None
        self._night_open: float | None = None
        self._night_open_bar: int = -1
        self._gap_history: deque[float] = deque(maxlen=100)
        self._in_night_session = False

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

                if ts.hour == 15 and ts.minute <= 5:
                    self._last_day_close = c
                    self._in_night_session = False

                if ts.hour == 21 and ts.minute < 10 and not self._in_night_session:
                    self._in_night_session = True
                    self._night_open = c
                    self._night_open_bar = self._bar_count

                    if self._last_day_close is not None and self._last_day_close > 0:
                        gap_pct = (c - self._last_day_close) / self._last_day_close
                        self._gap_history.append(gap_pct)

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
            max_hold = self.get_param("max_hold_bars", 30)
            tp_mult = self.get_param("tp_atr_mult", 2.0)
            sl_mult = self.get_param("sl_atr_mult", 1.2)

            pnl = (c - self._entry_price) / self._entry_price if self._position_side == "long" else (self._entry_price - c) / self._entry_price
            if pnl >= tp_mult * atr / self._entry_price or pnl <= -sl_mult * atr / self._entry_price or self._hold_bars >= max_hold:
                exit_type = SignalType.LONG_EXIT if self._position_side == "long" else SignalType.SHORT_EXIT
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=exit_type, strength=0.8, price=c,
                    reason=f"night_exit: hold={self._hold_bars}",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = None
                self._hold_bars = 0
                self._cd = self.get_param("cooldown_bars", 3)
                return signals

        if self._cd > 0:
            self._cd -= 1

        confirm_bars = self.get_param("momentum_confirm_bars", 3)
        bars_since_open = self._bar_count - self._night_open_bar

        if not self._position_side and self._cd <= 0 and self._in_night_session and \
           self._night_open is not None and self._last_day_close is not None and \
           bars_since_open >= confirm_bars and bars_since_open <= confirm_bars + 2:

            gap_pct = (self._night_open - self._last_day_close) / self._last_day_close

            gap_std = np.std(list(self._gap_history)) if len(self._gap_history) >= 10 else abs(gap_pct) * 2
            gap_zscore = gap_pct / max(gap_std, 1e-10) if gap_std > 0 else 0.0

            momentum = (c - self._night_open) / atr
            mom_threshold = self.get_param("momentum_threshold_atr", 0.5)

            vol_list = list(self._volumes)
            vol_avg = np.mean(vol_list[-20:]) if len(vol_list) >= 20 else v
            vol_surge = v > vol_avg * self.get_param("volume_surge_mult", 2.0)

            gap_threshold = self.get_param("gap_zscore_threshold", 1.5)

            if abs(gap_zscore) > gap_threshold and momentum > mom_threshold and gap_pct > 0:
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY,
                    strength=min(abs(gap_zscore) / 3.0, 1.0), price=c,
                    reason=f"night_buy: gap={gap_pct*100:.2f}% z={gap_zscore:.2f} mom={momentum:.2f}" + (" vol_surge" if vol_surge else ""),
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = "long"
                self._entry_price = c
                self._hold_bars = 0

            elif abs(gap_zscore) > gap_threshold and momentum < -mom_threshold and gap_pct < 0:
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY,
                    strength=min(abs(gap_zscore) / 3.0, 1.0), price=c,
                    reason=f"night_sell: gap={gap_pct*100:.2f}% z={gap_zscore:.2f} mom={momentum:.2f}" + (" vol_surge" if vol_surge else ""),
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = "short"
                self._entry_price = c
                self._hold_bars = 0

        return signals

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        return []
