"""午间跳空因子策略 — 利用午休前后的价格跳空捕捉方向性机会。

经典日内交易理论 + 时段效应：
  - 午休期间（11:30-13:30）外部信息持续流入但无法交易
  - 午后开盘（13:30）消化这些信息后产生方向性跳空
  - 跳空方向反映市场对午间信息的集体判断
  - 午后首根 K 线的成交量确认跳空有效性

具体信号：
  1. Gap Direction: 午后开盘价 vs 午前收盘价的跳空方向
  2. Gap Magnitude: 跳空幅度超过 N ATR 才视为有效
  3. Volume Confirmation: 午后首根 K 线成交量 > 均量 M 倍确认动量
  4. Momentum Follow: 跳空后前 K 根 K 线延续同方向则加强信号

Method: 经典日内交易理论 (gap trading) + 时段效应 (session effects)
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np

from ..base import BaseStrategy, Signal, SignalType, StrategyConfig
from ..indicators import calc_atr
from ..registry import auto_register
from .session import FuturesSessionType, get_session

logger = logging.getLogger(__name__)

DEFAULT_PARAMS: dict[str, Any] = {
    "gap_atr_threshold": 0.8,
    "volume_confirm_mult": 1.5,
    "momentum_bars": 3,
    "momentum_confirm_ratio": 0.6,
    "atr_period": 14,
    "tp_atr_mult": 2.0,
    "sl_atr_mult": 1.2,
    "max_hold_bars": 30,
    "cooldown_bars": 5,
    "entry_window_bars": 5,
}


@auto_register("lunch_gap")
class LunchGapStrategy(BaseStrategy):
    """午间跳空因子策略 — 午休跳空 + 成交量确认 + 动量跟随。"""

    def __init__(self, config: StrategyConfig) -> None:
        config = config.model_copy(
            update={"params": {**DEFAULT_PARAMS, **config.params}}
        )
        super().__init__(config)
        self._highs: deque[float] = deque(maxlen=200)
        self._lows: deque[float] = deque(maxlen=200)
        self._closes: deque[float] = deque(maxlen=200)
        self._volumes: deque[float] = deque(maxlen=200)
        self._bar_count = 0

        self._pre_lunch_close: float | None = None
        self._post_lunch_open: float | None = None
        self._gap_direction: int = 0
        self._gap_magnitude: float = 0.0
        self._afternoon_bar_idx: int = 0

        self._position_side: str | None = None
        self._entry_price = 0.0
        self._hold_bars = 0
        self._cd = 0

        self._prev_session: FuturesSessionType | None = None
        self._post_lunch_closes: list[float] = []

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

        current_session = FuturesSessionType.CLOSED
        dt = bar.get("datetime")
        if dt is not None:
            try:
                import pandas as pd
                ts = pd.Timestamp(dt)
                from datetime import datetime, timezone, timedelta
                cst = timezone(timedelta(hours=8))
                cst_dt = ts.to_pydatetime().replace(tzinfo=cst) if ts.tzinfo is None else ts.to_pydatetime().astimezone(cst)
                current_session = get_session(cst_dt)
            except Exception:
                pass

        if (self._prev_session == FuturesSessionType.MORNING_LATE
                and current_session == FuturesSessionType.LUNCH_BREAK):
            self._pre_lunch_close = c
            self._gap_direction = 0
            self._gap_magnitude = 0.0
            self._afternoon_bar_idx = 0
            self._post_lunch_closes = []

        if (self._prev_session in (FuturesSessionType.LUNCH_BREAK, None)
                and current_session == FuturesSessionType.AFTERNOON
                and self._pre_lunch_close is not None
                and self._afternoon_bar_idx == 0):
            self._post_lunch_open = c
            self._afternoon_bar_idx = 1

        if current_session == FuturesSessionType.AFTERNOON and self._afternoon_bar_idx > 0:
            self._post_lunch_closes.append(c)
            self._afternoon_bar_idx += 1

        self._prev_session = current_session

        if self._bar_count < 30:
            return []

        atr = calc_atr(list(self._highs), list(self._lows), list(self._closes),
                       self.get_param("atr_period", 14))
        if atr is None or atr < 1e-10:
            return []

        signals: list[Signal] = []

        if self._position_side:
            self._hold_bars += 1
            tp_mult = self.get_param("tp_atr_mult", 2.0)
            sl_mult = self.get_param("sl_atr_mult", 1.2)
            max_hold = self.get_param("max_hold_bars", 30)

            if self._position_side == "long":
                pnl = (c - self._entry_price) / self._entry_price
            else:
                pnl = (self._entry_price - c) / self._entry_price

            tp_hit = pnl >= tp_mult * atr / self._entry_price
            sl_hit = pnl <= -sl_mult * atr / self._entry_price

            if sl_hit or tp_hit or self._hold_bars >= max_hold:
                exit_type = (SignalType.LONG_EXIT if self._position_side == "long"
                             else SignalType.SHORT_EXIT)
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=exit_type, strength=0.8, price=c,
                    reason=f"lgap_exit: pnl={pnl:.4f} hold={self._hold_bars}",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = None
                self._hold_bars = 0
                self._cd = self.get_param("cooldown_bars", 5)
                return signals

        if self._cd > 0:
            self._cd -= 1

        entry_window = self.get_param("entry_window_bars", 5)
        if (not self._position_side
                and self._cd <= 0
                and self._pre_lunch_close is not None
                and self._post_lunch_open is not None
                and current_session == FuturesSessionType.AFTERNOON
                and 1 <= self._afternoon_bar_idx <= entry_window + 1):

            gap = (self._post_lunch_open - self._pre_lunch_close) / atr
            gap_threshold = self.get_param("gap_atr_threshold", 0.8)

            if abs(gap) >= gap_threshold:
                self._gap_direction = 1 if gap > 0 else -1
                self._gap_magnitude = abs(gap)

                vol_list = list(self._volumes)
                vol_avg = float(np.mean(vol_list[-20:])) if len(vol_list) >= 20 else v
                vol_confirmed = v > vol_avg * self.get_param("volume_confirm_mult", 1.5)

                momentum_confirmed = False
                momentum_bars = self.get_param("momentum_bars", 3)
                confirm_ratio = self.get_param("momentum_confirm_ratio", 0.6)
                if len(self._post_lunch_closes) >= 2:
                    same_dir_count = 0
                    check_bars = self._post_lunch_closes[-momentum_bars:]
                    for i in range(1, len(check_bars)):
                        delta = check_bars[i] - check_bars[i - 1]
                        if (self._gap_direction > 0 and delta > 0) or \
                           (self._gap_direction < 0 and delta < 0):
                            same_dir_count += 1
                    if len(check_bars) > 1:
                        momentum_confirmed = same_dir_count / (len(check_bars) - 1) >= confirm_ratio

                if vol_confirmed or momentum_confirmed:
                    strength = min(0.3 + self._gap_magnitude * 0.15, 1.0)
                    if vol_confirmed and momentum_confirmed:
                        strength = min(strength + 0.2, 1.0)

                    entry_reasons = [f"gap={gap:.2f}ATR"]
                    if vol_confirmed:
                        entry_reasons.append(f"vol_confirm={v/max(vol_avg,1):.1f}x")
                    if momentum_confirmed:
                        entry_reasons.append("momentum_ok")

                    if self._gap_direction > 0:
                        sig = Signal(
                            strategy_id=self.strategy_id, symbol=symbol,
                            signal_type=SignalType.LONG_ENTRY,
                            strength=strength, price=c,
                            reason=f"lgap_buy: {' + '.join(entry_reasons)}",
                        )
                        signals.append(sig)
                        self.record_signal(sig)
                        self._position_side = "long"
                        self._entry_price = c
                        self._hold_bars = 0

                    else:
                        sig = Signal(
                            strategy_id=self.strategy_id, symbol=symbol,
                            signal_type=SignalType.SHORT_ENTRY,
                            strength=strength, price=c,
                            reason=f"lgap_sell: {' + '.join(entry_reasons)}",
                        )
                        signals.append(sig)
                        self.record_signal(sig)
                        self._position_side = "short"
                        self._entry_price = c
                        self._hold_bars = 0

        return signals

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        return []
