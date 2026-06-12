"""日内反转策略 — 利用期货特有的时段效应进行均值回归。

经典理论 + 日内独特现象：
  - 开盘 15min 过度反应 → 反转机会（行为金融学 overreaction bias）
  - 午间跳空消化 → 回归合理价位
  - 尾盘清仓压力 → 反向信号
  - VWAP 作为日内"公平价格"锚定

具体信号：
  1. Opening Gap Fade: 开盘跳空超过 N ATR 后的反转
  2. VWAP Reversion: 价格偏离 VWAP 超过 M ATR 后回归
  3. RSI Extreme Bounce: RSI 进入极端区域后的快速反转
  4. Volume Climax: 成交量突然放大（恐慌/贪婪顶点）后反转

Method: 行为金融学 overreaction (De Bondt & Thaler 1985, 经典)
        + VWAP 均值回归（经典量化方法）
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
    "gap_atr_threshold": 1.5,
    "vwap_deviation_atr": 2.0,
    "rsi_period": 8,
    "rsi_extreme_high": 80,
    "rsi_extreme_low": 20,
    "volume_climax_mult": 3.0,
    "atr_period": 14,
    "tp_atr_mult": 1.5,
    "sl_atr_mult": 1.0,
    "max_hold_bars": 20,
    "cooldown_bars": 3,
    "vwap_period": 60,
}


def _rsi(closes: list[float], period: int) -> float:
    if len(closes) < period + 1:
        return 50.0
    gains, losses = [], []
    for i in range(-period, 0):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    avg_g = np.mean(gains)
    avg_l = np.mean(losses)
    if avg_l < 1e-10:
        return 100.0
    return 100.0 - 100.0 / (1.0 + avg_g / avg_l)


@auto_register("intraday_reversal")
class IntradayReversalStrategy(BaseStrategy):
    """日内反转策略 — 捕捉过度反应后的均值回归。"""

    def __init__(self, config: StrategyConfig) -> None:
        config = config.model_copy(
            update={"params": {**DEFAULT_PARAMS, **config.params}}
        )
        super().__init__(config)
        self._highs: deque[float] = deque(maxlen=200)
        self._lows: deque[float] = deque(maxlen=200)
        self._closes: deque[float] = deque(maxlen=200)
        self._volumes: deque[float] = deque(maxlen=200)
        self._tp_volumes: deque[float] = deque(maxlen=200)
        self._bar_count = 0
        self._position_side: str | None = None
        self._entry_price = 0.0
        self._hold_bars = 0
        self._cd = 0
        self._session_open_price: float | None = None
        self._last_session_bar = -1

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        h = float(bar.get("high", 0))
        l = float(bar.get("low", 0))
        c = float(bar.get("close", 0))
        v = float(bar.get("volume", 0))

        self._highs.append(h)
        self._lows.append(l)
        self._closes.append(c)
        self._volumes.append(v)
        self._tp_volumes.append(c * v)
        self._bar_count += 1

        dt = bar.get("datetime")
        if dt is not None:
            try:
                import pandas as pd
                ts = pd.Timestamp(dt)
                if ts.hour == 9 and ts.minute < 10 and self._bar_count - self._last_session_bar > 5:
                    self._session_open_price = c
                    self._last_session_bar = self._bar_count
                elif ts.hour == 21 and ts.minute < 10 and self._bar_count - self._last_session_bar > 5:
                    self._session_open_price = c
                    self._last_session_bar = self._bar_count
            except Exception:
                pass

        if self._bar_count < 30:
            return []

        atr = calc_atr(list(self._highs), list(self._lows), list(self._closes),
                       self.get_param("atr_period", 14))
        if atr is None or atr < 1e-10:
            return []

        signals = []
        close_list = list(self._closes)
        vol_list = list(self._volumes)

        rsi = _rsi(close_list, self.get_param("rsi_period", 8))

        vwap_period = min(self.get_param("vwap_period", 60), len(self._tp_volumes))
        tp_sum = sum(list(self._tp_volumes)[-vwap_period:])
        v_sum = sum(vol_list[-vwap_period:])
        vwap = tp_sum / max(v_sum, 1e-10)
        vwap_dev = (c - vwap) / atr

        vol_avg = np.mean(vol_list[-20:]) if len(vol_list) >= 20 else v
        vol_climax = v > vol_avg * self.get_param("volume_climax_mult", 3.0)

        gap_signal = 0
        if self._session_open_price is not None and self._bar_count - self._last_session_bar <= 3:
            gap = (c - self._session_open_price) / atr
            gap_threshold = self.get_param("gap_atr_threshold", 1.5)
            if gap > gap_threshold:
                gap_signal = -1  # gap up → fade short
            elif gap < -gap_threshold:
                gap_signal = 1  # gap down → fade long

        if self._position_side:
            self._hold_bars += 1
            max_hold = self.get_param("max_hold_bars", 20)
            tp_mult = self.get_param("tp_atr_mult", 1.5)
            sl_mult = self.get_param("sl_atr_mult", 1.0)

            pnl = (c - self._entry_price) / self._entry_price if self._position_side == "long" else (self._entry_price - c) / self._entry_price
            tp_hit = pnl >= tp_mult * atr / self._entry_price
            sl_hit = pnl <= -sl_mult * atr / self._entry_price

            if sl_hit or tp_hit or self._hold_bars >= max_hold:
                exit_type = SignalType.LONG_EXIT if self._position_side == "long" else SignalType.SHORT_EXIT
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=exit_type, strength=0.8, price=c,
                    reason=f"rev_exit: pnl={pnl:.4f} hold={self._hold_bars}",
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
            rsi_high = self.get_param("rsi_extreme_high", 80)
            rsi_low = self.get_param("rsi_extreme_low", 20)
            vwap_threshold = self.get_param("vwap_deviation_atr", 2.0)

            entry_reason = []
            entry_direction = 0

            if rsi < rsi_low:
                entry_direction += 1
                entry_reason.append(f"rsi_oversold={rsi:.0f}")
            elif rsi > rsi_high:
                entry_direction -= 1
                entry_reason.append(f"rsi_overbought={rsi:.0f}")

            if vwap_dev < -vwap_threshold:
                entry_direction += 1
                entry_reason.append(f"below_vwap={vwap_dev:.1f}ATR")
            elif vwap_dev > vwap_threshold:
                entry_direction -= 1
                entry_reason.append(f"above_vwap={vwap_dev:.1f}ATR")

            if gap_signal != 0:
                entry_direction += gap_signal
                entry_reason.append(f"gap_fade={'up' if gap_signal < 0 else 'down'}")

            if vol_climax and abs(entry_direction) > 0:
                entry_reason.append("vol_climax")

            if entry_direction >= 2 or (entry_direction >= 1 and vol_climax):
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY,
                    strength=min(abs(entry_direction) * 0.3, 1.0), price=c,
                    reason=f"rev_buy: {' + '.join(entry_reason)}",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = "long"
                self._entry_price = c
                self._hold_bars = 0

            elif entry_direction <= -2 or (entry_direction <= -1 and vol_climax):
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY,
                    strength=min(abs(entry_direction) * 0.3, 1.0), price=c,
                    reason=f"rev_sell: {' + '.join(entry_reason)}",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = "short"
                self._entry_price = c
                self._hold_bars = 0

        return signals

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        return []
