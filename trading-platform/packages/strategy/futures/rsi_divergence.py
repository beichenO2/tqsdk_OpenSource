"""RSI 背离策略 — 价格与 RSI 指标的方向不一致信号。

经典背离理论：
  - 看涨背离：价格新低但 RSI 不新低 → 下跌动力减弱 → 反弹
  - 看跌背离：价格新高但 RSI 不新高 → 上涨动力减弱 → 回调
  - 隐藏看涨：价格高低但 RSI 新低 → 趋势继续
  - 隐藏看跌：价格低高但 RSI 新高 → 趋势继续

Method: RSI (Welles Wilder 1978, 经典技术分析)
背离理论是经典技术分析核心概念。
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
    "rsi_period": 14,
    "divergence_lookback": 20,
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    "price_new_extreme_pct": 0.001,
    "atr_period": 14,
    "tp_atr_mult": 2.0,
    "sl_atr_mult": 1.0,
    "max_hold_bars": 30,
    "cooldown_bars": 3,
}


def _compute_rsi(closes: list[float], period: int) -> list[float]:
    rsi_values = [50.0] * min(period, len(closes))
    if len(closes) < period + 1:
        return rsi_values

    gains = []
    losses = []
    for i in range(1, period + 1):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))

    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)

    for i in range(period, len(closes)):
        d = closes[i] - closes[i - 1]
        avg_gain = (avg_gain * (period - 1) + max(d, 0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-d, 0)) / period

        if avg_loss < 1e-10:
            rsi_values.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsi_values.append(100.0 - 100.0 / (1.0 + rs))

    return rsi_values


@auto_register("rsi_divergence")
class RSIDivergenceStrategy(BaseStrategy):
    """RSI 背离策略 — 检测价格-RSI 方向不一致信号。"""

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

        rsi_period = self.get_param("rsi_period", 14)
        div_lb = self.get_param("divergence_lookback", 20)

        if self._bar_count < rsi_period + div_lb + 5:
            return []

        atr = calc_atr(list(self._highs), list(self._lows), list(self._closes),
                       self.get_param("atr_period", 14))
        if atr is None or atr < 1e-10:
            return []

        close_list = list(self._closes)
        rsi_values = _compute_rsi(close_list, rsi_period)

        if len(rsi_values) < div_lb:
            return []

        signals = []
        current_rsi = rsi_values[-1]
        recent_prices = close_list[-div_lb:]
        recent_rsi = rsi_values[-div_lb:]

        price_min_idx = int(np.argmin(recent_prices))
        price_max_idx = int(np.argmax(recent_prices))
        rsi_min_idx = int(np.argmin(recent_rsi))
        rsi_max_idx = int(np.argmax(recent_rsi))

        new_extreme_pct = self.get_param("price_new_extreme_pct", 0.001)
        oversold = self.get_param("rsi_oversold", 30)
        overbought = self.get_param("rsi_overbought", 70)

        bullish_div = (
            price_min_idx > div_lb // 2 and
            close_list[-1] <= min(recent_prices) * (1 + new_extreme_pct) and
            rsi_values[-1] > min(recent_rsi) + 2.0 and
            current_rsi < oversold + 10
        )

        bearish_div = (
            price_max_idx > div_lb // 2 and
            close_list[-1] >= max(recent_prices) * (1 - new_extreme_pct) and
            rsi_values[-1] < max(recent_rsi) - 2.0 and
            current_rsi > overbought - 10
        )

        if self._position_side:
            self._hold_bars += 1
            max_hold = self.get_param("max_hold_bars", 30)
            tp_mult = self.get_param("tp_atr_mult", 2.0)
            sl_mult = self.get_param("sl_atr_mult", 1.0)

            pnl = (c - self._entry_price) / self._entry_price if self._position_side == "long" else (self._entry_price - c) / self._entry_price
            if pnl >= tp_mult * atr / self._entry_price or pnl <= -sl_mult * atr / self._entry_price or self._hold_bars >= max_hold:
                exit_type = SignalType.LONG_EXIT if self._position_side == "long" else SignalType.SHORT_EXIT
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=exit_type, strength=0.8, price=c,
                    reason=f"rsi_div_exit: hold={self._hold_bars}",
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
            if bullish_div:
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY,
                    strength=0.8, price=c,
                    reason=f"rsi_bullish_div: RSI={current_rsi:.0f} price_near_low, RSI_higher",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = "long"
                self._entry_price = c
                self._hold_bars = 0

            elif bearish_div:
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY,
                    strength=0.8, price=c,
                    reason=f"rsi_bearish_div: RSI={current_rsi:.0f} price_near_high, RSI_lower",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = "short"
                self._entry_price = c
                self._hold_bars = 0

        return signals

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        return []
