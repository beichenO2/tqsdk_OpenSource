"""Tick 微结构策略 — 利用逐笔成交数据的微观结构信号。

国内期货 tick 数据包含：
  - LastPrice: 最新成交价
  - Volume: 累积成交量（日内递增）
  - OpenInterest: 持仓量
  - BidPrice1/AskPrice1: 买一/卖一价
  - BidVolume1/AskVolume1: 买一/卖一量

微结构信号（适配 5min K 线近似）：
  1. 价格加速度：连续几根 K 线的收益率变化率
  2. 成交强度：单位时间成交量骤变
  3. 持仓量加速：OI 变化率的变化率（二阶导）
  4. 高低差/实体比：K 线形态的量化特征（类 K 线理论）

Method: 市场微结构理论 (O'Hara 1995, 经典教科书)
        + 经典 K 线形态量化
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
    "acceleration_period": 5,
    "acceleration_threshold": 0.003,
    "volume_intensity_period": 10,
    "volume_intensity_mult": 2.5,
    "oi_accel_period": 5,
    "oi_accel_threshold": 0.001,
    "body_ratio_threshold": 0.7,
    "atr_period": 14,
    "tp_atr_mult": 2.0,
    "sl_atr_mult": 1.0,
    "max_hold_bars": 30,
    "cooldown_bars": 3,
}


def _body_ratio(open_: float, high: float, low: float, close: float) -> float:
    """K 线实体占比 = |close - open| / (high - low)。大实体 = 方向性强。"""
    hl_range = high - low
    if hl_range < 1e-10:
        return 0.0
    return abs(close - open_) / hl_range


def _upper_shadow_ratio(open_: float, high: float, low: float, close: float) -> float:
    """上影线比例。长上影 = 卖压。"""
    hl_range = high - low
    if hl_range < 1e-10:
        return 0.0
    return (high - max(open_, close)) / hl_range


def _lower_shadow_ratio(open_: float, high: float, low: float, close: float) -> float:
    """下影线比例。长下影 = 买盘支撑。"""
    hl_range = high - low
    if hl_range < 1e-10:
        return 0.0
    return (min(open_, close) - low) / hl_range


@auto_register("tick_microstructure")
class TickMicrostructureStrategy(BaseStrategy):
    """Tick 微结构日内策略 — 价格加速度 + 成交强度 + K 线形态。"""

    def __init__(self, config: StrategyConfig) -> None:
        config = config.model_copy(
            update={"params": {**DEFAULT_PARAMS, **config.params}}
        )
        super().__init__(config)
        self._opens: deque[float] = deque(maxlen=200)
        self._highs: deque[float] = deque(maxlen=200)
        self._lows: deque[float] = deque(maxlen=200)
        self._closes: deque[float] = deque(maxlen=200)
        self._volumes: deque[float] = deque(maxlen=200)
        self._ois: deque[float] = deque(maxlen=200)
        self._returns: deque[float] = deque(maxlen=200)
        self._bar_count = 0
        self._position_side: str | None = None
        self._entry_price = 0.0
        self._hold_bars = 0
        self._cd = 0

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        o = float(bar.get("open", 0))
        h = float(bar.get("high", 0))
        l = float(bar.get("low", 0))
        c = float(bar.get("close", 0))
        v = float(bar.get("volume", 0))
        oi = float(bar.get("open_interest", 0))

        self._opens.append(o)
        self._highs.append(h)
        self._lows.append(l)
        self._closes.append(c)
        self._volumes.append(v)
        self._ois.append(oi)

        if self._closes and len(self._closes) >= 2:
            ret = (c - list(self._closes)[-2]) / list(self._closes)[-2] if list(self._closes)[-2] > 0 else 0.0
            self._returns.append(ret)

        self._bar_count += 1

        if self._bar_count < 30:
            return []

        atr = calc_atr(list(self._highs), list(self._lows), list(self._closes),
                       self.get_param("atr_period", 14))
        if atr is None or atr < 1e-10:
            return []

        signals = []

        ret_list = list(self._returns)
        accel_period = self.get_param("acceleration_period", 5)
        if len(ret_list) >= accel_period * 2:
            recent_ret = np.mean(ret_list[-accel_period:])
            prev_ret = np.mean(ret_list[-accel_period * 2:-accel_period])
            acceleration = recent_ret - prev_ret
        else:
            acceleration = 0.0

        vol_list = list(self._volumes)
        vi_period = self.get_param("volume_intensity_period", 10)
        if len(vol_list) >= vi_period + 1:
            vol_avg = np.mean(vol_list[-vi_period - 1:-1])
            vol_intensity = v / max(vol_avg, 1e-10)
        else:
            vol_intensity = 1.0

        oi_list = list(self._ois)
        oi_period = self.get_param("oi_accel_period", 5)
        if len(oi_list) >= oi_period * 2 + 1:
            oi_recent = (oi_list[-1] - oi_list[-oi_period]) / max(abs(oi_list[-oi_period]), 1e-10)
            oi_prev = (oi_list[-oi_period] - oi_list[-oi_period * 2]) / max(abs(oi_list[-oi_period * 2]), 1e-10)
            oi_accel = oi_recent - oi_prev
        else:
            oi_accel = 0.0

        body_r = _body_ratio(o, h, l, c)
        upper_s = _upper_shadow_ratio(o, h, l, c)
        lower_s = _lower_shadow_ratio(o, h, l, c)
        is_bullish_bar = c > o
        is_bearish_bar = c < o
        strong_body = body_r > self.get_param("body_ratio_threshold", 0.7)
        hammer = lower_s > 0.6 and body_r < 0.3  # 锤子线（做多信号）
        shooting_star = upper_s > 0.6 and body_r < 0.3  # 射击之星（做空信号）

        if self._position_side:
            self._hold_bars += 1
            max_hold = self.get_param("max_hold_bars", 30)
            tp_mult = self.get_param("tp_atr_mult", 2.0)
            sl_mult = self.get_param("sl_atr_mult", 1.0)

            pnl = (c - self._entry_price) / self._entry_price if self._position_side == "long" else (self._entry_price - c) / self._entry_price
            tp_hit = pnl >= tp_mult * atr / self._entry_price
            sl_hit = pnl <= -sl_mult * atr / self._entry_price

            if sl_hit or tp_hit or self._hold_bars >= max_hold:
                exit_type = SignalType.LONG_EXIT if self._position_side == "long" else SignalType.SHORT_EXIT
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=exit_type, strength=0.8, price=c,
                    reason=f"micro_exit: hold={self._hold_bars}",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = None
                self._hold_bars = 0
                self._cd = self.get_param("cooldown_bars", 3)
                return signals

        if self._cd > 0:
            self._cd -= 1

        accel_thresh = self.get_param("acceleration_threshold", 0.003)
        vi_mult = self.get_param("volume_intensity_mult", 2.5)
        oi_thresh = self.get_param("oi_accel_threshold", 0.001)

        if not self._position_side and self._cd <= 0:
            buy_score = 0
            sell_score = 0
            reasons = []

            if acceleration > accel_thresh:
                buy_score += 1
                reasons.append(f"accel={acceleration:.5f}")
            elif acceleration < -accel_thresh:
                sell_score += 1
                reasons.append(f"accel={acceleration:.5f}")

            if vol_intensity > vi_mult and is_bullish_bar:
                buy_score += 1
                reasons.append(f"vol_intensity={vol_intensity:.1f}")
            elif vol_intensity > vi_mult and is_bearish_bar:
                sell_score += 1
                reasons.append(f"vol_intensity={vol_intensity:.1f}")

            if oi_accel > oi_thresh:
                buy_score += 1
                reasons.append(f"oi_accel={oi_accel:.5f}")
            elif oi_accel < -oi_thresh:
                sell_score += 1
                reasons.append(f"oi_accel={oi_accel:.5f}")

            if strong_body and is_bullish_bar:
                buy_score += 1
                reasons.append("strong_bull_bar")
            elif strong_body and is_bearish_bar:
                sell_score += 1
                reasons.append("strong_bear_bar")

            if hammer:
                buy_score += 1
                reasons.append("hammer")
            if shooting_star:
                sell_score += 1
                reasons.append("shooting_star")

            if buy_score >= 3:
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY,
                    strength=min(buy_score / 5.0, 1.0), price=c,
                    reason=f"micro_buy({buy_score}): {' + '.join(reasons)}",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = "long"
                self._entry_price = c
                self._hold_bars = 0

            elif sell_score >= 3:
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY,
                    strength=min(sell_score / 5.0, 1.0), price=c,
                    reason=f"micro_sell({sell_score}): {' + '.join(reasons)}",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = "short"
                self._entry_price = c
                self._hold_bars = 0

        return signals

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        return []
