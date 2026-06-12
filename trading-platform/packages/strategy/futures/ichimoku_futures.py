"""一目均衡表策略（期货适配版）— 日本经典趋势分析系统。

Goichi Hosoda (1930s-1960s, 经典日本技术分析):
  转换线 (Tenkan-sen) = (9-period High + 9-period Low) / 2
  基准线 (Kijun-sen) = (26-period High + 26-period Low) / 2
  先行A (Senkou A) = (Tenkan + Kijun) / 2, 移位26周期
  先行B (Senkou B) = (52-period High + 52-period Low) / 2, 移位26周期
  延迟线 (Chikou) = 收盘价移位-26周期

期货适配（参数调整为日内）：
  - 标准参数 9/26/52 → 日内适配 7/22/44（一天交易时间更短）
  - 云层突破是主要信号
  - 转换线/基准线交叉是次要信号

Method: 一目均衡表 (Hosoda 1930s-1960s, 经典日本技术分析)
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
    "tenkan_period": 7,
    "kijun_period": 22,
    "senkou_b_period": 44,
    "atr_period": 14,
    "tp_atr_mult": 2.5,
    "sl_atr_mult": 1.2,
    "max_hold_bars": 50,
    "cooldown_bars": 3,
}


def _donchian_mid(highs: list[float], lows: list[float], period: int) -> float:
    if len(highs) < period:
        return (highs[-1] + lows[-1]) / 2 if highs else 0.0
    return (max(highs[-period:]) + min(lows[-period:])) / 2


@auto_register("ichimoku_futures")
class IchimokuFuturesStrategy(BaseStrategy):
    """一目均衡表期货策略 — 云层突破 + TK 交叉。"""

    def __init__(self, config: StrategyConfig) -> None:
        config = config.model_copy(update={"params": {**DEFAULT_PARAMS, **config.params}})
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

        senkou_b_period = self.get_param("senkou_b_period", 44)
        if self._bar_count < senkou_b_period + 5:
            return []

        atr = calc_atr(list(self._highs), list(self._lows), list(self._closes),
                       self.get_param("atr_period", 14))
        if atr is None or atr < 1e-10:
            return []

        high_list = list(self._highs)
        low_list = list(self._lows)

        tenkan = _donchian_mid(high_list, low_list, self.get_param("tenkan_period", 7))
        kijun = _donchian_mid(high_list, low_list, self.get_param("kijun_period", 22))
        senkou_a = (tenkan + kijun) / 2
        senkou_b = _donchian_mid(high_list, low_list, senkou_b_period)

        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)
        above_cloud = c > cloud_top
        below_cloud = c < cloud_bottom
        tk_bullish = tenkan > kijun
        tk_bearish = tenkan < kijun

        signals = []

        if self._position_side:
            self._hold_bars += 1
            max_hold = self.get_param("max_hold_bars", 50)
            tp_mult = self.get_param("tp_atr_mult", 2.5)
            sl_mult = self.get_param("sl_atr_mult", 1.2)

            pnl = (c - self._entry_price) / self._entry_price if self._position_side == "long" else (self._entry_price - c) / self._entry_price
            if pnl >= tp_mult * atr / self._entry_price or pnl <= -sl_mult * atr / self._entry_price or self._hold_bars >= max_hold:
                exit_type = SignalType.LONG_EXIT if self._position_side == "long" else SignalType.SHORT_EXIT
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=exit_type, strength=0.8, price=c,
                    reason=f"ichi_exit: hold={self._hold_bars}",
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
            if above_cloud and tk_bullish:
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY,
                    strength=0.9, price=c,
                    reason=f"ichi_buy: above_cloud T={tenkan:.1f}>K={kijun:.1f} cloud=[{cloud_bottom:.1f},{cloud_top:.1f}]",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = "long"
                self._entry_price = c
                self._hold_bars = 0

            elif below_cloud and tk_bearish:
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY,
                    strength=0.9, price=c,
                    reason=f"ichi_sell: below_cloud T={tenkan:.1f}<K={kijun:.1f} cloud=[{cloud_bottom:.1f},{cloud_top:.1f}]",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = "short"
                self._entry_price = c
                self._hold_bars = 0

        return signals

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        return []
