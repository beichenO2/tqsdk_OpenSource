"""供需区域策略 — 识别关键价格区域并在突破/回拉时交易。

理论：
  - 供需区域是"制度交易者"(institutions) 留下的大单痕迹
  - 强势离开的价格区域 → 可能形成供需失衡 → 价格回来时再次反应
  - 类似支撑/阻力但有更严格的定义标准

识别规则（经典技术分析 + 量化增强）：
  1. 急涨/急跌前的基底区域（窄幅整理 → 急速突破）
  2. 基底区域的成交量异常（大单建仓痕迹）
  3. 离开速度（用 ATR 衡量突破力度）
  4. 区域有效性（首次测试最有效，多次测试弱化）

日内适配：
  - 每个交易日重新扫描区域（跨日区域参考但降权）
  - 开盘 gap 可能创造新的供需区域

Method: 经典价格行为分析 (Price Action), 结合量化定义。
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..base import BaseStrategy, Signal, SignalType, StrategyConfig
from ..indicators import calc_atr
from ..registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS: dict[str, Any] = {
    "zone_lookback": 50,
    "consolidation_bars": 3,
    "consolidation_range_atr": 0.5,
    "breakout_atr_mult": 1.5,
    "zone_proximity_atr": 0.3,
    "zone_max_tests": 3,
    "atr_period": 14,
    "tp_atr_mult": 2.0,
    "sl_atr_mult": 1.0,
    "max_hold_bars": 30,
    "cooldown_bars": 3,
}


@dataclass
class Zone:
    """供需区域。"""
    high: float
    low: float
    zone_type: str  # "demand" (support) or "supply" (resistance)
    strength: float  # 0-1, decreases with each test
    bar_index: int
    tests: int = 0


@auto_register("supply_demand_zone")
class SupplyDemandZoneStrategy(BaseStrategy):
    """供需区域日内策略 — 识别并交易关键价格区域。"""

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
        self._bar_count = 0
        self._zones: list[Zone] = []
        self._position_side: str | None = None
        self._entry_price = 0.0
        self._hold_bars = 0
        self._cd = 0

    def _scan_zones(self, atr: float) -> None:
        """Scan recent bars for consolidation → breakout patterns → zone creation."""
        if len(self._closes) < 10:
            return

        consol_bars = self.get_param("consolidation_bars", 3)
        consol_range = self.get_param("consolidation_range_atr", 0.5) * atr
        breakout_mult = self.get_param("breakout_atr_mult", 1.5)

        highs = list(self._highs)
        lows = list(self._lows)
        closes = list(self._closes)

        for i in range(consol_bars, len(closes) - 1):
            range_start = i - consol_bars
            zone_high = max(highs[range_start:i])
            zone_low = min(lows[range_start:i])
            zone_range = zone_high - zone_low

            if zone_range > consol_range:
                continue

            breakout_move = abs(closes[i] - closes[i - 1])

            if breakout_move > breakout_mult * atr:
                if closes[i] > zone_high:
                    zone = Zone(
                        high=zone_high, low=zone_low,
                        zone_type="demand", strength=0.9,
                        bar_index=self._bar_count - (len(closes) - i),
                    )
                    self._zones.append(zone)
                elif closes[i] < zone_low:
                    zone = Zone(
                        high=zone_high, low=zone_low,
                        zone_type="supply", strength=0.9,
                        bar_index=self._bar_count - (len(closes) - i),
                    )
                    self._zones.append(zone)

        if len(self._zones) > 20:
            self._zones = self._zones[-20:]

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        o = float(bar.get("open", 0))
        h = float(bar.get("high", 0))
        l = float(bar.get("low", 0))
        c = float(bar.get("close", 0))
        v = float(bar.get("volume", 0))

        self._opens.append(o)
        self._highs.append(h)
        self._lows.append(l)
        self._closes.append(c)
        self._volumes.append(v)
        self._bar_count += 1

        if self._bar_count < 20:
            return []

        atr = calc_atr(list(self._highs), list(self._lows), list(self._closes),
                       self.get_param("atr_period", 14))
        if atr is None or atr < 1e-10:
            return []

        if self._bar_count % 10 == 0:
            self._scan_zones(atr)

        signals = []
        proximity = self.get_param("zone_proximity_atr", 0.3) * atr
        max_tests = self.get_param("zone_max_tests", 3)

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
                    reason=f"sdz_exit: hold={self._hold_bars}",
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
            for zone in self._zones:
                if zone.tests >= max_tests:
                    continue

                if zone.zone_type == "demand" and abs(c - zone.high) < proximity and c > zone.low:
                    zone.tests += 1
                    zone.strength *= 0.8
                    sig = Signal(
                        strategy_id=self.strategy_id, symbol=symbol,
                        signal_type=SignalType.LONG_ENTRY,
                        strength=zone.strength, price=c,
                        reason=f"sdz_demand: zone=[{zone.low:.1f},{zone.high:.1f}] test#{zone.tests}",
                    )
                    signals.append(sig)
                    self.record_signal(sig)
                    self._position_side = "long"
                    self._entry_price = c
                    self._hold_bars = 0
                    break

                elif zone.zone_type == "supply" and abs(c - zone.low) < proximity and c < zone.high:
                    zone.tests += 1
                    zone.strength *= 0.8
                    sig = Signal(
                        strategy_id=self.strategy_id, symbol=symbol,
                        signal_type=SignalType.SHORT_ENTRY,
                        strength=zone.strength, price=c,
                        reason=f"sdz_supply: zone=[{zone.low:.1f},{zone.high:.1f}] test#{zone.tests}",
                    )
                    signals.append(sig)
                    self.record_signal(sig)
                    self._position_side = "short"
                    self._entry_price = c
                    self._hold_bars = 0
                    break

        return signals

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        return []
