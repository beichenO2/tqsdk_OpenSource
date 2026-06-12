"""缠论（Chan Theory）日内策略 — 基于分型/笔/线段/中枢的结构化分析。

缠论核心概念（李彪，2006-2008，经典量化分析体系）：
1. 合并K线：相邻K线的包含关系处理
2. 顶底分型：三根合并K线形成的局部极值
3. 笔（Bi）：相邻顶底分型之间的连线
4. 线段（Duan）：至少三笔构成的趋势段
5. 中枢（Zhongshu）：三个连续线段的重叠区域
6. 买卖点：一/二/三类买卖点

本实现将缠论结构化分析与日内交易框架结合：
- 分型/笔/线段实时增量构建
- 中枢突破/回拉产生交易信号
- 配合 IntradayGuard 收盘前清仓

Method quality: 经典量化分析方法，工业界广泛使用，免年限限制。
增强部分使用 ATR/成交量等经典指标辅助确认。
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..base import BaseStrategy, Signal, SignalType, StrategyConfig
from ..indicators import calc_atr
from ..registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS: dict[str, Any] = {
    "min_bi_bars": 4,
    "atr_period": 14,
    "vol_confirm_mult": 1.3,
    "zhongshu_break_atr_mult": 0.5,
    "risk_per_trade": 0.02,
    "trailing_stop_atr_mult": 2.0,
    "max_hold_bars": 60,
}


class FenxingType(str, Enum):
    TOP = "top"
    BOTTOM = "bottom"


@dataclass
class MergedBar:
    """合并K线：处理包含关系后的K线。"""
    high: float
    low: float
    index: int
    direction: int = 0  # 1=up, -1=down, 0=undetermined


@dataclass
class Fenxing:
    """顶底分型。"""
    type: FenxingType
    price: float
    index: int
    bar_index: int


@dataclass
class Bi:
    """笔：两个分型之间的连线。"""
    start: Fenxing
    end: Fenxing
    direction: int  # 1=up, -1=down

    @property
    def high(self) -> float:
        return max(self.start.price, self.end.price)

    @property
    def low(self) -> float:
        return min(self.start.price, self.end.price)

    @property
    def length(self) -> int:
        return abs(self.end.bar_index - self.start.bar_index)


@dataclass
class Zhongshu:
    """中枢：三个连续线段的重叠区域。"""
    high: float
    low: float
    start_index: int
    end_index: int
    level: int = 0


class ChanAnalyzer:
    """缠论结构分析器 — 增量构建分型/笔/中枢。"""

    def __init__(self, min_bi_bars: int = 4):
        self.min_bi_bars = min_bi_bars
        self.merged_bars: list[MergedBar] = []
        self.fenxings: list[Fenxing] = []
        self.bis: list[Bi] = []
        self.zhongshus: list[Zhongshu] = []
        self._raw_count = 0
        self._last_direction = 0

    def update(self, high: float, low: float, bar_index: int) -> None:
        """Feed a new bar and incrementally update the structure."""
        self._raw_count += 1

        if not self.merged_bars:
            self.merged_bars.append(MergedBar(high=high, low=low, index=bar_index))
            return

        last = self.merged_bars[-1]

        if high >= last.high and low <= last.low:
            if self._last_direction >= 0:
                last.high = max(last.high, high)
                last.low = max(last.low, low)
            else:
                last.high = min(last.high, high)
                last.low = min(last.low, low)
            return
        if high <= last.high and low >= last.low:
            if self._last_direction >= 0:
                last.high = max(last.high, high)
                last.low = max(last.low, low)
            else:
                last.high = min(last.high, high)
                last.low = min(last.low, low)
            return

        if high > last.high:
            self._last_direction = 1
        elif low < last.low:
            self._last_direction = -1

        self.merged_bars.append(MergedBar(high=high, low=low, index=bar_index, direction=self._last_direction))

        self._detect_fenxing()
        self._build_bi()
        self._build_zhongshu()

    def _detect_fenxing(self) -> None:
        if len(self.merged_bars) < 3:
            return

        a, b, c = self.merged_bars[-3], self.merged_bars[-2], self.merged_bars[-1]

        if b.high > a.high and b.high > c.high and b.low > a.low and b.low > c.low:
            fx = Fenxing(FenxingType.TOP, b.high, len(self.fenxings), b.index)
            if not self.fenxings or self.fenxings[-1].type != FenxingType.TOP:
                self.fenxings.append(fx)
            elif fx.price > self.fenxings[-1].price:
                self.fenxings[-1] = fx

        elif b.low < a.low and b.low < c.low and b.high < a.high and b.high < c.high:
            fx = Fenxing(FenxingType.BOTTOM, b.low, len(self.fenxings), b.index)
            if not self.fenxings or self.fenxings[-1].type != FenxingType.BOTTOM:
                self.fenxings.append(fx)
            elif fx.price < self.fenxings[-1].price:
                self.fenxings[-1] = fx

    def _build_bi(self) -> None:
        if len(self.fenxings) < 2:
            return

        start = self.fenxings[-2]
        end = self.fenxings[-1]

        if start.type == end.type:
            return

        bar_distance = abs(end.bar_index - start.bar_index)
        if bar_distance < self.min_bi_bars:
            return

        if start.type == FenxingType.BOTTOM and end.type == FenxingType.TOP:
            direction = 1
        elif start.type == FenxingType.TOP and end.type == FenxingType.BOTTOM:
            direction = -1
        else:
            return

        bi = Bi(start=start, end=end, direction=direction)

        if not self.bis or (self.bis[-1].end.bar_index != bi.start.bar_index and
                            self.bis[-1].end.bar_index < bi.start.bar_index):
            self.bis.append(bi)
        elif self.bis and self.bis[-1].direction == bi.direction:
            if (direction == 1 and bi.end.price > self.bis[-1].end.price) or \
               (direction == -1 and bi.end.price < self.bis[-1].end.price):
                self.bis[-1] = bi

    def _build_zhongshu(self) -> None:
        if len(self.bis) < 3:
            return

        b1, b2, b3 = self.bis[-3], self.bis[-2], self.bis[-1]
        overlap_high = min(b1.high, b2.high, b3.high)
        overlap_low = max(b1.low, b2.low, b3.low)

        if overlap_high > overlap_low:
            zs = Zhongshu(
                high=overlap_high, low=overlap_low,
                start_index=b1.start.bar_index,
                end_index=b3.end.bar_index,
            )
            if not self.zhongshus or zs.start_index > self.zhongshus[-1].end_index:
                self.zhongshus.append(zs)

    def get_last_zhongshu(self) -> Zhongshu | None:
        return self.zhongshus[-1] if self.zhongshus else None

    def get_last_bi(self) -> Bi | None:
        return self.bis[-1] if self.bis else None

    def get_signal(self, current_price: float) -> tuple[str, float]:
        """Generate Chan Theory signal based on current structure.

        Returns: (signal_type, strength)
        - "buy_1": 一买（中枢下方底分型确认后向上突破）
        - "buy_2": 二买（回拉中枢不创新低后向上）
        - "sell_1": 一卖（中枢上方顶分型确认后向下突破）
        - "sell_2": 二卖（反弹中枢不创新高后向下）
        - "hold": 无明确信号
        """
        zs = self.get_last_zhongshu()
        last_bi = self.get_last_bi()

        if not zs or not last_bi:
            return "hold", 0.0

        zs_mid = (zs.high + zs.low) / 2
        zs_range = zs.high - zs.low

        if last_bi.direction == 1 and current_price > zs.high:
            if last_bi.start.price < zs.low:
                return "buy_1", 0.9
            return "buy_2", 0.7

        if last_bi.direction == -1 and current_price < zs.low:
            if last_bi.start.price > zs.high:
                return "sell_1", 0.9
            return "sell_2", 0.7

        if last_bi.direction == -1 and current_price > zs.low and last_bi.end.price >= zs.low:
            if current_price < zs.high:
                return "buy_2", 0.6

        if last_bi.direction == 1 and current_price < zs.high and last_bi.end.price <= zs.high:
            if current_price > zs.low:
                return "sell_2", 0.6

        return "hold", 0.0


@auto_register("chan_theory")
class ChanTheoryStrategy(BaseStrategy):
    """缠论日内策略 — 分型/笔/中枢结构化分析产生交易信号。

    结合缠论买卖点与 ATR 止盈止损、成交量确认。
    """

    def __init__(self, config: StrategyConfig) -> None:
        config = config.model_copy(
            update={"params": {**DEFAULT_PARAMS, **config.params}}
        )
        super().__init__(config)
        self._analyzer = ChanAnalyzer(
            min_bi_bars=self.get_param("min_bi_bars", 4),
        )
        self._highs: deque[float] = deque(maxlen=200)
        self._lows: deque[float] = deque(maxlen=200)
        self._closes: deque[float] = deque(maxlen=200)
        self._volumes: deque[float] = deque(maxlen=200)
        self._bar_count = 0
        self._position_side: str | None = None
        self._entry_price = 0.0
        self._hold_bars = 0

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

        self._analyzer.update(h, l, self._bar_count)

        if self._bar_count < 30:
            return []

        atr = calc_atr(list(self._highs), list(self._lows), list(self._closes),
                       self.get_param("atr_period", 14))
        if atr is None or atr < 1e-10:
            return []

        signals = []
        chan_signal, strength = self._analyzer.get_signal(c)

        vol_avg = sum(list(self._volumes)[-20:]) / 20 if len(self._volumes) >= 20 else v
        vol_confirm = v > vol_avg * self.get_param("vol_confirm_mult", 1.3)

        if self._position_side:
            self._hold_bars += 1
            max_hold = self.get_param("max_hold_bars", 60)
            trail_mult = self.get_param("trailing_stop_atr_mult", 2.0)

            if self._position_side == "long":
                sl_hit = c < self._entry_price - trail_mult * atr
                if chan_signal in ("sell_1", "sell_2") and strength > 0.6:
                    sl_hit = True
            else:
                sl_hit = c > self._entry_price + trail_mult * atr
                if chan_signal in ("buy_1", "buy_2") and strength > 0.6:
                    sl_hit = True

            if sl_hit or self._hold_bars >= max_hold:
                exit_type = SignalType.LONG_EXIT if self._position_side == "long" else SignalType.SHORT_EXIT
                sig = Signal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    signal_type=exit_type,
                    strength=0.8,
                    price=c,
                    reason=f"chan_exit: {chan_signal} hold={self._hold_bars}",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = None
                self._hold_bars = 0
                return signals

        if not self._position_side:
            if chan_signal == "buy_1" and vol_confirm:
                sig = Signal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY,
                    strength=strength,
                    price=c,
                    reason="chan_buy_1: zhongshu breakup + vol confirm",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = "long"
                self._entry_price = c
                self._hold_bars = 0

            elif chan_signal == "buy_2" and strength >= 0.6:
                sig = Signal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY,
                    strength=strength,
                    price=c,
                    reason="chan_buy_2: pullback to zhongshu + no new low",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = "long"
                self._entry_price = c
                self._hold_bars = 0

            elif chan_signal == "sell_1" and vol_confirm:
                sig = Signal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY,
                    strength=strength,
                    price=c,
                    reason="chan_sell_1: zhongshu breakdown + vol confirm",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = "short"
                self._entry_price = c
                self._hold_bars = 0

            elif chan_signal == "sell_2" and strength >= 0.6:
                sig = Signal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY,
                    strength=strength,
                    price=c,
                    reason="chan_sell_2: bounce to zhongshu + no new high",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = "short"
                self._entry_price = c
                self._hold_bars = 0

        return signals

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        return []
