"""BTC 网格交易策略 - 在指定价格区间内按网格分档挂单。"""

from __future__ import annotations

import logging
from typing import Any

from ..base import BaseStrategy, Signal, SignalType, StrategyConfig
from ..registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "grid_upper": 100000.0,
    "grid_lower": 50000.0,
    "grid_count": 20,
    "order_qty_per_grid": 0.001,
    "rebalance_on_break": True,
}


class GridLevel:
    """单个网格档位。"""

    __slots__ = ("price", "is_filled", "side")

    def __init__(self, price: float) -> None:
        self.price = price
        self.is_filled = False
        self.side: str | None = None


@auto_register("btc_grid")
class BTCGridStrategy(BaseStrategy):
    """BTC 网格交易策略。

    核心逻辑：
    - 在 [grid_lower, grid_upper] 区间均匀划分 grid_count 个档位
    - 价格下穿网格线时买入，上穿时卖出
    - 破网时可选择重新定价
    """

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)
        self._grids: dict[str, list[GridLevel]] = {}
        self._last_price: dict[str, float] = {}
        self._active_upper: float = float(self.get_param("grid_upper"))
        self._active_lower: float = float(self.get_param("grid_lower"))

    def _init_grids(self, symbol: str) -> None:
        upper = self._active_upper
        lower = self._active_lower
        count = self.get_param("grid_count")
        if count <= 0:
            logger.warning("grid_count <= 0 for %s, skipping grid init", symbol)
            self._grids[symbol] = []
            return
        step = (upper - lower) / count
        self._grids[symbol] = [GridLevel(lower + i * step) for i in range(count + 1)]
        logger.info(
            "网格初始化: %s, range=[%.2f, %.2f], count=%d, step=%.2f",
            symbol, lower, upper, count, step,
        )

    def _find_crossed_grids(
        self, symbol: str, prev_price: float, curr_price: float
    ) -> list[tuple[GridLevel, str]]:
        """找出价格穿越的网格线及方向。"""
        crossed: list[tuple[GridLevel, str]] = []
        for level in self._grids.get(symbol, []):
            if prev_price > level.price >= curr_price:
                crossed.append((level, "down_cross"))
            elif prev_price < level.price <= curr_price:
                crossed.append((level, "up_cross"))
        return crossed

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        if symbol not in self._grids:
            self._init_grids(symbol)

        curr_price = bar["close"]
        prev_price = self._last_price.get(symbol, curr_price)
        self._last_price[symbol] = curr_price

        if prev_price == curr_price:
            return []

        upper = self._active_upper
        lower = self._active_lower
        if self.get_param("rebalance_on_break") and (curr_price > upper * 1.05 or curr_price < lower * 0.95):
            new_mid = curr_price
            half_range = (upper - lower) / 2
            self._active_upper = new_mid + half_range
            self._active_lower = new_mid - half_range
            self._init_grids(symbol)
            logger.info("网格重新定价: symbol=%s, new_center=%.2f", symbol, new_mid)
            return []

        crossed = self._find_crossed_grids(symbol, prev_price, curr_price)
        signals: list[Signal] = []
        qty = self.get_param("order_qty_per_grid")

        for level, direction in crossed:
            if direction == "down_cross" and not level.is_filled:
                sig = Signal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY,
                    strength=0.7,
                    price=level.price,
                    suggested_qty=qty,
                    reason=f"网格买入@{level.price:.2f}(下穿)",
                    metadata={"grid_price": level.price, "direction": direction},
                )
                level.is_filled = True
                level.side = "buy"
                signals.append(sig)
                self.record_signal(sig)

            elif direction == "up_cross" and level.is_filled and level.side == "buy":
                sig = Signal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    signal_type=SignalType.LONG_EXIT,
                    strength=0.7,
                    price=level.price,
                    suggested_qty=qty,
                    reason=f"网格卖出@{level.price:.2f}(上穿)",
                    metadata={"grid_price": level.price, "direction": direction},
                )
                level.is_filled = False
                level.side = None
                signals.append(sig)
                self.record_signal(sig)

        if signals:
            logger.info("网格触发: symbol=%s, signals=%d", symbol, len(signals))
        return signals

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        all_signals: list[Signal] = []
        for symbol in self.config.symbols:
            bar = market_data.get(symbol)
            if bar:
                sigs = await self.on_bar(symbol, bar)
                all_signals.extend(sigs)
        return all_signals

    def get_grid_status(self, symbol: str) -> list[dict[str, Any]]:
        """返回当前网格各档位状态。"""
        grids = self._grids.get(symbol, [])
        return [
            {"price": g.price, "is_filled": g.is_filled, "side": g.side}
            for g in grids
        ]
