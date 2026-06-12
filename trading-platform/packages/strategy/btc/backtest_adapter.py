"""Convenience wrapper: load a registered strategy by name for BTCBacktestEngine.

`BTCBacktestEngine.set_strategy` expects `strategy.base.BaseStrategy` (file: `packages/strategy/base.py`).
This class is itself a `BaseStrategy` that delegates to the registry-created inner.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from btc.models.types import Fill
from ..base import BaseStrategy, Signal, StrategyConfig
from ..registry import StrategyRegistry

logger = logging.getLogger(__name__)


class BacktestStrategyAdapter(BaseStrategy):
    """Delegates to `StrategyRegistry.create(strategy_name, config)`.

    Usage:
        adapter = BacktestStrategyAdapter(
            "btc_momentum",
            StrategyConfig(name="momentum_bt", symbols=["BTCUSDT"]),
        )
        engine.set_strategy(adapter)
        result = engine.run()
    """

    def __init__(
        self,
        strategy_name: str,
        config: StrategyConfig | None = None,
    ) -> None:
        cfg = config or StrategyConfig(name=strategy_name, symbols=["BTCUSDT"])
        super().__init__(cfg)
        self._strategy_name = strategy_name
        self._inner: BaseStrategy | None = None
        self._fill_history: deque[Fill] = deque(maxlen=500)

    def _ensure_inner(self) -> BaseStrategy:
        if self._inner is None:
            self._inner = StrategyRegistry.create(self._strategy_name, self.config)
        return self._inner

    async def on_start(self) -> None:
        inner = self._ensure_inner()
        await inner.on_start()
        await super().on_start()

    async def on_stop(self) -> None:
        if self._inner is not None:
            await self._inner.on_stop()
        await super().on_stop()

    def update_position(self, position: Any) -> None:
        super().update_position(position)
        if self._inner is not None:
            self._inner.update_position(position)

    def remove_position(self, symbol: str) -> None:
        super().remove_position(symbol)
        if self._inner is not None:
            self._inner.remove_position(symbol)

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        inner = self._ensure_inner()
        return await inner.on_bar(symbol, bar)

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        inner = self._ensure_inner()
        return await inner.generate_signals(market_data)

    def on_fill(self, fill: Any) -> None:
        if isinstance(fill, Fill):
            self._fill_history.append(fill)
        inner = self._inner
        if inner is not None:
            inner.on_fill(fill)

    def on_backtest_complete(self, result: Any) -> None:
        inner = self._inner
        if inner is not None:
            inner.on_backtest_complete(result)
        logger.info(
            "回测完成: strategy=%s, fills=%d",
            self._strategy_name,
            len(self._fill_history),
        )
