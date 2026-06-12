"""Adapter bridging Ch33's async signal-based strategies into Ch29's sync Strategy.

Ch33 strategies (BaseStrategy) are async and return Signal objects.
Ch29's BacktestEngine expects a sync Strategy that calls engine.buy()/sell().
This adapter translates signals into orders transparently.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from backtest.models import Bar
from backtest.strategy import Strategy
from strategy.base import BaseStrategy, Signal, SignalType

logger = logging.getLogger(__name__)

DEFAULT_ORDER_QTY_PCT = 0.02  # 2% of equity per signal


class BTCStrategyAdapter(Strategy):
    """Wraps a Ch33 BaseStrategy for use with Ch29's BacktestEngine.

    Signal-to-order translation:
      LONG_ENTRY  → engine.buy()
      LONG_EXIT   → engine.sell() (close long)
      SHORT_ENTRY → engine.sell()
      SHORT_EXIT  → engine.buy() (close short)
    """

    def __init__(
        self,
        inner: BaseStrategy,
        qty_pct: float = DEFAULT_ORDER_QTY_PCT,
        fixed_qty: int | None = None,
    ) -> None:
        super().__init__()
        self._inner = inner
        self._qty_pct = qty_pct
        self._fixed_qty = fixed_qty
        self._loop: asyncio.AbstractEventLoop | None = None

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        return self._loop

    def _run_async(self, coro: Any) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()

    def on_init(self) -> None:
        self._run_async(self._inner.on_start())

    def on_stop(self) -> None:
        self._run_async(self._inner.on_stop())

    def on_bar(self, bar: Bar) -> None:
        bar_dict = {
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": bar.volume,
            "timestamp": bar.dt,
        }

        signals: list[Signal] = self._run_async(
            self._inner.on_bar(bar.symbol, bar_dict)
        )

        for sig in signals:
            self._execute_signal(sig, bar)

    def _execute_signal(self, signal: Signal, bar: Bar) -> None:
        qty = self._compute_qty(signal, bar)
        if qty <= 0:
            return

        price = signal.price if signal.price else None

        if signal.signal_type == SignalType.LONG_ENTRY:
            self.buy(signal.symbol, qty, price)
            logger.debug("LONG_ENTRY %s qty=%d price=%s reason=%s", signal.symbol, qty, price, signal.reason)

        elif signal.signal_type == SignalType.SHORT_ENTRY:
            self.sell(signal.symbol, qty, price)
            logger.debug("SHORT_ENTRY %s qty=%d price=%s reason=%s", signal.symbol, qty, price, signal.reason)

        elif signal.signal_type == SignalType.LONG_EXIT:
            pos = self.get_position(signal.symbol)
            if pos and pos.long_volume > 0:
                self.sell(signal.symbol, pos.long_volume, price)
                logger.debug("LONG_EXIT %s qty=%d", signal.symbol, pos.long_volume)

        elif signal.signal_type == SignalType.SHORT_EXIT:
            pos = self.get_position(signal.symbol)
            if pos and pos.short_volume > 0:
                self.buy(signal.symbol, pos.short_volume, price)
                logger.debug("SHORT_EXIT %s qty=%d", signal.symbol, pos.short_volume)

    def _compute_qty(self, signal: Signal, bar: Bar) -> int:
        if self._fixed_qty is not None:
            return self._fixed_qty

        if signal.suggested_qty is not None and signal.suggested_qty > 0:
            return int(signal.suggested_qty)

        equity = float(self.engine.cash)
        price = signal.price or float(bar.close)
        if price <= 0:
            return 0

        notional = equity * self._qty_pct * signal.strength
        qty = round(notional / price, 6)
        return max(qty, 0)
