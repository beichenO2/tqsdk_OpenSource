"""策略适配器 — 将 strategy.base.BaseStrategy 桥接到回测引擎的 Strategy 接口。

BaseStrategy 是信号驱动的（on_bar → list[Signal]），
回测引擎的 Strategy 是订单驱动的（on_bar → self.buy/sell）。
此适配器在两者之间转换，让所有 BaseStrategy 子类无修改即可回测。
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from .models import Bar
from .strategy import Strategy

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class StrategyAdapter(Strategy):
    """将 BaseStrategy 的 Signal 输出转为回测引擎的 buy/sell 调用。

    Parameters
    ----------
    base_strategy
        任何 strategy.base.BaseStrategy 子类的实例。
    default_volume
        信号未指定 suggested_qty 时的默认下单手数。
    """

    def __init__(self, base_strategy: Any, default_volume: int = 1) -> None:
        super().__init__()
        self._inner = base_strategy
        self._default_volume = default_volume
        self._loop: asyncio.AbstractEventLoop | None = None

    def _run_async(self, coro: Any) -> Any:
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        return self._loop.run_until_complete(coro)

    def on_init(self) -> None:
        self._run_async(self._inner.on_start())
        logger.info("StrategyAdapter: %s initialized", self._inner.name)

    def on_stop(self) -> None:
        self._run_async(self._inner.on_stop())

    def on_bar(self, bar: Bar) -> None:
        bar_dict: dict[str, Any] = {
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": bar.volume,
            "datetime": bar.dt.isoformat(),
            "symbol": bar.symbol,
        }
        if bar.open_interest:
            bar_dict["open_interest"] = bar.open_interest
        if bar.extra:
            bar_dict.update(bar.extra)

        signals = self._run_async(self._inner.on_bar(bar.symbol, bar_dict))

        bt_pos = self.get_position(bar.symbol)

        for sig in signals:
            vol = int(sig.suggested_qty) if sig.suggested_qty else self._default_volume
            sig_type = sig.signal_type.value

            if sig_type == "long_entry":
                if bt_pos and bt_pos.short_volume > 0:
                    self.buy(bar.symbol, bt_pos.short_volume)
                self.buy(bar.symbol, vol)
            elif sig_type == "short_entry":
                if bt_pos and bt_pos.long_volume > 0:
                    self.sell(bar.symbol, bt_pos.long_volume)
                self.sell(bar.symbol, vol)
            elif sig_type == "long_exit":
                if bt_pos and bt_pos.long_volume > 0:
                    self.sell(bar.symbol, bt_pos.long_volume)
            elif sig_type == "short_exit":
                if bt_pos and bt_pos.short_volume > 0:
                    self.buy(bar.symbol, bt_pos.short_volume)

            logger.debug(
                "Signal %s %s vol=%d @ %.2f | %s",
                sig_type, bar.symbol, vol, float(bar.close), sig.reason,
            )

            bt_pos = self.get_position(bar.symbol)

    def on_trade(self, trade: Any) -> None:
        pos = self.get_position(trade.symbol)
        if pos:
            from strategy.base import OrderSide as StrategySide, Position as StrategyPosition
            if pos.long_volume > 0:
                self._inner.update_position(StrategyPosition(
                    symbol=trade.symbol,
                    side=StrategySide.BUY,
                    qty=float(pos.long_volume),
                    avg_price=float(pos.long_avg_price),
                    unrealized_pnl=float(pos.unrealized_pnl),
                    realized_pnl=float(pos.realized_pnl),
                ))
            elif pos.short_volume > 0:
                self._inner.update_position(StrategyPosition(
                    symbol=trade.symbol,
                    side=StrategySide.SELL,
                    qty=float(pos.short_volume),
                    avg_price=float(pos.short_avg_price),
                    unrealized_pnl=float(pos.unrealized_pnl),
                    realized_pnl=float(pos.realized_pnl),
                ))
            else:
                self._inner.remove_position(trade.symbol)

        self._inner.on_fill(trade)
