"""Helpers to run unified BaseStrategy inside the BTC backtest engine."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from strategy.base import (
    BaseStrategy,
    OrderSide as StratOrderSide,
    Position as StratPosition,
    Signal,
    SignalType,
)

if TYPE_CHECKING:
    from .models.types import OHLCV, Order as BTOrder, Position as BTPosition

logger = logging.getLogger(__name__)


def run_async(coro: Any) -> Any:
    """Run *coro* synchronously (backtest has no running event loop)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def ohlcv_to_bar_dict(bar: OHLCV) -> dict[str, Any]:
    return {
        "open": float(bar.open),
        "high": float(bar.high),
        "low": float(bar.low),
        "close": float(bar.close),
        "volume": float(bar.volume),
        "timestamp": bar.timestamp,
    }


def sync_bt_positions_to_strategy(
    strategy: BaseStrategy,
    positions: dict[str, BTPosition],
) -> None:
    """Mirror simulated exchange positions into the strategy's position cache."""

    seen: set[str] = set()
    for symbol, p in positions.items():
        seen.add(symbol)
        if p.is_flat:
            strategy.remove_position(symbol)
            continue
        side = StratOrderSide.BUY if p.is_long else StratOrderSide.SELL
        strat_pos = StratPosition(
            symbol=symbol,
            side=side,
            qty=float(abs(p.quantity)),
            avg_price=float(p.avg_entry_price),
            unrealized_pnl=float(p.unrealized_pnl),
            realized_pnl=float(p.realized_pnl),
        )
        strategy.update_position(strat_pos)

    for sym in list(strategy.get_all_positions().keys()):
        if sym not in seen:
            strategy.remove_position(sym)


def signals_to_btc_orders(
    signals: list[Signal],
    positions: dict[str, BTPosition],
    equity: float,
    current_price: float,
    *,
    position_size_pct: float,
    max_open_orders: int,
) -> list[BTOrder]:
    """Translate strategy signals into BTC simulated exchange orders."""
    from .models.types import Order as BTOrderModel
    from .models.types import OrderSide as BTOrderSide
    from .models.types import OrderType as BTOrderType

    orders: list[BTOrderModel] = []

    def calc_qty() -> float:
        if current_price <= 0:
            return 0.0
        notional = equity * position_size_pct
        return round(notional / current_price, 6)

    for sig in signals:
        pos = positions.get(sig.symbol)

        if sig.signal_type == SignalType.LONG_ENTRY:
            if pos is not None and pos.is_long:
                continue
            qty = calc_qty()
            if qty <= 0:
                continue
            orders.append(
                BTOrderModel(
                    symbol=sig.symbol,
                    side=BTOrderSide.BUY,
                    order_type=BTOrderType.MARKET,
                    quantity=Decimal(str(qty)),
                    metadata={"signal_id": sig.signal_id, "reason": sig.reason},
                )
            )

        elif sig.signal_type == SignalType.SHORT_ENTRY:
            if pos is not None and pos.is_short:
                continue
            qty = calc_qty()
            if qty <= 0:
                continue
            orders.append(
                BTOrderModel(
                    symbol=sig.symbol,
                    side=BTOrderSide.SELL,
                    order_type=BTOrderType.MARKET,
                    quantity=Decimal(str(qty)),
                    metadata={"signal_id": sig.signal_id, "reason": sig.reason},
                )
            )

        elif sig.signal_type == SignalType.LONG_EXIT:
            if pos is None or pos.is_flat or pos.is_short:
                continue
            exit_qty = abs(pos.quantity)
            if sig.suggested_qty is not None and 0 < Decimal(str(sig.suggested_qty)) < exit_qty:
                exit_qty = Decimal(str(round(float(sig.suggested_qty), 6)))
            orders.append(
                BTOrderModel(
                    symbol=sig.symbol,
                    side=BTOrderSide.SELL,
                    order_type=BTOrderType.MARKET,
                    quantity=exit_qty,
                    metadata={"signal_id": sig.signal_id, "reason": sig.reason},
                )
            )

        elif sig.signal_type == SignalType.SHORT_EXIT:
            if pos is None or pos.is_flat or pos.is_long:
                continue
            exit_qty = abs(pos.quantity)
            if sig.suggested_qty is not None and 0 < Decimal(str(sig.suggested_qty)) < exit_qty:
                exit_qty = Decimal(str(round(float(sig.suggested_qty), 6)))
            orders.append(
                BTOrderModel(
                    symbol=sig.symbol,
                    side=BTOrderSide.BUY,
                    order_type=BTOrderType.MARKET,
                    quantity=exit_qty,
                    metadata={"signal_id": sig.signal_id, "reason": sig.reason},
                )
            )

    hedge_orders = _build_hedge_orders(signals, positions, equity, current_price, position_size_pct=position_size_pct)
    orders.extend(hedge_orders)

    return orders[:max_open_orders]


def _build_hedge_orders(
    signals: list[Signal],
    positions: dict[str, Any],
    equity: float,
    current_price: float,
    *,
    position_size_pct: float,
    last_prices: dict[str, float] | None = None,
) -> list:
    """Generate hedge-leg orders from signal metadata (pairs/spread strategies)."""
    from .models.types import Order as BTOrderModel
    from .models.types import OrderSide as BTOrderSide
    from .models.types import OrderType as BTOrderType

    hedge_orders = []
    for sig in signals:
        meta = sig.metadata or {}
        hedge_sym = meta.get("hedge_symbol")
        hedge_side_str = meta.get("hedge_side")
        if not hedge_sym or not hedge_side_str:
            continue

        beta = float(meta.get("beta", 1.0))

        if sig.signal_type in (SignalType.LONG_ENTRY, SignalType.SHORT_ENTRY):
            hedge_price = (last_prices or {}).get(hedge_sym, current_price)
            if hedge_price <= 0:
                continue
            notional = equity * position_size_pct
            qty = round(notional / hedge_price * beta, 6)
            if qty <= 0:
                continue
            side = BTOrderSide.BUY if hedge_side_str == "buy" else BTOrderSide.SELL
            hedge_orders.append(
                BTOrderModel(
                    symbol=hedge_sym,
                    side=side,
                    order_type=BTOrderType.MARKET,
                    quantity=Decimal(str(qty)),
                    metadata={"hedge_for": sig.signal_id, "reason": f"hedge: {sig.reason}"},
                )
            )

        elif sig.signal_type in (SignalType.LONG_EXIT, SignalType.SHORT_EXIT):
            hedge_pos = positions.get(hedge_sym)
            if hedge_pos is None or hedge_pos.is_flat:
                continue
            close_side = BTOrderSide.SELL if hedge_pos.is_long else BTOrderSide.BUY
            hedge_orders.append(
                BTOrderModel(
                    symbol=hedge_sym,
                    side=close_side,
                    order_type=BTOrderType.MARKET,
                    quantity=abs(hedge_pos.quantity),
                    metadata={"hedge_for": sig.signal_id, "reason": f"hedge close: {sig.reason}"},
                )
            )

    return hedge_orders
