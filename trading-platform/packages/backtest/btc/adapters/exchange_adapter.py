"""Simulated exchange adapter implementing Ch31's ExchangeAdapter interface.

Enables strategy code to run identically in backtest and live mode by
providing the same ExchangeAdapter contract backed by the simulated exchange.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import AsyncIterator

from broker_crypto.base import ExchangeAdapter
from broker_crypto.models import (
    Balance,
    Exchange,
    ExchangeCredentials,
    OHLCV,
    OrderBook,
    OrderRequest,
    OrderResponse,
    OrderStatus,
    Position as CryptoPosition,
    Side,
    OrderType as CryptoOrderType,
    Ticker,
    Trade as CryptoTrade,
)

from ..exchange.simulated import SimulatedExchange
from ..models.types import (
    Order,
    OrderSide,
    OrderType,
)

logger = logging.getLogger(__name__)


class SimulatedExchangeAdapter(ExchangeAdapter):
    """Paper-trading adapter wrapping the BTC SimulatedExchange.

    Implements the ExchangeAdapter contract so strategies can switch between
    real and simulated exchanges with zero code changes.
    """

    def __init__(
        self,
        credentials: ExchangeCredentials,
        initial_capital: Decimal = Decimal("100000"),
        commission_rate: Decimal = Decimal("0.001"),
        slippage_bps: Decimal = Decimal("5"),
    ) -> None:
        super().__init__(credentials)
        self._sim = SimulatedExchange(
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            slippage_bps=slippage_bps,
        )
        self._connected = False
        self._kline_data: dict[str, list[OHLCV]] = {}
        self._leverage: dict[str, int] = {}

    @property
    def exchange(self) -> Exchange:
        return self._credentials.exchange

    @property
    def sim_exchange(self) -> SimulatedExchange:
        return self._sim

    # ── Lifecycle ────────────────────────────────────────────────────

    async def connect(self) -> None:
        self._connected = True
        logger.info("SimulatedExchangeAdapter connected (paper mode)")

    async def disconnect(self) -> None:
        self._connected = False

    # ── Market Data (REST) ───────────────────────────────────────────

    async def get_ticker(self, symbol: str) -> Ticker:
        raise NotImplementedError("Simulated adapter does not provide live ticker")

    async def get_orderbook(self, symbol: str, depth: int = 20) -> OrderBook:
        raise NotImplementedError("Simulated adapter does not provide order book")

    async def get_ohlcv(
        self, symbol: str, interval: str = "1m", limit: int = 500,
    ) -> list[OHLCV]:
        data = self._kline_data.get(symbol, [])
        return data[-limit:]

    async def get_recent_trades(self, symbol: str, limit: int = 100) -> list[CryptoTrade]:
        fills = [f for f in self._sim.fills if f.symbol == symbol]
        now = datetime.now(timezone.utc)
        return [
            CryptoTrade(
                exchange=self._credentials.exchange,
                symbol=f.symbol,
                trade_id=f.order_id,
                side=Side.BUY if f.side == OrderSide.BUY else Side.SELL,
                price=f.price,
                quantity=f.quantity,
                timestamp=getattr(f, "timestamp", now),
            )
            for f in fills[-limit:]
        ]

    # ── Market Data (WebSocket) — not applicable for backtest ────────

    async def stream_ticker(self, symbol: str) -> AsyncIterator[Ticker]:
        raise NotImplementedError("Simulated adapter does not support WebSocket streams")
        yield  # type: ignore[misc]  # make it a generator

    async def stream_orderbook(self, symbol: str) -> AsyncIterator[OrderBook]:
        raise NotImplementedError("Simulated adapter does not support WebSocket streams")
        yield  # type: ignore[misc]

    async def stream_trades(self, symbol: str) -> AsyncIterator[CryptoTrade]:
        raise NotImplementedError("Simulated adapter does not support WebSocket streams")
        yield  # type: ignore[misc]

    async def stream_ohlcv(
        self, symbol: str, interval: str = "1m",
    ) -> AsyncIterator[OHLCV]:
        raise NotImplementedError("Simulated adapter does not support WebSocket streams")
        yield  # type: ignore[misc]

    # ── Trading ──────────────────────────────────────────────────────

    async def place_order(self, request: OrderRequest) -> OrderResponse:
        bt_side = OrderSide.BUY if request.side == Side.BUY else OrderSide.SELL
        bt_type = _map_order_type(request.order_type)

        order = Order(
            id=request.client_order_id or uuid.uuid4().hex[:12],
            symbol=request.symbol,
            side=bt_side,
            order_type=bt_type,
            quantity=request.quantity,
            price=request.price,
        )
        self._sim.submit_order(order)

        now = datetime.now(timezone.utc)
        return OrderResponse(
            exchange=self._credentials.exchange,
            order_id=order.id,
            client_order_id=request.client_order_id,
            symbol=request.symbol,
            side=request.side,
            order_type=request.order_type,
            status=OrderStatus(order.status.value),
            quantity=request.quantity,
            created_at=now,
            updated_at=now,
        )

    async def cancel_order(self, order_id: str, symbol: str) -> OrderResponse:
        self._sim.cancel_order(order_id)
        now = datetime.now(timezone.utc)
        return OrderResponse(
            exchange=self._credentials.exchange,
            order_id=order_id,
            symbol=symbol,
            side=Side.BUY,
            order_type=CryptoOrderType.LIMIT,
            status=OrderStatus.CANCELLED,
            quantity=Decimal(0),
            created_at=now,
            updated_at=now,
        )

    async def get_order(self, order_id: str, symbol: str) -> OrderResponse:
        raise NotImplementedError("get_order not implemented for simulated adapter")

    async def get_open_orders(self, symbol: str | None = None) -> list[OrderResponse]:
        return []

    # ── Account ──────────────────────────────────────────────────────

    async def get_balances(self) -> list[Balance]:
        return [
            Balance(
                exchange=self._credentials.exchange,
                asset="USDT",
                free=self._sim.cash,
                locked=Decimal(0),
            )
        ]

    async def get_positions(self, symbol: str | None = None) -> list[CryptoPosition]:
        result: list[CryptoPosition] = []
        now = datetime.now(timezone.utc)
        for sym, pos in self._sim.positions.items():
            if symbol and sym != symbol:
                continue
            if pos.is_flat:
                continue
            result.append(
                CryptoPosition(
                    exchange=self._credentials.exchange,
                    symbol=sym,
                    side=Side.BUY if pos.is_long else Side.SELL,
                    quantity=abs(pos.quantity),
                    entry_price=pos.avg_entry_price,
                    unrealized_pnl=pos.unrealized_pnl,
                    leverage=self._leverage.get(sym, 1),
                    timestamp=now,
                )
            )
        return result

    # ── Backtest helpers (not part of ExchangeAdapter contract) ──────

    def preload_klines(self, symbol: str, klines: list[OHLCV]) -> None:
        self._kline_data[symbol] = klines

    async def set_leverage(self, symbol: str, leverage: int) -> None:
        self._leverage[symbol] = leverage


def _map_order_type(ct: CryptoOrderType) -> OrderType:
    mapping = {
        CryptoOrderType.MARKET: OrderType.MARKET,
        CryptoOrderType.LIMIT: OrderType.LIMIT,
        CryptoOrderType.STOP_LIMIT: OrderType.STOP_LIMIT,
        CryptoOrderType.STOP_MARKET: OrderType.STOP,
    }
    return mapping.get(ct, OrderType.MARKET)
