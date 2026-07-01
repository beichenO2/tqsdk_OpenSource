"""BTC BrokerAdapter - 将 broker_crypto 桥接到 Ch27 的 ExecutionEngine。

实现 Ch27 定义的 BrokerAdapter 接口，底层通过 Ch31/Ch32 的
ExchangeAdapter 与真实交易所通信。
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Optional

from core.enums.direction import Direction, Offset
from core.enums.order_status import OrderStatus
from core.enums.order_type import OrderType
from core.models.order import Order
from core.models.position import Position
from execution.broker_adapter import BrokerAdapter

from broker_crypto.base import ExchangeAdapter
from broker_crypto.models import (
    Exchange as CryptoExchange,
    OrderRequest as CryptoOrderRequest,
    OrderResponse,
    Side,
    OrderType as CryptoOrderType,
)

logger = logging.getLogger(__name__)

_CORE_TO_CRYPTO_ORDER_TYPE = {
    OrderType.MARKET: CryptoOrderType.MARKET,
    OrderType.LIMIT: CryptoOrderType.LIMIT,
    OrderType.STOP: CryptoOrderType.STOP_MARKET,
    OrderType.STOP_LIMIT: CryptoOrderType.STOP_LIMIT,
}


class CryptoBrokerAdapter(BrokerAdapter):
    """加密货币 BrokerAdapter 实现。

    桥接 Ch27 执行引擎（BrokerAdapter 接口）和 Ch32 交易所层
    （ExchangeAdapter 接口），负责两套模型之间的转换。

    用法::

        from strategy.btc.broker_adapter import CryptoBrokerAdapter
        adapter = CryptoBrokerAdapter(exchange_adapter, default_exchange="weex")
        execution_engine.register_broker("crypto", adapter)
    """

    def __init__(
        self,
        exchange_adapter: ExchangeAdapter,
        default_exchange: str = "weex",
    ) -> None:
        self._adapter = exchange_adapter
        self._default_exchange = default_exchange
        self._connected = False
        self._order_map: dict[str, tuple[str, str]] = {}

    async def connect(self) -> None:
        await self._adapter.connect()
        self._connected = True
        logger.info("CryptoBrokerAdapter 已连接: exchange=%s", self._default_exchange)

    async def disconnect(self) -> None:
        await self._adapter.disconnect()
        self._connected = False
        logger.info("CryptoBrokerAdapter 已断开")

    async def is_connected(self) -> bool:
        return self._connected

    async def submit_order(
        self,
        symbol: str,
        direction: Direction,
        offset: Offset,
        price: Decimal,
        volume: int,
        strategy_id: str = "",
        order_type: OrderType = OrderType.LIMIT,
    ) -> Order:
        crypto_symbol = self._convert_symbol(symbol)
        side = Side.BUY if direction == Direction.LONG else Side.SELL
        crypto_ot = _CORE_TO_CRYPTO_ORDER_TYPE.get(
            order_type, CryptoOrderType.LIMIT
        )

        try:
            exchange_enum = CryptoExchange(self._default_exchange)
        except ValueError:
            exchange_enum = CryptoExchange.BINANCE

        request = CryptoOrderRequest(
            exchange=exchange_enum,
            symbol=crypto_symbol,
            side=side,
            order_type=crypto_ot,
            quantity=Decimal(str(volume)),
            price=price if price else None,
        )

        try:
            response: OrderResponse = await self._adapter.place_order(request)

            order = Order(
                symbol=symbol,
                direction=direction,
                offset=offset,
                price=price,
                volume=volume,
                status=OrderStatus.SUBMITTED,
                strategy_id=strategy_id,
                broker_order_id=response.order_id,
            )
            self._order_map[order.order_id] = (response.order_id, crypto_symbol)
            logger.info(
                "提交订单: %s %s qty=%d price=%s -> exchange_id=%s",
                crypto_symbol, side.value, volume, price, response.order_id,
            )
            return order

        except Exception as e:
            logger.error("下单失败: %s", e)
            return Order(
                symbol=symbol,
                direction=direction,
                offset=offset,
                price=price,
                volume=volume,
                status=OrderStatus.FAILED,
                strategy_id=strategy_id,
            )

    async def cancel_order(self, order_id: str) -> bool:
        mapping = self._order_map.get(order_id)
        if not mapping:
            logger.warning("无法取消订单: 未找到交易所映射, order_id=%s", order_id)
            return False

        exchange_id, crypto_symbol = mapping
        try:
            await self._adapter.cancel_order(exchange_id, crypto_symbol)
            logger.info("取消订单: order_id=%s, exchange_id=%s", order_id, exchange_id)
            return True
        except Exception as e:
            logger.error("取消订单失败: %s", e)
            return False

    async def query_order(self, order_id: str) -> Optional[Order]:
        mapping = self._order_map.get(order_id)
        if not mapping:
            return None

        exchange_id, crypto_symbol = mapping
        _STATUS_MAP = {
            "open": OrderStatus.SUBMITTED,
            "partially_filled": OrderStatus.PARTIAL_FILLED,
            "filled": OrderStatus.FILLED,
            "cancelled": OrderStatus.CANCELLED,
            "rejected": OrderStatus.FAILED,
            "pending": OrderStatus.PENDING,
        }

        try:
            resp: OrderResponse = await self._adapter.get_order(exchange_id, crypto_symbol)
            return Order(
                order_id=order_id,
                symbol=crypto_symbol,
                direction=Direction.LONG if resp.side == Side.BUY else Direction.SHORT,
                offset=Offset.OPEN,
                price=resp.price or Decimal(0),
                volume=int(resp.quantity),
                filled_volume=int(resp.filled_quantity),
                avg_fill_price=resp.avg_fill_price or Decimal(0),
                status=_STATUS_MAP.get(resp.status.value, OrderStatus.PENDING),
                broker_order_id=exchange_id,
            )
        except Exception as e:
            logger.error("查询订单失败: %s", e)
            return None

    async def query_positions(self) -> list[Position]:
        try:
            crypto_positions = await self._adapter.get_positions()
            result: list[Position] = []
            for cp in crypto_positions:
                result.append(Position(
                    symbol=cp.symbol,
                    direction=Direction.LONG if cp.side == Side.BUY else Direction.SHORT,
                    volume=int(cp.quantity),
                    avg_price=cp.entry_price,
                    unrealized_pnl=cp.unrealized_pnl,
                ))
            return result
        except Exception as e:
            logger.error("查询持仓失败: %s", e)
            return []

    async def get_account_info(self) -> dict:
        try:
            balances = await self._adapter.get_balances()
            return {
                "exchange": self._default_exchange,
                "balances": [
                    {
                        "asset": b.asset,
                        "free": str(b.free),
                        "locked": str(b.locked),
                        "total": str(b.total),
                    }
                    for b in balances
                ],
            }
        except Exception as e:
            logger.error("查询账户失败: %s", e)
            return {"exchange": self._default_exchange, "balances": [], "error": str(e)}

    @staticmethod
    def _convert_symbol(instrument: str) -> str:
        """将内部合约代码转换为交易所格式。BINANCE.BTCUSDT -> BTC/USDT"""
        parts = instrument.split(".")
        raw = parts[-1] if len(parts) > 1 else instrument
        for quote in ("USDT", "BUSD", "USD", "BTC", "ETH"):
            if raw.upper().endswith(quote) and len(raw) > len(quote):
                base = raw[: -len(quote)]
                return f"{base.upper()}/{quote.upper()}"
        return raw.upper()
