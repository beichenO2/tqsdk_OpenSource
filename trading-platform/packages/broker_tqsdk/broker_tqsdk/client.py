"""TqSdk Broker Client — 封装 TqSdk API 为统一的交易接口."""

from __future__ import annotations

import asyncio
import logging
from decimal import Decimal
from typing import Any

from core.enums.direction import Direction, Offset
from core.models.position import Position

logger = logging.getLogger(__name__)


class TqBrokerClient:
    """封装 TqSdk 的 TqApi，提供异步友好的接口.

    生命周期:
        async with TqBrokerClient(auth, account) as client:
            await client.place_order(...)
    """

    def __init__(
        self,
        auth_email: str | None = None,
        auth_password: str | None = None,
        broker_id: str | None = None,
        account_id: str | None = None,
        td_url: str | None = None,
    ) -> None:
        self._auth_email = auth_email
        self._auth_password = auth_password
        self._broker_id = broker_id
        self._account_id = account_id
        self._td_url = td_url
        self._api: Any = None
        self._account: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def tqsdk_api(self) -> Any:
        """Underlying TqApi instance for market data (TqMarketAdapter)."""
        return self._api

    async def connect(self, max_retries: int = 2, retry_delay: float = 5.0) -> None:
        """建立 TqSdk 连接 (with retry on timeout)."""
        try:
            from tqsdk import TqApi, TqAuth, TqAccount, TqSim
        except ImportError:
            logger.warning(
                "⚠️ tqsdk NOT installed — STUB MODE active. "
                "All orders will return 'stub-order-id' and NOT reach any exchange."
            )
            self._api = None
            return

        auth = TqAuth(self._auth_email, self._auth_password) if self._auth_email else None

        if self._broker_id and self._account_id:
            self._account = TqAccount(self._broker_id, self._account_id, self._auth_password or "")
        else:
            self._account = TqSim(init_balance=1_000_000)

        last_err: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                self._api = TqApi(account=self._account, auth=auth)
                self._loop = asyncio.get_event_loop()
                logger.info("TqSdk connected (account=%s, attempt=%d)", type(self._account).__name__, attempt)
                return
            except Exception as exc:
                last_err = exc
                logger.warning(
                    "TqSdk connect attempt %d/%d failed: %s",
                    attempt, max_retries, exc,
                )
                if attempt < max_retries:
                    await asyncio.sleep(retry_delay)

        logger.warning(
            "⚠️ TqSdk connection failed after %d attempts — falling back to STUB MODE. "
            "Last error: %s",
            max_retries, last_err,
        )
        self._api = None

    async def disconnect(self) -> None:
        """关闭 TqSdk 连接."""
        if self._api is not None:
            self._api.close()
            self._api = None
            logger.info("TqSdk disconnected")

    async def __aenter__(self) -> TqBrokerClient:
        await self.connect()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.disconnect()

    async def place_order(
        self,
        symbol: str,
        direction: Direction,
        offset: Offset,
        price: Decimal,
        volume: int,
    ) -> str:
        """下单，返回 order_id."""
        if self._api is None:
            logger.warning(
                "⚠️ STUB: place_order(%s %s %s vol=%d) → stub-order-id (NOT sent to exchange)",
                symbol, direction.name, offset.name, volume,
            )
            return "stub-order-id"

        tq_direction = "BUY" if direction == Direction.LONG else "SELL"
        tq_offset = offset.value
        order = self._api.insert_order(
            symbol=symbol,
            direction=tq_direction,
            offset=tq_offset,
            limit_price=float(price),
            volume=volume,
        )
        return order.order_id

    async def cancel_order(self, order_id: str) -> bool:
        """撤单."""
        if self._api is None:
            logger.warning("⚠️ STUB: cancel_order(%s) — no exchange connection", order_id)
            return False
        self._api.cancel_order(order_id)
        return True

    async def get_positions(self) -> list[Position]:
        """获取全部持仓."""
        if self._api is None:
            logger.debug("STUB: get_positions() → empty (no exchange connection)")
            return []
        # TqSdk positions are keyed by symbol
        result: list[Position] = []
        positions = self._api.get_position()
        for symbol, pos in positions.items():
            if pos.pos_long > 0:
                result.append(Position(
                    symbol=symbol,
                    exchange=_extract_exchange(symbol),
                    direction=Direction.LONG,
                    volume=pos.pos_long,
                    available=pos.pos_long - pos.pos_long_his,
                    float_pnl=Decimal(str(pos.float_profit_long)),
                ))
            if pos.pos_short > 0:
                result.append(Position(
                    symbol=symbol,
                    exchange=_extract_exchange(symbol),
                    direction=Direction.SHORT,
                    volume=pos.pos_short,
                    available=pos.pos_short - pos.pos_short_his,
                    float_pnl=Decimal(str(pos.float_profit_short)),
                ))
        return result

    async def get_account_info(self) -> dict[str, Any]:
        """获取账户资金信息."""
        if self._api is None:
            logger.debug("STUB: get_account_info() → zeros (no exchange connection)")
            return {"balance": 0, "available": 0, "margin": 0}
        acc = self._api.get_account()
        return {
            "balance": acc.balance,
            "available": acc.available,
            "margin": acc.margin,
            "float_profit": acc.float_profit,
            "commission": acc.commission,
        }


def _extract_exchange(symbol: str) -> str:
    """从 TqSdk 合约格式 'SHFE.cu2401' 提取交易所."""
    parts = symbol.split(".")
    return parts[0] if len(parts) > 1 else "UNKNOWN"
