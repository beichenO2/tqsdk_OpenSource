"""委托单路由 — 下单、查询、撤单."""

from __future__ import annotations

from decimal import Decimal
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from core.enums.direction import Direction, Offset
from core.enums.market import Exchange
from core.enums.order_status import OrderStatus
from core.exceptions import OrderCancelFailedError, OrderNotFoundError

from app.deps import get_execution_service
from execution.service import ExecutionService

router = APIRouter(prefix="/orders", tags=["orders"])


_DIRECTION_ALIASES = {"BUY": Direction.LONG, "SELL": Direction.SHORT}


class CreateOrderRequest(BaseModel):
    strategy_id: str
    symbol: str
    exchange: Exchange
    direction: Direction
    offset: Offset
    price: Decimal
    volume: int = Field(..., gt=0)

    @classmethod
    def __get_validators__(cls):
        yield from super().__get_validators__()

    from pydantic import field_validator

    @field_validator("direction", mode="before")
    @classmethod
    def _normalize_direction(cls, v: str) -> str:
        if isinstance(v, str):
            up = v.upper()
            if up in _DIRECTION_ALIASES:
                return _DIRECTION_ALIASES[up].value
        return v


class OrderResponse(BaseModel):
    order_id: str
    status: str
    symbol: str = ""
    direction: str = ""
    price: str = ""
    volume: int = 0
    filled_volume: int = 0
    message: str = ""


@router.post("", response_model=OrderResponse)
async def create_order(
    req: CreateOrderRequest,
    svc: ExecutionService = Depends(get_execution_service),
) -> OrderResponse:
    order = await svc.place_order(
        strategy_id=req.strategy_id,
        symbol=req.symbol,
        exchange=req.exchange.value,
        direction=req.direction,
        offset=req.offset,
        price=req.price,
        volume=req.volume,
    )
    msg = ""
    if order.status == OrderStatus.REJECTED:
        msg = "Risk rejected"
    return OrderResponse(
        order_id=order.order_id,
        status=order.status.value,
        symbol=order.symbol,
        direction=order.direction.value,
        price=str(order.price),
        volume=order.volume,
        filled_volume=order.filled_volume,
        message=msg,
    )


@router.get("/{order_id}", response_model=OrderResponse)
async def get_order(
    order_id: str,
    svc: ExecutionService = Depends(get_execution_service),
) -> OrderResponse:
    order = svc.get_order(order_id)
    if order is None:
        raise OrderNotFoundError(f"Order {order_id} not found")
    return OrderResponse(
        order_id=order.order_id,
        status=order.status.value,
        symbol=order.symbol,
        direction=order.direction.value,
        price=str(order.price),
        volume=order.volume,
        filled_volume=order.filled_volume,
    )


@router.delete("/{order_id}")
async def cancel_order(
    order_id: str,
    svc: ExecutionService = Depends(get_execution_service),
) -> dict:
    success = await svc.cancel_order(order_id)
    if not success:
        raise OrderCancelFailedError(f"Failed to cancel order {order_id}")
    return {"order_id": order_id, "status": "CANCELLED"}


@router.get("", response_model=list[OrderResponse])
async def list_orders(
    strategy_id: Optional[str] = None,
    active_only: bool = False,
    svc: ExecutionService = Depends(get_execution_service),
) -> list[OrderResponse]:
    if active_only:
        orders = svc.get_active_orders()
    else:
        orders = svc.get_all_orders()

    return [
        OrderResponse(
            order_id=o.order_id,
            status=o.status.value,
            symbol=o.symbol,
            direction=o.direction.value,
            price=str(o.price),
            volume=o.volume,
            filled_volume=o.filled_volume,
        )
        for o in orders
        if strategy_id is None or o.strategy_id == strategy_id
    ]
