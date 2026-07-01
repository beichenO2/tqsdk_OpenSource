"""BTC API 路由 — 通过 broker_crypto WEEX 适配器访问交易所

端点前缀: /api/v1/btc/
覆盖: 行情、交易、账户、策略、回测
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from app.deps import get_btc_broker_manager
from core.exceptions import (
    BacktestError,
    BacktestUnavailableError,
    ExchangeError,
    ExchangeNotConnectedError,
    OrderNotFoundError,
    PermissionDeniedError,
    ServiceNotReadyError,
    StrategyNotFoundError,
    ValidationError,
)

router = APIRouter()
logger = logging.getLogger(__name__)


# ── Helpers ──


def _resolve_exchange(exchange_str: str):
    from broker_crypto import CryptoExchange
    try:
        return CryptoExchange(exchange_str.upper())
    except ValueError:
        raise ValidationError(f"Unknown exchange: {exchange_str}")


def _require_manager(manager=Depends(get_btc_broker_manager)):
    if manager is None:
        raise ServiceNotReadyError("BTC broker not initialized")
    return manager


def _require_adapter(exchange_str: str, manager):
    ex = _resolve_exchange(exchange_str)
    try:
        return manager.get_adapter(ex)
    except KeyError:
        raise ExchangeNotConnectedError(
            f"Exchange {exchange_str} not connected",
            detail={"connected": [e.value for e in manager.exchanges]},
        )


# ── Request / Response Schemas ──


class PlaceOrderRequest(BaseModel):
    exchange: str = Field(default="weex", description="交易所: weex")
    symbol: str = Field(default="BTCUSDT")
    side: str = Field(..., description="buy / sell")
    order_type: Optional[str] = Field(default=None, alias="type")
    type: Optional[str] = None
    quantity: Optional[Decimal] = None
    amount: Optional[Decimal] = None
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = Field(default=None, alias="stopPrice")
    stopPrice: Optional[Decimal] = None
    time_in_force: str = "gtc"
    client_order_id: Optional[str] = None

    model_config = {"populate_by_name": True}

    @property
    def resolved_quantity(self) -> Decimal:
        return self.quantity or self.amount or Decimal("0")

    @property
    def resolved_order_type(self) -> str:
        return self.order_type or self.type or "limit"


class CancelOrderRequest(BaseModel):
    exchange: str
    order_id: str
    symbol: str = "BTCUSDT"


class StrategyToggleRequest(BaseModel):
    strategy_name: str
    enabled: bool


class BacktestRequest(BaseModel):
    model_config = {"populate_by_name": True}

    strategy_name: str = Field(alias="strategyId", default="btc_momentum")
    symbol: str = "BTCUSDT"
    exchange: str = "weex"
    interval: str = "1h"
    start_date: str = Field(alias="startDate")
    end_date: str = Field(alias="endDate")
    initial_capital: Decimal = Field(Decimal("10000"), alias="initialCapital")
    commission: Decimal | None = Field(None, alias="commission")
    slippage: Decimal | None = Field(None, alias="slippage")
    params: dict = Field(default_factory=dict)


# ── Market Data ──


@router.get("/market/ticker/{symbol}")
async def get_ticker(
    symbol: str,
    exchange: str = Query(default="weex"),
    manager=Depends(_require_manager),
) -> dict:
    adapter = _require_adapter(exchange, manager)
    try:
        ticker = await adapter.get_ticker(symbol)
        return ticker.model_dump(mode="json")
    except Exception as e:
        logger.exception("get_ticker failed: %s %s", exchange, symbol)
        raise ExchangeError(f"Exchange error: {e}") from e


@router.get("/market/klines/{symbol}")
async def get_klines(
    symbol: str,
    exchange: str = Query(default="weex"),
    interval: Optional[str] = Query(default=None),
    timeframe: Optional[str] = Query(default=None),
    limit: int = Query(default=500, le=1500),
    manager=Depends(_require_manager),
) -> list:
    adapter = _require_adapter(exchange, manager)
    resolved_interval = timeframe or interval or "1h"
    try:
        ohlcv_list = await adapter.get_ohlcv(symbol, resolved_interval, limit)
        return [bar.model_dump(mode="json") for bar in ohlcv_list]
    except Exception as e:
        logger.exception("get_klines failed: %s %s", exchange, symbol)
        raise ExchangeError(f"Exchange error: {e}") from e


@router.get("/market/orderbook/{symbol}")
async def get_orderbook(
    symbol: str,
    exchange: str = Query(default="weex"),
    depth: Optional[int] = Query(default=None, le=100),
    limit: Optional[int] = Query(default=None, le=100),
    manager=Depends(_require_manager),
) -> dict:
    adapter = _require_adapter(exchange, manager)
    resolved_depth = limit or depth or 20
    try:
        book = await adapter.get_orderbook(symbol, resolved_depth)
        return book.model_dump(mode="json")
    except Exception as e:
        logger.exception("get_orderbook failed: %s %s", exchange, symbol)
        raise ExchangeError(f"Exchange error: {e}") from e


@router.get("/market/trades/{symbol}")
async def get_recent_trades(
    symbol: str,
    exchange: str = Query(default="weex"),
    limit: int = Query(default=50, le=500),
    manager=Depends(_require_manager),
) -> list:
    adapter = _require_adapter(exchange, manager)
    try:
        trades = await adapter.get_recent_trades(symbol, limit)
        return [t.model_dump(mode="json") for t in trades]
    except Exception as e:
        logger.exception("get_recent_trades failed: %s %s", exchange, symbol)
        raise ExchangeError(f"Exchange error: {e}") from e


# ── Trading ──


@router.post("/orders")
async def place_order(
    req: PlaceOrderRequest,
    manager=Depends(_require_manager),
) -> dict:
    try:
        from broker_crypto.models import OrderRequest, OrderType, Side, TimeInForce
    except (ImportError, ModuleNotFoundError):
        from broker_crypto.src.broker_crypto.models import OrderRequest, OrderType, Side, TimeInForce

    ex = _resolve_exchange(req.exchange)
    try:
        order_req = OrderRequest(
            exchange=ex,
            symbol=req.symbol,
            side=Side(req.side.lower()),
            order_type=OrderType(req.resolved_order_type),
            quantity=req.resolved_quantity,
            price=req.price,
            stop_price=req.stop_price or req.stopPrice,
            time_in_force=TimeInForce(req.time_in_force.lower()),
            client_order_id=req.client_order_id,
        )
        result = await manager.place_order(order_req)
        return result.model_dump(mode="json")
    except PermissionError as e:
        raise PermissionDeniedError(str(e)) from e
    except KeyError:
        raise ExchangeNotConnectedError(f"Exchange {req.exchange} not connected")
    except Exception as e:
        logger.exception("place_order failed: %s", req.exchange)
        raise ExchangeError(f"Exchange error: {e}") from e


@router.delete("/orders/{order_id}")
async def cancel_order(
    order_id: str,
    exchange: str = Query(default="weex"),
    symbol: str = Query(default="BTCUSDT"),
    manager=Depends(_require_manager),
) -> dict:
    ex = _resolve_exchange(exchange)
    try:
        result = await manager.cancel_order(ex, order_id, symbol)
        return result.model_dump(mode="json")
    except KeyError:
        raise ExchangeNotConnectedError(f"Exchange {exchange} not connected")
    except Exception as e:
        logger.exception("cancel_order failed: %s %s", exchange, order_id)
        raise ExchangeError(f"Exchange error: {e}") from e


@router.get("/orders")
async def get_orders(
    exchange: str = Query(default="weex"),
    symbol: Optional[str] = None,
    status: Optional[str] = None,
    manager=Depends(_require_manager),
) -> list:
    ex = _resolve_exchange(exchange)
    try:
        orders = await manager.get_open_orders(ex, symbol)
        result = [o.model_dump(mode="json") for o in orders]
        if status:
            result = [o for o in result if o.get("status", "").lower() == status.lower()]
        return result
    except KeyError:
        raise ExchangeNotConnectedError(f"Exchange {exchange} not connected")
    except Exception as e:
        logger.exception("get_orders failed: %s", exchange)
        raise ExchangeError(f"Exchange error: {e}") from e


@router.get("/orders/{order_id}")
async def get_order(
    order_id: str,
    exchange: str = Query(default="weex"),
    symbol: str = Query(default="BTCUSDT"),
    manager=Depends(_require_manager),
) -> dict:
    ex = _resolve_exchange(exchange)
    try:
        order = await manager.get_order(ex, order_id, symbol)
        return order.model_dump(mode="json")
    except KeyError:
        raise ExchangeNotConnectedError(f"Exchange {exchange} not connected")
    except Exception as e:
        logger.exception("get_order failed: %s %s", exchange, order_id)
        raise ExchangeError(f"Exchange error: {e}") from e


# ── Account ──


@router.get("/trades")
async def get_trade_history(
    limit: int = Query(default=100, le=500),
) -> list:
    return []


@router.get("/account/balances")
async def get_balances(
    exchange: Optional[str] = None,
    manager=Depends(_require_manager),
) -> list:
    try:
        if exchange:
            ex = _resolve_exchange(exchange)
            balances = await manager.get_balances(ex)
            return [b.model_dump(mode="json") for b in balances]

        all_balances = await manager.get_all_balances()
        result = []
        for bal_list in all_balances.values():
            result.extend(b.model_dump(mode="json") for b in bal_list)
        return result
    except KeyError:
        raise ExchangeNotConnectedError(f"Exchange {exchange} not connected")
    except Exception as e:
        logger.exception("get_balances failed")
        raise ExchangeError(f"Exchange error: {e}") from e


@router.get("/account/positions")
async def get_positions(
    exchange: Optional[str] = None,
    symbol: Optional[str] = None,
    manager=Depends(_require_manager),
) -> list:
    try:
        if exchange:
            ex = _resolve_exchange(exchange)
            positions = await manager.get_positions(ex, symbol)
            return [p.model_dump(mode="json") for p in positions]

        all_positions = await manager.get_all_positions()
        result = []
        for pos_list in all_positions.values():
            result.extend(p.model_dump(mode="json") for p in pos_list)
        return result
    except KeyError:
        raise ExchangeNotConnectedError(f"Exchange {exchange} not connected")
    except Exception as e:
        logger.exception("get_positions failed")
        raise ExchangeError(f"Exchange error: {e}") from e


# ── Strategies ──


@router.get("/strategies")
async def list_strategies() -> list:
    try:
        from strategy.registry import StrategyRegistry
        all_names = StrategyRegistry.list_registered()
        return [
            {"name": n, "enabled": False}
            for n in all_names
            if n.startswith("btc_")
        ]
    except ImportError:
        return [
            {"name": "btc_momentum", "description": "动量趋势跟踪", "enabled": False},
            {"name": "btc_mean_reversion", "description": "均值回归", "enabled": False},
            {"name": "btc_grid", "description": "网格交易", "enabled": False},
            {"name": "btc_trend_following", "description": "趋势跟踪", "enabled": False},
        ]


@router.get("/strategies/{name}")
async def get_strategy(name: str) -> dict:
    try:
        from strategy.registry import StrategyRegistry
        cls = StrategyRegistry.get(name)
        if cls is None:
            raise StrategyNotFoundError(f"Strategy not found: {name}")
        return {"name": name, "class": cls.__name__, "enabled": False}
    except ImportError:
        raise StrategyNotFoundError(f"Strategy not found: {name}")


@router.put("/strategies/{name}/toggle")
async def toggle_strategy(name: str, req: StrategyToggleRequest) -> dict:
    return {
        "name": name,
        "enabled": req.enabled,
        "message": "Strategy toggle requires strategy runner (v2)",
    }


# ── Backtest ──


@router.post("/backtest")
async def run_backtest(req: BacktestRequest) -> dict:
    try:
        from backtest.btc.engine.btc_backtest_engine import BTCBacktestEngine
        engine = BTCBacktestEngine()
        result = engine.run(
            strategy_name=req.strategy_name,
            symbol=req.symbol,
            interval=req.interval,
            start_date=req.start_date,
            end_date=req.end_date,
            initial_capital=float(req.initial_capital),
            params=req.params,
        )
        return {"status": "completed", "result": result}
    except ImportError:
        raise BacktestUnavailableError("BTC backtest engine not available")
    except Exception as e:
        logger.exception("BTC backtest failed")
        raise BacktestError(f"Backtest error: {e}") from e


@router.get("/backtest/results/{strategy_id}")
async def get_backtest_results_by_strategy(strategy_id: str) -> list:
    return []


@router.get("/backtest/{backtest_id}")
async def get_backtest_result(backtest_id: str) -> dict:
    raise OrderNotFoundError(f"Backtest result not found: {backtest_id}")


@router.get("/backtest")
async def list_backtests(
    strategy: Optional[str] = None,
    limit: int = Query(default=20, le=100),
) -> list:
    return []


# ── Exchanges ──


@router.get("/exchanges")
async def list_exchanges(
    manager=Depends(get_btc_broker_manager),
) -> dict:
    all_exchanges = [
        {"id": "weex", "name": "WEEX", "enum_value": "WEEX"},
    ]
    connected = {e.value for e in (manager.exchanges if manager else [])}
    return {
        "exchanges": [
            {"id": ex["id"], "name": ex["name"], "connected": ex["enum_value"] in connected}
            for ex in all_exchanges
        ],
    }


@router.get("/exchanges/{exchange_id}/status")
async def exchange_status(
    exchange_id: str,
    manager=Depends(get_btc_broker_manager),
) -> dict:
    if manager is None:
        return {"exchange": exchange_id, "connected": False, "latency_ms": None}
    try:
        ex = _resolve_exchange(exchange_id)
        adapter = manager.get_adapter(ex)
        return {
            "exchange": exchange_id,
            "connected": True,
            "adapter": type(adapter).__name__,
        }
    except (KeyError, ValidationError):
        return {"exchange": exchange_id, "connected": False, "latency_ms": None}
