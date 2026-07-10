"""TDD tests for live execution wiring and close-all positions."""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.enums.direction import Direction, Offset
from core.enums.market import Exchange
from core.enums.order_status import OrderStatus
from core.models.order import Order
from core.models.position import Position
from execution.service import ExecutionService
from sim_live.account_manager import AccountManager
from sim_live.live_scheduler import LiveScheduler, TradingMode
from strategy.base import BaseStrategy, Signal, SignalType, StrategyConfig
from tests.conftest import MockBrokerAdapter


class PositionsMockBroker(MockBrokerAdapter):
    """MockBrokerAdapter that returns configured positions."""

    def __init__(self, positions: list[Position] | None = None) -> None:
        super().__init__()
        self._positions = positions or []
        self.submitted: list[dict[str, Any]] = []
        self._fail_symbols: set[str] = set()

    def fail_on_symbol(self, symbol: str) -> None:
        self._fail_symbols.add(symbol)

    async def query_positions(self) -> list[Position]:
        return list(self._positions)

    async def submit_order(
        self,
        symbol: str,
        direction: Direction,
        offset: Offset,
        price: Decimal,
        volume: int,
        strategy_id: str = "",
    ) -> Order:
        if symbol in self._fail_symbols:
            raise RuntimeError(f"broker rejected {symbol}")
        self.submit_order_calls += 1
        self.submitted.append(
            {
                "symbol": symbol,
                "direction": direction,
                "offset": offset,
                "price": price,
                "volume": volume,
                "strategy_id": strategy_id,
            }
        )
        self._order_seq += 1
        return Order(
            order_id=f"mock-{self._order_seq}",
            strategy_id=strategy_id or "close-all",
            symbol=symbol,
            exchange=Exchange.SHFE,
            direction=direction,
            offset=offset,
            price=price,
            volume=volume,
            status=OrderStatus.SUBMITTED,
        )


class OneShotLongEntryStrategy(BaseStrategy):
    """Emits a single LONG_ENTRY signal on the first bar."""

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self._fired = False

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        if self._fired:
            return []
        self._fired = True
        return [
            Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                signal_type=SignalType.LONG_ENTRY,
                strength=0.8,
                price=bar["close"],
                reason="test entry",
            )
        ]

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        return []


def _make_long_position(symbol: str, volume: int = 2) -> Position:
    return Position(
        symbol=symbol,
        exchange=Exchange.SHFE,
        direction=Direction.LONG,
        volume=volume,
        available=volume,
        avg_price=Decimal("3500"),
    )


@pytest.mark.asyncio
async def test_close_all_positions_submits_close_orders() -> None:
    broker = PositionsMockBroker(
        positions=[
            _make_long_position("rb2510", 2),
            _make_long_position("cu2509", 1),
        ]
    )
    svc = ExecutionService(broker)

    result = await svc.close_all_positions()

    assert result["requested"] == 2
    assert len(result["submitted"]) == 2
    assert result["failed"] == []
    assert broker.submit_order_calls == 2

    by_symbol = {item["symbol"]: item for item in broker.submitted}
    assert by_symbol["rb2510"]["direction"] == Direction.SHORT
    assert by_symbol["rb2510"]["offset"] == Offset.CLOSE
    assert by_symbol["rb2510"]["volume"] == 2
    assert by_symbol["cu2509"]["direction"] == Direction.SHORT
    assert by_symbol["cu2509"]["offset"] == Offset.CLOSE
    assert by_symbol["cu2509"]["volume"] == 1

    for item in result["submitted"]:
        assert "order_id" in item
        assert item["symbol"] in ("rb2510", "cu2509")


@pytest.mark.asyncio
async def test_close_all_partial_failure() -> None:
    broker = PositionsMockBroker(
        positions=[
            _make_long_position("rb2510", 1),
            _make_long_position("cu2509", 1),
        ]
    )
    broker.fail_on_symbol("cu2509")
    svc = ExecutionService(broker)

    result = await svc.close_all_positions()

    assert result["requested"] == 2
    assert len(result["submitted"]) == 1
    assert result["submitted"][0]["symbol"] == "rb2510"
    assert len(result["failed"]) == 1
    assert result["failed"][0]["symbol"] == "cu2509"
    assert "broker rejected cu2509" in result["failed"][0]["error"]


@pytest.mark.asyncio
async def test_scheduler_live_signal_reaches_execution() -> None:
    accounts = AccountManager(crypto_count=1, futures_count=0)
    accounts.assign_strategy(1, "test-strategy")

    config = StrategyConfig(
        name="test-strategy",
        symbols=["BTCUSDT"],
    )
    strategy = OneShotLongEntryStrategy(config)

    mock_exec = MagicMock()
    mock_exec.place_order = AsyncMock(
        return_value=Order(
            order_id="live-1",
            strategy_id="test-strategy",
            symbol="BTCUSDT",
            exchange=Exchange.SHFE,
            direction=Direction.LONG,
            offset=Offset.OPEN,
            price=Decimal("50000"),
            volume=1,
            status=OrderStatus.SUBMITTED,
        )
    )

    scheduler = LiveScheduler(
        accounts=accounts,
        strategies={1: strategy},
        mode=TradingMode.LIVE,
        execution_service=mock_exec,
    )

    bar = {
        "timestamp": "2026-07-10T00:00:00+00:00",
        "open": 50000.0,
        "high": 50100.0,
        "low": 49900.0,
        "close": 50000.0,
        "volume": 100.0,
    }
    result = await scheduler.run_bar("2026-07-10T00:00:00+00:00", {"BTCUSDT": bar})

    mock_exec.place_order.assert_awaited_once()
    call_kwargs = mock_exec.place_order.await_args.kwargs
    assert call_kwargs["symbol"] == "BTCUSDT"
    assert call_kwargs["direction"] == Direction.LONG
    assert call_kwargs["offset"] == Offset.OPEN
    assert call_kwargs["volume"] >= 1
    assert result["total_signals"] == 1
    assert result["total_fills"] == 1


@pytest.mark.asyncio
async def test_scheduler_live_order_failure_does_not_crash() -> None:
    accounts = AccountManager(crypto_count=1, futures_count=0)
    accounts.assign_strategy(1, "test-strategy")

    config = StrategyConfig(
        name="test-strategy",
        symbols=["BTCUSDT"],
    )
    strategy = OneShotLongEntryStrategy(config)

    mock_exec = MagicMock()
    mock_exec.place_order = AsyncMock(side_effect=RuntimeError("exchange down"))

    scheduler = LiveScheduler(
        accounts=accounts,
        strategies={1: strategy},
        mode=TradingMode.LIVE,
        execution_service=mock_exec,
    )

    bar = {
        "timestamp": "2026-07-10T00:00:00+00:00",
        "open": 50000.0,
        "high": 50100.0,
        "low": 49900.0,
        "close": 50000.0,
        "volume": 100.0,
    }

    result = await scheduler.run_bar("2026-07-10T00:00:00+00:00", {"BTCUSDT": bar})
    second = await scheduler.run_bar("2026-07-10T00:01:00+00:00", {"BTCUSDT": bar})

    mock_exec.place_order.assert_awaited_once()
    assert result["total_signals"] == 1
    assert result["total_fills"] == 0
    assert second["bar_index"] == 2
