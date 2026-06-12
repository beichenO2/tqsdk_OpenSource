"""BTC backtesting domain models."""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field


class OHLCV(BaseModel):
    """Single candlestick bar."""

    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    turnover: Decimal = Decimal(0)

    @property
    def mid(self) -> Decimal:
        return (self.high + self.low) / 2


class OrderSide(str, enum.Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, enum.Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, enum.Enum):
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIAL_FILLED = "partial_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class Order(BaseModel):
    """A trading order in the simulation."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Decimal | None = None
    stop_price: Decimal | None = None
    filled_quantity: Decimal = Decimal(0)
    avg_fill_price: Decimal = Decimal(0)
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime | None = None
    updated_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Fill(BaseModel):
    """An executed fill for an order."""

    order_id: str
    symbol: str
    side: OrderSide
    price: Decimal
    quantity: Decimal
    commission: Decimal = Decimal(0)
    slippage: Decimal = Decimal(0)
    timestamp: datetime


class Position(BaseModel):
    """Current position for a symbol."""

    symbol: str
    quantity: Decimal = Decimal(0)
    avg_entry_price: Decimal = Decimal(0)
    unrealized_pnl: Decimal = Decimal(0)
    realized_pnl: Decimal = Decimal(0)

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        return self.quantity < 0

    @property
    def is_flat(self) -> bool:
        return self.quantity == 0

    def market_value(self, current_price: Decimal) -> Decimal:
        return self.quantity * current_price


class BacktestConfig(BaseModel):
    """Configuration for a backtest run."""

    strategy_id: str
    symbols: list[str]
    start_date: datetime
    end_date: datetime
    initial_capital: Decimal = Decimal("100000")
    commission_rate: Decimal = Decimal("0.001")
    slippage_bps: Decimal = Decimal("5")
    bar_interval: str = "1m"
    leverage_limit: Decimal = Decimal(1)
    max_position_pct: Decimal = Decimal("0.1")
    funding_rate_enabled: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class PerformanceMetrics(BaseModel):
    """Aggregated performance statistics."""

    total_return: Decimal = Decimal(0)
    annualized_return: Decimal = Decimal(0)
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: Decimal = Decimal(0)
    max_drawdown_duration_days: int = 0
    win_rate: float = 0.0
    profit_factor: Decimal = Decimal(0)
    total_trades: int = 0
    avg_trade_pnl: Decimal = Decimal(0)
    avg_win: Decimal = Decimal(0)
    avg_loss: Decimal = Decimal(0)
    calmar_ratio: float = 0.0
    volatility: float = 0.0
    avg_holding_period_hours: float = 0.0
    total_commission: Decimal = Decimal(0)
    total_slippage: Decimal = Decimal(0)


class BacktestResult(BaseModel):
    """Complete backtest result."""

    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    config: BacktestConfig
    metrics: PerformanceMetrics
    equity_curve: list[tuple[datetime, Decimal]] = Field(default_factory=list)
    fills: list[Fill] = Field(default_factory=list)
    daily_returns: list[tuple[datetime, float]] = Field(default_factory=list)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
