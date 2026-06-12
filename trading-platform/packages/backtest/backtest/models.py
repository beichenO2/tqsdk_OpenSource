"""回测引擎核心数据模型。"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4


class OrderSide(str, enum.Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, enum.Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(str, enum.Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL_FILLED = "PARTIAL_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass(frozen=True, slots=True)
class Bar:
    """K线数据。"""
    symbol: str
    dt: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    open_interest: int = 0
    turnover: Decimal = Decimal(0)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Tick:
    """逐笔行情。"""
    symbol: str
    dt: datetime
    last_price: Decimal
    volume: int
    bid_price: Decimal
    ask_price: Decimal
    bid_volume: int
    ask_volume: int
    open_interest: int = 0


@dataclass(slots=True)
class Order:
    """订单。"""
    id: UUID = field(default_factory=uuid4)
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    price: Decimal = Decimal(0)
    limit_price: Decimal | None = None
    volume: int = 0
    filled_volume: int = 0
    avg_fill_price: Decimal = Decimal(0)
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    strategy_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def remaining_volume(self) -> int:
        return self.volume - self.filled_volume

    @property
    def is_active(self) -> bool:
        return self.status in (
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIAL_FILLED,
        )


@dataclass(frozen=True, slots=True)
class Trade:
    """成交记录。"""
    id: UUID = field(default_factory=uuid4)
    order_id: UUID = field(default_factory=uuid4)
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    price: Decimal = Decimal(0)
    volume: int = 0
    commission: Decimal = Decimal(0)
    slippage: Decimal = Decimal(0)
    dt: datetime = field(default_factory=datetime.now)


@dataclass(slots=True)
class Position:
    """持仓。"""
    symbol: str = ""
    long_volume: int = 0
    short_volume: int = 0
    long_avg_price: Decimal = Decimal(0)
    short_avg_price: Decimal = Decimal(0)
    unrealized_pnl: Decimal = Decimal(0)
    realized_pnl: Decimal = Decimal(0)
    margin: Decimal = Decimal(0)

    @property
    def net_volume(self) -> int:
        return self.long_volume - self.short_volume


@dataclass(slots=True)
class BacktestConfig:
    """回测配置。"""
    strategy_id: str = ""
    symbols: list[str] = field(default_factory=list)
    start_date: datetime | None = None
    end_date: datetime | None = None
    initial_capital: Decimal = Decimal("1000000")
    commission_rate: Decimal = Decimal("0.0001")
    slippage_ticks: int = 1
    tick_size: Decimal = Decimal("1")
    contract_multiplier: int = 1
    margin_ratio: Decimal = Decimal("0.1")
    data_frequency: str = "1min"
    benchmark_symbol: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EquityCurvePoint:
    """权益曲线点。"""
    dt: datetime = field(default_factory=datetime.now)
    equity: Decimal = Decimal(0)
    cash: Decimal = Decimal(0)
    position_value: Decimal = Decimal(0)
    drawdown: Decimal = Decimal(0)
    drawdown_pct: Decimal = Decimal(0)


@dataclass(slots=True)
class BacktestResult:
    """回测结果。"""
    config: BacktestConfig = field(default_factory=BacktestConfig)
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[EquityCurvePoint] = field(default_factory=list)
    final_equity: Decimal = Decimal(0)
    total_return: Decimal = Decimal(0)
    annual_return: Decimal = Decimal(0)
    max_drawdown: Decimal = Decimal(0)
    max_drawdown_pct: Decimal = Decimal(0)
    sharpe_ratio: Decimal = Decimal(0)
    sortino_ratio: Decimal = Decimal(0)
    win_rate: Decimal = Decimal(0)
    profit_factor: Decimal = Decimal(0)
    total_trades: int = 0
    avg_trade_pnl: Decimal = Decimal(0)
    avg_holding_period: float = 0.0
    calmar_ratio: Decimal = Decimal(0)
    start_time: datetime | None = None
    end_time: datetime | None = None
    elapsed_seconds: float = 0.0
