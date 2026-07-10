"""策略基类和核心数据模型。"""

from __future__ import annotations

import enum
import uuid
from abc import ABC, abstractmethod
from collections import deque
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class SignalType(str, enum.Enum):
    LONG_ENTRY = "long_entry"
    LONG_EXIT = "long_exit"
    SHORT_ENTRY = "short_entry"
    SHORT_EXIT = "short_exit"
    HOLD = "hold"


class OrderSide(str, enum.Enum):
    BUY = "buy"
    SELL = "sell"


class Signal(BaseModel):
    """策略产出的交易信号。"""

    signal_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    strategy_id: str
    symbol: str
    signal_type: SignalType
    strength: float = Field(ge=0.0, le=1.0, description="信号强度 0~1")
    price: float | None = None
    suggested_qty: float | None = None
    reason: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class Position(BaseModel):
    """当前持仓快照。"""

    symbol: str
    side: OrderSide
    qty: float
    avg_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


class StrategyConfig(BaseModel):
    """策略配置基类，子类可扩展。"""

    strategy_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str
    version: str = "1.0.0"
    symbols: list[str] = Field(default_factory=list)
    features: list[str] = Field(default_factory=list, description="策略消费的因子名")
    params: dict[str, Any] = Field(default_factory=dict)
    risk_limits: dict[str, float] = Field(default_factory=dict)
    enabled: bool = True


class StrategyState(str, enum.Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"


class BaseStrategy(ABC):
    """所有策略的抽象基类。

    子类必须实现:
    - on_bar(): 收到 K 线时的处理逻辑
    - on_tick(): 收到 Tick 时的处理逻辑（可选）
    - generate_signals(): 根据当前市场状态生成交易信号
    """

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config
        self.state = StrategyState.IDLE
        self._positions: dict[str, Position] = {}
        self._signals: deque[Signal] = deque(maxlen=500)

    @property
    def strategy_id(self) -> str:
        return self.config.strategy_id

    @property
    def name(self) -> str:
        return self.config.name

    @abstractmethod
    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        """收到新 K 线数据时调用。返回产出的信号列表。"""
        ...

    async def on_tick(self, symbol: str, tick: dict[str, Any]) -> list[Signal]:
        """收到 Tick 数据时调用。默认不处理，子类可覆盖。"""
        return []

    @abstractmethod
    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        """基于当前市场状态批量生成信号。"""
        ...

    async def on_start(self) -> None:
        """策略启动时的初始化逻辑。"""
        self.state = StrategyState.RUNNING

    async def on_stop(self) -> None:
        """策略停止时的清理逻辑。"""
        self.state = StrategyState.STOPPED

    async def on_error(self, error: Exception) -> None:
        """策略出错时的处理逻辑。"""
        self.state = StrategyState.ERROR

    def on_fill(self, fill: Any) -> None:
        """回测成交回调（同步）；实盘可忽略或覆盖。"""

    def on_backtest_complete(self, result: Any) -> None:
        """回测结果产出后的同步钩子（metrics 已计算）。"""

    def update_position(self, position: Position) -> None:
        self._positions[position.symbol] = position

    def remove_position(self, symbol: str) -> None:
        """Remove cached position (e.g. flat in backtest mirror)."""
        self._positions.pop(symbol, None)

    def get_position(self, symbol: str) -> Position | None:
        return self._positions.get(symbol)

    def get_all_positions(self) -> dict[str, Position]:
        return dict(self._positions)

    def record_signal(self, signal: Signal) -> None:
        self._signals.append(signal)

    def get_recent_signals(self, limit: int = 50) -> list[Signal]:
        return list(self._signals)[-limit:]

    def get_param(self, key: str, default: Any = None) -> Any:
        return self.config.params.get(key, default)
