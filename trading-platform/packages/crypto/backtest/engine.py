"""BTC backtest engine — adapts the standard futures backtest framework for crypto.

Key adaptations for crypto markets:
- 24/7/365 trading (no session breaks or settlement windows)
- Pluggable cost and slippage models (maker/taker tiers, volume impact)
- Funding rate settlement every 8 hours for perpetual futures
- Fractional quantity support (Decimal precision)
- Optional LOB-based matching via CryptoOrderBook
- Integration with Ch32's BTCDataPipeline / ParquetStorage

Usage:
    from backtest.crypto.engine import CryptoBacktestEngine
    from backtest.crypto.cost_model import MakerTakerCostModel
    from backtest.crypto.slippage import VolumeImpactSlippage

    engine = CryptoBacktestEngine(
        config=config,
        cost_model=MakerTakerCostModel(),
        slippage_model=VolumeImpactSlippage(),
    )
    engine.load_replayer(replayer)
    engine.set_strategy(strategy)
    result = engine.run()
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from .cost_model import CostBreakdown, CostModel, FeeRole, FlatRateCostModel
from .orderbook import CryptoOrderBook
from .slippage import FixedBpsSlippage, SlippageModel, SlippageResult

logger = logging.getLogger(__name__)

_ZERO = Decimal(0)


def _run_maybe_async(result: Any) -> Any:
    """Run an awaitable synchronously, safely handling nested event loops."""
    if not inspect.isawaitable(result):
        return result
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(result)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, result).result()
_FUNDING_INTERVAL = timedelta(hours=8)


class CryptoBacktestConfig:
    """Configuration for a crypto backtest run.

    Mirrors ``BacktestConfig`` from ``btc.models.types`` but adds
    crypto-specific options (funding, leverage, session mode).
    """

    def __init__(
        self,
        strategy_id: str,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
        *,
        initial_capital: Decimal = Decimal("100000"),
        bar_interval: str = "1m",
        leverage_limit: Decimal = Decimal(1),
        max_position_pct: Decimal = Decimal("0.1"),
        funding_rate_enabled: bool = True,
        use_orderbook_matching: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.strategy_id = strategy_id
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.bar_interval = bar_interval
        self.leverage_limit = leverage_limit
        self.max_position_pct = max_position_pct
        self.funding_rate_enabled = funding_rate_enabled
        self.use_orderbook_matching = use_orderbook_matching
        self.metadata = metadata or {}


class CryptoPosition:
    """Tracks a single symbol position with Decimal precision."""

    __slots__ = (
        "symbol", "quantity", "avg_entry_price",
        "unrealized_pnl", "realized_pnl",
    )

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.quantity: Decimal = _ZERO
        self.avg_entry_price: Decimal = _ZERO
        self.unrealized_pnl: Decimal = _ZERO
        self.realized_pnl: Decimal = _ZERO

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        return self.quantity < 0

    @property
    def is_flat(self) -> bool:
        return self.quantity == 0


class CryptoFill:
    """Executed fill with cost/slippage breakdown."""

    __slots__ = (
        "order_id", "symbol", "is_buy", "price", "quantity",
        "cost", "slippage", "timestamp",
    )

    def __init__(
        self,
        order_id: str,
        symbol: str,
        is_buy: bool,
        price: Decimal,
        quantity: Decimal,
        cost: CostBreakdown,
        slippage: SlippageResult,
        timestamp: datetime,
    ) -> None:
        self.order_id = order_id
        self.symbol = symbol
        self.is_buy = is_buy
        self.price = price
        self.quantity = quantity
        self.cost = cost
        self.slippage = slippage
        self.timestamp = timestamp


class CryptoOrder:
    """A simple order representation for the crypto engine."""

    __slots__ = (
        "id", "symbol", "is_buy", "quantity", "limit_price",
        "is_market", "created_at",
    )

    def __init__(
        self,
        symbol: str,
        is_buy: bool,
        quantity: Decimal,
        limit_price: Decimal | None = None,
        created_at: datetime | None = None,
    ) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.symbol = symbol
        self.is_buy = is_buy
        self.quantity = quantity
        self.limit_price = limit_price
        self.is_market = limit_price is None
        self.created_at = created_at


class CryptoBacktestResult:
    """Complete backtest result with crypto-specific metrics."""

    def __init__(
        self,
        config: CryptoBacktestConfig,
        equity_curve: list[tuple[datetime, Decimal]],
        fills: list[CryptoFill],
        total_funding_paid: Decimal,
        total_commission: Decimal,
        total_slippage: Decimal,
        bar_count: int,
        wall_seconds: float,
        started_at: datetime,
        finished_at: datetime,
    ) -> None:
        self.config = config
        self.equity_curve = equity_curve
        self.fills = fills
        self.total_funding_paid = total_funding_paid
        self.total_commission = total_commission
        self.total_slippage = total_slippage
        self.bar_count = bar_count
        self.wall_seconds = wall_seconds
        self.started_at = started_at
        self.finished_at = finished_at

    @property
    def initial_equity(self) -> Decimal:
        return self.equity_curve[0][1] if self.equity_curve else _ZERO

    @property
    def final_equity(self) -> Decimal:
        return self.equity_curve[-1][1] if self.equity_curve else _ZERO

    @property
    def total_return(self) -> Decimal:
        if self.initial_equity == 0:
            return _ZERO
        return (self.final_equity - self.initial_equity) / self.initial_equity


class CryptoBacktestEngine:
    """Main engine: orchestrates bar replay, matching, strategy, and accounting.

    Differences from the standard futures ``BacktestEngine``:
    1. No session breaks — bars stream continuously across midnight
    2. Funding rate settlement every 8 h when enabled
    3. Pluggable ``CostModel`` and ``SlippageModel``
    4. Optional order book matching via ``CryptoOrderBook``
    """

    def __init__(
        self,
        config: CryptoBacktestConfig,
        cost_model: CostModel | None = None,
        slippage_model: SlippageModel | None = None,
    ) -> None:
        self._config = config
        self._cost = cost_model or FlatRateCostModel()
        self._slip = slippage_model or FixedBpsSlippage()

        self._cash: Decimal = config.initial_capital
        self._positions: dict[str, CryptoPosition] = {}
        self._pending_orders: list[CryptoOrder] = []
        self._fills: list[CryptoFill] = []
        self._equity_curve: list[tuple[datetime, Decimal]] = []

        self._orderbooks: dict[str, CryptoOrderBook] = {}
        if config.use_orderbook_matching:
            for sym in config.symbols:
                self._orderbooks[sym] = CryptoOrderBook(symbol=sym)

        self._funding_rates: dict[datetime, Decimal] = {}
        self._last_funding_time: datetime | None = None
        self._total_funding: Decimal = _ZERO
        self._total_commission: Decimal = _ZERO
        self._total_slippage: Decimal = _ZERO

        self._bar_count = 0
        self._replayer: Any = None
        self._strategy: Any = None

    @property
    def config(self) -> CryptoBacktestConfig:
        return self._config

    @property
    def cash(self) -> Decimal:
        return self._cash

    @property
    def equity(self) -> Decimal:
        unrealized = sum(p.unrealized_pnl for p in self._positions.values())
        return self._cash + unrealized

    @property
    def positions(self) -> dict[str, CryptoPosition]:
        return dict(self._positions)

    def load_replayer(self, replayer: Any) -> None:
        """Attach a data replayer (from btc.replayer or custom iterator)."""
        self._replayer = replayer

    def set_strategy(self, strategy: Any) -> None:
        """Attach a strategy instance (BaseStrategy or duck-typed)."""
        self._strategy = strategy

    def set_funding_rates(self, rates: dict[datetime, Decimal]) -> None:
        """Pre-load historical funding rates keyed by settlement timestamp."""
        self._funding_rates = rates

    def submit_order(self, order: CryptoOrder) -> None:
        """Submit an order (called by strategy during on_bar)."""
        if not self._validate_order(order):
            logger.warning("Order rejected: %s", order.id)
            return
        self._pending_orders.append(order)

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        for i, o in enumerate(self._pending_orders):
            if o.id == order_id:
                self._pending_orders.pop(i)
                return True
        return False

    def run(self) -> CryptoBacktestResult:
        """Execute the backtest. Returns full result."""
        if self._replayer is None:
            raise RuntimeError("No replayer attached — call load_replayer() first")
        if self._strategy is None:
            raise RuntimeError("No strategy attached — call set_strategy() first")

        started_at = datetime.now(timezone.utc)
        wall_start = time.monotonic()

        if hasattr(self._strategy, "on_start"):
            _run_maybe_async(self._strategy.on_start())

        logger.info(
            "Crypto backtest starting: strategy=%s symbols=%s period=%s→%s",
            self._config.strategy_id,
            self._config.symbols,
            self._config.start_date,
            self._config.end_date,
        )

        for bar in self._replayer.replay(
            start=self._config.start_date,
            end=self._config.end_date,
        ):
            self._process_bar(bar)

        finished_at = datetime.now(timezone.utc)
        wall_elapsed = time.monotonic() - wall_start

        if hasattr(self._strategy, "on_stop"):
            _run_maybe_async(self._strategy.on_stop())

        logger.info(
            "Crypto backtest done: %d bars, %d fills, %.2fs",
            self._bar_count, len(self._fills), wall_elapsed,
        )

        return CryptoBacktestResult(
            config=self._config,
            equity_curve=list(self._equity_curve),
            fills=list(self._fills),
            total_funding_paid=self._total_funding,
            total_commission=self._total_commission,
            total_slippage=self._total_slippage,
            bar_count=self._bar_count,
            wall_seconds=wall_elapsed,
            started_at=started_at,
            finished_at=finished_at,
        )

    def _process_bar(self, bar: Any) -> None:
        self._bar_count += 1

        if self._config.use_orderbook_matching:
            symbol = getattr(bar, "symbol", self._config.symbols[0]) if hasattr(bar, "symbol") else self._config.symbols[0]
            if symbol in self._orderbooks:
                self._orderbooks[symbol].seed_from_bar(
                    mid_price=(bar.high + bar.low) / 2,
                    bar_volume=bar.volume,
                )

        new_fills = self._match_pending_orders(bar)
        for fill in new_fills:
            self._apply_fill(fill)
            self._fills.append(fill)
            if hasattr(self._strategy, "on_fill"):
                self._strategy.on_fill(fill)

        self._settle_funding(bar)
        self._update_unrealized(bar)
        self._equity_curve.append((bar.timestamp, self.equity))

        if hasattr(self._strategy, "on_bar"):
            self._strategy.on_bar(bar, self)

        if self._bar_count % 10000 == 0:
            logger.debug(
                "Progress: %d bars, equity=%.2f",
                self._bar_count, self.equity,
            )

    def _match_pending_orders(self, bar: Any) -> list[CryptoFill]:
        fills: list[CryptoFill] = []
        remaining: list[CryptoOrder] = []

        for order in self._pending_orders:
            fill_price = self._try_match_price(order, bar)
            if fill_price is not None:
                slip_result = self._slip.apply(
                    price=fill_price,
                    quantity=order.quantity,
                    is_buy=order.is_buy,
                    bar_volume=bar.volume,
                    volatility=bar.high - bar.low,
                )
                cost_result = self._cost.calculate(
                    price=slip_result.slipped_price,
                    quantity=order.quantity,
                    role=FeeRole.TAKER if order.is_market else FeeRole.MAKER,
                )
                fills.append(CryptoFill(
                    order_id=order.id,
                    symbol=order.symbol,
                    is_buy=order.is_buy,
                    price=slip_result.slipped_price,
                    quantity=order.quantity,
                    cost=cost_result,
                    slippage=slip_result,
                    timestamp=bar.timestamp,
                ))
            else:
                remaining.append(order)

        self._pending_orders = remaining
        return fills

    def _try_match_price(self, order: CryptoOrder, bar: Any) -> Decimal | None:
        if order.is_market:
            return bar.open

        assert order.limit_price is not None
        if order.is_buy and bar.low <= order.limit_price:
            return min(order.limit_price, bar.open)
        if not order.is_buy and bar.high >= order.limit_price:
            return max(order.limit_price, bar.open)
        return None

    def _apply_fill(self, fill: CryptoFill) -> None:
        pos = self._positions.setdefault(
            fill.symbol, CryptoPosition(fill.symbol),
        )
        total_cost = fill.cost.total
        self._total_commission += fill.cost.commission
        self._total_slippage += fill.slippage.slippage_amount * fill.quantity

        if fill.is_buy:
            new_qty = pos.quantity + fill.quantity
            if pos.quantity >= 0:
                if new_qty != 0:
                    pos.avg_entry_price = (
                        (pos.avg_entry_price * pos.quantity + fill.price * fill.quantity)
                        / new_qty
                    )
            else:
                closed = min(fill.quantity, abs(pos.quantity))
                pnl = closed * (pos.avg_entry_price - fill.price)
                pos.realized_pnl += pnl
                if new_qty > 0:
                    pos.avg_entry_price = fill.price
            pos.quantity = new_qty
            self._cash -= fill.price * fill.quantity + total_cost
        else:
            new_qty = pos.quantity - fill.quantity
            if pos.quantity <= 0:
                if new_qty != 0:
                    pos.avg_entry_price = (
                        (pos.avg_entry_price * abs(pos.quantity) + fill.price * fill.quantity)
                        / abs(new_qty)
                    )
            else:
                closed = min(fill.quantity, pos.quantity)
                pnl = closed * (fill.price - pos.avg_entry_price)
                pos.realized_pnl += pnl
                if new_qty < 0:
                    pos.avg_entry_price = fill.price
            pos.quantity = new_qty
            self._cash += fill.price * fill.quantity - total_cost

    def _settle_funding(self, bar: Any) -> None:
        if not self._config.funding_rate_enabled:
            return
        ts = bar.timestamp
        if self._last_funding_time is None:
            self._last_funding_time = ts
            return
        if ts - self._last_funding_time < _FUNDING_INTERVAL:
            return

        rate = self._funding_rates.get(ts, _ZERO)
        if rate == _ZERO:
            self._last_funding_time = ts
            return

        for pos in self._positions.values():
            if pos.is_flat:
                continue
            notional = abs(pos.quantity) * bar.close
            payment = self._cost.funding_cost(notional, rate, pos.is_long)
            self._cash -= payment
            self._total_funding += payment

        self._last_funding_time = ts

    def _update_unrealized(self, bar: Any) -> None:
        price = bar.close
        for pos in self._positions.values():
            if pos.is_flat:
                pos.unrealized_pnl = _ZERO
            elif pos.is_long:
                pos.unrealized_pnl = pos.quantity * (price - pos.avg_entry_price)
            else:
                pos.unrealized_pnl = abs(pos.quantity) * (pos.avg_entry_price - price)

    def _validate_order(self, order: CryptoOrder) -> bool:
        if order.quantity <= 0:
            return False
        if self._config.max_position_pct > 0 and self._equity_curve:
            last_eq = self._equity_curve[-1][1]
            price = order.limit_price
            if price is None and self._equity_curve:
                for sym, pos in self._positions.items():
                    if sym == order.symbol and pos.avg_entry_price > 0:
                        price = pos.avg_entry_price
                        break
            if price is None or price <= 0:
                return True
            notional = order.quantity * price
            if notional > last_eq * self._config.max_position_pct:
                return False
        return True
