"""Position tracking and PnL computation using Ch26's Position model."""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Optional

from core.enums.direction import Direction, Offset
from core.models.position import Position
from core.models.trade import Trade

logger = logging.getLogger(__name__)


class PositionManager:
    """Tracks positions per (symbol, direction) and computes PnL."""

    def __init__(self) -> None:
        self._positions: dict[str, Position] = {}
        self._balance: Decimal = Decimal("0")
        self._available: Decimal = Decimal("0")

    @staticmethod
    def _key(symbol: str, direction: Direction) -> str:
        return f"{symbol}:{direction.value}"

    def on_trade(self, trade: Trade) -> None:
        """Update positions based on a new trade."""
        if trade.offset == Offset.OPEN:
            self._open(trade)
        else:
            self._close(trade)

    def _open(self, trade: Trade) -> None:
        key = self._key(trade.symbol, trade.direction)
        pos = self._positions.get(key)

        if pos is None:
            pos = Position(
                symbol=trade.symbol,
                exchange=trade.exchange,
                direction=trade.direction,
            )
            self._positions[key] = pos

        total_cost = pos.avg_price * pos.volume + trade.price * trade.volume
        pos.volume += trade.volume
        pos.available += trade.volume
        pos.avg_price = total_cost / Decimal(pos.volume) if pos.volume > 0 else Decimal("0")

        logger.info("Position opened: %s vol=%d avg=%s", key, pos.volume, pos.avg_price)

    def _close(self, trade: Trade) -> None:
        opposite = Direction.SHORT if trade.direction == Direction.LONG else Direction.LONG
        key = self._key(trade.symbol, opposite)
        pos = self._positions.get(key)

        if pos is None or pos.volume <= 0:
            logger.error("Close trade but no position: %s", key)
            return

        close_vol = min(trade.volume, pos.volume)
        multiplier = 1 if pos.direction == Direction.LONG else -1
        pnl = (trade.price - pos.avg_price) * close_vol * multiplier
        pos.close_pnl += pnl
        pos.volume -= close_vol
        pos.available = min(pos.available, pos.volume)

        if pos.volume == 0:
            del self._positions[key]

        logger.info("Position closed: %s vol=%d pnl=%s", key, close_vol, pnl)

    def freeze(self, symbol: str, direction: Direction, volume: int) -> bool:
        """Freeze volume for a pending close order."""
        key = self._key(symbol, direction)
        pos = self._positions.get(key)
        if pos is None or pos.available < volume:
            return False
        pos.available -= volume
        return True

    def unfreeze(self, symbol: str, direction: Direction, volume: int) -> None:
        key = self._key(symbol, direction)
        pos = self._positions.get(key)
        if pos:
            pos.available = min(pos.volume, pos.available + volume)

    def update_prices(self, prices: dict[str, Decimal], multipliers: Optional[dict[str, int]] = None) -> None:
        """Batch update last prices and recalculate floating PnL."""
        for key, pos in self._positions.items():
            if pos.symbol in prices:
                last = prices[pos.symbol]
                m = (multipliers or {}).get(pos.symbol, 1)
                sign = 1 if pos.direction == Direction.LONG else -1
                pos.float_pnl = (last - pos.avg_price) * pos.volume * m * sign

    def get_position(self, symbol: str, direction: Direction) -> Optional[Position]:
        return self._positions.get(self._key(symbol, direction))

    def get_all_positions(self) -> list[Position]:
        return [p for p in self._positions.values() if p.volume > 0]

    def get_net_volume(self, symbol: str) -> int:
        long = self._positions.get(self._key(symbol, Direction.LONG))
        short = self._positions.get(self._key(symbol, Direction.SHORT))
        return (long.volume if long else 0) - (short.volume if short else 0)

    def sync_from_broker(self, broker_positions: list[Position]) -> None:
        """Reconcile local state with broker's ground truth."""
        self._positions.clear()
        for pos in broker_positions:
            if pos.volume > 0:
                self._positions[self._key(pos.symbol, pos.direction)] = pos
        logger.info("Positions synced: %d entries", len(self._positions))

    def total_margin(self) -> Decimal:
        return sum(p.margin for p in self._positions.values())

    def total_float_pnl(self) -> Decimal:
        return sum(p.float_pnl for p in self._positions.values())

    def total_close_pnl(self) -> Decimal:
        return sum(p.close_pnl for p in self._positions.values())
