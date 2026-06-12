"""实盘/模拟调度器 — 策略信号路由到真实交易所或本地撮合。

支持两种模式：
- paper: 信号 → SimMatchingEngine 本地撮合（与 PaperScheduler 行为一致）
- live:  信号 → ExecutionService → BrokerAdapter → 真实交易所

核心设计：复用 PaperScheduler 的策略调度逻辑，仅替换信号执行路径。
"""

from __future__ import annotations

import asyncio
import enum
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable

from strategy.base import BaseStrategy, Signal, SignalType

from .account_manager import AccountManager, SimAccount

logger = logging.getLogger(__name__)


class TradingMode(str, enum.Enum):
    PAPER = "paper"
    LIVE = "live"


class LiveScheduler:
    """统一调度器，支持 paper/live 模式一键切换。"""

    def __init__(
        self,
        accounts: AccountManager,
        strategies: dict[int, BaseStrategy],
        mode: TradingMode = TradingMode.PAPER,
        execution_service: Any | None = None,
        on_signal: Callable[[Signal, str], None] | None = None,
        on_fill: Callable[[dict], None] | None = None,
    ) -> None:
        self.accounts = accounts
        self.strategies = strategies
        self._mode = mode
        self._execution_service = execution_service
        self._on_signal = on_signal
        self._on_fill = on_fill
        self._bar_count = 0
        self._snapshot_interval = 100
        self._started = False
        self._running = False

    @property
    def mode(self) -> TradingMode:
        return self._mode

    def switch_mode(self, new_mode: TradingMode) -> None:
        if new_mode == TradingMode.LIVE and self._execution_service is None:
            raise ValueError("Cannot switch to LIVE mode without ExecutionService")
        old = self._mode
        self._mode = new_mode
        logger.info("Trading mode switched: %s → %s", old.value, new_mode.value)

    def _commission_rate(self, market: str) -> float:
        if market == "crypto":
            return 0.0004
        return 0.00005

    async def _execute_signal_paper(
        self, signal: Signal, account: SimAccount, timestamp: str,
    ) -> bool:
        """Paper 模式：本地撮合（复用 PaperScheduler 逻辑）。"""
        symbol = signal.symbol
        price = signal.price or 0
        if price <= 0:
            return False
        comm_rate = self._commission_rate(account.market)

        if signal.signal_type == SignalType.LONG_ENTRY:
            if symbol in account.positions:
                return False
            if account.total_equity < account.initial_capital * 0.70:
                return False
            strength = signal.strength
            alloc_pct = 0.80 if strength >= 0.7 else 0.50 if strength >= 0.4 else 0.25
            alloc_capital = account.capital * alloc_pct
            qty = alloc_capital / price
            if qty <= 0:
                return False
            commission = price * qty * comm_rate
            if account.capital < alloc_capital + commission:
                alloc_capital = account.capital * 0.9
                qty = alloc_capital / price
                commission = price * qty * comm_rate
            account.open_position(
                symbol=symbol, side="long", qty=qty, price=price,
                commission=commission, reason=signal.reason, timestamp=timestamp,
            )
            return True

        elif signal.signal_type == SignalType.SHORT_ENTRY:
            if account.market == "crypto":
                return False
            if symbol in account.positions:
                return False
            if account.total_equity < account.initial_capital * 0.70:
                return False
            strength = signal.strength
            alloc_pct = 0.80 if strength >= 0.7 else 0.50 if strength >= 0.4 else 0.25
            alloc_capital = account.capital * alloc_pct
            qty = alloc_capital / price
            if qty <= 0:
                return False
            commission = price * qty * comm_rate
            account.open_position(
                symbol=symbol, side="short", qty=qty, price=price,
                commission=commission, reason=signal.reason, timestamp=timestamp,
            )
            return True

        elif signal.signal_type in (SignalType.LONG_EXIT, SignalType.SHORT_EXIT):
            if symbol not in account.positions:
                return False
            commission = price * account.positions[symbol].qty * comm_rate
            account.close_position(
                symbol=symbol, price=price, commission=commission,
                reason=signal.reason, timestamp=timestamp,
            )
            return True

        return False

    async def _execute_signal_live(
        self, signal: Signal, account: SimAccount, timestamp: str,
    ) -> bool:
        """Live 模式：通过 ExecutionService 发送到真实交易所。"""
        if self._execution_service is None:
            logger.error("ExecutionService not available in LIVE mode")
            return False

        from core.enums.direction import Direction, Offset

        symbol = signal.symbol
        price = signal.price
        if not price or price <= 0:
            return False

        sig_to_dir_offset = {
            SignalType.LONG_ENTRY: (Direction.LONG, Offset.OPEN),
            SignalType.SHORT_ENTRY: (Direction.SHORT, Offset.OPEN),
            SignalType.LONG_EXIT: (Direction.LONG, Offset.CLOSE),
            SignalType.SHORT_EXIT: (Direction.SHORT, Offset.CLOSE),
        }
        mapping = sig_to_dir_offset.get(signal.signal_type)
        if not mapping:
            return False
        direction, offset = mapping

        strength = signal.strength
        alloc_pct = 0.80 if strength >= 0.7 else 0.50 if strength >= 0.4 else 0.25

        if signal.signal_type in (SignalType.LONG_ENTRY, SignalType.SHORT_ENTRY):
            if symbol in account.positions:
                return False
            alloc_capital = account.capital * alloc_pct
            volume = max(1, int(alloc_capital / price))
        else:
            pos = account.positions.get(symbol)
            if not pos:
                return False
            volume = max(1, int(pos.qty))

        try:
            exchange = symbol.split(".")[0] if "." in symbol else "SHFE"
            order = await self._execution_service.place_order(
                strategy_id=account.strategy_name or f"live_{account.account_id}",
                symbol=symbol,
                exchange=exchange,
                direction=direction,
                offset=offset,
                price=Decimal(str(price)),
                volume=volume,
            )
            logger.info(
                "LIVE order placed: %s %s %s vol=%d price=%s → %s",
                symbol, direction.value, offset.value, volume, price, order.status.value,
            )
            fill_info = {
                "order_id": order.order_id,
                "symbol": symbol,
                "direction": direction.value,
                "offset": offset.value,
                "price": float(price),
                "volume": volume,
                "status": order.status.value,
                "strategy": account.strategy_name,
                "timestamp": timestamp,
            }
            if self._on_fill:
                self._on_fill(fill_info)
            return True

        except Exception as e:
            logger.error("LIVE order failed: %s %s — %s", symbol, signal.signal_type.value, e)
            return False

    async def run_bar(
        self, timestamp: str, market_data: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """处理一根 bar 的数据，驱动所有策略。"""
        if not self._started:
            self._started = True
            self._running = True
            for strategy in self.strategies.values():
                try:
                    await strategy.on_start()
                except Exception as exc:
                    logger.warning("on_start failed for %s: %s", strategy.name, exc)

        self._bar_count += 1
        total_signals = 0
        total_fills = 0

        execute_fn = (
            self._execute_signal_live
            if self._mode == TradingMode.LIVE
            else self._execute_signal_paper
        )

        for account_id, strategy in self.strategies.items():
            account = self.accounts.get(account_id)
            if account is None or not account.is_active:
                continue

            for symbol in list(account.positions.keys()):
                if symbol in market_data:
                    account.update_unrealized(symbol, market_data[symbol]["close"])

            for symbol in strategy.config.symbols:
                bar = market_data.get(symbol)
                if not bar:
                    continue
                try:
                    signals = await strategy.on_bar(symbol, bar)
                except Exception as e:
                    logger.warning(
                        "Strategy %s (account %d) error on %s: %s",
                        strategy.name, account_id, symbol, e,
                    )
                    continue

                for sig in signals:
                    total_signals += 1
                    if self._on_signal:
                        self._on_signal(sig, self._mode.value)
                    executed = await execute_fn(sig, account, timestamp)
                    if executed:
                        total_fills += 1

            if self._bar_count % self._snapshot_interval == 0:
                account.snapshot_equity(timestamp)

        return {
            "bar_index": self._bar_count,
            "timestamp": timestamp,
            "total_signals": total_signals,
            "total_fills": total_fills,
            "mode": self._mode.value,
        }

    async def shutdown(self) -> None:
        self._running = False
        for strategy in self.strategies.values():
            try:
                await strategy.on_stop()
            except Exception as exc:
                logger.warning("on_stop failed for %s: %s", strategy.name, exc)
        logger.info("LiveScheduler shutdown complete")

    def get_status(self) -> dict[str, Any]:
        return {
            "mode": self._mode.value,
            "running": self._running,
            "bar_count": self._bar_count,
            "strategy_count": len(self.strategies),
            "active_accounts": sum(
                1 for a in self.accounts.accounts.values() if a.is_active
            ),
        }
