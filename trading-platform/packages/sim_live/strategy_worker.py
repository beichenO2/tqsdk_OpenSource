"""Strategy worker — config-driven standalone paper/live trading process."""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from strategy.base import BaseStrategy, StrategyConfig
from strategy.registry import StrategyRegistry

from .account_manager import AccountManager, SimAccount
from .live_feed import TqGatewayLiveFeed
from .live_scheduler import LiveScheduler, TradingMode
from .worker_state import WorkerCheckpoint, WorkerStateStore

logger = logging.getLogger(__name__)

DEFAULT_STATE_DIR = Path("data/worker_state")


@dataclass
class StrategyEntry:
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    symbols: list[str] = field(default_factory=list)


@dataclass
class WorkerConfig:
    worker_id: str
    market: str
    mode: str
    strategies: list[StrategyEntry]
    checkpoint_interval_s: int = 60
    interval: str = "5m"
    gateway_url: str | None = None

    def __post_init__(self) -> None:
        normalized: list[StrategyEntry] = []
        for entry in self.strategies:
            if isinstance(entry, StrategyEntry):
                normalized.append(entry)
            elif isinstance(entry, dict):
                normalized.append(StrategyEntry(**entry))
            else:
                raise TypeError(f"Invalid strategy entry: {entry!r}")
        self.strategies = normalized


def load_worker_config(path: str | Path) -> WorkerConfig:
    """Load and validate worker JSON config."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    strategies = [
        StrategyEntry(
            name=s["name"],
            params=s.get("params", {}),
            symbols=s.get("symbols", []),
        )
        for s in raw.get("strategies", [])
    ]
    return WorkerConfig(
        worker_id=raw["worker_id"],
        market=raw["market"],
        mode=raw["mode"],
        strategies=strategies,
        checkpoint_interval_s=raw.get("checkpoint_interval_s", 60),
        interval=raw.get("interval", "5m"),
        gateway_url=raw.get("gateway_url"),
    )


def validate_startup_mode(config: WorkerConfig, *, allow_live: bool) -> None:
    """Reject live mode until M2 fill-feedback loop is wired."""
    if config.mode == "live" and not allow_live:
        logger.error(
            "LIVE mode is not enabled in this worker build. "
            "Use paper mode, or pass --allow-live after M2 execution feedback is verified."
        )
        sys.exit(1)


def build_strategies_from_config(
    config: WorkerConfig,
) -> tuple[dict[int, BaseStrategy], AccountManager]:
    """Instantiate strategies and accounts from worker config."""
    strategies: dict[int, BaseStrategy] = {}
    accounts = AccountManager(
        crypto_count=len(config.strategies) if config.market == "crypto" else 0,
        futures_count=len(config.strategies) if config.market == "futures" else 0,
        crypto_capital=100_000.0,
        futures_capital=1_000_000.0,
    )

    for idx, entry in enumerate(config.strategies, start=1):
        cls = StrategyRegistry.get(entry.name)
        if cls is None:
            from .observer_strategy import ObserverStrategy

            logger.warning("Strategy '%s' not registered — using ObserverStrategy", entry.name)
            cls = ObserverStrategy

        strat_config = StrategyConfig(
            strategy_id=f"worker_{config.worker_id}_{idx:03d}",
            name=entry.name,
            symbols=entry.symbols,
            params=entry.params,
        )
        strategies[idx] = cls(strat_config)
        accounts.assign_strategy(idx, entry.name, entry.params)

    return strategies, accounts


def _realized_pnl(account: SimAccount) -> float:
    return sum(t.pnl for t in account.trades if t.pnl != 0)


def _strategy_state(account: SimAccount, last_bar_ts: str | None) -> dict[str, Any]:
    return {
        "strategy_name": account.strategy_name,
        "positions": {
            sym: {
                "side": pos.side,
                "qty": pos.qty,
                "avg_price": pos.avg_price,
                "unrealized": round(pos.unrealized_pnl, 2),
            }
            for sym, pos in account.positions.items()
        },
        "capital": round(account.capital, 2),
        "total_equity": round(account.total_equity, 2),
        "realized_pnl": round(_realized_pnl(account), 2),
        "last_bar_ts": last_bar_ts,
    }


class StrategyWorker:
    """Standalone strategy worker with checkpoint/heartbeat lifecycle."""

    def __init__(
        self,
        config: WorkerConfig,
        *,
        state_dir: str | Path = DEFAULT_STATE_DIR,
        feed_factory: Callable[[WorkerConfig], Any] | None = None,
        allow_live: bool = False,
    ) -> None:
        self.config = config
        self.allow_live = allow_live
        self.store = WorkerStateStore(state_dir, config.worker_id)
        self._feed_factory = feed_factory or _default_feed_factory
        self._scheduler: LiveScheduler | None = None
        self._feed: Any = None
        self._last_bar_ts: str | None = None
        self._running = False
        self._checkpoint_task: asyncio.Task[None] | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._shutdown_requested = False

    def _collect_checkpoint(self) -> WorkerCheckpoint:
        now = WorkerStateStore.now_iso()
        strategies: dict[str, dict[str, Any]] = {}
        if self._scheduler is not None:
            for account_id, strategy in self._scheduler.strategies.items():
                account = self._scheduler.accounts.get(account_id)
                if account is None:
                    continue
                strategies[str(account_id)] = _strategy_state(account, self._last_bar_ts)

        return WorkerCheckpoint(
            worker_id=self.config.worker_id,
            updated_at=now,
            heartbeat_ts=now,
            market=self.config.market,
            mode=self.config.mode,
            last_bar_ts=self._last_bar_ts,
            strategies=strategies,
        )

    def _restore_from_checkpoint(self) -> WorkerCheckpoint | None:
        checkpoint = self.store.load()
        if checkpoint is None:
            return None
        self._last_bar_ts = checkpoint.last_bar_ts
        return checkpoint

    def _apply_checkpoint_to_accounts(self, checkpoint: WorkerCheckpoint) -> None:
        if self._scheduler is None:
            return
        for account_id_str, state in checkpoint.strategies.items():
            account = self._scheduler.accounts.get(int(account_id_str))
            if account is None:
                continue
            account.capital = float(state.get("capital", account.capital))
            account.positions.clear()
            from .account_manager import AccountPosition

            for sym, pos in state.get("positions", {}).items():
                account.positions[sym] = AccountPosition(
                    symbol=sym,
                    side=pos.get("side", ""),
                    qty=float(pos.get("qty", 0)),
                    avg_price=float(pos.get("avg_price", 0)),
                    unrealized_pnl=float(pos.get("unrealized", 0)),
                )

    def _build_runtime(self) -> None:
        strategies, accounts = build_strategies_from_config(self.config)
        trading_mode = TradingMode.LIVE if self.config.mode == "live" else TradingMode.PAPER
        # TODO(M2): wire ExecutionService + fill feedback for live mode when --allow-live.
        execution_service = None
        if trading_mode == TradingMode.LIVE:
            logger.warning(
                "LIVE mode started with --allow-live; exchange execution not yet connected."
            )

        self._scheduler = LiveScheduler(
            accounts=accounts,
            strategies=strategies,
            mode=trading_mode,
            execution_service=execution_service,
        )
        self._feed = self._feed_factory(self.config)

    async def _on_bar(self, symbol: str, bar: dict[str, Any]) -> None:
        if self._scheduler is None:
            return
        ts = bar.get("timestamp", "")
        self._last_bar_ts = ts
        market_data = {symbol: bar}
        await self._scheduler.run_bar(ts, market_data)

    async def _checkpoint_loop(self) -> None:
        interval = max(self.config.checkpoint_interval_s, 1)
        while self._running:
            await asyncio.sleep(interval)
            if not self._running:
                break
            self.store.save(self._collect_checkpoint())

    async def _heartbeat_loop(self) -> None:
        while self._running:
            self.store.touch_heartbeat()
            await asyncio.sleep(30)

    def handle_shutdown_signal(self) -> None:
        """SIGTERM/SIGINT handler — persist final checkpoint."""
        self._shutdown_requested = True
        self.store.save(self._collect_checkpoint())
        logger.info("Final checkpoint written for worker %s", self.config.worker_id)

    def _register_signal_handlers(self) -> None:
        def _handler(signum: int, _frame: Any) -> None:
            logger.info("Received signal %s — shutting down gracefully", signum)
            self.handle_shutdown_signal()
            raise SystemExit(0)

        signal.signal(signal.SIGTERM, _handler)
        signal.signal(signal.SIGINT, _handler)

    async def run(self) -> None:
        validate_startup_mode(self.config, allow_live=self.allow_live)
        self._register_signal_handlers()

        checkpoint = self._restore_from_checkpoint()
        self._build_runtime()
        if checkpoint is not None:
            self._apply_checkpoint_to_accounts(checkpoint)
            logger.info("Restored checkpoint for worker %s", self.config.worker_id)

        assert self._feed is not None
        self._feed._on_bar = self._on_bar  # type: ignore[attr-defined]

        self._running = True
        self.store.save(self._collect_checkpoint())
        self._checkpoint_task = asyncio.create_task(self._checkpoint_loop())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        try:
            await self._feed.start()
        finally:
            self._running = False
            if self._checkpoint_task:
                self._checkpoint_task.cancel()
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
            if self._scheduler:
                await self._scheduler.shutdown()
            if hasattr(self._feed, "stop"):
                await self._feed.stop()
            if not self._shutdown_requested:
                self.store.save(self._collect_checkpoint())


def _default_feed_factory(config: WorkerConfig) -> Any:
    symbols: list[str] = []
    for entry in config.strategies:
        symbols.extend(entry.symbols)
    symbols = list(dict.fromkeys(symbols))

    if config.market == "futures":
        return TqGatewayLiveFeed(
            symbols=symbols,
            interval=config.interval,
            gateway_url=config.gateway_url,
        )
    if config.market == "crypto":
        from .realtime_feed import BinanceKlineFeed

        return BinanceKlineFeed(
            symbols=symbols,
            interval=config.interval,
        )
    raise ValueError(f"Unsupported market: {config.market}")
