"""Unit tests for strategy worker config and lifecycle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from sim_live.strategy_worker import (
    StrategyWorker,
    WorkerConfig,
    build_strategies_from_config,
    load_worker_config,
    validate_startup_mode,
)
from sim_live.worker_state import WorkerCheckpoint, WorkerStateStore
from strategy.base import BaseStrategy, Signal, StrategyConfig
from strategy.registry import StrategyRegistry


class DummyWorkerStrategy(BaseStrategy):
    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        return []

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        return []


@pytest.fixture(autouse=True)
def _register_dummy_strategy() -> None:
    StrategyRegistry.register("dummy", DummyWorkerStrategy)
    yield
    StrategyRegistry.unregister("dummy")


def _config_dict(**overrides: Any) -> dict[str, Any]:
    base = {
        "worker_id": "paper-futures-01",
        "market": "futures",
        "mode": "paper",
        "checkpoint_interval_s": 60,
        "interval": "5m",
        "strategies": [
            {
                "name": "dummy",
                "params": {"alpha": 1},
                "symbols": ["rb2510", "cu2509"],
            }
        ],
    }
    base.update(overrides)
    return base


def test_load_worker_config_parses_fields(tmp_path: Path) -> None:
    cfg_path = tmp_path / "strategy_worker.json"
    cfg_path.write_text(json.dumps(_config_dict()), encoding="utf-8")

    cfg = load_worker_config(cfg_path)

    assert isinstance(cfg, WorkerConfig)
    assert cfg.worker_id == "paper-futures-01"
    assert cfg.market == "futures"
    assert cfg.mode == "paper"
    assert cfg.checkpoint_interval_s == 60
    assert len(cfg.strategies) == 1
    assert cfg.strategies[0].name == "dummy"
    assert cfg.strategies[0].symbols == ["rb2510", "cu2509"]


def test_build_strategies_from_config(tmp_path: Path) -> None:
    cfg = WorkerConfig(**_config_dict())
    strategies, accounts = build_strategies_from_config(cfg)

    assert len(strategies) == 1
    strategy = strategies[1]
    assert strategy.name == "dummy"
    assert strategy.config.symbols == ["rb2510", "cu2509"]
    account = accounts.get(1)
    assert account is not None
    assert account.strategy_name == "dummy"
    assert account.market == "futures"


def test_live_mode_rejected_without_allow_live() -> None:
    cfg = WorkerConfig(**_config_dict(mode="live"))

    with pytest.raises(SystemExit):
        validate_startup_mode(cfg, allow_live=False)


def test_live_mode_allowed_with_flag() -> None:
    cfg = WorkerConfig(**_config_dict(mode="live"))
    validate_startup_mode(cfg, allow_live=True)


def test_sigterm_handler_writes_final_checkpoint(tmp_path: Path) -> None:
    cfg = WorkerConfig(**_config_dict())
    store = WorkerStateStore(tmp_path, cfg.worker_id)
    worker = StrategyWorker(cfg, state_dir=tmp_path, feed_factory=lambda _cfg: MagicMock())

    worker._restore_from_checkpoint = MagicMock(return_value=None)  # type: ignore[method-assign]
    worker._build_runtime = MagicMock()  # type: ignore[method-assign]

    checkpoint = WorkerCheckpoint(
        worker_id=cfg.worker_id,
        updated_at="2026-07-10T02:00:00+00:00",
        heartbeat_ts="2026-07-10T02:00:00+00:00",
        market="futures",
        mode="paper",
        last_bar_ts="2026-07-10T01:55:00+00:00",
        strategies={
            "1": {
                "strategy_name": "dummy",
                "positions": {},
                "capital": 1_000_000.0,
                "total_equity": 1_000_000.0,
                "realized_pnl": 0.0,
                "last_bar_ts": "2026-07-10T01:55:00+00:00",
            }
        },
    )
    worker._collect_checkpoint = MagicMock(return_value=checkpoint)  # type: ignore[method-assign]

    worker.handle_shutdown_signal()

    loaded = store.load()
    assert loaded is not None
    assert loaded.last_bar_ts == "2026-07-10T01:55:00+00:00"
    assert loaded.strategies["1"]["strategy_name"] == "dummy"
