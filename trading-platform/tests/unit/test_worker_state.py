"""Unit tests for WorkerStateStore checkpoint persistence."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from sim_live.worker_state import WorkerCheckpoint, WorkerStateStore


def _sample_checkpoint(worker_id: str = "test-worker") -> WorkerCheckpoint:
    return WorkerCheckpoint(
        worker_id=worker_id,
        updated_at="2026-07-10T02:00:00+00:00",
        heartbeat_ts="2026-07-10T02:00:00+00:00",
        market="futures",
        mode="paper",
        last_bar_ts="2026-07-10T01:55:00+00:00",
        strategies={
            "1": {
                "strategy_name": "supertrend",
                "positions": {
                    "rb2510": {
                        "side": "long",
                        "qty": 2.0,
                        "avg_price": 3500.0,
                        "unrealized": 120.0,
                    }
                },
                "capital": 950_000.0,
                "total_equity": 1_007_120.0,
                "realized_pnl": 5000.0,
                "last_bar_ts": "2026-07-10T01:55:00+00:00",
            }
        },
    )


def test_checkpoint_save_load_roundtrip(tmp_path: Path) -> None:
    store = WorkerStateStore(tmp_path, "test-worker")
    original = _sample_checkpoint()

    store.save(original)
    loaded = store.load()

    assert loaded is not None
    assert loaded.worker_id == original.worker_id
    assert loaded.last_bar_ts == original.last_bar_ts
    assert loaded.strategies["1"]["positions"]["rb2510"]["qty"] == 2.0
    assert loaded.strategies["1"]["realized_pnl"] == 5000.0
    assert loaded.strategies["1"]["total_equity"] == 1_007_120.0


def test_atomic_write_no_partial_main_file(tmp_path: Path) -> None:
    store = WorkerStateStore(tmp_path, "atomic-worker")
    main_path = store.checkpoint_path
    real_open = open

    def guarded_open(path: str | Path, mode: str = "r", *args, **kwargs):
        if Path(path) == main_path and "w" in mode:
            raise AssertionError("Must not write main checkpoint directly")
        return real_open(path, mode, *args, **kwargs)

    with patch("builtins.open", side_effect=guarded_open):
        store.save(_sample_checkpoint("atomic-worker"))

    assert main_path.exists()
    json.loads(main_path.read_text())


def test_corrupt_checkpoint_backs_up_and_returns_empty(tmp_path: Path) -> None:
    store = WorkerStateStore(tmp_path, "corrupt-worker")
    main_path = store.checkpoint_path
    main_path.parent.mkdir(parents=True, exist_ok=True)
    main_path.write_text("{not valid json", encoding="utf-8")

    loaded = store.load()

    assert loaded is None
    corrupt_backup = main_path.with_suffix(".json.corrupt")
    assert corrupt_backup.exists()
    assert not main_path.exists()


def test_touch_heartbeat_creates_file(tmp_path: Path) -> None:
    store = WorkerStateStore(tmp_path, "hb-worker")
    store.touch_heartbeat()
    assert store.heartbeat_path.exists()
    assert store.heartbeat_path.stat().st_mtime > 0
