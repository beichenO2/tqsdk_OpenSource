"""Worker checkpoint persistence — atomic JSON writes and recovery."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class WorkerCheckpoint:
    """Serializable worker state snapshot."""

    worker_id: str
    updated_at: str
    heartbeat_ts: str
    market: str
    mode: str
    last_bar_ts: str | None = None
    strategies: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "updated_at": self.updated_at,
            "heartbeat_ts": self.heartbeat_ts,
            "market": self.market,
            "mode": self.mode,
            "last_bar_ts": self.last_bar_ts,
            "strategies": self.strategies,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkerCheckpoint:
        return cls(
            worker_id=data["worker_id"],
            updated_at=data["updated_at"],
            heartbeat_ts=data["heartbeat_ts"],
            market=data["market"],
            mode=data["mode"],
            last_bar_ts=data.get("last_bar_ts"),
            strategies=data.get("strategies", {}),
        )


class WorkerStateStore:
    """Persist worker checkpoints under data/worker_state/{worker_id}.json."""

    def __init__(self, state_dir: str | Path, worker_id: str) -> None:
        self.state_dir = Path(state_dir)
        self.worker_id = worker_id
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.state_dir / f"{worker_id}.json"
        self.heartbeat_path = self.state_dir / f"{worker_id}.heartbeat"

    def save(self, checkpoint: WorkerCheckpoint) -> None:
        """Atomically write checkpoint via tmp file + rename."""
        tmp_path = self.checkpoint_path.with_suffix(".json.tmp")
        payload = json.dumps(checkpoint.to_dict(), indent=2, ensure_ascii=False)
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self.checkpoint_path)
        logger.debug("Checkpoint saved: %s", self.checkpoint_path)

    def load(self) -> WorkerCheckpoint | None:
        """Load checkpoint; corrupt files are backed up and ignored."""
        if not self.checkpoint_path.exists():
            return None

        try:
            with open(self.checkpoint_path, encoding="utf-8") as f:
                data = json.load(f)
            return WorkerCheckpoint.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            corrupt_path = self.checkpoint_path.with_suffix(".json.corrupt")
            logger.error(
                "Corrupt checkpoint for %s: %s — backing up to %s",
                self.worker_id,
                exc,
                corrupt_path.name,
            )
            if self.checkpoint_path.exists():
                os.replace(self.checkpoint_path, corrupt_path)
            return None

    def touch_heartbeat(self) -> None:
        """Update heartbeat file mtime (external liveness probe)."""
        self.heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
        self.heartbeat_path.touch()
        now = time.time()
        os.utime(self.heartbeat_path, (now, now))

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()
