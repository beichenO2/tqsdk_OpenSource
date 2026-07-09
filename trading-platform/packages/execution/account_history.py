"""Account equity history — persist snapshots so closed-market UI shows the
last trading session's curve (frozen) instead of live-querying the gateway.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_PATH = _REPO / "data" / "account_history.jsonl"

# keep at most one snapshot per minute
_MIN_INTERVAL_S = 55.0


class AccountHistoryStore:
    """Append-only JSONL of account equity snapshots."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DEFAULT_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._last_ts: float = 0.0

    def append(self, account: dict[str, Any]) -> bool:
        """Record a snapshot; returns False when throttled or unusable."""
        now = time.time()
        if now - self._last_ts < _MIN_INTERVAL_S:
            return False
        balance = account.get("balance") or account.get("equity")
        if balance in (None, "", 0, "0"):
            return False
        row = {
            "ts": round(now, 1),
            "balance": float(balance),
            "available": float(account.get("available") or 0),
            "float_pnl": float(account.get("float_pnl") or account.get("float_profit") or 0),
        }
        try:
            with open(self.path, "a") as f:
                f.write(json.dumps(row) + "\n")
            self._last_ts = now
            return True
        except OSError as e:
            logger.warning("account history append failed: %s", e)
            return False

    def load(self, days: int = 30, max_points: int = 500) -> list[dict[str, Any]]:
        """Load snapshots within *days*, downsampled to *max_points*."""
        if not self.path.exists():
            return []
        cutoff = time.time() - days * 86400
        rows: list[dict[str, Any]] = []
        try:
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if float(row.get("ts", 0)) >= cutoff:
                        rows.append(row)
        except OSError as e:
            logger.warning("account history load failed: %s", e)
            return []

        if len(rows) > max_points:
            step = len(rows) / max_points
            rows = [rows[int(i * step)] for i in range(max_points)]

        out: list[dict[str, Any]] = []
        for r in rows:
            ts = float(r["ts"])
            out.append({
                "ts": ts,
                "date": time.strftime("%m-%d %H:%M", time.localtime(ts)),
                "pnl": r.get("balance", 0.0),
                "float_pnl": r.get("float_pnl", 0.0),
            })
        return out
