"""SQLite 持久化层 — 交易日志、订单记录、净值快照。

使用 WAL 模式提高并发读写性能。
服务启动时从 DB 恢复状态；运行时实时写入。
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_DB = Path(__file__).resolve().parents[2] / "data" / "trading.db"


class TradingDB:
    """交易数据持久化。"""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._path = Path(db_path) if db_path else _DEFAULT_DB
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> None:
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._create_tables()
        logger.info("TradingDB connected: %s", self._path)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def _create_tables(self) -> None:
        assert self._conn is not None
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                strategy_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                exchange TEXT,
                direction TEXT NOT NULL,
                offset_type TEXT NOT NULL,
                price REAL NOT NULL,
                volume INTEGER NOT NULL,
                filled_volume INTEGER DEFAULT 0,
                status TEXT NOT NULL,
                mode TEXT DEFAULT 'paper',
                created_at TEXT NOT NULL,
                updated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS fills (
                fill_id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT,
                strategy_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                price REAL NOT NULL,
                volume INTEGER NOT NULL,
                commission REAL DEFAULT 0,
                pnl REAL DEFAULT 0,
                mode TEXT DEFAULT 'paper',
                filled_at TEXT NOT NULL,
                FOREIGN KEY (order_id) REFERENCES orders(order_id)
            );

            CREATE TABLE IF NOT EXISTS equity_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_id INTEGER NOT NULL,
                strategy_name TEXT,
                equity REAL NOT NULL,
                capital REAL NOT NULL,
                unrealized_pnl REAL DEFAULT 0,
                position_count INTEGER DEFAULT 0,
                snapshot_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS strategy_sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id TEXT NOT NULL,
                strategy_name TEXT,
                mode TEXT DEFAULT 'paper',
                market TEXT,
                symbols TEXT,
                params TEXT,
                started_at TEXT NOT NULL,
                stopped_at TEXT,
                total_signals INTEGER DEFAULT 0,
                total_fills INTEGER DEFAULT 0,
                final_return_pct REAL
            );

            CREATE INDEX IF NOT EXISTS idx_orders_strategy ON orders(strategy_id);
            CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol);
            CREATE INDEX IF NOT EXISTS idx_fills_strategy ON fills(strategy_id);
            CREATE INDEX IF NOT EXISTS idx_equity_account ON equity_snapshots(account_id);
            CREATE INDEX IF NOT EXISTS idx_equity_time ON equity_snapshots(snapshot_at);
        """)

    def record_order(
        self, order_id: str, strategy_id: str, symbol: str,
        direction: str, offset_type: str, price: float,
        volume: int, status: str, mode: str = "paper",
        exchange: str = "",
    ) -> None:
        assert self._conn is not None
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """INSERT OR REPLACE INTO orders
               (order_id, strategy_id, symbol, exchange, direction, offset_type,
                price, volume, status, mode, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (order_id, strategy_id, symbol, exchange, direction, offset_type,
             price, volume, status, mode, now, now),
        )
        self._conn.commit()

    def update_order_status(self, order_id: str, status: str, filled_volume: int = 0) -> None:
        assert self._conn is not None
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "UPDATE orders SET status=?, filled_volume=?, updated_at=? WHERE order_id=?",
            (status, filled_volume, now, order_id),
        )
        self._conn.commit()

    def record_fill(
        self, order_id: str | None, strategy_id: str, symbol: str,
        direction: str, price: float, volume: int,
        commission: float = 0, pnl: float = 0, mode: str = "paper",
    ) -> int:
        assert self._conn is not None
        now = datetime.now(timezone.utc).isoformat()
        cursor = self._conn.execute(
            """INSERT INTO fills
               (order_id, strategy_id, symbol, direction, price, volume,
                commission, pnl, mode, filled_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (order_id, strategy_id, symbol, direction, price, volume,
             commission, pnl, mode, now),
        )
        self._conn.commit()
        return cursor.lastrowid or 0

    def record_equity_snapshot(
        self, account_id: int, equity: float, capital: float,
        unrealized_pnl: float = 0, position_count: int = 0,
        strategy_name: str = "",
    ) -> None:
        assert self._conn is not None
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """INSERT INTO equity_snapshots
               (account_id, strategy_name, equity, capital, unrealized_pnl,
                position_count, snapshot_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (account_id, strategy_name, equity, capital, unrealized_pnl,
             position_count, now),
        )
        self._conn.commit()

    def start_session(
        self, strategy_id: str, strategy_name: str, mode: str,
        market: str, symbols: list[str], params: dict[str, Any],
    ) -> int:
        assert self._conn is not None
        now = datetime.now(timezone.utc).isoformat()
        cursor = self._conn.execute(
            """INSERT INTO strategy_sessions
               (strategy_id, strategy_name, mode, market, symbols, params, started_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (strategy_id, strategy_name, mode, market,
             json.dumps(symbols), json.dumps(params), now),
        )
        self._conn.commit()
        return cursor.lastrowid or 0

    def end_session(
        self, session_id: int, total_signals: int = 0,
        total_fills: int = 0, final_return_pct: float = 0,
    ) -> None:
        assert self._conn is not None
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """UPDATE strategy_sessions
               SET stopped_at=?, total_signals=?, total_fills=?, final_return_pct=?
               WHERE session_id=?""",
            (now, total_signals, total_fills, final_return_pct, session_id),
        )
        self._conn.commit()

    def get_recent_orders(self, limit: int = 100) -> list[dict[str, Any]]:
        assert self._conn is not None
        rows = self._conn.execute(
            "SELECT * FROM orders ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_recent_fills(self, limit: int = 100) -> list[dict[str, Any]]:
        assert self._conn is not None
        rows = self._conn.execute(
            "SELECT * FROM fills ORDER BY filled_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_equity_curve(self, account_id: int, limit: int = 500) -> list[dict[str, Any]]:
        assert self._conn is not None
        rows = self._conn.execute(
            """SELECT snapshot_at, equity, capital, unrealized_pnl
               FROM equity_snapshots WHERE account_id=?
               ORDER BY snapshot_at DESC LIMIT ?""",
            (account_id, limit),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def get_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        assert self._conn is not None
        rows = self._conn.execute(
            "SELECT * FROM strategy_sessions ORDER BY started_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
