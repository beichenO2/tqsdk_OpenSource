"""Single TqApi session — credentials stay in this process only."""

from __future__ import annotations

import logging
import threading
import time
from decimal import Decimal
from typing import Any

logger = logging.getLogger(__name__)


class SessionBusyError(RuntimeError):
    """Raised when the TqApi lock cannot be acquired in time (closed market)."""


class _TimedLock:
    """RLock guard with acquire timeout — prevents thread-pool starvation
    when a TqSdk call blocks forever during closed market."""

    def __init__(self, lock: threading.RLock, timeout: float) -> None:
        self._lock = lock
        self._timeout = timeout

    def __enter__(self):
        if not self._lock.acquire(timeout=self._timeout):
            raise SessionBusyError(
                f"TqSdk session busy (lock not acquired in {self._timeout}s; market closed?)"
            )
        return self

    def __exit__(self, *exc) -> None:
        self._lock.release()


class TqSdkSession:
    """Thread-safe wrapper around one TqApi instance."""

    LOCK_TIMEOUT_S = 5.0

    def __init__(self) -> None:
        self._api: Any = None
        self._lock = threading.RLock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._connected = False
        self._account_mode = "unknown"

    def _locked(self) -> _TimedLock:
        return _TimedLock(self._lock, self.LOCK_TIMEOUT_S)

    @property
    def connected(self) -> bool:
        return self._connected and self._api is not None

    @property
    def account_mode(self) -> str:
        return self._account_mode

    def connect(self, creds: dict[str, str]) -> None:
        if self.connected:
            return
        from tqsdk import TqAccount, TqApi, TqAuth, TqSim

        auth = TqAuth(creds["auth_user"], creds["auth_password"])
        mode = creds.get("mode", "live").lower()
        if mode in ("sim", "tqsim") or not creds.get("broker_id"):
            account = TqSim(init_balance=1_000_000)
            self._account_mode = "tqsim"
        else:
            account = TqAccount(
                creds["broker_id"],
                creds["account_id"],
                creds.get("password", ""),
            )
            self._account_mode = f"live:{creds['broker_id']}"

        self._api = TqApi(account=account, auth=auth)
        self._running = True
        self._thread = threading.Thread(target=self._update_loop, name="tqsdk-wait-update", daemon=True)
        self._thread.start()
        self._connected = True
        logger.info("TqSdk session connected (%s)", self._account_mode)

    def disconnect(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        with self._lock:
            if self._api is not None:
                try:
                    self._api.close()
                except Exception:
                    logger.exception("error closing TqApi")
                self._api = None
        self._connected = False
        logger.info("TqSdk session disconnected")

    def _update_loop(self) -> None:
        """Drive TqSdk updates without holding the API lock.

        Holding ``_lock`` across ``wait_update`` starves HTTP handlers during
        both closed and open sessions (real TqSdk can exceed the deadline).
        """
        while self._running and self._api is not None:
            try:
                api = self._api
                if api is not None:
                    api.wait_update(deadline=time.time() + 1)
            except Exception:
                if self._running:
                    time.sleep(0.5)

    def get_account_info(self) -> dict[str, float]:
        with self._locked():
            if self._api is None:
                raise RuntimeError("TqSdk not connected")
            acc = self._api.get_account()
            return {
                "balance": float(acc.balance),
                "available": float(acc.available),
                "margin": float(acc.margin),
                "float_profit": float(acc.float_profit),
                "commission": float(acc.commission),
            }

    def get_positions(self) -> list[dict[str, Any]]:
        with self._locked():
            if self._api is None:
                raise RuntimeError("TqSdk not connected")
            result: list[dict[str, Any]] = []
            positions = self._api.get_position()
            for symbol, pos in positions.items():
                if pos.pos_long > 0:
                    result.append({
                        "symbol": symbol,
                        "direction": "LONG",
                        "volume": int(pos.pos_long),
                        "available": int(pos.pos_long - pos.pos_long_his),
                        "float_pnl": float(pos.float_profit_long),
                    })
                if pos.pos_short > 0:
                    result.append({
                        "symbol": symbol,
                        "direction": "SHORT",
                        "volume": int(pos.pos_short),
                        "available": int(pos.pos_short - pos.pos_short_his),
                        "float_pnl": float(pos.float_profit_short),
                    })
            return result

    def place_order(
        self,
        symbol: str,
        direction: str,
        offset: str,
        price: float,
        volume: int,
    ) -> str:
        with self._locked():
            if self._api is None:
                raise RuntimeError("TqSdk not connected")
            order = self._api.insert_order(
                symbol=symbol,
                direction=direction,
                offset=offset,
                limit_price=price,
                volume=volume,
            )
            return str(order.order_id)

    def cancel_order(self, order_id: str) -> bool:
        with self._locked():
            if self._api is None:
                return False
            self._api.cancel_order(order_id)
            return True

    def get_quote(self, symbol: str) -> dict[str, Any]:
        with self._locked():
            if self._api is None:
                raise RuntimeError("TqSdk not connected")
            q = self._api.get_quote(symbol)
            raw_dt = q.datetime
            dt_val: int | None
            try:
                dt_val = int(raw_dt) if raw_dt not in (None, "", "NaN") else None
            except (TypeError, ValueError):
                dt_val = None
            return {
                "symbol": symbol,
                "datetime": dt_val,
                "last_price": float(q.last_price) if q.last_price == q.last_price else 0.0,
                "highest": float(q.highest) if q.highest == q.highest else 0.0,
                "lowest": float(q.lowest) if q.lowest == q.lowest else 0.0,
                "volume": int(q.volume) if q.volume else 0,
                "amount": float(q.amount) if q.amount == q.amount else 0.0,
                "open_interest": int(q.open_interest) if q.open_interest else 0,
                "bid_price1": float(q.bid_price1) if q.bid_price1 else None,
                "bid_volume1": int(q.bid_volume1) if q.bid_volume1 else None,
                "ask_price1": float(q.ask_price1) if q.ask_price1 else None,
                "ask_volume1": int(q.ask_volume1) if q.ask_volume1 else None,
            }

    def get_klines(self, symbol: str, duration: int, length: int) -> list[dict[str, Any]]:
        with self._locked():
            if self._api is None:
                raise RuntimeError("TqSdk not connected")
            klines = self._api.get_kline_serial(symbol, duration, length)
            rows: list[dict[str, Any]] = []
            for _, row in klines.iterrows():
                rows.append({
                    "datetime": int(row["datetime"]),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": int(row["volume"]),
                    "open_interest": int(row["close_oi"]) if "close_oi" in row else None,
                })
            return rows

    def list_instruments(self, exchange_id: str | None = None, ins_class: str = "FUTURE") -> list[str]:
        with self._locked():
            if self._api is None:
                raise RuntimeError("TqSdk not connected")
            kwargs: dict[str, Any] = {"ins_class": ins_class}
            if exchange_id:
                kwargs["exchange_id"] = exchange_id
            return list(self._api.query_quotes(**kwargs))


_session: TqSdkSession | None = None


def get_session() -> TqSdkSession:
    global _session
    if _session is None:
        _session = TqSdkSession()
    return _session
