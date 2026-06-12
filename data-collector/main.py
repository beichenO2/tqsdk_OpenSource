#!/usr/bin/env python3
"""TqSdk Data Collector — long-running service managed by SOTAgent.

Architecture:
  Main thread   → orchestrator + health server + crypto kline poller
  Thread 1      → TqSdk wait_update() loop (futures tick recording + kline snapshots)
  Thread 2      → Binance WebSocket (crypto trade recording)
  Daemon threads→ TickBuffer flush timers (futures + crypto)
"""

from __future__ import annotations

import http.cookiejar
import json
import logging
import os
import signal
import sys
import time
import urllib.request
import urllib.parse
from datetime import datetime
from threading import Thread, Event

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("data-collector")

PRIVPORTAL_URL = os.getenv("PRIVPORTAL_URL", "http://127.0.0.1:12790")
SERVICE_NAME = "tqsdk-data-collector"


try:
    from polarisor_port_sdk import submit_task as _sdk_submit, complete_task as _sdk_complete
except ImportError:
    _sdk_submit = _sdk_complete = None

_session_task_id: str | None = None

CRYPTO_KLINE_INTERVAL = 600   # 10 min
FUTURES_KLINE_INTERVAL = 300   # 5 min
PP_RETRY_INTERVAL = 10
PP_MAX_RETRIES = 30

_shutdown = Event()


def _handle_signal(signum: int, frame: object) -> None:
    logger.info("received signal %d, shutting down gracefully...", signum)
    _shutdown.set()


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


# ═══════════════════════════════════════════════════════════════
# PolarPrivate client (unchanged)
# ═══════════════════════════════════════════════════════════════

class PrivPortalClient:
    """PolarPrivate API client with cookie-based session management."""

    def __init__(self, base_url: str = PRIVPORTAL_URL) -> None:
        self.base_url = base_url
        self._cj = http.cookiejar.CookieJar()
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            urllib.request.HTTPCookieProcessor(self._cj),
        )

    def request(self, method: str, path: str, body: dict | None = None) -> dict:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode() if body else None
        headers = {"Content-Type": "application/json"} if data else {}
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        with self._opener.open(req, timeout=15) as resp:
            return json.loads(resp.read().decode())

    def wait_for_availability(self, max_retries: int = PP_MAX_RETRIES) -> bool:
        for attempt in range(1, max_retries + 1):
            if _shutdown.is_set():
                return False
            try:
                self.request("GET", "/health")
                logger.info("PolarPrivate is available (attempt %d)", attempt)
                return True
            except Exception:
                logger.info("waiting for PolarPrivate... (attempt %d/%d)", attempt, max_retries)
                time.sleep(PP_RETRY_INTERVAL)
        return False

    def grant_d_class_secrets(self) -> dict[str, str] | None:
        """Request TqSdk credentials via D-class controlled grant (260505 batch).

        Replaces the legacy reveal-based fetch. Requires the data-collector binary
        to be present in ~/.privportal/d-class-allowlist.json under service_name
        "tqsdk-login" (or another configured entry).
        """
        try:
            import hashlib
            import sys

            sha = hashlib.sha256()
            with open(sys.executable, "rb") as f:
                for chunk in iter(lambda: f.read(64 * 1024), b""):
                    sha.update(chunk)
            payload = {
                "service_name": "tqsdk-login",
                "caller_executable_sha256": sha.hexdigest(),
            }
            result = self.request("POST", "/api/d-class/grant", payload)
            secrets = result.get("secrets", {}) if isinstance(result, dict) else {}
            return secrets if secrets else None
        except urllib.error.HTTPError as e:
            body = e.read().decode() if hasattr(e, "read") else ""
            logger.error("d-class grant failed: %d %s", e.code, body)
            return None


def fetch_tqsdk_credentials(client: PrivPortalClient) -> dict[str, str] | None:
    secrets = client.grant_d_class_secrets()
    if not secrets:
        logger.error(
            "Failed to obtain TqSdk credentials via D-class grant. "
            "Add this binary's SHA256 to ~/.privportal/d-class-allowlist.json "
            "under service_name 'tqsdk-login' with allowed_secret_keys "
            "[exchange.tqsdk.auth_user, exchange.tqsdk.auth_password]."
        )
        return None
    auth_user = secrets.get("exchange.tqsdk.auth_user")
    auth_password = secrets.get("exchange.tqsdk.auth_password")
    if not auth_user or not auth_password:
        logger.error("D-class allowlist missing required keys for tqsdk-login")
        return None
    logger.info("TqSdk credentials fetched via D-class: auth_user=%s", auth_user)
    return {"auth_user": auth_user, "auth_password": auth_password}


# ═══════════════════════════════════════════════════════════════
# Futures tick recorder thread
# ═══════════════════════════════════════════════════════════════

def run_futures_tick_thread(creds: dict[str, str], tick_buf: "TickBuffer") -> None:
    """TqSdk event loop: subscribe to all futures, record ticks + periodic kline snapshots."""
    from collector import FUTURES_SYMBOLS, collect_futures_klines

    try:
        from tqsdk import TqApi, TqAuth, TqSim
    except ImportError:
        logger.error("tqsdk not installed, futures tick recording disabled")
        return

    try:
        auth = TqAuth(creds["auth_user"], creds["auth_password"])
        api = TqApi(account=TqSim(init_balance=1_000_000), auth=auth)
    except Exception as e:
        logger.error("failed to create TqApi for tick recording: %s", e)
        return

    logger.info("TqApi created for tick recording, subscribing to %d symbols", len(FUTURES_SYMBOLS))

    tick_serials: dict[str, object] = {}
    tick_positions: dict[str, int] = {}

    for sym in FUTURES_SYMBOLS:
        try:
            ts = api.get_tick_serial(sym, 200)
            tick_serials[sym] = ts
            tick_positions[sym] = len(ts) if ts is not None else 0
        except Exception as e:
            logger.warning("failed to subscribe tick for %s: %s", sym, e)

    logger.info("subscribed to %d tick streams", len(tick_serials))

    last_kline_snapshot = 0.0

    try:
        while not _shutdown.is_set():
            try:
                api.wait_update(deadline=time.time() + 5)
            except Exception:
                if _shutdown.is_set():
                    break
                continue

            for sym, ts in tick_serials.items():
                if ts is None:
                    continue
                try:
                    current_len = len(ts)
                    prev_pos = tick_positions.get(sym, current_len)

                    if current_len > prev_pos:
                        new_ticks = []
                        for i in range(prev_pos, current_len):
                            row = ts.iloc[i]
                            new_ticks.append({
                                "datetime": str(row.get("datetime", "")),
                                "last_price": float(row.get("last_price", 0)),
                                "highest": float(row.get("highest", 0)),
                                "lowest": float(row.get("lowest", 0)),
                                "volume": int(row.get("volume", 0)),
                                "amount": float(row.get("amount", 0)),
                                "open_interest": int(row.get("open_interest", 0)),
                            })
                        if new_ticks:
                            tick_buf.record_batch(sym, new_ticks)

                    tick_positions[sym] = current_len
                except Exception as e:
                    logger.debug("tick read error for %s: %s", sym, e)

            now = time.time()
            if now - last_kline_snapshot >= FUTURES_KLINE_INTERVAL:
                try:
                    n = collect_futures_klines(api)
                    logger.info("futures kline snapshot: %d symbols", n)
                except Exception as e:
                    logger.error("futures kline snapshot error: %s", e)
                last_kline_snapshot = now

    finally:
        try:
            api.close()
            logger.info("TqApi closed (tick thread)")
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════

def run_collection_loop() -> None:
    from health import start_health_server, update_status
    from collector import CRYPTO_SYMBOLS, collect_crypto_klines
    from tick_recorder import TickBuffer

    start_health_server()
    update_status(status="initializing")
    logger.info("health server started on port 18900")

    pp = PrivPortalClient()
    if not pp.wait_for_availability():
        update_status(status="error", error="PolarPrivate not available")
        logger.error("PolarPrivate not available after retries, exiting")
        sys.exit(1)

    update_status(credentials=True)

    creds = fetch_tqsdk_credentials(pp)
    if not creds:
        update_status(status="error", error="Cannot obtain TqSdk credentials via D-class grant")
        logger.error("D-class grant failed, exiting")
        sys.exit(1)

    # ── tick buffers ──
    futures_tick_buf = TickBuffer("futures", flush_interval=60)
    crypto_tick_buf = TickBuffer("crypto", flush_interval=30)
    futures_tick_buf.start()
    crypto_tick_buf.start()

    # ── futures tick thread ──
    futures_thread = None
    if creds:
        futures_thread = Thread(
            target=run_futures_tick_thread,
            args=(creds, futures_tick_buf),
            daemon=True,
            name="futures-tick",
        )
        futures_thread.start()
        logger.info("futures tick thread started")
    else:
        logger.warning("no TqSdk credentials, futures tick recording disabled")

    # ── binance websocket ──
    binance_stream = None
    try:
        from binance_ws import BinanceTradeStream
        binance_stream = BinanceTradeStream(CRYPTO_SYMBOLS, crypto_tick_buf)
        binance_stream.start()
        logger.info("binance trade stream started")
    except ImportError:
        logger.warning("websocket-client not installed, crypto tick recording disabled (install: pip install websocket-client)")
    except Exception as e:
        logger.error("failed to start binance trade stream: %s", e)

    update_status(status="running", collecting=True, tick_recording=True)
    logger.info("=== all collection systems active ===")

    global _session_task_id
    if _sdk_submit:
        try:
            _tr = _sdk_submit(
                task_type="data-collection",
                command="data-collector/main.py",
                requester="tqsdk-data-collector",
            )
            _session_task_id = _tr.get("task_id")
            logger.info("SOTAgent task submitted: %s", _session_task_id)
        except Exception as e:
            logger.debug("Task submit failed: %s", e)

    last_crypto_kline = 0.0

    while not _shutdown.is_set():
        now = time.time()

        if now - last_crypto_kline >= CRYPTO_KLINE_INTERVAL:
            try:
                n = collect_crypto_klines()
                update_status(last_crypto_collection=datetime.now().isoformat(), crypto_pairs=n)
                logger.info("crypto kline snapshot: %d pairs", n)
            except Exception as e:
                logger.error("crypto kline error: %s", e)
            last_crypto_kline = now

        ft_stats = futures_tick_buf.get_stats()
        ct_stats = crypto_tick_buf.get_stats()
        bs_stats = binance_stream.get_stats() if binance_stream else {}

        update_status(
            futures_ticks=ft_stats.get("total_ticks", 0),
            crypto_trades=ct_stats.get("total_ticks", 0),
            binance_ws=bs_stats,
        )

        _shutdown.wait(timeout=30)

    # ── graceful shutdown ──
    logger.info("shutting down...")
    if binance_stream:
        binance_stream.stop()
    futures_tick_buf.stop()
    crypto_tick_buf.stop()

    if _session_task_id and _sdk_complete:
        try:
            _sdk_complete(_session_task_id)
        except Exception:
            pass

    update_status(status="stopped", collecting=False, tick_recording=False)
    logger.info("data collector stopped")


if __name__ == "__main__":
    logger.info("=== TqSdk Data Collector starting (tick mode) ===")
    run_collection_loop()
