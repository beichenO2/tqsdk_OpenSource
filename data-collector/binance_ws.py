"""Binance WebSocket trade stream → TickBuffer.

Connects to Binance combined stream for all configured symbols,
parses trade events, and feeds them into a TickBuffer for
periodic Parquet flush.

Handles reconnection with exponential backoff.
"""

from __future__ import annotations

import json
import logging
import ssl
import threading
import time
from datetime import datetime, timezone

import websocket  # websocket-client

from tick_recorder import TickBuffer

logger = logging.getLogger(__name__)

BINANCE_WS_BASE = "wss://stream.binance.com:9443/stream?streams="
RECONNECT_BASE_SEC = 2
RECONNECT_MAX_SEC = 120


class BinanceTradeStream:
    """WebSocket client that records real-time Binance trades."""

    def __init__(self, symbols: list[str], tick_buffer: TickBuffer) -> None:
        self._symbols = [s.lower() for s in symbols]
        self._buf = tick_buffer
        self._ws: websocket.WebSocketApp | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._reconnect_delay = RECONNECT_BASE_SEC
        self._stats = {"trades_received": 0, "reconnects": 0, "errors": 0}

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="binance-ws")
        self._thread.start()
        logger.info("binance trade stream started: %d symbols", len(self._symbols))

    def stop(self) -> None:
        self._running = False
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("binance trade stream stopped: %s", self._stats)

    def get_stats(self) -> dict:
        return dict(self._stats)

    # ── internal ────────────────────────────────────────────────

    def _build_url(self) -> str:
        streams = "/".join(f"{s}@trade" for s in self._symbols)
        return BINANCE_WS_BASE + streams

    def _run_loop(self) -> None:
        while self._running:
            try:
                self._connect()
            except Exception as e:
                self._stats["errors"] += 1
                logger.error("binance ws error: %s", e)

            if not self._running:
                break

            delay = min(self._reconnect_delay, RECONNECT_MAX_SEC)
            logger.info("reconnecting in %ds...", delay)
            time.sleep(delay)
            self._reconnect_delay = min(self._reconnect_delay * 2, RECONNECT_MAX_SEC)
            self._stats["reconnects"] += 1

    def _detect_proxy(self) -> tuple[str, int, str] | None:
        """Auto-detect Clash Verge mixed-port from its config file."""
        import os, subprocess
        proxy_env = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
        if proxy_env:
            from urllib.parse import urlparse
            p = urlparse(proxy_env)
            return (p.hostname or "127.0.0.1", p.port or 7897, p.scheme or "http")

        cfg_path = os.path.expanduser(
            "~/Library/Application Support/io.github.clash-verge-rev.clash-verge-rev/clash-verge.yaml"
        )
        if os.path.exists(cfg_path):
            try:
                import yaml
                with open(cfg_path) as f:
                    d = yaml.safe_load(f)
                port = d.get("mixed-port", 7897)
                logger.info("detected Clash Verge mixed-port: %d", port)
                return ("127.0.0.1", port, "http")
            except Exception as e:
                logger.warning("failed to parse clash config: %s", e)
        return None

    def _connect(self) -> None:
        url = self._build_url()
        logger.info("connecting to binance: %s", url[:120] + "...")

        proxy_info = self._detect_proxy()
        http_proxy_host = proxy_info[0] if proxy_info else None
        http_proxy_port = proxy_info[1] if proxy_info else None
        if proxy_info:
            logger.info("using proxy: %s:%d", http_proxy_host, http_proxy_port)

        self._ws = websocket.WebSocketApp(
            url,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=self._on_open,
        )

        self._ws.run_forever(
            sslopt={"cert_reqs": ssl.CERT_NONE},
            ping_interval=30,
            ping_timeout=10,
            http_proxy_host=http_proxy_host,
            http_proxy_port=http_proxy_port,
        )

    def _on_open(self, ws: websocket.WebSocket) -> None:
        self._reconnect_delay = RECONNECT_BASE_SEC
        logger.info("binance ws connected")

    def _on_message(self, ws: websocket.WebSocket, message: str) -> None:
        try:
            msg = json.loads(message)
            data = msg.get("data", msg)
            if data.get("e") != "trade":
                return

            symbol = data["s"].lower()
            row = {
                "timestamp": datetime.fromtimestamp(data["T"] / 1000, tz=timezone.utc).isoformat(),
                "trade_id": data["t"],
                "price": float(data["p"]),
                "quantity": float(data["q"]),
                "is_buyer_maker": data["m"],
            }
            self._buf.record(symbol, row)
            self._stats["trades_received"] += 1

        except Exception as e:
            self._stats["errors"] += 1
            if self._stats["errors"] % 1000 == 1:
                logger.warning("binance msg parse error: %s", e)

    def _on_error(self, ws: websocket.WebSocket, error: Exception) -> None:
        logger.warning("binance ws error: %s", error)

    def _on_close(self, ws: websocket.WebSocket, close_code: int | None, close_msg: str | None) -> None:
        logger.info("binance ws closed: code=%s msg=%s", close_code, close_msg)
