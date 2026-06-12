"""Lightweight HTTP health check endpoint for SOTAgent process manager."""

from __future__ import annotations

import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

_status: dict = {"status": "starting", "credentials": False, "collecting": False}
_lock = threading.Lock()

HEALTH_PORT = 18900


def update_status(**kwargs: object) -> None:
    with _lock:
        _status.update(kwargs)


def get_status() -> dict:
    with _lock:
        return dict(_status)


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/health":
            body = json.dumps(get_status()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def log_message(self, fmt: str, *args: object) -> None:
        pass  # suppress noisy request logs


def start_health_server() -> HTTPServer:
    """Start the health HTTP server in a daemon thread."""
    server = HTTPServer(("0.0.0.0", HEALTH_PORT), _Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server
