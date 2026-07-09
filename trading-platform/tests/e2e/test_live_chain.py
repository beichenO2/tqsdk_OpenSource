"""E2E live-session chain tests — run during market hours.

Usage:
    pytest tests/e2e/test_live_chain.py -m live -v
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import httpx
import pytest

from tests.e2e.conftest import API_BASE, GW_BASE, LIVE_SYMBOL, api_live, gateway_live

_CST = ZoneInfo("Asia/Shanghai")

pytestmark = pytest.mark.live


def _get(client: httpx.Client, url: str) -> httpx.Response:
    return client.get(url, timeout=12.0)


def _post(client: httpx.Client, url: str, json: dict) -> httpx.Response:
    return client.post(url, json=json, timeout=12.0)


# ── L1 Gateway ──────────────────────────────────────────────────────────────


@pytest.mark.skipif(not gateway_live(), reason="gateway offline")
def test_g02_gateway_health_connected() -> None:
    """G-02: gateway connected to TqSdk."""
    r = httpx.get(f"{GW_BASE}/health", timeout=5.0)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["connected"] is True


@pytest.mark.skipif(not gateway_live(), reason="gateway offline")
def test_g03_gateway_quote_not_busy(live_symbol: str) -> None:
    """G-03: quote returns 200 without session-busy 503."""
    r = httpx.get(f"{GW_BASE}/api/v1/market/quote/{live_symbol}", timeout=12.0)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("last_price") is not None
    assert body["last_price"] > 0


@pytest.mark.skipif(not gateway_live(), reason="gateway offline")
def test_g04_gateway_account() -> None:
    """G-04: account info reachable."""
    r = httpx.get(f"{GW_BASE}/api/v1/account", timeout=12.0)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("balance", 0) > 0


# ── L2 API proxy ────────────────────────────────────────────────────────────


@pytest.mark.skipif(not api_live(), reason="API offline")
def test_a01_api_liveness() -> None:
    """A-01: API healthz + readyz."""
    assert httpx.get(f"{API_BASE}/healthz", timeout=5.0).json() == {"status": "ok"}
    ready = httpx.get(f"{API_BASE}/readyz", timeout=5.0)
    assert ready.status_code == 200
    assert ready.json().get("status") == "ready"


@pytest.mark.skipif(not (api_live() and gateway_live()), reason="API or gateway offline")
def test_a02_system_health_all_ok() -> None:
    """A-02: aggregated system health."""
    r = httpx.get(f"{API_BASE}/api/v1/system/health", timeout=10.0)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    comps = body["components"]
    assert comps["api"]["ok"] is True
    assert comps["execution"]["ok"] is True
    assert comps["tqsdk_gateway"]["ok"] is True


@pytest.mark.skipif(not (api_live() and gateway_live()), reason="API or gateway offline")
def test_a03_api_quote_live_not_cache(live_symbol: str) -> None:
    """A-03: API quote uses live gateway, not closed_market_cache."""
    r = httpx.get(f"{API_BASE}/api/v1/market/quote/{live_symbol}", timeout=12.0)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("message") == "ok", f"expected live quote, got cache fallback: {body}"
    last_price = body.get("last_price", 0)
    if isinstance(last_price, str):
        last_price = float(last_price)
    assert last_price > 0


@pytest.mark.skipif(not (api_live() and gateway_live()), reason="API or gateway offline")
def test_a04_account_not_stale() -> None:
    """A-04: account info from live gateway (stale=false)."""
    r = httpx.get(f"{API_BASE}/api/v1/positions/account/info", timeout=12.0)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("stale") is not True, f"account still stale: {body}"
    assert body.get("balance", 0) > 0


@pytest.mark.skipif(not api_live(), reason="API offline")
def test_a05_pnl_history_has_points() -> None:
    """A-05: equity history has snapshots."""
    r = httpx.get(f"{API_BASE}/api/v1/positions/pnl-history?days=7", timeout=10.0)
    assert r.status_code == 200
    rows = r.json()
    assert isinstance(rows, list)
    assert len(rows) >= 1


# ── L3 Risk / Paper ─────────────────────────────────────────────────────────


@pytest.mark.skipif(not api_live(), reason="API offline")
def test_r01_risk_probe_allows_valid_order(live_symbol: str) -> None:
    """R-01: RiskGate passes a reasonable probe order."""
    # Use a mid-range price for rb (~3000)
    payload = {
        "symbol": "rb2510",
        "exchange": "SHFE",
        "direction": "LONG",
        "offset": "OPEN",
        "price": "3100",
        "volume": 1,
    }
    r = httpx.post(f"{API_BASE}/api/v1/live-trading/risk-probe", json=payload, timeout=10.0)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["allowed"] is True


@pytest.mark.skipif(not api_live(), reason="API offline")
def test_r02_risk_probe_rejects_oversized_order() -> None:
    """R-02: RiskGate rejects absurd volume."""
    payload = {
        "symbol": "rb2510",
        "exchange": "SHFE",
        "direction": "LONG",
        "offset": "OPEN",
        "price": "3100",
        "volume": 99999,
    }
    r = httpx.post(f"{API_BASE}/api/v1/live-trading/risk-probe", json=payload, timeout=10.0)
    assert r.status_code == 200
    body = r.json()
    assert body["allowed"] is False


@pytest.mark.skipif(not api_live(), reason="API offline")
def test_r03_paper_order_accepted() -> None:
    """R-03: paper mode order passes gate without exchange submit."""
    payload = {
        "symbol": "rb2510",
        "exchange": "SHFE",
        "direction": "LONG",
        "offset": "OPEN",
        "price": "3100",
        "volume": 1,
        "strategy_id": "e2e-probe",
    }
    r = httpx.post(f"{API_BASE}/api/v1/live-trading/order", json=payload, timeout=10.0)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ACCEPTED_PAPER"
    assert body["mode"] == "paper"


# ── L4 Research ─────────────────────────────────────────────────────────────


@pytest.mark.skipif(not api_live(), reason="API offline")
def test_f01_factors_list_nonempty() -> None:
    """F-01: factor registry populated."""
    r = httpx.get(f"{API_BASE}/api/v1/factors", timeout=15.0)
    assert r.status_code == 200
    body = r.json()
    items = body.get("factors", body) if isinstance(body, dict) else body
    assert isinstance(items, list)
    assert len(items) >= 10


@pytest.mark.skipif(not api_live(), reason="API offline")
def test_f02_factor_analyze_cs_returns_ic() -> None:
    """F-02: cross-sectional IC analysis runs."""
    payload = {
        "factor_name": "wq001",
        "symbols": ["i", "rb", "cu"],
        "horizon": 4,
    }
    r = httpx.post(f"{API_BASE}/api/v1/factors/analyze-cs", json=payload, timeout=60.0)
    assert r.status_code == 200, r.text[:500]
    body = r.json()
    summary = body.get("summary", body)
    assert "ic_mean" in summary or "ic_mean" in body
    assert len(body.get("ic_series", [])) > 0


# ── L5 WebSocket ────────────────────────────────────────────────────────────


@pytest.mark.skipif(not api_live(), reason="API offline")
def test_w01_ws_ping_pong() -> None:
    """W-01: WebSocket accepts connection and responds to ping."""
    import json

    import websockets

    ws_url = API_BASE.replace("http://", "ws://").replace("https://", "wss://") + "/ws"

    async def _run() -> None:
        async with websockets.connect(ws_url, open_timeout=5) as ws:
            await ws.send(json.dumps({"action": "ping"}))
            msg = await asyncio_wait(ws.recv(), timeout=5.0)
            data = json.loads(msg)
            assert data.get("action") == "pong"

    import asyncio

    def asyncio_wait(coro, timeout):
        return asyncio.wait_for(coro, timeout=timeout)

    asyncio.run(_run())
