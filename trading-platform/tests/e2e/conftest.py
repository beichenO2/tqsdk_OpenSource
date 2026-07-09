"""E2E fixtures — skip when live services are not reachable."""

from __future__ import annotations

import os

import httpx
import pytest

API_BASE = os.getenv("LIVE_API_URL", "http://127.0.0.1:8600").rstrip("/")
GW_BASE = os.getenv("LIVE_GATEWAY_URL", "http://127.0.0.1:12890").rstrip("/")
LIVE_SYMBOL = os.getenv("LIVE_TEST_SYMBOL", "KQ.m@SHFE.rb")


def _probe(url: str, path: str, timeout: float = 3.0) -> bool:
    try:
        r = httpx.get(f"{url}{path}", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def gateway_live() -> bool:
    try:
        r = httpx.get(f"{GW_BASE}/health", timeout=3.0)
        return r.status_code == 200 and r.json().get("connected") is True
    except Exception:
        return False


def api_live() -> bool:
    return _probe(API_BASE, "/healthz")


@pytest.fixture(scope="session")
def live_gateway_url() -> str:
    if not gateway_live():
        pytest.skip(f"Gateway not reachable at {GW_BASE}")
    return GW_BASE


@pytest.fixture(scope="session")
def live_api_url() -> str:
    if not api_live():
        pytest.skip(f"API not reachable at {API_BASE}")
    return API_BASE


@pytest.fixture(scope="session")
def live_symbol() -> str:
    return LIVE_SYMBOL
