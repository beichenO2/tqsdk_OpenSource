"""Gateway session lock — update_loop must not starve API readers."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from session import SessionBusyError, TqSdkSession


def _mock_quote(**overrides):
    q = MagicMock()
    q.datetime = overrides.get("datetime", 1)
    q.last_price = overrides.get("last_price", 3086.0)
    q.highest = overrides.get("highest", 3090.0)
    q.lowest = overrides.get("lowest", 3080.0)
    q.volume = overrides.get("volume", 100)
    q.amount = overrides.get("amount", 1_000_000.0)
    q.open_interest = overrides.get("open_interest", 5000)
    q.bid_price1 = overrides.get("bid_price1", 3085.0)
    q.bid_volume1 = overrides.get("bid_volume1", 10)
    q.ask_price1 = overrides.get("ask_price1", 3087.0)
    q.ask_volume1 = overrides.get("ask_volume1", 8)
    return q


def test_update_loop_does_not_block_get_quote() -> None:
    """G-01: background wait_update must not hold the API lock."""
    session = TqSdkSession()
    mock_api = MagicMock()
    wait_entered = threading.Event()

    def slow_wait_update(**_kwargs):
        wait_entered.set()
        time.sleep(1.5)

    mock_api.wait_update = slow_wait_update
    mock_api.get_quote.return_value = _mock_quote()

    session._api = mock_api
    session._connected = True
    session._running = True

    loop = threading.Thread(target=session._update_loop, name="test-update-loop", daemon=True)
    loop.start()
    assert wait_entered.wait(timeout=2.0), "update loop never entered wait_update"

    # Must succeed while wait_update is sleeping — not SessionBusyError
    result = session.get_quote("KQ.m@SHFE.rb")
    assert result["last_price"] == 3086.0

    session._running = False
    loop.join(timeout=3.0)


def test_concurrent_quotes_do_not_raise_busy() -> None:
    """G-01b: two sequential quotes succeed under active update loop."""
    session = TqSdkSession()
    mock_api = MagicMock()
    mock_api.wait_update = lambda **_kwargs: time.sleep(0.3)
    mock_api.get_quote.return_value = _mock_quote()

    session._api = mock_api
    session._connected = True
    session._running = True

    loop = threading.Thread(target=session._update_loop, daemon=True)
    loop.start()
    time.sleep(0.05)

    for _ in range(3):
        session.get_quote("KQ.m@SHFE.rb")

    session._running = False
    loop.join(timeout=2.0)


def test_get_quote_raises_busy_when_lock_held_too_long() -> None:
    """Sanity: timed lock still protects against true deadlock."""
    session = TqSdkSession()
    session._api = MagicMock()
    session._connected = True

    started = threading.Event()
    release = threading.Event()

    def hold_lock() -> None:
        session._lock.acquire()
        started.set()
        release.wait(timeout=10.0)
        session._lock.release()

    holder = threading.Thread(target=hold_lock, daemon=True)
    holder.start()
    assert started.wait(timeout=2.0)

    try:
        with pytest.raises(SessionBusyError):
            session.get_quote("KQ.m@SHFE.rb")
    finally:
        release.set()
        holder.join(timeout=3.0)
    session._api.get_quote.assert_not_called()
