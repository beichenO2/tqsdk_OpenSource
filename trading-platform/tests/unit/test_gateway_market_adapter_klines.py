"""Regression: gateway klines rows with null/NaN OHLC must be skipped, not 500.

Right after a gateway restart TqSdk may return the newest kline rows with
NaN/None OHLC before real data arrives; Decimal(str(nan_or_none)) raised
decimal.InvalidOperation and broke GET /market/klines end-to-end (2026-07-10).
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from broker_tqsdk.gateway_market_adapter import TqGatewayMarketAdapter


def _make_adapter(items: list[dict]) -> TqGatewayMarketAdapter:
    adapter = TqGatewayMarketAdapter.__new__(TqGatewayMarketAdapter)
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"items": items}
    client = MagicMock()
    client.get = AsyncMock(return_value=resp)
    adapter._client = client
    return adapter


@pytest.mark.asyncio
async def test_klines_skip_nan_and_null_rows() -> None:
    good = {
        "datetime": 1_783_651_500_000_000_000,
        "open": 3088.0,
        "high": 3088.0,
        "low": 3087.0,
        "close": 3087.0,
        "volume": 1701,
        "open_interest": 2_031_284,
    }
    nan_row = dict(good, open=float("nan"))
    null_row = dict(good, close=None)
    no_dt_row = dict(good, datetime=None)

    adapter = _make_adapter([nan_row, null_row, no_dt_row, good])
    bars = await adapter.get_klines("KQ.m@SHFE.rb", duration_seconds=300)

    assert len(bars) == 1
    assert bars[0].open == Decimal("3088.0")
    assert bars[0].volume == 1701
