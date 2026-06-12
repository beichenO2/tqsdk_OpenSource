"""Crypto data aggregation router — BlockBeats, CMC, CoinAnk, Dune."""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Query

from core.exceptions import ProvidersUnavailableError

router = APIRouter(prefix="/crypto-data", tags=["crypto-data"])
logger = logging.getLogger(__name__)

_IMPORT_ERROR: str | None = None

try:
    from datahub.providers.blockbeats import BlockBeatsProvider
    from datahub.providers.coinmarketcap import CoinMarketCapProvider
    from datahub.providers.coinank import CoinAnkProvider
    from datahub.providers.dune import DuneProvider
except ImportError as exc:
    _IMPORT_ERROR = str(exc)
    BlockBeatsProvider = None  # type: ignore[assignment, misc]
    CoinMarketCapProvider = None  # type: ignore[assignment, misc]
    CoinAnkProvider = None  # type: ignore[assignment, misc]
    DuneProvider = None  # type: ignore[assignment, misc]


def _check_available() -> None:
    if _IMPORT_ERROR:
        raise ProvidersUnavailableError(
            "Crypto data providers unavailable",
            detail={"import_error": _IMPORT_ERROR},
        )


@router.get("/news")
async def get_crypto_news(
    category: str = "all",
    limit: int = Query(default=20, le=100),
) -> list[dict[str, Any]]:
    _check_available()
    assert BlockBeatsProvider is not None
    async with BlockBeatsProvider() as bb:
        return await bb.get_news(category, limit)


@router.get("/fund-flows/{symbol}")
async def get_fund_flows(symbol: str = "BTC") -> dict[str, Any]:
    _check_available()
    assert BlockBeatsProvider is not None
    async with BlockBeatsProvider() as bb:
        return await bb.get_fund_flows(symbol)


@router.get("/macro")
async def get_macro_data() -> dict[str, Any]:
    _check_available()
    assert BlockBeatsProvider is not None
    async with BlockBeatsProvider() as bb:
        return await bb.get_macro_data()


@router.get("/quotes")
async def get_quotes(
    symbols: Optional[str] = None,
    limit: int = Query(default=100, le=5000),
) -> list[dict[str, Any]]:
    _check_available()
    assert CoinMarketCapProvider is not None
    sym_list = symbols.split(",") if symbols else None
    async with CoinMarketCapProvider() as cmc:
        return await cmc.get_latest_quotes(sym_list, limit)


@router.get("/global-metrics")
async def get_global_metrics() -> dict[str, Any]:
    _check_available()
    assert CoinMarketCapProvider is not None
    async with CoinMarketCapProvider() as cmc:
        return await cmc.get_global_metrics()


@router.get("/open-interest/{symbol}")
async def get_open_interest(
    symbol: str = "BTC",
    exchange: Optional[str] = None,
) -> dict[str, Any]:
    _check_available()
    assert CoinAnkProvider is not None
    async with CoinAnkProvider() as ca:
        return await ca.get_open_interest(symbol, exchange)


@router.get("/funding-rates/{symbol}")
async def get_funding_rates(
    symbol: str = "BTC",
    exchange: Optional[str] = None,
) -> list[dict[str, Any]]:
    _check_available()
    assert CoinAnkProvider is not None
    async with CoinAnkProvider() as ca:
        return await ca.get_funding_rates(symbol, exchange)


@router.get("/liquidations/{symbol}")
async def get_liquidations(
    symbol: str = "BTC",
    period: str = "24h",
) -> dict[str, Any]:
    _check_available()
    assert CoinAnkProvider is not None
    async with CoinAnkProvider() as ca:
        return await ca.get_liquidations(symbol, period)


@router.get("/long-short-ratio/{symbol}")
async def get_long_short_ratio(
    symbol: str = "BTC",
    exchange: str = "binance",
) -> dict[str, Any]:
    _check_available()
    assert CoinAnkProvider is not None
    async with CoinAnkProvider() as ca:
        return await ca.get_long_short_ratio(symbol, exchange)


@router.get("/onchain/{query_id}")
async def get_onchain_data(
    query_id: int,
    max_wait: int = Query(default=60, le=120),
) -> dict[str, Any]:
    _check_available()
    assert DuneProvider is not None
    async with DuneProvider() as dune:
        return await dune.run_query_and_wait(query_id, max_wait=max_wait)


@router.get("/onchain/{query_id}/cached")
async def get_onchain_cached(query_id: int) -> dict[str, Any]:
    _check_available()
    assert DuneProvider is not None
    async with DuneProvider() as dune:
        return await dune.get_latest_result(query_id)
