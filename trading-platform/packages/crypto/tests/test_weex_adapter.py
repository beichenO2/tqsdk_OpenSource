"""WEEX 交易所适配器单元测试。"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from broker_crypto import WEEXAdapter, create_adapter
from broker_crypto.models import (
    Balance,
    Exchange,
    ExchangeCredentials,
    OHLCV,
    OrderBook,
    OrderRequest,
    OrderResponse,
    OrderStatus,
    OrderType,
    Side,
    Ticker,
    TimeInForce,
    Trade,
)


@pytest.fixture
def creds() -> ExchangeCredentials:
    return ExchangeCredentials(
        exchange=Exchange.WEEX,
        api_key="test_key",
        api_secret="test_secret",
        passphrase="test_pass",
        testnet=False,
    )


@pytest.fixture
def adapter(creds: ExchangeCredentials) -> WEEXAdapter:
    return WEEXAdapter(creds)


class TestWEEXAdapterCreation:
    def test_exchange_property(self, adapter: WEEXAdapter):
        assert adapter.exchange == Exchange.WEEX

    def test_factory_creates_weex(self, creds: ExchangeCredentials):
        a = create_adapter(creds)
        assert isinstance(a, WEEXAdapter)

    def test_exchange_enum_has_weex(self):
        assert Exchange.WEEX.value == "WEEX"


class TestWEEXSignature:
    def test_sign_get_without_query(self, adapter: WEEXAdapter):
        sig = adapter._sign("1591089508404", "GET", "/api/v3/market/depth")
        assert isinstance(sig, str)
        assert len(sig) > 0

    def test_sign_get_with_query(self, adapter: WEEXAdapter):
        sig = adapter._sign(
            "1591089508404", "GET", "/api/v3/market/depth",
            query_string="symbol=BTCUSDT&limit=20",
        )
        assert isinstance(sig, str)

    def test_sign_post_with_body(self, adapter: WEEXAdapter):
        body = '{"symbol":"BTCUSDT","side":"BUY"}'
        sig = adapter._sign("1591089508404", "POST", "/api/v3/order", body=body)
        assert isinstance(sig, str)

    def test_headers_contain_required_fields(self, adapter: WEEXAdapter):
        headers = adapter._headers("GET", "/api/v3/account/")
        assert "ACCESS-KEY" in headers
        assert "ACCESS-SIGN" in headers
        assert "ACCESS-TIMESTAMP" in headers
        assert "ACCESS-PASSPHRASE" in headers
        assert headers["ACCESS-KEY"] == "test_key"
        assert headers["ACCESS-PASSPHRASE"] == "test_pass"


class TestWEEXParseOrder:
    def test_parse_filled_order(self, adapter: WEEXAdapter):
        raw = {
            "symbol": "BTCUSDT",
            "orderId": 702345678901234567,
            "clientOrderId": "my-order",
            "price": "68900",
            "origQty": "0.01",
            "executedQty": "0.01",
            "cummulativeQuoteQty": "689.00",
            "status": "FILLED",
            "type": "LIMIT",
            "side": "BUY",
            "time": 1764506000456,
            "updateTime": 1764506001556,
        }
        result = adapter._parse_order(raw)
        assert result.exchange == Exchange.WEEX
        assert result.status == OrderStatus.FILLED
        assert result.filled_quantity == Decimal("0.01")
        assert result.avg_fill_price == Decimal("689.00") / Decimal("0.01")
        assert result.side == Side.BUY

    def test_parse_new_order(self, adapter: WEEXAdapter):
        raw = {
            "symbol": "ETHUSDT",
            "orderId": 123,
            "price": "3500",
            "origQty": "1",
            "executedQty": "0",
            "cummulativeQuoteQty": "0",
            "status": "NEW",
            "type": "MARKET",
            "side": "SELL",
            "time": 1764506000000,
            "updateTime": 1764506000000,
        }
        result = adapter._parse_order(raw)
        assert result.status == OrderStatus.OPEN
        assert result.side == Side.SELL
        assert result.order_type == OrderType.MARKET
        assert result.avg_fill_price is None


class TestWEEXTimestamp:
    def test_ts_converts_ms(self, adapter: WEEXAdapter):
        dt = adapter._ts(1764506000456)
        assert isinstance(dt, datetime)
        assert dt.tzinfo == timezone.utc


class TestWEEXModelsWithWeexExchange:
    def test_ticker_with_weex(self):
        t = Ticker(
            exchange=Exchange.WEEX,
            symbol="BTCUSDT",
            bid=Decimal("68919.90"),
            ask=Decimal("68920.70"),
            last=Decimal("68920.40"),
            volume_24h=Decimal("1524.361"),
            timestamp=datetime.now(tz=timezone.utc),
        )
        assert t.exchange == Exchange.WEEX

    def test_balance_with_weex(self):
        b = Balance(
            exchange=Exchange.WEEX,
            asset="USDT",
            free=Decimal("1200"),
            locked=Decimal("0"),
        )
        assert b.total == Decimal("1200")

    def test_orderbook_with_weex(self):
        book = OrderBook(
            exchange=Exchange.WEEX,
            symbol="BTCUSDT",
            bids=[(Decimal("68950.10"), Decimal("2.345"))],
            asks=[(Decimal("68950.20"), Decimal("1.104"))],
            timestamp=datetime.now(tz=timezone.utc),
        )
        assert book.asks[0][0] > book.bids[0][0]

    def test_ohlcv_with_weex(self):
        candle = OHLCV(
            exchange=Exchange.WEEX,
            symbol="BTCUSDT",
            interval="1m",
            open=Decimal("68940.10"),
            high=Decimal("68955.00"),
            low=Decimal("68938.50"),
            close=Decimal("68952.40"),
            volume=Decimal("12.345"),
            timestamp=datetime.now(tz=timezone.utc),
        )
        assert candle.high > candle.low
