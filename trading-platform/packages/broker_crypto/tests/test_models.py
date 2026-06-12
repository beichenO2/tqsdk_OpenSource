"""broker_crypto 模型单元测试"""

from datetime import datetime, timezone
from decimal import Decimal

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
    Position,
    Side,
    Ticker,
    TimeInForce,
    Trade,
)


def test_ticker():
    t = Ticker(
        exchange=Exchange.BINANCE,
        symbol="BTCUSDT",
        bid=Decimal("64999"),
        ask=Decimal("65001"),
        last=Decimal("65000"),
        volume_24h=Decimal("12345.6"),
        timestamp=datetime.now(tz=timezone.utc),
    )
    assert t.last == Decimal("65000")
    assert t.exchange == Exchange.BINANCE


def test_ohlcv():
    candle = OHLCV(
        exchange=Exchange.OKX,
        symbol="BTCUSDT",
        interval="1m",
        open=Decimal("64000"),
        high=Decimal("65000"),
        low=Decimal("63500"),
        close=Decimal("64800"),
        volume=Decimal("100.5"),
        timestamp=datetime.now(tz=timezone.utc),
    )
    assert candle.high > candle.low
    assert candle.interval == "1m"


def test_order_request():
    req = OrderRequest(
        exchange=Exchange.BINANCE,
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("0.01"),
        price=Decimal("65000"),
    )
    assert req.time_in_force == TimeInForce.GTC
    assert req.stop_price is None


def test_order_response_defaults():
    now = datetime.now(tz=timezone.utc)
    resp = OrderResponse(
        exchange=Exchange.BINANCE,
        order_id="test-001",
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type=OrderType.LIMIT,
        status=OrderStatus.OPEN,
        quantity=Decimal("0.01"),
        price=Decimal("65000"),
        created_at=now,
        updated_at=now,
    )
    assert resp.filled_quantity == Decimal("0")
    assert resp.avg_fill_price is None


def test_balance_total_auto():
    bal = Balance(
        exchange=Exchange.BINANCE,
        asset="USDT",
        free=Decimal("1000"),
        locked=Decimal("200"),
    )
    assert bal.total == Decimal("1200")


def test_position():
    pos = Position(
        exchange=Exchange.OKX,
        symbol="BTC-USDT-SWAP",
        side=Side.BUY,
        quantity=Decimal("0.5"),
        entry_price=Decimal("64000"),
        unrealized_pnl=Decimal("500"),
        leverage=10,
        timestamp=datetime.now(tz=timezone.utc),
    )
    assert pos.leverage == 10
    assert pos.unrealized_pnl == Decimal("500")


def test_trade():
    trade = Trade(
        exchange=Exchange.BINANCE,
        symbol="BTCUSDT",
        trade_id="t-001",
        side=Side.BUY,
        price=Decimal("65000"),
        quantity=Decimal("0.01"),
        timestamp=datetime.now(tz=timezone.utc),
    )
    assert trade.trade_id == "t-001"


def test_credentials():
    cred = ExchangeCredentials(
        exchange=Exchange.BINANCE,
        api_key="test_key",
        api_secret="test_secret",
        testnet=True,
    )
    assert cred.testnet is True
    assert cred.passphrase is None


def test_orderbook():
    book = OrderBook(
        exchange=Exchange.BINANCE,
        symbol="BTCUSDT",
        bids=[(Decimal("64999"), Decimal("1.5"))],
        asks=[(Decimal("65001"), Decimal("0.8"))],
        timestamp=datetime.now(tz=timezone.utc),
    )
    assert len(book.bids) == 1
    assert book.asks[0][0] > book.bids[0][0]
