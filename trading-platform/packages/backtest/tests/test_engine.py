"""回测引擎集成测试。"""

from datetime import datetime, timedelta
from decimal import Decimal

from backtest.models import BacktestConfig, Bar
from backtest.engine import BacktestEngine
from backtest.datafeed import BarDataFeed
from backtest.strategy import Strategy


class SimpleMAStrategy(Strategy):
    """简单均线策略 - 用于测试。"""

    def __init__(self, window: int = 5) -> None:
        super().__init__()
        self._window = window
        self._closes: list[Decimal] = []
        self._holding = False

    def on_bar(self, bar: Bar) -> None:
        self._closes.append(bar.close)
        if len(self._closes) < self._window:
            return

        ma = sum(self._closes[-self._window:]) / self._window

        if bar.close > ma and not self._holding:
            self.buy(bar.symbol, 1)
            self._holding = True
        elif bar.close < ma and self._holding:
            self.sell(bar.symbol, 1)
            self._holding = False


def _make_bars(symbol: str, n: int, base_price: float = 100.0) -> list[Bar]:
    """生成测试用K线数据。"""
    bars = []
    import math
    base = datetime(2024, 1, 1, 9, 0)
    for i in range(n):
        price = base_price + 10 * math.sin(i * 0.3)
        bars.append(
            Bar(
                symbol=symbol,
                dt=base + timedelta(minutes=i),
                open=Decimal(str(round(price, 2))),
                high=Decimal(str(round(price + 2, 2))),
                low=Decimal(str(round(price - 2, 2))),
                close=Decimal(str(round(price + 0.5, 2))),
                volume=1000,
            )
        )
    return bars


def test_engine_basic_run() -> None:
    """测试引擎基本运行。"""
    config = BacktestConfig(
        strategy_id="test_ma",
        symbols=["SHFE.rb2405"],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 2),
        initial_capital=Decimal("1000000"),
        commission_rate=Decimal("0.0001"),
        slippage_ticks=1,
        tick_size=Decimal("1"),
        contract_multiplier=10,
    )

    engine = BacktestEngine(config)

    datafeed = BarDataFeed(engine.event_bus)
    datafeed.add_bars(_make_bars("SHFE.rb2405", 100))

    engine.set_datafeed(datafeed)
    engine.set_strategy(SimpleMAStrategy(window=5))

    result = engine.run()

    assert result.total_trades >= 0
    assert len(result.equity_curve) == 100
    assert result.final_equity > 0
    assert result.config.strategy_id == "test_ma"


def test_engine_no_trades() -> None:
    """无交易场景。"""

    class DoNothingStrategy(Strategy):
        def on_bar(self, bar: Bar) -> None:
            pass

    config = BacktestConfig(
        strategy_id="noop",
        symbols=["SHFE.rb2405"],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 2),
        initial_capital=Decimal("500000"),
    )
    engine = BacktestEngine(config)
    datafeed = BarDataFeed(engine.event_bus)
    datafeed.add_bars(_make_bars("SHFE.rb2405", 50))
    engine.set_datafeed(datafeed)
    engine.set_strategy(DoNothingStrategy())

    result = engine.run()

    assert result.total_trades == 0
    assert result.final_equity == Decimal("500000")
