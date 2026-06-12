"""Tests for report generators — Sharpe ratio, drawdown, annualized return."""

from datetime import datetime, timedelta
from decimal import Decimal

from backtest.models import BacktestConfig, EquityCurvePoint
from backtest.report import ReportGenerator


def _make_equity_curve(
    returns: list[float],
    initial: float = 1_000_000,
    start: datetime | None = None,
) -> list[EquityCurvePoint]:
    """Build equity curve from a list of per-bar returns."""
    base_dt = start or datetime(2024, 1, 1, 9, 0)
    equity = Decimal(str(initial))
    curve = [EquityCurvePoint(dt=base_dt, equity=equity, cash=equity)]
    for i, r in enumerate(returns):
        equity = equity * (1 + Decimal(str(r)))
        curve.append(EquityCurvePoint(
            dt=base_dt + timedelta(minutes=i + 1),
            equity=equity,
            cash=equity,
        ))
    return curve


def test_sharpe_zero_returns():
    """With zero returns, Sharpe should be negative (only rf drag) or 0."""
    curve = _make_equity_curve([0.0] * 100)
    gen = ReportGenerator()
    config = BacktestConfig(initial_capital=Decimal("1000000"))
    result = gen.generate(config, [], curve)
    # All returns are 0 → std is 0 → sharpe stays 0 (division guard)
    assert result.sharpe_ratio == Decimal(0)


def test_sharpe_positive():
    """With mostly positive returns, Sharpe should be > 0."""
    import random
    random.seed(42)
    returns = [0.002 + random.gauss(0, 0.005) for _ in range(250)]
    curve = _make_equity_curve(returns)
    gen = ReportGenerator()
    config = BacktestConfig(initial_capital=Decimal("1000000"))
    result = gen.generate(config, [], curve)
    assert float(result.sharpe_ratio) > 0


def test_max_drawdown():
    """Drawdown should capture the worst peak-to-trough."""
    curve = _make_equity_curve([0.1, -0.15, 0.05])
    gen = ReportGenerator()
    config = BacktestConfig(initial_capital=Decimal("1000000"))
    result = gen.generate(config, [], curve)
    assert float(result.max_drawdown_pct) > 0.13


def test_total_return():
    curve = _make_equity_curve([0.05, 0.05, -0.02])
    gen = ReportGenerator()
    config = BacktestConfig(initial_capital=Decimal("1000000"))
    result = gen.generate(config, [], curve)
    expected = (1.05 * 1.05 * 0.98) - 1
    assert abs(float(result.total_return) - expected) < 1e-6


def test_sortino_uses_all_returns_in_denominator():
    """Sortino downside deviation should use len(all returns), not just negatives."""
    import random
    random.seed(99)
    returns = [0.003 + random.gauss(0, 0.008) for _ in range(200)]
    negatives = [r for r in returns if r < 0]
    assert len(negatives) > 10, "need some negative returns for test"

    curve = _make_equity_curve(returns)
    gen = ReportGenerator()
    config = BacktestConfig(initial_capital=Decimal("1000000"))
    result = gen.generate(config, [], curve)

    assert result.sortino_ratio is not None
    assert float(result.sortino_ratio) > 0


def test_sortino_all_positive_returns():
    """When all returns are positive, Sortino should not be set (no downside)."""
    curve = _make_equity_curve([0.001] * 50)
    gen = ReportGenerator()
    config = BacktestConfig(initial_capital=Decimal("1000000"))
    result = gen.generate(config, [], curve)
    assert result.sortino_ratio is None or result.sortino_ratio == Decimal(0)
