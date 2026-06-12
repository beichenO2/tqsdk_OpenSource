"""Tests for BTC backtest analyzer — verifies annualized return uses backtest time.

This test imports the analyzer module directly (bypassing btc/__init__.py which
pulls in strategy-dependent code) to keep the test self-contained.
"""

import sys
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock

# Stub out the strategy package so btc.engine doesn't fail on import
sys.modules.setdefault("strategy", MagicMock())
sys.modules.setdefault("strategy.base", MagicMock())

# Ensure the backtest package root is importable
_pkg_root = Path(__file__).resolve().parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from btc.report.analyzer import BacktestAnalyzer, _std  # noqa: E402
from btc.models.types import BacktestConfig  # noqa: E402


def _make_config() -> BacktestConfig:
    return BacktestConfig(
        strategy_id="test",
        symbols=["BTCUSDT"],
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2024, 1, 1),
    )


def test_annualized_uses_backtest_time():
    """Annualized return should reflect the backtest period, not wall-clock."""
    config = _make_config()

    bt_start = datetime(2023, 1, 1)
    n_bars = 365 * 6  # 4h bars for ~1 year

    initial = Decimal("100000")
    final = Decimal("125000")  # 25% total return over 1 year

    curve: list[tuple[datetime, Decimal]] = []
    for i in range(n_bars + 1):
        t = bt_start + timedelta(hours=4 * i)
        eq = initial + (final - initial) * Decimal(str(i / n_bars))
        curve.append((t, eq))

    wall_start = datetime(2026, 4, 17, 10, 0, 0)
    wall_end = datetime(2026, 4, 17, 10, 0, 5)  # 5 seconds

    analyzer = BacktestAnalyzer(
        config=config,
        equity_curve=curve,
        fills=[],
        started_at=wall_start,
        finished_at=wall_end,
    )

    result = analyzer.compute()
    ann_return = float(result.metrics.annualized_return)

    assert 0.20 < ann_return < 0.30, (
        f"Annualized return {ann_return:.4f} is not in expected range [0.20, 0.30]. "
        "Likely using wall-clock time instead of backtest time."
    )


def test_std_helper():
    assert _std([1.0, 2.0, 3.0]) > 0
    assert _std([5.0]) == 0.0
