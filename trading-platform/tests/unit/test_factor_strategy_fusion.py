"""Factor layer ↔ strategy layer fusion — FeatureMixin, FactorStrategy, evolution registry."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

_repo = Path(__file__).resolve().parents[2]
for p in [_repo, _repo / "packages" / "core", _repo / "packages" / "strategy", _repo / "packages"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from decimal import Decimal  # noqa: E402

from backtest.futures_matrix import backtest_strategy_on_bars  # noqa: E402
from features.registry import FactorRegistry, factor  # noqa: E402
from strategy.base import BaseStrategy, SignalType, StrategyConfig  # noqa: E402
from strategy.feature_mixin import FeatureMixin  # noqa: E402
from strategy.registry import StrategyRegistry  # noqa: E402


def _make_bar(
    close: float,
    *,
    open_: float | None = None,
    high: float | None = None,
    low: float | None = None,
    volume: float = 1000.0,
) -> dict[str, float]:
    o = open_ if open_ is not None else close
    h = high if high is not None else close + 0.5
    l = low if low is not None else close - 0.5
    return {"open": o, "high": h, "low": l, "close": close, "volume": volume}


def _feed_bars(mixin: Any, symbol: str, closes: list[float]) -> None:
    for c in closes:
        mixin.record_bar(symbol, _make_bar(c))


def _make_ohlcv_df(n: int, *, flat: float = 100.0, step: float = 0.0) -> pd.DataFrame:
    closes = [flat + step * i for i in range(n)]
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2024-06-01 09:00", periods=n, freq="5min"),
            "open": closes,
            "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes],
            "close": closes,
            "volume": np.full(n, 5000.0),
            "instrument": ["TEST"] * n,
        }
    )


@pytest.fixture
def _register_test_momentum_factor() -> str:
    """Controllable momentum factor for FactorStrategy logic tests."""

    @factor("test_momentum", category="test", output_columns=["test_momentum"])
    def _test_momentum(df: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
        df["test_momentum"] = df["close"] - df["close"].shift(lag)
        return df

    yield "test_momentum"
    reg = FactorRegistry()
    if "test_momentum" in reg._factors:  # noqa: SLF001
        del reg._factors["test_momentum"]  # noqa: SLF001


class _FeatureProbe(FeatureMixin, BaseStrategy):
    """Minimal strategy shell to exercise FeatureMixin."""

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self._init_features()

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list:
        return []

    async def generate_signals(self, market_data: dict[str, Any]) -> list:
        return []


# --- FeatureMixin ---


def test_feature_mixin_returns_real_factor_value() -> None:
    cfg = StrategyConfig(name="probe", features=["rsi"], params={"feature_window": 200})
    probe = _FeatureProbe(cfg)
    _feed_bars(probe, "SYM", [100.0 + math.sin(i / 3) for i in range(60)])

    vals = probe.factor_values("SYM")
    assert "rsi" in vals
    assert vals["rsi"] is not None
    assert isinstance(vals["rsi"], float)
    assert 0.0 <= vals["rsi"] <= 100.0


def test_feature_mixin_insufficient_data_returns_none_not_raises() -> None:
    cfg = StrategyConfig(name="probe", features=["rsi"], params={"feature_window": 200})
    probe = _FeatureProbe(cfg)
    _feed_bars(probe, "SYM", [100.0, 101.0, 99.0, 100.5, 101.5])

    vals = probe.factor_values("SYM")
    assert vals.get("rsi") is None


# --- FactorStrategy ---


@pytest.mark.asyncio
async def test_factor_strategy_warmup_emits_no_signals(
    _register_test_momentum_factor: str,
) -> None:
    import strategy.templates.factor_strategy  # noqa: F401, PLC0415

    cfg = StrategyConfig(
        name="factor_strategy",
        features=["test_momentum"],
        params={
            "factors": {"test_momentum": 1.0},
            "zscore_window": 30,
            "entry_z": 1.0,
            "exit_z": 0.3,
        },
    )
    strat = StrategyRegistry.create("factor_strategy", cfg)

    signals = []
    for i in range(25):
        bar = _make_bar(100.0 + i * 0.01)
        signals.extend(await strat.on_bar("SYM", bar))

    assert signals == []


@pytest.mark.asyncio
async def test_factor_strategy_entry_and_exit_on_controlled_momentum(
    _register_test_momentum_factor: str,
) -> None:
    import strategy.templates.factor_strategy  # noqa: F401, PLC0415

    cfg = StrategyConfig(
        name="factor_strategy",
        features=["test_momentum"],
        params={
            "factors": {"test_momentum": 1.0},
            "zscore_window": 20,
            "entry_z": 1.2,
            "exit_z": 0.4,
            "allow_short": False,
        },
    )
    strat = StrategyRegistry.create("factor_strategy", cfg)

    closes: list[float] = [100.0] * 50
    closes += [100.0 + 5.0 * (i + 1) for i in range(25)]
    closes += [225.0] * 40

    all_signals = []
    for c in closes:
        all_signals.extend(await strat.on_bar("SYM", _make_bar(c)))

    types = [s.signal_type for s in all_signals]
    assert SignalType.LONG_ENTRY in types, "expected LONG_ENTRY during momentum burst"
    assert SignalType.LONG_EXIT in types, "expected LONG_EXIT when momentum fades"

    entry_idx = types.index(SignalType.LONG_ENTRY)
    exit_idx = next(i for i, t in enumerate(types) if t == SignalType.LONG_EXIT and i > entry_idx)
    assert exit_idx > entry_idx


@pytest.mark.asyncio
async def test_factor_strategy_smoke_with_real_rsi_factor() -> None:
    import strategy.templates.factor_strategy  # noqa: F401, PLC0415

    cfg = StrategyConfig(
        name="factor_strategy",
        features=["rsi"],
        params={
            "factors": {"rsi": 1.0},
            "zscore_window": 30,
            "entry_z": 2.0,
            "exit_z": 0.5,
        },
    )
    strat = StrategyRegistry.create("factor_strategy", cfg)

    closes = [100.0 + math.sin(i / 4) * 2 for i in range(80)]
    signals = []
    for c in closes:
        signals.extend(await strat.on_bar("SYM", _make_bar(c)))

    assert isinstance(signals, list)


def test_registry_create_factor_strategy() -> None:
    import strategy.templates.factor_strategy  # noqa: F401, PLC0415

    cfg = StrategyConfig(
        name="factor_strategy",
        features=["rsi"],
        params={"factors": {"rsi": 1.0}},
    )
    strat = StrategyRegistry.create("factor_strategy", cfg)
    assert strat is not None
    assert strat.name == "factor_strategy"


def test_factor_strategy_backtest_smoke() -> None:
    import strategy.templates.factor_strategy  # noqa: F401, PLC0415

    cfg = StrategyConfig(
        name="factor_strategy",
        strategy_id="factor_bt",
        features=["rsi"],
        params={
            "factors": {"rsi": 1.0},
            "zscore_window": 30,
            "entry_z": 1.5,
            "exit_z": 0.3,
        },
    )
    strat = StrategyRegistry.create("factor_strategy", cfg)
    bars = _make_ohlcv_df(150, flat=3000.0, step=3.0)

    report = backtest_strategy_on_bars(strat, bars, symbol="rb2505", initial_capital=Decimal("100000"))
    assert "error" not in report
    assert report["total_trades"] >= 0


# --- evolution_registry ---


def test_register_evolved_factors_from_fake_json(tmp_path: Path) -> None:
    from factor.evolution_registry import register_evolved_factors  # noqa: PLC0415
    from factor.registry import get_registry  # noqa: PLC0415

    payload = {
        "candidates": [
            {
                "expr": "roc(close, 3)",
                "score": 0.42,
                "ic_mean": 0.15,
                "error": None,
            },
            {
                "expr": "invalid syntax !!!",
                "score": 0.1,
                "ic_mean": 0.05,
                "error": "syntax",
            },
            {
                "expr": "delta(close, 1)",
                "score": None,
                "ic_mean": None,
                "error": None,
            },
        ]
    }
    path = tmp_path / "latest.json"
    path.write_text(json.dumps(payload))

    names = register_evolved_factors(path)
    assert names == ["evolved_0"]

    reg = get_registry()
    df = pd.DataFrame(
        {
            "open": np.linspace(100, 110, 40),
            "high": np.linspace(101, 111, 40),
            "low": np.linspace(99, 109, 40),
            "close": np.linspace(100, 110, 40),
            "volume": np.full(40, 1000.0),
        }
    )
    out = reg.compute("evolved_0", df)
    assert "evolved_0" in out.columns
    assert out["evolved_0"].notna().sum() > 0


def test_register_evolved_factors_missing_file_returns_empty(tmp_path: Path) -> None:
    from factor.evolution_registry import register_evolved_factors  # noqa: PLC0415

    missing = tmp_path / "nope.json"
    assert register_evolved_factors(missing) == []
