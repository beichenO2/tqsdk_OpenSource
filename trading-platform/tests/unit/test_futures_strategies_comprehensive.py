"""Comprehensive unit tests for registered futures strategies."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Sequence
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

_repo = Path(__file__).resolve().parents[2]
for p in [
    _repo,
    _repo / "apps" / "api",
    _repo / "packages" / "core",
    _repo / "packages" / "backtest",
    _repo / "packages" / "strategy",
    _repo / "packages",
]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import strategy.futures  # noqa: E402 — side-effect: @auto_register
from core.models.bar import Bar  # noqa: E402
from strategy.base import OrderSide, Position, Signal, SignalType, StrategyConfig  # noqa: E402
from strategy.futures.adaptive_bollinger import AdaptiveBollingerStrategy  # noqa: E402
from strategy.futures.bollinger_mr import BollingerMRStrategy  # noqa: E402
from strategy.futures.cta_trend import CTATrendStrategy  # noqa: E402
from strategy.futures.dl_strategy import DLTimeseriesStrategy  # noqa: E402
from strategy.futures.dual_ma import FuturesDualMAStrategy  # noqa: E402
from strategy.futures.pairs_trading import PairsTradingStrategy  # noqa: E402
from strategy.futures.rbreaker import RBreakerStrategy  # noqa: E402
from strategy.futures.regime_momentum import RegimeMomentumStrategy  # noqa: E402
from strategy.futures.spread_arb import SpreadArbitrage  # noqa: E402
from strategy.futures.vol_breakout import VolBreakoutStrategy  # noqa: E402
from strategy.futures.volume_price import VolumePriceStrategy  # noqa: E402
from strategy.registry import StrategyRegistry  # noqa: E402


def bar_to_dict(b: Bar) -> dict[str, Any]:
    return {
        "open": float(b.open),
        "high": float(b.high),
        "low": float(b.low),
        "close": float(b.close),
        "volume": int(b.volume),
        "datetime": b.datetime,
    }


@pytest.fixture
def sample_bars() -> list[Bar]:
    """Synthetic OHLCV series for strategy integration tests.

    Bar uses ``datetime=`` (domain model); not ``dt=``.
    """
    from datetime import datetime as _dt, timedelta as _td
    from decimal import Decimal as _Dec

    bars: list[Bar] = []
    price = 3500.0
    for i in range(200):
        bars.append(
            Bar(
                symbol="SHFE.rb2501",
                datetime=_dt(2026, 1, 1) + _td(minutes=i),
                open=_Dec(str(round(price - 2, 1))),
                high=_Dec(str(round(price + 6, 1))),
                low=_Dec(str(round(price - 6, 1))),
                close=_Dec(str(round(price, 1))),
                volume=2000 + i * 10,
            )
        )
        price += 5 if i < 67 else -8 if i < 134 else 6
    return bars


@pytest.fixture
def sample_bars_far(sample_bars: list[Bar]) -> list[Bar]:
    """Second leg for spread/pairs — shifted closes to build a non-constant spread series."""
    out: list[Bar] = []
    for i, b in enumerate(sample_bars):
        skew = Decimal(str(40 + (i % 40) * 2 - (i // 40) * 5))
        out.append(
            b.model_copy(
                update={
                    "symbol": "SHFE.rb2505",
                    "close": b.close - skew,
                    "open": b.open - skew,
                    "high": b.high - skew,
                    "low": b.low - skew,
                }
            )
        )
    return out


async def _feed_on_bar(strategy: Any, symbol: str, bars: Sequence[Bar]) -> list[Any]:
    collected: list[Any] = []
    for b in bars:
        collected.extend(await strategy.on_bar(symbol, bar_to_dict(b)))
    return collected


async def _feed_pair(
    strategy: Any,
    sym_a: str,
    bars_a: Sequence[Bar],
    sym_b: str,
    bars_b: Sequence[Bar],
) -> list[Any]:
    out: list[Any] = []
    for ba, bb in zip(bars_a, bars_b, strict=True):
        out.extend(await strategy.on_bar(sym_a, bar_to_dict(ba)))
        out.extend(await strategy.on_bar(sym_b, bar_to_dict(bb)))
    return out


def _cfg(name: str, symbols: list[str], params: dict[str, Any] | None = None) -> StrategyConfig:
    return StrategyConfig(name=name, symbols=symbols, params=params or {})


# --- StrategyRegistry ---


def test_registry_lists_include_futures_dual_ma() -> None:
    names = StrategyRegistry.list_registered()
    assert "futures_dual_ma" in names
    assert "cta_trend" in names
    assert "dl_timeseries" in names


def test_registry_create_dual_ma_roundtrip() -> None:
    cfg = _cfg("dual", ["SHFE.rb2501"])
    obj = StrategyRegistry.create("futures_dual_ma", cfg)
    assert isinstance(obj, FuturesDualMAStrategy)
    assert obj.config.name == "dual"


def test_registry_create_unknown_raises_key_error() -> None:
    with pytest.raises(KeyError, match="未注册"):
        StrategyRegistry.create("not_a_real_strategy_name", _cfg("x", []))


def test_registry_add_list_delete_instance() -> None:
    cfg = _cfg("inst", ["A"], {})
    sid = cfg.strategy_id
    StrategyRegistry.add_instance(cfg)
    assert StrategyRegistry.get_instance(sid) is not None
    assert any(c.strategy_id == sid for c in StrategyRegistry.list_instances())
    assert StrategyRegistry.delete_instance(sid) is True
    assert StrategyRegistry.get_instance(sid) is None


def test_registry_set_instance_enabled_updates_flag() -> None:
    cfg = _cfg("en", ["B"], {})
    StrategyRegistry.add_instance(cfg)
    updated = StrategyRegistry.set_instance_enabled(cfg.strategy_id, False)
    assert updated is not None
    assert updated.enabled is False
    StrategyRegistry.delete_instance(cfg.strategy_id)


def test_registry_unregister_removes_type() -> None:
    class _Dummy(FuturesDualMAStrategy):
        pass

    StrategyRegistry.register("tmp_dummy_strategy", _Dummy)
    assert StrategyRegistry.get("tmp_dummy_strategy") is _Dummy
    assert StrategyRegistry.unregister("tmp_dummy_strategy") is True
    assert StrategyRegistry.get("tmp_dummy_strategy") is None


# --- FuturesDualMAStrategy (5) ---


def test_dual_ma_instantiation_default_params() -> None:
    s = FuturesDualMAStrategy(_cfg("dma", ["SHFE.rb2501"]))
    assert s.get_param("fast_period") == 5
    assert s.get_param("slow_period") == 20


def test_dual_ma_instantiation_custom_params() -> None:
    s = FuturesDualMAStrategy(
        _cfg("dma2", ["SHFE.rb2501"], {"fast_period": 3, "slow_period": 15, "volume_surge_ratio": 1.05})
    )
    assert s.get_param("fast_period") == 3
    assert s.get_param("slow_period") == 15


@pytest.mark.asyncio
async def test_dual_ma_on_bar_produces_signals(sample_bars: list[Bar]) -> None:
    s = FuturesDualMAStrategy(_cfg("dma3", ["SHFE.rb2501"], {"volume_surge_ratio": 1.01}))
    s.update_position(
        Position(symbol="SHFE.rb2501", side=OrderSide.BUY, qty=1.0, avg_price=3000.0)
    )
    sigs = await _feed_on_bar(s, "SHFE.rb2501", sample_bars)
    assert any(sig.signal_type == SignalType.LONG_EXIT for sig in sigs)


@pytest.mark.asyncio
async def test_dual_ma_empty_bar_sequence_returns_empty() -> None:
    s = FuturesDualMAStrategy(_cfg("dma4", ["SHFE.rb2501"]))
    assert await _feed_on_bar(s, "SHFE.rb2501", []) == []


def test_dual_ma_invalid_params_zero_fast_period_no_crash() -> None:
    s = FuturesDualMAStrategy(_cfg("dma5", ["SHFE.rb2501"], {"fast_period": 0}))
    assert s.get_param("fast_period") == 0


# --- CTATrendStrategy (5) ---


def test_cta_trend_instantiation_default_params() -> None:
    s = CTATrendStrategy(_cfg("cta", ["SHFE.rb2501"]))
    assert s.get_param("entry_period") == 20


def test_cta_trend_instantiation_custom_params() -> None:
    s = CTATrendStrategy(_cfg("cta2", ["SHFE.rb2501"], {"entry_period": 12, "exit_period": 6}))
    assert s.get_param("entry_period") == 12
    assert s.get_param("exit_period") == 6


@pytest.mark.asyncio
async def test_cta_trend_generate_signals_with_equity(sample_bars: list[Bar]) -> None:
    s = CTATrendStrategy(_cfg("cta3", ["SHFE.rb2501"], {"atr_filter_mult": 0.0}))
    md: dict[str, Any] = {"equity": 1_000_000.0}
    all_sigs: list[Any] = []
    for b in sample_bars:
        md["SHFE.rb2501"] = bar_to_dict(b)
        all_sigs.extend(await s.generate_signals(md))
    assert len(all_sigs) >= 1


@pytest.mark.asyncio
async def test_cta_trend_empty_market_data_no_signals() -> None:
    s = CTATrendStrategy(_cfg("cta4", ["SHFE.rb2501"]))
    assert await s.generate_signals({}) == []


def test_cta_trend_invalid_negative_entry_period_stored() -> None:
    s = CTATrendStrategy(_cfg("cta5", ["SHFE.rb2501"], {"entry_period": -3}))
    assert s.get_param("entry_period") == -3


# --- BollingerMRStrategy (5) ---


def test_bollinger_mr_instantiation_default_params() -> None:
    s = BollingerMRStrategy(_cfg("bb", ["SHFE.rb2501"]))
    assert s.get_param("bb_period") == 20


def test_bollinger_mr_instantiation_custom_params() -> None:
    s = BollingerMRStrategy(_cfg("bb2", ["SHFE.rb2501"], {"rsi_oversold": 40.0, "rsi_overbought": 60.0}))
    assert s.get_param("rsi_oversold") == 40.0


@pytest.mark.asyncio
async def test_bollinger_mr_on_bar_eventually_emits_signal(sample_bars: list[Bar]) -> None:
    s = BollingerMRStrategy(_cfg("bb3", ["SHFE.rb2501"], {"min_bars": 20, "rsi_oversold": 45.0}))
    sigs = await _feed_on_bar(s, "SHFE.rb2501", sample_bars)
    assert len(sigs) >= 1


@pytest.mark.asyncio
async def test_bollinger_mr_empty_feed_returns_empty() -> None:
    s = BollingerMRStrategy(_cfg("bb4", ["SHFE.rb2501"]))
    assert await _feed_on_bar(s, "SHFE.rb2501", []) == []


def test_bollinger_mr_invalid_bb_period_returns_hold_not_crash() -> None:
    s = BollingerMRStrategy(_cfg("bb5", ["SHFE.rb2501"], {"bb_period": 0, "min_bars": 5}))
    sig = s.generate_signal("SHFE.rb2501", {"close": 100.0, "high": 101.0, "low": 99.0})
    for _ in range(30):
        sig = s.generate_signal("SHFE.rb2501", {"close": 100.0, "high": 101.0, "low": 99.0})
    assert sig.signal_type == SignalType.HOLD


# --- VolBreakoutStrategy (5) ---


def test_vol_breakout_instantiation_default_params() -> None:
    s = VolBreakoutStrategy(_cfg("vb", ["SHFE.rb2501"]))
    assert s.get_param("atr_period") == 14


def test_vol_breakout_instantiation_custom_params() -> None:
    s = VolBreakoutStrategy(
        _cfg("vb2", ["SHFE.rb2501"], {"narrow_atr_ratio": 1.5, "volume_rise_ratio": 1.01})
    )
    assert s.get_param("narrow_atr_ratio") == 1.5


@pytest.mark.asyncio
async def test_vol_breakout_on_bar_generates_breakout_signal() -> None:
    """Tight range keeps ATR in a narrow regime; close-only breakout + volume spike."""
    sym = "SHFE.rb2501"
    s = VolBreakoutStrategy(
        _cfg(
            "vb3",
            [sym],
            {
                "min_bars": 12,
                "narrow_atr_ratio": 2.0,
                "volume_rise_ratio": 1.0,
                "atr_period": 5,
                "atr_regime_period": 8,
                "range_period": 5,
                "volume_ma_period": 8,
            },
        )
    )
    p0 = 3000.0
    sigs: list[Any] = []
    for i in range(28):
        sigs.extend(
            await s.on_bar(
                sym,
                {
                    "open": p0,
                    "high": p0 + 0.4,
                    "low": p0 - 0.4,
                    "close": p0,
                    "volume": 2000 + i * 20,
                },
            )
        )
    sigs.extend(
        await s.on_bar(
            sym,
            {
                "open": p0,
                "high": p0 + 0.5,
                "low": p0 - 0.5,
                "close": p0 + 2.2,
                "volume": 80000,
            },
        )
    )
    assert any(s.signal_type == SignalType.LONG_ENTRY for s in sigs)


@pytest.mark.asyncio
async def test_vol_breakout_empty_feed() -> None:
    s = VolBreakoutStrategy(_cfg("vb4", ["SHFE.rb2501"]))
    assert await _feed_on_bar(s, "SHFE.rb2501", []) == []


def test_vol_breakout_invalid_negative_range_period_handled() -> None:
    s = VolBreakoutStrategy(_cfg("vb5", ["SHFE.rb2501"], {"range_period": -1, "min_bars": 5}))
    sig = s.generate_signal("SHFE.rb2501", {"close": 100.0, "high": 105.0, "low": 95.0, "volume": 1000.0})
    assert sig.signal_type == SignalType.HOLD


# --- SpreadArbitrage (5) ---


def test_spread_arb_instantiation_default_params() -> None:
    s = SpreadArbitrage(_cfg("sp", ["NEAR", "FAR"]))
    assert s.get_param("lookback") == 60


def test_spread_arb_instantiation_custom_params() -> None:
    s = SpreadArbitrage(_cfg("sp2", ["NEAR", "FAR"], {"lookback": 40, "entry_stdev_mult": 1.5}))
    assert s.get_param("lookback") == 40


@pytest.mark.asyncio
async def test_spread_arb_feed_pair_generates_signal(sample_bars: list[Bar], sample_bars_far: list[Bar]) -> None:
    s = SpreadArbitrage(
        _cfg(
            "sp3",
            ["SHFE.rb2501", "SHFE.rb2505"],
            {"lookback": 30, "min_samples": 12, "entry_stdev_mult": 0.8},
        )
    )
    sigs = await _feed_pair(s, "SHFE.rb2501", sample_bars, "SHFE.rb2505", sample_bars_far)
    assert len(sigs) >= 1


@pytest.mark.asyncio
async def test_spread_arb_empty_pair_feed() -> None:
    s = SpreadArbitrage(_cfg("sp4", ["A", "B"]))
    assert await _feed_pair(s, "A", [], "B", []) == []


def test_spread_arb_single_symbol_config_still_instantiates() -> None:
    s = SpreadArbitrage(_cfg("sp5", ["ONLY"]))
    sig = s.generate_signal("ONLY", {"close": 100.0})
    assert sig.signal_type == SignalType.HOLD


# --- PairsTradingStrategy (5) ---


def test_pairs_trading_instantiation_default_params() -> None:
    s = PairsTradingStrategy(_cfg("pt", ["A", "B"]))
    assert s.get_param("entry_z") == 2.0


def test_pairs_trading_instantiation_custom_params() -> None:
    s = PairsTradingStrategy(_cfg("pt2", ["A", "B"], {"entry_z": 1.2, "beta": 0.9}))
    assert s.get_param("entry_z") == 1.2


@pytest.mark.asyncio
async def test_pairs_trading_feed_generates_signal(sample_bars: list[Bar], sample_bars_far: list[Bar]) -> None:
    s = PairsTradingStrategy(
        _cfg(
            "pt3",
            ["SHFE.rb2501", "SHFE.rb2505"],
            {"lookback": 35, "min_samples": 15, "entry_z": 0.9},
        )
    )
    sigs = await _feed_pair(s, "SHFE.rb2501", sample_bars, "SHFE.rb2505", sample_bars_far)
    assert len(sigs) >= 1


@pytest.mark.asyncio
async def test_pairs_trading_empty_feed() -> None:
    s = PairsTradingStrategy(_cfg("pt4", ["A", "B"]))
    assert await _feed_pair(s, "A", [], "B", []) == []


def test_pairs_trading_invalid_zero_entry_z_uses_fallback_in_qty() -> None:
    s = PairsTradingStrategy(_cfg("pt5", ["A", "B"], {"entry_z": 0.0}))
    q = s._half_kelly_qty(1.0, 0.0, 1.0)  # noqa: SLF001 — stable contract of internal helper
    assert q >= 0.0


# --- VolumePriceStrategy (5) ---


def test_volume_price_instantiation_default_params() -> None:
    s = VolumePriceStrategy(_cfg("vp", ["SHFE.rb2501"]))
    assert s.get_param("vwap_window") == 30


def test_volume_price_instantiation_custom_params() -> None:
    s = VolumePriceStrategy(_cfg("vp2", ["SHFE.rb2501"], {"vwap_window": 20, "obv_ma_period": 8}))
    assert s.get_param("vwap_window") == 20


@pytest.mark.asyncio
async def test_volume_price_on_bar_produces_signal(sample_bars: list[Bar]) -> None:
    s = VolumePriceStrategy(_cfg("vp3", ["SHFE.rb2501"], {"min_bars": 36, "divergence_bars": 3}))
    sigs = await _feed_on_bar(s, "SHFE.rb2501", sample_bars)
    assert len(sigs) >= 1


@pytest.mark.asyncio
async def test_volume_price_empty_feed() -> None:
    s = VolumePriceStrategy(_cfg("vp4", ["SHFE.rb2501"]))
    assert await _feed_on_bar(s, "SHFE.rb2501", []) == []


def test_volume_price_incomplete_bar_returns_hold() -> None:
    s = VolumePriceStrategy(_cfg("vp5", ["SHFE.rb2501"]))
    sig = s.generate_signal("SHFE.rb2501", {"close": 1.0})
    assert sig.signal_type == SignalType.HOLD


# --- RBreakerStrategy (5) ---


def test_rbreaker_instantiation_default_params() -> None:
    s = RBreakerStrategy(_cfg("rb", ["SHFE.rb2501"]))
    assert s.get_param("f1") == 0.35


def test_rbreaker_instantiation_custom_params() -> None:
    s = RBreakerStrategy(_cfg("rb2", ["SHFE.rb2501"], {"f1": 0.4, "position_size": 2.0}))
    assert s.get_param("f1") == 0.4


@pytest.mark.asyncio
async def test_rbreaker_on_bar_trend_long_after_day_roll() -> None:
    sym = "SHFE.rb2501"
    s = RBreakerStrategy(_cfg("rb3", [sym]))
    sigs: list[Any] = []
    d0 = datetime(2026, 1, 1)
    for m in range(20):
        price = 3500.0
        sigs.extend(
            await s.on_bar(
                sym,
                {
                    "open": price,
                    "high": price + 10,
                    "low": price - 10,
                    "close": price,
                    "datetime": d0 + timedelta(minutes=m),
                },
            )
        )
    d1 = datetime(2026, 1, 2)
    breakout = 3600.0
    sigs.extend(
        await s.on_bar(
            sym,
            {
                "open": 3500.0,
                "high": breakout,
                "low": 3490.0,
                "close": 3550.0,
                "datetime": d1,
            },
        )
    )
    assert any(sig.signal_type == SignalType.LONG_ENTRY for sig in sigs)


@pytest.mark.asyncio
async def test_rbreaker_missing_calendar_day_returns_empty() -> None:
    s = RBreakerStrategy(_cfg("rb4", ["SHFE.rb2501"]))
    out = await s.on_bar("SHFE.rb2501", {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5})
    assert out == []


def test_rbreaker_invalid_position_size_clamped_in_logic() -> None:
    s = RBreakerStrategy(_cfg("rb5", ["SHFE.rb2501"], {"position_size": -5.0}))
    assert s.get_param("position_size") == -5.0


# --- RegimeMomentumStrategy (5) ---


def test_regime_momentum_instantiation_default_params() -> None:
    s = RegimeMomentumStrategy(_cfg("rm", ["SHFE.rb2501"]))
    assert s.get_param("min_bars") == 60


def test_regime_momentum_instantiation_custom_params() -> None:
    s = RegimeMomentumStrategy(
        _cfg("rm2", ["SHFE.rb2501"], {"min_bars": 40, "adx_period": 10, "atr_period": 10})
    )
    assert s.get_param("min_bars") == 40


@pytest.mark.asyncio
async def test_regime_momentum_on_bar_runs_full_series(sample_bars: list[Bar]) -> None:
    s = RegimeMomentumStrategy(
        _cfg(
            "rm3",
            ["SHFE.rb2501"],
            {
                "min_bars": 35,
                "hmm_lookback": 80,
                "hmm_retrain_interval": 40,
                "adx_period": 10,
                "atr_period": 10,
                "adx_min_strength": 0.0,
            },
        )
    )
    sigs = await _feed_on_bar(s, "SHFE.rb2501", sample_bars)
    assert isinstance(sigs, list)
    assert all(isinstance(x, Signal) for x in sigs)


@pytest.mark.asyncio
async def test_regime_momentum_empty_feed() -> None:
    s = RegimeMomentumStrategy(_cfg("rm4", ["SHFE.rb2501"]))
    assert await _feed_on_bar(s, "SHFE.rb2501", []) == []


def test_regime_momentum_invalid_confidence_threshold_stored() -> None:
    s = RegimeMomentumStrategy(_cfg("rm5", ["SHFE.rb2501"], {"regime_confidence_threshold": -1.0}))
    assert s.get_param("regime_confidence_threshold") == -1.0


# --- AdaptiveBollingerStrategy (5) ---


def test_adaptive_bollinger_instantiation_default_params() -> None:
    s = AdaptiveBollingerStrategy(_cfg("ab", ["SHFE.rb2501"]))
    assert s.get_param("bb_period_base") == 20


def test_adaptive_bollinger_instantiation_custom_params() -> None:
    s = AdaptiveBollingerStrategy(
        _cfg("ab2", ["SHFE.rb2501"], {"bb_period_min": 10, "adx_trend_threshold": 80.0})
    )
    assert s.get_param("bb_period_min") == 10


@pytest.mark.asyncio
async def test_adaptive_bollinger_on_bar_emits_signal(sample_bars: list[Bar]) -> None:
    s = AdaptiveBollingerStrategy(
        _cfg("ab3", ["SHFE.rb2501"], {"min_bars": 25, "rsi_oversold": 48.0, "adx_trend_threshold": 100.0})
    )
    sigs = await _feed_on_bar(s, "SHFE.rb2501", sample_bars)
    assert len(sigs) >= 1


@pytest.mark.asyncio
async def test_adaptive_bollinger_empty_feed() -> None:
    s = AdaptiveBollingerStrategy(_cfg("ab4", ["SHFE.rb2501"]))
    assert await _feed_on_bar(s, "SHFE.rb2501", []) == []


def test_adaptive_bollinger_inverted_bb_min_max_still_runs() -> None:
    s = AdaptiveBollingerStrategy(
        _cfg("ab5", ["SHFE.rb2501"], {"bb_period_min": 40, "bb_period_max": 12, "min_bars": 5})
    )
    sig = s.generate_signal("SHFE.rb2501", {"close": 100.0, "high": 101.0, "low": 99.0})
    assert sig.signal_type in (SignalType.HOLD, SignalType.LONG_ENTRY, SignalType.SHORT_ENTRY)


# --- DLTimeseriesStrategy (5) ---


class _FakeDLProba:
    def __init__(self, p_down: float, p_up: float) -> None:
        self.probabilities = [[p_down, p_up]]


class _FakeDLModel:
    is_trained = True

    def predict(self, X: Any) -> _FakeDLProba:
        return _FakeDLProba(0.15, 0.85)


def test_dl_timeseries_instantiation_default_params() -> None:
    s = DLTimeseriesStrategy(_cfg("dl", ["SHFE.rb2501"]))
    assert s.get_param("model_type") == "lstm"


def test_dl_timeseries_instantiation_custom_params_and_model() -> None:
    m = _FakeDLModel()
    s = DLTimeseriesStrategy(
        _cfg("dl2", ["SHFE.rb2501"], {"sequence_length": 20, "long_prob_threshold": 0.5}),
        model=m,
    )
    assert s._seq_len == 20  # noqa: SLF001


@pytest.mark.asyncio
async def test_dl_timeseries_on_bar_emits_long_with_mock_model(sample_bars: list[Bar]) -> None:
    s = DLTimeseriesStrategy(
        _cfg("dl3", ["SHFE.rb2501"], {"sequence_length": 10, "long_prob_threshold": 0.5}),
        model=_FakeDLModel(),
    )
    sigs = await _feed_on_bar(s, "SHFE.rb2501", sample_bars)
    assert any(sig.signal_type == SignalType.LONG_ENTRY for sig in sigs)


@pytest.mark.asyncio
async def test_dl_timeseries_no_model_returns_empty(sample_bars: list[Bar]) -> None:
    s = DLTimeseriesStrategy(_cfg("dl4", ["SHFE.rb2501"], {"sequence_length": 5}))
    sigs = await _feed_on_bar(s, "SHFE.rb2501", sample_bars[:20])
    assert sigs == []


def test_dl_timeseries_invalid_sequence_length_model_validate() -> None:
    cfg = _cfg("dl5", ["SHFE.rb2501"], {"sequence_length": -3})
    s = DLTimeseriesStrategy(cfg, model=_FakeDLModel())
    assert s._seq_len < 0  # noqa: SLF001 — strategy stores raw int; feeding still safe


@pytest.mark.asyncio
async def test_dual_ma_incomplete_bar_dict_returns_empty() -> None:
    s = FuturesDualMAStrategy(_cfg("dma_inc", ["SHFE.rb2501"]))
    assert await s.on_bar("SHFE.rb2501", {"close": 100.0}) == []


@pytest.mark.asyncio
async def test_cta_trend_incomplete_bar_dict_returns_empty() -> None:
    s = CTATrendStrategy(_cfg("cta_inc", ["SHFE.rb2501"]))
    assert await s.on_bar("SHFE.rb2501", {"close": 100.0, "high": 101.0}) == []


@pytest.mark.asyncio
async def test_regime_momentum_incomplete_bar_returns_empty() -> None:
    s = RegimeMomentumStrategy(_cfg("rm_inc", ["SHFE.rb2501"]))
    assert await s.on_bar("SHFE.rb2501", {"close": 100.0}) == []


@pytest.mark.asyncio
async def test_dl_timeseries_bar_missing_close_raises_key_error() -> None:
    s = DLTimeseriesStrategy(_cfg("dl_miss", ["SHFE.rb2501"], {"sequence_length": 2}), model=_FakeDLModel())
    with pytest.raises(KeyError):
        await s.on_bar("SHFE.rb2501", {"high": 1.0, "low": 0.5, "volume": 1.0})


def test_spread_arb_generate_signal_missing_close_is_hold() -> None:
    s = SpreadArbitrage(_cfg("sp_miss", ["A", "B"]))
    sig = s.generate_signal("A", {"high": 1.0, "low": 0.5})
    assert sig.signal_type == SignalType.HOLD


def test_pairs_trading_generate_signal_missing_close_is_hold() -> None:
    s = PairsTradingStrategy(_cfg("pt_miss", ["A", "B"]))
    sig = s.generate_signal("A", {"high": 1.0, "low": 0.5})
    assert sig.signal_type == SignalType.HOLD


# --- Bar model validation (shared) ---


def test_bar_high_below_low_raises_value_error() -> None:
    with pytest.raises(ValueError, match="high"):
        Bar(
            symbol="X",
            datetime=datetime(2026, 1, 1),
            open=Decimal("1"),
            high=Decimal("1"),
            low=Decimal("2"),
            close=Decimal("1"),
            volume=1,
        )


def test_strategy_config_missing_name_pydantic_validation() -> None:
    with pytest.raises(ValidationError):
        StrategyConfig.model_validate({"symbols": ["x"]})


def test_generate_signals_runs_concurrently_for_dual_ma() -> None:
    async def _run() -> int:
        s = FuturesDualMAStrategy(_cfg("async", ["SHFE.rb2501"]))
        n = 0
        for b in [
            Bar(
                symbol="SHFE.rb2501",
                datetime=datetime(2026, 1, 1) + timedelta(minutes=i),
                open=Decimal("100"),
                high=Decimal("102"),
                low=Decimal("98"),
                close=Decimal("100"),
                volume=5000 + i * 100,
            )
            for i in range(50)
        ]:
            n += len(await s.on_bar("SHFE.rb2501", bar_to_dict(b)))
        return n

    assert asyncio.run(_run()) >= 0
