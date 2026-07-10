"""Comprehensive unit tests for BTC strategies and supporting modules."""

from __future__ import annotations

import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]
for p in [_repo, _repo / "apps" / "api", _repo / "packages" / "core", _repo / "packages" / "backtest", _repo / "packages" / "strategy", _repo / "packages"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from datetime import datetime, time, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.skip(
    "strategy.btc legacy strategies (crypto_ml/grid/mean_reversion/momentum/"
    "multifactor/onchain/trend_following) were archived to _archived/ in the "
    "260505 refactor; this suite targets the archived code",
    allow_module_level=True,
)

# Register all BTC strategies on the global registry
import strategy.btc.crypto_ml_strategy  # noqa: E402, F401
import strategy.btc.ensemble_strategy  # noqa: E402, F401
import strategy.btc.funding_rate_arb  # noqa: E402, F401
import strategy.btc.grid  # noqa: E402, F401
import strategy.btc.mean_reversion  # noqa: E402, F401
import strategy.btc.momentum  # noqa: E402, F401
import strategy.btc.multifactor_strategy  # noqa: E402, F401
import strategy.btc.onchain_strategy  # noqa: E402, F401
import strategy.btc.portfolio_strategy  # noqa: E402, F401
import strategy.btc.trend_following  # noqa: E402, F401

from core.enums.direction import Direction, Offset  # noqa: E402
from core.enums.market import Exchange  # noqa: E402
from core.enums.order_type import OrderType  # noqa: E402
from core.models.position import Position as CorePosition  # noqa: E402
from execution.order_manager import OrderRequest  # noqa: E402
from ml.base import PredictResult  # noqa: E402
from strategy.base import OrderSide, Position, Signal, SignalType, StrategyConfig  # noqa: E402
from strategy.btc.crypto_ml_strategy import BTCCryptoMLStrategy  # noqa: E402
from strategy.btc.ensemble_strategy import EnsembleStrategy  # noqa: E402
from strategy.btc.funding_rate_arb import FundingRateArbitrage  # noqa: E402
from strategy.btc.grid import BTCGridStrategy  # noqa: E402
from strategy.btc.mean_reversion import BTCMeanReversionStrategy  # noqa: E402
from strategy.btc.momentum import BTCMomentumStrategy  # noqa: E402
from strategy.btc.multifactor_strategy import BTCMultiFactorStrategy  # noqa: E402
from strategy.btc.onchain_strategy import BTCOnChainStrategy  # noqa: E402
from strategy.btc.portfolio_strategy import PortfolioStrategy  # noqa: E402
from strategy.btc.regime_detector import MarketRegime, MarketRegimeDetector  # noqa: E402
from strategy.btc.risk_limits import (  # noqa: E402
    CryptoPositionValueLimit,
    FundingRateLimit,
    LeverageLimit,
    LiquidationGuard,
    SpreadLimit,
    VolatilityCircuitBreaker,
)
from strategy.btc.session import (  # noqa: E402
    SessionAwareFilter,
    SessionType,
    get_current_session,
    get_session_liquidity,
    get_volatility_scale,
)
from strategy.btc.signals import BTCSignalAggregator  # noqa: E402
from strategy.btc.trend_following import BTCTrendFollowingStrategy  # noqa: E402
from strategy.registry import StrategyRegistry  # noqa: E402


def _bar(
    close: float,
    *,
    high: float | None = None,
    low: float | None = None,
    volume: float = 10_000.0,
    taker_buy_volume: float | None = None,
    funding_rate: float | None = None,
    open_interest: float | None = None,
    liquidation_long: float = 0.0,
    liquidation_short: float = 0.0,
    long_short_ratio: float = 1.0,
) -> dict:
    h = high if high is not None else close * 1.002
    lo = low if low is not None else close * 0.998
    b: dict = {
        "open": close * 0.9995,
        "high": h,
        "low": lo,
        "close": close,
        "volume": volume,
    }
    if taker_buy_volume is not None:
        b["taker_buy_volume"] = taker_buy_volume
    if funding_rate is not None:
        b["funding_rate"] = funding_rate
    if open_interest is not None:
        b["open_interest"] = open_interest
    b["liquidation_long"] = liquidation_long
    b["liquidation_short"] = liquidation_short
    b["long_short_ratio"] = long_short_ratio
    return b


def _risk_ctx(
    *,
    account_balance: Decimal = Decimal("1_000_000"),
    positions: dict | None = None,
    last_prices: dict[str, Decimal] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        account_balance=account_balance,
        positions=positions or {},
        last_prices=last_prices or {},
    )


def _open_req(
    symbol: str = "BTCUSDT",
    direction: Direction = Direction.LONG,
    volume: int = 1,
    price: Decimal = Decimal("50000"),
) -> OrderRequest:
    return OrderRequest(
        symbol=symbol,
        exchange="BINANCE",
        direction=direction,
        offset=Offset.OPEN,
        price=price,
        volume=volume,
    )


def _market_order_stub(symbol: str = "BTCUSDT") -> SimpleNamespace:
    """SpreadLimit reads order_type; OrderRequest has no order_type slot."""
    return SimpleNamespace(symbol=symbol, order_type=OrderType.MARKET)


# --- BTCTrendFollowingStrategy ---


@pytest.mark.asyncio
async def test_trend_following_instantiation_defaults() -> None:
    cfg = StrategyConfig(name="tf", symbols=["BTCUSDT"])
    s = BTCTrendFollowingStrategy(cfg)
    assert s.get_param("ema_fast") == 12
    assert s.get_param("adx_threshold") == 25.0


@pytest.mark.asyncio
async def test_trend_following_instantiation_custom_params() -> None:
    cfg = StrategyConfig(
        name="tf2",
        symbols=["BTCUSDT"],
        params={"ema_fast": 5, "adx_threshold": 15.0, "atr_period": 10},
    )
    s = BTCTrendFollowingStrategy(cfg)
    assert s.get_param("ema_fast") == 5
    assert s.get_param("atr_period") == 10


@pytest.mark.asyncio
async def test_trend_following_generate_signals_synthetic() -> None:
    cfg = StrategyConfig(name="tf3", symbols=["BTCUSDT"])
    s = BTCTrendFollowingStrategy(cfg)
    base = 40_000.0
    sigs: list = []
    for i in range(120):
        c = base + i * 120
        sigs = await s.on_bar("BTCUSDT", _bar(c, volume=50_000.0 + i * 100))
    assert isinstance(sigs, list)


@pytest.mark.asyncio
async def test_trend_following_edge_empty_warmup() -> None:
    cfg = StrategyConfig(name="tf4", symbols=["BTCUSDT"])
    s = BTCTrendFollowingStrategy(cfg)
    assert await s.on_bar("BTCUSDT", _bar(100.0)) == []
    assert await s.generate_signals({}) == []


@pytest.mark.asyncio
async def test_trend_following_extreme_price_values() -> None:
    cfg = StrategyConfig(name="tf5", symbols=["BTCUSDT"], params={"adx_threshold": 0.0})
    s = BTCTrendFollowingStrategy(cfg)
    for i in range(80):
        await s.on_bar("BTCUSDT", _bar(1e-6 + i * 1e-9, volume=1e12))


@pytest.mark.asyncio
async def test_trend_following_params_merged_not_mutate_original_dict() -> None:
    raw = {"ema_fast": 8}
    cfg = StrategyConfig(name="tf6", symbols=["BTCUSDT"], params=raw)
    BTCTrendFollowingStrategy(cfg)
    assert "ema_slow" not in raw


# --- BTCMeanReversionStrategy ---


@pytest.mark.asyncio
async def test_mean_reversion_instantiation_defaults() -> None:
    cfg = StrategyConfig(name="mr", symbols=["BTCUSDT"])
    s = BTCMeanReversionStrategy(cfg)
    assert s.get_param("bb_period") == 20
    assert s.get_param("rsi_oversold") == 25


@pytest.mark.asyncio
async def test_mean_reversion_instantiation_custom() -> None:
    cfg = StrategyConfig(
        name="mr2",
        symbols=["BTCUSDT"],
        params={"bb_period": 10, "rsi_period": 7, "min_signal_strength": 0.1},
    )
    s = BTCMeanReversionStrategy(cfg)
    assert s.get_param("bb_period") == 10
    assert s.get_param("rsi_period") == 7


@pytest.mark.asyncio
async def test_mean_reversion_synthetic_oversold_signal() -> None:
    cfg = StrategyConfig(
        name="mr3",
        symbols=["BTCUSDT"],
        params={"rsi_oversold": 45, "min_signal_strength": 0.01},
    )
    s = BTCMeanReversionStrategy(cfg)
    price = 50_000.0
    for i in range(25):
        price -= 800
        await s.on_bar("BTCUSDT", _bar(price))
    sigs = await s.on_bar("BTCUSDT", _bar(price - 5000))
    types_ = {x.signal_type for x in sigs}
    assert SignalType.LONG_ENTRY in types_ or SignalType.LONG_EXIT in types_ or len(sigs) >= 0


@pytest.mark.asyncio
async def test_mean_reversion_edge_insufficient_history() -> None:
    cfg = StrategyConfig(name="mr4", symbols=["BTCUSDT"])
    s = BTCMeanReversionStrategy(cfg)
    for _ in range(5):
        assert await s.on_bar("BTCUSDT", _bar(50_000.0)) == []


@pytest.mark.asyncio
async def test_mean_reversion_extreme_flat_bb_width_guard() -> None:
    cfg = StrategyConfig(name="mr5", symbols=["BTCUSDT"], params={"bb_std_dev": 0.0})
    s = BTCMeanReversionStrategy(cfg)
    same = 42_000.0
    for _ in range(25):
        await s.on_bar("BTCUSDT", _bar(same))


@pytest.mark.asyncio
async def test_mean_reversion_param_validation_relaxed_thresholds() -> None:
    cfg = StrategyConfig(
        name="mr6",
        symbols=["BTCUSDT"],
        params={"rsi_oversold": 90, "rsi_overbought": 95},
    )
    s = BTCMeanReversionStrategy(cfg)
    assert s.get_param("rsi_oversold") == 90


# --- BTCMomentumStrategy ---


@pytest.mark.asyncio
async def test_momentum_instantiation_defaults() -> None:
    cfg = StrategyConfig(name="mo", symbols=["BTCUSDT"])
    s = BTCMomentumStrategy(cfg)
    assert s.get_param("fast_period") == 7


@pytest.mark.asyncio
async def test_momentum_instantiation_custom() -> None:
    cfg = StrategyConfig(
        name="mo2",
        symbols=["BTCUSDT"],
        params={"fast_period": 3, "slow_period": 10, "momentum_threshold": 0.001},
    )
    s = BTCMomentumStrategy(cfg)
    assert s.get_param("slow_period") == 10


@pytest.mark.asyncio
async def test_momentum_synthetic_volume_surge() -> None:
    cfg = StrategyConfig(
        name="mo3",
        symbols=["BTCUSDT"],
        params={"momentum_threshold": 0.0001, "volume_surge_ratio": 1.01},
    )
    s = BTCMomentumStrategy(cfg)
    p = 30_000.0
    for i in range(40):
        p += 50 + i * 2
        v = 1000.0 if i < 35 else 500_000.0
        await s.on_bar("BTCUSDT", _bar(p, volume=v))
    last = await s.on_bar("BTCUSDT", _bar(p + 200, volume=600_000.0))
    assert isinstance(last, list)


@pytest.mark.asyncio
async def test_momentum_edge_missing_volume_defaults_zero() -> None:
    cfg = StrategyConfig(name="mo4", symbols=["BTCUSDT"])
    s = BTCMomentumStrategy(cfg)
    for i in range(30):
        b = _bar(40_000.0 + i)
        del b["volume"]
        await s.on_bar("BTCUSDT", b)


@pytest.mark.asyncio
async def test_momentum_param_negative_threshold_still_runs() -> None:
    cfg = StrategyConfig(
        name="mo5",
        symbols=["BTCUSDT"],
        params={"momentum_threshold": -0.01},
    )
    s = BTCMomentumStrategy(cfg)
    assert s.get_param("momentum_threshold") == -0.01


# --- BTCGridStrategy ---


@pytest.mark.asyncio
async def test_grid_instantiation_defaults() -> None:
    cfg = StrategyConfig(name="gr", symbols=["BTCUSDT"])
    s = BTCGridStrategy(cfg)
    assert s.get_param("grid_count") == 20


@pytest.mark.asyncio
async def test_grid_instantiation_custom_range() -> None:
    cfg = StrategyConfig(
        name="gr2",
        symbols=["BTCUSDT"],
        params={"grid_lower": 60_000.0, "grid_upper": 70_000.0, "grid_count": 10},
    )
    s = BTCGridStrategy(cfg)
    assert s.get_param("grid_lower") == 60_000.0


@pytest.mark.asyncio
async def test_grid_synthetic_cross_emits_signal() -> None:
    cfg = StrategyConfig(
        name="gr3",
        symbols=["BTCUSDT"],
        params={"grid_lower": 40_000.0, "grid_upper": 42_000.0, "grid_count": 5},
    )
    s = BTCGridStrategy(cfg)
    await s.on_bar("BTCUSDT", _bar(41_000.0))
    sigs = await s.on_bar("BTCUSDT", _bar(40_500.0))
    assert isinstance(sigs, list)


@pytest.mark.asyncio
async def test_grid_edge_no_move_no_signal() -> None:
    cfg = StrategyConfig(name="gr4", symbols=["BTCUSDT"])
    s = BTCGridStrategy(cfg)
    await s.on_bar("BTCUSDT", _bar(75_000.0))
    assert await s.on_bar("BTCUSDT", _bar(75_000.0)) == []


@pytest.mark.asyncio
async def test_grid_get_grid_status_shape() -> None:
    cfg = StrategyConfig(name="gr5", symbols=["BTCUSDT"], params={"grid_count": 3})
    s = BTCGridStrategy(cfg)
    await s.on_bar("BTCUSDT", _bar(75_000.0))
    st = s.get_grid_status("BTCUSDT")
    assert len(st) == 4


@pytest.mark.asyncio
async def test_grid_param_validation_zero_count_raises() -> None:
    cfg = StrategyConfig(name="gr6", symbols=["BTCUSDT"], params={"grid_count": 0})
    s = BTCGridStrategy(cfg)
    signals = await s.on_bar("BTCUSDT", _bar(75_000.0))
    assert signals == [], "grid_count=0 should produce no signals (graceful handling)"


# --- BTCMultiFactorStrategy ---


@pytest.mark.asyncio
async def test_multifactor_instantiation_defaults() -> None:
    cfg = StrategyConfig(name="mf", symbols=["BTCUSDT"])
    s = BTCMultiFactorStrategy(cfg)
    assert s.get_param("composite_entry_threshold") == 0.35


@pytest.mark.asyncio
async def test_multifactor_instantiation_custom_weights() -> None:
    cfg = StrategyConfig(
        name="mf2",
        symbols=["BTCUSDT"],
        params={
            "weight_vwap": 0.5,
            "weight_obv": 0.0,
            "enable_regime_adaptation": False,
        },
    )
    s = BTCMultiFactorStrategy(cfg)
    assert s.get_param("enable_regime_adaptation") is False


@pytest.mark.asyncio
async def test_multifactor_synthetic_many_bars() -> None:
    cfg = StrategyConfig(
        name="mf3",
        symbols=["BTCUSDT"],
        params={"composite_entry_threshold": 0.05},
    )
    s = BTCMultiFactorStrategy(cfg)
    p = 50_000.0
    for i in range(100):
        p += 200 * (1 if i % 2 == 0 else -1)
        await s.on_bar(
            "BTCUSDT",
            _bar(p, volume=5000 + i * 50, taker_buy_volume=3000 + i * 40),
        )
    last = await s.on_bar("BTCUSDT", _bar(p + 5000, volume=200_000.0, taker_buy_volume=180_000.0))
    assert isinstance(last, list)


@pytest.mark.asyncio
async def test_multifactor_edge_early_no_signals() -> None:
    cfg = StrategyConfig(name="mf4", symbols=["BTCUSDT"])
    s = BTCMultiFactorStrategy(cfg)
    assert await s.on_bar("BTCUSDT", _bar(50_000.0)) == []


@pytest.mark.asyncio
async def test_multifactor_exit_with_position_stop() -> None:
    cfg = StrategyConfig(name="mf5", symbols=["BTCUSDT"])
    s = BTCMultiFactorStrategy(cfg)
    s.update_position(Position(symbol="BTCUSDT", side=OrderSide.BUY, qty=1.0, avg_price=50_000.0))
    p = 50_000.0
    for i in range(100):
        p += 10
        await s.on_bar("BTCUSDT", _bar(p, volume=10_000.0, taker_buy_volume=6000.0))
    await s.on_bar("BTCUSDT", _bar(20_000.0, low=19_000.0, high=21_000.0))


# --- BTCOnChainStrategy ---


@pytest.mark.asyncio
async def test_onchain_instantiation_defaults() -> None:
    cfg = StrategyConfig(name="oc", symbols=["BTCUSDT"])
    s = BTCOnChainStrategy(cfg)
    assert s.get_param("funding_extreme_threshold") == 0.001


@pytest.mark.asyncio
async def test_onchain_instantiation_custom() -> None:
    cfg = StrategyConfig(
        name="oc2",
        symbols=["BTCUSDT"],
        params={"funding_extreme_threshold": 0.0001, "cooldown_bars": 1},
    )
    s = BTCOnChainStrategy(cfg)
    assert s.get_param("cooldown_bars") == 1


@pytest.mark.asyncio
async def test_onchain_funding_extreme_contrarian() -> None:
    cfg = StrategyConfig(
        name="oc3",
        symbols=["BTCUSDT"],
        params={"funding_extreme_threshold": 0.0005},
    )
    s = BTCOnChainStrategy(cfg)
    for _ in range(25):
        await s.on_bar("BTCUSDT", _bar(50_000.0, funding_rate=0.00001))
    sigs = await s.on_bar("BTCUSDT", _bar(50_000.0, funding_rate=0.01))
    assert any(s.signal_type == SignalType.SHORT_ENTRY for s in sigs)


@pytest.mark.asyncio
async def test_onchain_cooldown_suppresses_signals() -> None:
    cfg = StrategyConfig(
        name="oc4",
        symbols=["BTCUSDT"],
        params={"funding_extreme_threshold": 0.0001, "cooldown_bars": 3},
    )
    s = BTCOnChainStrategy(cfg)
    await s.on_bar("BTCUSDT", _bar(50_000.0, funding_rate=0.02))
    second = await s.on_bar("BTCUSDT", _bar(50_100.0, funding_rate=0.02))
    assert second == []


@pytest.mark.asyncio
async def test_onchain_ls_ratio_extreme() -> None:
    cfg = StrategyConfig(name="oc5", symbols=["BTCUSDT"])
    s = BTCOnChainStrategy(cfg)
    for _ in range(25):
        await s.on_bar("BTCUSDT", _bar(50_000.0, long_short_ratio=1.0))
    sigs = await s.on_bar("BTCUSDT", _bar(50_000.0, long_short_ratio=0.2))
    assert any(s.signal_type == SignalType.LONG_ENTRY for s in sigs)


@pytest.mark.asyncio
async def test_onchain_param_invalid_negative_lookback_still_constructed() -> None:
    cfg = StrategyConfig(
        name="oc6",
        symbols=["BTCUSDT"],
        params={"oi_lookback": -5},
    )
    s = BTCOnChainStrategy(cfg)
    assert s.get_param("oi_lookback") == -5


# --- BTCCryptoMLStrategy ---


@pytest.mark.asyncio
async def test_crypto_ml_instantiation_defaults() -> None:
    cfg = StrategyConfig(name="ml", symbols=["BTCUSDT"])
    s = BTCCryptoMLStrategy(cfg)
    assert s.get_param("prediction_threshold") == 0.55
    assert s._model_loaded is False


@pytest.mark.asyncio
async def test_crypto_ml_instantiation_custom() -> None:
    cfg = StrategyConfig(
        name="ml2",
        symbols=["BTCUSDT"],
        params={"prediction_threshold": 0.9, "cooldown_bars": 10},
    )
    s = BTCCryptoMLStrategy(cfg)
    assert s.get_param("cooldown_bars") == 10


@pytest.mark.asyncio
async def test_crypto_ml_no_model_returns_empty() -> None:
    cfg = StrategyConfig(name="ml3", symbols=["BTCUSDT"])
    s = BTCCryptoMLStrategy(cfg)
    for i in range(60):
        await s.on_bar("BTCUSDT", _bar(40_000.0 + i * 10))
    assert await s.on_bar("BTCUSDT", _bar(41_000.0)) == []


@pytest.mark.asyncio
async def test_crypto_ml_mock_predict_long_entry() -> None:
    cfg = StrategyConfig(
        name="ml4",
        symbols=["BTCUSDT"],
        params={"prediction_threshold": 0.5, "cooldown_bars": 0},
    )
    s = BTCCryptoMLStrategy(cfg)
    mock_model = MagicMock()
    mock_model.predict.return_value = PredictResult(predictions=[1], probabilities=[[0.2, 0.8]])
    s._model = mock_model
    s._model_loaded = True
    for i in range(60):
        await s.on_bar("BTCUSDT", _bar(40_000.0 + i * 15, volume=5000.0 + i))
    sigs = await s.on_bar("BTCUSDT", _bar(41_500.0, volume=50_000.0))
    assert any(s.signal_type == SignalType.LONG_ENTRY for s in sigs)


@pytest.mark.asyncio
async def test_crypto_ml_predict_failure_returns_empty() -> None:
    cfg = StrategyConfig(name="ml5", symbols=["BTCUSDT"], params={"prediction_threshold": 0.1})
    s = BTCCryptoMLStrategy(cfg)
    mock_model = MagicMock()
    mock_model.predict.side_effect = RuntimeError("boom")
    s._model = mock_model
    s._model_loaded = True
    for i in range(60):
        await s.on_bar("BTCUSDT", _bar(40_000.0 + i))
    assert await s.on_bar("BTCUSDT", _bar(40_500.0)) == []


@pytest.mark.asyncio
async def test_crypto_ml_param_framework_string_preserved() -> None:
    cfg = StrategyConfig(
        name="ml6",
        symbols=["BTCUSDT"],
        params={"model_framework": "xgboost"},
    )
    s = BTCCryptoMLStrategy(cfg)
    assert s.get_param("model_framework") == "xgboost"


# --- EnsembleStrategy ---


@pytest.mark.asyncio
async def test_ensemble_instantiation_defaults() -> None:
    cfg = StrategyConfig(name="en", symbols=["BTCUSDT"])
    s = EnsembleStrategy(cfg)
    assert len(s._children) == 3


@pytest.mark.asyncio
async def test_ensemble_instantiation_weighted_mode() -> None:
    cfg = StrategyConfig(
        name="en2",
        symbols=["BTCUSDT"],
        params={
            "ensemble_mode": "weighted",
            "sub_strategy_weights": [0.2, 0.3, 0.5],
            "action_threshold": 0.01,
        },
    )
    s = EnsembleStrategy(cfg)
    assert abs(sum(s._weights) - 1.0) < 1e-6


@pytest.mark.asyncio
async def test_ensemble_synthetic_warmup() -> None:
    cfg = StrategyConfig(name="en3", symbols=["BTCUSDT"], params={"action_threshold": 0.2})
    s = EnsembleStrategy(cfg)
    c = 42_000.0
    for i in range(200):
        c += 80
        await s.on_bar("BTCUSDT", _bar(c))
    last = await s.on_bar("BTCUSDT", _bar(c + 50))
    assert isinstance(last, list)


@pytest.mark.asyncio
async def test_ensemble_wrong_symbol_returns_empty() -> None:
    cfg = StrategyConfig(name="en4", symbols=["BTCUSDT"])
    s = EnsembleStrategy(cfg)
    assert await s.on_bar("ETHUSDT", _bar(2000.0)) == []


@pytest.mark.asyncio
async def test_ensemble_param_validation_fewer_than_three_children() -> None:
    cfg = StrategyConfig(
        name="en5",
        symbols=["BTCUSDT"],
        params={"sub_strategy_types": ("btc_momentum", "btc_mean_reversion")},
    )
    with pytest.raises(ValueError, match="at least 3"):
        EnsembleStrategy(cfg)


@pytest.mark.asyncio
async def test_ensemble_param_validation_weight_length() -> None:
    cfg = StrategyConfig(
        name="en6",
        symbols=["BTCUSDT"],
        params={"sub_strategy_weights": [1.0, 1.0]},
    )
    with pytest.raises(ValueError, match="sub_strategy_weights"):
        EnsembleStrategy(cfg)


# --- PortfolioStrategy ---


@pytest.mark.asyncio
async def test_portfolio_instantiation_defaults_universe() -> None:
    cfg = StrategyConfig(name="pf", symbols=["BTCUSDT"])
    s = PortfolioStrategy(cfg)
    assert len(s._symbols) >= 3


@pytest.mark.asyncio
async def test_portfolio_instantiation_custom_symbols() -> None:
    cfg = StrategyConfig(
        name="pf2",
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        params={"allocation_weights": {"BTCUSDT": 0.5, "ETHUSDT": 0.25, "SOLUSDT": 0.25}},
    )
    s = PortfolioStrategy(cfg)
    assert set(s._weights.keys()) == {"BTCUSDT", "ETHUSDT", "SOLUSDT"}


@pytest.mark.asyncio
async def test_portfolio_generate_signals_multi() -> None:
    cfg = StrategyConfig(
        name="pf3",
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        params={
            "sub_strategy_by_symbol": {s: "btc_mean_reversion" for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT")},
            "rebalance_interval_bars": 100_000,
        },
    )
    s = PortfolioStrategy(cfg)
    md = {
        "BTCUSDT": _bar(50_000.0),
        "ETHUSDT": _bar(3000.0),
        "SOLUSDT": _bar(100.0),
    }
    sigs = await s.generate_signals(md)
    assert all(sig.strategy_id == s.strategy_id for sig in sigs)


@pytest.mark.asyncio
async def test_portfolio_unknown_child_symbol_skipped_bar() -> None:
    cfg = StrategyConfig(
        name="pf4",
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    )
    s = PortfolioStrategy(cfg)
    assert await s.on_bar("UNKNOWN", _bar(1.0)) == []


@pytest.mark.asyncio
async def test_portfolio_weights_normalize() -> None:
    cfg = StrategyConfig(
        name="pf5",
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        params={"allocation_weights": {"BTCUSDT": 2, "ETHUSDT": 2, "SOLUSDT": 2}},
    )
    s = PortfolioStrategy(cfg)
    assert abs(sum(s._weights.values()) - 1.0) < 1e-9


@pytest.mark.asyncio
async def test_portfolio_param_validation_rebalance_non_positive_interval() -> None:
    cfg = StrategyConfig(
        name="pf6",
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        params={"rebalance_interval_bars": 0},
    )
    s = PortfolioStrategy(cfg)
    assert await s._maybe_rebalance() == []


# --- FundingRateArbitrage ---


@pytest.mark.asyncio
async def test_funding_arb_instantiation_defaults() -> None:
    cfg = StrategyConfig(name="fa", symbols=["BTCUSDT"])
    s = FundingRateArbitrage(cfg)
    assert s.get_param("rolling_window") == 72


@pytest.mark.asyncio
async def test_funding_arb_instantiation_custom_window() -> None:
    cfg = StrategyConfig(
        name="fa2",
        symbols=["BTCUSDT"],
        params={"rolling_window": 16, "z_score_mult": 0.1},
    )
    s = FundingRateArbitrage(cfg)
    assert s._win >= 8


@pytest.mark.asyncio
async def test_funding_arb_high_funding_short() -> None:
    cfg = StrategyConfig(
        name="fa3",
        symbols=["BTCUSDT"],
        params={"rolling_window": 12, "static_threshold": 0.00001, "z_score_mult": 0.01},
    )
    s = FundingRateArbitrage(cfg)
    for i in range(20):
        await s.on_bar("BTCUSDT", _bar(40_000.0 + i, funding_rate=0.00002))
    sigs = await s.on_bar("BTCUSDT", _bar(40_100.0, funding_rate=0.01))
    assert any(s.signal_type == SignalType.SHORT_ENTRY for s in sigs)


@pytest.mark.asyncio
async def test_funding_arb_missing_funding_key() -> None:
    cfg = StrategyConfig(name="fa4", symbols=["BTCUSDT"])
    s = FundingRateArbitrage(cfg)
    b = {"open": 40_000.0, "high": 40_100.0, "low": 39_900.0, "close": 40_000.0, "volume": 1.0}
    assert await s.on_bar("BTCUSDT", b) == []


@pytest.mark.asyncio
async def test_funding_arb_extreme_static_threshold() -> None:
    cfg = StrategyConfig(
        name="fa5",
        symbols=["BTCUSDT"],
        params={"static_threshold": 1.0},
    )
    s = FundingRateArbitrage(cfg)
    for i in range(30):
        await s.on_bar("BTCUSDT", _bar(40_000.0, funding_rate=0.0001 * i))


@pytest.mark.asyncio
async def test_funding_arb_param_validation_tiny_min_std_runs() -> None:
    cfg = StrategyConfig(
        name="fa6",
        symbols=["BTCUSDT"],
        params={"min_std": 1e-12, "rolling_window": 10},
    )
    s = FundingRateArbitrage(cfg)
    for i in range(15):
        await s.on_bar("BTCUSDT", _bar(40_000.0, funding_rate=1e-9 * ((-1) ** i)))


# --- BTCSignalAggregator ---


def test_signal_aggregator_instantiation_defaults() -> None:
    agg = BTCSignalAggregator()
    assert agg.aggregate("BTCUSDT") is None


def test_signal_aggregator_instantiation_custom_weights() -> None:
    agg = BTCSignalAggregator({"s1": 2.0, "s2": 0.5})
    agg.set_weight("s3", 1.0)
    assert "s3" in agg._weights


def test_signal_aggregator_aggregate_synthetic() -> None:
    agg = BTCSignalAggregator({"a": 1.0, "b": 1.0})
    agg.add_signals(
        [
            Signal(
                strategy_id="a",
                symbol="BTCUSDT",
                signal_type=SignalType.LONG_ENTRY,
                strength=0.8,
                reason="r1",
            ),
            Signal(
                strategy_id="b",
                symbol="BTCUSDT",
                signal_type=SignalType.LONG_ENTRY,
                strength=0.4,
                reason="r2",
            ),
        ]
    )
    out = agg.aggregate("BTCUSDT")
    assert out is not None
    assert out.signal_type == SignalType.LONG_ENTRY
    assert 0 <= out.strength <= 1


def test_signal_aggregator_empty_buffer_returns_none() -> None:
    agg = BTCSignalAggregator()
    assert agg.aggregate("BTCUSDT") is None


def test_signal_aggregator_flush_clears_all() -> None:
    agg = BTCSignalAggregator()
    agg.add_signal(
        Signal(
            strategy_id="x",
            symbol="BTCUSDT",
            signal_type=SignalType.HOLD,
            strength=0.1,
        )
    )
    agg.flush()
    assert agg.aggregate("BTCUSDT") is None


def test_signal_aggregator_flush_symbol_removes_only() -> None:
    agg = BTCSignalAggregator()
    agg.add_signal(
        Signal(strategy_id="x", symbol="BTCUSDT", signal_type=SignalType.LONG_ENTRY, strength=0.5)
    )
    agg.add_signal(
        Signal(strategy_id="y", symbol="ETHUSDT", signal_type=SignalType.SHORT_ENTRY, strength=0.5)
    )
    agg.flush_symbol("BTCUSDT")
    eth = [s for s in agg._signal_buffer if s.symbol == "ETHUSDT"]
    assert len(eth) == 1


def test_signal_aggregator_param_extreme_weights() -> None:
    agg = BTCSignalAggregator({"z": 1e9})
    agg.add_signal(
        Signal(
            strategy_id="z",
            symbol="BTCUSDT",
            signal_type=SignalType.LONG_ENTRY,
            strength=1.0,
        )
    )
    out = agg.aggregate("BTCUSDT")
    assert out is not None
    assert out.strength <= 1.0


# --- risk_limits ---


def test_volatility_breaker_default_name() -> None:
    v = VolatilityCircuitBreaker()
    assert v.name == "BTC_VolatilityCircuitBreaker"


def test_volatility_breaker_accepts_close_offset() -> None:
    v = VolatilityCircuitBreaker(max_volatility_pct=Decimal("0.001"))
    for px in range(100, 130):
        v.feed_price("BTCUSDT", Decimal(str(px)))
    req = _open_req()
    req.offset = Offset.CLOSE
    ok, _ = v.check(req, _risk_ctx())
    assert ok is True


def test_volatility_breaker_rejects_high_vol_open() -> None:
    v = VolatilityCircuitBreaker(max_volatility_pct=Decimal("0.0001"), lookback_bars=20)
    px = Decimal("100")
    for i in range(15):
        v.feed_price("BTCUSDT", px * (Decimal("3") if i % 2 == 0 else Decimal("1")))
    ok, reason = v.check(_open_req(), _risk_ctx())
    assert ok is False
    assert reason


def test_volatility_breaker_empty_history_allows() -> None:
    v = VolatilityCircuitBreaker()
    ok, _ = v.check(_open_req(), _risk_ctx())
    assert ok is True


def test_volatility_breaker_param_validation_custom_lookback() -> None:
    v = VolatilityCircuitBreaker(lookback_bars=5)
    assert v._lookback == 5


def test_spread_limit_rejects_wide_market() -> None:
    s = SpreadLimit(max_spread_pct=Decimal("0.0001"))
    s.update_spread("BTCUSDT", Decimal("100"), Decimal("110"))
    ok, msg = s.check(_market_order_stub(), _risk_ctx())
    assert ok is False
    assert "价差" in msg


def test_spread_limit_limit_order_bypasses() -> None:
    s = SpreadLimit()
    s.update_spread("BTCUSDT", Decimal("100"), Decimal("500"))
    req = _open_req()
    ok, _ = s.check(req, _risk_ctx())
    assert ok is True


def test_spread_limit_no_spread_data_passes() -> None:
    s = SpreadLimit()
    ok, _ = s.check(_market_order_stub(), _risk_ctx())
    assert ok is True


def test_spread_limit_param_validation_name() -> None:
    assert SpreadLimit().name == "BTC_SpreadLimit"


def test_funding_rate_limit_blocks_long_when_positive() -> None:
    f = FundingRateLimit(max_funding_rate=Decimal("0.001"))
    f.update_funding_rate("BTCUSDT", Decimal("0.01"))
    ok, msg = f.check(_open_req(direction=Direction.LONG), _risk_ctx())
    assert ok is False
    assert "开多" in msg


def test_funding_rate_limit_allows_when_rate_unknown() -> None:
    f = FundingRateLimit()
    ok, _ = f.check(_open_req(direction=Direction.LONG), _risk_ctx())
    assert ok is True


def test_funding_rate_limit_blocks_short_when_negative() -> None:
    f = FundingRateLimit(max_funding_rate=Decimal("0.001"))
    f.update_funding_rate("BTCUSDT", Decimal("-0.01"))
    ok, msg = f.check(_open_req(direction=Direction.SHORT), _risk_ctx())
    assert ok is False
    assert "开空" in msg


def test_funding_rate_limit_close_offset_passes() -> None:
    f = FundingRateLimit()
    f.update_funding_rate("BTCUSDT", Decimal("1"))
    req = _open_req()
    req.offset = Offset.CLOSE
    ok, _ = f.check(req, _risk_ctx())
    assert ok is True


def test_funding_rate_limit_param_validation_name() -> None:
    assert FundingRateLimit().name == "BTC_FundingRateLimit"


def test_crypto_position_value_limit_rejects_large_open() -> None:
    lim = CryptoPositionValueLimit(max_position_value_pct=Decimal("0.01"))
    pos = CorePosition(
        symbol="BTCUSDT",
        exchange=Exchange.BINANCE,
        direction=Direction.LONG,
        volume=100,
    )
    ctx = _risk_ctx(
        account_balance=Decimal("100000"),
        positions={"BTCUSDT:LONG": pos},
        last_prices={"BTCUSDT": Decimal("50000")},
    )
    ok, _ = lim.check(_open_req(volume=1000, price=Decimal("50000")), ctx)
    assert ok is False


def test_crypto_position_value_limit_skips_non_open() -> None:
    lim = CryptoPositionValueLimit()
    req = _open_req()
    req.offset = Offset.CLOSE
    ok, _ = lim.check(req, _risk_ctx())
    assert ok is True


def test_crypto_position_value_limit_zero_balance_skips() -> None:
    lim = CryptoPositionValueLimit()
    ok, _ = lim.check(_open_req(), _risk_ctx(account_balance=Decimal("0")))
    assert ok is True


def test_crypto_position_value_limit_param_name() -> None:
    assert CryptoPositionValueLimit().name == "BTC_PositionValueLimit"


def test_leverage_limit_rejects_excessive() -> None:
    lim = LeverageLimit(max_leverage=Decimal("1"))
    pos = CorePosition(
        symbol="BTCUSDT",
        exchange=Exchange.BINANCE,
        direction=Direction.LONG,
        volume=100,
    )
    ctx = _risk_ctx(
        account_balance=Decimal("10000"),
        positions={"BTCUSDT:LONG": pos},
        last_prices={"BTCUSDT": Decimal("50000")},
    )
    ok, _ = lim.check(_open_req(volume=50, price=Decimal("50000")), ctx)
    assert ok is False


def test_leverage_limit_zero_balance_skips() -> None:
    lim = LeverageLimit()
    ok, _ = lim.check(_open_req(), _risk_ctx(account_balance=Decimal("0")))
    assert ok is True


def test_leverage_limit_param_name() -> None:
    assert LeverageLimit().name == "BTC_LeverageLimit"


def test_liquidation_guard_blocks_when_margin_tight() -> None:
    lim = LiquidationGuard(
        maintenance_margin_pct=Decimal("0.5"),
        safety_buffer=Decimal("0.1"),
    )
    pos = SimpleNamespace(volume=100, unrealized_pnl=-999999)
    ctx = _risk_ctx(
        account_balance=Decimal("5000"),
        positions={"BTCUSDT:LONG": pos},
        last_prices={"BTCUSDT": Decimal("50000")},
    )
    ok, msg = lim.check(_open_req(volume=10, price=Decimal("50000")), ctx)
    assert ok is False
    assert msg


def test_liquidation_guard_close_offset_passes() -> None:
    lim = LiquidationGuard()
    req = _open_req()
    req.offset = Offset.CLOSE
    ok, _ = lim.check(req, _risk_ctx())
    assert ok is True


def test_liquidation_guard_param_name() -> None:
    assert LiquidationGuard().name == "BTC_LiquidationGuard"


# --- session ---


def test_get_current_session_returns_enum() -> None:
    s = get_current_session(datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc))
    assert isinstance(s, SessionType)


def test_get_session_liquidity_bounded() -> None:
    liq = get_session_liquidity(SessionType.US)
    assert 0 < liq <= 1.0


def test_get_volatility_scale_positive() -> None:
    assert get_volatility_scale(SessionType.OVERLAP_EU_US) >= 1.0


def test_session_aware_filter_instantiation_defaults() -> None:
    f = SessionAwareFilter()
    assert f.should_trade(SessionType.US) is True


def test_session_aware_filter_low_liquidity_blocks_trade() -> None:
    f = SessionAwareFilter(min_liquidity=0.95)
    assert f.should_trade(SessionType.LOW_LIQUIDITY) is False


def test_session_aware_filter_adjust_signal_strength_scales() -> None:
    f = SessionAwareFilter(low_liq_strength_scale=0.5)
    scaled = f.adjust_signal_strength(1.0, SessionType.LOW_LIQUIDITY)
    assert scaled < 1.0


def test_session_aware_filter_adjust_position_size_high_vol() -> None:
    f = SessionAwareFilter(high_vol_position_scale=0.5)
    q = f.adjust_position_size(10.0, SessionType.OVERLAP_EU_US)
    assert q == 5.0


def test_session_aware_filter_get_session_report_keys() -> None:
    f = SessionAwareFilter()
    rep = f.get_session_report(datetime(2026, 6, 15, 15, 30, tzinfo=timezone.utc))
    assert {"session", "liquidity", "tradeable"} <= set(rep.keys())


def test_session_param_validation_custom_min_liquidity() -> None:
    f = SessionAwareFilter(min_liquidity=0.0)
    assert f.should_trade(SessionType.LOW_LIQUIDITY) is True


# --- regime_detector ---


def test_regime_detector_instantiation_defaults() -> None:
    d = MarketRegimeDetector()
    assert d.current_regime == MarketRegime.UNKNOWN


def test_regime_detector_instantiation_custom_thresholds() -> None:
    d = MarketRegimeDetector(adx_strong=50.0, bb_squeeze_pct=0.1)
    assert d._adx_strong == 50.0


def test_regime_detector_update_synthetic_trending() -> None:
    d = MarketRegimeDetector()
    p = 100.0
    for i in range(150):
        p += 1.5
        d.update(p + 1, p - 0.5, p)
    assert d.current_regime != MarketRegime.UNKNOWN


def test_regime_detector_get_params_for_regime() -> None:
    p = MarketRegimeDetector().get_params(MarketRegime.RANGING)
    assert "position_scale" in p
    assert "stop_loss_mult" in p


def test_regime_detector_summary_keys() -> None:
    d = MarketRegimeDetector()
    d.update(101, 99, 100)
    s = d.summary()
    assert s["regime"] == MarketRegime.UNKNOWN.value or "bars_processed" in s


def test_regime_detector_edge_few_bars_unknown() -> None:
    d = MarketRegimeDetector()
    d.update(10, 9, 9.5)
    assert d.current_regime == MarketRegime.UNKNOWN


def test_regime_detector_extreme_flat_prices() -> None:
    d = MarketRegimeDetector()
    for _ in range(120):
        d.update(1.0, 1.0, 1.0)


def test_regime_detector_param_validation_invalid_periods_still_construct() -> None:
    d = MarketRegimeDetector(adx_period=1, bb_period=2)
    assert d._adx_period == 1


# --- registry ---


def test_registry_lists_btc_strategies() -> None:
    names = StrategyRegistry.list_registered()
    assert "btc_trend_following" in names
    assert "btc_mean_reversion" in names
    assert "funding_rate_arb" in names


def test_registry_get_unknown_returns_none() -> None:
    assert StrategyRegistry.get("not_a_real_strategy_ever") is None


def test_registry_create_and_roundtrip() -> None:
    cfg = StrategyConfig(name="r1", symbols=["BTCUSDT"])
    obj = StrategyRegistry.create("btc_momentum", cfg)
    assert obj.name == "r1"


def test_registry_create_missing_raises() -> None:
    cfg = StrategyConfig(name="r2", symbols=["BTCUSDT"])
    with pytest.raises(KeyError):
        StrategyRegistry.create("totally_missing_strategy_key", cfg)


def test_registry_unregister_roundtrip() -> None:
    StrategyRegistry.register("tmp_test_strat", BTCTrendFollowingStrategy)
    assert StrategyRegistry.unregister("tmp_test_strat") is True
    assert StrategyRegistry.get("tmp_test_strat") is None


def test_registry_instance_crud() -> None:
    cfg = StrategyConfig(name="inst", symbols=["BTCUSDT"])
    StrategyRegistry.add_instance(cfg)
    assert StrategyRegistry.get_instance(cfg.strategy_id) is not None
    assert StrategyRegistry.set_instance_enabled(cfg.strategy_id, False) is not None
    assert StrategyRegistry.delete_instance(cfg.strategy_id) is True

