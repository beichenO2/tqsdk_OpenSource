"""Unit tests for AlphaGate acceptance framework (TDD)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor.alpha_gate import AlphaGate


def _make_ohlcv(close: np.ndarray, freq: str = "4h") -> pd.DataFrame:
    n = len(close)
    rng = np.random.default_rng(0)
    dt = pd.date_range("2020-01-01", periods=n, freq=freq, tz="UTC")
    spread = rng.uniform(0.001, 0.005, n) * close
    return pd.DataFrame(
        {
            "open_time": dt,
            "open": close * (1 + rng.normal(0, 0.0002, n)),
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": rng.integers(100, 5000, n).astype(float),
        }
    )


def _total_return_no_shift(
    position: pd.Series, close: pd.Series, one_way_cost_bps: float = 5.0
) -> float:
    """Same-bar execution (look-ahead) — must outperform causal shift(1) engine."""
    one_way = one_way_cost_bps / 10_000.0
    price_ret = close.pct_change().fillna(0.0)
    ret_gross = position * price_ret
    ret_cost = position.diff().abs().fillna(0.0) * one_way
    return float((1.0 + ret_gross - ret_cost).prod() - 1.0)


def _make_alpha_synthetic(n: int = 3000, seed: int = 42) -> tuple[pd.Series, pd.DataFrame]:
    """Construct causal signal + price drift so strategy has durable alpha."""
    rng = np.random.default_rng(seed)
    signal = np.ones(n)
    for i in range(1, n):
        if rng.random() < 0.004:
            signal[i] = -signal[i - 1]
        else:
            signal[i] = signal[i - 1]
    drift = 0.0015
    noise = rng.normal(0, 0.0003, n)
    lagged = np.roll(signal, 1)
    lagged[0] = 0.0
    ret = drift * lagged + noise
    close = 100.0 * np.cumprod(1.0 + ret)
    ohlcv = _make_ohlcv(close)
    position = pd.Series(signal, index=ohlcv.index, dtype=float)
    return position, ohlcv


def _make_random_synthetic(n: int = 3000, seed: int = 7) -> tuple[pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    ret = rng.normal(0, 0.01, n)
    close = 100.0 * np.cumprod(1.0 + ret)
    ohlcv = _make_ohlcv(close)
    position = pd.Series(rng.uniform(-1, 1, n), index=ohlcv.index)
    position = position.rolling(5, min_periods=1).mean().clip(-1, 1)
    return position, ohlcv


def _make_high_turnover_synthetic(n: int = 2000, seed: int = 3) -> tuple[pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    ret = rng.normal(0, 0.005, n)
    close = 100.0 * np.cumprod(1.0 + ret)
    ohlcv = _make_ohlcv(close)
    position = pd.Series(np.where(np.arange(n) % 2 == 0, 1.0, -1.0), index=ohlcv.index)
    return position, ohlcv


def _make_timing_vs_trend_synthetic(
    n: int = 4000, seed: int = 37
) -> tuple[pd.Series, pd.DataFrame]:
    """Low-return high-Sharpe timing signal on a high-vol bull trend price."""
    rng = np.random.default_rng(seed)
    cycle = 500
    exposure = 0.2
    spike = 0.08
    price_ret = np.zeros(n)
    position = np.zeros(n)
    for i in range(1, n):
        phase = i % cycle
        if phase == 0:
            price_ret[i] = spike
            position[i] = 0.0
        elif phase < 350:
            price_ret[i] = 0.0009 + rng.normal(0, 0.0015)
            position[i] = exposure
        elif phase < 400:
            price_ret[i] = rng.normal(-0.004, 0.012)
            position[i] = 0.0
        else:
            price_ret[i] = 0.0005 + rng.normal(0, 0.0012)
            position[i] = exposure
    close = 100.0 * np.cumprod(1.0 + price_ret)
    ohlcv = _make_ohlcv(close)
    return pd.Series(position, index=ohlcv.index, dtype=float), ohlcv


def _legacy_g5_total_return_pass(strat_r: float, bh_r: float, st_r: float) -> bool:
    """Old G5 rule: strategy total net return beats both benchmarks."""
    return strat_r > bh_r and strat_r > st_r


class TestAlphaGateSynthetic:
    def test_known_alpha_passes_all_gates(self):
        position, ohlcv = _make_alpha_synthetic()
        gate = AlphaGate(one_way_cost_bps=5.0, bars_per_year=2190)
        report = gate.evaluate(position, ohlcv)
        assert report.verdict == "PASS"
        assert sum(g.passed for g in report.gates.values()) == 5

    def test_random_signal_not_pass(self):
        position, ohlcv = _make_random_synthetic()
        gate = AlphaGate(one_way_cost_bps=5.0, bars_per_year=2190)
        report = gate.evaluate(position, ohlcv)
        assert report.verdict != "PASS"
        g3 = report.gates["G3_trade_expectancy"]
        g5 = report.gates["G5_benchmark"]
        assert not g3.passed or not g5.passed

    def test_high_turnover_fails_g4(self):
        position, ohlcv = _make_high_turnover_synthetic()
        gate = AlphaGate(one_way_cost_bps=5.0, bars_per_year=2190)
        report = gate.evaluate(position, ohlcv)
        assert not report.gates["G4_turnover_cost"].passed

    def test_leakage_shift_beats_causal(self):
        rng = np.random.default_rng(99)
        n = 2000
        ret = rng.normal(0, 0.008, n)
        close = 100.0 * np.cumprod(1.0 + ret)
        ohlcv = _make_ohlcv(close)
        future_sign = pd.Series(np.sign(ret), index=ohlcv.index)
        gate = AlphaGate(one_way_cost_bps=5.0, bars_per_year=2190)
        causal = gate.evaluate(future_sign, ohlcv)
        leaked = _total_return_no_shift(future_sign, ohlcv["close"])
        assert leaked > causal.metrics["total_net_return"] + 0.5

    def test_cost_monotonicity(self):
        position, ohlcv = _make_alpha_synthetic()
        gate = AlphaGate(one_way_cost_bps=5.0, bars_per_year=2190)
        report = gate.evaluate(position, ohlcv)
        s = report.metrics
        assert s["net_return_2bp"] >= s["net_return_5bp"] >= s["net_return_10bp"]

    def test_gate_report_markdown(self):
        position, ohlcv = _make_alpha_synthetic(n=500)
        gate = AlphaGate(one_way_cost_bps=5.0, bars_per_year=2190)
        md = gate.evaluate(position, ohlcv).to_markdown()
        assert "AlphaGate" in md
        assert "G1" in md
        assert "Verdict" in md
        assert "G5 benchmark details" in md
        assert "G5a" in md
        assert "G5b" in md

    def test_g5_risk_adjusted_passes_timing_beats_legacy_total_return(self):
        """Low-return timing signal should pass new G5 but fail old total-return rule."""
        position, ohlcv = _make_timing_vs_trend_synthetic()
        gate = AlphaGate(one_way_cost_bps=5.0, bars_per_year=2190)
        report = gate.evaluate(position, ohlcv)
        g5 = report.gates["G5_benchmark"]
        d = g5.details
        assert g5.passed
        assert d["g5a_passed"]
        assert d["g5b_passed"]
        assert d["strategy_return"] < d["buy_hold_return"]
        assert d["strategy_sharpe"] > d["buy_hold_sharpe"]
        assert not _legacy_g5_total_return_pass(
            d["strategy_return"], d["buy_hold_return"], d["supertrend_return"]
        )


class TestAlphaGateValidation:
    def test_position_out_of_range_raises(self):
        position, ohlcv = _make_alpha_synthetic(n=100)
        position.iloc[10] = 1.5
        gate = AlphaGate()
        with pytest.raises(ValueError, match="position"):
            gate.evaluate(position, ohlcv)

    def test_mismatched_length_raises(self):
        position, ohlcv = _make_alpha_synthetic(n=100)
        gate = AlphaGate()
        with pytest.raises(ValueError, match="length"):
            gate.evaluate(position.iloc[:50], ohlcv)
