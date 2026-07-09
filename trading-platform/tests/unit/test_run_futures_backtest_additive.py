"""回归：确保 scripts/run_futures_backtest.py 的 PnL 模型不会再爆 e+197。

Root bug: 原实现 capital *= (1 + pnl_pct) 每笔交易乘法复利，几百根 bar
就把 intraday_reversal 的 final_capital 放大到 e+197 级别。修复：固定名义仓位
+ clamp 单笔 pnl_pct 到 [-0.99, +10] + 加法累计。本测试跑 500 bar × 3 种行情，
断言 final_capital 在 5 个数量级内有界。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
for sub in ("packages", "scripts"):
    sys.path.insert(0, str(REPO_ROOT / sub))
for pkg in ("core", "backtest", "sim_live"):
    sys.path.insert(0, str(REPO_ROOT / "packages" / pkg))

import run_futures_backtest as rfb  # noqa: E402
from strategy.base import BaseStrategy, Signal, SignalType, StrategyConfig  # noqa: E402


class _AlternatingStrategy(BaseStrategy):
    """每 4 bar 开一次多头，2 bar 后平仓 — 500 bar 产 125 笔交易."""

    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self._tick = 0
        self._in_pos = False

    async def on_bar(self, symbol, bar):
        self._tick += 1
        out = []
        if not self._in_pos and self._tick % 4 == 0:
            out.append(Signal(
                symbol=symbol,
                signal_type=SignalType.LONG_ENTRY,
                strategy_id=self.config.strategy_id,
                strength=1.0,
            ))
            self._in_pos = True
        elif self._in_pos and self._tick % 4 == 2:
            out.append(Signal(
                symbol=symbol,
                signal_type=SignalType.LONG_EXIT,
                strategy_id=self.config.strategy_id,
                strength=1.0,
            ))
            self._in_pos = False
        return out

    async def generate_signals(self, market_data):
        return []


def _make_bars(n: int = 500, drift: float = 0.0, vol: float = 0.01, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, n)
    prices = 3000.0 * np.cumprod(1.0 + rets)
    return pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=n, freq="5min"),
        "open": prices,
        "high": prices * 1.002,
        "low": prices * 0.998,
        "close": prices,
        "volume": np.full(n, 1000.0),
        "instrument": ["rb2510"] * n,
    })


@pytest.mark.parametrize(
    "scenario,drift,vol",
    [
        ("bull", 0.001, 0.01),
        ("bear", -0.001, 0.01),
        ("chop", 0.0, 0.03),
        ("extreme-vol", 0.0, 0.2),
    ],
)
def test_backtest_additive_pnl_bounded(scenario, drift, vol):
    bars = _make_bars(n=500, drift=drift, vol=vol, seed=abs(hash(scenario)) & 0xFFFF)
    strategy = _AlternatingStrategy(StrategyConfig(name="alt", strategy_id=f"alt-{scenario}"))

    result = rfb.backtest_strategy_on_bars(strategy, bars, initial_capital=100_000.0)

    assert result["trades"] > 0, f"{scenario}: no trades"
    final = result["final_capital"]
    assert -1e7 < final < 1e7, f"{scenario}: final_capital={final} exceeds bounded range"
    assert abs(result["total_return"]) < 100, f"{scenario}: total_return={result['total_return']} unbounded"
    assert not np.isnan(result["sharpe"]) and not np.isinf(result["sharpe"]), f"{scenario}: sharpe NaN/Inf"


def test_backtest_pnl_clamp_extreme_gap():
    """模拟 entry→exit 价格直接翻 100 倍：旧实现 capital *= 100 爆炸，新实现 clamp 到 +10."""
    bars = _make_bars(n=4, drift=0.0, vol=0.0)
    bars.loc[bars.index[-1], ["open", "high", "low", "close"]] = 300_000.0

    class _JumpStrategy(_AlternatingStrategy):
        async def on_bar(self, symbol, bar):
            self._tick += 1
            out = []
            if self._tick == 1:
                out.append(Signal(symbol=symbol, signal_type=SignalType.LONG_ENTRY, strategy_id=self.config.strategy_id, strength=1.0))
            elif self._tick == 4:
                out.append(Signal(symbol=symbol, signal_type=SignalType.LONG_EXIT, strategy_id=self.config.strategy_id, strength=1.0))
            return out

    strategy = _JumpStrategy(StrategyConfig(name="jump", strategy_id="jump-test"))
    result = rfb.backtest_strategy_on_bars(strategy, bars, initial_capital=100_000.0)

    notional = 100_000.0 * rfb.POSITION_NOTIONAL_PCT
    max_single_trade_gain = notional * rfb.PER_TRADE_RETURN_CAP
    assert result["final_capital"] < 100_000.0 + max_single_trade_gain + 100, (
        f"clamp broke: final_capital={result['final_capital']} exceeds 100k + {max_single_trade_gain}"
    )
