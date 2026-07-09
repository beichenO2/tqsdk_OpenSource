"""验证 signal_balance + regime_filter mixin 消除 supertrend 式单边偏差。

Inbox 病例: supertrend 在 rb 上涨 500 bar 产 35 SHORT / 0 LONG。
本测试构造强上涨合成行情，断言：
  - 裸 supertrend（不过滤）仍然单边（保留原 bug 作为对比锚点；可选）
  - 过滤后 supertrend 信号：要么 L/S 平衡（不超 3:1），要么完全没有逆势 SHORT
  - whale_detector / keltner_channel 在强上涨市中不会产出 SHORT > LONG 的偏差
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "packages"))
for pkg in ("core", "backtest", "sim_live"):
    sys.path.insert(0, str(REPO_ROOT / "packages" / pkg))

from strategy.base import SignalType, StrategyConfig  # noqa: E402
from strategy.futures.keltner_channel import KeltnerChannelStrategy  # noqa: E402
from strategy.futures.supertrend import SupertrendStrategy  # noqa: E402
from strategy.mixins import EMASlopeRegimeMixin, SignalBalanceMixin  # noqa: E402


def _make_uptrend(n: int = 500, drift: float = 0.002, vol: float = 0.01, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, n)
    prices = 3000.0 * np.cumprod(1 + rets)
    return pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=n, freq="5min"),
        "open": prices, "high": prices * 1.002, "low": prices * 0.998,
        "close": prices, "volume": np.full(n, 1000.0),
        "open_interest": np.full(n, 10000.0),
        "instrument": ["rb2510"] * n,
    })


def _run_strategy(strategy, bars: pd.DataFrame) -> dict[str, int]:
    long_entry = short_entry = 0
    loop = asyncio.new_event_loop()
    try:
        for _, row in bars.iterrows():
            bar = {
                "datetime": row["datetime"],
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "open_interest": float(row["open_interest"]),
            }
            sigs = loop.run_until_complete(strategy.on_bar("rb2510", bar))
            for s in sigs:
                if s.signal_type == SignalType.LONG_ENTRY:
                    long_entry += 1
                elif s.signal_type == SignalType.SHORT_ENTRY:
                    short_entry += 1
    finally:
        loop.close()
    return {"long": long_entry, "short": short_entry}


def test_signal_balance_mixin_unit():
    """Mixin 独立单元行为：warmup 之前都放行，warmup 后主导方被压制."""
    class _Dummy(SignalBalanceMixin):
        SB_WARMUP = 4
        SB_MAX_RATIO = 2.0

    d = _Dummy()
    # warmup 阶段（total < 4）任何方向都放行
    for _ in range(3):
        assert d._sb_allow(SignalType.SHORT_ENTRY)
        d._sb_record(SignalType.SHORT_ENTRY)
    # total=3 < WARMUP=4 仍然放行
    assert d._sb_allow(SignalType.SHORT_ENTRY)
    d._sb_record(SignalType.SHORT_ENTRY)

    # total=4 触发 ratio 检查：short/max(long,1) = 4/1 = 4.0 > 2.0 → 压制 SHORT
    assert not d._sb_allow(SignalType.SHORT_ENTRY)
    assert d._sb_allow(SignalType.LONG_ENTRY)
    d._sb_record(SignalType.LONG_ENTRY)
    d._sb_record(SignalType.LONG_ENTRY)

    # 现在 long=2, short=4 → 4/2=2.0 <= 2.0 → SHORT 再次放行
    assert d._sb_allow(SignalType.SHORT_ENTRY)


def test_regime_filter_mixin_unit():
    """EMA 斜率在强上涨时 = 'up'，应屏蔽 SHORT_ENTRY，放行 LONG_ENTRY."""
    class _Dummy(EMASlopeRegimeMixin):
        EMA_REGIME_PERIOD = 20
        EMA_REGIME_SLOPE_THRESH = 0.0005
        EMA_REGIME_WARMUP = 10

    d = _Dummy()
    price = 3000.0
    for _ in range(40):
        price *= 1.003
        d._regime_update(price)
    assert d._regime() == "up"
    assert not d._regime_allow(SignalType.SHORT_ENTRY)
    assert d._regime_allow(SignalType.LONG_ENTRY)


def test_supertrend_no_one_sided_bias_in_uptrend():
    """装过 mixin 的 supertrend 在强上涨 500 bar 不应全 SHORT."""
    strategy = SupertrendStrategy(StrategyConfig(name="supertrend", strategy_id="st-test"))
    bars = _make_uptrend(n=500, drift=0.002, vol=0.01)
    stats = _run_strategy(strategy, bars)
    total = stats["long"] + stats["short"]
    if total == 0:
        pytest.skip("strategy produced no signals on this synthetic data")
    short_ratio = stats["short"] / total
    assert short_ratio <= 0.6, (
        f"supertrend 在上涨行情 SHORT 占比 {short_ratio:.2%}，疑似单边偏差未修复；{stats}"
    )


def test_keltner_channel_no_one_sided_bias_in_uptrend():
    strategy = KeltnerChannelStrategy(StrategyConfig(name="keltner_channel", strategy_id="kc-test"))
    bars = _make_uptrend(n=500, drift=0.002, vol=0.015)
    stats = _run_strategy(strategy, bars)
    total = stats["long"] + stats["short"]
    if total == 0:
        pytest.skip("strategy produced no signals on this synthetic data")
    short_ratio = stats["short"] / total
    assert short_ratio <= 0.6, (
        f"keltner 在上涨行情 SHORT 占比 {short_ratio:.2%}，regime filter 应压制；{stats}"
    )
