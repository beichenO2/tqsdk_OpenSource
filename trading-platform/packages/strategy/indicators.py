"""Shared technical indicators used across strategies.

All strategies should import indicators from here to avoid reimplementation.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Sequence

_Floats = list[float] | deque[float] | Sequence[float]


def calc_atr(
    highs: _Floats,
    lows: _Floats,
    closes: _Floats,
    period: int,
) -> float | None:
    """Simple average of True Range over *period* bars.

    Requires at least ``period + 1`` data points (TR needs a previous close).
    Returns ``None`` when there is insufficient data.
    """
    h = list(highs)
    l = list(lows)
    c = list(closes)
    if len(h) < period + 1:
        return None
    trs: list[float] = []
    for i in range(-period, 0):
        tr = max(
            h[i] - l[i],
            abs(h[i] - c[i - 1]),
            abs(l[i] - c[i - 1]),
        )
        trs.append(tr)
    return sum(trs) / len(trs)


def ema_update(prev: float | None, value: float, period: int) -> float:
    """Incremental EMA update. Returns *value* when *prev* is ``None``."""
    if prev is None:
        return value
    k = 2.0 / (period + 1)
    return value * k + prev * (1 - k)


def sma(data: _Floats, period: int) -> float | None:
    """Simple moving average over the last *period* values."""
    if period <= 0 or len(data) < period:
        return None
    return sum(list(data)[-period:]) / period


def rsi(closes: _Floats, period: int = 14) -> float | None:
    """RSI using Wilder's smoothed averaging.

    Returns None if fewer than ``period + 1`` data points.
    """
    c = list(closes)
    if len(c) < period + 1:
        return None
    gains = 0.0
    losses = 0.0
    for i in range(-period, 0):
        delta = c[i] - c[i - 1]
        if delta > 0:
            gains += delta
        else:
            losses -= delta
    avg_gain = gains / period
    avg_loss = losses / period
    if avg_loss < 1e-12:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def bollinger_bands(
    closes: _Floats,
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[float, float, float] | None:
    """Returns (lower, mid, upper) Bollinger Bands.

    Returns None when insufficient data.
    """
    c = list(closes)
    if len(c) < period:
        return None
    window = c[-period:]
    mid = sum(window) / period
    variance = sum((x - mid) ** 2 for x in window) / period
    std = math.sqrt(variance)
    return (mid - num_std * std, mid, mid + num_std * std)


def macd(
    closes: _Floats,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> tuple[float, float, float] | None:
    """MACD via EMA. Returns (macd_line, signal_line, histogram).

    Returns None when fewer than ``slow`` data points.
    """
    c = list(closes)
    if len(c) < slow:
        return None
    ema_fast: float | None = None
    ema_slow: float | None = None
    signal_ema: float | None = None
    macd_line = 0.0
    for price in c:
        ema_fast = ema_update(ema_fast, price, fast)
        ema_slow = ema_update(ema_slow, price, slow)
        macd_line = ema_fast - ema_slow
        signal_ema = ema_update(signal_ema, macd_line, signal_period)
    assert signal_ema is not None
    return (macd_line, signal_ema, macd_line - signal_ema)


def adx(
    highs: _Floats,
    lows: _Floats,
    closes: _Floats,
    period: int = 14,
) -> float | None:
    """Average Directional Index. Returns None when insufficient data.

    Requires at least ``period * 2 + 1`` bars for reasonable smoothing.
    """
    h = list(highs)
    l = list(lows)
    c = list(closes)
    n = len(h)
    if n < period + 1:
        return None

    plus_dm_sum = 0.0
    minus_dm_sum = 0.0
    tr_sum = 0.0

    for i in range(1, period + 1):
        up = h[i] - h[i - 1]
        down = l[i - 1] - l[i]
        plus_dm_sum += max(up, 0.0) if up > down else 0.0
        minus_dm_sum += max(down, 0.0) if down > up else 0.0
        tr_sum += max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))

    alpha = 1.0 / period
    smoothed_tr = tr_sum
    smoothed_plus = plus_dm_sum
    smoothed_minus = minus_dm_sum
    dx_sum = 0.0
    dx_count = 0

    for i in range(period + 1, n):
        up = h[i] - h[i - 1]
        down = l[i - 1] - l[i]
        pdm = max(up, 0.0) if up > down else 0.0
        mdm = max(down, 0.0) if down > up else 0.0
        tr = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))

        smoothed_tr = smoothed_tr - smoothed_tr * alpha + tr
        smoothed_plus = smoothed_plus - smoothed_plus * alpha + pdm
        smoothed_minus = smoothed_minus - smoothed_minus * alpha + mdm

        if smoothed_tr > 0:
            di_plus = smoothed_plus / smoothed_tr
            di_minus = smoothed_minus / smoothed_tr
            di_sum = di_plus + di_minus
            if di_sum > 0:
                dx = abs(di_plus - di_minus) / di_sum * 100.0
                dx_sum += dx
                dx_count += 1

    if dx_count == 0:
        return None
    return dx_sum / dx_count


def rolling_zscore(
    data: _Floats,
    window: int,
    clamp: float = 10.0,
) -> float | None:
    """Z-score of the last value vs a rolling window.

    Returns None when insufficient data. Clamps output to [-clamp, clamp].
    """
    d = list(data)
    if len(d) < window:
        return None
    w = d[-window:]
    mean = sum(w) / len(w)
    var = sum((x - mean) ** 2 for x in w) / len(w)
    std = math.sqrt(var) if var > 0 else 0.0
    if std < 1e-10:
        return None
    z = (d[-1] - mean) / std
    return max(-clamp, min(clamp, z))


def check_atr_exit(
    side: str,
    close: float,
    avg_price: float,
    atr: float,
    hold_bars: int,
    sl_mult: float = 2.5,
    tp_mult: float = 4.0,
    max_hold: int = 48,
) -> tuple[bool, str]:
    """Shared ATR-based exit logic.

    Returns (should_exit, reason). ``side`` is "buy" or "sell".
    """
    if hold_bars >= max_hold:
        return True, f"最大持仓({max_hold}bars)"
    if side == "buy":
        if close < avg_price - atr * sl_mult:
            return True, f"止损({sl_mult}xATR)"
        if close > avg_price + atr * tp_mult:
            return True, f"止盈({tp_mult}xATR)"
    elif side == "sell":
        if close > avg_price + atr * sl_mult:
            return True, f"止损({sl_mult}xATR)"
        if close < avg_price - atr * tp_mult:
            return True, f"止盈({tp_mult}xATR)"
    return False, ""
