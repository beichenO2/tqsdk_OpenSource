"""Tests for shared indicator utilities."""

from collections import deque

from strategy.indicators import calc_atr, ema_update, sma


def test_calc_atr_insufficient_data():
    assert calc_atr([10.0], [8.0], [9.0], 14) is None


def test_calc_atr_basic():
    highs = [12.0, 13.0, 14.0, 13.5, 15.0]
    lows = [10.0, 11.0, 12.0, 11.5, 13.0]
    closes = [11.0, 12.0, 13.0, 12.5, 14.0]
    result = calc_atr(highs, lows, closes, 3)
    assert result is not None
    assert result > 0


def test_calc_atr_with_deque():
    highs = deque([12.0, 13.0, 14.0, 13.5, 15.0], maxlen=10)
    lows = deque([10.0, 11.0, 12.0, 11.5, 13.0], maxlen=10)
    closes = deque([11.0, 12.0, 13.0, 12.5, 14.0], maxlen=10)
    result = calc_atr(highs, lows, closes, 3)
    assert result is not None
    assert result > 0


def test_calc_atr_known_value():
    highs = [10.0, 12.0, 11.0]
    lows = [8.0, 9.0, 9.5]
    closes = [9.0, 11.0, 10.0]
    result = calc_atr(highs, lows, closes, 2)
    assert result is not None
    # Bar index -2: TR = max(12-9, |12-9|, |9-9|) = 3.0
    # Bar index -1: TR = max(11-9.5, |11-11|, |9.5-11|) = 1.5
    assert abs(result - 2.25) < 1e-9


def test_ema_update_first_value():
    assert ema_update(None, 100.0, 10) == 100.0


def test_ema_update_convergence():
    value = None
    for _ in range(200):
        value = ema_update(value, 50.0, 10)
    assert abs(value - 50.0) < 1e-6


def test_ema_update_weighting():
    prev = 100.0
    result = ema_update(prev, 110.0, 9)
    k = 2.0 / (9 + 1)
    expected = 110.0 * k + 100.0 * (1 - k)
    assert abs(result - expected) < 1e-9


def test_sma_basic():
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert sma(data, 3) == 4.0
    assert sma(data, 5) == 3.0


def test_sma_insufficient_data():
    assert sma([1.0], 5) is None
    assert sma([], 1) is None
    assert sma([1.0], 0) is None


def test_sma_with_deque():
    data = deque([10.0, 20.0, 30.0], maxlen=5)
    assert sma(data, 3) == 20.0
