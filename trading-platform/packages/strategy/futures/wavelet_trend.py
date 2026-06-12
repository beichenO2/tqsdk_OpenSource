"""小波去噪趋势策略 — 用离散小波变换提取趋势信号。

理论：
  - DWT (Discrete Wavelet Transform) 将价格序列分解为多尺度成分
  - 高频分量 = 噪声，低频分量 = 趋势
  - 重构去噪信号后检测趋势方向

实现方式（无 pywt 依赖，使用 Haar 小波的快速实现）：
  - Haar 小波是最简单的正交小波（经典信号处理，Haar 1910）
  - 多级分解：level-1 细节 = 最高频噪声
  - 软阈值去噪（Donoho & Johnstone 1994 — 经典统计信号处理）
  - 重构后趋势方向 + 斜率 → 交易信号

Method:
  - Haar wavelet: Haar 1910 (经典)
  - Wavelet denoising: Donoho & Johnstone 1994 (经典统计方法)
  - 应用于金融信号处理是标准做法
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np

from ..base import BaseStrategy, Signal, SignalType, StrategyConfig
from ..indicators import calc_atr
from ..registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS: dict[str, Any] = {
    "wavelet_levels": 3,
    "denoise_threshold_mult": 1.0,
    "trend_slope_threshold": 0.0003,
    "signal_window": 30,
    "atr_period": 14,
    "tp_atr_mult": 2.5,
    "sl_atr_mult": 1.2,
    "max_hold_bars": 50,
    "cooldown_bars": 3,
}


def _haar_decompose(signal: np.ndarray, levels: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """Multi-level Haar wavelet decomposition.

    Returns (approximation_coeffs, [detail_coeffs_per_level])
    """
    approx = signal.copy()
    details = []
    for _ in range(levels):
        n = len(approx)
        if n < 2:
            break
        half = n // 2
        a = np.zeros(half)
        d = np.zeros(half)
        for i in range(half):
            a[i] = (approx[2 * i] + approx[2 * i + 1]) / np.sqrt(2)
            d[i] = (approx[2 * i] - approx[2 * i + 1]) / np.sqrt(2)
        details.append(d)
        approx = a
    return approx, details


def _soft_threshold(coeffs: np.ndarray, threshold: float) -> np.ndarray:
    """Donoho-Johnstone soft thresholding."""
    return np.sign(coeffs) * np.maximum(np.abs(coeffs) - threshold, 0)


def _haar_reconstruct(approx: np.ndarray, details: list[np.ndarray]) -> np.ndarray:
    """Reconstruct signal from Haar wavelet coefficients."""
    signal = approx.copy()
    for d in reversed(details):
        n = len(d)
        recon = np.zeros(2 * n)
        for i in range(n):
            recon[2 * i] = (signal[i] + d[i]) / np.sqrt(2)
            recon[2 * i + 1] = (signal[i] - d[i]) / np.sqrt(2)
        signal = recon
    return signal


def wavelet_denoise(prices: np.ndarray, levels: int = 3, threshold_mult: float = 1.0) -> np.ndarray:
    """Denoise a price series using Haar DWT + soft thresholding."""
    n_orig = len(prices)
    n_padded = 2 ** int(np.ceil(np.log2(max(n_orig, 4))))
    padded = np.pad(prices, (0, n_padded - n_orig), mode="edge")

    approx, details = _haar_decompose(padded, levels)

    denoised_details = []
    for d in details:
        if len(d) > 0:
            sigma = np.median(np.abs(d)) / 0.6745  # MAD estimator
            threshold = threshold_mult * sigma * np.sqrt(2 * np.log(len(d)))
            denoised_details.append(_soft_threshold(d, threshold))
        else:
            denoised_details.append(d)

    reconstructed = _haar_reconstruct(approx, denoised_details)
    return reconstructed[:n_orig]


@auto_register("wavelet_trend")
class WaveletTrendStrategy(BaseStrategy):
    """小波去噪趋势策略 — 从噪声中提取纯趋势信号。"""

    def __init__(self, config: StrategyConfig) -> None:
        config = config.model_copy(
            update={"params": {**DEFAULT_PARAMS, **config.params}}
        )
        super().__init__(config)
        self._closes: deque[float] = deque(maxlen=300)
        self._highs: deque[float] = deque(maxlen=300)
        self._lows: deque[float] = deque(maxlen=300)
        self._bar_count = 0
        self._position_side: str | None = None
        self._entry_price = 0.0
        self._hold_bars = 0
        self._cd = 0

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        h = float(bar.get("high", 0))
        l = float(bar.get("low", 0))
        c = float(bar.get("close", 0))

        self._highs.append(h)
        self._lows.append(l)
        self._closes.append(c)
        self._bar_count += 1

        window = self.get_param("signal_window", 30)
        if self._bar_count < max(window + 5, 35):
            return []

        atr = calc_atr(list(self._highs), list(self._lows), list(self._closes),
                       self.get_param("atr_period", 14))
        if atr is None or atr < 1e-10:
            return []

        close_arr = np.array(list(self._closes)[-window:])
        denoised = wavelet_denoise(
            close_arr,
            levels=self.get_param("wavelet_levels", 3),
            threshold_mult=self.get_param("denoise_threshold_mult", 1.0),
        )

        if len(denoised) < 5:
            return []

        slope = (denoised[-1] - denoised[-5]) / (5 * max(abs(denoised[-5]), 1e-10))
        slope_threshold = self.get_param("trend_slope_threshold", 0.0003)

        signals = []

        if self._position_side:
            self._hold_bars += 1
            max_hold = self.get_param("max_hold_bars", 50)
            tp_mult = self.get_param("tp_atr_mult", 2.5)
            sl_mult = self.get_param("sl_atr_mult", 1.2)

            pnl = (c - self._entry_price) / self._entry_price if self._position_side == "long" else (self._entry_price - c) / self._entry_price
            tp_hit = pnl >= tp_mult * atr / self._entry_price
            sl_hit = pnl <= -sl_mult * atr / self._entry_price

            trend_reversed = (self._position_side == "long" and slope < -slope_threshold) or \
                            (self._position_side == "short" and slope > slope_threshold)

            if sl_hit or tp_hit or self._hold_bars >= max_hold or trend_reversed:
                exit_type = SignalType.LONG_EXIT if self._position_side == "long" else SignalType.SHORT_EXIT
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=exit_type, strength=0.8, price=c,
                    reason=f"wavelet_exit: slope={slope:.6f} hold={self._hold_bars}",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = None
                self._hold_bars = 0
                self._cd = self.get_param("cooldown_bars", 3)
                return signals

        if self._cd > 0:
            self._cd -= 1

        if not self._position_side and self._cd <= 0:
            if slope > slope_threshold:
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY,
                    strength=min(abs(slope) * 1000, 1.0), price=c,
                    reason=f"wavelet_buy: slope={slope:.6f} denoised_trend=up",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = "long"
                self._entry_price = c
                self._hold_bars = 0

            elif slope < -slope_threshold:
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY,
                    strength=min(abs(slope) * 1000, 1.0), price=c,
                    reason=f"wavelet_sell: slope={slope:.6f} denoised_trend=down",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = "short"
                self._entry_price = c
                self._hold_bars = 0

        return signals

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        return []
