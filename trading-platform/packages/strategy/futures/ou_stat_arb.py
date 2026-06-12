"""Ornstein-Uhlenbeck 统计套利策略 — 价格均值回归的数学建模。

理论：Ornstein-Uhlenbeck 过程 (Uhlenbeck & Ornstein 1930, 经典物理)
      dX_t = θ(μ - X_t)dt + σdW_t

应用于金融（经典量化金融方法）：
  - θ: 均值回归速度（越大回归越快，策略越有效）
  - μ: 长期均值水平
  - σ: 波动率
  - Half-life = ln(2)/θ — 预期回归一半所需时间

信号构建：
  1. 滚动估计 OU 参数 (θ, μ, σ)
  2. 计算当前价格偏离 μ 的标准化距离
  3. 偏离 > N σ → 反向入场（预期回归）
  4. 回到 μ 附近 → 平仓

适用场景：
  - 价差交易（两个相关品种的价差服从 OU）
  - 单品种短期均值回归（日内价格围绕 VWAP/SMA 波动）

Method: Ornstein-Uhlenbeck process (1930, 经典物理/随机过程)
        Maximum Likelihood 估计 OU 参数 (经典统计方法)
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
    "ou_window": 60,
    "entry_zscore": 2.0,
    "exit_zscore": 0.3,
    "max_half_life": 30,
    "min_half_life": 3,
    "atr_period": 14,
    "sl_atr_mult": 2.0,
    "max_hold_bars": 40,
    "cooldown_bars": 3,
}


def estimate_ou_params(prices: np.ndarray, dt: float = 1.0) -> dict[str, float]:
    """Estimate OU process parameters via discrete AR(1) regression.

    X_{t+1} - X_t = a + b * X_t + ε
    Then: θ = -b/dt, μ = -a/b, σ = std(ε) * sqrt(-2*b / (dt*(1-exp(2*b*dt))))
    """
    if len(prices) < 10:
        return {"theta": 0, "mu": 0, "sigma": 0, "half_life": float("inf")}

    dx = np.diff(prices)
    x = prices[:-1]

    if np.std(x) < 1e-10:
        return {"theta": 0, "mu": np.mean(prices), "sigma": 0, "half_life": float("inf")}

    b = np.cov(dx, x)[0, 1] / np.var(x)
    a = np.mean(dx) - b * np.mean(x)
    residuals = dx - (a + b * x)
    sigma_e = np.std(residuals)

    if b >= 0:
        return {"theta": 0, "mu": np.mean(prices), "sigma": sigma_e, "half_life": float("inf")}

    theta = -b / dt
    mu = -a / b
    half_life = np.log(2) / theta if theta > 0 else float("inf")

    try:
        sigma = sigma_e * np.sqrt(-2 * b / (dt * (1 - np.exp(2 * b * dt))))
    except Exception:
        sigma = sigma_e

    return {
        "theta": float(theta),
        "mu": float(mu),
        "sigma": float(sigma),
        "half_life": float(half_life),
        "r_squared": float(1 - np.var(residuals) / np.var(dx)) if np.var(dx) > 0 else 0,
    }


@auto_register("ou_stat_arb")
class OUStatArbStrategy(BaseStrategy):
    """Ornstein-Uhlenbeck 统计套利策略 — 数学化的均值回归。"""

    def __init__(self, config: StrategyConfig) -> None:
        config = config.model_copy(
            update={"params": {**DEFAULT_PARAMS, **config.params}}
        )
        super().__init__(config)
        self._closes: deque[float] = deque(maxlen=200)
        self._highs: deque[float] = deque(maxlen=200)
        self._lows: deque[float] = deque(maxlen=200)
        self._bar_count = 0
        self._position_side: str | None = None
        self._entry_price = 0.0
        self._hold_bars = 0
        self._cd = 0
        self._last_ou: dict[str, float] = {}

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        h = float(bar.get("high", 0))
        l = float(bar.get("low", 0))
        c = float(bar.get("close", 0))

        self._highs.append(h)
        self._lows.append(l)
        self._closes.append(c)
        self._bar_count += 1

        ou_window = self.get_param("ou_window", 60)
        if self._bar_count < ou_window + 5:
            return []

        atr = calc_atr(list(self._highs), list(self._lows), list(self._closes),
                       self.get_param("atr_period", 14))
        if atr is None or atr < 1e-10:
            return []

        close_arr = np.array(list(self._closes)[-ou_window:])
        self._last_ou = estimate_ou_params(close_arr)

        theta = self._last_ou["theta"]
        mu = self._last_ou["mu"]
        sigma = self._last_ou["sigma"]
        half_life = self._last_ou["half_life"]

        min_hl = self.get_param("min_half_life", 3)
        max_hl = self.get_param("max_half_life", 30)

        if theta <= 0 or half_life < min_hl or half_life > max_hl:
            if self._position_side:
                exit_type = SignalType.LONG_EXIT if self._position_side == "long" else SignalType.SHORT_EXIT
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=exit_type, strength=0.7, price=c,
                    reason=f"ou_regime_exit: theta={theta:.4f} hl={half_life:.1f}",
                )
                self.record_signal(sig)
                self._position_side = None
                self._hold_bars = 0
                return [sig]
            return []

        zscore = (c - mu) / max(sigma, 1e-10)

        signals = []

        if self._position_side:
            self._hold_bars += 1
            max_hold = self.get_param("max_hold_bars", 40)
            exit_z = self.get_param("exit_zscore", 0.3)
            sl_mult = self.get_param("sl_atr_mult", 2.0)

            pnl = (c - self._entry_price) / self._entry_price if self._position_side == "long" else (self._entry_price - c) / self._entry_price
            sl_hit = pnl <= -sl_mult * atr / self._entry_price

            reverted = (self._position_side == "long" and zscore > -exit_z) or \
                      (self._position_side == "short" and zscore < exit_z)

            if reverted or sl_hit or self._hold_bars >= max_hold:
                exit_type = SignalType.LONG_EXIT if self._position_side == "long" else SignalType.SHORT_EXIT
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=exit_type, strength=0.8, price=c,
                    reason=f"ou_exit: z={zscore:.2f} theta={theta:.4f} hl={half_life:.1f}",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = None
                self._hold_bars = 0
                self._cd = self.get_param("cooldown_bars", 3)
                return signals

        if self._cd > 0:
            self._cd -= 1

        entry_z = self.get_param("entry_zscore", 2.0)

        if not self._position_side and self._cd <= 0:
            if zscore < -entry_z:
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY,
                    strength=min(abs(zscore) / 3.0, 1.0), price=c,
                    reason=f"ou_buy: z={zscore:.2f} mu={mu:.1f} theta={theta:.4f} hl={half_life:.1f}",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = "long"
                self._entry_price = c
                self._hold_bars = 0

            elif zscore > entry_z:
                sig = Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY,
                    strength=min(abs(zscore) / 3.0, 1.0), price=c,
                    reason=f"ou_sell: z={zscore:.2f} mu={mu:.1f} theta={theta:.4f} hl={half_life:.1f}",
                )
                signals.append(sig)
                self.record_signal(sig)
                self._position_side = "short"
                self._entry_price = c
                self._hold_bars = 0

        return signals

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        return []
