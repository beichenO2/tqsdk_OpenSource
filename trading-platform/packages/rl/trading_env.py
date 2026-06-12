"""期货交易环境 — Gymnasium 兼容的 RL 训练环境。"""

from __future__ import annotations

import logging
from typing import Any, SupportsFloat

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .base import BaseTradingEnv

logger = logging.getLogger(__name__)


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average (in-place safe)."""
    alpha = 2.0 / (period + 1)
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def compute_ta_features(bars: np.ndarray, t: int, window: int) -> np.ndarray:
    """Compute 10 technical indicator features from OHLCV bars at index *t*.

    Returns a float32 array of shape (10,) with values clipped to [-1, 1].
    Features: RSI, MACD_hist, ATR_norm, BB_width, OBV_norm,
              ADX_norm, regime_trend, regime_range, regime_vol, regime_breakout.
    """
    lookback = max(window, 30)
    start = max(0, t - lookback + 1)
    seg = bars[start : t + 1]
    n = seg.shape[0]
    close = seg[:, 3].astype(np.float64)
    high = seg[:, 1].astype(np.float64)
    low = seg[:, 2].astype(np.float64)
    volume = seg[:, 4].astype(np.float64)

    feats = np.zeros(10, dtype=np.float32)
    if n < 3:
        return feats

    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    period = min(14, n - 1)
    avg_gain = np.mean(gains[-period:]) + 1e-12
    avg_loss = np.mean(losses[-period:]) + 1e-12
    rs = avg_gain / avg_loss
    rsi = 100.0 - 100.0 / (1.0 + rs)
    feats[0] = np.clip((rsi - 50.0) / 50.0, -1.0, 1.0)

    if n >= 26:
        ema12 = _ema(close, 12)
        ema26 = _ema(close, 26)
        macd_line = ema12 - ema26
        signal = _ema(macd_line, 9)
        hist = macd_line[-1] - signal[-1]
        price_scale = max(abs(close[-1]), 1e-8)
        feats[1] = np.clip(hist / price_scale * 100.0, -1.0, 1.0)

    tr = np.maximum(high[1:] - low[1:],
                    np.maximum(np.abs(high[1:] - close[:-1]),
                               np.abs(low[1:] - close[:-1])))
    atr_period = min(14, len(tr))
    atr = np.mean(tr[-atr_period:])
    feats[2] = np.clip(atr / max(close[-1], 1e-8) * 10.0, -1.0, 1.0)

    bb_period = min(20, n)
    bb_mean = np.mean(close[-bb_period:])
    bb_std = np.std(close[-bb_period:]) + 1e-12
    bb_width = 4.0 * bb_std / max(bb_mean, 1e-8)
    feats[3] = np.clip(bb_width - 0.1, -1.0, 1.0)

    obv = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i - 1]:
            obv[i] = obv[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            obv[i] = obv[i - 1] - volume[i]
        else:
            obv[i] = obv[i - 1]
    if n >= 10:
        obv_sma = np.mean(obv[-10:])
        obv_std = np.std(obv[-10:]) + 1e-12
        feats[4] = np.clip((obv[-1] - obv_sma) / obv_std / 3.0, -1.0, 1.0)

    if n >= 14:
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]),
                           np.maximum(high[1:] - high[:-1], 0.0), 0.0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]),
                            np.maximum(low[:-1] - low[1:], 0.0), 0.0)
        atr_14 = np.mean(tr[-14:]) + 1e-12
        di_plus = np.mean(dm_plus[-14:]) / atr_14 * 100.0
        di_minus = np.mean(dm_minus[-14:]) / atr_14 * 100.0
        di_sum = di_plus + di_minus + 1e-12
        dx = abs(di_plus - di_minus) / di_sum * 100.0
        adx = dx
        feats[5] = np.clip(adx / 50.0 - 0.5, -1.0, 1.0)

        is_trending = adx > 25.0
        is_ranging = adx < 20.0
        is_high_vol = bb_width > 0.08
        is_breakout = is_trending and is_high_vol

        feats[6] = 1.0 if is_trending and not is_breakout else 0.0
        feats[7] = 1.0 if is_ranging else 0.0
        feats[8] = 1.0 if is_high_vol and not is_breakout else 0.0
        feats[9] = 1.0 if is_breakout else 0.0

    return feats

# 动作语义：Hold / 做多 / 做空 / 平仓
ACTION_HOLD = 0
ACTION_LONG = 1
ACTION_SHORT = 2
ACTION_CLOSE = 3


class FuturesTradingEnv(gym.Env, BaseTradingEnv):
    """期货连续合约简化仿真环境。

    观测：滑动窗口 OHLCV（滚动窗口归一化到 [-1,1]）+ 仓位 one-hot + 权益比 + 持仓时长 + 浮盈。
    奖励：MTM Delta（密集信号）+ Trend Bonus（顺势持仓）- 手续费 - 回撤惩罚。

    Parameters
    ----------
    bars
        形状 ``(N, 5)`` 的 OHLCV 数组，列顺序为开高低收量。
    config
        可选配置键：``window_size`` (默认 30)、``initial_balance`` (默认 100_000)、
        ``commission_rate`` (默认 3e-4)、``slippage`` (默认 1e-4)、
        ``trend_bonus_scale`` (默认 0.5)、``dd_penalty_scale`` (默认 2.0)、
        ``dd_threshold`` (默认 0.05)、``bankruptcy_fraction`` (默认 0.01)。
    """

    metadata = {"render_modes": []}

    def __init__(self, bars: np.ndarray, config: dict[str, Any] | None = None) -> None:
        BaseTradingEnv.__init__(self, config)
        gym.Env.__init__(self)

        if bars.ndim != 2 or bars.shape[1] != 5:
            raise ValueError(f"bars 必须为形状 (N, 5) 的 OHLCV，当前为 {bars.shape}")

        self._bars = np.asarray(bars, dtype=np.float64)
        n = self._bars.shape[0]

        self._window = int(self.config.get("window_size", 30))
        self._initial_balance = float(self.config.get("initial_balance", 100_000.0))
        self._commission_rate = float(self.config.get("commission_rate", 0.0003))
        self._slippage = float(self.config.get("slippage", 0.0001))
        self._trend_bonus_scale = float(self.config.get("trend_bonus_scale", 0.5))
        self._dd_penalty_scale = float(self.config.get("dd_penalty_scale", 2.0))
        self._dd_threshold = float(self.config.get("dd_threshold", 0.05))
        self._bankruptcy_fraction = float(self.config.get("bankruptcy_fraction", 0.01))

        if self._window < 2:
            raise ValueError("window_size 至少为 2")
        if n < self._window + 2:
            raise ValueError(f"K 线数量 {n} 不足以覆盖 window_size={self._window} 与至少一步转移")

        self._use_ta = bool(self.config.get("use_ta_features", True))
        ta_dim = 10 if self._use_ta else 0
        obs_dim = self._window * 5 + 3 + 1 + 2 + ta_dim
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(4)

        self._t: int = 0
        self._position: int = 0  # -1 空, 0 无仓, 1 多
        self._entry_price: float = 0.0
        self._equity: float = self._initial_balance
        self._opened_this_step: bool = False

        self._last_realized_pnl: float = 0.0
        self._last_commission_paid: float = 0.0

        self._prev_equity: float = self._initial_balance
        self._peak_equity: float = self._initial_balance
        self._hold_steps: int = 0

        self._dsr_eta = float(self.config.get("dsr_eta", 0.01))
        self._dsr_A: float = 0.0
        self._dsr_B: float = 0.0
        self._dsr_weight = float(self.config.get("dsr_weight", 1.0))
        self._reward_mode: str = self.config.get("reward_mode", "dsr")

        self.np_random: Any = np.random.default_rng()

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self._t = self._window - 1
        self._position = 0
        self._entry_price = 0.0
        self._equity = self._initial_balance
        self._opened_this_step = False
        self._last_realized_pnl = 0.0
        self._last_commission_paid = 0.0
        self._prev_equity = self._initial_balance
        self._peak_equity = self._initial_balance
        self._hold_steps = 0
        self._dsr_A = 0.0
        self._dsr_B = 0.0

        obs = self._get_observation()
        info: dict[str, Any] = {
            "equity": self._equity,
            "step_index": self._t,
            "position": self._position,
        }
        return obs, info

    def step(
        self, action: SupportsFloat | np.ndarray | int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        a = int(np.asarray(action).item())
        if a not in (0, 1, 2, 3):
            raise ValueError(f"非法动作: {a}，期望 0–3")

        close_px = float(self._bars[self._t, 3])

        self._last_realized_pnl = 0.0
        self._last_commission_paid = 0.0
        self._prev_equity = self._equity
        self._opened_this_step = False
        if self._position != 0:
            self._hold_steps += 1

        # 执行动作（在当前 bar 收盘价附近成交）
        self._apply_action(a, close_px)

        # 推进时间；在下一根 K 上用新收盘价盯市（简化：先推进再盯市）
        self._t += 1
        terminated = False
        truncated = False

        if self._t >= self._bars.shape[0]:
            # 数据结束：强制平仓以实现最终盈亏
            if self._position != 0:
                final_close = float(self._bars[self._bars.shape[0] - 1, 3])
                self._close_position(final_close, reason="eod")
            terminated = True
        else:
            new_close = float(self._bars[self._t, 3])
            mtm_ref = (
                self._entry_price
                if self._opened_this_step and self._position != 0
                else close_px
            )
            self._mark_to_market(new_close, mtm_ref)

            min_equity = self._initial_balance * self._bankruptcy_fraction
            if self._equity <= min_equity:
                terminated = True

        obs = self._get_observation() if not terminated else self._get_observation_terminal()
        reward = float(self._calculate_reward(a))

        info: dict[str, Any] = {
            "equity": self._equity,
            "step_index": self._t,
            "position": self._position,
            "realized_pnl_step": self._last_realized_pnl,
            "commission_step": self._last_commission_paid,
        }

        return obs, reward, terminated, truncated, info

    def _get_observation_terminal(self) -> np.ndarray:
        """终止步仍返回合法向量（使用最后一窗）。"""
        self._t = min(self._t, self._bars.shape[0] - 1)
        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        start = self._t - self._window + 1
        window = self._bars[start : self._t + 1, :].astype(np.float64)

        w_min = window.min(axis=0)
        w_max = window.max(axis=0)
        w_span = np.maximum(w_max - w_min, 1e-8)
        norm = 2.0 * (window - w_min) / w_span - 1.0
        norm = np.clip(norm, -1.0, 1.0).astype(np.float32).ravel()

        pos_enc = np.zeros(3, dtype=np.float32)
        if self._position < 0:
            pos_enc[0] = 1.0
        elif self._position == 0:
            pos_enc[1] = 1.0
        else:
            pos_enc[2] = 1.0

        ratio = self._equity / max(self._initial_balance, 1e-8)
        balance_feat = np.clip(2.0 * ratio - 1.0, -1.0, 1.0).astype(np.float32)
        balance_arr = np.array([balance_feat], dtype=np.float32)

        hold_duration = np.array(
            [np.clip(self._hold_steps / 100.0, 0.0, 1.0)], dtype=np.float32
        )
        unrealized = np.array([self._unrealized_pnl_pct()], dtype=np.float32)

        parts = [norm, pos_enc, balance_arr, hold_duration, unrealized]
        if self._use_ta:
            ta = compute_ta_features(self._bars, self._t, self._window)
            parts.append(ta)

        return np.concatenate(parts, axis=0)

    def _calculate_reward(self, action: Any) -> float:
        """Reward function with selectable modes.

        Modes (``reward_mode`` in config):
        - ``"dsr"``: Differential Sharpe Ratio (Moody & Saffell, modernised).
          Provides a risk-adjusted dense signal via exponential-moving
          estimates of return mean/variance.
        - ``"mtm"``: Legacy MTM Delta + Trend Bonus - Drawdown Penalty.
        """
        del action
        scale = max(self._initial_balance, 1e-8)
        equity_delta = (self._equity - self._prev_equity) / scale

        cost_term = self._last_commission_paid / scale

        self._peak_equity = max(self._peak_equity, self._equity)
        dd = 0.0
        if self._peak_equity > 1e-12:
            dd_frac = (self._peak_equity - self._equity) / self._peak_equity
            dd = max(0.0, dd_frac - self._dd_threshold)
        dd_penalty = self._dd_penalty_scale * dd

        if self._reward_mode == "dsr":
            reward = self._dsr_reward(equity_delta) - cost_term - dd_penalty
        else:
            trend_bonus = 0.0
            if self._t >= 1 and self._position != 0:
                prev_close = float(self._bars[self._t - 1, 3])
                curr_close = float(self._bars[min(self._t, self._bars.shape[0] - 1), 3])
                if prev_close > 1e-12:
                    price_ret = (curr_close - prev_close) / prev_close
                    trend_bonus = self._trend_bonus_scale * self._position * price_ret
            reward = equity_delta + trend_bonus - cost_term - dd_penalty

        return float(np.clip(reward, -1.0, 1.0))

    def _dsr_reward(self, ret: float) -> float:
        """Differential Sharpe Ratio — EMA-based incremental Sharpe update.

        At each step the running first and second moments (A, B) of returns
        are updated with learning rate ``eta``.  The DSR is the marginal
        change in the Sharpe ratio attributable to the latest return, giving
        a dense, risk-aware reward signal.
        """
        eta = self._dsr_eta
        delta_A = ret - self._dsr_A
        delta_B = ret * ret - self._dsr_B

        self._dsr_A += eta * delta_A
        self._dsr_B += eta * delta_B

        var = self._dsr_B - self._dsr_A ** 2
        if var < 1e-12:
            dsr = delta_A
        else:
            dsr = (self._dsr_B * delta_A - 0.5 * self._dsr_A * delta_B) / (var ** 1.5)

        return float(self._dsr_weight * np.clip(dsr, -2.0, 2.0))

    def _apply_action(self, action: int, close_px: float) -> None:
        if action == ACTION_HOLD:
            return
        if action == ACTION_CLOSE:
            self._close_position(close_px, reason="action")
            return
        if action == ACTION_LONG:
            if self._position == 1:
                return
            if self._position == -1:
                self._close_position(close_px, reason="reverse")
            self._open_position(1, close_px)
            return
        if action == ACTION_SHORT:
            if self._position == -1:
                return
            if self._position == 1:
                self._close_position(close_px, reason="reverse")
            self._open_position(-1, close_px)

    def _fee(self, notional: float) -> float:
        return abs(notional) * self._commission_rate

    def _open_position(self, direction: int, close_px: float) -> None:
        if direction == 1:
            fill = close_px * (1.0 + self._slippage)
        else:
            fill = close_px * (1.0 - self._slippage)
        fee = self._fee(fill)
        self._equity -= fee
        self._last_commission_paid += fee
        self._position = direction
        self._entry_price = fill
        self._opened_this_step = True

    def _unrealized_pnl_pct(self) -> float:
        """Current unrealized PnL as a fraction, clipped to [-1, 1]."""
        if self._position == 0 or self._entry_price < 1e-12:
            return 0.0
        curr_close = float(self._bars[min(self._t, self._bars.shape[0] - 1), 3])
        if self._position == 1:
            pnl_pct = (curr_close - self._entry_price) / self._entry_price
        else:
            pnl_pct = (self._entry_price - curr_close) / self._entry_price
        return float(np.clip(pnl_pct, -1.0, 1.0))

    def _close_position(self, close_px: float, reason: str) -> None:
        del reason
        if self._position == 0:
            return
        if self._position == 1:
            fill = close_px * (1.0 - self._slippage)
            pnl = fill - self._entry_price
        else:
            fill = close_px * (1.0 + self._slippage)
            pnl = self._entry_price - fill
        fee = self._fee(fill)
        self._equity += pnl - fee
        self._last_realized_pnl += pnl
        self._last_commission_paid += fee
        self._position = 0
        self._entry_price = 0.0
        self._hold_steps = 0

    def _mark_to_market(self, new_close: float, ref_close: float) -> None:
        """从未平仓盯市：按 ref_close → new_close 的价差更新权益。"""
        if self._position == 0:
            return
        if self._position == 1:
            self._equity += new_close - ref_close
        else:
            self._equity += ref_close - new_close

    def close(self) -> None:
        logger.debug("FuturesTradingEnv.close")
