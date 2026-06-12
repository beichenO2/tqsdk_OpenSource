"""Soft Actor-Critic (SAC) Reinforcement Learning Trading Strategy.

Uses continuous action space with entropy regularization for exploration-exploitation
balance. The agent learns optimal position sizing and direction simultaneously,
with a richer state representation than the existing PPO implementation.

Key differences from existing PPO:
  - Continuous actions: position weight ∈ [-1, +1] instead of discrete {hold, long, short, close}
  - Entropy bonus: automatic temperature tuning prevents premature convergence
  - Multi-feature state: technical indicators + market regime features
  - Risk-aware reward: Sortino-based shaping penalizes downside more than upside

Architecture:
  State (OHLCV + indicators + regime) → SAC Agent → Continuous Position Weight → Execution

Usage:
    python scripts/research/frontier/sac_rl_strategy.py [--symbol BTCUSDT] [--weeks 80]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import os as _os
import sys as _sys

try:
    from polarisor_port_sdk import submit_task as _sdk_submit, complete_task as _sdk_complete
except ImportError:
    _sdk_submit = _sdk_complete = None

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import add_technical_features, compute_metrics, load_crypto_bars

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:
    logger.error("stable-baselines3 required: pip install stable-baselines3[extra]")
    sys.exit(1)


class ContinuousTradingEnv(gym.Env):
    """Enhanced Gymnasium env with continuous position sizing for SAC.

    Action: single float in [-1, 1] representing target position weight
      - Positive → long, negative → short, near zero → flat
    State: sliding window of normalized features + position info + equity ratio
    Reward: PnL with Sortino-style asymmetric shaping + transaction cost penalty
    """

    metadata = {"render_modes": []}

    def __init__(self, bars: pd.DataFrame, config: dict[str, Any] | None = None):
        super().__init__()
        config = config or {}

        self._df = add_technical_features(bars)
        self._feature_cols = [
            "returns", "log_returns", "vol_ratio", "rsi", "bb_pctb",
            "macd_hist", "atr_14", "high_low_range", "volatility_20",
        ]
        self._features = self._df[self._feature_cols].values.astype(np.float64)
        self._closes = self._df["close"].values.astype(np.float64)

        self._window = int(config.get("window_size", 30))
        self._initial_balance = float(config.get("initial_balance", 100_000.0))
        self._commission_rate = float(config.get("commission_rate", 0.0003))
        self._slippage = float(config.get("slippage", 0.0001))
        self._leverage = float(config.get("leverage", 5.0))

        n_features = len(self._feature_cols)
        obs_dim = self._window * n_features + 3  # features + [position, equity_ratio, unrealized_pnl]

        feat_flat = self._features.ravel()
        self._feat_mean = self._features.mean(axis=0)
        self._feat_std = self._features.std(axis=0) + 1e-8

        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self._t = 0
        self._position = 0.0
        self._entry_price = 0.0
        self._equity = self._initial_balance
        self._reward_history: list[float] = []
        self._trades: list[dict] = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self._t = self._window
        self._position = 0.0
        self._entry_price = 0.0
        self._equity = self._initial_balance
        self._reward_history.clear()
        self._trades.clear()
        return self._get_obs(), {"equity": self._equity}

    def step(self, action):
        target_pos = float(np.clip(action[0], -1.0, 1.0))

        close_px = self._closes[self._t]
        prev_equity = self._equity

        pos_change = target_pos - self._position
        if abs(pos_change) > 0.05:
            trade_cost = abs(pos_change) * close_px * (self._commission_rate + self._slippage)
            self._equity -= trade_cost

            if self._position != 0 and abs(target_pos) < abs(self._position):
                pnl = (close_px - self._entry_price) / self._entry_price * self._position * self._leverage
                self._trades.append({"pnl_pct": pnl})

            if abs(target_pos) > 0.05:
                self._entry_price = close_px
            self._position = target_pos

        self._t += 1
        terminated = self._t >= len(self._closes) - 1
        truncated = False

        if not terminated:
            new_close = self._closes[self._t]
            price_return = (new_close - close_px) / close_px
            position_pnl = self._position * price_return * self._leverage * self._equity
            self._equity += position_pnl

        if self._equity < self._initial_balance * 0.1:
            terminated = True

        reward = self._compute_reward(prev_equity)

        obs = self._get_obs()
        info = {"equity": self._equity, "position": self._position, "step": self._t}

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        t = min(self._t, len(self._features) - 1)
        start = max(0, t - self._window + 1)
        window_feats = self._features[start:t + 1]

        normalized = (window_feats - self._feat_mean) / self._feat_std
        normalized = np.clip(normalized, -3.0, 3.0)

        if len(normalized) < self._window:
            pad = np.zeros((self._window - len(normalized), normalized.shape[1]))
            normalized = np.vstack([pad, normalized])

        flat = normalized.flatten().astype(np.float32)

        meta = np.array([
            self._position,
            self._equity / self._initial_balance - 1.0,
            (self._closes[min(self._t, len(self._closes) - 1)] - self._entry_price) / (self._entry_price + 1e-10) * self._position if self._entry_price > 0 else 0.0,
        ], dtype=np.float32)

        return np.concatenate([flat, meta])

    def _compute_reward(self, prev_equity: float) -> float:
        pnl = (self._equity - prev_equity) / self._initial_balance

        self._reward_history.append(pnl)
        if len(self._reward_history) > 50:
            self._reward_history.pop(0)

        sortino_bonus = 0.0
        if len(self._reward_history) >= 10:
            arr = np.array(self._reward_history)
            downside = arr[arr < 0]
            downside_std = np.std(downside) if len(downside) > 1 else 1e-8
            mean_ret = np.mean(arr)
            sortino_bonus = 0.01 * mean_ret / (downside_std + 1e-8)
            sortino_bonus = np.clip(sortino_bonus, -0.5, 0.5)

        turnover_penalty = abs(self._position) * 1e-5

        return float(pnl + sortino_bonus - turnover_penalty)


def train_sac_agent(bars: pd.DataFrame, config: dict | None = None,
                    total_timesteps: int = 100_000) -> tuple:
    config = config or {}

    def make_env():
        return ContinuousTradingEnv(bars, config)

    env = DummyVecEnv([make_env])

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=50_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        ent_coef="auto",
        train_freq=1,
        gradient_steps=1,
        verbose=0,
        seed=42,
    )

    logger.info("Training SAC agent for %d timesteps...", total_timesteps)
    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    return model, env


def backtest_sac(
    bars: pd.DataFrame, model, leverage: int = 8,
    initial_capital: float = 100.0, commission_pct: float = 0.0004,
    slippage_pct: float = 0.0003,
) -> dict:
    env = ContinuousTradingEnv(bars, {
        "leverage": leverage,
        "initial_balance": initial_capital * 1000,
        "commission_rate": commission_pct,
        "slippage": slippage_pct,
    })
    obs, info = env.reset()

    capital = initial_capital
    peak = capital
    equity_curve = [capital]
    trades = []
    prev_position = 0.0
    entry_price = 0.0
    closes = bars["close"].values.astype(np.float64)
    cost = commission_pct + slippage_pct

    for step in range(len(closes) - env._window - 2):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        target_pos = float(np.clip(action[0], -1.0, 1.0))
        current_bar = min(env._t, len(closes) - 1)
        current_price = closes[current_bar]

        if abs(target_pos - prev_position) > 0.1:
            if prev_position != 0 and entry_price > 0:
                direction = 1 if prev_position > 0 else -1
                pnl_pct = (current_price / entry_price - 1) * direction * leverage
                realized = capital * abs(pnl_pct) * (1 if pnl_pct > 0 else -1) - capital * cost * 2
                capital += realized
                trades.append({
                    "entry_bar": step, "exit_bar": step,
                    "direction": direction, "pnl_pct": pnl_pct,
                    "realized": realized,
                })

            if abs(target_pos) > 0.1:
                entry_price = current_price
                capital -= capital * cost
            else:
                entry_price = 0.0

            prev_position = target_pos

        elif prev_position != 0 and entry_price > 0:
            direction = 1 if prev_position > 0 else -1
            unrealized = (current_price / entry_price - 1) * direction * leverage
            equity_val = capital * (1 + unrealized)
            equity_curve.append(max(equity_val, 0))
            continue

        equity_curve.append(capital)

        if terminated or truncated:
            break

    if prev_position != 0 and entry_price > 0:
        current_price = closes[-1]
        direction = 1 if prev_position > 0 else -1
        pnl_pct = (current_price / entry_price - 1) * direction * leverage
        realized = capital * abs(pnl_pct) * (1 if pnl_pct > 0 else -1) - capital * cost * 2
        capital += realized
        trades.append({
            "entry_bar": len(closes) - 1, "exit_bar": len(closes) - 1,
            "direction": direction, "pnl_pct": pnl_pct,
            "realized": realized,
        })
        equity_curve.append(capital)

    return compute_metrics(trades, initial_capital, capital, equity_curve)


def main():

    _task_id = None
    if _sdk_submit:
        try:
            _tr = _sdk_submit(task_type="rl-training", command="sac_rl_strategy.py", requester="sac-rl-strategy", estimated_duration_sec=3600)
            _task_id = _tr.get("task_id")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="SAC RL Trading Strategy")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--weeks", type=int, default=80)
    parser.add_argument("--leverage", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=100_000)
    args = parser.parse_args()

    bars = load_crypto_bars(args.symbol, args.timeframe, args.weeks)
    if len(bars) < 200:
        logger.error("Insufficient data: %d bars", len(bars))
        return

    split_idx = int(len(bars) * 0.7)
    train_bars = bars.iloc[:split_idx].reset_index(drop=True)
    test_bars = bars.iloc[split_idx:].reset_index(drop=True)

    logger.info("Training SAC agent on %d bars...", len(train_bars))
    model, _ = train_sac_agent(train_bars, total_timesteps=args.timesteps)

    logger.info("Backtesting on out-of-sample %d bars...", len(test_bars))
    results = backtest_sac(test_bars, model, leverage=args.leverage)

    logger.info("\n%s", "=" * 60)
    logger.info("SAC RL STRATEGY — %s (OOS)", args.symbol)
    logger.info("=" * 60)
    for k, v in results.items():
        logger.info("  %s: %s", k, v)

    out_dir = Path(__file__).resolve().parent.parent.parent.parent / "models" / "frontier"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sac_results.json"
    with open(out_path, "w") as f:
        json.dump({"strategy": "SAC_RL", "symbol": args.symbol, "leverage": args.leverage,
                    "timesteps": args.timesteps, **results}, f, indent=2)
    logger.info("Results saved to %s", out_path)

    return results


    if _task_id and _sdk_complete:
        try:
            _sdk_complete(_task_id)
        except Exception:
            pass


if __name__ == "__main__":
    main()
