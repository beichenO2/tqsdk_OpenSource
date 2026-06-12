"""PPO Reinforcement Learning Trading Agent — Uses existing Gymnasium env.

Trains a PPO agent (via Stable-Baselines3) on crypto OHLCV bars,
evaluates via backtest metrics compatible with V4 comparison.

Usage:
    python scripts/research/ppo_rl_strategy.py [--symbol BTCUSDT] [--weeks 80] [--leverage 8]
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

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from datahub.crypto_loader import CryptoDataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    logger.error("gymnasium required: pip install gymnasium")
    sys.exit(1)

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:
    logger.error("stable-baselines3 required: pip install stable-baselines3")
    sys.exit(1)


class CryptoTradingEnv(gym.Env):
    """Crypto trading env with richer observation space for PPO training.

    Observation: [window_size x 10] flattened features per bar:
        returns, volume_ratio, rsi, atr_norm, adx_norm,
        ema_fast_diff, ema_slow_diff, taker_buy_ratio,
        position_encoding (one-hot), equity_ratio
    """

    metadata = {"render_modes": []}

    def __init__(self, bars: np.ndarray, features: np.ndarray,
                 window: int = 30, leverage: int = 8, commission: float = 0.0004):
        super().__init__()
        self._bars = bars  # (N, 5) OHLCV
        self._features = features  # (N, n_feat)
        self._window = window
        self._leverage = leverage
        self._commission = commission
        self._initial_balance = 100.0

        n_obs = window * features.shape[1] + 3  # +3 for position, equity, bars_held
        self.observation_space = spaces.Box(low=-10, high=10, shape=(n_obs,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)  # hold, long, short, close

        self._idx = window
        self._balance = self._initial_balance
        self._position = 0  # -1, 0, 1
        self._entry_price = 0.0
        self._bars_held = 0
        self._peak = self._initial_balance

    def _get_obs(self):
        feat_window = self._features[self._idx - self._window : self._idx].flatten()
        pos_enc = np.array([float(self._position == 1), float(self._position == -1),
                           self._balance / self._initial_balance], dtype=np.float32)
        return np.concatenate([feat_window, pos_enc]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._idx = self._window
        self._balance = self._initial_balance
        self._position = 0
        self._entry_price = 0.0
        self._bars_held = 0
        self._peak = self._initial_balance
        return self._get_obs(), {}

    def step(self, action):
        close = self._bars[self._idx, 3]  # close price
        reward = 0.0
        cost = self._commission + 0.0003  # commission + slippage

        if self._position != 0:
            self._bars_held += 1
            pnl = (close / self._entry_price - 1) * self._position * self._leverage
            unrealized = self._balance * pnl

            should_close = (
                action == 3 or
                self._bars_held >= 48 or
                pnl < -0.90 / self._leverage  # liquidation
            )
            if should_close:
                realized = self._balance * abs(pnl) * (1 if pnl > 0 else -1) - self._balance * cost * 2
                self._balance += realized
                reward = realized / self._initial_balance
                self._position = 0
                self._entry_price = 0.0
                self._bars_held = 0

        if self._position == 0:
            if action == 1:  # long
                self._position = 1
                self._entry_price = close
                self._balance -= self._balance * cost
                self._bars_held = 0
            elif action == 2:  # short
                self._position = -1
                self._entry_price = close
                self._balance -= self._balance * cost
                self._bars_held = 0

        reward -= 1e-5  # small hold penalty

        self._peak = max(self._peak, self._balance)
        dd = (self._peak - self._balance) / self._peak if self._peak > 0 else 0
        if dd > 0.5:
            reward -= 0.1  # drawdown penalty

        self._idx += 1
        terminated = self._idx >= len(self._bars) - 1 or self._balance <= self._initial_balance * 0.01
        truncated = False

        return self._get_obs(), reward, terminated, truncated, {}


def build_features(bars: pd.DataFrame) -> np.ndarray:
    """Build normalized feature matrix from OHLCV."""
    closes = bars["close"].values.astype(np.float64)
    highs = bars["high"].values.astype(np.float64)
    lows = bars["low"].values.astype(np.float64)
    volumes = bars["volume"].values.astype(np.float64)
    tbv = bars.get("taker_buy_volume", bars["volume"] * 0.5).values.astype(np.float64)

    returns = np.diff(np.log(closes + 1e-10), prepend=np.log(closes[0] + 1e-10))
    vol_ratio = volumes / (pd.Series(volumes).rolling(20).mean().fillna(1).values + 1e-10)

    delta = pd.Series(closes).diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = (100 - (100 / (1 + gain / (loss + 1e-10)))).fillna(50).values / 100.0

    tr = np.maximum(highs - lows, np.maximum(abs(highs - np.roll(closes, 1)), abs(lows - np.roll(closes, 1))))
    atr = pd.Series(tr).rolling(14).mean().fillna(0).values
    atr_norm = atr / (closes + 1e-10)

    ema12 = pd.Series(closes).ewm(span=12).mean().values
    ema26 = pd.Series(closes).ewm(span=26).mean().values
    ema_fast_diff = (closes - ema12) / (closes + 1e-10)
    ema_slow_diff = (closes - ema26) / (closes + 1e-10)

    tbr = tbv / (volumes + 1e-10)

    features = np.column_stack([
        returns, vol_ratio, rsi, atr_norm, ema_fast_diff, ema_slow_diff, tbr
    ])

    mean = np.nanmean(features, axis=0)
    std = np.nanstd(features, axis=0) + 1e-8
    features = np.clip((features - mean) / std, -5, 5)
    features = np.nan_to_num(features, nan=0.0)

    return features


def evaluate_agent(env, model, bars_df: pd.DataFrame) -> dict:
    """Run trained agent through env and collect trading metrics."""
    obs, _ = env.reset()
    done = False
    trades = []
    current_trade = None

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        prev_pos = env._position
        prev_balance = env._balance

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        if prev_pos == 0 and env._position != 0:
            current_trade = {"entry_bar": env._idx, "direction": env._position, "entry_balance": prev_balance}
        elif prev_pos != 0 and env._position == 0 and current_trade:
            pnl = (env._balance - current_trade["entry_balance"]) / current_trade["entry_balance"]
            trades.append({"pnl_pct": pnl, "direction": current_trade["direction"]})
            current_trade = None

    capital = env._balance
    initial = env._initial_balance
    total_return = (capital - initial) / initial * 100
    n_trades = len(trades)
    wins = [t for t in trades if t["pnl_pct"] > 0]
    win_rate = len(wins) / n_trades * 100 if n_trades else 0
    losses = [t for t in trades if t["pnl_pct"] <= 0]

    pf = 999
    if losses and sum(abs(t["pnl_pct"]) for t in losses) > 0:
        pf = sum(t["pnl_pct"] for t in wins) / sum(abs(t["pnl_pct"]) for t in losses)

    returns_arr = [t["pnl_pct"] for t in trades]
    sharpe = (np.mean(returns_arr) / (np.std(returns_arr) + 1e-10)) * np.sqrt(252 / max(1, n_trades)) if returns_arr else 0

    peak = initial
    max_dd = 0.0
    running = initial
    for t in trades:
        running *= (1 + t["pnl_pct"])
        peak = max(peak, running)
        dd = (peak - running) / peak
        max_dd = max(max_dd, dd)

    return {
        "total_return_pct": round(total_return, 2),
        "n_trades": n_trades,
        "win_rate": round(win_rate, 1),
        "profit_factor": round(pf, 2),
        "max_drawdown_pct": round(max_dd * 100, 1),
        "sharpe": round(sharpe, 2),
        "final_capital": round(capital, 2),
    }


def main():

    _task_id = None
    if _sdk_submit:
        try:
            _tr = _sdk_submit(task_type="rl-training", command="ppo_rl_strategy.py", requester="ppo-rl-strategy", estimated_duration_sec=3600)
            _task_id = _tr.get("task_id")
        except Exception:
            pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--weeks", type=int, default=80)
    parser.add_argument("--leverage", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=100_000)
    args = parser.parse_args()

    loader = CryptoDataLoader()
    try:
        bars = loader.load_with_funding(args.symbol, args.timeframe)
    except Exception:
        bars = loader.load(args.symbol, args.timeframe)

    if args.weeks:
        bars = bars.tail(args.weeks * 7 * 24).reset_index(drop=True)

    logger.info(f"Loaded {len(bars)} bars for {args.symbol}")

    features = build_features(bars)
    ohlcv = bars[["open", "high", "low", "close", "volume"]].values.astype(np.float64)

    split = int(len(bars) * 0.7)
    train_ohlcv, test_ohlcv = ohlcv[:split], ohlcv[split:]
    train_feat, test_feat = features[:split], features[split:]

    def make_train_env():
        return CryptoTradingEnv(train_ohlcv, train_feat, leverage=args.leverage)

    vec_env = DummyVecEnv([make_train_env])

    logger.info(f"Training PPO for {args.timesteps} timesteps...")
    model = PPO(
        "MlpPolicy", vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=True)

    logger.info("Evaluating on out-of-sample data...")
    test_env = CryptoTradingEnv(test_ohlcv, test_feat, leverage=args.leverage)
    results = evaluate_agent(test_env, model, bars.iloc[split:])

    logger.info(f"\n{'='*60}")
    logger.info(f"PPO RL STRATEGY — {args.symbol} (OOS)")
    logger.info(f"{'='*60}")
    for k, v in results.items():
        logger.info(f"  {k}: {v}")

    out_path = Path(__file__).resolve().parent.parent.parent / "models" / "ppo_rl_results.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"symbol": args.symbol, "leverage": args.leverage, "timesteps": args.timesteps, **results}, f, indent=2)
    logger.info(f"Results saved to {out_path}")

    model_path = Path(__file__).resolve().parent.parent.parent / "models" / "ppo_crypto_agent"
    model.save(str(model_path))
    logger.info(f"Model saved to {model_path}")

    return results


    if _task_id and _sdk_complete:
        try:
            _sdk_complete(_task_id)
        except Exception:
            pass


if __name__ == "__main__":
    main()
