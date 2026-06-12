"""Train RL PPO agent on crypto data with improved environment.

Uses the v4.0 Phase 25 improved FuturesTradingEnv which has:
- MTM Delta dense reward (no hold_penalty)
- Trend Bonus for trend-following behavior
- Drawdown Penalty (progressive)
- Rolling window normalization (no future data leakage)
- Discrete actions {hold, long, short, close}
- Position duration + unrealized PnL features

Run: python3 packages/crypto/scripts/train_crypto_rl.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

DATA_DIR = Path.home() / "Downloads" / "crypto_data"
MODEL_DIR = Path(__file__).resolve().parents[3] / "models" / "rl_crypto"
SYMBOL = "BTCUSDT"
TIMEFRAME = "1h"
TOTAL_TIMESTEPS = 2_000_000
WINDOW_SIZE = 30


def load_ohlcv(symbol: str, timeframe: str) -> np.ndarray:
    path = DATA_DIR / symbol.lower() / f"{timeframe}.parquet"
    if not path.exists():
        print(f"Data not found: {path}")
        sys.exit(1)
    df = pd.read_parquet(path)
    bars = df[["open", "high", "low", "close", "volume"]].values.astype(np.float64)
    print(f"Loaded {len(bars)} bars from {path}")
    return bars


def main():
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError:
        print("stable-baselines3 not installed. Run: pip install stable-baselines3")
        sys.exit(1)

    from rl.trading_env import FuturesTradingEnv

    bars = load_ohlcv(SYMBOL, TIMEFRAME)
    train_end = int(len(bars) * 0.8)
    train_bars = bars[:train_end]
    test_bars = bars[train_end:]

    print(f"Train: {len(train_bars)} bars, Test: {len(test_bars)} bars")

    env_config = {
        "window_size": WINDOW_SIZE,
        "initial_balance": 10000.0,
        "commission_rate": 0.0004,
        "slippage": 0.0003,
        "trend_bonus_scale": 0.5,
        "dd_penalty_scale": 2.0,
        "dd_threshold": 0.05,
        "bankruptcy_fraction": 0.01,
    }

    def make_env():
        return FuturesTradingEnv(train_bars, env_config)

    vec_env = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
    )

    print(f"\nTraining PPO for {TOTAL_TIMESTEPS:,} timesteps...")
    t0 = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    elapsed = time.time() - t0
    print(f"Training completed in {elapsed:.0f}s")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"ppo_crypto_{SYMBOL}_{TIMEFRAME}_{datetime.now().strftime('%Y%m%d')}"
    model.save(str(model_path))
    print(f"Model saved to {model_path}")

    print("\n=== OOS Evaluation ===")
    test_env = FuturesTradingEnv(test_bars, env_config)
    obs, info = test_env.reset()
    total_reward = 0.0
    steps = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated:
            break

    final_equity = info.get("equity", 0)
    oos_return = (final_equity - 10000) / 10000 * 100
    print(f"OOS Steps: {steps}")
    print(f"OOS Final Equity: {final_equity:.2f}")
    print(f"OOS Return: {oos_return:.2f}%")
    print(f"OOS Total Reward: {total_reward:.4f}")

    results = {
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "train_bars": len(train_bars),
        "test_bars": len(test_bars),
        "total_timesteps": TOTAL_TIMESTEPS,
        "training_time_sec": elapsed,
        "oos_return_pct": oos_return,
        "oos_final_equity": final_equity,
        "oos_total_reward": total_reward,
        "model_path": str(model_path),
    }

    results_path = MODEL_DIR / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
