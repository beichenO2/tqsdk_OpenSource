"""PPO training with MambaFormer feature extractor.

Integrates the v4 MambaFormerExtractor (S4D + Transformer) into the PPO
training loop, replacing the default MLP, and uses DSR reward instead of
the legacy MTM Delta.

Usage:
    python train_ppo.py --checkpoint models/rl_v4_mambaformer/final_model.zip --timesteps 1000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
CHECKPOINT_DEFAULT = (
    SCRIPT_DIR.parent.parent / "models" / "rl_v4_mambaformer" / "final_model.zip"
)


def _generate_synthetic_bars(n_bars: int = 5000, seed: int = 42) -> np.ndarray:
    """Generate synthetic futures-like OHLCV bar data for training validation."""
    rng = np.random.default_rng(seed)
    price = 5000.0
    bars = []
    for _ in range(n_bars):
        ret = rng.normal(0.0001, 0.015)
        high_ext = abs(rng.normal(0, 0.005))
        low_ext = abs(rng.normal(0, 0.005))
        open_p = price
        close_p = price * (1 + ret)
        high_p = max(open_p, close_p) * (1 + high_ext)
        low_p = min(open_p, close_p) * (1 - low_ext)
        volume = max(100, rng.normal(10000, 3000))
        bars.append([open_p, high_p, low_p, close_p, volume])
        price = close_p
    return np.array(bars)


def _max_drawdown(equity_curve: list[float]) -> float:
    peak = equity_curve[0]
    worst = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        if peak > 1e-12:
            dd = (eq - peak) / peak
            if dd < worst:
                worst = dd
    return float(worst)


def _create_mambaformer_policy(env_obs_space, action_space, extractor_kwargs, d_model, extra_features):
    """Build a CustomPPOPolicy that uses MambaFormerExtractor as the feature extractor.

    This creates a policy class compatible with SB3's ActorCriticPolicy,
    where the feature extractor is replaced with MambaFormer (pre-trained).
    """
    from stable_baselines3.common.policies import ActorCriticPolicy
    from packages.rl.mambaformer_extractor import MambaFormerExtractor

    # MambaFormer returns (latent_pi, latent_vf) but SB3's BaseFeaturesExtractor
    # interface expects a single tensor. We wrap with an adapter that returns
    # only latent_pi (the policy latent), since MambaFormer already shares
    # features between pi and vf.
    class MambaFormerFeaturesAdapter(nn.Module):
        """SB3-compatible adapter that wraps a MambaFormerExtractor.

        MambaFormerExtractor.forward() returns (latent_pi, latent_vf).
        This adapter returns only latent_pi so it satisfies the
        BaseFeaturesExtractor single-tensor contract.
        """

        def __init__(self, observation_space, **kw):
            super().__init__()
            self._mambaformer = MambaFormerExtractor(observation_space=observation_space, **kw)
            self._features_dim = self._mambaformer.features_dim

        @property
        def features_dim(self) -> int:
            return self._features_dim

        def forward(self, observations) -> torch.Tensor:  # type: ignore[name-defined]
            latent_pi, _ = self._mambaformer(observations)
            return latent_pi

        def load_pretrained(self, checkpoint_path: str) -> None:
            """Load pre-trained MambaFormer weights from an SB3 model zip."""
            import io, json, zipfile
            import torch

            with zipfile.ZipFile(checkpoint_path, "r") as zf:
                policy_state = torch.load(
                    io.BytesIO(zf.read("policy.pth")),
                    map_location="cpu",
                    weights_only=False,
                )

            prefix = "features_extractor."
            prefix_len = len(prefix)
            sd = {}
            for key, val in policy_state.items():
                if key.startswith(prefix):
                    sd[key[prefix_len:]] = val

            self._mambaformer.load_state_dict(sd)
            logger.info("Loaded %d pre-trained weights into MambaFormer adapter", len(sd))

    class MambaFormerPolicy(ActorCriticPolicy):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    return MambaFormerPolicy, MambaFormerFeaturesAdapter


def train_with_mambaformer(
    bars: np.ndarray,
    *,
    checkpoint_path: str | None = None,
    total_timesteps: int = 1000,
    eval_freq: int = 500,
    window_size: int = 30,
    seed: int = 42,
    initial_balance: float = 100_000.0,
    commission_rate: float = 0.0003,
    slippage: float = 0.0001,
) -> dict:
    """Train PPO with MambaFormerExtractor on bar data."""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import EvalCallback
        from stable_baselines3.common.vec_env import DummyVecEnv
        import torch
    except ImportError as exc:  # pragma: no cover
        logger.error("stable-baselines3 not installed: %s", exc)
        sys.exit(1)

    from packages.rl.mambaformer_extractor import MambaFormerExtractor
    from packages.rl.trading_env import FuturesTradingEnv

    # Infer observation space from the environment
    _env = FuturesTradingEnv(bars[: window_size + 10], {"window_size": window_size})
    obs_space = _env.observation_space
    obs_dim = int(obs_space.shape[0])
    seq_features = window_size * 5
    extra_features = obs_dim - seq_features
    logger.info(
        "obs_dim=%d, seq_features=%d (window=%d, bars=5), extra_features=%d",
        obs_dim, seq_features, window_size, extra_features,
    )

    extractor_kwargs = dict(
        d_model=64,
        n_s4d_layers=2,
        n_attn_layers=1,
        nhead=4,
        d_state=16,
        features_per_bar=5,
        extra_features=extra_features,
        dropout=0.1,
    )

    # Walk-forward split: 80% train, 20% eval
    split = int(len(bars) * 0.8)
    train_bars = bars[:split]
    eval_bars = bars[split:]
    logger.info("Train bars: %d, Eval bars: %d", len(train_bars), len(eval_bars))

    env_cfg = {
        "window_size": window_size,
        "initial_balance": initial_balance,
        "commission_rate": commission_rate,
        "slippage": slippage,
        "reward_mode": "dsr",  # DSR instead of legacy MTM Delta
    }

    train_env = DummyVecEnv([lambda: FuturesTradingEnv(train_bars, env_cfg)])
    eval_env = DummyVecEnv([lambda: FuturesTradingEnv(eval_bars, env_cfg)])

    save_dir = str(SCRIPT_DIR / "checkpoints_mambaformer")
    os.makedirs(save_dir, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=save_dir,
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=3,
    )

    logger.info("Starting PPO+MambaFormer training for %d steps", total_timesteps)

    # Build the adapter and policy
    MambaFormerPolicy, MambaFormerFeaturesAdapter = _create_mambaformer_policy(
        obs_space,
        train_env.action_space,
        extractor_kwargs,
        d_model=64,
        extra_features=extra_features,
    )

    model = PPO(
        MambaFormerPolicy,
        train_env,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            features_extractor_class=MambaFormerFeaturesAdapter,
            features_extractor_kwargs=extractor_kwargs,
        ),
        tensorboard_log=os.path.join(save_dir, "tb_logs"),
        verbose=1,
        seed=seed,
    )

    # Load pre-trained weights if checkpoint provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        _load_pretrained_adapter(model, checkpoint_path, extractor_kwargs)

    model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=True)

    final_path = os.path.join(save_dir, "ppo_mambaformer_final")
    model.save(final_path)
    logger.info("Model saved to %s", final_path)

    # Evaluate on hold-out bars
    eval_raw_env = FuturesTradingEnv(eval_bars, env_cfg)
    results = _evaluate(model, eval_raw_env, initial_balance)
    results["model_path"] = final_path + ".zip"

    results_path = os.path.join(save_dir, "training_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s", results_path)

    return results


def _load_pretrained_adapter(model: object, checkpoint_path: str, extractor_kwargs: dict) -> None:
    """Load pre-trained MambaFormer weights into the policy's feature extractor adapter."""
    import io, json, zipfile
    import torch

    adapter = model.policy.features_extractor
    with zipfile.ZipFile(checkpoint_path, "r") as zf:
        policy_state = torch.load(
            io.BytesIO(zf.read("policy.pth")),
            map_location="cpu",
            weights_only=False,
        )

    prefix = "features_extractor."
    prefix_len = len(prefix)
    sd = {}
    for key, val in policy_state.items():
        if key.startswith(prefix):
            sd[key[prefix_len:]] = val

    adapter._mambaformer.load_state_dict(sd)
    logger.info("Loaded %d pre-trained weights into MambaFormer adapter", len(sd))


def _evaluate(model: object, env: object, initial_balance: float) -> dict:
    """Run deterministic evaluation and return summary metrics."""
    obs, _ = env.reset()
    equity_curve = [env._equity]
    total_reward = 0.0
    step = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
        equity_curve.append(env._equity)
        step += 1
        if terminated or truncated or step >= 10_000:
            break

    final_return = (env._equity - initial_balance) / initial_balance
    max_dd = _max_drawdown(equity_curve)

    rets = np.diff(equity_curve) / np.maximum(np.abs(equity_curve[:-1]), 1e-10)
    sharpe = 0.0
    if len(rets) > 1 and np.std(rets) > 1e-12:
        sharpe = float(np.mean(rets) / np.std(rets) * np.sqrt(252))

    results = {
        "total_return": round(final_return, 6),
        "final_equity": round(env._equity, 2),
        "max_drawdown": round(max_dd, 6),
        "sharpe_ratio": round(sharpe, 4),
        "total_reward": round(total_reward, 4),
        "steps": step,
    }
    logger.info("=== Evaluation Results ===")
    for k, v in results.items():
        logger.info("  %-20s: %s", k, v)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO with MambaFormerExtractor")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(CHECKPOINT_DEFAULT),
        help="Path to MambaFormerExtractor checkpoint",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1000,
        help="Total training timesteps (default 1000 for quick validation)",
    )
    parser.add_argument(
        "--bars",
        type=int,
        default=5000,
        help="Number of synthetic bars to generate",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-freq", type=int, default=500)
    args = parser.parse_args()

    checkpoint = args.checkpoint if os.path.exists(args.checkpoint) else None
    if checkpoint:
        logger.info("Using checkpoint: %s", checkpoint)
    else:
        logger.info("Building fresh MambaFormerExtractor (no checkpoint)")

    bars = _generate_synthetic_bars(n_bars=args.bars, seed=args.seed)
    logger.info("Generated %d synthetic bars", len(bars))

    results = train_with_mambaformer(
        bars,
        checkpoint_path=checkpoint,
        total_timesteps=args.timesteps,
        eval_freq=args.eval_freq,
        seed=args.seed,
    )

    if results["total_return"] > 0:
        logger.info("✅ PASSED: Return %.4f > 0", results["total_return"])
    else:
        logger.warning(
            "⚠️  Return %.4f <= 0 — may need more training steps",
            results["total_return"],
        )


if __name__ == "__main__":
    main()
