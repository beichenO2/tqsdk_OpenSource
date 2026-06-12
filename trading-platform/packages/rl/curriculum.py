"""Curriculum learning pipeline for RL trading agent.

Implements 4-stage progressive training that starts from easy market
conditions and gradually introduces harder regimes.  Also supports
multi-instrument parallel environments via SB3 SubprocVecEnv.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False

from .trading_env import FuturesTradingEnv, compute_ta_features


@dataclass
class CurriculumStage:
    name: str
    description: str
    volatility_range: tuple[float, float]
    timesteps: int
    data_filter: Callable[[np.ndarray], np.ndarray] | None = None


DEFAULT_STAGES: list[CurriculumStage] = [
    CurriculumStage(
        name="low_vol_trend",
        description="低波动趋势行情 — 容易学习基本交易逻辑",
        volatility_range=(0.0, 0.02),
        timesteps=200_000,
    ),
    CurriculumStage(
        name="mixed",
        description="混合市况 — 趋势与震荡交替",
        volatility_range=(0.0, 0.05),
        timesteps=300_000,
    ),
    CurriculumStage(
        name="high_vol",
        description="高波动行情 — 学习风险控制",
        volatility_range=(0.02, 0.15),
        timesteps=300_000,
    ),
    CurriculumStage(
        name="full",
        description="全量数据验证 — 所有市况",
        volatility_range=(0.0, 1.0),
        timesteps=500_000,
    ),
]


def _bar_volatility(bars: np.ndarray, window: int = 20) -> np.ndarray:
    """Per-bar rolling volatility (close-to-close returns std)."""
    close = bars[:, 3]
    returns = np.diff(np.log(np.maximum(close, 1e-12)))
    vol = np.full(len(bars), 0.0)
    for i in range(window, len(returns)):
        vol[i + 1] = np.std(returns[i - window : i])
    return vol


def filter_bars_by_volatility(
    bars: np.ndarray,
    vol_low: float,
    vol_high: float,
    min_segment: int = 100,
) -> np.ndarray:
    """Select contiguous segments of bars within the volatility range.

    Returns a subset of bars (preserving order) whose rolling volatility
    falls within [vol_low, vol_high].  Short fragments (< min_segment)
    are dropped.
    """
    vol = _bar_volatility(bars)
    mask = (vol >= vol_low) & (vol <= vol_high)

    segments: list[np.ndarray] = []
    start = None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if i - start >= min_segment:
                segments.append(bars[start:i])
            start = None
    if start is not None and len(bars) - start >= min_segment:
        segments.append(bars[start:])

    if not segments:
        logger.warning(
            "No segments found for vol [%.4f, %.4f]; using full data", vol_low, vol_high
        )
        return bars

    return np.concatenate(segments, axis=0)


def make_env(
    bars: np.ndarray,
    env_config: dict[str, Any] | None = None,
) -> Callable[[], FuturesTradingEnv]:
    """Factory for creating FuturesTradingEnv instances (for VecEnv)."""
    def _init() -> FuturesTradingEnv:
        return FuturesTradingEnv(bars.copy(), config=env_config)
    return _init


@dataclass
class CurriculumTrainer:
    """Orchestrates multi-stage curriculum training.

    Parameters
    ----------
    bars_dict
        Mapping of instrument name → OHLCV ndarray.
        Single instrument: ``{"rb": bars_array}``.
    env_config
        Base config for FuturesTradingEnv.
    policy_kwargs
        PPO policy kwargs (e.g. MambaFormer extractor config).
    stages
        Curriculum stages.  Default: 4 stages (low_vol → full).
    save_dir
        Directory for saving checkpoints.
    """

    bars_dict: dict[str, np.ndarray]
    env_config: dict[str, Any] = field(default_factory=dict)
    policy_kwargs: dict[str, Any] = field(default_factory=dict)
    stages: list[CurriculumStage] = field(default_factory=lambda: DEFAULT_STAGES)
    save_dir: str = "models/rl_curriculum"
    ppo_kwargs: dict[str, Any] = field(default_factory=dict)
    fast_mode: bool = False

    def train(self) -> dict[str, Any]:
        """Run all curriculum stages sequentially."""
        if not _SB3_AVAILABLE:
            raise RuntimeError("stable-baselines3 required for curriculum training")

        save_path = Path(self.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        results: dict[str, Any] = {}
        model: PPO | None = None

        if self.fast_mode:
            from .mambaformer_extractor import MambaFormerExtractor
            self.policy_kwargs.setdefault("features_extractor_class", MambaFormerExtractor)
            fast_kw = self.policy_kwargs.setdefault("features_extractor_kwargs", {})
            fast_kw.setdefault("d_model", 32)
            fast_kw.setdefault("n_s4d_layers", 1)
            fast_kw.setdefault("n_attn_layers", 1)
            fast_kw.setdefault("d_state", 4)
            logger.info("Fast mode: d_model=32, n_s4d=1, d_state=4")

        for i, stage in enumerate(self.stages):
            logger.info("=== Stage %d/%d: %s ===", i + 1, len(self.stages), stage.name)

            envs = self._build_envs(stage)
            use_subproc = len(envs) > 2 and hasattr(sys.modules.get("__main__", None), "__file__")
            vec_env = SubprocVecEnv(envs) if use_subproc else DummyVecEnv(envs)

            try:
                if model is None:
                    default_ppo = dict(
                        n_steps=2048,
                        batch_size=64,
                        n_epochs=10,
                        learning_rate=3e-4,
                        gamma=0.99,
                        clip_range=0.2,
                        verbose=0,
                    )
                    default_ppo.update(self.ppo_kwargs)
                    model = PPO(
                        "MlpPolicy",
                        vec_env,
                        policy_kwargs=self.policy_kwargs,
                        **default_ppo,
                    )
                else:
                    model.set_env(vec_env)

                model.learn(
                    total_timesteps=stage.timesteps,
                    reset_num_timesteps=False,
                    progress_bar=False,
                )

                ckpt = save_path / f"stage_{i}_{stage.name}"
                model.save(str(ckpt))
                logger.info("Checkpoint saved: %s", ckpt)

                results[stage.name] = {
                    "timesteps": stage.timesteps,
                    "checkpoint": str(ckpt),
                }
            finally:
                vec_env.close()

        final_path = save_path / "final_model"
        if model is not None:
            model.save(str(final_path))
            results["final_model"] = str(final_path)

        return results

    def _build_envs(self, stage: CurriculumStage) -> list[Callable]:
        """Create one env factory per instrument, filtered by stage volatility."""
        factories = []
        for name, bars in self.bars_dict.items():
            filtered = filter_bars_by_volatility(
                bars, stage.volatility_range[0], stage.volatility_range[1]
            )
            if filtered.shape[0] < 100:
                logger.warning(
                    "Instrument %s has only %d bars for stage %s, skipping",
                    name, filtered.shape[0], stage.name,
                )
                continue
            factories.append(make_env(filtered, self.env_config))

        if not factories:
            logger.warning("No valid envs for stage %s, using full data", stage.name)
            for name, bars in self.bars_dict.items():
                factories.append(make_env(bars, self.env_config))

        return factories
