"""PPO 智能体 — 基于 Stable-Baselines3 的 PPO 实现。"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import numpy as np

from .base import (
    BaseRLAgent,
    BaseTradingEnv,
    EvalResult,
    RLAgentMeta,
    RLAgentStatus,
    TrainConfig,
)

logger = logging.getLogger(__name__)

try:
    from stable_baselines3 import PPO as SB3PPO
    from stable_baselines3.common.base_class import BaseAlgorithm

    _SB3_AVAILABLE = True
    _SB3_IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover - 可选依赖环境
    _SB3_AVAILABLE = False
    _SB3_IMPORT_ERROR = exc
    SB3PPO = None  # type: ignore[misc, assignment]
    BaseAlgorithm = object  # type: ignore[misc, assignment]


def is_stable_baselines3_available() -> bool:
    """是否在环境中成功导入 stable-baselines3。"""
    return _SB3_AVAILABLE


def stable_baselines3_import_error() -> Exception | None:
    """若导入失败则返回异常对象，否则为 None。"""
    return _SB3_IMPORT_ERROR


class PPOAgent(BaseRLAgent):
    """封装 Stable-Baselines3 ``PPO`` 的交易智能体。

    训练、推理与持久化均依赖 SB3；若未安装相关包，构造仍可成功，
    但在调用 ``train`` / ``predict`` 等方法时会抛出明确的 ``RuntimeError``。
    """

    def __init__(self, meta: RLAgentMeta) -> None:
        super().__init__(meta)
        self._model: BaseAlgorithm | None = None

    def _require_sb3(self) -> None:
        if not _SB3_AVAILABLE or SB3PPO is None:
            msg = "未安装 stable-baselines3 或导入失败，无法使用 PPOAgent"
            if _SB3_IMPORT_ERROR is not None:
                msg = f"{msg}: {_SB3_IMPORT_ERROR}"
            raise RuntimeError(msg)

    def _require_model(self) -> Any:
        self._require_sb3()
        if self._model is None:
            raise RuntimeError("模型未初始化，请先训练或 load")
        return self._model

    def _ensure_gym_env(self, env: BaseTradingEnv) -> gym.Env:
        if not isinstance(env, gym.Env):
            raise TypeError(
                "PPOAgent 需要 Gymnasium 环境（例如 FuturesTradingEnv），"
                f"当前类型: {type(env).__name__}"
            )
        return cast(gym.Env, env)

    async def train(self, env: BaseTradingEnv, config: TrainConfig) -> EvalResult:
        """在线程池中执行 SB3 的 ``learn``，避免阻塞事件循环。"""
        self._require_sb3()
        gym_env = self._ensure_gym_env(env)

        self.meta.status = RLAgentStatus.TRAINING
        self.meta.hyperparams.update(
            {
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "n_epochs": config.n_epochs,
                "gamma": config.gamma,
                "clip_range": config.clip_range,
            }
        )

        extra = config.extra or {}
        tensorboard_log = extra.get("tensorboard_log")
        policy = extra.get("policy", "MlpPolicy")

        def _train_sync() -> None:
            assert SB3PPO is not None
            bs = max(int(config.batch_size), 1)
            target_rollout = max(2048, bs * 32)
            n_steps = ((target_rollout + bs - 1) // bs) * bs

            self._model = SB3PPO(
                policy,
                gym_env,
                learning_rate=config.learning_rate,
                n_steps=n_steps,
                batch_size=config.batch_size,
                n_epochs=config.n_epochs,
                gamma=config.gamma,
                clip_range=config.clip_range,
                tensorboard_log=tensorboard_log,
                verbose=extra.get("verbose", 0),
                seed=extra.get("seed"),
            )
            logger.info("PPO 开始训练，total_timesteps=%s", config.total_timesteps)
            self._model.learn(
                total_timesteps=config.total_timesteps,
                progress_bar=bool(extra.get("progress_bar", False)),
            )

        try:
            await asyncio.to_thread(_train_sync)
        except Exception:
            self.meta.status = RLAgentStatus.FAILED
            logger.exception("PPO 训练失败")
            raise

        self.meta.status = RLAgentStatus.TRAINED
        self.meta.trained_at = datetime.now(UTC)
        self.meta.total_timesteps = config.total_timesteps

        n_eval = int(extra.get("eval_episodes", config.eval_episodes))
        result = await asyncio.to_thread(self.evaluate, env, n_eval)
        self.meta.metrics.update(
            {
                "mean_reward": result.mean_reward,
                "sharpe_ratio": result.sharpe_ratio or 0.0,
                "max_drawdown": result.max_drawdown or 0.0,
            }
        )
        return result

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Any:
        model = self._require_model()
        obs = np.asarray(observation, dtype=np.float32)
        action, _states = model.predict(obs, deterministic=deterministic)
        return action

    def evaluate(self, env: BaseTradingEnv, n_episodes: int = 10) -> EvalResult:
        """在环境中滚动若干回合，汇总奖励与权益衍生指标。"""
        model = self._require_model()
        gym_env = self._ensure_gym_env(env)

        initial_balance = float(getattr(gym_env, "config", {}).get("initial_balance", 100_000.0))

        episode_rewards: list[float] = []
        episode_lengths: list[int] = []
        episode_returns: list[float] = []
        all_simple_returns: list[float] = []
        episode_max_dds: list[float] = []

        for ep in range(n_episodes):
            obs, info = gym_env.reset()
            done = False
            total_r = 0.0
            steps = 0
            equities: list[float] = [float(info.get("equity", initial_balance))]

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = gym_env.step(action)
                total_r += float(reward)
                steps += 1
                eq = float(info.get("equity", equities[-1]))
                prev = equities[-1]
                if prev > 1e-12:
                    all_simple_returns.append((eq - prev) / prev)
                equities.append(eq)
                done = bool(terminated or truncated)

            episode_rewards.append(total_r)
            episode_lengths.append(steps)
            if equities:
                episode_returns.append((equities[-1] - initial_balance) / max(initial_balance, 1e-8))
            episode_max_dds.append(_max_drawdown_ratio(equities))

        mean_r = float(np.mean(episode_rewards)) if episode_rewards else 0.0
        std_r = float(np.std(episode_rewards)) if episode_rewards else 0.0
        mean_len = float(np.mean(episode_lengths)) if episode_lengths else 0.0

        sharpe: float | None = None
        if len(all_simple_returns) > 1:
            arr = np.array(all_simple_returns, dtype=np.float64)
            mu = float(np.mean(arr))
            sig = float(np.std(arr, ddof=1))
            sharpe = float(mu / (sig + 1e-12) * np.sqrt(len(arr)))

        max_dd = float(np.min(episode_max_dds)) if episode_max_dds else None
        mean_total_ret = float(np.mean(episode_returns)) if episode_returns else None

        return EvalResult(
            mean_reward=mean_r,
            std_reward=std_r,
            mean_episode_length=mean_len,
            total_episodes=n_episodes,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            total_return=mean_total_ret,
        )

    def save(self, path: str) -> str:
        model = self._require_model()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(p))
        self.meta.artifact_path = str(p)
        logger.info("PPO 模型已保存: %s", p)
        return str(p)

    def load(self, path: str) -> None:
        self._require_sb3()
        assert SB3PPO is not None
        self._model = SB3PPO.load(path)
        self.meta.artifact_path = path
        self.meta.status = RLAgentStatus.TRAINED
        logger.info("PPO 模型已加载: %s", path)


def _max_drawdown_ratio(equity_series: list[float]) -> float:
    """给定权益序列，返回最大回撤比例（非正数，例如 -0.2 表示 -20%）。"""
    if not equity_series:
        return 0.0
    peak = equity_series[0]
    worst = 0.0
    for eq in equity_series:
        if eq > peak:
            peak = eq
        if peak > 1e-12:
            dd = (eq - peak) / peak
            if dd < worst:
                worst = dd
    return float(worst)
