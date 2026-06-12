"""RL 环境与智能体基础接口 — 强化学习组件的抽象基类。

提供统一的 TradingEnv（Gymnasium 兼容）和 BaseRLAgent 接口，
支持 Stable-Baselines3、自定义 PPO/A2C/SAC 等实现。
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, Field


class RLAlgorithm(str, enum.Enum):
    PPO = "ppo"
    A2C = "a2c"
    SAC = "sac"
    TD3 = "td3"
    DQN = "dqn"
    CUSTOM = "custom"


class RLAgentStatus(str, enum.Enum):
    UNTRAINED = "untrained"
    TRAINING = "training"
    TRAINED = "trained"
    EVALUATING = "evaluating"
    DEPLOYED = "deployed"
    FAILED = "failed"


class RLAgentMeta(BaseModel):
    """RL 智能体元信息。"""

    agent_id: str
    name: str
    algorithm: RLAlgorithm
    version: str = "1.0.0"
    status: RLAgentStatus = RLAgentStatus.UNTRAINED
    env_id: str = ""
    hyperparams: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    artifact_path: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    trained_at: Optional[datetime] = None
    total_timesteps: int = 0


class TrainConfig(BaseModel):
    """RL 训练配置。"""

    total_timesteps: int = 100_000
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    clip_range: float = 0.2
    eval_freq: int = 10_000
    eval_episodes: int = 5
    extra: dict[str, Any] = Field(default_factory=dict)


class EvalResult(BaseModel):
    """RL 评估结果。"""

    mean_reward: float = 0.0
    std_reward: float = 0.0
    mean_episode_length: float = 0.0
    total_episodes: int = 0
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    total_return: Optional[float] = None
    custom_metrics: dict[str, float] = Field(default_factory=dict)


class BaseTradingEnv(ABC):
    """交易环境抽象基类 — Gymnasium 兼容接口。

    子类需要实现 reset / step / _get_observation / _calculate_reward。
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """重置环境，返回 (observation, info)。"""
        ...

    @abstractmethod
    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """执行一步动作，返回 (obs, reward, terminated, truncated, info)。"""
        ...

    @abstractmethod
    def _get_observation(self) -> np.ndarray:
        """构建当前观测向量。"""
        ...

    @abstractmethod
    def _calculate_reward(self, action: Any) -> float:
        """根据动作计算奖励。"""
        ...

    def close(self) -> None:
        """释放环境资源。"""


class BaseRLAgent(ABC):
    """RL 智能体抽象基类。"""

    def __init__(self, meta: RLAgentMeta) -> None:
        self.meta = meta
        self._agent: Any = None

    @property
    def agent_id(self) -> str:
        return self.meta.agent_id

    @property
    def is_trained(self) -> bool:
        return self.meta.status in (RLAgentStatus.TRAINED, RLAgentStatus.DEPLOYED)

    @abstractmethod
    async def train(self, env: BaseTradingEnv, config: TrainConfig) -> EvalResult:
        """训练智能体。"""
        ...

    @abstractmethod
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Any:
        """根据观测给出动作。"""
        ...

    @abstractmethod
    def evaluate(self, env: BaseTradingEnv, n_episodes: int = 10) -> EvalResult:
        """评估智能体表现。"""
        ...

    @abstractmethod
    def save(self, path: str) -> str:
        """保存智能体到磁盘。"""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """从磁盘加载智能体。"""
        ...
