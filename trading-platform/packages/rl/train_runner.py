"""RL 训练编排器 — 从真实数据加载到 PPO Agent 训练的完整流程。

用法:
    runner = RLTrainRunner(instrument="rb", timeframe="5m")
    result = await runner.run()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from datahub.futures_loader import FuturesDataLoader

from .base import RLAgentMeta, TrainConfig
from .ppo_agent import PPOAgent
from .trading_env import FuturesTradingEnv

logger = logging.getLogger(__name__)


@dataclass
class RLTrainConfig:
    instrument: str = "rb"
    timeframe: str = "5m"
    start_date: str | None = "2024-01-01"
    end_date: str | None = "2024-06-30"
    eval_start_date: str | None = "2024-07-01"
    eval_end_date: str | None = "2024-12-31"
    total_timesteps: int = 50_000
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    clip_range: float = 0.2
    window_size: int = 30
    initial_balance: float = 100_000.0
    save_dir: str = "models"
    cache_dir: str | None = ".cache/bars"
    eval_episodes: int = 5


@dataclass
class RLTrainReport:
    instrument: str
    timeframe: str
    train_bars: int = 0
    eval_bars: int = 0
    mean_reward: float = 0.0
    total_return: float | None = None
    sharpe_ratio: float | None = None
    max_drawdown: float | None = None
    model_path: str = ""


class RLTrainRunner:
    """Orchestrates PPO training with real futures data."""

    def __init__(self, config: RLTrainConfig | None = None, **kwargs: Any):
        self.config = config or RLTrainConfig(**kwargs)
        self.loader = FuturesDataLoader()

    def _load_bars_array(
        self, start: str | None, end: str | None
    ) -> np.ndarray:
        cfg = self.config
        bars = self.loader.load_main_contract_bars(
            cfg.instrument, cfg.timeframe, start, end, cache_dir=cfg.cache_dir,
        )
        if bars.empty:
            raise ValueError(f"No data for {cfg.instrument} {start}→{end}")

        ohlcv = bars[["open", "high", "low", "close", "volume"]].values.astype(np.float64)
        return ohlcv

    async def run(self) -> RLTrainReport:
        cfg = self.config
        report = RLTrainReport(instrument=cfg.instrument, timeframe=cfg.timeframe)

        logger.info("=== RL Train: %s %s ===", cfg.instrument, cfg.timeframe)

        train_bars = self._load_bars_array(cfg.start_date, cfg.end_date)
        report.train_bars = len(train_bars)
        logger.info("Train bars: %d", len(train_bars))

        eval_bars = self._load_bars_array(cfg.eval_start_date, cfg.eval_end_date)
        report.eval_bars = len(eval_bars)
        logger.info("Eval bars: %d", len(eval_bars))

        train_env = FuturesTradingEnv(train_bars, {
            "window_size": cfg.window_size,
            "initial_balance": cfg.initial_balance,
        })

        meta = RLAgentMeta(
            agent_id=f"ppo_{cfg.instrument}_{cfg.timeframe}",
            name=f"PPO {cfg.instrument}",
            algorithm="PPO",
        )
        agent = PPOAgent(meta)

        train_config = TrainConfig(
            total_timesteps=cfg.total_timesteps,
            learning_rate=cfg.learning_rate,
            batch_size=cfg.batch_size,
            n_epochs=cfg.n_epochs,
            gamma=cfg.gamma,
            clip_range=cfg.clip_range,
            eval_episodes=cfg.eval_episodes,
        )

        logger.info("Training PPO for %d timesteps...", cfg.total_timesteps)
        eval_result = await agent.train(train_env, train_config)

        report.mean_reward = eval_result.mean_reward
        report.total_return = eval_result.total_return
        report.sharpe_ratio = eval_result.sharpe_ratio
        report.max_drawdown = eval_result.max_drawdown

        logger.info(
            "Train eval: reward=%.2f return=%s sharpe=%s dd=%s",
            eval_result.mean_reward,
            f"{eval_result.total_return:.4f}" if eval_result.total_return else "N/A",
            f"{eval_result.sharpe_ratio:.4f}" if eval_result.sharpe_ratio else "N/A",
            f"{eval_result.max_drawdown:.4f}" if eval_result.max_drawdown else "N/A",
        )

        # Evaluate on out-of-sample data
        eval_env = FuturesTradingEnv(eval_bars, {
            "window_size": cfg.window_size,
            "initial_balance": cfg.initial_balance,
        })
        oos_result = agent.evaluate(eval_env, n_episodes=cfg.eval_episodes)
        logger.info(
            "OOS eval: reward=%.2f return=%s sharpe=%s dd=%s",
            oos_result.mean_reward,
            f"{oos_result.total_return:.4f}" if oos_result.total_return else "N/A",
            f"{oos_result.sharpe_ratio:.4f}" if oos_result.sharpe_ratio else "N/A",
            f"{oos_result.max_drawdown:.4f}" if oos_result.max_drawdown else "N/A",
        )

        save_dir = Path(cfg.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        model_path = str(save_dir / f"ppo_{cfg.instrument}_{cfg.timeframe}")
        agent.save(model_path)
        report.model_path = model_path

        logger.info("=== RL training complete, model saved to %s ===", model_path)
        return report
