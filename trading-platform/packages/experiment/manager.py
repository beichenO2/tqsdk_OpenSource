"""实验管理器 — 创建实验、运行策略回测、对比结果。

ExperimentManager 整合 StrategyRegistryCenter 和回测引擎，
提供：创建实验 → 配置参数 → 运行 → 收集结果 → 多实验对比 的完整流程。
"""

from __future__ import annotations

import enum
import logging
import uuid
from datetime import UTC, datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from .registry import StrategyRegistryCenter, StrategyStatus

logger = logging.getLogger(__name__)


class ExperimentStatus(str, enum.Enum):
    CREATED = "created"
    CONFIGURED = "configured"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExperimentConfig(BaseModel):
    """单次实验的配置。"""

    strategy_id: str
    strategy_params: dict[str, Any] = Field(default_factory=dict)
    symbols: list[str] = Field(default_factory=list)
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    initial_capital: float = 1_000_000.0
    benchmark: Optional[str] = None
    data_source: str = "default"
    extra: dict[str, Any] = Field(default_factory=dict)


class ExperimentResult(BaseModel):
    """单次实验的结果摘要。"""

    total_return: float = 0.0
    annualized_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    avg_holding_period: float = 0.0
    custom_metrics: dict[str, float] = Field(default_factory=dict)


class Experiment(BaseModel):
    """实验实体 — 一次策略回测/模拟运行的完整记录。"""

    experiment_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str
    description: str = ""
    config: ExperimentConfig
    status: ExperimentStatus = ExperimentStatus.CREATED
    result: Optional[ExperimentResult] = None
    error_message: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ComparisonReport(BaseModel):
    """多实验对比报告。"""

    report_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    experiment_ids: list[str]
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    rankings: dict[str, list[str]] = Field(
        default_factory=dict,
        description="指标名 → 按该指标排名的 experiment_id 列表（最优在前）",
    )
    summary_table: list[dict[str, Any]] = Field(
        default_factory=list,
        description="每行一个实验，列为各指标值",
    )


class ExperimentManager:
    """实验管理器 — 单例，管理实验的全生命周期。"""

    _instance: Optional[ExperimentManager] = None
    _experiments: dict[str, Experiment]

    def __new__(cls) -> ExperimentManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._experiments = {}
        return cls._instance

    def create(
        self,
        name: str,
        config: ExperimentConfig,
        description: str = "",
        tags: Optional[list[str]] = None,
    ) -> Experiment:
        """创建新实验。策略必须已在注册中心且状态为 APPROVED。"""
        registry = StrategyRegistryCenter()
        entry = registry.get(config.strategy_id)
        if entry is None:
            raise KeyError(f"策略 '{config.strategy_id}' 未在注册中心登记")
        if entry.status != StrategyStatus.APPROVED:
            raise ValueError(
                f"策略 '{entry.name}' 状态为 {entry.status.value}，"
                "只有 APPROVED 策略可用于实验"
            )

        experiment = Experiment(
            name=name,
            description=description,
            config=config,
            status=ExperimentStatus.CONFIGURED,
            tags=tags or [],
        )
        self._experiments[experiment.experiment_id] = experiment
        logger.info(
            "创建实验: %s (%s) 使用策略 %s",
            name, experiment.experiment_id, entry.name,
        )
        return experiment

    async def run(self, experiment_id: str) -> Experiment:
        """运行实验（调用回测引擎）。

        当前为框架实现，回测引擎集成后会调用
        packages/backtest 的 EventDrivenBacktester。
        """
        exp = self._get_or_raise(experiment_id)
        if exp.status not in (ExperimentStatus.CONFIGURED, ExperimentStatus.FAILED):
            raise ValueError(f"实验状态 {exp.status.value} 不可运行")

        exp = exp.model_copy(update={
            "status": ExperimentStatus.RUNNING,
            "started_at": datetime.now(UTC),
            "error_message": None,
        })
        self._experiments[experiment_id] = exp
        logger.info("开始运行实验: %s", experiment_id)

        try:
            result = await self._execute_backtest(exp)
            exp = exp.model_copy(update={
                "status": ExperimentStatus.COMPLETED,
                "result": result,
                "finished_at": datetime.now(UTC),
            })
            self._experiments[experiment_id] = exp

            registry = StrategyRegistryCenter()
            registry.update(
                exp.config.strategy_id,
                performance_summary=result.model_dump(),
            )
            logger.info("实验完成: %s (Sharpe=%.2f)", experiment_id, result.sharpe_ratio)
        except Exception as exc:
            exp = exp.model_copy(update={
                "status": ExperimentStatus.FAILED,
                "error_message": str(exc),
                "finished_at": datetime.now(UTC),
            })
            self._experiments[experiment_id] = exp
            logger.error("实验失败: %s — %s", experiment_id, exc)
            raise

        return exp

    def cancel(self, experiment_id: str) -> Experiment:
        """取消实验。"""
        exp = self._get_or_raise(experiment_id)
        if exp.status == ExperimentStatus.COMPLETED:
            raise ValueError("已完成的实验不可取消")
        exp = exp.model_copy(update={
            "status": ExperimentStatus.CANCELLED,
            "finished_at": datetime.now(UTC),
        })
        self._experiments[experiment_id] = exp
        logger.info("取消实验: %s", experiment_id)
        return exp

    def get(self, experiment_id: str) -> Optional[Experiment]:
        return self._experiments.get(experiment_id)

    def list_experiments(
        self,
        strategy_id: Optional[str] = None,
        status: Optional[ExperimentStatus] = None,
        tags: Optional[list[str]] = None,
    ) -> list[Experiment]:
        """按条件筛选实验。"""
        results = list(self._experiments.values())
        if strategy_id is not None:
            results = [e for e in results if e.config.strategy_id == strategy_id]
        if status is not None:
            results = [e for e in results if e.status == status]
        if tags:
            tag_set = set(tags)
            results = [e for e in results if tag_set.issubset(set(e.tags))]
        return sorted(results, key=lambda e: e.created_at, reverse=True)

    def compare(self, experiment_ids: list[str]) -> ComparisonReport:
        """对比多个已完成实验的结果。"""
        experiments: list[Experiment] = []
        for eid in experiment_ids:
            exp = self._get_or_raise(eid)
            if exp.status != ExperimentStatus.COMPLETED or exp.result is None:
                raise ValueError(f"实验 {eid} 未完成或无结果，无法对比")
            experiments.append(exp)

        if len(experiments) < 2:
            raise ValueError("对比至少需要 2 个已完成的实验")

        metric_keys = [
            "total_return", "annualized_return", "sharpe_ratio",
            "sortino_ratio", "max_drawdown", "win_rate",
            "profit_factor", "total_trades",
        ]

        summary_table: list[dict[str, Any]] = []
        for exp in experiments:
            assert exp.result is not None
            row: dict[str, Any] = {
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "strategy_id": exp.config.strategy_id,
            }
            for key in metric_keys:
                row[key] = getattr(exp.result, key)
            summary_table.append(row)

        higher_is_better = {
            "total_return", "annualized_return", "sharpe_ratio",
            "sortino_ratio", "win_rate", "profit_factor",
        }
        lower_is_better = {"max_drawdown"}

        rankings: dict[str, list[str]] = {}
        for key in metric_keys:
            reverse = key in higher_is_better
            if key in lower_is_better:
                reverse = False
            sorted_exps = sorted(
                experiments,
                key=lambda e: getattr(e.result, key),  # type: ignore[union-attr]
                reverse=reverse,
            )
            rankings[key] = [e.experiment_id for e in sorted_exps]

        return ComparisonReport(
            experiment_ids=experiment_ids,
            rankings=rankings,
            summary_table=summary_table,
        )

    def delete(self, experiment_id: str) -> bool:
        """删除实验记录。"""
        if experiment_id not in self._experiments:
            return False
        exp = self._experiments[experiment_id]
        if exp.status == ExperimentStatus.RUNNING:
            raise ValueError("运行中的实验不可删除")
        del self._experiments[experiment_id]
        logger.info("删除实验: %s", experiment_id)
        return True

    @property
    def count(self) -> int:
        return len(self._experiments)

    def _get_or_raise(self, experiment_id: str) -> Experiment:
        exp = self._experiments.get(experiment_id)
        if exp is None:
            raise KeyError(f"未找到实验: {experiment_id}")
        return exp

    async def _execute_backtest(self, exp: Experiment) -> ExperimentResult:
        """回测执行桩 — 回测引擎集成后替换为真实调用。

        TODO: 集成 packages/backtest EventDrivenBacktester
        """
        logger.warning(
            "使用桩实现运行实验 %s — 回测引擎集成后将产出真实结果",
            exp.experiment_id,
        )
        return ExperimentResult(
            total_return=0.0,
            annualized_return=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            total_trades=0,
        )

    @classmethod
    def _reset(cls) -> None:
        """仅用于测试——重置单例。"""
        cls._instance = None
