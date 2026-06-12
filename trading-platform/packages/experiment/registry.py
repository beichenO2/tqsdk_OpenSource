"""策略注册中心 — 策略全生命周期治理与审批。

与 packages/strategy/registry.py (本地策略类注册) 不同，
本模块管理策略的元信息、审批流程、家族分类和来源追踪。
策略从 draft → pending_review → approved/rejected，只有 approved 的策略
才能被 ExperimentManager 用于实验。
"""

from __future__ import annotations

import enum
import logging
import uuid
from datetime import UTC, datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class StrategyFamily(str, enum.Enum):
    """策略家族分类。"""
    RULE_BASED = "rule_based"
    MANUAL = "manual"
    SEMI_AUTO = "semi_auto"
    ML = "ml"
    DL = "dl"
    RL = "rl"
    HYBRID = "hybrid"
    ENSEMBLE = "ensemble"


class StrategySource(str, enum.Enum):
    """策略来源。"""
    BUILTIN = "builtin"
    USER_DEFINED = "user_defined"
    ML_TRAINED = "ml_trained"
    RL_TRAINED = "rl_trained"
    IMPORTED = "imported"


class StrategyStatus(str, enum.Enum):
    """策略审批状态。"""
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    DEPRECATED = "deprecated"
    SUSPENDED = "suspended"


class StrategyEntry(BaseModel):
    """策略注册条目 — 策略元信息的完整记录。"""

    strategy_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str
    family: StrategyFamily
    source: StrategySource
    status: StrategyStatus = StrategyStatus.DRAFT
    reject_reason: Optional[str] = None
    description: str = ""
    version: str = "1.0.0"
    author: str = ""
    tags: list[str] = Field(default_factory=list)
    config_schema: dict[str, Any] = Field(
        default_factory=dict,
        description="策略可接受的参数 JSON Schema",
    )
    performance_summary: dict[str, Any] = Field(
        default_factory=dict,
        description="最近一次回测/实验的绩效摘要",
    )
    registered_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class StrategyRegistryCenter:
    """策略注册中心 — 单例，管理所有策略条目的生命周期。"""

    _instance: Optional[StrategyRegistryCenter] = None
    _entries: dict[str, StrategyEntry]

    def __new__(cls) -> StrategyRegistryCenter:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._entries = {}
        return cls._instance

    def register(self, entry: StrategyEntry) -> StrategyEntry:
        """注册新策略条目。"""
        if entry.strategy_id in self._entries:
            raise ValueError(f"策略 '{entry.strategy_id}' 已存在，使用 update() 修改")
        self._entries[entry.strategy_id] = entry
        logger.info(
            "注册策略: %s (%s) [%s/%s]",
            entry.name, entry.strategy_id, entry.family.value, entry.source.value,
        )
        return entry

    def update(self, strategy_id: str, **kwargs: Any) -> StrategyEntry:
        """更新策略条目字段。"""
        entry = self._get_or_raise(strategy_id)
        kwargs["updated_at"] = datetime.now(UTC)
        updated = entry.model_copy(update=kwargs)
        self._entries[strategy_id] = updated
        logger.info("更新策略: %s -> %s", strategy_id, list(kwargs.keys()))
        return updated

    def submit_for_review(self, strategy_id: str) -> StrategyEntry:
        """将 draft 策略提交审核。"""
        entry = self._get_or_raise(strategy_id)
        if entry.status != StrategyStatus.DRAFT:
            raise ValueError(f"只有 DRAFT 状态可以提交审核，当前: {entry.status.value}")
        return self.update(strategy_id, status=StrategyStatus.PENDING_REVIEW, reject_reason=None)

    def approve(self, strategy_id: str) -> StrategyEntry:
        """审批通过策略。"""
        entry = self._get_or_raise(strategy_id)
        if entry.status != StrategyStatus.PENDING_REVIEW:
            raise ValueError(f"只有 PENDING_REVIEW 可以审批，当前: {entry.status.value}")
        return self.update(strategy_id, status=StrategyStatus.APPROVED, reject_reason=None)

    def reject(self, strategy_id: str, reason: str) -> StrategyEntry:
        """驳回策略，附上原因。被驳回后可修改后重新提交。"""
        entry = self._get_or_raise(strategy_id)
        if entry.status != StrategyStatus.PENDING_REVIEW:
            raise ValueError(f"只有 PENDING_REVIEW 可以驳回，当前: {entry.status.value}")
        return self.update(
            strategy_id,
            status=StrategyStatus.REJECTED,
            reject_reason=reason,
        )

    def resubmit(self, strategy_id: str) -> StrategyEntry:
        """被驳回的策略修改后重新提交审核。"""
        entry = self._get_or_raise(strategy_id)
        if entry.status != StrategyStatus.REJECTED:
            raise ValueError(f"只有 REJECTED 可以重新提交，当前: {entry.status.value}")
        return self.update(strategy_id, status=StrategyStatus.PENDING_REVIEW, reject_reason=None)

    def deprecate(self, strategy_id: str) -> StrategyEntry:
        """将策略标记为已废弃。"""
        entry = self._get_or_raise(strategy_id)
        if entry.status not in (StrategyStatus.APPROVED, StrategyStatus.SUSPENDED):
            raise ValueError(f"只有 APPROVED/SUSPENDED 可以废弃，当前: {entry.status.value}")
        return self.update(strategy_id, status=StrategyStatus.DEPRECATED)

    def suspend(self, strategy_id: str, reason: str = "") -> StrategyEntry:
        """暂停已批准的策略（出现问题时紧急使用）。"""
        entry = self._get_or_raise(strategy_id)
        if entry.status != StrategyStatus.APPROVED:
            raise ValueError(f"只有 APPROVED 可以暂停，当前: {entry.status.value}")
        metadata = {**entry.metadata, "suspend_reason": reason}
        return self.update(strategy_id, status=StrategyStatus.SUSPENDED, metadata=metadata)

    def get(self, strategy_id: str) -> Optional[StrategyEntry]:
        return self._entries.get(strategy_id)

    def list_entries(
        self,
        family: Optional[StrategyFamily] = None,
        source: Optional[StrategySource] = None,
        status: Optional[StrategyStatus] = None,
        tags: Optional[list[str]] = None,
    ) -> list[StrategyEntry]:
        """按条件筛选策略条目。"""
        results = list(self._entries.values())
        if family is not None:
            results = [e for e in results if e.family == family]
        if source is not None:
            results = [e for e in results if e.source == source]
        if status is not None:
            results = [e for e in results if e.status == status]
        if tags:
            tag_set = set(tags)
            results = [e for e in results if tag_set.issubset(set(e.tags))]
        return sorted(results, key=lambda e: e.registered_at, reverse=True)

    def list_approved(self) -> list[StrategyEntry]:
        """仅列出已审批通过、可用于实验的策略。"""
        return self.list_entries(status=StrategyStatus.APPROVED)

    def remove(self, strategy_id: str) -> bool:
        """移除策略条目（仅允许 DRAFT/REJECTED）。"""
        entry = self._entries.get(strategy_id)
        if entry is None:
            return False
        if entry.status not in (StrategyStatus.DRAFT, StrategyStatus.REJECTED):
            raise ValueError(f"只有 DRAFT/REJECTED 可以删除，当前: {entry.status.value}")
        del self._entries[strategy_id]
        logger.info("删除策略: %s (%s)", entry.name, strategy_id)
        return True

    @property
    def count(self) -> int:
        return len(self._entries)

    def summary(self) -> dict[str, int]:
        """各状态的策略数量统计。"""
        counts: dict[str, int] = {}
        for entry in self._entries.values():
            counts[entry.status.value] = counts.get(entry.status.value, 0) + 1
        return counts

    def _get_or_raise(self, strategy_id: str) -> StrategyEntry:
        entry = self._entries.get(strategy_id)
        if entry is None:
            raise KeyError(f"未找到策略: {strategy_id}")
        return entry

    @classmethod
    def _reset(cls) -> None:
        """仅用于测试——重置单例。"""
        cls._instance = None
