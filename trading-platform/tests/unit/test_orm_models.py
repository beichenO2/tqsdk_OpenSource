"""Unit tests for ORM model definitions and Alembic migration validity."""

from __future__ import annotations

import ast
import importlib


def test_all_models_importable():
    """Verify all ORM model classes can be imported."""
    from core.db.models import (
        User, ApiKey,
        Instrument, Exchange,
        Strategy, StrategyVersion, StrategyParam,
        Order, Fill,
        Position, PositionSnapshot,
        RiskRule, RiskAlert,
        BacktestRun, BacktestTrade, BacktestMetric,
        MLModel, MLExperiment, MLModelVersion,
        EvidenceRecord, DecisionLog,
        DataSource, DataSubscription,
    )
    assert User.__tablename__ == "users"
    assert Order.__tablename__ == "orders"
    assert Strategy.__tablename__ == "strategies"
    assert BacktestRun.__tablename__ == "backtest_runs"
    assert EvidenceRecord.__tablename__ == "evidence_records"
    assert MLModel.__tablename__ == "ml_models"
    assert DataSource.__tablename__ == "data_sources"


def test_base_metadata_has_naming_convention():
    from core.db.base import Base
    nc = Base.metadata.naming_convention
    assert "ix" in nc
    assert "uq" in nc
    assert "fk" in nc


def test_timestamp_mixin_columns():
    from core.db.base import TimestampMixin
    assert hasattr(TimestampMixin, "created_at")
    assert hasattr(TimestampMixin, "updated_at")


def test_migration_script_syntax():
    """Verify the Alembic migration is syntactically correct Python."""
    with open("infra/migrations/versions/001_initial_schema.py") as f:
        tree = ast.parse(f.read())
    func_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    assert "upgrade" in func_names
    assert "downgrade" in func_names


def test_model_count():
    """Ensure __all__ has the expected number of models."""
    from core.db import models
    assert len(models.__all__) == 23
