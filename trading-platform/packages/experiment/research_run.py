"""Research Run data structure for the strategy research workbench.

A research run captures the full lifecycle of a strategy research session:
  prompt → config → code generation → backtest → diagnostics → validation → iteration.

Runs are stored as JSON files in the research/ output directory.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DiagnosticCategory(str, Enum):
    RUNTIME = "runtime"
    LOGIC = "logic"
    DATA = "data"


@dataclass
class DiagnosticEntry:
    category: DiagnosticCategory
    code: str
    message: str
    severity: str = "warning"
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    gate: str
    passed: bool
    metrics: dict[str, float] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)


@dataclass
class IterationRecord:
    iteration: int
    timestamp: float = field(default_factory=time.time)
    prompt: str = ""
    changes: str = ""
    metrics_before: dict[str, float] = field(default_factory=dict)
    metrics_after: dict[str, float] = field(default_factory=dict)
    diagnostics: list[dict] = field(default_factory=list)


@dataclass
class ResearchArtifact:
    """Exportable research artifact for KnowLever/AutoOffice consumption."""
    run_id: str
    variant: str
    data_range: str
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    risk: dict[str, float] = field(default_factory=dict)
    paths: dict[str, str] = field(default_factory=dict)
    decision: str = ""


@dataclass
class ResearchRun:
    """A strategy research run with full provenance."""

    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    status: RunStatus = RunStatus.PENDING

    prompt: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    generated_code: str = ""
    strategy_name: str = ""

    symbols: list[str] = field(default_factory=list)
    timeframe: str = "5m"
    data_range: str = ""

    backtest_results: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    diagnostics: list[dict] = field(default_factory=list)
    validation: list[dict] = field(default_factory=list)
    iterations: list[dict] = field(default_factory=list)

    artifact: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> ResearchRun:
        data = dict(data)
        if "status" in data:
            data["status"] = RunStatus(data["status"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def add_diagnostic(self, category: str, code: str, message: str,
                       severity: str = "warning", **ctx: Any) -> None:
        self.diagnostics.append({
            "category": category, "code": code, "message": message,
            "severity": severity, "context": ctx,
            "timestamp": time.time(),
        })
        self.updated_at = time.time()

    def add_iteration(self, prompt: str, changes: str,
                      before: dict | None = None, after: dict | None = None) -> None:
        self.iterations.append({
            "iteration": len(self.iterations) + 1,
            "timestamp": time.time(),
            "prompt": prompt, "changes": changes,
            "metrics_before": before or {},
            "metrics_after": after or {},
        })
        self.updated_at = time.time()

    def add_validation(self, gate: str, passed: bool,
                       metrics: dict | None = None, thresholds: dict | None = None) -> None:
        self.validation.append({
            "gate": gate, "passed": passed,
            "metrics": metrics or {}, "thresholds": thresholds or {},
            "timestamp": time.time(),
        })
        self.updated_at = time.time()

    def build_artifact(self) -> ResearchArtifact:
        return ResearchArtifact(
            run_id=self.run_id,
            variant=self.strategy_name,
            data_range=self.data_range,
            params=self.config,
            metrics=self.metrics,
            validation={"gates": self.validation, "all_passed": all(v.get("passed") for v in self.validation)},
            risk={k: v for k, v in self.metrics.items() if k in ("max_dd", "sharpe", "calmar", "var_95")},
            paths={},
            decision="approved" if all(v.get("passed") for v in self.validation) else "rejected",
        )


class RunStore:
    """File-based store for research runs."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, run_id: str) -> Path:
        return self.base_dir / f"{run_id}.json"

    def save(self, run: ResearchRun) -> None:
        run.updated_at = time.time()
        with open(self._path(run.run_id), "w") as f:
            json.dump(run.to_dict(), f, indent=2, default=str)

    def load(self, run_id: str) -> ResearchRun | None:
        p = self._path(run_id)
        if not p.exists():
            return None
        with open(p) as f:
            return ResearchRun.from_dict(json.load(f))

    def list_runs(self, limit: int = 50) -> list[dict]:
        runs = []
        for p in sorted(self.base_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                with open(p) as f:
                    data = json.load(f)
                runs.append({
                    "run_id": data["run_id"],
                    "status": data.get("status", "unknown"),
                    "strategy_name": data.get("strategy_name", ""),
                    "created_at": data.get("created_at", 0),
                    "prompt": data.get("prompt", "")[:100],
                    "metrics": {k: data.get("metrics", {}).get(k) for k in ("sharpe", "total_return", "max_dd")},
                })
            except Exception:
                continue
            if len(runs) >= limit:
                break
        return runs

    def delete(self, run_id: str) -> bool:
        p = self._path(run_id)
        if p.exists():
            p.unlink()
            return True
        return False
