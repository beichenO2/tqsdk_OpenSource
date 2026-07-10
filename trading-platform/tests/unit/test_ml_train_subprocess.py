"""Unit tests for ML training subprocess isolation in the API layer."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.testclient import TestClient

_repo = Path(__file__).resolve().parents[2]
for p in (
    _repo,
    _repo / "apps" / "api",
    _repo / "packages" / "core",
    _repo / "packages",
):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tests.integration.route_harness import build_test_app

_TRAIN_RESPONSE_KEYS = {
    "model_id",
    "model_path",
    "report_path",
    "train_accuracy",
    "val_accuracy",
    "test_metrics",
    "feature_importance",
    "duration_seconds",
    "data_info",
}


def _sample_train_json() -> dict:
    return {
        "model_id": "xgb_20260710_120000",
        "model_path": "models/xgb_20260710_120000.json",
        "report_path": "models/xgb_20260710_120000_report.json",
        "train_accuracy": 0.91,
        "val_accuracy": 0.88,
        "test_metrics": {"accuracy": 0.87, "precision": 0.85, "recall": 0.84, "f1": 0.84},
        "feature_importance": {"open": 0.1, "close": 0.2},
        "duration_seconds": 1.23,
        "data_info": {
            "source": "parquet (2000 bars)",
            "total_samples": 1800,
            "train_size": 1080,
            "val_size": 360,
            "test_size": 360,
        },
    }


@pytest.fixture
def ml_client() -> TestClient:
    return TestClient(build_test_app(routers=("ml",)), raise_server_exceptions=False)


def test_train_endpoint_returns_compatible_response_via_subprocess(
    monkeypatch: pytest.MonkeyPatch, ml_client: TestClient
) -> None:
    import app.routers.ml as ml_mod

    payload = _sample_train_json()
    stdout = json.dumps(payload).encode()

    async def _fake_exec(*_args, **_kwargs):
        proc = MagicMock()
        proc.returncode = 0
        proc.communicate = AsyncMock(return_value=(stdout, b""))
        proc.kill = MagicMock()
        proc.wait = AsyncMock()
        return proc

    monkeypatch.setattr(ml_mod, "_ML_IMPORT_ERROR", None, raising=False)
    with patch.object(ml_mod.asyncio, "create_subprocess_exec", side_effect=_fake_exec):
        r = ml_client.post("/api/v1/ml/train", json={"n_bars": 2000})

    assert r.status_code == 200
    body = r.json()
    assert _TRAIN_RESPONSE_KEYS <= set(body.keys())
    assert body["model_id"] == payload["model_id"]
    assert body["train_accuracy"] == payload["train_accuracy"]
    assert body["test_metrics"]["accuracy"] == payload["test_metrics"]["accuracy"]


def test_train_subprocess_timeout_returns_504(
    monkeypatch: pytest.MonkeyPatch, ml_client: TestClient
) -> None:
    import app.routers.ml as ml_mod

    async def _slow_exec(*_args, **_kwargs):
        proc = MagicMock()
        proc.kill = MagicMock()
        proc.wait = AsyncMock()
        return proc

    async def _timeout_wait_for(coro, timeout=None):
        coro.close()
        raise asyncio.TimeoutError()

    monkeypatch.setattr(ml_mod, "_ML_IMPORT_ERROR", None, raising=False)
    monkeypatch.setattr(ml_mod, "ML_TRAIN_TIMEOUT_S", 0.01, raising=False)
    with (
        patch.object(ml_mod.asyncio, "create_subprocess_exec", side_effect=_slow_exec),
        patch.object(ml_mod.asyncio, "wait_for", side_effect=_timeout_wait_for),
    ):
        r = ml_client.post("/api/v1/ml/train", json={"n_bars": 2000})

    assert r.status_code == 504
    body = r.json()
    assert body["error"] == "ML_TRAIN_TIMEOUT"


def test_train_subprocess_nonzero_exit_returns_500_with_stderr(
    monkeypatch: pytest.MonkeyPatch, ml_client: TestClient
) -> None:
    import app.routers.ml as ml_mod

    stderr_tail = "E" * 2500

    async def _failing_exec(*_args, **_kwargs):
        proc = MagicMock()
        proc.returncode = 1
        proc.communicate = AsyncMock(return_value=(b"", stderr_tail.encode()))
        proc.kill = MagicMock()
        proc.wait = AsyncMock()
        return proc

    monkeypatch.setattr(ml_mod, "_ML_IMPORT_ERROR", None, raising=False)
    with patch.object(ml_mod.asyncio, "create_subprocess_exec", side_effect=_failing_exec):
        r = ml_client.post("/api/v1/ml/train", json={"n_bars": 2000})

    assert r.status_code == 500
    body = r.json()
    assert body["error"] == "ML_TRAIN_FAILED"
    assert "detail" in body
    assert len(body["detail"]["stderr"]) == 2000
    assert body["detail"]["stderr"] == stderr_tail[-2000:]


def test_worker_lightgbm_branch_does_not_import_torch() -> None:
    """Worker --framework lightgbm must lazy-load only LightGBM (no torch)."""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(_repo),
            str(_repo / "packages"),
            str(_repo / "packages" / "core"),
            env.get("PYTHONPATH", ""),
        ]
    )
    script = """
import sys
from apps.worker import train_ml

model_cls, ml_fw, prefix = train_ml._import_framework("lightgbm")
assert "torch" not in sys.modules, f"torch loaded: {[m for m in sys.modules if 'torch' in m]}"
assert model_cls.__name__ == "LightGBMModel"
assert prefix == "lgb"
print("WORKER_LGB_NO_TORCH_OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(_repo),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"stderr={result.stderr!r} stdout={result.stdout!r}"
    assert "WORKER_LGB_NO_TORCH_OK" in result.stdout


def test_worker_source_has_lazy_framework_imports() -> None:
    import re

    src = (_repo / "apps" / "worker" / "train_ml.py").read_text()
    assert "OpenMP" in src or "openmp" in src.lower()
    assert not re.search(r"^\s*import\s+lightgbm\b", src, re.MULTILINE)
    assert not re.search(r"^\s*import\s+torch\b", src, re.MULTILINE)
    assert not re.search(r"^\s*from\s+torch\b", src, re.MULTILINE)
    assert "--framework" in src
