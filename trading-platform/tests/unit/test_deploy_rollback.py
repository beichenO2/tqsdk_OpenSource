"""Unit tests for deploy router — deploy, rollback, history (no live trading needed)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def deploy_dirs(tmp_path: Path):
    """Patch deploy module paths to use tmp_path."""
    import app.routers.deploy as deploy_mod

    orig_params = deploy_mod.PARAMS_DIR
    orig_log = deploy_mod.DEPLOY_LOG

    deploy_mod.PARAMS_DIR = tmp_path / "params"
    deploy_mod.DEPLOY_LOG = tmp_path / "history.json"

    yield tmp_path

    deploy_mod.PARAMS_DIR = orig_params
    deploy_mod.DEPLOY_LOG = orig_log


@pytest.fixture()
def client(deploy_dirs: Path):
    from app.main import create_app

    app = create_app()
    return TestClient(app)


def test_deploy_and_get_params(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/deploy/params/test_strat",
        json={"params": {"fast": 5, "slow": 20}, "source": "test", "note": "initial"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "deployed"
    assert data["params"]["fast"] == 5

    resp = client.get("/api/v1/deploy/params/test_strat")
    assert resp.status_code == 200
    assert resp.json()["deployed"] is True
    assert resp.json()["params"]["slow"] == 20


def test_rollback_restores_old_params(client: TestClient) -> None:
    client.post(
        "/api/v1/deploy/params/rollback_strat",
        json={"params": {"v": 1}, "source": "v1"},
    )
    client.post(
        "/api/v1/deploy/params/rollback_strat",
        json={"params": {"v": 2}, "source": "v2"},
    )

    resp = client.get("/api/v1/deploy/params/rollback_strat")
    assert resp.json()["params"]["v"] == 2

    resp = client.post("/api/v1/deploy/rollback/rollback_strat")
    assert resp.status_code == 200
    assert resp.json()["status"] == "rolled_back"
    assert resp.json()["params"]["v"] == 1

    resp = client.get("/api/v1/deploy/params/rollback_strat")
    assert resp.json()["params"]["v"] == 1


def test_rollback_no_history_404(client: TestClient) -> None:
    resp = client.post("/api/v1/deploy/rollback/no_such_strat")
    assert resp.status_code == 404


def test_deploy_history(client: TestClient) -> None:
    client.post(
        "/api/v1/deploy/params/hist_strat",
        json={"params": {"a": 1}, "source": "a"},
    )
    client.post(
        "/api/v1/deploy/params/hist_strat",
        json={"params": {"a": 2}, "source": "b"},
    )

    resp = client.get("/api/v1/deploy/history?limit=10")
    assert resp.status_code == 200
    history = resp.json()
    assert len(history) >= 2
    assert history[0]["source"] == "b"


def test_get_nonexistent_strategy_params(client: TestClient) -> None:
    resp = client.get("/api/v1/deploy/params/nonexistent")
    assert resp.status_code == 200
    assert resp.json()["deployed"] is False
