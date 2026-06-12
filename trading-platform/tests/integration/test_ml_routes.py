"""Integration tests for ``/api/v1/ml`` routes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]
for p in (
    _repo,
    _repo / "apps" / "api",
    _repo / "packages" / "core",
    _repo / "packages" / "backtest",
    _repo / "packages" / "security" / "src",
    _repo / "packages",
):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pytest
from starlette.testclient import TestClient

from tests.integration.route_harness import build_test_app


@pytest.fixture
def client() -> TestClient:
    app = build_test_app(routers=("ml",))
    return TestClient(app, raise_server_exceptions=False)


def test_list_models_empty_when_dir_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, client: TestClient
) -> None:
    import app.routers.ml as ml_mod

    monkeypatch.setattr(ml_mod, "MODEL_DIR", str(tmp_path / "no_models"), raising=False)
    r = client.get("/api/v1/ml/models")
    assert r.status_code == 200
    assert r.json() == []


def test_list_models_skips_report_sidecars(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, client: TestClient
) -> None:
    import app.routers.ml as ml_mod

    d = tmp_path / "mdl"
    d.mkdir()
    (d / "m1.json").write_text("{}")
    (d / "m1_report.json").write_text("{}")
    monkeypatch.setattr(ml_mod, "MODEL_DIR", str(d), raising=False)
    r = client.get("/api/v1/ml/models")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 1
    assert rows[0]["model_id"] == "m1"


def test_get_model_unknown_returns_404(client: TestClient) -> None:
    r = client.get("/api/v1/ml/models/does-not-exist-xyz")
    assert r.status_code == 404
    assert r.json()["error"] == "MODEL_NOT_FOUND"


def test_train_returns_503_when_ml_unavailable(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    import app.routers.ml as ml_mod

    monkeypatch.setattr(ml_mod, "_ML_IMPORT_ERROR", "forced ml import error", raising=False)
    r = client.post("/api/v1/ml/train", json={})
    assert r.status_code == 503
    body = r.json()
    assert body["error"] == "ML_UNAVAILABLE"
    assert "import_error" in body.get("detail", {})


def test_predict_returns_503_when_ml_unavailable(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    import app.routers.ml as ml_mod

    monkeypatch.setattr(ml_mod, "_ML_IMPORT_ERROR", "forced", raising=False)
    body = {"model_id": "x", "features": {c: 0.0 for c in ml_mod.FEATURE_COLUMNS}}
    r = client.post("/api/v1/ml/predict", json=body)
    assert r.status_code == 503


def test_train_validation_n_bars_too_small_returns_422(client: TestClient) -> None:
    r = client.post("/api/v1/ml/train", json={"n_bars": 10})
    assert r.status_code == 422


def test_train_validation_max_depth_out_of_range_returns_422(client: TestClient) -> None:
    r = client.post("/api/v1/ml/train", json={"max_depth": 99})
    assert r.status_code == 422


def test_get_model_returns_metadata_when_file_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, client: TestClient
) -> None:
    import app.routers.ml as ml_mod

    d = tmp_path / "store"
    d.mkdir()
    (d / "abc.json").write_text("{}")
    report = {"hyperparams": {"max_depth": 3}}
    (d / "abc_report.json").write_text(json.dumps(report))
    monkeypatch.setattr(ml_mod, "MODEL_DIR", str(d), raising=False)
    r = client.get("/api/v1/ml/models/abc")
    assert r.status_code == 200
    payload = r.json()
    assert payload["model_id"] == "abc"
    assert payload["report"] is not None
    assert payload["report"]["hyperparams"]["max_depth"] == 3


def test_list_models_sorted_by_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, client: TestClient
) -> None:
    import app.routers.ml as ml_mod

    d = tmp_path / "multi"
    d.mkdir()
    (d / "z.json").write_text("{}")
    (d / "a.json").write_text("{}")
    monkeypatch.setattr(ml_mod, "MODEL_DIR", str(d), raising=False)
    r = client.get("/api/v1/ml/models")
    ids = [row["model_id"] for row in r.json()]
    assert ids == ["a", "z"]


def test_list_models_json_content_type(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, client: TestClient
) -> None:
    import app.routers.ml as ml_mod

    monkeypatch.setattr(ml_mod, "MODEL_DIR", str(tmp_path / "empty_ml"), raising=False)
    r = client.get("/api/v1/ml/models")
    assert "application/json" in r.headers.get("content-type", "")


def test_get_model_unknown_includes_model_not_found_message(client: TestClient) -> None:
    r = client.get("/api/v1/ml/models/ghost-model-123")
    body = r.json()
    assert "ghost-model-123" in body.get("message", "")


def test_train_defaults_accept_empty_body_when_ml_unavailable(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    import app.routers.ml as ml_mod

    monkeypatch.setattr(ml_mod, "_ML_IMPORT_ERROR", "x", raising=False)
    r = client.post("/api/v1/ml/train", json={})
    assert r.status_code == 503
