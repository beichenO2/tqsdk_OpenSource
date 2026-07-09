"""Integration tests for research router — upload validation + settings local-auth."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "packages"))
sys.path.insert(0, str(REPO_ROOT / "apps" / "api"))


def _make_research_app() -> FastAPI:
    app = FastAPI()
    from app.routers import research
    app.include_router(research.router, prefix="/api/v1")
    return app


def _make_settings_app() -> FastAPI:
    app = FastAPI()
    from app.routers import settings
    app.include_router(settings.router, prefix="/api/v1")
    return app


class TestUploadValidation:
    @pytest.fixture
    def client(self):
        return TestClient(_make_research_app())

    def test_upload_valid_csv(self, client, tmp_path):
        resp = client.post(
            "/api/v1/research/upload",
            files={"file": ("test.csv", b"col1,col2\n1,2\n", "text/csv")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["filename"] == "test.csv"
        assert data["size_bytes"] == 14

    def test_upload_valid_json(self, client):
        resp = client.post(
            "/api/v1/research/upload",
            files={"file": ("data.json", b'{"key":"value"}', "application/json")},
        )
        assert resp.status_code == 200

    def test_reject_disallowed_extension(self, client):
        resp = client.post(
            "/api/v1/research/upload",
            files={"file": ("malware.exe", b"evil", "application/octet-stream")},
        )
        assert resp.status_code == 415
        assert "not allowed" in resp.json()["detail"]

    def test_reject_empty_file(self, client):
        resp = client.post(
            "/api/v1/research/upload",
            files={"file": ("empty.csv", b"", "text/csv")},
        )
        assert resp.status_code == 400
        assert "Empty" in resp.json()["detail"]

    def test_reject_oversized_file(self, client):
        from app.routers import research
        original = research._UPLOAD_MAX_BYTES
        try:
            research._UPLOAD_MAX_BYTES = 10
            resp = client.post(
                "/api/v1/research/upload",
                files={"file": ("big.csv", b"x" * 20, "text/csv")},
            )
            assert resp.status_code == 413
            assert "too large" in resp.json()["detail"].lower()
        finally:
            research._UPLOAD_MAX_BYTES = original

    def test_upload_raw_body(self, client):
        resp = client.post(
            "/api/v1/research/upload",
            content=b"raw data here",
            headers={"x-filename": "raw.txt", "content-type": "application/octet-stream"},
        )
        assert resp.status_code == 200
        assert resp.json()["filename"] == "raw.txt"


class TestSettingsLocalAuth:
    @pytest.fixture
    def client(self):
        from app.routers import settings
        settings._LOCAL_ADDRS.add("testclient")
        yield TestClient(_make_settings_app())
        settings._LOCAL_ADDRS.discard("testclient")

    def test_get_all_settings(self, client):
        resp = client.get("/api/v1/settings")
        assert resp.status_code == 200
        data = resp.json()
        assert "research" in data
        assert "mcp" in data

    def test_get_section(self, client):
        resp = client.get("/api/v1/settings/research")
        assert resp.status_code == 200
        assert "research" in resp.json()

    def test_get_unknown_section_404(self, client):
        resp = client.get("/api/v1/settings/nonexistent")
        assert resp.status_code == 404

    def test_patch_setting(self, client):
        resp = client.patch(
            "/api/v1/settings",
            json={"path": "research.default_timeframe", "value": "1h"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["new_value"] == "1h"

        check = client.get("/api/v1/settings/research")
        assert check.json()["research"]["default_timeframe"] == "1h"

    def test_reset_settings(self, client):
        client.patch(
            "/api/v1/settings",
            json={"path": "research.default_timeframe", "value": "99m"},
        )
        resp = client.post("/api/v1/settings/reset")
        assert resp.status_code == 200

        check = client.get("/api/v1/settings/research")
        assert check.json()["research"]["default_timeframe"] == "5m"

    def test_invalid_patch_path(self, client):
        resp = client.patch(
            "/api/v1/settings",
            json={"path": "nonexistent.deep.path", "value": "x"},
        )
        assert resp.status_code == 400
