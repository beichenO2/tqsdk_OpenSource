"""Integration tests for MCP router — tool discovery, invocation, and auth."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "packages"))
sys.path.insert(0, str(REPO_ROOT / "apps" / "api"))

from app.routers import mcp


def _make_app(**env_overrides) -> FastAPI:
    app = FastAPI()
    app.include_router(mcp.router, prefix="/api/v1")
    return app


@pytest.fixture
def client():
    return TestClient(_make_app())


class TestToolDiscovery:
    def test_list_tools_returns_all(self, client):
        resp = client.get("/api/v1/mcp/tools")
        assert resp.status_code == 200
        data = resp.json()
        assert "tools" in data
        names = {t["name"] for t in data["tools"]}
        assert names >= {"list_strategies", "run_backtest", "get_run_status", "get_metrics"}

    def test_each_tool_has_input_schema(self, client):
        resp = client.get("/api/v1/mcp/tools")
        for tool in resp.json()["tools"]:
            assert "inputSchema" in tool, f"{tool['name']} missing inputSchema"
            assert tool["inputSchema"]["type"] == "object"


class TestListStrategies:
    def test_list_all(self, client):
        resp = client.post(
            "/api/v1/mcp/tools/call",
            json={"name": "list_strategies", "arguments": {"market": "all"}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "content" in data
        import json
        result = json.loads(data["content"][0]["text"])
        assert result["count"] > 0
        assert result["market_filter"] == "all"

    def test_filter_futures(self, client):
        resp = client.post(
            "/api/v1/mcp/tools/call",
            json={"name": "list_strategies", "arguments": {"market": "futures"}},
        )
        assert resp.status_code == 200
        import json
        result = json.loads(resp.json()["content"][0]["text"])
        for s in result["strategies"]:
            assert s["market"] == "futures"

    def test_filter_btc(self, client):
        resp = client.post(
            "/api/v1/mcp/tools/call",
            json={"name": "list_strategies", "arguments": {"market": "btc"}},
        )
        assert resp.status_code == 200
        import json
        result = json.loads(resp.json()["content"][0]["text"])
        for s in result["strategies"]:
            assert s["market"] == "btc"


class TestGetRunStatus:
    def test_missing_run_id(self, client):
        resp = client.post(
            "/api/v1/mcp/tools/call",
            json={"name": "get_run_status", "arguments": {}},
        )
        assert resp.status_code == 200
        import json
        result = json.loads(resp.json()["content"][0]["text"])
        assert "error" in result

    def test_nonexistent_run(self, client):
        resp = client.post(
            "/api/v1/mcp/tools/call",
            json={"name": "get_run_status", "arguments": {"run_id": "fake-id-123"}},
        )
        assert resp.status_code == 200
        import json
        result = json.loads(resp.json()["content"][0]["text"])
        assert "error" in result
        assert "not found" in result["error"]


class TestGetMetrics:
    def test_list_available(self, client):
        resp = client.post(
            "/api/v1/mcp/tools/call",
            json={"name": "get_metrics", "arguments": {}},
        )
        assert resp.status_code == 200
        import json
        result = json.loads(resp.json()["content"][0]["text"])
        assert "available" in result

    def test_nonexistent_strategy(self, client):
        resp = client.post(
            "/api/v1/mcp/tools/call",
            json={"name": "get_metrics", "arguments": {"strategy_name": "nonexistent_xyz"}},
        )
        assert resp.status_code == 200
        import json
        result = json.loads(resp.json()["content"][0]["text"])
        assert result.get("latest") is None


class TestUnknownTool:
    def test_unknown_tool_404(self, client):
        resp = client.post(
            "/api/v1/mcp/tools/call",
            json={"name": "nonexistent_tool", "arguments": {}},
        )
        assert resp.status_code == 404


class TestMCPAuth:
    def test_open_when_no_key_set(self):
        with patch.object(mcp, "_MCP_KEY", None):
            app = _make_app()
            c = TestClient(app)
            resp = c.get("/api/v1/mcp/tools")
            assert resp.status_code == 200

    def test_reject_when_key_set_no_header(self):
        with patch.object(mcp, "_MCP_KEY", "test-secret-key"):
            app = _make_app()
            c = TestClient(app)
            resp = c.get("/api/v1/mcp/tools")
            assert resp.status_code == 401

    def test_reject_wrong_key(self):
        with patch.object(mcp, "_MCP_KEY", "test-secret-key"):
            app = _make_app()
            c = TestClient(app)
            resp = c.get("/api/v1/mcp/tools", headers={"x-mcp-key": "wrong-key"})
            assert resp.status_code == 403

    def test_accept_correct_key(self):
        with patch.object(mcp, "_MCP_KEY", "test-secret-key"):
            app = _make_app()
            c = TestClient(app)
            resp = c.get("/api/v1/mcp/tools", headers={"x-mcp-key": "test-secret-key"})
            assert resp.status_code == 200
