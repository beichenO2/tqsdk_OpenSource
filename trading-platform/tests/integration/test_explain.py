"""Integration tests for evidence chain (explain) REST API."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from explain.chain import EvidenceChain, EvidenceEvent
from explain.views import (
    DecisionGraphView,
    DecisionNode,
    FactorContributionRow,
    FactorContributionView,
    TimelineEntry,
    TimelineView,
)


def _build_explain_app() -> FastAPI:
    from app.routers import explain
    from tests.integration.route_harness import register_platform_exception_handlers

    app = FastAPI()
    register_platform_exception_handlers(app)
    app.include_router(explain.router, prefix="/api/v1")
    return app


@pytest.fixture
def explain_client() -> TestClient:
    return TestClient(_build_explain_app())


MOCK_TIMELINE = TimelineView(
    trade_id="T001",
    symbol="IF2504",
    entries=[
        TimelineEntry(
            timestamp="2026-04-07T09:30:00+00:00",
            event_type="signal",
            data={"action": "BUY", "confidence": 0.85},
        ),
        TimelineEntry(
            timestamp="2026-04-07T09:30:01+00:00",
            event_type="risk_check",
            data={"passed": True, "margin_ok": True},
        ),
    ],
)


MOCK_FACTORS = FactorContributionView(
    trade_id="T001",
    symbol="IF2504",
    factors=[
        FactorContributionRow(factor="momentum", weight=0.45, source="signal"),
        FactorContributionRow(factor="volume", weight=0.30, source="signal"),
    ],
)


MOCK_GRAPH = DecisionGraphView(
    trade_id="T001",
    symbol="IF2504",
    root=DecisionNode(id="root", label="Trade T001", event_type="root"),
)

MOCK_CHAIN = EvidenceChain(
    trade_id="T001",
    symbol="IF2504",
    opened_at=datetime(2026, 4, 7, 9, 30, tzinfo=UTC),
    finalized_at=datetime(2026, 4, 7, 9, 35, tzinfo=UTC),
    events=[
        EvidenceEvent(
            timestamp=datetime(2026, 4, 7, 9, 30, tzinfo=UTC),
            event_type="signal",
            data={"action": "BUY"},
            metadata={},
        )
    ],
)


class TestTimelineEndpoint:
    @patch("app.routers.explain.timeline_view", new_callable=AsyncMock)
    @patch("app.routers.explain.get_session")
    def test_returns_timeline(self, mock_session, mock_view, explain_client):
        mock_view.return_value = MOCK_TIMELINE
        resp = explain_client.get("/api/v1/explain/timeline/T001")
        assert resp.status_code == 200
        body = resp.json()
        assert body["trade_id"] == "T001"
        assert len(body["entries"]) == 2

    @patch("app.routers.explain.timeline_view", new_callable=AsyncMock)
    @patch("app.routers.explain.get_session")
    def test_canonical_trade_path_returns_timeline(self, mock_session, mock_view, explain_client):
        mock_view.return_value = MOCK_TIMELINE
        resp = explain_client.get("/api/v1/explain/T001/timeline")
        assert resp.status_code == 200
        assert resp.json()["trade_id"] == "T001"

    @patch("app.routers.explain.timeline_view", new_callable=AsyncMock)
    @patch("app.routers.explain.get_session")
    def test_returns_404_when_not_found(self, mock_session, mock_view, explain_client):
        mock_view.return_value = None
        resp = explain_client.get("/api/v1/explain/timeline/MISSING")
        assert resp.status_code == 404


class TestFactorsEndpoint:
    @patch("app.routers.explain.factor_contribution_view", new_callable=AsyncMock)
    @patch("app.routers.explain.get_session")
    def test_returns_factors(self, mock_session, mock_view, explain_client):
        mock_view.return_value = MOCK_FACTORS
        resp = explain_client.get("/api/v1/explain/factors/T001")
        assert resp.status_code == 200
        assert len(resp.json()["factors"]) == 2

    @patch("app.routers.explain.factor_contribution_view", new_callable=AsyncMock)
    @patch("app.routers.explain.get_session")
    def test_returns_404_when_not_found(self, mock_session, mock_view, explain_client):
        mock_view.return_value = None
        resp = explain_client.get("/api/v1/explain/factors/MISSING")
        assert resp.status_code == 404


class TestGraphEndpoint:
    @patch("app.routers.explain.decision_graph_view", new_callable=AsyncMock)
    @patch("app.routers.explain.get_session")
    def test_returns_graph(self, mock_session, mock_view, explain_client):
        mock_view.return_value = MOCK_GRAPH
        resp = explain_client.get("/api/v1/explain/graph/T001")
        assert resp.status_code == 200
        assert resp.json()["root"]["event_type"] == "root"

    @patch("app.routers.explain.decision_graph_view", new_callable=AsyncMock)
    @patch("app.routers.explain.get_session")
    def test_returns_404_when_not_found(self, mock_session, mock_view, explain_client):
        mock_view.return_value = None
        resp = explain_client.get("/api/v1/explain/graph/MISSING")
        assert resp.status_code == 404


class TestListEndpoint:
    @patch("app.routers.explain.EvidenceStore.list_chains", new_callable=AsyncMock)
    @patch("app.routers.explain.get_session")
    def test_returns_timeline_list(self, mock_session, mock_list, explain_client):
        mock_list.return_value = [MOCK_CHAIN]
        resp = explain_client.get(
            "/api/v1/explain",
            params={
                "symbol": "IF2504",
                "start": "2026-04-07T09:00:00+00:00",
                "end": "2026-04-07T10:00:00+00:00",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 1
        assert body[0]["trade_id"] == "T001"
        assert body[0]["entries"][0]["event_type"] == "signal"
