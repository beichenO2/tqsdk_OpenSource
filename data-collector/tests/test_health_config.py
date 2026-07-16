"""Runtime-governance tests for the collector health listener."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import health  # noqa: E402


def test_health_port_comes_from_polarport_runtime_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TQSDK_COLLECTOR_PORT", "19005")
    assert hasattr(health, "resolve_health_port"), "health listener must expose runtime port resolution"
    assert health.resolve_health_port() == 19005


def test_health_port_rejects_invalid_runtime_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TQSDK_COLLECTOR_PORT", "not-a-port")
    assert hasattr(health, "resolve_health_port"), "health listener must expose runtime port resolution"
    with pytest.raises(ValueError, match="TQSDK_COLLECTOR_PORT"):
        health.resolve_health_port()
