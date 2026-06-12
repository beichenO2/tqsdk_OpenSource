"""Unit tests for FastAPI DI helpers in ``app.deps``."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

_repo = Path(__file__).resolve().parents[2]
for p in [_repo, _repo / "apps" / "api", _repo / "packages" / "core", _repo / "packages" / "security" / "src", _repo / "packages"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pytest

from core.exceptions import ServiceNotReadyError

import app.deps as deps


@pytest.fixture(autouse=True)
def _reset_deps_globals() -> None:
    deps.set_execution_service(None)  # type: ignore[arg-type]
    deps.set_market_adapter(None)
    deps.set_btc_broker_manager(None)
    deps.get_market_service.cache_clear()
    yield
    deps.set_execution_service(None)  # type: ignore[arg-type]
    deps.set_market_adapter(None)
    deps.set_btc_broker_manager(None)
    deps.get_market_service.cache_clear()


def test_get_execution_service_raises_when_not_set() -> None:
    with pytest.raises(ServiceNotReadyError) as exc_info:
        deps.get_execution_service()
    assert exc_info.value.code == "SERVICE_NOT_READY"
    assert "ExecutionService" in str(exc_info.value)


def test_set_get_execution_service_round_trip() -> None:
    svc = MagicMock()
    deps.set_execution_service(svc)
    assert deps.get_execution_service() is svc


def test_is_execution_service_ready_false_initially() -> None:
    assert deps.is_execution_service_ready() is False


def test_is_execution_service_ready_true_after_set() -> None:
    deps.set_execution_service(MagicMock())
    assert deps.is_execution_service_ready() is True


def test_is_execution_service_ready_false_after_reset() -> None:
    deps.set_execution_service(MagicMock())
    deps.set_execution_service(None)  # type: ignore[arg-type]
    assert deps.is_execution_service_ready() is False


def test_get_market_adapter_raises_when_not_set() -> None:
    with pytest.raises(ServiceNotReadyError) as exc_info:
        deps.get_market_adapter()
    assert exc_info.value.code == "SERVICE_NOT_READY"
    assert "TqMarketAdapter" in str(exc_info.value)


def test_set_get_market_adapter_round_trip() -> None:
    adapter = MagicMock()
    deps.set_market_adapter(adapter)
    assert deps.get_market_adapter() is adapter


def test_set_market_adapter_none_unsets_adapter() -> None:
    deps.set_market_adapter(MagicMock())
    deps.set_market_adapter(None)
    with pytest.raises(ServiceNotReadyError):
        deps.get_market_adapter()


def test_get_market_service_returns_market_service() -> None:
    from app.services.market import MarketService

    deps.set_market_adapter(MagicMock())
    deps.get_market_service.cache_clear()
    ms = deps.get_market_service()
    assert isinstance(ms, MarketService)


def test_get_market_service_raises_service_not_ready_without_adapter() -> None:
    deps.get_market_service.cache_clear()
    with pytest.raises(ServiceNotReadyError):
        deps.get_market_service()


def test_get_market_service_uses_current_adapter() -> None:
    a1 = MagicMock(name="a1")
    a2 = MagicMock(name="a2")
    deps.set_market_adapter(a1)
    deps.get_market_service.cache_clear()
    assert deps.get_market_service().adapter is a1
    deps.set_market_adapter(a2)
    deps.get_market_service.cache_clear()
    assert deps.get_market_service().adapter is a2


def test_get_btc_broker_manager_none_initially() -> None:
    assert deps.get_btc_broker_manager() is None


def test_set_get_btc_broker_manager_round_trip() -> None:
    mgr = MagicMock()
    deps.set_btc_broker_manager(mgr)
    assert deps.get_btc_broker_manager() is mgr


def test_set_btc_broker_manager_none_clears() -> None:
    deps.set_btc_broker_manager(MagicMock())
    deps.set_btc_broker_manager(None)
    assert deps.get_btc_broker_manager() is None


def test_service_not_ready_error_is_imported_from_core_exceptions() -> None:
    assert ServiceNotReadyError.__module__ == "core.exceptions"


def test_get_execution_service_returns_same_instance_twice() -> None:
    svc = MagicMock()
    deps.set_execution_service(svc)
    assert deps.get_execution_service() is deps.get_execution_service()


def test_get_market_adapter_returns_same_instance_twice() -> None:
    ad = MagicMock()
    deps.set_market_adapter(ad)
    assert deps.get_market_adapter() is deps.get_market_adapter()


def test_get_market_service_cached_until_cleared() -> None:
    deps.set_market_adapter(MagicMock())
    deps.get_market_service.cache_clear()
    m1 = deps.get_market_service()
    m2 = deps.get_market_service()
    assert m1 is m2


def test_execution_service_not_ready_has_status_503() -> None:
    with pytest.raises(ServiceNotReadyError) as ei:
        deps.get_execution_service()
    assert ei.value.status_code == 503


def test_market_adapter_not_ready_has_status_503() -> None:
    with pytest.raises(ServiceNotReadyError) as ei:
        deps.get_market_adapter()
    assert ei.value.status_code == 503


def test_get_market_service_not_ready_has_status_503() -> None:
    deps.get_market_service.cache_clear()
    with pytest.raises(ServiceNotReadyError) as ei:
        deps.get_market_service()
    assert ei.value.status_code == 503


def test_set_execution_service_round_trip_preserves_identity() -> None:
    svc = MagicMock(name="exec-svc")
    deps.set_execution_service(svc)
    assert deps.get_execution_service() is svc
    deps.set_execution_service(None)  # type: ignore[arg-type]


def test_set_market_adapter_round_trip_preserves_identity() -> None:
    adapter = MagicMock(name="adapter")
    deps.set_market_adapter(adapter)
    assert deps.get_market_adapter() is adapter


def test_get_market_service_after_adapter_configured_succeeds() -> None:
    deps.get_market_service.cache_clear()
    deps.set_market_adapter(MagicMock())
    deps.get_market_service.cache_clear()
    assert deps.get_market_service() is not None
