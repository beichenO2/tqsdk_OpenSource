"""Compatibility shim for the canonical FastAPI app entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_repo_import_paths() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    extra_paths = (
        repo_root,
        repo_root / "apps" / "api",
        repo_root / "packages" / "core",
        repo_root / "packages" / "backtest",
        repo_root / "packages" / "broker_tqsdk",
        repo_root / "packages" / "risk",
        repo_root / "packages" / "sim_live",
        repo_root / "packages",
    )
    for path in reversed(extra_paths):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


_bootstrap_repo_import_paths()

from app.main import app, create_app

__all__ = ["app", "create_app"]


if __name__ == "__main__":
    import os
    import uvicorn

    preferred_port = int(os.environ.get("TQTRADER_PORT", "8000"))
    host = os.environ.get("TQTRADER_HOST", "127.0.0.1")

    port = preferred_port
    try:
        sdk_path = str(Path(__file__).resolve().parents[4] / "PolarPort" / "src" / "sdk" / "python")
        sys.path.insert(0, sdk_path)
        from polarisor_port_sdk import claim_port_sync, register_capabilities_sync
        port = claim_port_sync(service="tqtrader-api", project="tqsdk", preferred=preferred_port)
        cap_path = str(Path(__file__).resolve().parents[2] / "capabilities.json")
        if os.path.exists(cap_path):
            try:
                register_capabilities_sync(cap_path)
            except Exception as e:
                print(f"[tqtrader] capability registration failed (non-fatal): {e}")
    except Exception as e:
        print(f"[tqtrader] port-sdk claim failed (non-fatal): {e}, using port {port}")

    uvicorn.run(app, host=host, port=port)
