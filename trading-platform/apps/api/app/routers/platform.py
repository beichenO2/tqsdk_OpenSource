"""平台数据 / 技能只读 API。

端点:
- GET /platform/data     — 缓存覆盖、freshness、collector 探测
- GET /platform/skills   — docs/skills/*.yaml 目录
- GET /platform/skills/{name} — 单个 skill 详情
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/platform", tags=["platform"])

REPO = Path(__file__).resolve().parents[4]  # trading-platform/
DATA_DIR = REPO / "data"
SKILLS_DIR = REPO / "docs" / "skills"


def _probe_http(url: str, timeout: float = 1.5) -> dict[str, Any]:
    try:
        req = Request(url, method="GET")
        with urlopen(req, timeout=timeout) as resp:  # noqa: S310 — local health only
            body = resp.read().decode("utf-8", errors="replace")[:800]
            return {"ok": 200 <= resp.status < 300, "status_code": resp.status, "body": body}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024**2:
        return f"{n / 1024:.1f} KB"
    if n < 1024**3:
        return f"{n / 1024**2:.1f} MB"
    return f"{n / 1024**3:.2f} GB"


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _scan_cache(cache_dir: Path, kind: str) -> dict[str, Any]:
    if not cache_dir.exists():
        return {
            "kind": kind,
            "path": str(cache_dir.relative_to(REPO)) if cache_dir.is_relative_to(REPO) else str(cache_dir),
            "exists": False,
            "file_count": 0,
            "total_bytes": 0,
            "symbols": [],
            "newest_mtime": None,
            "oldest_mtime": None,
            "files": [],
        }

    files: list[dict[str, Any]] = []
    symbols: set[str] = set()
    total = 0
    newest: float | None = None
    oldest: float | None = None

    for p in cache_dir.rglob("*"):
        if not p.is_file():
            continue
        st = p.stat()
        total += st.st_size
        mtime = st.st_mtime
        newest = mtime if newest is None else max(newest, mtime)
        oldest = mtime if oldest is None else min(oldest, mtime)
        rel = str(p.relative_to(cache_dir))
        files.append({
            "path": rel,
            "bytes": st.st_size,
            "mtime": _iso(mtime),
        })
        if kind == "futures":
            # KQ_m_SHFE_rb_5m.parquet / rb_5m_....parquet
            name = p.stem
            parts = name.split("_")
            if name.startswith("KQ_m_") and len(parts) >= 4:
                symbols.add(parts[3])
            elif parts:
                symbols.add(parts[0])
        else:
            # crypto: {symbol}/1h.parquet
            try:
                symbols.add(p.relative_to(cache_dir).parts[0])
            except ValueError:
                pass

    files.sort(key=lambda f: f.get("mtime") or "", reverse=True)
    return {
        "kind": kind,
        "path": str(cache_dir.relative_to(REPO)),
        "exists": True,
        "file_count": len(files),
        "total_bytes": total,
        "total_human": _fmt_bytes(total),
        "symbol_count": len(symbols),
        "symbols": sorted(symbols),
        "newest_mtime": _iso(newest),
        "oldest_mtime": _iso(oldest),
        "files": files[:40],
    }


@router.get("/data")
async def platform_data() -> dict[str, Any]:
    """采集缓存覆盖 + data-collector 健康探测。"""
    futures = _scan_cache(DATA_DIR / "futures_cache", "futures")
    crypto = _scan_cache(DATA_DIR / "crypto_cache", "crypto")

    extras: list[dict[str, Any]] = []
    for name in ("deployed_params", "overfit_validation", "paper_probe", "tick", "daily_reports"):
        d = DATA_DIR / name
        if not d.exists():
            continue
        n_files = sum(1 for p in d.rglob("*") if p.is_file())
        extras.append({"name": name, "file_count": n_files, "exists": True})

    collector_url = os.getenv("DATA_COLLECTOR_URL", "http://127.0.0.1:18900").rstrip("/")
    probe = _probe_http(f"{collector_url}/health")
    collector: dict[str, Any] = {
        "url": collector_url,
        "ok": probe.get("ok", False),
    }
    if probe.get("ok") and probe.get("body"):
        try:
            import json
            collector["status"] = json.loads(probe["body"])
        except Exception:
            collector["raw"] = probe.get("body")
    elif probe.get("error"):
        collector["error"] = probe["error"]

    gateway_url = os.getenv("TQSDK_GATEWAY_URL", "http://127.0.0.1:12890").rstrip("/")
    gw = _probe_http(f"{gateway_url}/health")

    return {
        "data_dir": str(DATA_DIR.relative_to(REPO)),
        "caches": {"futures": futures, "crypto": crypto},
        "extras": extras,
        "collector": collector,
        "tqsdk_gateway": {
            "ok": gw.get("ok", False),
            "url": gateway_url,
            **{k: v for k, v in gw.items() if k != "ok"},
        },
    }


def _parse_skill_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml
        data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            data = {}
    except Exception:
        data = {}
        # minimal fallback: top-level scalars
        for key in ("name", "description", "category"):
            m = re.search(rf"^{key}:\s*(.+)$", text, re.MULTILINE)
            if m:
                data[key] = m.group(1).strip().strip("\"'")

    name = data.get("name") or path.stem
    iface = data.get("interface") or {}
    inputs = iface.get("input") if isinstance(iface, dict) else None
    outputs = iface.get("output") if isinstance(iface, dict) else None
    steps = data.get("steps")
    step_names: list[str] = []
    if isinstance(steps, dict):
        step_names = list(steps.keys())

    return {
        "name": name,
        "file": path.name,
        "description": data.get("description") or "",
        "category": data.get("category") or "unknown",
        "bindings": data.get("tqsdk_bindings") or [],
        "inputs": inputs if isinstance(inputs, list) else [],
        "outputs": outputs if isinstance(outputs, list) else [],
        "steps": step_names,
        "raw_keys": sorted(data.keys()) if data else [],
    }


@router.get("/skills")
async def list_skills() -> dict[str, Any]:
    """列出 docs/skills/*.yaml。"""
    if not SKILLS_DIR.exists():
        return {"skills": [], "count": 0, "dir": str(SKILLS_DIR)}

    skills: list[dict[str, Any]] = []
    for f in sorted(SKILLS_DIR.glob("*.yaml")) + sorted(SKILLS_DIR.glob("*.yml")):
        try:
            skills.append(_parse_skill_yaml(f))
        except Exception as e:
            logger.warning("Failed to parse skill %s: %s", f, e)
            skills.append({
                "name": f.stem,
                "file": f.name,
                "description": f"(parse error: {e})",
                "category": "error",
                "bindings": [],
                "inputs": [],
                "outputs": [],
                "steps": [],
            })
    return {
        "dir": str(SKILLS_DIR.relative_to(REPO)),
        "skills": skills,
        "count": len(skills),
    }


@router.get("/skills/{name}")
async def get_skill(name: str) -> dict[str, Any]:
    """按 name / stem 取 skill 全文摘要。"""
    if not SKILLS_DIR.exists():
        raise HTTPException(status_code=404, detail="Skills directory missing")

    candidates = [
        SKILLS_DIR / f"{name}.yaml",
        SKILLS_DIR / f"{name}.yml",
    ]
    # also match by YAML `name:` field
    path: Path | None = None
    for c in candidates:
        if c.exists():
            path = c
            break
    if path is None:
        for f in list(SKILLS_DIR.glob("*.yaml")) + list(SKILLS_DIR.glob("*.yml")):
            meta = _parse_skill_yaml(f)
            if meta.get("name") == name:
                path = f
                break
    if path is None:
        raise HTTPException(status_code=404, detail=f"Skill not found: {name}")

    meta = _parse_skill_yaml(path)
    meta["content"] = path.read_text(encoding="utf-8")
    meta["path"] = str(path.relative_to(REPO))
    return meta
