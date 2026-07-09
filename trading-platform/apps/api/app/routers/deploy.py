"""策略参数部署 API — 从回测/优化结果导出参数并部署到实盘策略。

端点:
- GET  /deploy/params/{strategy_name}     — 查看策略当前参数
- POST /deploy/params/{strategy_name}     — 更新策略参数
- GET  /deploy/optuna-best                — 列出 Optuna 最优参数
- POST /deploy/apply-optuna/{study_name}  — 应用 Optuna 结果到策略
- GET  /deploy/history                    — 部署历史
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/deploy", tags=["deploy"])

PARAMS_DIR = Path(__file__).resolve().parents[4] / "data" / "deployed_params"
OPTUNA_DIR = Path(__file__).resolve().parents[4] / "models"
DEPLOY_LOG = Path(__file__).resolve().parents[4] / "data" / "deploy_history.json"


def _ensure_dirs() -> None:
    PARAMS_DIR.mkdir(parents=True, exist_ok=True)


def _load_deploy_history() -> list[dict[str, Any]]:
    if DEPLOY_LOG.exists():
        return json.loads(DEPLOY_LOG.read_text())
    return []


def _save_deploy_history(history: list[dict[str, Any]]) -> None:
    DEPLOY_LOG.parent.mkdir(parents=True, exist_ok=True)
    DEPLOY_LOG.write_text(json.dumps(history, indent=2, ensure_ascii=False))


class DeployParamsRequest(BaseModel):
    params: dict[str, Any]
    source: str = "manual"
    note: str = ""


@router.get("/params/{strategy_name}")
async def get_strategy_params(strategy_name: str) -> dict[str, Any]:
    """查看策略当前部署参数。"""
    _ensure_dirs()
    param_file = PARAMS_DIR / f"{strategy_name}.json"
    if not param_file.exists():
        return {"strategy_name": strategy_name, "params": {}, "deployed": False}
    data = json.loads(param_file.read_text())
    return {"strategy_name": strategy_name, "params": data, "deployed": True}


@router.post("/params/{strategy_name}")
async def deploy_strategy_params(
    strategy_name: str, req: DeployParamsRequest,
) -> dict[str, Any]:
    """部署策略参数。"""
    _ensure_dirs()
    param_file = PARAMS_DIR / f"{strategy_name}.json"

    old_params = {}
    if param_file.exists():
        old_params = json.loads(param_file.read_text())

    param_file.write_text(json.dumps(req.params, indent=2))

    record = {
        "strategy_name": strategy_name,
        "source": req.source,
        "note": req.note,
        "old_params": old_params,
        "new_params": req.params,
        "deployed_at": datetime.now(timezone.utc).isoformat(),
    }
    history = _load_deploy_history()
    history.append(record)
    _save_deploy_history(history)

    logger.info("Deployed params for %s from %s", strategy_name, req.source)
    return {"status": "deployed", "strategy_name": strategy_name, "params": req.params}


@router.get("/optuna-best")
async def list_optuna_best(
    market: str | None = Query(None),
    top_n: int = Query(10, ge=1, le=50),
) -> list[dict[str, Any]]:
    """列出 Optuna 最优参数。扫描 models/ 目录下的 Optuna 结果。"""
    results: list[dict[str, Any]] = []

    if not OPTUNA_DIR.exists():
        return results

    for f in sorted(OPTUNA_DIR.glob("optuna_*.json")):
        try:
            data = json.loads(f.read_text())
            if isinstance(data, dict) and "best_params" in data:
                entry = {
                    "file": f.name,
                    "strategy": data.get("strategy", f.stem),
                    "market": data.get("market", "unknown"),
                    "best_score": data.get("best_value", data.get("best_score")),
                    "sharpe": data.get("sharpe_ratio"),
                    "total_return": data.get("total_return"),
                    "best_params": data["best_params"],
                    "n_trials": data.get("n_trials", 0),
                }
                if market and entry["market"] != market:
                    continue
                results.append(entry)
        except (json.JSONDecodeError, KeyError):
            continue

    results.sort(key=lambda x: x.get("best_score") or 0, reverse=True)
    return results[:top_n]


@router.post("/apply-optuna/{study_name}")
async def apply_optuna_result(
    study_name: str,
    strategy_name: str | None = Query(None),
) -> dict[str, Any]:
    """应用 Optuna 优化结果到策略。"""
    optuna_file = OPTUNA_DIR / f"optuna_{study_name}.json"
    if not optuna_file.exists():
        optuna_file = OPTUNA_DIR / f"{study_name}.json"
    if not optuna_file.exists():
        raise HTTPException(status_code=404, detail=f"Optuna result not found: {study_name}")

    data = json.loads(optuna_file.read_text())
    best_params = data.get("best_params")
    if not best_params:
        raise HTTPException(status_code=400, detail="No best_params in Optuna result")

    target_name = strategy_name or data.get("strategy", study_name)

    _ensure_dirs()
    param_file = PARAMS_DIR / f"{target_name}.json"
    old_params = json.loads(param_file.read_text()) if param_file.exists() else {}
    param_file.write_text(json.dumps(best_params, indent=2))

    record = {
        "strategy_name": target_name,
        "source": f"optuna:{study_name}",
        "note": f"Applied from {optuna_file.name}, score={data.get('best_value', 'N/A')}",
        "old_params": old_params,
        "new_params": best_params,
        "deployed_at": datetime.now(timezone.utc).isoformat(),
    }
    history = _load_deploy_history()
    history.append(record)
    _save_deploy_history(history)

    return {
        "status": "applied",
        "strategy_name": target_name,
        "source": study_name,
        "params": best_params,
    }


@router.post("/rollback/{strategy_name}")
async def rollback_strategy_params(strategy_name: str) -> dict[str, Any]:
    """回滚策略参数到上一版本（从部署历史中恢复 old_params）。"""
    history = _load_deploy_history()
    target = [r for r in reversed(history) if r["strategy_name"] == strategy_name]
    if not target:
        raise HTTPException(status_code=404, detail=f"No deploy history for {strategy_name}")

    last_deploy = target[0]
    old_params = last_deploy.get("old_params", {})

    _ensure_dirs()
    param_file = PARAMS_DIR / f"{strategy_name}.json"
    current_params = json.loads(param_file.read_text()) if param_file.exists() else {}

    if not old_params:
        if param_file.exists():
            param_file.unlink()
        logger.info("Rolled back %s to empty (no prior params)", strategy_name)
    else:
        param_file.write_text(json.dumps(old_params, indent=2))

    record = {
        "strategy_name": strategy_name,
        "source": "rollback",
        "note": f"Rolled back from deploy at {last_deploy.get('deployed_at', '?')}",
        "old_params": current_params,
        "new_params": old_params,
        "deployed_at": datetime.now(timezone.utc).isoformat(),
    }
    history.append(record)
    _save_deploy_history(history)

    logger.info("Rolled back params for %s", strategy_name)
    return {"status": "rolled_back", "strategy_name": strategy_name, "params": old_params}


@router.get("/history")
async def get_deploy_history(
    limit: int = Query(20, ge=1, le=100),
) -> list[dict[str, Any]]:
    """部署历史。"""
    history = _load_deploy_history()
    return list(reversed(history[-limit:]))
