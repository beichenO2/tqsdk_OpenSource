"""策略管理路由."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from core.exceptions import StrategyNotFoundError
from strategy.base import StrategyConfig
from strategy.registry import StrategyRegistry

router = APIRouter(prefix="/strategies", tags=["strategies"])


class StrategyCreateRequest(BaseModel):
    name: str
    symbols: list[str]
    params: dict = {}
    enabled: bool = True
    max_position: int = 10
    capital: float = 1_000_000.0


@router.post("")
async def create_strategy(req: StrategyCreateRequest) -> dict:
    merged_params = {
        **req.params,
        "max_position": req.max_position,
        "capital": req.capital,
    }
    config = StrategyConfig(
        name=req.name,
        symbols=req.symbols,
        params=merged_params,
        enabled=req.enabled,
    )
    StrategyRegistry.add_instance(config)
    return {"strategy_id": config.strategy_id, "status": "created"}


@router.get("")
async def list_strategies() -> list[dict]:
    return [c.model_dump(mode="json") for c in StrategyRegistry.list_instances()]


@router.get("/{strategy_id}")
async def get_strategy(strategy_id: str) -> dict:
    cfg = StrategyRegistry.get_instance(strategy_id)
    if cfg is None:
        raise StrategyNotFoundError(f"Strategy {strategy_id} not found")
    return cfg.model_dump(mode="json")


@router.put("/{strategy_id}/toggle")
async def toggle_strategy(strategy_id: str, enabled: bool = True) -> dict:
    updated = StrategyRegistry.set_instance_enabled(strategy_id, enabled)
    if updated is None:
        raise StrategyNotFoundError(f"Strategy {strategy_id} not found")
    return updated.model_dump(mode="json")


@router.post("/pause-all")
async def pause_all_strategies() -> dict:
    instances = StrategyRegistry.list_instances()
    count = 0
    for cfg in instances:
        if cfg.enabled:
            StrategyRegistry.set_instance_enabled(cfg.strategy_id, False)
            count += 1
    return {"status": "ok", "paused": count}
