"""策略工厂 — 从 strategy_catalog 动态实例化策略对象。

通过 class_path 字符串动态导入策略类，用 params 构造 StrategyConfig。
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

from strategy.base import BaseStrategy, StrategyConfig

from .strategy_catalog import get_catalog

logger = logging.getLogger(__name__)


def _import_class(class_path: str) -> type:
    """从 'module.path.ClassName' 导入类。"""
    parts = class_path.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid class_path: {class_path}")
    module_path, class_name = parts
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def create_strategy(entry: dict[str, Any]) -> BaseStrategy:
    """从目录条目创建策略实例。"""
    cls = _import_class(entry["class_path"])
    config = StrategyConfig(
        strategy_id=f"sim_{entry['account_id']:03d}",
        name=entry["name"],
        symbols=entry["symbols"],
        params=entry.get("params", {}),
    )
    return cls(config)


def create_all_strategies(market: str | None = None) -> dict[int, BaseStrategy]:
    """批量创建策略实例。返回 {account_id: strategy}。"""
    catalog = get_catalog(market)
    strategies: dict[int, BaseStrategy] = {}
    failed: list[str] = []

    for entry in catalog:
        aid = entry["account_id"]
        try:
            strategies[aid] = create_strategy(entry)
            logger.debug("Created strategy %s for account %d", entry["name"], aid)
        except Exception as e:
            failed.append(f"Account {aid} ({entry['name']}): {e}")
            logger.warning("Failed to create strategy for account %d: %s", aid, e)

    logger.info(
        "StrategyFactory: %d/%d strategies created (%d failed)",
        len(strategies), len(catalog), len(failed),
    )
    if failed:
        for f in failed[:10]:
            logger.warning("  - %s", f)

    if not strategies and catalog:
        from .observer_strategy import ObserverStrategy

        logger.warning(
            "All %d catalog strategies failed — deploying ObserverStrategy fallbacks "
            "to keep paper engine running",
            len(catalog),
        )
        for entry in catalog:
            aid = entry["account_id"]
            config = StrategyConfig(
                strategy_id=f"obs_{aid:03d}",
                name=f"Observer-{entry['name']}",
                symbols=entry["symbols"],
            )
            strategies[aid] = ObserverStrategy(config)

    return strategies
