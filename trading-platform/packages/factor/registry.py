"""桥接 packages/features.FactorRegistry → 统一元数据视图。"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_LOADED = False


def ensure_features_loaded() -> None:
    """导入 features 子模块以触发 @factor 注册。"""
    global _LOADED
    if _LOADED:
        return
    try:
        import features  # noqa: F401 — side-effect: register factors
        for mod in (
            "features.technical",
            "features.statistical",
            "features.microstructure",
            "features.research_factors",
            "features.alpha_sets",
        ):
            try:
                __import__(mod)
            except ImportError as e:
                logger.debug("Optional factor module %s unavailable: %s", mod, e)
        _LOADED = True
    except ImportError as e:
        logger.warning("features package unavailable: %s", e)


def get_registry():
    ensure_features_loaded()
    from features.registry import FactorRegistry
    return FactorRegistry()


def list_factor_metas(category: str | None = None) -> list[dict[str, Any]]:
    """返回因子元数据列表（可 JSON 序列化）。"""
    reg = get_registry()
    rows: list[dict[str, Any]] = []
    for meta in reg.list_factors(category=category):
        rows.append({
            "name": meta.name,
            "category": meta.category,
            "description": (meta.description or "").strip(),
            "params": dict(meta.params),
            "output_columns": list(meta.output_columns),
        })
    rows.sort(key=lambda r: (r["category"], r["name"]))
    return rows


def get_factor_meta(name: str) -> dict[str, Any]:
    reg = get_registry()
    meta = reg.get(name)
    return {
        "name": meta.name,
        "category": meta.category,
        "description": (meta.description or "").strip(),
        "params": dict(meta.params),
        "output_columns": list(meta.output_columns),
    }


def compute_factor_frame(df, names: list[str], params: dict | None = None):
    """在 OHLCV DataFrame 上批量计算因子，返回含输出列的副本。"""
    from features.engine import FeatureEngine
    engine = FeatureEngine(get_registry())
    return engine.compute_factors(df.copy(), names, params=params)
