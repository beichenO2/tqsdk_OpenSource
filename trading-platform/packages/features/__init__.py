"""
Features - 因子与特征工程模块

提供：
- 技术指标因子（MA, RSI, MACD, Bollinger, ATR 等）
- 微观结构因子（OFI, VPIN, 买卖压力等）
- 统计因子（波动率、偏度、峰度、协整等）
- 特征管道（批量计算、标准化、特征选择）
- 因子注册中心（可扩展的因子注册机制）
"""

from features.registry import FactorRegistry, factor
from features.engine import FeatureEngine
from features.technical import (
    ma,
    ema,
    rsi,
    macd,
    bollinger_bands,
    atr,
    obv,
)

import importlib as _importlib
for _mod in ("features.statistical", "features.microstructure", "features.research_factors"):
    try:
        _importlib.import_module(_mod)
    except ImportError:
        pass

__all__ = [
    "FactorRegistry",
    "FeatureEngine",
    "factor",
    "ma",
    "ema",
    "rsi",
    "macd",
    "bollinger_bands",
    "atr",
    "obv",
]
