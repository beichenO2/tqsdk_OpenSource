"""FeatureMixin — 为策略提供 rolling bar 缓冲与因子值计算。"""

from __future__ import annotations

from collections import deque
from typing import Any

import pandas as pd


class FeatureMixin:
    """为策略注入因子计算能力（懒加载 FeatureEngine，零 features 零开销）。"""

    def _init_features(self) -> None:
        self._bar_buffers: dict[str, deque[dict[str, Any]]] = {}
        self._feature_window = int(self.get_param("feature_window", 200))

    def _ensure_bar_buffer(self, symbol: str) -> deque[dict[str, Any]]:
        if symbol not in self._bar_buffers:
            self._bar_buffers[symbol] = deque(maxlen=self._feature_window)
        return self._bar_buffers[symbol]

    def record_bar(self, symbol: str, bar: dict[str, Any]) -> None:
        self._ensure_bar_buffer(symbol).append(bar)

    def _bars_to_ohlcv_df(self, symbol: str) -> pd.DataFrame:
        buf = self._bar_buffers.get(symbol)
        if not buf:
            return pd.DataFrame()
        rows: list[dict[str, float]] = []
        for b in buf:
            close = float(b["close"])
            rows.append({
                "open": float(b.get("open", close)),
                "high": float(b.get("high", close)),
                "low": float(b.get("low", close)),
                "close": close,
                "volume": float(b.get("volume", 0.0)),
            })
        return pd.DataFrame(rows)

    def factor_values(self, symbol: str) -> dict[str, float | None]:
        features = list(getattr(self.config, "features", None) or [])
        if not features:
            return {}

        df = self._bars_to_ohlcv_df(symbol)
        if df.empty:
            return {name: None for name in features}

        from factor.registry import get_registry
        from features.engine import FeatureEngine

        engine = FeatureEngine(get_registry())
        registry = get_registry()
        out: dict[str, float | None] = {name: None for name in features}

        for name in features:
            try:
                result = engine.compute_factors(df.copy(), [name])
                meta = registry.get(name)
                col = meta.output_columns[0]
                if col not in result.columns:
                    continue
                series = result[col].dropna()
                if len(series):
                    out[name] = float(series.iloc[-1])
            except Exception:
                out[name] = None

        return out
