"""仓位管理模块 — 统一支持 fixed / risk_parity / kelly 三种模式。"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any, Mapping

Number = int | float | Decimal


def _as_float(value: Number) -> float:
    """将数值统一转换为 float。"""
    if isinstance(value, Decimal):
        return float(value)
    return float(value)


def _normalize(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        return {symbol: 0.0 for symbol in weights}
    return {symbol: weight / total for symbol, weight in weights.items()}


@dataclass(frozen=True, slots=True)
class KellyInput:
    """Kelly 计算输入。"""

    win_rate: Number
    avg_win: Number
    avg_loss: Number


class SizingMethod(str, Enum):
    FIXED = "fixed"
    RISK_PARITY = "risk_parity"
    KELLY = "kelly"


class _FixedSizer:
    def size(
        self,
        symbols: list[str],
        *,
        fixed_weights: Mapping[str, Number] | None = None,
    ) -> dict[str, float]:
        if not symbols:
            return {}

        if fixed_weights:
            raw = {symbol: max(_as_float(fixed_weights.get(symbol, 0)), 0.0) for symbol in symbols}
            if sum(raw.values()) > 0:
                return _normalize(raw)

        equal_weight = 1.0 / len(symbols)
        return {symbol: equal_weight for symbol in symbols}


class _RiskParitySizer:
    def size(
        self,
        symbols: list[str],
        *,
        volatilities: Mapping[str, Number],
    ) -> dict[str, float]:
        if not symbols:
            return {}

        inv_vol_weights: dict[str, float] = {}
        for symbol in symbols:
            vol = _as_float(volatilities.get(symbol, 0))
            inv_vol_weights[symbol] = 0.0 if vol <= 0 else (1.0 / vol)
        return _normalize(inv_vol_weights)


class _KellySizer:
    def __init__(self, cap: float) -> None:
        self._cap = max(cap, 0.0)

    @property
    def cap(self) -> float:
        return self._cap

    def size(
        self,
        symbols: list[str],
        *,
        kelly_inputs: Mapping[str, KellyInput | Mapping[str, Number]],
    ) -> dict[str, float]:
        if not symbols:
            return {}

        result: dict[str, float] = {}
        for symbol in symbols:
            result[symbol] = self._kelly_fraction(kelly_inputs.get(symbol))
        return result

    def _kelly_fraction(self, raw: KellyInput | Mapping[str, Number] | None) -> float:
        if raw is None:
            return 0.0

        if isinstance(raw, KellyInput):
            win_rate = _as_float(raw.win_rate)
            avg_win = _as_float(raw.avg_win)
            avg_loss = _as_float(raw.avg_loss)
        else:
            try:
                win_rate = _as_float(raw["win_rate"])
                avg_win = _as_float(raw["avg_win"])
                avg_loss = _as_float(raw["avg_loss"])
            except KeyError:
                return 0.0

        if not (0 < win_rate < 1):
            return 0.0
        if avg_win <= 0 or avg_loss <= 0:
            return 0.0

        odds = avg_win / avg_loss
        if odds <= 0:
            return 0.0

        fraction = win_rate - ((1.0 - win_rate) / odds)
        if fraction <= 0:
            return 0.0
        return min(fraction, self._cap)


class PositionSizer:
    """统一仓位管理入口，返回 symbol -> target ratio 映射。"""

    SUPPORTED_METHODS = frozenset(SizingMethod)

    def __init__(
        self,
        method: SizingMethod | str = SizingMethod.FIXED,
        *,
        kelly_cap: float = 0.25,
    ) -> None:
        if isinstance(method, str):
            method = SizingMethod(method.lower())
        if method not in self.SUPPORTED_METHODS:
            candidates = sorted(m.value for m in self.SUPPORTED_METHODS)
            raise ValueError(f"unknown method {method!r}, choose from {candidates}")

        self._method = method
        self._fixed_sizer = _FixedSizer()
        self._risk_parity_sizer = _RiskParitySizer()
        self._kelly_sizer = _KellySizer(cap=kelly_cap)

    @property
    def method(self) -> SizingMethod:
        return self._method

    def size(
        self,
        symbols: list[str] | tuple[str, ...] | None = None,
        *,
        fixed_weights: Mapping[str, Number] | None = None,
        volatility: Mapping[str, Number] | None = None,
        volatilities: Mapping[str, Number] | None = None,
        kelly_inputs: Mapping[str, KellyInput | Mapping[str, Number]] | None = None,
        kelly_stats: Mapping[str, KellyInput | Mapping[str, Number]] | None = None,
    ) -> dict[str, float]:
        """输出目标仓位（非负比例）:

        - fixed: 等权或按 fixed_weights 归一化
        - risk_parity: 按波动率倒数归一化
        - kelly: 使用 Kelly 公式并截断到 [0, kelly_cap]
        """
        if self._method == SizingMethod.FIXED:
            resolved_symbols = self._resolve_symbols(symbols, fixed_weights)
            return self._fixed_sizer.size(
                resolved_symbols,
                fixed_weights=fixed_weights,
            )

        if self._method == SizingMethod.RISK_PARITY:
            vol_map = volatility if volatility is not None else volatilities
            if vol_map is None:
                raise ValueError("risk_parity mode requires volatility or volatilities")
            resolved_symbols = self._resolve_symbols(symbols, vol_map)
            return self._risk_parity_sizer.size(
                resolved_symbols,
                volatilities=vol_map,
            )

        kelly_map = kelly_inputs if kelly_inputs is not None else kelly_stats
        if kelly_map is None:
            raise ValueError("kelly mode requires kelly_inputs or kelly_stats")
        resolved_symbols = self._resolve_symbols(symbols, kelly_map)
        return self._kelly_sizer.size(
            resolved_symbols,
            kelly_inputs=kelly_map,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self._method.value,
            "kelly_cap": self._kelly_sizer.cap,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PositionSizer:
        return cls(
            method=data.get("method", "fixed"),
            kelly_cap=_as_float(data.get("kelly_cap", 0.25)),
        )

    def _resolve_symbols(
        self,
        symbols: list[str] | tuple[str, ...] | None,
        source: Mapping[str, Any] | None,
    ) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()

        if symbols:
            for symbol in symbols:
                if symbol and symbol not in seen:
                    deduped.append(symbol)
                    seen.add(symbol)

        if deduped:
            return deduped

        if source:
            for symbol in source:
                if symbol and symbol not in seen:
                    deduped.append(symbol)
                    seen.add(symbol)
        return deduped


__all__ = ["KellyInput", "PositionSizer", "SizingMethod"]
