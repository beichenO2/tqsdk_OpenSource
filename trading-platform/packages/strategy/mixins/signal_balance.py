"""SignalBalanceMixin — 防单边信号偏差（supertrend 35 SHORT/0 LONG 类 bug）。

Root: 某些策略（尤其是 trend_change-based）在趋势行情中只会从一个方向入场，
例如 rb 上涨但 supertrend 只产 SHORT。本 mixin 统计 LONG/SHORT 入场次数，当
主导方 / 劣势方 > `max_ratio` 时，压制主导方的新入场信号，迫使策略放弃单边。

用法：
    class MyStrategy(SignalBalanceMixin, BaseStrategy):
        SB_WARMUP = 8           # 前 8 笔不管
        SB_MAX_RATIO = 3.0      # 主导方/劣势方 > 3 → 压制

        async def on_bar(...):
            ...
            if self._sb_allow(SignalType.LONG_ENTRY):
                # emit long entry
                self._sb_record(SignalType.LONG_ENTRY)

类属性可被子类覆盖。read 只需 `_sb_allow(signal_type)`；每次真正入场后调
`_sb_record(signal_type)`。
"""

from __future__ import annotations

from typing import Any


class SignalBalanceMixin:
    """混入一个轻量 L/S 计数器 + ratio-based 压制."""

    SB_WARMUP: int = 8
    SB_MAX_RATIO: float = 3.0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._sb_long_count = 0
        self._sb_short_count = 0

    def _sb_allow(self, signal_type: Any) -> bool:
        """是否允许此方向的新入场信号。对非 LONG/SHORT_ENTRY 一律放行。"""
        name = getattr(signal_type, "name", str(signal_type))
        total = self._sb_long_count + self._sb_short_count
        if total < self.SB_WARMUP:
            return True

        if name == "LONG_ENTRY":
            denom = max(self._sb_short_count, 1)
            return (self._sb_long_count / denom) <= self.SB_MAX_RATIO
        if name == "SHORT_ENTRY":
            denom = max(self._sb_long_count, 1)
            return (self._sb_short_count / denom) <= self.SB_MAX_RATIO
        return True

    def _sb_record(self, signal_type: Any) -> None:
        name = getattr(signal_type, "name", str(signal_type))
        if name == "LONG_ENTRY":
            self._sb_long_count += 1
        elif name == "SHORT_ENTRY":
            self._sb_short_count += 1

    def _sb_stats(self) -> dict[str, Any]:
        total = self._sb_long_count + self._sb_short_count
        if total == 0:
            return {"long": 0, "short": 0, "ratio": 1.0, "balance_pct": 0.5}
        balance_pct = self._sb_long_count / total
        ratio = self._sb_long_count / max(self._sb_short_count, 1) if self._sb_short_count else float("inf")
        return {"long": self._sb_long_count, "short": self._sb_short_count, "ratio": ratio, "balance_pct": balance_pct}
