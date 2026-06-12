"""波动率目标策略 — 风险平价思路的仓位管理 + 动量信号。

SOTA 要点 (源自 AQR / Man AHL 研究):
- 目标波动率: 将仓位 scale 到预设的 target_vol
- 信号: 时间序列动量 (TSMOM) — 过去 N 期收益为正则做多
- 仓位 = target_vol / realized_vol * signal_direction
- 在高波动率环境自动减仓，低波动率环境加仓
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any

from ..base import BaseStrategy, Signal, SignalType, StrategyConfig
from ..registry import auto_register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "lookback": 20,           # 动量回看期
    "vol_window": 20,         # 实际波动率窗口
    "target_vol": 0.15,       # 年化目标波动率
    "annualize_factor": 252,  # 年化因子 (日线=252, 小时=252*24)
    "max_leverage": 3.0,
    "rebalance_every": 5,     # 每 N 根 bar 重新计算仓位
    "trailing_stop_vol_mult": 2.0,
    "max_hold_bars": 300,
}


@auto_register("vol_target")
class VolTargetStrategy(BaseStrategy):
    """波动率目标 + TSMOM 信号的自适应策略。"""

    def __init__(self, config: StrategyConfig) -> None:
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._close_buf: dict[str, deque[float]] = {}
        self._bar_count: dict[str, int] = {}
        self._bars_in_pos: dict[str, int] = {}
        self._entry_price: dict[str, float] = {}

    def _ensure(self, symbol: str) -> None:
        if symbol not in self._close_buf:
            max_len = max(int(self.get_param("lookback")), int(self.get_param("vol_window"))) + 20
            self._close_buf[symbol] = deque(maxlen=max_len)
            self._bar_count[symbol] = 0

    def _realized_vol(self, symbol: str) -> float | None:
        """计算已实现波动率 (年化)。"""
        window = int(self.get_param("vol_window"))
        buf = list(self._close_buf[symbol])
        if len(buf) < window + 1:
            return None
        log_returns = [math.log(buf[i] / buf[i - 1]) for i in range(-window, 0) if buf[i - 1] > 0]
        if len(log_returns) < window:
            return None
        mean = sum(log_returns) / len(log_returns)
        var = sum((r - mean) ** 2 for r in log_returns) / len(log_returns)
        daily_vol = math.sqrt(var) if var > 0 else 1e-10
        ann_factor = float(self.get_param("annualize_factor"))
        return daily_vol * math.sqrt(ann_factor)

    def _tsmom_signal(self, symbol: str) -> int:
        """时间序列动量: 过去 lookback 期总收益的方向。"""
        lookback = int(self.get_param("lookback"))
        buf = list(self._close_buf[symbol])
        if len(buf) < lookback + 1:
            return 0
        if buf[-lookback - 1] == 0:
            return 0
        ret = buf[-1] / buf[-lookback - 1] - 1
        if ret > 0:
            return 1
        elif ret < 0:
            return -1
        return 0

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._ensure(symbol)
        close = float(bar["close"])
        self._close_buf[symbol].append(close)
        self._bar_count[symbol] = self._bar_count.get(symbol, 0) + 1

        rv = self._realized_vol(symbol)
        mom_signal = self._tsmom_signal(symbol)
        if rv is None or mom_signal == 0:
            return []

        target_vol = float(self.get_param("target_vol"))
        max_lev = float(self.get_param("max_leverage"))
        leverage = min(target_vol / max(rv, 1e-10), max_lev)

        signals: list[Signal] = []
        pos = self.get_position(symbol)
        _rebal = int(self.get_param("rebalance_every"))  # noqa: F841

        if pos is None:
            strength = min(leverage / max_lev, 1.0)
            if mom_signal > 0:
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.LONG_ENTRY, strength=round(strength, 4),
                    price=close,
                    reason=f"VolTarget做多 rv={rv:.3f} lev={leverage:.2f}",
                    metadata={"realized_vol": rv, "leverage": leverage, "tsmom": mom_signal},
                ))
                self._entry_price[symbol] = close
            else:
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=SignalType.SHORT_ENTRY, strength=round(strength, 4),
                    price=close,
                    reason=f"VolTarget做空 rv={rv:.3f} lev={leverage:.2f}",
                    metadata={"realized_vol": rv, "leverage": leverage, "tsmom": mom_signal},
                ))
                self._entry_price[symbol] = close

        elif pos is not None:
            self._bars_in_pos[symbol] = self._bars_in_pos.get(symbol, 0) + 1
            max_hold = int(self.get_param("max_hold_bars"))
            vol_stop_mult = float(self.get_param("trailing_stop_vol_mult"))

            expected_direction = 1 if pos.side.value == "buy" else -1
            signal_flipped = mom_signal != expected_direction

            entry = self._entry_price.get(symbol, pos.avg_price)
            ret_since_entry = (close / entry - 1) * expected_direction if entry > 0 else 0
            bar_vol = rv / math.sqrt(float(self.get_param("annualize_factor"))) if rv > 0 else 0
            vol_stop_hit = ret_since_entry < -bar_vol * vol_stop_mult if bar_vol > 0 else False

            if self._bars_in_pos.get(symbol, 0) >= max_hold or signal_flipped or vol_stop_hit:
                exit_t = SignalType.LONG_EXIT if pos.side.value == "buy" else SignalType.SHORT_EXIT
                reason = "信号反转" if signal_flipped else ("VolStop" if vol_stop_hit else f"超时{max_hold}")
                signals.append(Signal(
                    strategy_id=self.strategy_id, symbol=symbol,
                    signal_type=exit_t, strength=0.8, price=close,
                    reason=f"VolTarget平仓: {reason} rv={rv:.3f}",
                ))
                self._bars_in_pos[symbol] = 0
        else:
            self._bars_in_pos[symbol] = 0

        for s in signals:
            self.record_signal(s)
        return signals

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        all_sigs: list[Signal] = []
        for symbol in self.config.symbols:
            bar = market_data.get(symbol)
            if bar:
                all_sigs.extend(await self.on_bar(symbol, bar))
        return all_sigs
