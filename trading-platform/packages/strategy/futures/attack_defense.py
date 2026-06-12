"""攻防建模策略 — 两股力量的 Lotka-Volterra 市场动力学。

数学基础：
  假设市场只存在两股力量——攻（买方）和防（卖方）。
  二者此消彼长但永不消失，用改进的 Lotka-Volterra 竞争模型描述其动态：

  dB/dt = B · (α - β·S/(B+S))  +  η_B(t)
  dS/dt = S · (γ - δ·B/(B+S))  +  η_S(t)

  其中 η 项由观测到的量价数据驱动，而非纯随机噪声。

力量估计（从 OHLCV + OI 观测数据到隐变量）：
  - 量价分解（Lee-Ready 思想）：
      V_buy  = Volume × (Close - Low)  / (High - Low)
      V_sell = Volume × (High - Close) / (High - Low)

  - OI 增量修正：
      ΔOI > 0 且价格上涨 → 新多头入场，增强攻方
      ΔOI > 0 且价格下跌 → 新空头入场，增强防方
      ΔOI < 0             → 平仓退出，按方向弱化对应方

  - 离散时间递推（EMA + 交叉抑制 + 正性约束）：
      B(t) = max(ε, λ·B(t-1)·(1 - κ·S(t-1)/Σ) + (1-λ)·B_raw(t))
      S(t) = max(ε, λ·S(t-1)·(1 - κ·B(t-1)/Σ) + (1-λ)·S_raw(t))
      其中 Σ = B(t-1) + S(t-1), ε = 1e-8

衍生指标：
  - 力量比 R(t)     = B/(B+S) ∈ (0,1)，>0.5 买方占优
  - 力量失衡 I(t)   = (B-S)/(B+S) ∈ (-1,1)
  - 力量动量 M(t)   = R(t) - R(t-1)
  - 力量加速度 A(t)  = M(t) - M(t-1)
  - 力量背离：价格方向 vs 力量比方向不一致 → 反转信号

参考文献：
  - Montero et al. (2020) "Predator-prey model for stock market fluctuations"
    Journal of Economic Interaction and Coordination
  - 长江证券 (2020) "分布估计下的主动成交占比" — 博弈因子构建
  - Cont et al. (2014) Order Flow Imbalance 研究
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any

import numpy as np

from ..base import BaseStrategy, Signal, SignalType, StrategyConfig
from ..indicators import calc_atr, ema_update, rolling_zscore
from ..registry import auto_register

logger = logging.getLogger(__name__)

_EPS = 1e-8

DEFAULT_PARAMS: dict[str, Any] = {
    # ── 力量估计参数 ──
    "force_decay": 0.92,            # λ: EMA 衰减，越大越平滑（0.85→0.92 减少噪声）
    "cross_inhibit": 0.10,          # κ: 交叉抑制系数（Lotka-Volterra 竞争项）
    "oi_amplify": 1.5,              # OI 增量对力量的放大系数
    "warmup_bars": 60,              # 预热期（含均量 EMA 预热）
    "carrying_capacity": 10.0,      # K: logistic 承载容量，防止力量发散
    "vol_ema_period": 30,           # 成交量 EMA 归一化窗口
    "commodity_mode": False,        # 商品模式：禁用 OI 修正，纯量价分解

    # ── 信号参数 ──
    "ratio_upper": 0.72,            # 力量比做多阈值（极端化：只在明确占优时入场）
    "ratio_lower": 0.28,            # 力量比做空阈值
    "momentum_confirm": 0.006,      # 力量动量确认阈值
    "momentum_smooth": 5,           # 动量 EMA 平滑周期
    "sustain_bars": 3,              # 力量比需连续 N bars 超过阈值才入场
    "divergence_window": 15,        # 背离检测窗口
    "divergence_threshold": 0.7,    # 背离信号强度阈值（提高）

    # ── 趋势确认 ──
    "trend_ma_period": 30,          # 趋势均线周期
    "force_accel_threshold": 0.005, # 加速度确认阈值

    # ── 风控参数 ──
    "atr_period": 14,
    "tp_atr_mult": 4.5,
    "sl_atr_mult": 2.0,
    "max_hold_bars": 48,
    "min_hold_bars": 8,             # 最小持仓周期（5min × 8 = 40min）
    "cooldown_bars": 24,            # 冷却期（5min × 24 = 2h）
}


def _volume_decompose(h: float, l: float, c: float, v: float) -> tuple[float, float]:
    """将总成交量分解为买量和卖量（Lee-Ready 近似）。"""
    rng = h - l
    if rng < _EPS:
        return v * 0.5, v * 0.5
    buy_ratio = (c - l) / rng
    return v * buy_ratio, v * (1.0 - buy_ratio)


def _oi_contribution(
    delta_oi: float,
    oi_prev: float,
    price_ret: float,
    amplify: float,
) -> tuple[float, float]:
    """OI 增量修正：判断新资金流入方向并分配到攻/防。

    Returns (buy_boost, sell_boost) ≥ 0.
    """
    if oi_prev < _EPS:
        return 0.0, 0.0
    oi_ratio = abs(delta_oi) / oi_prev * amplify

    if delta_oi > 0:
        if price_ret > 0:
            return oi_ratio, 0.0
        else:
            return 0.0, oi_ratio
    elif delta_oi < 0:
        if price_ret > 0:
            return oi_ratio * 0.3, 0.0
        else:
            return 0.0, oi_ratio * 0.3
    return 0.0, 0.0


@auto_register("attack_defense")
class AttackDefenseStrategy(BaseStrategy):
    """攻防建模策略。

    基于修改的 Lotka-Volterra 竞争模型，从量价+OI 数据估计买卖双方
    力量强度，利用力量比、动量、背离等衍生指标产生交易信号。
    """

    def __init__(self, config: StrategyConfig) -> None:
        config = config.model_copy(
            update={"params": {**DEFAULT_PARAMS, **config.params}}
        )
        super().__init__(config)

        self._highs: deque[float] = deque(maxlen=400)
        self._lows: deque[float] = deque(maxlen=400)
        self._closes: deque[float] = deque(maxlen=400)
        self._volumes: deque[float] = deque(maxlen=400)
        self._ois: deque[float] = deque(maxlen=400)

        # 攻防力量序列
        self._attack: float = 1.0   # B(t) — 买方力量
        self._defense: float = 1.0  # S(t) — 卖方力量
        self._ratios: deque[float] = deque(maxlen=400)   # R(t)
        self._momentums: deque[float] = deque(maxlen=400) # M(t)
        self._vol_ema: float | None = None  # 成交量 EMA（归一化用）
        self._momentum_ema: float | None = None  # 平滑后的动量

        self._bar_count = 0
        self._position_side: str | None = None
        self._entry_price = 0.0
        self._hold_bars = 0
        self._cd = 0
        self._sustain_bull = 0   # 连续处于多头区间的 bar 数
        self._sustain_bear = 0   # 连续处于空头区间的 bar 数

    # ──────────────────────────────────────────────
    #  核心：Lotka-Volterra 力量递推
    # ──────────────────────────────────────────────

    def _update_forces(self, bar: dict[str, Any]) -> None:
        """从新 bar 数据递推攻防力量。

        V2 改进：
        - 成交量 EMA 归一化，消除日内量能差异
        - Logistic 承载容量项 (1 - F/K)，防止力量无界增长
        - 动量 EMA 平滑，减少高频噪声
        """
        h = float(bar.get("high", 0))
        l = float(bar.get("low", 0))
        c = float(bar.get("close", 0))
        v = float(bar.get("volume", 0))
        oi = float(bar.get("open_interest", bar.get("oi", 0)))

        self._highs.append(h)
        self._lows.append(l)
        self._closes.append(c)
        self._volumes.append(v)
        self._ois.append(oi)
        self._bar_count += 1

        p = self.config.params

        # 更新成交量 EMA（归一化基准）
        self._vol_ema = ema_update(self._vol_ema, max(v, 1.0), p["vol_ema_period"])

        if self._bar_count < 2:
            self._ratios.append(0.5)
            self._momentums.append(0.0)
            return

        lam = p["force_decay"]
        kappa = p["cross_inhibit"]
        K = p["carrying_capacity"]

        v_buy, v_sell = _volume_decompose(h, l, c, v)

        prev_c = self._closes[-2]
        price_ret = (c - prev_c) / prev_c if prev_c > _EPS else 0.0

        if p.get("commodity_mode", False):
            buy_boost, sell_boost = 0.0, 0.0
        else:
            prev_oi = self._ois[-2]
            delta_oi = oi - prev_oi
            buy_boost, sell_boost = _oi_contribution(
                delta_oi, prev_oi, price_ret, p["oi_amplify"]
            )

        # 归一化：除以成交量 EMA，使 raw force ≈ 1.0 量级
        vol_norm = max(self._vol_ema or 1.0, 1.0)
        b_raw = v_buy * (1.0 + buy_boost) / vol_norm
        s_raw = v_sell * (1.0 + sell_boost) / vol_norm

        total = self._attack + self._defense
        if total < _EPS:
            total = 2.0

        # Lotka-Volterra + logistic 承载容量
        # (1 - F/K) 项：力量越接近 K，增长越慢，防止发散
        self._attack = max(
            _EPS,
            lam * self._attack
            * (1.0 - kappa * self._defense / total)
            * (1.0 - max(0.0, self._attack - 1.0) / K)
            + (1.0 - lam) * b_raw,
        )
        self._defense = max(
            _EPS,
            lam * self._defense
            * (1.0 - kappa * self._attack / total)
            * (1.0 - max(0.0, self._defense - 1.0) / K)
            + (1.0 - lam) * s_raw,
        )

        new_total = self._attack + self._defense
        ratio = self._attack / new_total if new_total > _EPS else 0.5
        self._ratios.append(ratio)

        # 动量：先算 raw，再 EMA 平滑
        raw_mom = ratio - self._ratios[-2] if len(self._ratios) >= 2 else 0.0
        self._momentum_ema = ema_update(
            self._momentum_ema, raw_mom, p["momentum_smooth"]
        )
        self._momentums.append(self._momentum_ema)

    # ──────────────────────────────────────────────
    #  衍生指标
    # ──────────────────────────────────────────────

    def _force_acceleration(self) -> float:
        """力量加速度：动量的变化率。"""
        if len(self._momentums) < 2:
            return 0.0
        return self._momentums[-1] - self._momentums[-2]

    def _detect_divergence(self) -> tuple[float, str]:
        """检测力量-价格背离。

        Returns (strength, type): strength ∈ [0,1], type ∈ {"bullish","bearish","none"}
        """
        p = self.config.params
        win = p["divergence_window"]
        if len(self._closes) < win or len(self._ratios) < win:
            return 0.0, "none"

        closes_w = list(self._closes)[-win:]
        ratios_w = list(self._ratios)[-win:]

        price_slope = (closes_w[-1] - closes_w[0]) / (abs(closes_w[0]) + _EPS)
        ratio_slope = ratios_w[-1] - ratios_w[0]

        if price_slope > 0.005 and ratio_slope < -0.02:
            strength = min(1.0, abs(ratio_slope) / 0.1)
            return strength, "bearish"
        elif price_slope < -0.005 and ratio_slope > 0.02:
            strength = min(1.0, abs(ratio_slope) / 0.1)
            return strength, "bullish"
        return 0.0, "none"

    def _trend_filter(self) -> float:
        """趋势过滤器：当前价相对均线的位置，>0 多头，<0 空头。"""
        p = self.config.params
        period = p["trend_ma_period"]
        if len(self._closes) < period:
            return 0.0
        ma = sum(list(self._closes)[-period:]) / period
        return (self._closes[-1] - ma) / (ma + _EPS)

    # ──────────────────────────────────────────────
    #  交易信号
    # ──────────────────────────────────────────────

    def _check_entry(
        self, ratio: float, momentum: float, accel: float,
        div_strength: float, div_type: str, trend: float, atr: float, close: float,
    ) -> tuple[str, float, str] | None:
        """检查入场条件。Returns (side, strength, reason) or None."""
        p = self.config.params
        upper = p["ratio_upper"]
        lower = p["ratio_lower"]
        mom_thr = p["momentum_confirm"]
        div_thr = p["divergence_threshold"]
        sustain_req = p.get("sustain_bars", 3)

        # 更新持续计数器
        if ratio > upper:
            self._sustain_bull += 1
        else:
            self._sustain_bull = 0

        if ratio < lower:
            self._sustain_bear += 1
        else:
            self._sustain_bear = 0

        signals: list[tuple[str, float, str]] = []

        # 信号 1: 力量比持续突破 + 动量确认
        if self._sustain_bull >= sustain_req and momentum > mom_thr:
            s = min(1.0, (ratio - upper) / (1.0 - upper) + abs(momentum) * 3)
            if trend > -0.005:
                signals.append((
                    "buy", s,
                    f"攻方持续占优 R={ratio:.3f} M={momentum:.4f} 持续{self._sustain_bull}bars",
                ))

        if self._sustain_bear >= sustain_req and momentum < -mom_thr:
            s = min(1.0, (lower - ratio) / lower + abs(momentum) * 3)
            if trend < 0.005:
                signals.append((
                    "sell", s,
                    f"防方持续占优 R={ratio:.3f} M={momentum:.4f} 持续{self._sustain_bear}bars",
                ))

        # 信号 2: 力量背离反转（不需要持续确认，但阈值更高）
        if div_type == "bullish" and div_strength > div_thr and ratio > 0.45:
            signals.append(("buy", div_strength * 0.7, f"看涨背离 div={div_strength:.2f}"))

        if div_type == "bearish" and div_strength > div_thr and ratio < 0.55:
            signals.append(("sell", div_strength * 0.7, f"看跌背离 div={div_strength:.2f}"))

        if not signals:
            return None

        best = max(signals, key=lambda x: x[1])
        return best

    def _check_exit(
        self, close: float, atr: float, ratio: float,
        momentum: float, div_strength: float, div_type: str,
    ) -> tuple[bool, str]:
        """检查出场条件。"""
        p = self.config.params
        min_hold = p.get("min_hold_bars", 6)

        if self._hold_bars >= p["max_hold_bars"]:
            return True, f"最大持仓({p['max_hold_bars']}bars)"

        # 硬止损不受最小持仓限制
        if self._position_side == "buy":
            if close < self._entry_price - atr * p["sl_atr_mult"]:
                return True, f"止损({p['sl_atr_mult']}xATR)"
        elif self._position_side == "sell":
            if close > self._entry_price + atr * p["sl_atr_mult"]:
                return True, f"止损({p['sl_atr_mult']}xATR)"

        # 软出场条件需满足最小持仓期
        if self._hold_bars < min_hold:
            return False, ""

        if self._position_side == "buy":
            if close > self._entry_price + atr * p["tp_atr_mult"]:
                return True, f"止盈({p['tp_atr_mult']}xATR)"
            if ratio < 0.42 and momentum < -0.008:
                return True, "攻方失势 R<0.42"
            if div_type == "bearish" and div_strength > 0.6:
                return True, f"看跌背离 div={div_strength:.2f}"

        elif self._position_side == "sell":
            if close < self._entry_price - atr * p["tp_atr_mult"]:
                return True, f"止盈({p['tp_atr_mult']}xATR)"
            if ratio > 0.58 and momentum > 0.008:
                return True, "防方失势 R>0.58"
            if div_type == "bullish" and div_strength > 0.6:
                return True, f"看涨背离 div={div_strength:.2f}"

        return False, ""

    # ──────────────────────────────────────────────
    #  主循环
    # ──────────────────────────────────────────────

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._update_forces(bar)

        if self._cd > 0:
            self._cd -= 1
        if self._position_side:
            self._hold_bars += 1

        p = self.config.params
        if self._bar_count < p["warmup_bars"]:
            return []

        c = float(bar.get("close", 0))
        atr = calc_atr(self._highs, self._lows, self._closes, p["atr_period"])
        if not atr or atr < _EPS:
            return []

        ratio = self._ratios[-1]
        momentum = self._momentums[-1]
        accel = self._force_acceleration()
        div_strength, div_type = self._detect_divergence()
        trend = self._trend_filter()

        signals: list[Signal] = []

        if self._position_side:
            should_exit, reason = self._check_exit(
                c, atr, ratio, momentum, div_strength, div_type
            )
            if should_exit:
                sig_type = (
                    SignalType.LONG_EXIT
                    if self._position_side == "buy"
                    else SignalType.SHORT_EXIT
                )
                signals.append(Signal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    signal_type=sig_type,
                    strength=0.9,
                    price=c,
                    reason=f"攻防出场: {reason}",
                    metadata={
                        "ratio": round(ratio, 4),
                        "attack": round(self._attack, 2),
                        "defense": round(self._defense, 2),
                        "momentum": round(momentum, 5),
                    },
                ))
                self._position_side = None
                self._entry_price = 0.0
                self._hold_bars = 0
                self._cd = p["cooldown_bars"]
        else:
            if self._cd <= 0:
                entry = self._check_entry(
                    ratio, momentum, accel, div_strength, div_type, trend, atr, c
                )
                if entry:
                    side, strength, reason = entry
                    sig_type = (
                        SignalType.LONG_ENTRY
                        if side == "buy"
                        else SignalType.SHORT_ENTRY
                    )
                    signals.append(Signal(
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        signal_type=sig_type,
                        strength=strength,
                        price=c,
                        reason=f"攻防入场: {reason}",
                        metadata={
                            "ratio": round(ratio, 4),
                            "attack": round(self._attack, 2),
                            "defense": round(self._defense, 2),
                            "momentum": round(momentum, 5),
                            "accel": round(accel, 5),
                            "divergence": div_type,
                            "trend": round(trend, 4),
                        },
                    ))
                    self._position_side = side
                    self._entry_price = c

        return signals

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        """批量生成信号（框架要求实现）。"""
        all_signals: list[Signal] = []
        for sym, bars in market_data.items():
            if isinstance(bars, list):
                for bar in bars:
                    all_signals.extend(await self.on_bar(sym, bar))
            elif isinstance(bars, dict):
                all_signals.extend(await self.on_bar(sym, bars))
        return all_signals

    # ──────────────────────────────────────────────
    #  诊断接口
    # ──────────────────────────────────────────────

    def get_state_snapshot(self) -> dict[str, Any]:
        """返回当前攻防状态快照，用于调试和可视化。"""
        total = self._attack + self._defense
        return {
            "attack": round(self._attack, 4),
            "defense": round(self._defense, 4),
            "ratio": round(self._attack / total, 4) if total > _EPS else 0.5,
            "momentum": round(self._momentums[-1], 5) if self._momentums else 0.0,
            "accel": round(self._force_acceleration(), 5),
            "position": self._position_side,
            "hold_bars": self._hold_bars,
            "bar_count": self._bar_count,
        }
