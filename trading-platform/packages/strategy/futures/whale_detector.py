"""庄家识别策略 — 假设庄家存在，跟随庄家方向交易。

技术路线（独立于其他策略，避免交叉影响）

====================================================================
案例研究库 — 全球期货市场操纵处罚案例提取的行为模式
====================================================================

【中国案例】

1. 甲醇1501案 (2014, 郑商所)
   操纵者: 姜为 (欣华欣总经理), 42个账户, 资金4.15亿
   手法: 囤积现货42万吨 + 期货买持仓76% → 多逼空
   特征: OI集中度极高(第二名仅439手 vs 27517手), 逆市反弹4%
   结局: 资金链断裂, 连续3日跌停20%, 86客户穿仓1.77亿
   处罚: 顶格罚款100万 + 终身禁入 + 有期徒刑2年6个月
   可量化信号: OI单边占比>50%, 逆势价格走势, 资金链断裂后OI骤降

2. 普麦1601案 (2015, 郑商所)
   操纵者: 廖山焱, 9个账户
   手法: 买持仓占比最高99.68%, 自买自卖对倒72.84%
   特征: 现货价格平稳(2400)但期货被拉至2719(+13%), 连续拉升4.88%/3.98%
   结局: 逼仓失败, 交割月连续跌停
   可量化信号: 量价背离(高对倒量+低真实价格变动), 期现价差异常扩大

3. 焦炭焦煤2101案 (2020-2021, 大商所)
   操纵者: 渤海融幸, 规避持仓限制
   手法: 多账户规避300手限额 → 形成持仓优势 → 约定交易73.89%
   特征: 约定交易拉涨6.30%, 交割月寻求有利平仓价
   处罚: 罚没2040万元
   可量化信号: 特定时段成交集中度异常, OI持仓突破限额

4. 秦某虚假申报案 (2020-2021, 多品种)
   手法: 大额报撤单+反向成交(Spoofing), 6品种9合约62次
   特征: 不以成交为目的的大单, 快速撤单后反向交易
   处罚: 罚没110万
   可量化信号: Order-to-Trade Ratio异常高, 大单存活时间极短

【美国案例】

5. Navinder Sarao Flash Crash案 (2010, CME E-mini S&P 500)
   手法: 自制动态分层算法(Layering Algorithm), 4-6层大额卖单
   特征: 卖单占订单簿20-29%, $170-200M下行压力, 当日修改8.1万次仅成交81手
   结局: 触发2010年5月6日"Flash Crash", 道指瞬跌近1000点
   处罚: 罚款$38M, 认罪协议, 获利$12.8M
   可量化信号: 订单簿单边深度骤增, 极高撤单/修改率, 价格异常波动

6. JPMorgan Spoofing案 (2008-2016, COMEX/NYMEX/CBOT)
   手法: 贵金属+国债期货, 数十万虚假订单(hundreds of thousands)
   特征: 8年持续操纵, 涉及金/银/铂/钯/国债多品种
   处罚: $920M (CFTC史上最高罚款), 含$436M罚金+$312M赔偿+$172M退赃
   可量化信号: 跨品种同步异常订单模式, 长期持续的spoofing特征

【英国/欧洲案例】

7. Michael Coscia案 (2013, ICE Futures Europe)
   手法: 大宗商品期货layering, 快速挂撤单
   处罚: FCA罚款£597K, 后在美国也被起诉
   可量化信号: 多层同向订单+反向真实成交

8. Da Vinci Invest案 (2015, UK)
   手法: 算法交易layering, 制造锯齿形(saw-tooth)价格模式
   处罚: £7.5M
   可量化信号: 价格锯齿形模式, 算法驱动的规律性挂撤

9. Mizuho BTP期货案 (2016, Eurex)
   手法: 意大利国债期货spoofing, 大单(200+手)制造虚假深度
   特征: 3名交易员233次操纵, 大单距最优价3-4档
   处罚: 3名交易员终身禁入 (2025年上诉庭维持)
   可量化信号: 远离最优价的大单频繁出现又消失

====================================================================
五种操纵模式及检测方法
====================================================================

模式A - 持仓集中/逼仓 (Position Concentration)
  案例: 甲醇1501, 普麦1601, 焦炭焦煤2101
  信号: OI单边占比骤增, 持仓增速z-score异常

模式B - 虚假申报/Spoofing
  案例: 秦某, Sarao, JPMorgan, Coscia, Mizuho
  信号: 需tick级数据(本策略用bar级近似), 成交量/OI增量不匹配

模式C - 分层/Layering
  案例: Sarao, JPMorgan, Da Vinci
  信号: 价格锯齿形模式, 短周期方差比偏离

模式D - 对倒/Wash Trading
  案例: 普麦1601
  信号: 高成交量+低价格变动(量价背离)

模式E - 逼仓崩盘
  案例: 甲醇1501, 普麦1601
  信号: OI急降+价格暴跌+VR偏离
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..base import BaseStrategy, Signal, SignalType, StrategyConfig
from ..indicators import calc_atr, rolling_zscore, ema_update
from ..registry import auto_register

logger = logging.getLogger(__name__)


# ── 可调参数（增强灵活度） ──

DEFAULT_PARAMS: dict[str, Any] = {
    # ── 检测窗口 ──
    "oi_window": 30,
    "volume_window": 20,
    "price_window": 20,
    "vr_window": 10,

    # ── 异常阈值（可通过训练/Optuna优化） ──
    "oi_surge_zscore": 2.5,
    "volume_surge_zscore": 2.0,
    "vpd_threshold": 0.3,
    "vr_deviation": 0.4,

    # ── 模式权重（灵活调节各检测模式的影响力） ──
    "weight_oi_concentration": 1.0,    # 模式A
    "weight_spoofing_proxy": 0.8,      # 模式B (bar级近似)
    "weight_wash_trading": 1.0,        # 模式D
    "weight_vr_deviation": 0.6,        # 模式C
    "weight_crash_signal": 1.2,        # 模式E

    # ── 阶段识别 ──
    "accumulation_oi_bars": 5,
    "distribution_oi_bars": 3,
    "breakout_pct": 0.015,

    # ── 交易参数 ──
    "atr_period": 14,
    "tp_atr_mult": 3.0,
    "sl_atr_mult": 1.5,
    "max_hold_bars": 60,
    "cooldown_bars": 5,
    "min_signal_strength": 0.5,

    # ── 训练/标记参数 ──
    "feature_export": False,
    "label_column": None,
}


class WhalePhase:
    UNKNOWN = "unknown"
    ACCUMULATION = "accumulation"
    MARKUP = "markup"
    DISTRIBUTION = "distribution"
    MARKDOWN = "markdown"


@dataclass
class WhaleFeatureRow:
    """单个 bar 的庄家特征向量，用于训练/回测分析。"""
    bar_idx: int
    timestamp: str
    close: float
    volume: float
    oi: float
    oi_zscore: float
    vol_zscore: float
    vpd_score: float
    variance_ratio: float
    spoofing_proxy: float
    oi_vol_mismatch: float
    phase: str
    composite_score: float
    signal_side: str = ""
    signal_strength: float = 0.0
    label: int = -1  # -1=unlabeled, 0=no_whale, 1=whale_accumulation, 2=markup, 3=distribution


@auto_register("whale_detector")
class WhaleDetectorStrategy(BaseStrategy):
    """庄家识别跟庄策略。

    通过 OI/Volume/Price 多维异常检测识别庄家行为阶段，支持：
    - 5种操纵模式的加权检测
    - 可配置权重（Optuna 可调）
    - 特征导出（训练用）
    - 标签回测（有标签/无标签数据均可）
    """

    def __init__(self, config: StrategyConfig) -> None:
        config = config.model_copy(
            update={"params": {**DEFAULT_PARAMS, **config.params}}
        )
        super().__init__(config)

        self._highs: deque[float] = deque(maxlen=500)
        self._lows: deque[float] = deque(maxlen=500)
        self._closes: deque[float] = deque(maxlen=500)
        self._volumes: deque[float] = deque(maxlen=500)
        self._ois: deque[float] = deque(maxlen=500)

        self._oi_deltas: deque[float] = deque(maxlen=500)
        self._returns: deque[float] = deque(maxlen=500)
        self._vol_oi_ratios: deque[float] = deque(maxlen=500)

        self._bar_count = 0
        self._position_side: str | None = None
        self._entry_price = 0.0
        self._hold_bars = 0
        self._cd = 0

        self._phase = WhalePhase.UNKNOWN
        self._phase_bars = 0
        self._oi_trend_count = 0
        self._whale_direction: str | None = None

        self.feature_history: list[WhaleFeatureRow] = []

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        h = float(bar.get("high", 0))
        l = float(bar.get("low", 0))
        c = float(bar.get("close", 0))
        v = float(bar.get("volume", 0))
        oi = float(bar.get("open_interest", bar.get("oi", 0)))
        ts = str(bar.get("datetime", bar.get("timestamp", "")))

        self._highs.append(h)
        self._lows.append(l)
        self._closes.append(c)
        self._volumes.append(v)
        self._ois.append(oi)
        self._bar_count += 1

        if self._bar_count > 1:
            self._oi_deltas.append(oi - self._ois[-2])
            ret = (c - self._closes[-2]) / self._closes[-2] if self._closes[-2] else 0
            self._returns.append(ret)
            oi_delta = abs(self._oi_deltas[-1]) if self._oi_deltas[-1] else 1e-10
            self._vol_oi_ratios.append(v / max(oi_delta, 1e-10))

        if self._cd > 0:
            self._cd -= 1
        if self._position_side:
            self._hold_bars += 1

        p = self.config.params
        min_warmup = max(p["oi_window"], p["volume_window"], p["price_window"], p["atr_period"] + 1)
        if self._bar_count < min_warmup:
            return []

        atr = calc_atr(self._highs, self._lows, self._closes, p["atr_period"])
        if not atr or atr < 1e-10:
            return []

        # ── 计算全部检测维度 ──
        oi_zscore = self._calc_oi_zscore()
        vol_zscore = self._calc_volume_zscore()
        vpd_score = self._calc_volume_price_divergence()
        vr = self._calc_variance_ratio()
        spoof_proxy = self._calc_spoofing_proxy()
        oi_vol_mismatch = self._calc_oi_vol_mismatch()

        composite = self._calc_composite_score(
            oi_zscore, vol_zscore, vpd_score, vr, spoof_proxy, oi_vol_mismatch,
        )
        phase = self._detect_phase(oi_zscore, vol_zscore, vpd_score, vr, c, atr)

        # ── 特征导出（训练用）──
        if p.get("feature_export"):
            label_col = p.get("label_column")
            label = int(bar.get(label_col, -1)) if label_col else -1
            self.feature_history.append(WhaleFeatureRow(
                bar_idx=self._bar_count,
                timestamp=ts,
                close=c, volume=v, oi=oi,
                oi_zscore=oi_zscore,
                vol_zscore=vol_zscore,
                vpd_score=vpd_score,
                variance_ratio=vr,
                spoofing_proxy=spoof_proxy,
                oi_vol_mismatch=oi_vol_mismatch,
                phase=phase,
                composite_score=composite,
                label=label,
            ))

        # ── 交易信号 ──
        signals: list[Signal] = []

        if self._position_side:
            should_exit, reason = self._check_exit(c, atr, phase)
            if should_exit:
                sig_type = SignalType.LONG_EXIT if self._position_side == "buy" else SignalType.SHORT_EXIT
                signals.append(Signal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    signal_type=sig_type,
                    strength=0.9,
                    price=c,
                    reason=f"庄家出场: {reason}",
                    metadata={"phase": phase, "composite": composite},
                ))
                self._position_side = None
                self._entry_price = 0.0
                self._hold_bars = 0
                self._cd = p["cooldown_bars"]
        else:
            if self._cd <= 0:
                entry = self._check_entry(
                    oi_zscore, vol_zscore, vpd_score, vr,
                    spoof_proxy, oi_vol_mismatch, composite,
                    c, atr, phase,
                )
                if entry:
                    side, strength, reason = entry
                    sig_type = SignalType.LONG_ENTRY if side == "buy" else SignalType.SHORT_ENTRY
                    signals.append(Signal(
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        signal_type=sig_type,
                        strength=strength,
                        price=c,
                        reason=reason,
                        metadata={
                            "phase": phase,
                            "oi_z": oi_zscore, "vol_z": vol_zscore,
                            "vpd": vpd_score, "vr": vr,
                            "spoof": spoof_proxy, "mismatch": oi_vol_mismatch,
                            "composite": composite,
                        },
                    ))
                    self._position_side = side
                    self._entry_price = c
                    self._hold_bars = 0

                    if p.get("feature_export") and self.feature_history:
                        self.feature_history[-1].signal_side = side
                        self.feature_history[-1].signal_strength = strength

        for sig in signals:
            self.record_signal(sig)
        return signals

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        return []

    def get_features_array(self) -> np.ndarray:
        """导出特征矩阵 (N x 7)，用于外部训练。"""
        if not self.feature_history:
            return np.empty((0, 7))
        return np.array([
            [f.oi_zscore, f.vol_zscore, f.vpd_score, f.variance_ratio,
             f.spoofing_proxy, f.oi_vol_mismatch, f.composite_score]
            for f in self.feature_history
        ])

    def get_labels_array(self) -> np.ndarray:
        """导出标签向量，-1 表示无标签。"""
        if not self.feature_history:
            return np.empty(0, dtype=int)
        return np.array([f.label for f in self.feature_history], dtype=int)

    # ── 检测引擎: 5 种模式 ──

    def _calc_oi_zscore(self) -> float:
        """模式A: OI 变化率 z-score — 持仓集中度异常。

        甲醇1501: 买持仓从30.75%升至76.04%, z-score会非常高。
        """
        z = rolling_zscore(list(self._oi_deltas), self.config.params["oi_window"])
        return z if z is not None else 0.0

    def _calc_volume_zscore(self) -> float:
        """成交量 z-score。"""
        z = rolling_zscore(list(self._volumes), self.config.params["volume_window"])
        return z if z is not None else 0.0

    def _calc_volume_price_divergence(self) -> float:
        """模式D: 量价背离度 — 对倒/洗盘检测。

        普麦1601: 自买自卖对倒72.84%, 成交量巨大但价格仅微变。
        高 vpd = 高成交量但低价格影响 = 操纵嫌疑。
        """
        p = self.config.params
        if len(self._returns) < p["price_window"] or len(self._volumes) < p["volume_window"]:
            return 0.0

        vol_z = rolling_zscore(list(self._volumes), p["volume_window"])
        price_z = rolling_zscore(
            [abs(r) for r in list(self._returns)[-p["price_window"]:]],
            p["price_window"],
        )

        if vol_z is None or price_z is None or abs(vol_z) < 0.5:
            return 0.0
        return abs(vol_z) / max(abs(price_z), 0.1)

    def _calc_variance_ratio(self) -> float:
        """模式C: Variance Ratio 检验 — 偏离随机游走意味人为操纵。

        VR > 1: 正序列相关 (趋势被人为维持, 如Sarao的layering)
        VR < 1: 负序列相关 (pump-dump/锯齿形, 如Da Vinci)
        """
        p = self.config.params
        q = p["vr_window"]
        rets = list(self._returns)
        if len(rets) < q * 3:
            return 1.0

        r1 = rets[-q * 3:]
        var1 = float(np.var(r1))
        if var1 < 1e-15:
            return 1.0

        rq = [sum(r1[i:i + q]) for i in range(0, len(r1) - q + 1, q)]
        varq = float(np.var(rq))
        return varq / (q * var1) if var1 > 0 else 1.0

    def _calc_spoofing_proxy(self) -> float:
        """模式B: Spoofing 近似检测 (bar级)。

        无tick数据时的替代方案：
        高成交量 + OI几乎不变 = 大量对冲/撤单(类似spoofing效果)
        Sarao: 修改8.1万次仅成交81手, 意味着成交/OI变化极小。
        """
        if len(self._volumes) < 5 or len(self._oi_deltas) < 5:
            return 0.0

        recent_vol = list(self._volumes)[-5:]
        recent_oi_abs = [abs(d) for d in list(self._oi_deltas)[-5:]]

        avg_vol = sum(recent_vol) / len(recent_vol)
        avg_oi_change = sum(recent_oi_abs) / len(recent_oi_abs)

        if avg_oi_change < 1e-10:
            return min(avg_vol / 1000, 5.0) if avg_vol > 0 else 0.0

        ratio = avg_vol / avg_oi_change
        vol_z = rolling_zscore(list(self._volumes), self.config.params["volume_window"])

        if vol_z is not None and vol_z > 1.5 and ratio > 50:
            return min(ratio / 100, 5.0)
        return 0.0

    def _calc_oi_vol_mismatch(self) -> float:
        """OI-Volume 不匹配度: OI 大变但成交量不够大(或反之)。

        焦炭焦煤案: 规避限额, OI在限额附近集中, 但成交方式不自然。
        """
        if len(self._vol_oi_ratios) < self.config.params["volume_window"]:
            return 0.0
        z = rolling_zscore(list(self._vol_oi_ratios), self.config.params["volume_window"])
        return abs(z) if z is not None else 0.0

    def _calc_composite_score(
        self,
        oi_z: float, vol_z: float, vpd: float, vr: float,
        spoof: float, mismatch: float,
    ) -> float:
        """加权复合得分 — 综合所有检测模式。"""
        p = self.config.params
        raw = (
            abs(oi_z) * p["weight_oi_concentration"]
            + vpd * p["weight_wash_trading"]
            + spoof * p["weight_spoofing_proxy"]
            + abs(vr - 1.0) * p["weight_vr_deviation"] * 5
            + mismatch * 0.3
        )
        return min(raw / 5.0, 1.0)

    def _detect_phase(
        self,
        oi_zscore: float, vol_zscore: float, vpd_score: float,
        vr: float, close: float, atr: float,
    ) -> str:
        """识别庄家行为阶段。"""
        p = self.config.params

        if len(self._oi_deltas) < 3:
            return WhalePhase.UNKNOWN

        recent = list(self._oi_deltas)[-p["accumulation_oi_bars"]:]
        up = sum(1 for d in recent if d > 0)
        down = sum(1 for d in recent if d < 0)

        if up >= len(recent) - 1:
            self._oi_trend_count = max(self._oi_trend_count + 1, 1)
        elif down >= len(recent) - 1:
            self._oi_trend_count = min(self._oi_trend_count - 1, -1)
        else:
            self._oi_trend_count = 0

        if (self._oi_trend_count >= p["accumulation_oi_bars"]
                and abs(vol_zscore) < p["volume_surge_zscore"]
                and vpd_score > p["vpd_threshold"]):
            self._phase = WhalePhase.ACCUMULATION
            self._whale_direction = "long" if oi_zscore > 0 else "short"

        elif (self._phase == WhalePhase.ACCUMULATION
              and self._oi_trend_count > 0
              and vol_zscore > p["volume_surge_zscore"]
              and len(self._closes) > 5):
            recent_move = (close - self._closes[-5]) / close
            if abs(recent_move) > p["breakout_pct"]:
                self._phase = WhalePhase.MARKUP
                self._whale_direction = "long" if recent_move > 0 else "short"

        elif (self._oi_trend_count <= -p["distribution_oi_bars"]
              and vol_zscore > 1.0):
            self._phase = WhalePhase.DISTRIBUTION

        elif (oi_zscore < -p["oi_surge_zscore"]
              and len(self._returns) > 0
              and abs(self._returns[-1]) > 2 * atr / close
              and abs(vr - 1.0) > p["vr_deviation"]):
            self._phase = WhalePhase.MARKDOWN

        else:
            if self._phase not in (WhalePhase.ACCUMULATION, WhalePhase.MARKUP):
                self._phase = WhalePhase.UNKNOWN

        self._phase_bars += 1
        return self._phase

    # ── 入场 ──

    def _check_entry(
        self,
        oi_z: float, vol_z: float, vpd: float, vr: float,
        spoof: float, mismatch: float, composite: float,
        close: float, atr: float, phase: str,
    ) -> tuple[str, float, str] | None:
        p = self.config.params

        # 吸筹期 → 跟随建仓
        if phase == WhalePhase.ACCUMULATION and oi_z > p["oi_surge_zscore"]:
            direction = self._whale_direction or "long"
            side = "buy" if direction == "long" else "sell"
            strength = min(1.0, composite * 1.2)
            if strength >= p["min_signal_strength"]:
                return (
                    side, strength,
                    f"吸筹跟庄: OI_z={oi_z:.1f} VPD={vpd:.2f} VR={vr:.2f} comp={composite:.2f}",
                )

        # 拉升确认 → 追涨
        if phase == WhalePhase.MARKUP and self._whale_direction:
            side = "buy" if self._whale_direction == "long" else "sell"
            strength = min(1.0, composite * 1.1)
            if strength >= p["min_signal_strength"]:
                return (
                    side, strength,
                    f"拉升跟庄: dir={self._whale_direction} vol_z={vol_z:.1f} comp={composite:.2f}",
                )

        # 砸盘 → 反手做空（高门槛）
        if (phase == WhalePhase.MARKDOWN
                and oi_z < -3.0
                and abs(vr - 1.0) > p["vr_deviation"] * 1.5):
            strength = min(1.0, abs(oi_z) / 5.0 + abs(vr - 1.0))
            if strength >= 0.7:
                return (
                    "sell", strength,
                    f"砸盘做空: OI_z={oi_z:.1f} VR={vr:.2f} comp={composite:.2f}",
                )

        # 高复合得分但无明确阶段 → 弱信号（可用于训练标注）
        if composite > 0.8 and phase == WhalePhase.UNKNOWN and oi_z > 2.0:
            direction = "long" if oi_z > 0 else "short"
            side = "buy" if direction == "long" else "sell"
            strength = composite * 0.7
            if strength >= p["min_signal_strength"]:
                return (
                    side, strength,
                    f"高异常得分: comp={composite:.2f} OI_z={oi_z:.1f} spoof={spoof:.2f}",
                )

        return None

    # ── 出场 ──

    def _check_exit(self, close: float, atr: float, phase: str) -> tuple[bool, str]:
        p = self.config.params

        if self._hold_bars >= p["max_hold_bars"]:
            return True, f"最大持仓({p['max_hold_bars']}bars)"

        if self._position_side == "buy":
            if close < self._entry_price - atr * p["sl_atr_mult"]:
                return True, f"止损({p['sl_atr_mult']}xATR)"
            if close > self._entry_price + atr * p["tp_atr_mult"]:
                return True, f"止盈({p['tp_atr_mult']}xATR)"
        elif self._position_side == "sell":
            if close > self._entry_price + atr * p["sl_atr_mult"]:
                return True, f"止损({p['sl_atr_mult']}xATR)"
            if close < self._entry_price - atr * p["tp_atr_mult"]:
                return True, f"止盈({p['tp_atr_mult']}xATR)"

        if phase == WhalePhase.DISTRIBUTION:
            return True, "出货信号(OI下降+放量)"

        if phase == WhalePhase.MARKDOWN and self._position_side == "buy":
            return True, "砸盘信号(紧急离场)"

        return False, ""


# ── Optuna 参数空间模板（用于超参搜索） ──

OPTUNA_PARAM_SPACE: dict[str, tuple[str, Any, Any]] = {
    "oi_window": ("int", 10, 60),
    "volume_window": ("int", 10, 40),
    "price_window": ("int", 10, 40),
    "vr_window": ("int", 5, 20),
    "oi_surge_zscore": ("float", 1.5, 4.0),
    "volume_surge_zscore": ("float", 1.0, 3.5),
    "vpd_threshold": ("float", 0.1, 1.0),
    "vr_deviation": ("float", 0.2, 0.8),
    "weight_oi_concentration": ("float", 0.3, 2.0),
    "weight_spoofing_proxy": ("float", 0.3, 2.0),
    "weight_wash_trading": ("float", 0.3, 2.0),
    "weight_vr_deviation": ("float", 0.2, 1.5),
    "weight_crash_signal": ("float", 0.5, 2.0),
    "accumulation_oi_bars": ("int", 3, 10),
    "distribution_oi_bars": ("int", 2, 6),
    "breakout_pct": ("float", 0.005, 0.03),
    "tp_atr_mult": ("float", 2.0, 5.0),
    "sl_atr_mult": ("float", 1.0, 3.0),
    "max_hold_bars": ("int", 20, 120),
    "cooldown_bars": ("int", 2, 10),
    "min_signal_strength": ("float", 0.3, 0.8),
}
