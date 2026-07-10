#!/usr/bin/env python3
"""BTC surviving-factor → strategy backtest with strict no-leakage discipline.

Dedup / sign / weights / params use TRAIN only. TEST is a one-shot evaluation.

Usage:
    PYTHONPATH=packages .venv/bin/python3 scripts/run_btc_factor_strategy_backtest.py
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ROOT / "packages"
if str(PACKAGES) not in sys.path:
    sys.path.insert(0, str(PACKAGES))

from backtest.datafeed import BarDataFeed  # noqa: E402
from backtest.engine import BacktestEngine  # noqa: E402
from backtest.futures_matrix import result_to_report_dict  # noqa: E402
from backtest.models import BacktestConfig, Bar  # noqa: E402
from backtest.strategy_adapter import StrategyAdapter  # noqa: E402
from datahub.crypto_loader import CryptoDataLoader  # noqa: E402
from factor.combine import dynamic_combine  # noqa: E402
from factor.evolution import evaluate_expression  # noqa: E402
from factor.registry import get_registry  # noqa: E402
from strategy.base import BaseStrategy, Signal, SignalType, StrategyConfig  # noqa: E402
from strategy.templates.factor_strategy import FactorStrategy  # noqa: E402
from strategy.templates.supertrend import SupertrendStrategy  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("btc-factor-strategy-bt")

OHLCV_COLS = ("datetime", "open", "high", "low", "close", "volume")
DEFAULT_COMMISSION = Decimal("0.0005")
DEFAULT_INITIAL_CAPITAL = Decimal("1000000")
METRIC_KEYS = (
    "total_return",
    "annual_return",
    "sharpe",
    "max_drawdown_pct",
    "win_rate",
    "profit_factor",
    "total_trades",
    "turnover",
    "cost_ratio",
)


# ---------------------------------------------------------------------------
# Data helpers (aligned with run_btc_factor_mining.py)
# ---------------------------------------------------------------------------


def prepare_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "datetime" not in out.columns and "open_time" in out.columns:
        out = out.rename(columns={"open_time": "datetime"})
    missing = [c for c in OHLCV_COLS if c not in out.columns]
    if missing:
        raise ValueError(f"OHLCV missing columns: {missing}")
    out = out[list(OHLCV_COLS)].copy()
    out["datetime"] = pd.to_datetime(out["datetime"], utc=True)
    out = out.sort_values("datetime").reset_index(drop=True)
    return out


def load_btc_ohlcv(
    symbol: str = "BTCUSDT",
    timeframe: str = "1h",
    data_dir: str | None = None,
) -> pd.DataFrame:
    loader = CryptoDataLoader(data_dir)
    raw = loader.load(symbol, timeframe=timeframe)
    if raw.empty:
        raise FileNotFoundError(f"No data for {symbol} {timeframe} (data_dir={loader.data_dir})")
    return prepare_ohlcv(raw)


def split_train_test(
    df: pd.DataFrame,
    train_bars: int,
    test_bars: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tail split: earlier train_bars then later test_bars; identical to mining script."""
    need = train_bars + test_bars
    if len(df) < need:
        raise ValueError(f"Need >= {need} bars, got {len(df)}")
    tail = df.iloc[-need:].reset_index(drop=True)
    train = tail.iloc[:train_bars].reset_index(drop=True)
    test = tail.iloc[train_bars:].reset_index(drop=True)
    assert len(train) == train_bars and len(test) == test_bars
    assert train["datetime"].iloc[-1] < test["datetime"].iloc[0]
    return train, test


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BTC factor strategy backtest (no leakage)")
    p.add_argument(
        "--factors-json",
        type=str,
        default=str(ROOT / "output" / "research" / "btc_surviving_factors.json"),
    )
    p.add_argument("--symbol", type=str, default="BTCUSDT")
    p.add_argument("--timeframe", type=str, default="1h")
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--train-bars", type=int, default=30000)
    p.add_argument("--test-bars", type=int, default=8000)
    p.add_argument(
        "--dedup-eval-bars",
        type=int,
        default=20000,
        help="Use last N train bars for dedup eval (speed). 0 = full train.",
    )
    p.add_argument("--corr-threshold", type=float, default=0.7)
    p.add_argument("--entry-z", type=float, default=1.0, help="FactorStrategy entry_z (default; not tuned on test)")
    p.add_argument("--exit-z", type=float, default=0.3, help="FactorStrategy exit_z (default; not tuned on test)")
    p.add_argument("--composite-entry-z", type=float, default=1.0)
    p.add_argument("--composite-exit-z", type=float, default=0.5)
    p.add_argument("--zscore-window", type=int, default=60)
    p.add_argument("--commission-rate", type=float, default=0.0005)
    p.add_argument("--slippage-bps", type=float, default=1.0, help="Slippage in bps (1bp default)")
    p.add_argument("--initial-capital", type=float, default=1_000_000.0)
    p.add_argument("--default-volume", type=int, default=1)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Dedup (TRAIN only)
# ---------------------------------------------------------------------------


def load_surviving_factors(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    factors = data.get("factors") or []
    out: list[dict[str, Any]] = []
    for f in factors:
        expr = str(f.get("expr") or "").strip()
        if not expr:
            continue
        out.append(
            {
                "expr": expr,
                "train_ic": float(f["train_ic"]) if f.get("train_ic") is not None else None,
                "test_ic": float(f["test_ic"]) if f.get("test_ic") is not None else None,
                "test_ir": float(f["test_ir"]) if f.get("test_ir") is not None else None,
            }
        )
    return out


def evaluate_factors_on_df(df: pd.DataFrame, exprs: list[str]) -> pd.DataFrame:
    """Evaluate expressions; columns are exprs. Failures → all-NaN column."""
    cols: dict[str, pd.Series] = {}
    for expr in exprs:
        try:
            s = evaluate_expression(expr, df)
            cols[expr] = pd.Series(s.to_numpy(dtype=float), index=df.index)
        except Exception as exc:  # noqa: BLE001
            logger.warning("eval failed for %s: %s", expr[:60], exc)
            cols[expr] = pd.Series(np.nan, index=df.index)
    return pd.DataFrame(cols, index=df.index)


def greedy_cluster_by_corr(
    factors: list[dict[str, Any]],
    factor_frame: pd.DataFrame,
    *,
    threshold: float = 0.7,
) -> list[dict[str, Any]]:
    """Greedy clustering on |Spearman ρ|; keep highest |train_ic| as representative.

    Decision uses only train_ic + train-segment factor values (caller must pass train frame).
    """
    usable = [f for f in factors if f.get("train_ic") is not None and f["expr"] in factor_frame.columns]
    usable.sort(key=lambda f: abs(float(f["train_ic"])), reverse=True)

    corr = factor_frame[[f["expr"] for f in usable]].corr(method="spearman")
    clusters: list[dict[str, Any]] = []
    assigned: set[str] = set()

    for fac in usable:
        expr = fac["expr"]
        if expr in assigned:
            continue
        members = [fac]
        assigned.add(expr)
        for other in usable:
            oexpr = other["expr"]
            if oexpr in assigned:
                continue
            rho = corr.loc[expr, oexpr]
            if rho == rho and abs(float(rho)) >= threshold:
                members.append(other)
                assigned.add(oexpr)
        clusters.append(
            {
                "representative": fac,
                "size": len(members),
                "members": members,
            }
        )
    return clusters


def signed_weights_from_train_ic(reps: list[dict[str, Any]]) -> dict[str, float]:
    """Sign = sign(train_ic); magnitude = |train_ic| normalized. TRAIN only."""
    ics = [abs(float(r["train_ic"])) for r in reps]
    total = sum(ics) or 1.0
    weights: dict[str, float] = {}
    for i, r in enumerate(reps):
        sign = 1.0 if float(r["train_ic"]) >= 0 else -1.0
        weights[f"mined_btc_{i}"] = sign * (abs(float(r["train_ic"])) / total)
    return weights


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def register_mined_factors(reps: list[dict[str, Any]]) -> list[str]:
    """Register representative exprs into FactorRegistry (evolution_registry style)."""
    registry = get_registry()
    names: list[str] = []
    for i, rep in enumerate(reps):
        name = f"mined_btc_{i}"
        expr = rep["expr"]

        def _make_compute(expression: str, col: str):
            def compute(df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
                series = evaluate_expression(expression, df)
                out = df.copy()
                out[col] = series
                return out

            return compute

        registry.register(
            name=name,
            category="mined_btc",
            description=f"Mined BTC: {expr}",
            output_columns=[name],
            compute_fn=_make_compute(expr, name),
            expr=expr,
            train_ic=rep.get("train_ic"),
        )
        names.append(name)
        logger.info("Registered %s ← %s", name, expr[:80])
    return names


# ---------------------------------------------------------------------------
# Strategies (thin wrappers for precomputed signals / buy&hold)
# ---------------------------------------------------------------------------


class BarExtraFactorStrategy(FactorStrategy):
    """FactorStrategy that reads precomputed factor values from bar extras.

    Avoids per-bar FeatureEngine recompute while keeping FactorStrategy entry/exit
    and params.factors semantics. Factors are still registered in FactorRegistry.
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self._last_bar: dict[str, dict[str, Any]] = {}

    def factor_values(self, symbol: str) -> dict[str, float | None]:
        bar = self._last_bar.get(symbol) or {}
        out: dict[str, float | None] = {}
        for name in self._factor_weights:
            v = bar.get(name)
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                out[name] = None
            else:
                try:
                    out[name] = float(v)
                except (TypeError, ValueError):
                    out[name] = None
        return out

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        self._last_bar[symbol] = bar
        return await super().on_bar(symbol, bar)


class CompositeZStrategy(BaseStrategy):
    """Trade a precomputed composite z-score carried on the bar as ``signal_z``."""

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self._entry_z = float(self.get_param("entry_z", 1.0))
        self._exit_z = float(self.get_param("exit_z", 0.5))
        self._allow_short = bool(self.get_param("allow_short", True))
        self._pos_side: dict[str, str | None] = {}

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        raw = bar.get("signal_z")
        if raw is None:
            return []
        try:
            score = float(raw)
        except (TypeError, ValueError):
            return []
        if math.isnan(score) or math.isinf(score):
            return []

        close = float(bar["close"])
        strength = min(abs(score), 1.0)
        side = self._pos_side.get(symbol)
        signals: list[Signal] = []

        if side is None:
            if score > self._entry_z:
                signals.append(
                    Signal(
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        signal_type=SignalType.LONG_ENTRY,
                        strength=strength,
                        price=close,
                        reason=f"composite_z={score:.3f}",
                    )
                )
                self._pos_side[symbol] = "long"
            elif score < -self._entry_z and self._allow_short:
                signals.append(
                    Signal(
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        signal_type=SignalType.SHORT_ENTRY,
                        strength=strength,
                        price=close,
                        reason=f"composite_z={score:.3f}",
                    )
                )
                self._pos_side[symbol] = "short"
        elif side == "long":
            if abs(score) < self._exit_z:
                signals.append(
                    Signal(
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        signal_type=SignalType.LONG_EXIT,
                        strength=strength,
                        price=close,
                        reason=f"composite_exit z={score:.3f}",
                    )
                )
                self._pos_side[symbol] = None
        elif side == "short":
            if abs(score) < self._exit_z:
                signals.append(
                    Signal(
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        signal_type=SignalType.SHORT_EXIT,
                        strength=strength,
                        price=close,
                        reason=f"composite_exit z={score:.3f}",
                    )
                )
                self._pos_side[symbol] = None

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


class BuyAndHoldStrategy(BaseStrategy):
    """Long once on first bar, hold forever."""

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self._entered = False

    async def on_bar(self, symbol: str, bar: dict[str, Any]) -> list[Signal]:
        if self._entered:
            return []
        self._entered = True
        return [
            Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                signal_type=SignalType.LONG_ENTRY,
                strength=1.0,
                price=float(bar["close"]),
                reason="buy_and_hold",
            )
        ]

    async def generate_signals(self, market_data: dict[str, Any]) -> list[Signal]:
        out: list[Signal] = []
        for symbol in self.config.symbols:
            bar = market_data.get(symbol)
            if bar:
                out.extend(await self.on_bar(symbol, bar))
        return out


# ---------------------------------------------------------------------------
# Backtest runner with cost metrics
# ---------------------------------------------------------------------------


def _to_naive_dt(value: Any) -> datetime:
    """Coerce to timezone-naive datetime for BacktestEngine comparisons."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.to_pydatetime()


def dataframe_to_bars_with_extra(
    df: pd.DataFrame,
    symbol: str,
    extra_cols: list[str] | None = None,
) -> list[Bar]:
    extra_cols = extra_cols or []
    bars: list[Bar] = []
    for _, row in df.iterrows():
        extra: dict[str, Any] = {}
        for c in extra_cols:
            if c not in row.index:
                continue
            v = row[c]
            if pd.isna(v):
                continue
            extra[c] = float(v)
        bars.append(
            Bar(
                symbol=symbol,
                dt=_to_naive_dt(row["datetime"]),
                open=Decimal(str(row["open"])),
                high=Decimal(str(row["high"])),
                low=Decimal(str(row["low"])),
                close=Decimal(str(row["close"])),
                volume=int(float(row.get("volume", 0) or 0)),
                extra=extra,
            )
        )
    return bars


def run_strategy_backtest(
    strategy: BaseStrategy,
    bars_df: pd.DataFrame,
    *,
    symbol: str,
    initial_capital: Decimal,
    commission_rate: Decimal,
    slippage_ticks: int,
    tick_size: Decimal,
    default_volume: int = 1,
    extra_cols: list[str] | None = None,
) -> dict[str, Any]:
    if bars_df.empty:
        raise ValueError("bars empty")

    start = _to_naive_dt(bars_df["datetime"].iloc[0])
    end = _to_naive_dt(bars_df["datetime"].iloc[-1])

    config = BacktestConfig(
        strategy_id=strategy.config.strategy_id,
        symbols=[symbol],
        start_date=start,
        end_date=end,
        initial_capital=initial_capital,
        commission_rate=commission_rate,
        slippage_ticks=slippage_ticks,
        tick_size=tick_size,
        contract_multiplier=1,
    )
    engine = BacktestEngine(config)
    feed = BarDataFeed(engine.event_bus)
    feed.add_bars(dataframe_to_bars_with_extra(bars_df, symbol, extra_cols))
    engine.set_datafeed(feed)
    engine.set_strategy(StrategyAdapter(strategy, default_volume=default_volume))

    t0 = time.monotonic()
    result = engine.run()
    duration_s = time.monotonic() - t0

    report = result_to_report_dict(
        result,
        strategy=strategy.config.strategy_id,
        symbol=symbol,
        bars=len(bars_df),
        duration_s=duration_s,
    )

    total_commission = sum((float(t.commission) for t in result.trades), 0.0)
    total_slippage = sum((float(t.slippage) for t in result.trades), 0.0)
    total_notional = sum(
        (float(t.price) * int(t.volume) * float(config.contract_multiplier) for t in result.trades),
        0.0,
    )
    cap = float(initial_capital)
    report["total_commission"] = round(total_commission, 4)
    report["total_slippage"] = round(total_slippage, 4)
    report["cost_ratio"] = round(total_commission / cap, 6) if cap else 0.0
    report["turnover"] = round(total_notional / cap, 6) if cap else 0.0
    report["name"] = strategy.config.strategy_id
    # Normalize aliases expected by tests / report
    report["sharpe"] = report.get("sharpe", report.get("sharpe_ratio", 0.0))
    report["max_drawdown_pct"] = report.get("max_drawdown_pct", report.get("max_dd", 0.0))
    report["total_trades"] = report.get("total_trades", report.get("trades", 0))
    report["annual_return"] = report.get("annual_return", 0.0)
    report["win_rate"] = report.get("win_rate", 0.0)
    report["profit_factor"] = report.get("profit_factor", 0.0)
    report["total_return"] = report.get("total_return", 0.0)
    return report


def _causal_zscore_series(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window, min_periods=max(2, window // 2)).mean()
    sig = s.rolling(window, min_periods=max(2, window // 2)).std(ddof=0)
    return (s - mu) / sig.replace(0.0, np.nan)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt_pct(x: float | None) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    return f"{100.0 * float(x):.2f}%"


def _fmt_num(x: float | None, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    return f"{float(x):.{digits}f}"


def build_report_md(
    *,
    ts: str,
    split_info: dict[str, Any],
    clusters: list[dict[str, Any]],
    reps: list[dict[str, Any]],
    weights: dict[str, float],
    metrics_table: list[dict[str, Any]],
    params: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append(f"# BTC Factor Strategy Backtest Report ({ts})")
    lines.append("")
    lines.append("## 防泄漏声明 (No Information Leakage)")
    lines.append("")
    lines.append(
        "- **去重聚类、代表因子选择、符号方向、权重**：仅使用 **train** 段因子值与 **train_ic**。"
    )
    lines.append(
        "- **策略参数**（entry_z / exit_z）：使用预设默认值，**未在 test 段调参**；"
        "若未来调优，只能在 train 段进行，test 一次性评估后不得回改。"
    )
    lines.append(
        "- **test_ic / test_ir**：仅出现在最终评估表中作对照，**不参与任何决策**。"
    )
    lines.append(
        "- **dynamic_combine**：滚动 IC 权重因果（t 仅用到 t-1），可在 train+test 全程计算后切片 test 交易。"
    )
    lines.append(
        "- 数据划分与挖掘脚本一致：最后 `test_bars` 为 OOS，其前 `train_bars` 为 train。"
    )
    lines.append("")
    lines.append("## Split")
    lines.append("")
    lines.append(f"- train: `{split_info['train_start']}` → `{split_info['train_end']}` ({split_info['n_train']} bars)")
    lines.append(f"- test: `{split_info['test_start']}` → `{split_info['test_end']}` ({split_info['n_test']} bars)")
    lines.append(f"- dedup_eval_bars: `{params['dedup_eval_bars']}`")
    lines.append(f"- corr_threshold: `{params['corr_threshold']}`")
    lines.append("")
    lines.append("## Dedup Clusters (train-only)")
    lines.append("")
    lines.append("| cluster | size | representative expr | train_ic | test_ic (ref only) |")
    lines.append("|---:|---:|---|---:|---:|")
    for i, c in enumerate(clusters):
        r = c["representative"]
        lines.append(
            f"| {i} | {c['size']} | `{r['expr']}` | {_fmt_num(r.get('train_ic'), 4)} | "
            f"{_fmt_num(r.get('test_ic'), 4)} |"
        )
    lines.append("")
    lines.append("### Representatives & signed weights (train IC)")
    lines.append("")
    lines.append("| name | expr | train_ic | signed_weight |")
    lines.append("|---|---|---:|---:|")
    for i, r in enumerate(reps):
        name = f"mined_btc_{i}"
        lines.append(
            f"| `{name}` | `{r['expr']}` | {_fmt_num(r.get('train_ic'), 4)} | "
            f"{_fmt_num(weights.get(name), 4)} |"
        )
    lines.append("")
    lines.append("## Cost model")
    lines.append("")
    lines.append(f"- commission_rate: `{params['commission_rate']}`")
    lines.append(f"- slippage: `{params['slippage_bps']}` bp (tick_size ≈ median_price × bps/10000, slippage_ticks=1)")
    lines.append(f"- contract_multiplier: `1`")
    lines.append(f"- initial_capital: `{params['initial_capital']}`")
    lines.append(f"- FactorStrategy entry_z/exit_z: `{params['entry_z']}` / `{params['exit_z']}` (defaults, not test-tuned)")
    lines.append(
        f"- CompositeZ entry/exit: `z>{params['composite_entry_z']}` long / "
        f"`z<-{params['composite_entry_z']}` short / `|z|<{params['composite_exit_z']}` flat"
    )
    lines.append("")
    lines.append("## Test-segment metrics")
    lines.append("")
    lines.append(
        "| name | total_return | annual_return | Sharpe | maxDD | win_rate | "
        "profit_factor | trades | turnover | cost_ratio |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m in metrics_table:
        lines.append(
            f"| {m['name']} | {_fmt_pct(m.get('total_return'))} | {_fmt_pct(m.get('annual_return'))} | "
            f"{_fmt_num(m.get('sharpe'), 3)} | {_fmt_pct(m.get('max_drawdown_pct'))} | "
            f"{_fmt_pct(m.get('win_rate'))} | {_fmt_num(m.get('profit_factor'), 3)} | "
            f"{m.get('total_trades', 0)} | {_fmt_num(m.get('turnover'), 3)} | "
            f"{_fmt_pct(m.get('cost_ratio'))} |"
        )
    lines.append("")
    lines.append("## Conclusions")
    lines.append("")

    by_name = {m["name"]: m for m in metrics_table}
    fs = by_name.get("factor_strategy")
    cz = by_name.get("composite_z")
    bh = by_name.get("buy_and_hold")
    st = by_name.get("supertrend")

    def _ret(m: dict[str, Any] | None) -> float:
        return float(m.get("total_return") or 0.0) if m else 0.0

    def _cost(m: dict[str, Any] | None) -> float:
        return float(m.get("cost_ratio") or 0.0) if m else 0.0

    def _turn(m: dict[str, Any] | None) -> float:
        return float(m.get("turnover") or 0.0) if m else 0.0

    best_factor = max(
        [m for m in (fs, cz) if m],
        key=lambda m: _ret(m),
        default=None,
    )
    bh_ret = _ret(bh)
    best_ret = _ret(best_factor)

    if best_factor and best_ret > bh_ret and best_ret > 0:
        lines.append(
            f"- 扣成本后因子路径 **{best_factor['name']}** 总收益 `{_fmt_pct(best_ret)}` "
            f"高于买入持有 `{_fmt_pct(bh_ret)}`，IC→PnL 在本 OOS 窗口有正向证据。"
        )
    elif best_factor and best_ret > bh_ret:
        lines.append(
            f"- 本 OOS 为下跌市：因子路径 **{best_factor['name']}** 收益 `{_fmt_pct(best_ret)}` "
            f"仍优于买入持有 `{_fmt_pct(bh_ret)}`（空头/均值回归缓冲），"
            "但绝对收益为负，不能单独证明可交易 alpha。"
        )
    elif best_factor and best_ret > 0:
        lines.append(
            f"- 因子路径有正收益 `{_fmt_pct(best_ret)}`，但未跑赢买入持有 `{_fmt_pct(bh_ret)}`；"
            "alpha 可能被趋势 beta / 成本侵蚀。"
        )
    else:
        lines.append(
            f"- 扣成本后因子路径收益偏弱（best `{_fmt_pct(best_ret)}` vs buy&hold `{_fmt_pct(bh_ret)}`）；"
            "OOS IC 未必转化为可交易 PnL。"
        )

    st_ret = _ret(st)
    if best_factor and abs(best_ret - st_ret) < 0.01:
        lines.append(
            f"- 与 supertrend（`{_fmt_pct(st_ret)}`）接近，因子组合未展现稳定超额。"
        )

    if fs:
        lines.append(
            f"- FactorStrategy 换手 `{_fmt_num(_turn(fs), 2)}`，成本占比 `{_fmt_pct(_cost(fs))}`。"
        )
    if cz:
        lines.append(
            f"- CompositeZ 换手 `{_fmt_num(_turn(cz), 2)}`，成本占比 `{_fmt_pct(_cost(cz))}`。"
        )
    if st:
        lines.append(
            f"- Supertrend 参照：收益 `{_fmt_pct(_ret(st))}`，Sharpe `{_fmt_num(st.get('sharpe'), 3)}`。"
        )

    high_turn = max(_turn(fs), _turn(cz))
    if high_turn > 20:
        lines.append("- **换手偏高**：1h 频段 + z 阈值进出可能放大成本；建议加持仓冷却或降低交易频率。")
    else:
        lines.append("- 换手处于可接受区间（相对 1h 回测）；仍需结合实盘费率复核。")

    lines.append("")
    lines.append("### Overfitting / live readiness")
    lines.append("")
    lines.append(
        "- 本实验已隔离 test 决策泄漏，但仍存在 **多重检验 / 挖掘阶段选择偏差**："
        "存活因子来自同一挖掘流程，簇内高度共线（动量/均值偏离家族）。"
    )
    lines.append(
        "- 若因子路径未稳定优于 buy&hold + supertrend，**不宜直接实盘**；"
        "下一步：更长 OOS、多币种 CS 验证、降低换手、walk-forward 参数冻结。"
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    df: pd.DataFrame,
    factors: list[dict[str, Any]],
    *,
    train_bars: int = 30000,
    test_bars: int = 8000,
    dedup_eval_bars: int = 20000,
    corr_threshold: float = 0.7,
    entry_z: float = 1.0,
    exit_z: float = 0.3,
    composite_entry_z: float = 1.0,
    composite_exit_z: float = 0.5,
    zscore_window: int = 60,
    commission_rate: float = 0.0005,
    slippage_bps: float = 1.0,
    initial_capital: float = 1_000_000.0,
    default_volume: int = 1,
    symbol: str = "BTCUSDT",
    out_dir: Path | None = None,
    ts: str | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir) if out_dir else ROOT / "output" / "research"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = ts or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")

    train, test = split_train_test(df, train_bars, test_bars)
    split_info = {
        "n_full": len(df),
        "n_train": len(train),
        "n_test": len(test),
        "train_start": str(train["datetime"].iloc[0]),
        "train_end": str(train["datetime"].iloc[-1]),
        "test_start": str(test["datetime"].iloc[0]),
        "test_end": str(test["datetime"].iloc[-1]),
    }
    logger.info(
        "Split train %s→%s (%d) | test %s→%s (%d)",
        split_info["train_start"],
        split_info["train_end"],
        split_info["n_train"],
        split_info["test_start"],
        split_info["test_end"],
        split_info["n_test"],
    )

    # --- 1) Dedup on TRAIN only ---
    if dedup_eval_bars and dedup_eval_bars > 0:
        dedup_df = train.iloc[-min(dedup_eval_bars, len(train)) :].reset_index(drop=True)
    else:
        dedup_df = train
    logger.info("Dedup eval on %d train bars, %d factors", len(dedup_df), len(factors))
    t0 = time.monotonic()
    train_frame = evaluate_factors_on_df(dedup_df, [f["expr"] for f in factors])
    logger.info("Factor eval done in %.1fs", time.monotonic() - t0)

    clusters = greedy_cluster_by_corr(factors, train_frame, threshold=corr_threshold)
    for i, c in enumerate(clusters):
        r = c["representative"]
        logger.info(
            "Cluster %d size=%d rep=%s train_ic=%.4f",
            i,
            c["size"],
            r["expr"][:70],
            float(r["train_ic"]),
        )
    reps = [c["representative"] for c in clusters]
    if not reps:
        raise RuntimeError("No representative factors after dedup")

    # --- 2) Register ---
    factor_names = register_mined_factors(reps)
    weights = signed_weights_from_train_ic(reps)

    # --- 3) Precompute factor values on train+test (for signals); weights from train only ---
    full = pd.concat([train, test], ignore_index=True)
    full_frame = evaluate_factors_on_df(full, [r["expr"] for r in reps])
    full_frame.columns = factor_names

    test_bars_df = test.copy()
    for name in factor_names:
        test_bars_df[name] = full_frame[name].iloc[-len(test) :].to_numpy()

    # Composite via dynamic_combine on full (causal), then z-score; trade on test
    fwd = full["close"].shift(-1) / full["close"] - 1.0
    # Orient factors by train IC sign before combine
    oriented = full_frame.copy()
    for i, name in enumerate(factor_names):
        if float(reps[i]["train_ic"]) < 0:
            oriented[name] = -oriented[name]
    combined = dynamic_combine(oriented, fwd, window=120, min_periods=60, smoothing_halflife=20)
    signal_z = _causal_zscore_series(combined, zscore_window)
    test_bars_df["signal_z"] = signal_z.iloc[-len(test) :].to_numpy()

    # Slippage: 1bp of median test price
    median_px = float(test["close"].median())
    tick_size = Decimal(str(round(median_px * (slippage_bps / 10_000.0), 4)))
    if tick_size <= 0:
        tick_size = Decimal("1")
    commission = Decimal(str(commission_rate))
    capital = Decimal(str(initial_capital))

    metrics_table: list[dict[str, Any]] = []

    # --- 4a) FactorStrategy ---
    fs_cfg = StrategyConfig(
        name="factor_strategy",
        strategy_id="factor_strategy",
        symbols=[symbol],
        features=factor_names,
        params={
            "factors": weights,
            "entry_z": entry_z,
            "exit_z": exit_z,
            "zscore_window": zscore_window,
            "allow_short": True,
            "feature_window": max(200, zscore_window + 50),
        },
    )
    fs = BarExtraFactorStrategy(fs_cfg)
    m_fs = run_strategy_backtest(
        fs,
        test_bars_df,
        symbol=symbol,
        initial_capital=capital,
        commission_rate=commission,
        slippage_ticks=1,
        tick_size=tick_size,
        default_volume=default_volume,
        extra_cols=factor_names,
    )
    m_fs["name"] = "factor_strategy"
    metrics_table.append(m_fs)
    logger.info(
        "factor_strategy return=%.4f sharpe=%.3f trades=%s cost_ratio=%.4f",
        m_fs["total_return"],
        m_fs["sharpe"],
        m_fs["total_trades"],
        m_fs["cost_ratio"],
    )

    # --- 4b) Composite Z ---
    cz_cfg = StrategyConfig(
        name="composite_z",
        strategy_id="composite_z",
        symbols=[symbol],
        params={
            "entry_z": composite_entry_z,
            "exit_z": composite_exit_z,
            "allow_short": True,
        },
    )
    cz = CompositeZStrategy(cz_cfg)
    m_cz = run_strategy_backtest(
        cz,
        test_bars_df,
        symbol=symbol,
        initial_capital=capital,
        commission_rate=commission,
        slippage_ticks=1,
        tick_size=tick_size,
        default_volume=default_volume,
        extra_cols=["signal_z"],
    )
    m_cz["name"] = "composite_z"
    metrics_table.append(m_cz)
    logger.info(
        "composite_z return=%.4f sharpe=%.3f trades=%s cost_ratio=%.4f",
        m_cz["total_return"],
        m_cz["sharpe"],
        m_cz["total_trades"],
        m_cz["cost_ratio"],
    )

    # --- 5) Benchmarks ---
    bh = BuyAndHoldStrategy(
        StrategyConfig(name="buy_and_hold", strategy_id="buy_and_hold", symbols=[symbol])
    )
    m_bh = run_strategy_backtest(
        bh,
        test_bars_df,
        symbol=symbol,
        initial_capital=capital,
        commission_rate=commission,
        slippage_ticks=1,
        tick_size=tick_size,
        default_volume=default_volume,
    )
    m_bh["name"] = "buy_and_hold"
    metrics_table.append(m_bh)

    st = SupertrendStrategy(
        StrategyConfig(name="supertrend", strategy_id="supertrend", symbols=[symbol])
    )
    m_st = run_strategy_backtest(
        st,
        test_bars_df,
        symbol=symbol,
        initial_capital=capital,
        commission_rate=commission,
        slippage_ticks=1,
        tick_size=tick_size,
        default_volume=default_volume,
    )
    m_st["name"] = "supertrend"
    metrics_table.append(m_st)

    params = {
        "commission_rate": commission_rate,
        "slippage_bps": slippage_bps,
        "initial_capital": initial_capital,
        "entry_z": entry_z,
        "exit_z": exit_z,
        "composite_entry_z": composite_entry_z,
        "composite_exit_z": composite_exit_z,
        "corr_threshold": corr_threshold,
        "dedup_eval_bars": dedup_eval_bars if dedup_eval_bars else len(train),
        "tick_size": str(tick_size),
    }

    report_md = build_report_md(
        ts=ts,
        split_info=split_info,
        clusters=clusters,
        reps=reps,
        weights=weights,
        metrics_table=metrics_table,
        params=params,
    )
    report_path = out_dir / f"btc_factor_strategy_report_{ts}.md"
    report_path.write_text(report_md, encoding="utf-8")
    logger.info("Wrote report %s", report_path)

    summary_json = {
        "ts": ts,
        "split_info": split_info,
        "representatives": [
            {
                "name": f"mined_btc_{i}",
                "expr": r["expr"],
                "train_ic": r.get("train_ic"),
                "test_ic": r.get("test_ic"),
                "weight": weights.get(f"mined_btc_{i}"),
                "cluster_size": clusters[i]["size"],
            }
            for i, r in enumerate(reps)
        ],
        "metrics_table": metrics_table,
        "params": params,
        "report_path": str(report_path),
    }
    json_path = out_dir / f"btc_factor_strategy_metrics_{ts}.json"
    json_path.write_text(json.dumps(summary_json, indent=2, default=str), encoding="utf-8")

    return {
        "report_path": str(report_path),
        "json_path": str(json_path),
        "clusters": clusters,
        "representatives": reps,
        "weights": weights,
        "metrics_table": metrics_table,
        "split_info": split_info,
        "params": params,
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    factors = load_surviving_factors(args.factors_json)
    logger.info("Loaded %d surviving factors from %s", len(factors), args.factors_json)
    df = load_btc_ohlcv(args.symbol, args.timeframe, args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "output" / "research"
    return run_pipeline(
        df,
        factors,
        train_bars=args.train_bars,
        test_bars=args.test_bars,
        dedup_eval_bars=args.dedup_eval_bars,
        corr_threshold=args.corr_threshold,
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        composite_entry_z=args.composite_entry_z,
        composite_exit_z=args.composite_exit_z,
        zscore_window=args.zscore_window,
        commission_rate=args.commission_rate,
        slippage_bps=args.slippage_bps,
        initial_capital=args.initial_capital,
        default_volume=args.default_volume,
        symbol=args.symbol,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
