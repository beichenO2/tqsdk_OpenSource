"""多币 crypto 截面因子面板构建 + Alphalens 风格分析。"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from datahub.crypto_loader import CryptoDataLoader
from factor.evolution import evaluate_expression
from factor.registry import compute_factor_frame, get_factor_meta

logger = logging.getLogger(__name__)

MIN_ALIGNED_ROWS = 30
MIN_SYMBOLS = 2


def _primary_series(computed: pd.DataFrame, name: str, output_columns: list[str]) -> pd.Series:
    for col in output_columns:
        if col in computed.columns:
            return computed[col]
    candidates = [c for c in computed.columns if c == name or c.startswith(f"{name}_")]
    if candidates:
        return computed[candidates[0]]
    extras = [c for c in computed.columns if c not in ("open", "high", "low", "close", "volume")]
    if extras:
        return computed[extras[-1]]
    raise ValueError(f"No output column for factor {name}")


def _load_symbol_ohlcv(
    loader: CryptoDataLoader,
    symbol: str,
    timeframe: str,
    limit: int,
) -> pd.DataFrame:
    df = loader.load(symbol, timeframe=timeframe)
    if df.empty:
        raise ValueError(f"No data for symbol {symbol} ({timeframe})")
    out = df.copy()
    out["open_time"] = pd.to_datetime(out["open_time"], utc=True)
    out = out.set_index("open_time").sort_index()
    if limit and len(out) > limit:
        out = out.iloc[-limit:]
    return out[["open", "high", "low", "close", "volume"]]


def _align_panels(
    factor_cols: dict[str, pd.Series],
    close_cols: dict[str, pd.Series],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(factor_cols) < MIN_SYMBOLS:
        raise ValueError(f"Need at least {MIN_SYMBOLS} symbols with data, got {len(factor_cols)}")

    fpanel = pd.DataFrame(factor_cols).sort_index()
    cpanel = pd.DataFrame(close_cols).sort_index()
    common_idx = fpanel.index.intersection(cpanel.index)
    fpanel = fpanel.loc[common_idx]
    cpanel = cpanel.loc[common_idx]
    mask = fpanel.notna().all(axis=1) & cpanel.notna().all(axis=1)
    fpanel = fpanel.loc[mask]
    cpanel = cpanel.loc[mask]

    if len(fpanel) < MIN_ALIGNED_ROWS:
        raise ValueError(
            f"Insufficient aligned rows after dropna: {len(fpanel)} < {MIN_ALIGNED_ROWS}"
        )
    return fpanel, cpanel


def build_cs_panels(
    factor_name: str,
    symbols: list[str],
    timeframe: str = "1h",
    limit: int = 5000,
    data_dir: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """逐币计算注册因子，按 open_time 内连接对齐为截面 panel。"""
    if len(symbols) < MIN_SYMBOLS:
        raise ValueError(f"Need at least {MIN_SYMBOLS} symbols, got {len(symbols)}")

    loader = CryptoDataLoader(data_dir)
    meta = get_factor_meta(factor_name)
    factor_cols: dict[str, pd.Series] = {}
    close_cols: dict[str, pd.Series] = {}

    for sym in symbols:
        try:
            ohlcv = _load_symbol_ohlcv(loader, sym, timeframe, limit)
            computed = compute_factor_frame(ohlcv, [factor_name])
            factor_cols[sym] = _primary_series(computed, factor_name, meta["output_columns"])
            close_cols[sym] = computed["close"]
        except Exception as e:
            logger.debug("cs panel skip %s: %s", sym, e)
            continue

    return _align_panels(factor_cols, close_cols)


def build_cs_panels_from_expr(
    expr: str,
    symbols: list[str],
    timeframe: str = "1h",
    limit: int = 5000,
    data_dir: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """逐币用表达式沙箱计算因子值，对齐为截面 panel。"""
    if len(symbols) < MIN_SYMBOLS:
        raise ValueError(f"Need at least {MIN_SYMBOLS} symbols, got {len(symbols)}")

    loader = CryptoDataLoader(data_dir)
    factor_cols: dict[str, pd.Series] = {}
    close_cols: dict[str, pd.Series] = {}

    for sym in symbols:
        try:
            ohlcv = _load_symbol_ohlcv(loader, sym, timeframe, limit)
            factor_cols[sym] = evaluate_expression(expr, ohlcv)
            close_cols[sym] = ohlcv["close"]
        except Exception as e:
            logger.debug("cs expr panel skip %s: %s", sym, e)
            continue

    return _align_panels(factor_cols, close_cols)


def run_cs_analysis(
    factor_name: str,
    symbols: list[str],
    timeframe: str = "1h",
    limit: int = 5000,
    quantiles: int = 5,
    data_dir: str | None = None,
    *,
    horizon: int = 1,
) -> dict[str, Any]:
    """注册因子截面 IC + 分位收益一站式。"""
    from factor.alphalens_cs import cross_sectional_ic, quantile_returns
    from factor.analysis import summarize_ic

    fpanel, cpanel = build_cs_panels(
        factor_name, symbols, timeframe=timeframe, limit=limit, data_dir=data_dir
    )
    n_assets = len(fpanel.columns)
    ic = cross_sectional_ic(fpanel, cpanel, horizon=horizon, min_assets=min(3, n_assets))
    summary = summarize_ic(ic)
    qret = quantile_returns(
        fpanel, cpanel, horizon=horizon, quantiles=quantiles, min_assets=n_assets
    )
    ic_tail = ic.dropna().tail(80)
    result = {
        "mode": "cross_sectional",
        "horizon": horizon,
        "summary": summary,
        "quantile_returns": qret,
        "ic_series": [
            {
                "t": (idx.isoformat() if hasattr(idx, "isoformat") else str(idx)),
                "v": float(v),
            }
            for idx, v in ic_tail.items()
        ],
        "n_assets": n_assets,
        "factor_name": factor_name,
        "symbols_used": list(fpanel.columns),
    }
    return result


def run_cs_analysis_from_expr(
    expr: str,
    symbols: list[str],
    timeframe: str = "1h",
    limit: int = 5000,
    quantiles: int = 5,
    data_dir: str | None = None,
    *,
    horizon: int = 1,
) -> dict[str, Any]:
    """表达式因子截面 IC + 分位收益。"""
    from factor.alphalens_cs import cross_sectional_ic, quantile_returns
    from factor.analysis import summarize_ic

    fpanel, cpanel = build_cs_panels_from_expr(
        expr, symbols, timeframe=timeframe, limit=limit, data_dir=data_dir
    )
    n_assets = len(fpanel.columns)
    ic = cross_sectional_ic(fpanel, cpanel, horizon=horizon, min_assets=min(3, n_assets))
    summary = summarize_ic(ic)
    qret = quantile_returns(
        fpanel, cpanel, horizon=horizon, quantiles=quantiles, min_assets=n_assets
    )
    ic_tail = ic.dropna().tail(80)
    return {
        "mode": "cross_sectional",
        "horizon": horizon,
        "summary": summary,
        "quantile_returns": qret,
        "ic_series": [
            {
                "t": (idx.isoformat() if hasattr(idx, "isoformat") else str(idx)),
                "v": float(v),
            }
            for idx, v in ic_tail.items()
        ],
        "n_assets": n_assets,
        "expr": expr,
        "symbols_used": list(fpanel.columns),
    }
