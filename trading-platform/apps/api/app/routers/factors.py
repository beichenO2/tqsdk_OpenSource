"""因子 API — 列表 / 详情 / 计算 / 分析 / 合成。

端点:
- GET  /factors
- GET  /factors/{name}
- POST /factors/compute
- POST /factors/analyze
- POST /factors/combine
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/factors", tags=["factors"])

REPO = Path(__file__).resolve().parents[4]
PACKAGES = REPO / "packages"
for p in (PACKAGES, PACKAGES / "features", REPO):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


def _futures_files(base: str) -> list[Path]:
    """精确匹配期货缓存文件，避免单字符品种（i/m/j…）通配污染。

    优先级：KQ 主连 5m > KQ 主连 daily > 旧命名 {sym}_5m_* > {sym}_*。
    同一次只返回同一时间粒度的文件。
    """
    d = REPO / "data" / "futures_cache"
    if not d.exists():
        return []
    variants = [base]
    for v in (base.lower(), base.upper()):
        if v not in variants:
            variants.append(v)
    for b in variants:
        for tf in ("5m", "daily"):
            files = sorted(d.glob(f"KQ_m_*_{b}_{tf}.parquet"))
            if files:
                return files
    for b in variants:
        files = sorted(d.glob(f"{b}_5m_*.parquet"))
        if files:
            return files
        files = sorted(d.glob(f"{b}_*.parquet"))
        if files:
            return files
    return []


def _crypto_files(sym_lower: str) -> list[Path]:
    """crypto 目录取单一时间粒度，避免 1h/4h/1d 混载。"""
    d = REPO / "data" / "crypto_cache" / sym_lower
    if not d.is_dir():
        return []
    for tf in ("1h", "4h", "1d", "30m", "15m", "5m"):
        f = d / f"{tf}.parquet"
        if f.exists():
            return [f]
    rest = sorted(d.glob("*.parquet"))
    return rest[:1]


def _resolve_symbol_files(symbol: str) -> list[Path]:
    raw_sym = symbol.split(".")[-1] if "." in symbol else symbol
    base_sym = "".join(c for c in raw_sym if not c.isdigit()) or raw_sym
    return _futures_files(base_sym) or _crypto_files(raw_sym.lower())


def _load_ohlcv(symbol: str, limit: int = 500) -> "Any":
    """从 futures/crypto parquet 缓存加载 OHLCV DataFrame（单品种、单粒度）。"""
    import pandas as pd

    hits = _resolve_symbol_files(symbol)
    if not hits:
        raise HTTPException(status_code=404, detail=f"No parquet data for symbol: {symbol}")

    frames: list[pd.DataFrame] = []
    for fp in hits[:8]:
        try:
            df = pd.read_parquet(fp)
        except Exception as e:
            logger.warning("skip %s: %s", fp, e)
            continue
        # normalize columns
        colmap = {c.lower(): c for c in df.columns}
        rename = {}
        for want in ("open", "high", "low", "close", "volume"):
            if want in colmap:
                rename[colmap[want]] = want
            elif want.capitalize() in df.columns:
                rename[want.capitalize()] = want
        df = df.rename(columns=rename)
        ts_col = next(
            (c for c in ("datetime", "timestamp", "open_time", "dt", "date") if c in df.columns),
            None,
        )
        if ts_col:
            df[ts_col] = pd.to_datetime(df[ts_col])
            df = df.set_index(ts_col).sort_index()
        if not all(c in df.columns for c in ("open", "high", "low", "close")):
            continue
        if "volume" not in df.columns:
            df["volume"] = 0.0
        frames.append(df[["open", "high", "low", "close", "volume"]])

    if not frames:
        raise HTTPException(status_code=404, detail=f"Unreadable parquet for: {symbol}")

    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    if limit and len(out) > limit:
        out = out.iloc[-limit:]
    return out


def _default_panel_symbols() -> list[str]:
    """从 futures_cache 推断可用品种代码。"""
    cache = REPO / "data" / "futures_cache"
    if not cache.exists():
        return ["rb", "au", "cu", "ag", "IF", "i", "m"]
    syms: set[str] = set()
    for p in cache.glob("KQ_m_*_*_5m.parquet"):
        # KQ_m_SHFE_rb_5m.parquet
        parts = p.stem.split("_")
        if len(parts) >= 4:
            syms.add(parts[3])
    for p in cache.glob("*_5m_*.parquet"):
        if p.name.startswith("KQ_"):
            continue
        syms.add(p.stem.split("_")[0])
    return sorted(syms)[:24] if syms else ["rb", "au", "cu"]


def _tf_token(path: Path) -> str:
    """从文件名提取时间粒度 token（5m/daily/1h/…）。"""
    stem = path.stem
    for tf in ("5m", "daily", "1h", "4h", "1d", "30m", "15m"):
        if stem.endswith(f"_{tf}") or stem == tf or f"_{tf}_" in stem:
            return tf
    return "unknown"


def _build_factor_close_panels(
    symbols: list[str],
    factor_name: str,
    limit: int = 400,
) -> tuple[Any, Any]:
    """构建 (factor_panel, close_panel)，列=品种；强制同一时间粒度。"""
    import pandas as pd
    from factor.registry import compute_factor_frame, get_factor_meta

    meta = get_factor_meta(factor_name)
    factor_cols: dict[str, Any] = {}
    close_cols: dict[str, Any] = {}
    panel_tf: str | None = None
    for sym in symbols:
        try:
            files = _resolve_symbol_files(sym)
            if not files:
                continue
            tf = _tf_token(files[0])
            if panel_tf is None:
                panel_tf = tf
            elif tf != panel_tf:
                logger.debug("panel skip %s: tf %s != %s", sym, tf, panel_tf)
                continue
            df = _load_ohlcv(sym, limit=limit)
            computed = compute_factor_frame(df, [factor_name])
            s = _primary_series(computed, factor_name, meta["output_columns"])
            factor_cols[sym] = s
            close_cols[sym] = computed["close"]
        except Exception as e:
            logger.debug("panel skip %s: %s", sym, e)
            continue
    if len(factor_cols) < 3:
        raise HTTPException(
            status_code=400,
            detail=f"Need ≥3 symbols with data for cross-section, got {len(factor_cols)}",
        )
    fpanel = pd.DataFrame(factor_cols).sort_index()
    cpanel = pd.DataFrame(close_cols).sort_index()
    return fpanel, cpanel


def _primary_series(computed, name: str, output_columns: list[str]):
    """取因子主输出列。"""
    for col in output_columns:
        if col in computed.columns:
            return computed[col]
    # technical ma writes ma_{period}
    candidates = [c for c in computed.columns if c == name or c.startswith(f"{name}_")]
    if candidates:
        return computed[candidates[0]]
    # last added non-ohlcv
    extras = [c for c in computed.columns if c not in ("open", "high", "low", "close", "volume")]
    if extras:
        return computed[extras[-1]]
    raise HTTPException(status_code=500, detail=f"No output column for factor {name}")


@router.get("")
async def list_factors(
    category: str | None = Query(None),
) -> dict[str, Any]:
    from factor.registry import list_factor_metas, get_registry

    metas = list_factor_metas(category=category)
    reg = get_registry()
    return {
        "factors": metas,
        "count": len(metas),
        "categories": reg.categories,
    }


@router.get("/{name}")
async def get_factor(name: str) -> dict[str, Any]:
    from factor.registry import get_factor_meta

    try:
        return get_factor_meta(name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


class ComputeRequest(BaseModel):
    symbol: str = Field(..., examples=["rb", "btcusdt"])
    factor_names: list[str] = Field(..., min_length=1)
    limit: int = Field(400, ge=50, le=5000)
    params: dict[str, dict[str, Any]] | None = None


def _compute_factors_sync(req: ComputeRequest) -> dict[str, Any]:
    from factor.registry import compute_factor_frame, get_factor_meta

    df = _load_ohlcv(req.symbol, limit=req.limit)
    try:
        computed = compute_factor_frame(df, req.factor_names, params=req.params)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"compute failed: {e}") from e

    series_out: dict[str, Any] = {}
    for name in req.factor_names:
        try:
            meta = get_factor_meta(name)
            s = _primary_series(computed, name, meta["output_columns"])
        except Exception:
            continue
        tail = s.dropna().tail(80)
        series_out[name] = {
            "points": [
                {"t": (idx.isoformat() if hasattr(idx, "isoformat") else str(idx)), "v": float(v)}
                for idx, v in tail.items()
            ],
            "last": float(tail.iloc[-1]) if len(tail) else None,
            "n": int(s.notna().sum()),
        }

    return {
        "symbol": req.symbol,
        "bars": len(df),
        "factors": series_out,
    }


@router.post("/compute")
async def compute_factors(req: ComputeRequest) -> dict[str, Any]:
    return await asyncio.to_thread(_compute_factors_sync, req)


class AnalyzeRequest(BaseModel):
    symbol: str = "rb"
    factor_names: list[str] = Field(..., min_length=1)
    limit: int = Field(500, ge=80, le=5000)
    horizon: int = Field(1, ge=1, le=20)
    dedupe_threshold: float = Field(0.99, ge=0.5, le=1.0)


def _analyze_factors_sync(req: AnalyzeRequest) -> dict[str, Any]:
    from factor.analysis import (
        correlation_matrix,
        deduplicate_factors,
        factor_ic,
        ic_decay,
        summarize_ic,
    )
    from factor.registry import compute_factor_frame, get_factor_meta

    df = _load_ohlcv(req.symbol, limit=req.limit)
    try:
        computed = compute_factor_frame(df, req.factor_names)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"compute failed: {e}") from e

    close = computed["close"]
    factor_cols: dict[str, Any] = {}
    reports: list[dict[str, Any]] = []

    for name in req.factor_names:
        try:
            meta = get_factor_meta(name)
            s = _primary_series(computed, name, meta["output_columns"])
        except Exception as e:
            reports.append({"name": name, "error": str(e)})
            continue
        factor_cols[name] = s
        ic = factor_ic(s, close, horizon=req.horizon)
        summary = summarize_ic(ic)
        decay = ic_decay(s, close)
        ic_tail = ic.dropna().tail(60)
        reports.append({
            "name": name,
            "category": meta["category"],
            "summary": summary,
            "decay": decay,
            "ic_series": [
                {"t": (idx.isoformat() if hasattr(idx, "isoformat") else str(idx)), "v": float(v)}
                for idx, v in ic_tail.items()
            ],
        })

    import pandas as pd
    fdf = pd.DataFrame(factor_cols)
    corr = correlation_matrix(fdf)
    dedupe = deduplicate_factors(fdf, threshold=req.dedupe_threshold)

    return {
        "symbol": req.symbol,
        "horizon": req.horizon,
        "bars": len(df),
        "reports": reports,
        "correlation": corr,
        "dedupe": dedupe,
    }


@router.post("/analyze")
async def analyze_factors(req: AnalyzeRequest) -> dict[str, Any]:
    return await asyncio.to_thread(_analyze_factors_sync, req)


class CrossSectionRequest(BaseModel):
    factor_name: str = Field(..., examples=["a158_roc_20", "rsi", "wq101"])
    symbols: list[str] | None = None
    limit: int = Field(300, ge=80, le=2000)
    horizon: int = Field(1, ge=1, le=20)
    quantiles: int = Field(5, ge=2, le=10)


def _analyze_cs_sync(req: CrossSectionRequest) -> dict[str, Any]:
    from factor.alphalens_cs import analyze_cross_section
    from factor.registry import get_factor_meta

    try:
        meta = get_factor_meta(req.factor_name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    symbols = req.symbols or _default_panel_symbols()
    fpanel, cpanel = _build_factor_close_panels(symbols, req.factor_name, limit=req.limit)
    result = analyze_cross_section(
        fpanel,
        cpanel,
        horizon=req.horizon,
        quantiles=req.quantiles,
    )
    result["factor"] = meta
    result["symbols_used"] = list(fpanel.columns)
    return result


@router.post("/analyze-cs")
async def analyze_cross_section_api(req: CrossSectionRequest) -> dict[str, Any]:
    """Alphalens 风格截面 IC + 分位收益。"""
    return await asyncio.to_thread(_analyze_cs_sync, req)


class CombineRequest(BaseModel):
    symbol: str = "rb"
    factor_names: list[str] = Field(..., min_length=2)
    method: str = Field("equal", pattern="^(equal|ic|orth)$")
    limit: int = Field(400, ge=80, le=5000)


def _combine_factors_sync(req: CombineRequest) -> dict[str, Any]:
    from factor.analysis import factor_ic, summarize_ic
    from factor.combine import combine_equal_weight, combine_ic_weight, orthogonalize
    from factor.registry import compute_factor_frame, get_factor_meta
    import pandas as pd

    df = _load_ohlcv(req.symbol, limit=req.limit)
    computed = compute_factor_frame(df, req.factor_names)
    close = computed["close"]
    cols: dict[str, Any] = {}
    ic_means: dict[str, float] = {}
    for name in req.factor_names:
        meta = get_factor_meta(name)
        s = _primary_series(computed, name, meta["output_columns"])
        cols[name] = s
        summary = summarize_ic(factor_ic(s, close))
        if summary["ic_mean"] is not None:
            ic_means[name] = float(summary["ic_mean"])

    fdf = pd.DataFrame(cols).dropna()
    if req.method == "equal":
        combined = combine_equal_weight(fdf)
    elif req.method == "ic":
        combined = combine_ic_weight(fdf, ic_means)
    else:
        orth = orthogonalize(fdf)
        combined = orth.mean(axis=1).rename("combined_orth")

    tail = combined.dropna().tail(80)
    return {
        "symbol": req.symbol,
        "method": req.method,
        "ic_means": ic_means,
        "combined": {
            "points": [
                {"t": (idx.isoformat() if hasattr(idx, "isoformat") else str(idx)), "v": float(v)}
                for idx, v in tail.items()
            ],
            "last": float(tail.iloc[-1]) if len(tail) else None,
        },
    }


@router.post("/combine")
async def combine_factors(req: CombineRequest) -> dict[str, Any]:
    return await asyncio.to_thread(_combine_factors_sync, req)


class EvolveRequest(BaseModel):
    symbol: str = "rb"
    n_proposals: int = Field(5, ge=1, le=20)
    limit: int = Field(400, ge=100, le=3000)
    use_llm: bool = True
    existing_exprs: list[str] | None = None


def _evolve_factors_sync(req: EvolveRequest) -> dict[str, Any]:
    from factor.evolution import run_evolution_round

    df = _load_ohlcv(req.symbol, limit=req.limit)
    result = run_evolution_round(
        df,
        n_proposals=req.n_proposals,
        existing_exprs=req.existing_exprs,
        use_llm=req.use_llm,
    )
    result["symbol"] = req.symbol
    result["bars"] = len(df)
    return result


@router.post("/evolve")
async def evolve_factors(req: EvolveRequest) -> dict[str, Any]:
    """一轮因子表达式进化（bandit + LLM/模板变异 + IC/去重门控）。"""
    return await asyncio.to_thread(_evolve_factors_sync, req)


@router.get("/evolve/latest")
async def evolve_latest() -> dict[str, Any]:
    """读取最近一次进化产物。"""
    path = REPO / "output" / "factor_evolution" / "latest.json"
    if not path.exists():
        return {"exists": False, "latest": None}
    try:
        return {"exists": True, "latest": json.loads(path.read_text())}
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
