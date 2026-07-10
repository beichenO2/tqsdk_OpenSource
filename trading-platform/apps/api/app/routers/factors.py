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
from pydantic import BaseModel, Field, model_validator

from factor.evolution import run_evolution_round
from factor.evolution_registry import classify_candidates, register_elite_from_payload
from factor.mcts_search import run_mcts_search

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/factors", tags=["factors"])

DEFAULT_CRYPTO_DATA_DIR = None  # CryptoDataLoader 默认 ~/Downloads/crypto_data

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


def _is_crypto_usdt(symbol: str) -> bool:
    """全大写且 USDT 结尾视为 crypto 永续（如 BTCUSDT）。"""
    return symbol.isupper() and symbol.endswith("USDT")


def _load_crypto_ohlcv(
    symbol: str,
    limit: int = 500,
    timeframe: str = "1h",
    data_dir: str | None = None,
) -> "Any":
    """CryptoDataLoader 加载 OHLCV，index=open_time。"""
    import pandas as pd
    from datahub.crypto_loader import CryptoDataLoader

    loader = CryptoDataLoader(data_dir or DEFAULT_CRYPTO_DATA_DIR)
    df = loader.load(symbol, timeframe=timeframe)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No crypto data for symbol: {symbol}")
    out = df.copy()
    out["open_time"] = pd.to_datetime(out["open_time"], utc=True)
    out = out.set_index("open_time").sort_index()
    cols = ["open", "high", "low", "close", "volume"]
    if not all(c in out.columns for c in cols):
        raise HTTPException(status_code=404, detail=f"Unreadable crypto parquet for: {symbol}")
    out = out[cols]
    if limit and len(out) > limit:
        out = out.iloc[-limit:]
    return out


def _load_evolution_ohlcv(symbol: str, limit: int = 500) -> "Any":
    """进化 round 数据源：crypto USDT 走 CryptoDataLoader，否则期货缓存。"""
    if _is_crypto_usdt(symbol):
        return _load_crypto_ohlcv(symbol, limit=limit)
    return _load_ohlcv(symbol, limit=limit)


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
    method: str = Field("equal", pattern="^(equal|ic|orth|dynamic)$")
    limit: int = Field(400, ge=80, le=5000)
    window: int = Field(120, ge=20, le=2000)
    halflife: int = Field(20, ge=1, le=500)
    horizon: int = Field(1, ge=1, le=20)


def _combine_factors_sync(req: CombineRequest) -> dict[str, Any]:
    from factor.analysis import factor_ic, summarize_ic
    from factor.combine import (
        combine_equal_weight,
        combine_ic_weight,
        compare_static_vs_dynamic,
        dynamic_combine,
        orthogonalize,
    )
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
    compare: dict[str, Any] | None = None
    if req.method == "equal":
        combined = combine_equal_weight(fdf)
    elif req.method == "ic":
        combined = combine_ic_weight(fdf, ic_means)
    elif req.method == "dynamic":
        fwd = close.reindex(fdf.index).shift(-req.horizon) / close.reindex(fdf.index) - 1.0
        min_periods = max(20, req.window // 2)
        combined = dynamic_combine(
            fdf,
            fwd,
            window=req.window,
            min_periods=min_periods,
            smoothing_halflife=req.halflife,
        )
        compare = compare_static_vs_dynamic(
            fdf,
            close.reindex(fdf.index),
            horizon=req.horizon,
            window=req.window,
            min_periods=min_periods,
            smoothing_halflife=req.halflife,
        )
    else:
        orth = orthogonalize(fdf)
        combined = orth.mean(axis=1).rename("combined_orth")

    tail = combined.dropna().tail(80)
    out: dict[str, Any] = {
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
    if compare is not None:
        out["compare"] = compare
    return out

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
    df = _load_evolution_ohlcv(req.symbol, limit=req.limit)
    result = run_evolution_round(
        df,
        n_proposals=req.n_proposals,
        existing_exprs=req.existing_exprs,
        use_llm=req.use_llm,
    )
    classified = classify_candidates(result)
    registered = register_elite_from_payload(result)
    result["symbol"] = req.symbol
    result["bars"] = len(df)
    result["elite"] = classified["elite"]
    result["qualified"] = classified["qualified"]
    result["registered"] = registered
    return result


@router.post("/evolve")
async def evolve_factors(req: EvolveRequest) -> dict[str, Any]:
    """一轮因子表达式进化（bandit + LLM/模板变异 + IC/去重门控）。"""
    return await asyncio.to_thread(_evolve_factors_sync, req)


class EvolveMCTSRequest(BaseModel):
    symbol: str = "BTCUSDT"
    n_iterations: int = Field(50, ge=1, le=200)
    use_llm: bool = False
    timeframe: str = "1h"
    limit: int = Field(5000, ge=500, le=20000)


def _evolve_mcts_sync(req: EvolveMCTSRequest) -> dict[str, Any]:
    if _is_crypto_usdt(req.symbol):
        df = _load_crypto_ohlcv(req.symbol, limit=req.limit, timeframe=req.timeframe)
    else:
        df = _load_ohlcv(req.symbol, limit=req.limit)
    result = run_mcts_search(
        df,
        n_iterations=req.n_iterations,
        use_llm=req.use_llm,
    )
    classified = classify_candidates(result)
    registered = register_elite_from_payload(result)
    result["symbol"] = req.symbol
    result["bars"] = len(df)
    result["elite"] = classified["elite"]
    result["qualified"] = classified["qualified"]
    result["registered"] = registered
    return result


@router.post("/evolve-mcts")
async def evolve_mcts_factors(req: EvolveMCTSRequest) -> dict[str, Any]:
    """MCTS 因子表达式搜索（子树规避 + 失败经验库 + IC 门控）。"""
    return await asyncio.to_thread(_evolve_mcts_sync, req)


class AnalyzeCsCryptoRequest(BaseModel):
    factor_name: str | None = None
    expr: str | None = None
    symbols: list[str] = Field(
        default=["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"],
        min_length=2,
    )
    timeframe: str = "1h"
    limit: int = Field(5000, ge=80, le=20000)
    quantiles: int = Field(5, ge=2, le=10)

    @model_validator(mode="after")
    def _factor_or_expr(self) -> "AnalyzeCsCryptoRequest":
        if bool(self.factor_name) == bool(self.expr):
            raise ValueError("Provide exactly one of factor_name or expr")
        return self


def _analyze_cs_crypto_sync(req: AnalyzeCsCryptoRequest) -> dict[str, Any]:
    from factor.cs_pipeline import run_cs_analysis, run_cs_analysis_from_expr

    data_dir = DEFAULT_CRYPTO_DATA_DIR
    try:
        if req.expr:
            return run_cs_analysis_from_expr(
                req.expr,
                req.symbols,
                timeframe=req.timeframe,
                limit=req.limit,
                quantiles=req.quantiles,
                data_dir=data_dir,
            )
        return run_cs_analysis(
            req.factor_name or "",
            req.symbols,
            timeframe=req.timeframe,
            limit=req.limit,
            quantiles=req.quantiles,
            data_dir=data_dir,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.post("/analyze-cs-crypto")
async def analyze_cs_crypto(req: AnalyzeCsCryptoRequest) -> dict[str, Any]:
    """多币 crypto 截面 IC + 分位收益（CryptoDataLoader 数据源）。"""
    return await asyncio.to_thread(_analyze_cs_crypto_sync, req)


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
