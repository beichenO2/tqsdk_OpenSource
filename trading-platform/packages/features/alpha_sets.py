"""Alpha158 / WorldQuant 101 风格因子子集（期货 OHLCV 可算）。

参考：
- Microsoft Qlib Alpha158（动量/波动/量价滚动特征）
- WorldQuant 101 Formulaic Alphas（可复现表达式子集）
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.registry import factor


# ─── Alpha158-inspired ───


@factor("a158_roc_5", category="alpha158", output_columns=["a158_roc_5"], period=5)
def a158_roc_5(df: pd.DataFrame, period: int = 5) -> pd.DataFrame:
    """Rate of change over *period* bars (Qlib ROC-like)."""
    df["a158_roc_5"] = df["close"] / df["close"].shift(period) - 1.0
    return df


@factor("a158_roc_20", category="alpha158", output_columns=["a158_roc_20"], period=20)
def a158_roc_20(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    df["a158_roc_20"] = df["close"] / df["close"].shift(period) - 1.0
    return df


@factor("a158_ma_close_5", category="alpha158", output_columns=["a158_ma_close_5"], period=5)
def a158_ma_close_5(df: pd.DataFrame, period: int = 5) -> pd.DataFrame:
    """close / MA(close, period) - 1."""
    ma = df["close"].rolling(period).mean()
    df["a158_ma_close_5"] = df["close"] / ma.replace(0, np.nan) - 1.0
    return df


@factor("a158_ma_close_20", category="alpha158", output_columns=["a158_ma_close_20"], period=20)
def a158_ma_close_20(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    ma = df["close"].rolling(period).mean()
    df["a158_ma_close_20"] = df["close"] / ma.replace(0, np.nan) - 1.0
    return df


@factor("a158_std_20", category="alpha158", output_columns=["a158_std_20"], period=20)
def a158_std_20(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Rolling return std (volatility)."""
    df["a158_std_20"] = df["close"].pct_change().rolling(period).std()
    return df


@factor("a158_max_20", category="alpha158", output_columns=["a158_max_20"], period=20)
def a158_max_20(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """close / rolling max - 1 (drawdown from peak)."""
    mx = df["close"].rolling(period).max()
    df["a158_max_20"] = df["close"] / mx.replace(0, np.nan) - 1.0
    return df


@factor("a158_min_20", category="alpha158", output_columns=["a158_min_20"], period=20)
def a158_min_20(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    mn = df["close"].rolling(period).min()
    df["a158_min_20"] = df["close"] / mn.replace(0, np.nan) - 1.0
    return df


@factor("a158_qtlu_20", category="alpha158", output_columns=["a158_qtlu_20"], period=20)
def a158_qtlu_20(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Quantile upper: close vs rolling 80% quantile."""
    q = df["close"].rolling(period).quantile(0.8)
    df["a158_qtlu_20"] = df["close"] / q.replace(0, np.nan) - 1.0
    return df


@factor("a158_qtld_20", category="alpha158", output_columns=["a158_qtld_20"], period=20)
def a158_qtld_20(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    q = df["close"].rolling(period).quantile(0.2)
    df["a158_qtld_20"] = df["close"] / q.replace(0, np.nan) - 1.0
    return df


@factor("a158_rsv_20", category="alpha158", output_columns=["a158_rsv_20"], period=20)
def a158_rsv_20(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """RSV = (close - min) / (max - min)."""
    mn = df["low"].rolling(period).min()
    mx = df["high"].rolling(period).max()
    df["a158_rsv_20"] = (df["close"] - mn) / (mx - mn).replace(0, np.nan)
    return df


@factor("a158_corr_cv_20", category="alpha158", output_columns=["a158_corr_cv_20"], period=20)
def a158_corr_cv_20(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Rolling corr(close, volume)."""
    df["a158_corr_cv_20"] = df["close"].rolling(period).corr(df["volume"])
    return df


@factor("a158_vstd_20", category="alpha158", output_columns=["a158_vstd_20"], period=20)
def a158_vstd_20(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    df["a158_vstd_20"] = df["volume"].rolling(period).std() / df["volume"].rolling(period).mean().replace(0, np.nan)
    return df


@factor("a158_vwap_bias_20", category="alpha158", output_columns=["a158_vwap_bias_20"], period=20)
def a158_vwap_bias_20(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Approximate VWAP bias using typical price * volume."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].replace(0, np.nan)
    vwap = (tp * vol).rolling(period).sum() / vol.rolling(period).sum()
    df["a158_vwap_bias_20"] = df["close"] / vwap - 1.0
    return df


# ─── WorldQuant 101 subset ───


@factor("wq001", category="wq101", output_columns=["wq001"])
def wq001(df: pd.DataFrame) -> pd.DataFrame:
    """Alpha#1 simplified: rank(Ts_ArgMax(SignedPower(returns), 5)) style proxy.

    Uses: -1 * rank(rolling_std(returns, 20)) * sign as momentum-vol mix proxy
    when full cross-section rank is unavailable — single-asset Ts_Rank proxy.
    """
    rets = df["close"].pct_change()
    # Ts_Rank of (close - delay(close,1)) over 5 — use rolling rank percentile
    delta = df["close"].diff()
    # percentile rank within rolling window
    def _ts_rank(x: np.ndarray) -> float:
        if len(x) == 0 or np.isnan(x[-1]):
            return np.nan
        return float(pd.Series(x).rank(pct=True).iloc[-1])

    ts_rank = delta.rolling(5).apply(_ts_rank, raw=True)
    df["wq001"] = ts_rank * rets.rolling(20).std()
    return df


@factor("wq006", category="wq101", output_columns=["wq006"])
def wq006(df: pd.DataFrame) -> pd.DataFrame:
    """Alpha#6: -1 * correlation(open, volume, 10)."""
    df["wq006"] = -1.0 * df["open"].rolling(10).corr(df["volume"])
    return df


@factor("wq007", category="wq101", output_columns=["wq007"])
def wq007(df: pd.DataFrame) -> pd.DataFrame:
    """Alpha#7 proxy: (adv20 < volume) ? (-1 * ts_rank(abs(delta(close,7)), 60)) * sign(delta) : -1

    Simplified without adv20 cross-section: -ts_rank(|Δclose7|, 60) * sign(Δclose7)
    """
    delta7 = df["close"].diff(7)

    def _ts_rank(x: np.ndarray) -> float:
        if len(x) == 0 or np.isnan(x[-1]):
            return np.nan
        return float(pd.Series(x).rank(pct=True).iloc[-1])

    tr = delta7.abs().rolling(60).apply(_ts_rank, raw=True)
    df["wq007"] = -tr * np.sign(delta7)
    return df


@factor("wq012", category="wq101", output_columns=["wq012"])
def wq012(df: pd.DataFrame) -> pd.DataFrame:
    """Alpha#12: sign(delta(volume,1)) * (-1 * delta(close,1))."""
    df["wq012"] = np.sign(df["volume"].diff(1)) * (-1.0 * df["close"].diff(1))
    return df


@factor("wq026", category="wq101", output_columns=["wq026"])
def wq026(df: pd.DataFrame) -> pd.DataFrame:
    """Alpha#26: -1 * ts_max(correlation(ts_rank(volume,5), ts_rank(high,5), 5), 3) proxy."""
    def _ts_rank(s: pd.Series, w: int) -> pd.Series:
        return s.rolling(w).apply(
            lambda x: float(pd.Series(x).rank(pct=True).iloc[-1]) if len(x) else np.nan,
            raw=True,
        )

    rv = _ts_rank(df["volume"], 5)
    rh = _ts_rank(df["high"], 5)
    corr = rv.rolling(5).corr(rh)
    df["wq026"] = -1.0 * corr.rolling(3).max()
    return df


@factor("wq033", category="wq101", output_columns=["wq033"])
def wq033(df: pd.DataFrame) -> pd.DataFrame:
    """Alpha#33: rank((-1 * ((1 - open/close)^1))) — single-asset: -((1-open/close))."""
    df["wq033"] = -1.0 * (1.0 - df["open"] / df["close"].replace(0, np.nan))
    return df


@factor("wq041", category="wq101", output_columns=["wq041"])
def wq041(df: pd.DataFrame) -> pd.DataFrame:
    """Alpha#41: (((high * low)^0.5) - vwap) approx with typical price as vwap."""
    vwap = (df["high"] + df["low"] + df["close"]) / 3.0
    df["wq041"] = np.sqrt((df["high"] * df["low"]).clip(lower=0)) - vwap
    return df


@factor("wq054", category="wq101", output_columns=["wq054"])
def wq054(df: pd.DataFrame) -> pd.DataFrame:
    """Alpha#54: (-1 * ((low - close) * (open^5))) / ((low - high) * (close^5))."""
    num = -1.0 * (df["low"] - df["close"]) * (df["open"] ** 5)
    den = (df["low"] - df["high"]).replace(0, np.nan) * (df["close"] ** 5)
    df["wq054"] = num / den
    return df


@factor("wq101", category="wq101", output_columns=["wq101"])
def wq101(df: pd.DataFrame) -> pd.DataFrame:
    """Alpha#101: ((close - open) / ((high - low) + .001))."""
    df["wq101"] = (df["close"] - df["open"]) / ((df["high"] - df["low"]) + 0.001)
    return df


def _ts_rank(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window).apply(
        lambda x: float(pd.Series(x).rank(pct=True).iloc[-1]) if len(x) else np.nan,
        raw=True,
    )


@factor("wq002", category="wq101", output_columns=["wq002"])
def wq002(df: pd.DataFrame) -> pd.DataFrame:
    """Alpha#2 proxy: -corr(rank(delta(log(volume),2)), rank((close-open)/open), 6)."""
    dlogv = np.log(df["volume"].replace(0, np.nan)).diff(2)
    ret_oc = (df["close"] - df["open"]) / df["open"].replace(0, np.nan)
    df["wq002"] = -1.0 * dlogv.rank(pct=True).rolling(6).corr(ret_oc.rank(pct=True))
    return df


@factor("wq003", category="wq101", output_columns=["wq003"])
def wq003(df: pd.DataFrame) -> pd.DataFrame:
    """Alpha#3: -corr(rank(open), rank(volume), 10)."""
    df["wq003"] = -1.0 * df["open"].rank(pct=True).rolling(10).corr(df["volume"].rank(pct=True))
    return df


@factor("wq004", category="wq101", output_columns=["wq004"])
def wq004(df: pd.DataFrame) -> pd.DataFrame:
    """Alpha#4 proxy: -ts_rank(rank(low), 9)."""
    df["wq004"] = -1.0 * _ts_rank(df["low"].rank(pct=True), 9)
    return df


@factor("wq008", category="wq101", output_columns=["wq008"])
def wq008(df: pd.DataFrame) -> pd.DataFrame:
    """Alpha#8 proxy: -rank( (sum(open,5)*sum(returns,5)) - delay(...) )."""
    rets = df["close"].pct_change()
    s = df["open"].rolling(5).sum() * rets.rolling(5).sum()
    df["wq008"] = -1.0 * (s - s.shift(10))
    return df


@factor("wq009", category="wq101", output_columns=["wq009"])
def wq009(df: pd.DataFrame) -> pd.DataFrame:
    """Alpha#9: if(0<ts_min(delta(close,1),5), ts_min(...), if(ts_max(...)<0, ts_max(...), delta))."""
    d = df["close"].diff(1)
    ts_min = d.rolling(5).min()
    ts_max = d.rolling(5).max()
    df["wq009"] = np.where(
        0 < ts_min,
        ts_min,
        np.where(ts_max < 0, ts_max, d),
    )
    return df


@factor("wq010", category="wq101", output_columns=["wq010"])
def wq010(df: pd.DataFrame) -> pd.DataFrame:
    """Alpha#10: rank of Alpha#9-like conditional delta."""
    d = df["close"].diff(1)
    ts_min = d.rolling(4).min()
    ts_max = d.rolling(4).max()
    cond = np.where(0 < ts_min, ts_min, np.where(ts_max < 0, ts_max, d))
    df["wq010"] = pd.Series(cond, index=df.index).rank(pct=True)
    return df


@factor("wq018", category="wq101", output_columns=["wq018"])
def wq018(df: pd.DataFrame) -> pd.DataFrame:
    """Alpha#18: -std(abs(close-open),5) + corr(close, open, 10) style proxy."""
    df["wq018"] = -1.0 * (df["close"] - df["open"]).abs().rolling(5).std() + df["close"].rolling(10).corr(df["open"])
    return df


@factor("wq019", category="wq101", output_columns=["wq019"])
def wq019(df: pd.DataFrame) -> pd.DataFrame:
    """Alpha#19: (-1 * sign((close - delay(close,7)) + delta(close,7))) * (1 + rank(1+sum(returns,250)))."""
    rets = df["close"].pct_change()
    part = (df["close"] - df["close"].shift(7)) + df["close"].diff(7)
    df["wq019"] = -1.0 * np.sign(part) * (1.0 + (1.0 + rets.rolling(60).sum()).rank(pct=True))
    return df


@factor("wq020", category="wq101", output_columns=["wq020"])
def wq020(df: pd.DataFrame) -> pd.DataFrame:
    """Alpha#20 proxy: -rank(open-delay(high,1)) * rank(open-delay(close,1)) * rank(open-delay(low,1))."""
    a = (df["open"] - df["high"].shift(1)).rank(pct=True)
    b = (df["open"] - df["close"].shift(1)).rank(pct=True)
    c = (df["open"] - df["low"].shift(1)).rank(pct=True)
    df["wq020"] = -1.0 * a * b * c
    return df


@factor("wq022", category="wq101", output_columns=["wq022"])
def wq022(df: pd.DataFrame) -> pd.DataFrame:
    """Alpha#22: -delta(corr(high, volume, 5), 5) * rank(stddev(close, 20))."""
    corr = df["high"].rolling(5).corr(df["volume"])
    df["wq022"] = -1.0 * corr.diff(5) * df["close"].rolling(20).std().rank(pct=True)
    return df


@factor("wq024", category="wq101", output_columns=["wq024"])
def wq024(df: pd.DataFrame) -> pd.DataFrame:
    """Alpha#24 proxy: if delta(ts_mean(close,100),100)/delay < 0.05 then -delta else -delta(close,3)."""
    ma = df["close"].rolling(100).mean()
    cond = (ma.diff(100) / ma.shift(100).replace(0, np.nan)) < 0.05
    df["wq024"] = np.where(cond, -ma.diff(100), -df["close"].diff(3))
    return df


@factor("wq028", category="wq101", output_columns=["wq028"])
def wq028(df: pd.DataFrame) -> pd.DataFrame:
    """Alpha#28 proxy: scale(corr(adv20, low, 5) + (high+low)/2 - close) — use volume mean as adv."""
    adv = df["volume"].rolling(20).mean()
    mid = (df["high"] + df["low"]) / 2.0
    raw = adv.rolling(5).corr(df["low"]) + mid - df["close"]
    df["wq028"] = (raw - raw.rolling(20).mean()) / raw.rolling(20).std().replace(0, np.nan)
    return df


@factor("wq038", category="wq101", output_columns=["wq038"])
def wq038(df: pd.DataFrame) -> pd.DataFrame:
    """Alpha#38: -rank(ts_rank(close,10)) * rank(close/open)."""
    df["wq038"] = -1.0 * _ts_rank(df["close"], 10).rank(pct=True) * (df["close"] / df["open"].replace(0, np.nan)).rank(pct=True)
    return df


@factor("wq040", category="wq101", output_columns=["wq040"])
def wq040(df: pd.DataFrame) -> pd.DataFrame:
    """Alpha#40: -rank(stddev(high,10)) * corr(high, volume, 10)."""
    df["wq040"] = -1.0 * df["high"].rolling(10).std().rank(pct=True) * df["high"].rolling(10).corr(df["volume"])
    return df


@factor("wq044", category="wq101", output_columns=["wq044"])
def wq044(df: pd.DataFrame) -> pd.DataFrame:
    """Alpha#44: -corr(high, rank(volume), 5)."""
    df["wq044"] = -1.0 * df["high"].rolling(5).corr(df["volume"].rank(pct=True))
    return df


@factor("wq053", category="wq101", output_columns=["wq053"])
def wq053(df: pd.DataFrame) -> pd.DataFrame:
    """Alpha#53: -delta( ((close-low)-(high-close))/(close-low) , 9)."""
    x = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["close"] - df["low"]).replace(0, np.nan)
    df["wq053"] = -1.0 * x.diff(9)
    return df


@factor("wq055", category="wq101", output_columns=["wq055"])
def wq055(df: pd.DataFrame) -> pd.DataFrame:
    """Alpha#55: -corr(rank((close-ts_min(low,12))/(ts_max(high,12)-ts_min(low,12))), rank(volume), 6)."""
    mn = df["low"].rolling(12).min()
    mx = df["high"].rolling(12).max()
    rsv = (df["close"] - mn) / (mx - mn).replace(0, np.nan)
    df["wq055"] = -1.0 * rsv.rank(pct=True).rolling(6).corr(df["volume"].rank(pct=True))
    return df
