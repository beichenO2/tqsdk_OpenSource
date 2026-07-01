"""Data collection logic — TqSdk futures tick/kline + Binance crypto kline."""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_ROOT = Path(os.getenv("DATA_ROOT", os.path.expanduser("~/Polarisor/tqsdk/trading-platform/data")))
FUTURES_CACHE = DATA_ROOT / "futures_cache"
CRYPTO_CACHE = DATA_ROOT / "crypto_cache"

# 主力合约列表（期货）
FUTURES_SYMBOLS = [
    # SHFE 上期所
    "KQ.m@SHFE.rb",  # 螺纹钢
    "KQ.m@SHFE.au",  # 黄金
    "KQ.m@SHFE.ag",  # 白银
    "KQ.m@SHFE.cu",  # 铜
    "KQ.m@SHFE.al",  # 铝
    "KQ.m@SHFE.zn",  # 锌
    "KQ.m@SHFE.ni",  # 镍
    # DCE 大商所
    "KQ.m@DCE.i",    # 铁矿石
    "KQ.m@DCE.m",    # 豆粕
    "KQ.m@DCE.y",    # 豆油
    "KQ.m@DCE.p",    # 棕榈油
    "KQ.m@DCE.j",    # 焦炭
    "KQ.m@DCE.jm",   # 焦煤
    "KQ.m@DCE.eg",   # 乙二醇
    "KQ.m@DCE.pp",   # 聚丙烯
    # CZCE 郑商所
    "KQ.m@CZCE.MA",  # 甲醇
    "KQ.m@CZCE.SR",  # 白糖
    "KQ.m@CZCE.CF",  # 棉花
    "KQ.m@CZCE.TA",  # PTA
    "KQ.m@CZCE.FG",  # 玻璃
    "KQ.m@CZCE.SA",  # 纯碱
    "KQ.m@CZCE.AP",  # 苹果
    # INE 能源中心
    "KQ.m@INE.sc",   # 原油
    "KQ.m@INE.lu",   # 低硫燃料油
    # CFFEX 中金所
    "KQ.m@CFFEX.IF", # 沪深300
    "KQ.m@CFFEX.IC", # 中证500
    "KQ.m@CFFEX.IM", # 中证1000
    "KQ.m@CFFEX.IH", # 上证50
    "KQ.m@CFFEX.T",  # 十年期国债
]

# 加密货币（Binance kline）
CRYPTO_SYMBOLS = [
    "btcusdt", "ethusdt", "solusdt", "bnbusdt", "xrpusdt",
    "dogeusdt", "adausdt", "avaxusdt", "linkusdt", "ltcusdt",
]
CRYPTO_INTERVALS = ["1h", "4h", "1d"]


def collect_futures_klines(api: object, symbols: list[str] | None = None) -> int:
    """Fetch daily + 5min klines for futures symbols via TqSdk, save to Parquet.

    Returns the number of symbols successfully collected.
    """
    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas not installed, skipping futures kline collection")
        return 0

    symbols = symbols or FUTURES_SYMBOLS
    FUTURES_CACHE.mkdir(parents=True, exist_ok=True)
    collected = 0

    for sym in symbols:
        try:
            # 日线 (200 根)
            klines_daily = api.get_kline_serial(sym, 86400, 200)
            if klines_daily is not None and len(klines_daily) > 0:
                df = pd.DataFrame(klines_daily)
                safe_name = sym.replace("@", "_").replace(".", "_")
                path = FUTURES_CACHE / f"{safe_name}_daily.parquet"
                df.to_parquet(path, index=False)
                logger.info("saved %d daily bars for %s → %s", len(df), sym, path.name)

            # 5分钟线 (500 根)
            klines_5m = api.get_kline_serial(sym, 300, 500)
            if klines_5m is not None and len(klines_5m) > 0:
                df = pd.DataFrame(klines_5m)
                safe_name = sym.replace("@", "_").replace(".", "_")
                path = FUTURES_CACHE / f"{safe_name}_5m.parquet"
                df.to_parquet(path, index=False)
                logger.info("saved %d 5m bars for %s → %s", len(df), sym, path.name)

            collected += 1
        except Exception as e:
            logger.warning("failed to collect %s: %s", sym, e)

    return collected


def collect_futures_klines_via_gateway(
    gateway_url: str | None = None,
    symbols: list[str] | None = None,
) -> int:
    """Fetch daily + 5min klines via TqSdk Gateway HTTP API (no local credentials)."""
    import json
    import urllib.parse
    import urllib.request

    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas not installed, skipping futures kline collection")
        return 0

    base = (gateway_url or os.getenv("TQSDK_GATEWAY_URL", "http://127.0.0.1:12891")).rstrip("/")
    symbols = symbols or FUTURES_SYMBOLS
    FUTURES_CACHE.mkdir(parents=True, exist_ok=True)
    collected = 0

    def _fetch(sym: str, duration: int, length: int) -> list[dict]:
        url = f"{base}/api/v1/market/klines/{urllib.parse.quote(sym, safe='')}?duration={duration}&length={length}"
        with urllib.request.urlopen(url, timeout=60) as resp:
            body = json.loads(resp.read().decode())
        return body.get("items", [])

    for sym in symbols:
        try:
            daily = _fetch(sym, 86400, 200)
            if daily:
                df = pd.DataFrame(daily)
                safe_name = sym.replace("@", "_").replace(".", "_")
                path = FUTURES_CACHE / f"{safe_name}_daily.parquet"
                df.to_parquet(path, index=False)
                logger.info("saved %d daily bars for %s → %s", len(df), sym, path.name)

            bars_5m = _fetch(sym, 300, 500)
            if bars_5m:
                df = pd.DataFrame(bars_5m)
                safe_name = sym.replace("@", "_").replace(".", "_")
                path = FUTURES_CACHE / f"{safe_name}_5m.parquet"
                df.to_parquet(path, index=False)
                logger.info("saved %d 5m bars for %s → %s", len(df), sym, path.name)

            collected += 1
        except Exception as e:
            logger.warning("failed to collect %s via gateway: %s", sym, e)

    return collected


def _get_proxy_handler() -> urllib.request.ProxyHandler | None:
    """Auto-detect Clash Verge proxy for Binance REST API."""
    import urllib.request

    proxy_env = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if proxy_env:
        return urllib.request.ProxyHandler({"https": proxy_env, "http": proxy_env})

    cfg_path = os.path.expanduser(
        "~/Library/Application Support/io.github.clash-verge-rev.clash-verge-rev/clash-verge.yaml"
    )
    if os.path.exists(cfg_path):
        try:
            import yaml
            with open(cfg_path) as f:
                d = yaml.safe_load(f)
            port = d.get("mixed-port", 7897)
            proxy_url = f"http://127.0.0.1:{port}"
            return urllib.request.ProxyHandler({"https": proxy_url, "http": proxy_url})
        except Exception:
            pass
    return None


def collect_crypto_klines(symbols: list[str] | None = None, intervals: list[str] | None = None) -> int:
    """Fetch klines from Binance REST API (no auth needed), save to Parquet.

    Returns the number of symbol-interval pairs successfully collected.
    """
    import urllib.request
    import json

    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas not installed, skipping crypto kline collection")
        return 0

    symbols = symbols or CRYPTO_SYMBOLS
    intervals = intervals or CRYPTO_INTERVALS
    collected = 0

    proxy_handler = _get_proxy_handler()
    if proxy_handler:
        opener = urllib.request.build_opener(proxy_handler)
        logger.info("crypto kline collection using proxy")
    else:
        opener = urllib.request.build_opener()

    for sym in symbols:
        for interval in intervals:
            try:
                url = f"https://api.binance.com/api/v3/klines?symbol={sym.upper()}&interval={interval}&limit=500"
                req = urllib.request.Request(url)
                with opener.open(req, timeout=15) as resp:
                    data = json.loads(resp.read())

                if not data:
                    continue

                df = pd.DataFrame(data, columns=[
                    "open_time", "open", "high", "low", "close", "volume",
                    "close_time", "quote_volume", "trades", "taker_buy_base",
                    "taker_buy_quote", "ignore",
                ])
                for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
                    df[col] = pd.to_numeric(df[col])
                df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")

                out_dir = CRYPTO_CACHE / sym
                out_dir.mkdir(parents=True, exist_ok=True)
                path = out_dir / f"{interval}.parquet"
                df.to_parquet(path, index=False)
                logger.info("saved %d %s bars for %s → %s", len(df), interval, sym, path)
                collected += 1

            except Exception as e:
                logger.warning("failed to collect %s/%s: %s", sym, interval, e)

    return collected
