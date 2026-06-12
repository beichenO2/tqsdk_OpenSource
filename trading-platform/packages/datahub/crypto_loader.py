"""Crypto OHLCV data loader — reads Parquet files from ~/Downloads/crypto_data/.

Supports multiple symbols (BTCUSDT, ETHUSDT, ...) and timeframes (1m to 1d).
Provides resampling, train/test splitting, and feature-ready DataFrames.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(os.path.expanduser("~/Downloads/crypto_data"))


class CryptoDataLoader:
    """Load and prepare crypto OHLCV data from Parquet files."""

    def __init__(self, data_dir: str | Path | None = None) -> None:
        self._data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    def available_symbols(self) -> list[str]:
        if not self._data_dir.exists():
            return []
        return sorted(
            d.name.upper()
            for d in self._data_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

    def available_timeframes(self, symbol: str) -> list[str]:
        sym_dir = self._data_dir / symbol.lower()
        if not sym_dir.exists():
            return []
        return sorted(
            f.stem
            for f in sym_dir.glob("*.parquet")
        )

    def load(
        self,
        symbol: str = "BTCUSDT",
        timeframe: str = "1h",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load OHLCV data for a symbol and timeframe.

        Returns DataFrame with columns: open_time, open, high, low, close,
        volume, quote_volume, trades, taker_buy_volume, taker_buy_quote_volume.
        """
        path = self._data_dir / symbol.lower() / f"{timeframe}.parquet"
        if not path.exists():
            logger.warning("Data file not found: %s", path)
            return pd.DataFrame()

        df = pd.read_parquet(path)

        if "open_time" not in df.columns:
            logger.error("Missing open_time column in %s", path)
            return pd.DataFrame()

        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        df.sort_values("open_time", inplace=True)
        df.reset_index(drop=True, inplace=True)

        if start:
            df = df[df["open_time"] >= pd.Timestamp(start, tz="UTC")]
        if end:
            df = df[df["open_time"] <= pd.Timestamp(end, tz="UTC")]

        logger.info(
            "Loaded %s %s: %d rows [%s → %s]",
            symbol, timeframe, len(df),
            df["open_time"].iloc[0].strftime("%Y-%m-%d") if len(df) > 0 else "N/A",
            df["open_time"].iloc[-1].strftime("%Y-%m-%d") if len(df) > 0 else "N/A",
        )
        return df

    def load_multi_timeframe(
        self,
        symbol: str = "BTCUSDT",
        timeframes: list[str] | None = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> dict[str, pd.DataFrame]:
        """Load multiple timeframes for the same symbol."""
        if timeframes is None:
            timeframes = self.available_timeframes(symbol)
        return {
            tf: self.load(symbol, tf, start, end)
            for tf in timeframes
        }

    def resample(
        self,
        df: pd.DataFrame,
        target_tf: str,
    ) -> pd.DataFrame:
        """Resample OHLCV from lower to higher timeframe."""
        if df.empty:
            return df

        tf_map = {
            "5m": "5min", "15m": "15min", "30m": "30min",
            "1h": "1h", "4h": "4h", "1d": "1D",
        }
        pd_freq = tf_map.get(target_tf, target_tf)

        result = df.set_index("open_time")
        agg = result.resample(pd_freq).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "quote_volume": "sum",
            "trades": "sum",
            "taker_buy_volume": "sum",
            "taker_buy_quote_volume": "sum",
        }).dropna(subset=["close"])

        agg.reset_index(inplace=True)
        return agg

    def prepare_ml_dataframe(
        self,
        symbol: str = "BTCUSDT",
        timeframe: str = "1h",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load data and add derived columns needed for ML features."""
        df = self.load(symbol, timeframe, start, end)
        if df.empty:
            return df

        df["returns_1"] = df["close"].pct_change(1)
        df["returns_5"] = df["close"].pct_change(5)
        df["returns_10"] = df["close"].pct_change(10)
        df["log_returns"] = (df["close"] / df["close"].shift(1)).apply(
            lambda x: __import__("math").log(x) if x and x > 0 else 0
        )

        df["vol_10"] = df["returns_1"].rolling(10).std()
        df["vol_20"] = df["returns_1"].rolling(20).std()
        df["vol_50"] = df["returns_1"].rolling(50).std()

        if "volume" in df.columns and df["volume"].sum() > 0:
            df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
            df["taker_ratio"] = (
                df["taker_buy_volume"] / df["volume"]
            ).where(df["volume"] > 0, 0.5)
        else:
            df["volume_ratio"] = 1.0
            df["taker_ratio"] = 0.5

        df["high_low_range"] = (df["high"] - df["low"]) / df["close"]

        return df

    def load_with_funding(
        self,
        symbol: str = "BTCUSDT",
        timeframe: str = "4h",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load OHLCV data and merge real or synthetic funding rates.

        Tries to load real funding_rate.parquet first. If unavailable,
        generates synthetic rates correlated with price momentum and
        mean-reverting around zero — matching real perpetual funding dynamics.
        """
        df = self.load(symbol, timeframe, start, end)
        if df.empty:
            return df

        premium_path = self._data_dir / symbol.lower() / "premium_index_4h.parquet"
        funding_path = self._data_dir / symbol.lower() / "funding_rate.parquet"

        if premium_path.exists():
            pi = pd.read_parquet(premium_path)
            pi["open_time"] = pd.to_datetime(pi["open_time"], utc=True).dt.as_unit("ns")
            pi = pi[["open_time", "close"]].rename(columns={"close": "premium_index"})
            pi["premium_index"] = pd.to_numeric(pi["premium_index"], errors="coerce")
            pi.sort_values("open_time", inplace=True)

            df["open_time"] = df["open_time"].dt.as_unit("ns")
            df = pd.merge_asof(
                df.sort_values("open_time"), pi, on="open_time", direction="backward",
            )
            interest_per_8h = 0.0001
            df["funding_rate"] = df["premium_index"].fillna(0.0) + interest_per_8h
            df["funding_rate"] = df["funding_rate"].clip(-0.01, 0.01)
            df.drop(columns=["premium_index"], inplace=True)
            logger.info("Merged REAL premium index → funding rate for %s (%d rows)", symbol, pi.shape[0])

        elif funding_path.exists():
            fr = pd.read_parquet(funding_path)
            fr["funding_time"] = pd.to_datetime(fr["funding_time"], utc=True)
            fr.sort_values("funding_time", inplace=True)
            df = pd.merge_asof(
                df.sort_values("open_time"),
                fr[["funding_time", "funding_rate"]].rename(
                    columns={"funding_time": "open_time"}
                ),
                on="open_time",
                direction="backward",
            )
            df["funding_rate"] = df["funding_rate"].ffill().fillna(0.0)
            logger.info("Merged real funding rates for %s", symbol)
        else:
            import numpy as np
            rng = np.random.RandomState(42)
            n = len(df)
            returns = df["close"].pct_change().fillna(0).values
            mom_20 = pd.Series(returns).rolling(20).mean().fillna(0).values
            mom_60 = pd.Series(returns).rolling(60).mean().fillna(0).values
            vol_20 = pd.Series(returns).rolling(20).std().fillna(0.01).values

            base = mom_60 * 3.0 + mom_20 * 2.0
            regime_noise = rng.normal(0, 0.0003, n)
            spikes = rng.uniform(-1, 1, n)
            spike_mask = rng.random(n) < 0.03
            spikes[~spike_mask] = 0
            spikes *= vol_20 * 8

            raw = base + regime_noise + spikes
            funding = np.zeros(n)
            decay = 0.92
            for i in range(n):
                funding[i] = decay * (funding[i - 1] if i > 0 else 0) + (1 - decay) * raw[i]
            funding = np.clip(funding, -0.01, 0.01)
            df["funding_rate"] = funding
            logger.info("Generated synthetic funding rates for %s (no real data)", symbol)

        return df

    def load_multi_symbol(
        self,
        symbols: list[str],
        timeframe: str = "4h",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> dict[str, pd.DataFrame]:
        """Load data for multiple symbols, aligned by timestamp."""
        result = {}
        for sym in symbols:
            df = self.load(sym, timeframe, start, end)
            if not df.empty:
                result[sym] = df
        return result

    def train_test_split(
        self,
        df: pd.DataFrame,
        test_ratio: float = 0.2,
        val_ratio: float = 0.15,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Time-series split (no shuffle). Returns (train, val, test)."""
        n = len(df)
        test_start = int(n * (1 - test_ratio))
        val_start = int(test_start * (1 - val_ratio))
        return df.iloc[:val_start], df.iloc[val_start:test_start], df.iloc[test_start:]
