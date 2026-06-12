"""Shared dataset loading for volbar overfit gates (OOS / WF / MC)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from datahub.crypto_loader import CryptoDataLoader
from eternal_optimizer_volbar import convert_to_vol_bars

# SOL listed ~2020-09; validate multi-asset portfolio on common tradable era only.
COMMON_ERA_START = pd.Timestamp("2020-09-01", tz="UTC")


def load_volbar_gate_datasets(
    symbols: list[str],
    *,
    align_common_era: bool = True,
    data_dir: Path | None = None,
) -> tuple[dict, list, dict]:
    """Load full volbar series + OOS holdout slices for gate validation."""
    loader = CryptoDataLoader(data_dir=data_dir or Path.home() / "Downloads" / "crypto_data")
    full: dict = {}
    oos_list: list = []
    meta = {"align_common_era": align_common_era, "common_era_start": None}

    for sym in symbols:
        bars = loader.load_with_funding(sym, "1h")
        if bars.empty:
            bars = loader.load(sym, "1h")
        if bars.empty or len(bars) < 2000:
            continue
        if align_common_era:
            bars = bars[bars["open_time"] >= COMMON_ERA_START].copy()
            if meta["common_era_start"] is None:
                meta["common_era_start"] = COMMON_ERA_START.isoformat()
        if len(bars) < 2000:
            continue
        vb = convert_to_vol_bars(bars, atr_multiplier=1.0, atr_period=14)
        n = len(vb)
        val_end = int(n * 0.75)
        oos = vb.iloc[val_end:].copy()
        oos.attrs = vb.attrs.copy()
        full[sym] = vb
        oos_list.append(oos)

    meta["symbols_loaded"] = list(full.keys())
    return full, oos_list, meta
