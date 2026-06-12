"""交易所与资产类别枚举."""

from enum import StrEnum


class Exchange(StrEnum):
    SHFE = "SHFE"
    DCE = "DCE"
    CZCE = "CZCE"
    CFFEX = "CFFEX"
    INE = "INE"
    GFEX = "GFEX"
    BINANCE = "BINANCE"
    OKX = "OKX"


class AssetClass(StrEnum):
    FUTURES = "FUTURES"
    OPTIONS = "OPTIONS"
    CRYPTO_SPOT = "CRYPTO_SPOT"
    CRYPTO_PERP = "CRYPTO_PERP"
