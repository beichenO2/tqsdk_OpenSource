"""数据提供者 - 对接不同数据源的适配器"""

from datahub.providers.base import DataProvider
from datahub.providers.blockbeats import BlockBeatsProvider
from datahub.providers.coinmarketcap import CoinMarketCapProvider
from datahub.providers.coinank import CoinAnkProvider
from datahub.providers.dune import DuneProvider

__all__ = [
    "DataProvider",
    "BlockBeatsProvider",
    "CoinMarketCapProvider",
    "CoinAnkProvider",
    "DuneProvider",
]
