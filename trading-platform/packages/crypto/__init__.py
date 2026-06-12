"""Crypto trading package — fully decoupled from futures.

This package contains all cryptocurrency-specific trading logic:
- strategies/ — BTC/ETH/SOL trading strategies
- ml/ — Crypto ML training pipelines
- datahub/ — Crypto data loaders and providers
- backtest/ — Crypto-specific backtest engine (cost model, orderbook, slippage)
- scripts/ — Crypto-specific scripts (backtest runners, param optimizers)

Shared infrastructure (BaseStrategy, FeatureEngine, ML base classes) lives in
the parent packages/ namespace and is imported as needed.
"""
