"""Smoke import tests for advanced backtest analysis modules."""

from __future__ import annotations


def test_import_monte_carlo():
    from backtest.monte_carlo import MonteCarloAnalyzer, MonteCarloResult

    assert MonteCarloAnalyzer is not None
    assert MonteCarloResult is not None


def test_import_rolling():
    from backtest.rolling_analyzer import RollingAnalyzer

    assert RollingAnalyzer is not None


def test_import_trade():
    from backtest.trade_analyzer import TradeAnalysisSummary, TradeAnalyzer

    assert TradeAnalyzer is not None
    assert TradeAnalysisSummary is not None


def test_import_drawdown():
    from backtest.drawdown_analyzer import DrawdownAnalysis, DrawdownAnalyzer, DrawdownEvent

    assert DrawdownAnalyzer is not None
    assert DrawdownEvent is not None
    assert DrawdownAnalysis is not None


def test_import_exporter():
    from backtest.exporter import ReportExporter

    assert ReportExporter is not None
