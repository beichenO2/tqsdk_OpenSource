"""多品种并行回测编排器。"""

from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from decimal import Decimal
from typing import Any, Callable

logger = logging.getLogger(__name__)

SingleInstrumentRunner = Callable[[str], Any]


class MultiInstrumentBacktest:
    """通过组合单品种回测能力实现多品种并行执行。"""

    def __init__(
        self,
        single_instrument_runner: SingleInstrumentRunner,
        *,
        max_workers: int | None = None,
    ) -> None:
        if not callable(single_instrument_runner):
            raise TypeError("single_instrument_runner must be callable")
        self._single_instrument_runner = single_instrument_runner
        self._max_workers = max_workers

    @classmethod
    def from_engine_builder(
        cls,
        engine_builder: Callable[[str], Any],
        *,
        max_workers: int | None = None,
    ) -> "MultiInstrumentBacktest":
        """
        通过引擎构造函数创建多品种回测器。

        engine_builder(symbol) 应返回已配置好的单品种引擎实例（如 BacktestEngine/CryptoBacktestEngine），
        且该实例必须提供 run() 方法。
        """

        def _run_single(symbol: str) -> Any:
            engine = engine_builder(symbol)
            run_method = getattr(engine, "run", None)
            if not callable(run_method):
                raise TypeError(f"engine for {symbol} must implement run()")
            return run_method()

        return cls(_run_single, max_workers=max_workers)

    def run(self, symbol_inputs: list[str]) -> dict[str, Any]:
        """并行执行多品种回测并返回分品种结果与组合汇总。"""
        normalized_symbols = self._normalize_symbols(symbol_inputs)
        if len(normalized_symbols) < 3:
            raise ValueError("MultiInstrumentBacktest requires at least 3 symbols")

        worker_count = self._resolve_workers(len(normalized_symbols))
        wall_start = time.monotonic()

        per_instrument_results: dict[str, dict[str, Any]] = {}
        failed_symbols: dict[str, str] = {}

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures: dict[Future[Any], str] = {
                executor.submit(self._single_instrument_runner, symbol): symbol
                for symbol in normalized_symbols
            }

            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001 - keep per-symbol failure isolated
                    error_message = f"{type(exc).__name__}: {exc}"
                    failed_symbols[symbol] = error_message
                    per_instrument_results[symbol] = {
                        "status": "failed",
                        "error": error_message,
                    }
                    logger.exception("Backtest failed for %s", symbol)
                    continue

                per_instrument_results[symbol] = {
                    "status": "ok",
                    "result": result,
                    "metrics": self._extract_metrics(result),
                }

        wall_seconds = time.monotonic() - wall_start
        aggregate_statistics = self._build_portfolio_summary(
            per_symbol_results=per_instrument_results,
            failed_symbols=failed_symbols,
            symbols_requested=len(normalized_symbols),
            wall_seconds=wall_seconds,
        )

        return {
            "per_instrument_results": per_instrument_results,
            "aggregate_statistics": aggregate_statistics,
            # Compatibility aliases for callers expecting early draft keys.
            "per_symbol_results": per_instrument_results,
            "portfolio_summary": aggregate_statistics,
            "failed_symbols": failed_symbols,
        }

    def _resolve_workers(self, symbol_count: int) -> int:
        if self._max_workers is not None:
            return max(1, self._max_workers)
        return max(1, min(symbol_count, 8))

    def _normalize_symbols(self, symbols: list[str]) -> list[str]:
        seen: set[str] = set()
        normalized: list[str] = []
        for symbol in symbols:
            if not symbol:
                continue
            if symbol in seen:
                continue
            seen.add(symbol)
            normalized.append(symbol)
        return normalized

    def _extract_metrics(self, result: Any) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        metric_map = {
            "initial_equity": ("initial_equity", "initial_capital"),
            "final_equity": ("final_equity", "final_capital"),
            "total_return": ("total_return",),
            "max_drawdown": ("max_drawdown",),
            "sharpe_ratio": ("sharpe_ratio",),
            "total_trades": ("total_trades", "round_trips"),
            "elapsed_seconds": ("elapsed_seconds", "wall_seconds"),
        }

        for key, candidates in metric_map.items():
            value = self._read_field(result, candidates)
            if value is not None:
                metrics[key] = value
        return metrics

    def _read_field(self, obj: Any, names: tuple[str, ...]) -> Any:
        for name in names:
            if isinstance(obj, dict) and name in obj:
                return obj[name]
            if hasattr(obj, name):
                return getattr(obj, name)
        return None

    def _build_portfolio_summary(
        self,
        *,
        per_symbol_results: dict[str, dict[str, Any]],
        failed_symbols: dict[str, str],
        symbols_requested: int,
        wall_seconds: float,
    ) -> dict[str, Any]:
        succeeded = sum(1 for item in per_symbol_results.values() if item.get("status") == "ok")
        failed = len(failed_symbols)

        summary: dict[str, Any] = {
            "symbols_requested": symbols_requested,
            "symbols_succeeded": succeeded,
            "symbols_failed": failed,
            "success_rate": (succeeded / symbols_requested) if symbols_requested else 0.0,
            "wall_seconds": wall_seconds,
        }

        initial_sum = Decimal(0)
        final_sum = Decimal(0)
        has_equity = False
        total_trades = 0

        for item in per_symbol_results.values():
            if item.get("status") != "ok":
                continue
            metrics = item.get("metrics", {})
            initial = self._to_decimal(metrics.get("initial_equity"))
            final = self._to_decimal(metrics.get("final_equity"))
            trades = metrics.get("total_trades")

            if initial is not None and final is not None:
                initial_sum += initial
                final_sum += final
                has_equity = True
            if isinstance(trades, int):
                total_trades += trades

        if has_equity:
            summary["aggregate_initial_equity"] = initial_sum
            summary["aggregate_final_equity"] = final_sum
            if initial_sum != 0:
                summary["aggregate_return"] = (final_sum - initial_sum) / initial_sum

        summary["aggregate_total_trades"] = total_trades
        return summary

    def _to_decimal(self, value: Any) -> Decimal | None:
        if isinstance(value, Decimal):
            return value
        if isinstance(value, int):
            return Decimal(value)
        if isinstance(value, float):
            return Decimal(str(value))
        if isinstance(value, str):
            try:
                return Decimal(value)
            except Exception:  # noqa: BLE001 - tolerant conversion
                return None
        return None
