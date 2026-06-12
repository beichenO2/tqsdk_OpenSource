"""模拟实盘调度器 — 驱动 200 个策略在历史/实时数据上运行。

核心功能:
1. 按时间顺序推送 bar 数据给每个策略
2. 策略信号 → 通过 SimMatchingEngine 撮合 (200ms 延迟)
3. 成交结果 → 更新 AccountManager 中对应账号的持仓/资金
4. 定期快照净值曲线
"""

from __future__ import annotations

import logging
from typing import Any

from strategy.base import BaseStrategy, Signal, SignalType

from .account_manager import AccountManager, SimAccount
from .models import SimConfig

logger = logging.getLogger(__name__)


class PaperScheduler:
    """模拟实盘调度器。"""

    def __init__(
        self,
        accounts: AccountManager,
        strategies: dict[int, BaseStrategy],
        sim_config: SimConfig | None = None,
    ) -> None:
        self.accounts = accounts
        self.strategies = strategies

        if sim_config is None:
            sim_config = SimConfig(
                latency_ms=200,
                latency_jitter_ms=50,
                commission_rate=__import__("decimal").Decimal("0.0004"),
                slippage_ticks=1,
            )
        self.sim_config = sim_config
        self._bar_count = 0
        self._snapshot_interval = 100
        self._started = False

    def _commission_rate(self, market: str) -> float:
        if market == "crypto":
            return 0.0004
        return 0.00005

    def _process_signal(
        self,
        signal: Signal,
        account: SimAccount,
        timestamp: str,
    ) -> bool:
        """处理策略信号，更新账号持仓。返回是否实际执行。

        仓位管理:
        - 加密货币市场禁止做空 (9年数据含多轮牛市，做空长期必死)
        - 信号强度 ≥ 0.7: 重仓 (总资金 80%)
        - 信号强度 0.4-0.7: 中仓 (总资金 50%)
        - 信号强度 < 0.4: 轻仓 (总资金 25%)
        - 最大回撤保护: 净值跌破初始资金 70% 时停止开仓
        """
        symbol = signal.symbol
        price = signal.price or 0
        if price <= 0:
            return False
        comm_rate = self._commission_rate(account.market)

        if signal.signal_type == SignalType.LONG_ENTRY:
            if symbol in account.positions:
                return False
            if account.total_equity < account.initial_capital * 0.70:
                return False

            strength = signal.strength
            if strength >= 0.7:
                alloc_pct = 0.80
            elif strength >= 0.4:
                alloc_pct = 0.50
            else:
                alloc_pct = 0.25

            alloc_capital = account.capital * alloc_pct
            qty = alloc_capital / price
            if qty <= 0:
                return False
            commission = price * qty * comm_rate
            if account.capital < alloc_capital + commission:
                alloc_capital = account.capital * 0.9
                qty = alloc_capital / price
                commission = price * qty * comm_rate

            account.open_position(
                symbol=symbol, side="long", qty=qty, price=price,
                commission=commission, reason=signal.reason, timestamp=timestamp,
            )
            return True

        elif signal.signal_type == SignalType.SHORT_ENTRY:
            # 加密货币市场禁止做空
            if account.market == "crypto":
                return False
            if symbol in account.positions:
                return False
            if account.total_equity < account.initial_capital * 0.70:
                return False

            strength = signal.strength
            if strength >= 0.7:
                alloc_pct = 0.80
            elif strength >= 0.4:
                alloc_pct = 0.50
            else:
                alloc_pct = 0.25

            alloc_capital = account.capital * alloc_pct
            qty = alloc_capital / price
            if qty <= 0:
                return False
            commission = price * qty * comm_rate
            account.open_position(
                symbol=symbol, side="short", qty=qty, price=price,
                commission=commission, reason=signal.reason, timestamp=timestamp,
            )
            return True

        elif signal.signal_type in (SignalType.LONG_EXIT, SignalType.SHORT_EXIT):
            if symbol not in account.positions:
                return False
            commission = price * account.positions[symbol].qty * comm_rate
            account.close_position(
                symbol=symbol, price=price, commission=commission,
                reason=signal.reason, timestamp=timestamp,
            )
            return True

        return False

    async def run_bar(
        self,
        timestamp: str,
        market_data: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """处理一根 bar 的数据，驱动所有策略。

        Args:
            timestamp: ISO 时间戳
            market_data: {symbol: {open, high, low, close, volume, ...}}

        Returns:
            本轮统计: 信号数、成交数
        """
        if not self._started:
            self._started = True
            for strategy in self.strategies.values():
                try:
                    await strategy.on_start()
                except Exception as exc:
                    logger.warning("on_start failed for %s: %s", strategy.name, exc)

        self._bar_count += 1
        total_signals = 0
        total_fills = 0

        for account_id, strategy in self.strategies.items():
            account = self.accounts.get(account_id)
            if account is None or not account.is_active:
                continue

            # 更新未实现盈亏
            for symbol in list(account.positions.keys()):
                if symbol in market_data:
                    account.update_unrealized(symbol, market_data[symbol]["close"])

            # 只推送该策略关注的品种
            for symbol in strategy.config.symbols:
                bar = market_data.get(symbol)
                if not bar:
                    continue

                try:
                    signals = await strategy.on_bar(symbol, bar)
                except Exception as e:
                    logger.warning(
                        "Strategy %s (account %d) error on %s: %s",
                        strategy.name, account_id, symbol, e,
                    )
                    continue

                for sig in signals:
                    total_signals += 1
                    executed = self._process_signal(sig, account, timestamp)
                    if executed:
                        total_fills += 1

            # 定期快照
            if self._bar_count % self._snapshot_interval == 0:
                account.snapshot_equity(timestamp)

        return {
            "bar_index": self._bar_count,
            "timestamp": timestamp,
            "total_signals": total_signals,
            "total_fills": total_fills,
        }

    async def shutdown(self) -> None:
        """Call on_stop on all strategies."""
        for strategy in self.strategies.values():
            try:
                await strategy.on_stop()
            except Exception as exc:
                logger.warning("on_stop failed for %s: %s", strategy.name, exc)

    async def run_history(
        self,
        bars_by_symbol: dict[str, list[dict[str, Any]]],
        progress_every: int = 500,
    ) -> dict[str, Any]:
        """回放历史数据运行所有策略。

        Args:
            bars_by_symbol: {symbol: [bar_dicts]} 按时间顺序
            progress_every: 每 N 根 bar 打印进度

        Returns:
            运行统计
        """
        # 构建统一时间线
        all_timestamps: set[str] = set()
        for bars in bars_by_symbol.values():
            for bar in bars:
                ts = bar.get("timestamp") or bar.get("datetime") or ""
                if ts:
                    all_timestamps.add(ts)

        sorted_ts = sorted(all_timestamps)
        total_bars = len(sorted_ts)
        logger.info("PaperScheduler: %d timestamps, %d symbols", total_bars, len(bars_by_symbol))

        # 建索引
        ts_index: dict[str, dict[str, dict[str, Any]]] = {}
        for symbol, bars in bars_by_symbol.items():
            for bar in bars:
                ts = bar.get("timestamp") or bar.get("datetime") or ""
                if ts not in ts_index:
                    ts_index[ts] = {}
                ts_index[ts][symbol] = bar

        total_signals = 0
        total_fills = 0

        for i, ts in enumerate(sorted_ts):
            market_data = ts_index.get(ts, {})
            stats = await self.run_bar(ts, market_data)
            total_signals += stats["total_signals"]
            total_fills += stats["total_fills"]

            if (i + 1) % progress_every == 0:
                logger.info(
                    "Progress: %d/%d (%.1f%%) | signals=%d fills=%d",
                    i + 1, total_bars, (i + 1) / total_bars * 100,
                    total_signals, total_fills,
                )

        # 最终快照
        for account in self.accounts.accounts.values():
            account.snapshot_equity(sorted_ts[-1] if sorted_ts else "end")

        summary = self.accounts.summary()
        logger.info(
            "PaperScheduler complete: %d bars, %d signals, %d fills",
            total_bars, total_signals, total_fills,
        )
        logger.info("Crypto avg return: %.2f%%", summary["crypto"]["avg_return"])
        logger.info("Futures avg return: %.2f%%", summary["futures"]["avg_return"])

        return {
            "total_bars": total_bars,
            "total_signals": total_signals,
            "total_fills": total_fills,
            "summary": summary,
        }
