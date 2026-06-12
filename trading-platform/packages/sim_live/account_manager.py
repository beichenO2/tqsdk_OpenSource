"""模拟账号管理器 — 管理200个独立账号的资金/持仓/PnL。

每个账号绑定一个策略实例，拥有独立的：
- 初始/当前资金
- 持仓(symbol→Position)
- 交易历史
- 净值曲线
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AccountPosition:
    """单品种持仓。"""
    symbol: str
    side: str = ""  # "long" / "short" / ""
    qty: float = 0.0
    avg_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class TradeRecord:
    """单笔成交记录。"""
    account_id: int
    symbol: str
    side: str
    price: float
    qty: float
    pnl: float = 0.0
    commission: float = 0.0
    timestamp: str = ""
    reason: str = ""


@dataclass
class SimAccount:
    """模拟账号。"""
    account_id: int
    market: str  # "crypto" / "futures"
    strategy_name: str = ""
    strategy_params: dict[str, Any] = field(default_factory=dict)
    initial_capital: float = 100_000.0
    capital: float = 100_000.0
    positions: dict[str, AccountPosition] = field(default_factory=dict)
    trades: list[TradeRecord] = field(default_factory=list)
    equity_curve: list[tuple[str, float]] = field(default_factory=list)
    is_active: bool = True

    @property
    def total_position_value(self) -> float:
        """所有持仓的可回收价值 = 初始保证金 + 未实现盈亏。"""
        return sum(p.avg_price * p.qty + p.unrealized_pnl for p in self.positions.values())

    @property
    def total_unrealized(self) -> float:
        return sum(p.unrealized_pnl for p in self.positions.values())

    @property
    def total_equity(self) -> float:
        return self.capital + self.total_position_value

    @property
    def total_return_pct(self) -> float:
        if self.initial_capital == 0:
            return 0.0
        return (self.total_equity - self.initial_capital) / self.initial_capital * 100

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        exits = [t for t in self.trades if t.pnl != 0]
        if not exits:
            return 0.0
        wins = [t for t in exits if t.pnl > 0]
        return len(wins) / len(exits) * 100

    def open_position(
        self, symbol: str, side: str, qty: float, price: float,
        commission: float = 0.0, reason: str = "", timestamp: str = "",
    ) -> None:
        cost = price * qty + commission
        self.capital -= cost
        self.positions[symbol] = AccountPosition(
            symbol=symbol, side=side, qty=qty, avg_price=price,
        )
        self.trades.append(TradeRecord(
            account_id=self.account_id, symbol=symbol,
            side=f"{side}_entry", price=price, qty=qty,
            commission=commission, reason=reason, timestamp=timestamp,
        ))

    def close_position(
        self, symbol: str, price: float, commission: float = 0.0,
        reason: str = "", timestamp: str = "",
    ) -> float:
        pos = self.positions.get(symbol)
        if not pos or pos.qty == 0:
            return 0.0

        if pos.side == "long":
            pnl = (price - pos.avg_price) * pos.qty
            # 卖出获得的现金
            proceeds = price * pos.qty
        else:
            pnl = (pos.avg_price - price) * pos.qty
            # 做空平仓: 归还保证金 + 盈亏
            proceeds = pos.avg_price * pos.qty + pnl

        self.capital += proceeds - commission
        self.trades.append(TradeRecord(
            account_id=self.account_id, symbol=symbol,
            side=f"{pos.side}_exit", price=price, qty=pos.qty,
            pnl=pnl - commission, commission=commission,
            reason=reason, timestamp=timestamp,
        ))
        del self.positions[symbol]
        return pnl - commission

    def update_unrealized(self, symbol: str, current_price: float) -> None:
        pos = self.positions.get(symbol)
        if not pos:
            return
        if pos.side == "long":
            pos.unrealized_pnl = (current_price - pos.avg_price) * pos.qty
        elif pos.side == "short":
            pos.unrealized_pnl = (pos.avg_price - current_price) * pos.qty

    def snapshot_equity(self, timestamp: str) -> None:
        self.equity_curve.append((timestamp, self.total_equity))

    def to_dict(self) -> dict[str, Any]:
        return {
            "account_id": self.account_id,
            "market": self.market,
            "strategy_name": self.strategy_name,
            "initial_capital": self.initial_capital,
            "capital": round(self.capital, 2),
            "total_equity": round(self.total_equity, 2),
            "total_return_pct": round(self.total_return_pct, 2),
            "total_trades": self.total_trades,
            "win_rate": round(self.win_rate, 1),
            "is_active": self.is_active,
            "positions": {
                k: {"side": v.side, "qty": v.qty, "avg_price": v.avg_price, "unrealized": round(v.unrealized_pnl, 2)}
                for k, v in self.positions.items()
            },
        }


class AccountManager:
    """200个模拟账号的中央管理器。"""

    def __init__(
        self,
        crypto_count: int = 100,
        futures_count: int = 100,
        crypto_capital: float = 100_000.0,
        futures_capital: float = 1_000_000.0,
    ) -> None:
        self.accounts: dict[int, SimAccount] = {}

        for i in range(1, crypto_count + 1):
            self.accounts[i] = SimAccount(
                account_id=i, market="crypto",
                initial_capital=crypto_capital, capital=crypto_capital,
            )

        for i in range(crypto_count + 1, crypto_count + futures_count + 1):
            self.accounts[i] = SimAccount(
                account_id=i, market="futures",
                initial_capital=futures_capital, capital=futures_capital,
            )

        logger.info(
            "AccountManager: %d crypto + %d futures = %d accounts",
            crypto_count, futures_count, len(self.accounts),
        )

    def get(self, account_id: int) -> SimAccount | None:
        return self.accounts.get(account_id)

    def crypto_accounts(self) -> list[SimAccount]:
        return [a for a in self.accounts.values() if a.market == "crypto"]

    def futures_accounts(self) -> list[SimAccount]:
        return [a for a in self.accounts.values() if a.market == "futures"]

    def assign_strategy(self, account_id: int, name: str, params: dict[str, Any] | None = None) -> None:
        acct = self.accounts.get(account_id)
        if acct:
            acct.strategy_name = name
            acct.strategy_params = params or {}

    def leaderboard(self, market: str | None = None, top_n: int = 20) -> list[dict[str, Any]]:
        accts = list(self.accounts.values())
        if market:
            accts = [a for a in accts if a.market == market]
        accts.sort(key=lambda a: a.total_return_pct, reverse=True)
        return [a.to_dict() for a in accts[:top_n]]

    def summary(self) -> dict[str, Any]:
        crypto = self.crypto_accounts()
        futures = self.futures_accounts()
        return {
            "crypto": {
                "count": len(crypto),
                "avg_return": round(sum(a.total_return_pct for a in crypto) / max(len(crypto), 1), 2),
                "best": max((a.total_return_pct for a in crypto), default=0),
                "worst": min((a.total_return_pct for a in crypto), default=0),
                "profitable": sum(1 for a in crypto if a.total_return_pct > 0),
            },
            "futures": {
                "count": len(futures),
                "avg_return": round(sum(a.total_return_pct for a in futures) / max(len(futures), 1), 2),
                "best": max((a.total_return_pct for a in futures), default=0),
                "worst": min((a.total_return_pct for a in futures), default=0),
                "profitable": sum(1 for a in futures if a.total_return_pct > 0),
            },
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            str(k): v.to_dict()
            for k, v in self.accounts.items()
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("Saved %d accounts to %s", len(self.accounts), path)
