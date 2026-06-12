"""模拟实盘报告生成器 — 排行榜、分类统计、净值曲线输出。"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .account_manager import AccountManager

logger = logging.getLogger(__name__)


class PaperReporter:
    """从 AccountManager 生成各类报告。"""

    def __init__(self, accounts: AccountManager) -> None:
        self.accounts = accounts

    def leaderboard_report(self, top_n: int = 20) -> str:
        """生成排行榜文本报告。"""
        lines = ["=" * 80, "模拟实盘排行榜", "=" * 80, ""]

        for market in ("crypto", "futures"):
            market_name = "加密货币" if market == "crypto" else "期货"
            lines.append(f"── {market_name} Top {top_n} ──")
            lines.append(f"{'排名':<4} {'账号':<6} {'策略':<30} {'收益率':>10} {'交易数':>8} {'胜率':>8}")
            lines.append("-" * 70)

            lb = self.accounts.leaderboard(market=market, top_n=top_n)
            for i, acct in enumerate(lb, 1):
                lines.append(
                    f"{i:<4} {acct['account_id']:<6} {acct['strategy_name']:<30} "
                    f"{acct['total_return_pct']:>9.2f}% {acct['total_trades']:>8} "
                    f"{acct['win_rate']:>7.1f}%"
                )
            lines.append("")

        return "\n".join(lines)

    def category_report(self) -> str:
        """按策略家族分类统计。"""
        from .strategy_catalog import FULL_CATALOG

        family_stats: dict[str, list[float]] = {}
        for entry in FULL_CATALOG:
            aid = entry["account_id"]
            acct = self.accounts.get(aid)
            if acct is None:
                continue
            # 提取策略家族 (class_path 的最后一段类名)
            family = entry["class_path"].rsplit(".", 1)[0].rsplit(".", 1)[-1]
            if family not in family_stats:
                family_stats[family] = []
            family_stats[family].append(acct.total_return_pct)

        lines = ["=" * 60, "按策略家族分类统计", "=" * 60, ""]
        lines.append(f"{'家族':<25} {'数量':>5} {'平均收益':>10} {'最佳':>10} {'最差':>10}")
        lines.append("-" * 60)

        for family in sorted(family_stats.keys()):
            returns = family_stats[family]
            count = len(returns)
            avg = sum(returns) / count
            best = max(returns)
            worst = min(returns)
            lines.append(f"{family:<25} {count:>5} {avg:>9.2f}% {best:>9.2f}% {worst:>9.2f}%")

        return "\n".join(lines)

    def save_full_report(self, output_dir: str | Path) -> None:
        """保存完整报告到文件。"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 排行榜
        lb_path = output_dir / "leaderboard.txt"
        lb_path.write_text(self.leaderboard_report(), encoding="utf-8")
        logger.info("Leaderboard saved to %s", lb_path)

        # 分类统计
        cat_path = output_dir / "category_stats.txt"
        cat_path.write_text(self.category_report(), encoding="utf-8")

        # 全账号 JSON
        self.accounts.save(output_dir / "accounts.json")

        # 净值曲线 JSON
        equity_data: dict[str, list[tuple[str, float]]] = {}
        for aid, acct in self.accounts.accounts.items():
            if acct.equity_curve:
                equity_data[str(aid)] = acct.equity_curve

        eq_path = output_dir / "equity_curves.json"
        with open(eq_path, "w") as f:
            json.dump(equity_data, f, indent=2, ensure_ascii=False)
        logger.info("Equity curves saved to %s", eq_path)

        # 摘要
        summary = self.accounts.summary()
        sum_path = output_dir / "summary.json"
        with open(sum_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info("Full report saved to %s", output_dir)
