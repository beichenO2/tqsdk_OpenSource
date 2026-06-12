"""Export :class:`BacktestResult` to JSON, CSV, and standalone HTML."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from decimal import Decimal
from html import escape
from io import StringIO
from pathlib import Path
from typing import Any
from uuid import UUID

from .models import BacktestResult, EquityCurvePoint, Trade


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class ReportExporter:
    """Serialize backtest results for reporting and archival."""

    def to_json_dict(self, result: BacktestResult) -> dict[str, Any]:
        return {
            "config": _config_dict(result.config),
            "metrics": _metrics_dict(result),
            "trades": [_trade_dict(t) for t in result.trades],
            "equity_curve": [_equity_dict(p) for p in result.equity_curve],
        }

    def export_json(self, result: BacktestResult, path: str | Path) -> None:
        path = Path(path)
        path.write_text(
            json.dumps(self.to_json_dict(result), indent=2, default=_json_default),
            encoding="utf-8",
        )

    def export_csv(self, result: BacktestResult, path: str | Path) -> None:
        path = Path(path)
        buf = StringIO()
        w = csv.writer(buf)
        w.writerow(["# summary"])
        for k, v in _metrics_dict(result).items():
            w.writerow([k, v])
        w.writerow([])
        w.writerow(
            [
                "trade_id",
                "order_id",
                "symbol",
                "side",
                "price",
                "volume",
                "commission",
                "slippage",
                "dt",
            ]
        )
        for t in sorted(result.trades, key=lambda x: x.dt):
            w.writerow(
                [
                    str(t.id),
                    str(t.order_id),
                    t.symbol,
                    t.side.value,
                    str(t.price),
                    t.volume,
                    str(t.commission),
                    str(t.slippage),
                    t.dt.isoformat(),
                ]
            )
        path.write_text(buf.getvalue(), encoding="utf-8")

    def export_html(self, result: BacktestResult, path: str | Path) -> None:
        path = Path(path)
        body = _build_html(self.to_json_dict(result), result)
        path.write_text(body, encoding="utf-8")


def _config_dict(cfg: Any) -> dict[str, Any]:
    return {
        "strategy_id": cfg.strategy_id,
        "symbols": list(cfg.symbols),
        "initial_capital": str(cfg.initial_capital),
        "commission_rate": str(cfg.commission_rate),
        "data_frequency": cfg.data_frequency,
    }


def _metrics_dict(result: BacktestResult) -> dict[str, Any]:
    return {
        "final_equity": str(result.final_equity),
        "total_return": str(result.total_return),
        "annual_return": str(result.annual_return),
        "max_drawdown": str(result.max_drawdown),
        "max_drawdown_pct": str(result.max_drawdown_pct),
        "sharpe_ratio": str(result.sharpe_ratio),
        "sortino_ratio": str(result.sortino_ratio),
        "win_rate": str(result.win_rate),
        "profit_factor": str(result.profit_factor),
        "total_trades": result.total_trades,
        "avg_trade_pnl": str(result.avg_trade_pnl),
        "avg_holding_period": result.avg_holding_period,
        "calmar_ratio": str(result.calmar_ratio),
    }


def _trade_dict(t: Trade) -> dict[str, Any]:
    return {
        "id": str(t.id),
        "order_id": str(t.order_id),
        "symbol": t.symbol,
        "side": t.side.value,
        "price": str(t.price),
        "volume": t.volume,
        "commission": str(t.commission),
        "slippage": str(t.slippage),
        "dt": t.dt.isoformat(),
    }


def _equity_dict(p: EquityCurvePoint) -> dict[str, Any]:
    return {
        "dt": p.dt.isoformat(),
        "equity": str(p.equity),
        "cash": str(p.cash),
        "position_value": str(p.position_value),
        "drawdown": str(p.drawdown),
        "drawdown_pct": str(p.drawdown_pct),
    }


def _equity_svg(points: list[EquityCurvePoint], width: int = 800, height: int = 220) -> str:
    if not points:
        return (
            f'<svg width="{width}" height="{height}" '
            'xmlns="http://www.w3.org/2000/svg">'
            '<text x="10" y="30">No equity data</text></svg>'
        )
    ys = [float(p.equity) for p in points]
    y_min, y_max = min(ys), max(ys)
    pad = 10
    if y_max <= y_min:
        y_max = y_min + 1.0
    n = len(ys)
    coords: list[tuple[float, float]] = []
    for i, y in enumerate(ys):
        x = pad + (width - 2 * pad) * i / max(1, n - 1)
        yn = height - pad - (height - 2 * pad) * (y - y_min) / (y_max - y_min)
        coords.append((x, yn))
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
    return (
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
        f'<rect width="100%" height="100%" fill="#0f1419"/>'
        f'<polyline fill="none" stroke="#3b82f6" stroke-width="1.5" points="{pts}"/>'
        f'<text x="{pad}" y="20" fill="#94a3b8" font-size="12">Equity</text>'
        f"</svg>"
    )


def _build_html(data: dict[str, Any], result: BacktestResult) -> str:
    metrics_rows = "".join(
        f"<tr><td>{escape(str(k))}</td><td>{escape(str(v))}</td></tr>"
        for k, v in data["metrics"].items()
    )
    trade_rows = "".join(
        "<tr>"
        f"<td>{escape(str(t['dt']))}</td>"
        f"<td>{escape(str(t['symbol']))}</td>"
        f"<td>{escape(str(t['side']))}</td>"
        f"<td>{escape(str(t['price']))}</td>"
        f"<td>{escape(str(t['volume']))}</td>"
        f"<td>{escape(str(t['commission']))}</td>"
        "</tr>"
        for t in data["trades"][:500]
    )
    chart = _equity_svg(result.equity_curve)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Backtest Report</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; background: #111827; color: #e5e7eb; }}
h1 {{ font-size: 1.4rem; }}
section {{ margin-bottom: 28px; }}
table {{ border-collapse: collapse; width: 100%; max-width: 960px; }}
th, td {{ border: 1px solid #374151; padding: 8px 10px; text-align: left; }}
th {{ background: #1f2937; }}
.metrics td:first-child {{ color: #9ca3af; width: 220px; }}
.chart-wrap {{
  background: #0f1419; border: 1px solid #374151; border-radius: 8px;
  padding: 12px; max-width: 840px;
}}
</style>
</head>
<body>
<h1>Backtest report</h1>
<section>
<h2>Equity curve</h2>
<div class="chart-wrap">{chart}</div>
</section>
<section>
<h2>Key metrics</h2>
<table class="metrics">
{metrics_rows}
</table>
</section>
<section>
<h2>Trades ({len(data["trades"])})</h2>
<table>
<thead><tr>
<th>Time</th><th>Symbol</th><th>Side</th><th>Price</th><th>Vol</th><th>Commission</th>
</tr></thead>
<tbody>
{trade_rows}
</tbody>
</table>
</section>
</body>
</html>
"""
