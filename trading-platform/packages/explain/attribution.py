"""Post-hoc attribution helpers for evidence chains."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from explain.chain import EvidenceChain, EvidenceEvent


class FactorAttribution(BaseModel):
    """Relative contribution of a named factor to the trade decision."""

    factor: str
    weight: float
    source: str
    detail: dict[str, Any] = Field(default_factory=dict)


class RiskAttributionItem(BaseModel):
    """Impact of a single risk check on the outcome path."""

    rule_name: str
    passed: bool
    impact_weight: float
    detail: dict[str, Any] = Field(default_factory=dict)


class RiskAttributionSummary(BaseModel):
    """Aggregated view of how risk checks shaped the chain."""

    checks: list[RiskAttributionItem] = Field(default_factory=list)
    pass_ratio: float
    blocking_weight: float


def _abs_numeric_values(obj: Any, prefix: str = "") -> list[tuple[str, float]]:
    """Flatten nested dicts/lists into (name, abs value) pairs for weighting."""
    out: list[tuple[str, float]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            name = f"{prefix}.{k}" if prefix else str(k)
            out.extend(_abs_numeric_values(v, name))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            name = f"{prefix}[{i}]"
            out.extend(_abs_numeric_values(v, name))
    elif isinstance(obj, bool):
        return out
    elif isinstance(obj, int | float):
        val = float(obj)
        if val != 0.0:
            key = prefix or "value"
            out.append((key, abs(val)))
    return out


def compute_factor_attribution(chain: EvidenceChain) -> list[FactorAttribution]:
    """
    Estimate each factor's contribution weight from signal (and related) events.

    Uses absolute magnitudes of numeric fields in event ``data`` (e.g. ``model_scores``,
    ``momentum``) and normalizes to sum to 1.0 per chain. Non-signal events are skipped.
    """
    raw: list[tuple[str, float, str]] = []
    for ev in chain.events:
        if ev.event_type != "signal":
            continue
        pairs = _abs_numeric_values(ev.data)
        for name, mag in pairs:
            raw.append((name, mag, ev.timestamp.isoformat()))

    if not raw:
        return []

    total = sum(m for _, m, _ in raw)
    if total <= 0:
        return []

    # Merge duplicate factor names by summing mass before normalization
    merged: dict[str, tuple[float, str]] = {}
    for name, mag, ts in raw:
        if name in merged:
            prev_mag, prev_ts = merged[name]
            merged[name] = (prev_mag + mag, prev_ts)
        else:
            merged[name] = (mag, ts)

    out: list[FactorAttribution] = []
    total_merged = sum(m for m, _ in merged.values())
    for name, (mag, ts) in sorted(merged.items(), key=lambda x: -x[1][0]):
        out.append(
            FactorAttribution(
                factor=name,
                weight=mag / total_merged,
                source=f"signal@{ts}",
                detail={"raw_mass": mag},
            )
        )
    return out


def _risk_event_weight(ev: EvidenceEvent) -> float:
    """Default severity weight; larger when the check failed."""
    data = ev.data
    w = 1.0
    if isinstance(data.get("severity"), int | float):
        w = max(w, float(data["severity"]))
    if data.get("passed") is False:
        w += 1.0
    return w


def compute_risk_attribution(chain: EvidenceChain) -> RiskAttributionSummary:
    """
    Analyze risk-check events: pass ratio and relative blocking weight.

    ``blocking_weight`` is the share of total risk weight attributed to failed checks.
    """
    items: list[RiskAttributionItem] = []
    total_w = 0.0
    block_w = 0.0
    passed_n = 0
    total_n = 0

    for ev in chain.events:
        if ev.event_type != "risk_check":
            continue
        passed = bool(ev.data.get("passed", True))
        rule_name = str(ev.data.get("rule_name") or ev.data.get("rule") or "unknown")
        iw = _risk_event_weight(ev)
        total_w += iw
        if not passed:
            block_w += iw
        if passed:
            passed_n += 1
        total_n += 1
        items.append(
            RiskAttributionItem(
                rule_name=rule_name,
                passed=passed,
                impact_weight=iw,
                detail={k: v for k, v in ev.data.items() if k not in ("passed",)},
            )
        )

    pass_ratio = passed_n / total_n if total_n else 0.0
    blocking = block_w / total_w if total_w > 0 else 0.0
    return RiskAttributionSummary(
        checks=items,
        pass_ratio=pass_ratio,
        blocking_weight=blocking,
    )
