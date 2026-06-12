"""Read models for UI: timeline, factor contributions, and decision graph."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from explain.attribution import compute_factor_attribution
from explain.persistence import EvidenceStore

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from explain.chain import EvidenceChain


class TimelineEntry(BaseModel):
    """One row in a chronological evidence timeline."""

    timestamp: str
    event_type: str
    data: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TimelineView(BaseModel):
    """Chronological list of events for a trade."""

    trade_id: str
    symbol: str
    entries: list[TimelineEntry] = Field(default_factory=list)


class FactorContributionRow(BaseModel):
    """Single factor row for dashboards."""

    factor: str
    weight: float
    source: str


class FactorContributionView(BaseModel):
    """Factors that drove the decision (from signal-derived attribution)."""

    trade_id: str
    symbol: str
    factors: list[FactorContributionRow] = Field(default_factory=list)


class DecisionNode(BaseModel):
    """Node in a decision tree view."""

    id: str
    label: str
    event_type: str
    children: list[DecisionNode] = Field(default_factory=list)


class DecisionGraphView(BaseModel):
    """Structured decision tree derived from the chain order."""

    trade_id: str
    symbol: str
    root: DecisionNode


def _chain_to_timeline(chain: EvidenceChain) -> TimelineView:
    entries = [
        TimelineEntry(
            timestamp=e.timestamp.isoformat(),
            event_type=e.event_type,
            data=dict(e.data),
            metadata=dict(e.metadata),
        )
        for e in sorted(chain.events, key=lambda x: x.timestamp)
    ]
    return TimelineView(trade_id=chain.trade_id, symbol=chain.symbol, entries=entries)


def _build_decision_graph(chain: EvidenceChain) -> DecisionGraphView:
    """Ordered events as children of root; signal nodes include factor leaves."""
    root = DecisionNode(
        id=f"trade:{chain.trade_id}",
        label=f"Trade {chain.trade_id} ({chain.symbol})",
        event_type="root",
        children=[],
    )
    ordered = sorted(chain.events, key=lambda x: x.timestamp)
    for i, ev in enumerate(ordered):
        nid = f"evt:{i}:{ev.event_type}"
        node = DecisionNode(
            id=nid,
            label=ev.event_type.replace("_", " ").title(),
            event_type=ev.event_type,
            children=[],
        )
        if ev.event_type == "signal":
            attrs = compute_factor_attribution(chain)
            for j, fa in enumerate(attrs[:12]):
                node.children.append(
                    DecisionNode(
                        id=f"{nid}:factor:{j}",
                        label=f"{fa.factor} ({fa.weight:.2%})",
                        event_type="factor",
                        children=[],
                    )
                )
        root.children.append(node)

    return DecisionGraphView(trade_id=chain.trade_id, symbol=chain.symbol, root=root)


async def timeline_view(session: AsyncSession, trade_id: str) -> TimelineView | None:
    """
    Build a chronological event list for ``trade_id``.

    Parameters
    ----------
    session
        Async SQLAlchemy session (from ``core.db``).
    trade_id
        Trade identifier matching the stored chain anchor.
    """
    store = EvidenceStore(session)
    chain = await store.get_chain(trade_id)
    if chain is None:
        return None
    return _chain_to_timeline(chain)


async def factor_contribution_view(
    session: AsyncSession,
    trade_id: str,
) -> FactorContributionView | None:
    """Return factor weights for ``trade_id`` using :func:`compute_factor_attribution`."""
    store = EvidenceStore(session)
    chain = await store.get_chain(trade_id)
    if chain is None:
        return None
    attrs = compute_factor_attribution(chain)
    rows = [
        FactorContributionRow(factor=a.factor, weight=a.weight, source=a.source)
        for a in attrs
    ]
    return FactorContributionView(
        trade_id=chain.trade_id,
        symbol=chain.symbol,
        factors=rows,
    )


async def decision_graph_view(session: AsyncSession, trade_id: str) -> DecisionGraphView | None:
    """Return a tree view: root → ordered events, with factors under the signal node."""
    store = EvidenceStore(session)
    chain = await store.get_chain(trade_id)
    if chain is None:
        return None
    return _build_decision_graph(chain)
