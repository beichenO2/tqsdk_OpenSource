"""证据链路由 — 交易决策可解释性 REST API."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from core.db.base import get_session
from core.exceptions import DataNotAvailableError
from explain.chain import EvidenceChain
from explain.persistence import EvidenceStore
from explain.views import (
    DecisionGraphView,
    FactorContributionView,
    TimelineEntry,
    TimelineView,
    decision_graph_view,
    factor_contribution_view,
    timeline_view,
)

router = APIRouter(prefix="/explain", tags=["explain"])


def _timeline_from_chain(chain: EvidenceChain) -> TimelineView:
    entries = [
        TimelineEntry(
            timestamp=event.timestamp.isoformat(),
            event_type=event.event_type,
            data=dict(event.data),
            metadata=dict(event.metadata),
        )
        for event in sorted(chain.events, key=lambda item: item.timestamp)
    ]
    return TimelineView(trade_id=chain.trade_id, symbol=chain.symbol, entries=entries)


@router.get("", response_model=list[TimelineView])
async def list_timelines(
    symbol: str,
    start: datetime,
    end: datetime,
    session: AsyncSession = Depends(get_session),
) -> list[TimelineView]:
    store = EvidenceStore(session)
    chains = await store.list_chains(symbol, start, end)
    return [_timeline_from_chain(chain) for chain in chains]


@router.get("/{trade_id}/timeline", response_model=TimelineView)
@router.get("/timeline/{trade_id}", response_model=TimelineView, include_in_schema=False)
async def get_timeline(
    trade_id: str,
    session: AsyncSession = Depends(get_session),
) -> TimelineView:
    result = await timeline_view(session, trade_id)
    if result is None:
        raise DataNotAvailableError(f"No evidence chain for trade {trade_id}")
    return result


@router.get("/{trade_id}/factors", response_model=FactorContributionView)
@router.get("/factors/{trade_id}", response_model=FactorContributionView, include_in_schema=False)
async def get_factors(
    trade_id: str,
    session: AsyncSession = Depends(get_session),
) -> FactorContributionView:
    result = await factor_contribution_view(session, trade_id)
    if result is None:
        raise DataNotAvailableError(f"No evidence chain for trade {trade_id}")
    return result


@router.get("/{trade_id}/graph", response_model=DecisionGraphView)
@router.get("/graph/{trade_id}", response_model=DecisionGraphView, include_in_schema=False)
async def get_decision_graph(
    trade_id: str,
    session: AsyncSession = Depends(get_session),
) -> DecisionGraphView:
    result = await decision_graph_view(session, trade_id)
    if result is None:
        raise DataNotAvailableError(f"No evidence chain for trade {trade_id}")
    return result
