"""Persist evidence chains to PostgreSQL via SQLAlchemy ORM."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import orjson
from sqlalchemy import select

from core.db.base import generate_uuid
from core.db.models.evidence import DecisionLog
from explain.chain import EvidenceChain

if TYPE_CHECKING:
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncSession

EVIDENCE_CHAIN_ACTION = "EVIDENCE_CHAIN"
"""Value stored in :attr:`DecisionLog.action` for serialized evidence chains."""


def _trade_anchor_json(trade_id: str) -> str:
    """Canonical JSON for lookup rows (stable key order)."""
    return json.dumps({"trade_id": trade_id}, sort_keys=True, separators=(",", ":"))


def _chain_to_summary_bytes(chain: EvidenceChain) -> str:
    """Serialize chain for ``DecisionLog.summary`` (JSON text)."""
    payload: dict[str, Any] = chain.model_dump(mode="json")
    return orjson.dumps(payload).decode("utf-8")


def _summary_to_chain(summary: str) -> EvidenceChain:
    return EvidenceChain.model_validate(orjson.loads(summary))


class EvidenceStore:
    """Async persistence for :class:`EvidenceChain` using :class:`DecisionLog`."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save_chain(
        self,
        chain: EvidenceChain,
        user_id: str,
        *,
        strategy_id: str | None = None,
        order_id: str | None = None,
    ) -> str:
        """
        Insert or update a chain row.

        Full chain JSON is stored in ``DecisionLog.summary``. ``evidence_ids_json``
        holds a small JSON object ``{"trade_id": "<id>"}`` for lookup.

        Returns
        -------
        str
            The primary key of the ``decision_logs`` row.
        """
        anchor = _trade_anchor_json(chain.trade_id)
        stmt = select(DecisionLog).where(
            DecisionLog.action == EVIDENCE_CHAIN_ACTION,
            DecisionLog.evidence_ids_json == anchor,
        )
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()
        summary = _chain_to_summary_bytes(chain)
        decided_at = chain.finalized_at or chain.opened_at
        if row is None:
            row = DecisionLog(
                id=generate_uuid(),
                user_id=user_id,
                strategy_id=strategy_id,
                order_id=order_id,
                action=EVIDENCE_CHAIN_ACTION,
                instrument_symbol=chain.symbol,
                summary=summary,
                evidence_ids_json=anchor,
                decided_at=decided_at,
            )
            self._session.add(row)
        else:
            row.user_id = user_id
            row.strategy_id = strategy_id
            row.order_id = order_id
            row.instrument_symbol = chain.symbol
            row.summary = summary
            row.decided_at = decided_at
        await self._session.flush()
        return row.id

    async def get_chain(self, trade_id: str) -> EvidenceChain | None:
        """Load a chain by ``trade_id``, or ``None`` if missing."""
        anchor = _trade_anchor_json(trade_id)
        stmt = select(DecisionLog).where(
            DecisionLog.action == EVIDENCE_CHAIN_ACTION,
            DecisionLog.evidence_ids_json == anchor,
        )
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()
        if row is None:
            return None
        return _summary_to_chain(row.summary)

    async def list_chains(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> list[EvidenceChain]:
        """Return chains for ``instrument_symbol`` with ``decided_at`` in ``[start, end]``."""
        stmt = (
            select(DecisionLog)
            .where(DecisionLog.action == EVIDENCE_CHAIN_ACTION)
            .where(DecisionLog.instrument_symbol == symbol)
            .where(DecisionLog.decided_at >= start)
            .where(DecisionLog.decided_at <= end)
            .order_by(DecisionLog.decided_at)
        )
        result = await self._session.execute(stmt)
        chains: list[EvidenceChain] = []
        for row in result.scalars():
            try:
                chains.append(_summary_to_chain(row.summary))
            except (orjson.JSONDecodeError, ValueError):
                continue
        return chains
