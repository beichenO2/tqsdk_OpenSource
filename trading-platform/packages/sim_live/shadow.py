"""Shadow trading: paper fills alongside live executions and quality metrics."""

from __future__ import annotations

from collections import defaultdict, deque
from decimal import Decimal

from core.models.order import Order
from core.models.tick import Tick

from .engine import SimMatchingEngine, SubmitOrderResult
from .models import ExecutionQuality, Fill, SimConfig


def _mid_from_tick(tick: Tick | None) -> Decimal | None:
    if tick is None:
        return None
    if tick.bid_price1 is not None and tick.ask_price1 is not None:
        return (tick.bid_price1 + tick.ask_price1) / Decimal(2)
    return tick.last_price


def _signed_slippage_vs_mid(fill: Fill, mid: Decimal | None) -> Decimal | None:
    """Positive = paid more than mid (bad for buys) / received less (bad for sells)."""
    if mid is None or mid == 0:
        return None
    if fill.aggressor_buy:
        return (fill.price - mid) / mid
    return (mid - fill.price) / mid


class ShadowTrader:
    """
    Runs ``SimMatchingEngine`` against the same tick stream as production while
    recording live broker fills. Pairs shadow vs live FIFO per symbol when sides
    match and exposes aggregate execution-quality metrics.
    """

    def __init__(
        self,
        config: SimConfig | None = None,
        *,
        pair_queue_maxlen: int = 10_000,
    ) -> None:
        self._config = config or SimConfig()
        self._engine = SimMatchingEngine(self._config)
        self._last_mid: dict[str, Decimal] = {}
        self._shadow_slips: list[Decimal] = []
        self._live_slips: list[Decimal] = []
        self._live_queue: dict[str, deque[Fill]] = defaultdict(
            lambda: deque(maxlen=pair_queue_maxlen)
        )
        self._shadow_queue: dict[str, deque[Fill]] = defaultdict(
            lambda: deque(maxlen=pair_queue_maxlen)
        )
        self._improvements: list[Decimal] = []

    @property
    def engine(self) -> SimMatchingEngine:
        """Underlying simulation engine (submit orders, cancel, inspect books)."""
        return self._engine

    def update_mid(self, tick: Tick) -> None:
        """Cache mid-price for a symbol (optional if every fill carries an explicit mid)."""
        m = _mid_from_tick(tick)
        if m is not None:
            self._last_mid[tick.symbol] = m

    async def process_tick(self, tick: Tick) -> list[Fill]:
        """Advance shadow state on each market tick and record shadow fills vs mid."""
        self.update_mid(tick)
        mid = self._last_mid.get(tick.symbol) or _mid_from_tick(tick)
        fills = await self._engine.process_tick(tick)
        for f in fills:
            self._append_shadow_fill(f, mid)
        self._pair_all_symbols()
        return fills

    async def shadow_submit(self, order: Order) -> SubmitOrderResult:
        """Submit to the shadow engine and mirror fills into the comparison buffers."""
        res = await self._engine.submit_order(order)
        tick = self._engine.last_tick
        mid = _mid_from_tick(tick) if tick else self._last_mid.get(order.symbol)
        for f in res.fills:
            self._append_shadow_fill(f, mid)
        self._pair_all_symbols()
        return res

    def record_live_fills(
        self,
        fills: list[Fill],
        *,
        mid: Decimal | None = None,
    ) -> None:
        """
        Ingest real executions. When ``mid`` is omitted, the last tick mid for the
        symbol is used (see ``update_mid`` / ``process_tick``).
        """
        for f in fills:
            m = mid if mid is not None else self._last_mid.get(f.symbol)
            slip = _signed_slippage_vs_mid(f, m)
            if slip is not None:
                self._live_slips.append(slip)
            self._live_queue[f.symbol].append(f)
        self._pair_all_symbols()

    def _append_shadow_fill(self, f: Fill, mid: Decimal | None) -> None:
        slip = _signed_slippage_vs_mid(f, mid)
        if slip is not None:
            self._shadow_slips.append(slip)
        self._shadow_queue[f.symbol].append(f)

    def _pair_all_symbols(self) -> None:
        symbols = set(self._live_queue.keys()) | set(self._shadow_queue.keys())
        for sym in symbols:
            self._pair_symbol(sym)

    def _pair_symbol(self, symbol: str) -> None:
        lq = self._live_queue[symbol]
        sq = self._shadow_queue[symbol]
        while lq and sq:
            lf = lq[0]
            sf = sq[0]
            if lf.aggressor_buy != sf.aggressor_buy:
                break
            lq.popleft()
            sq.popleft()
            if lf.aggressor_buy:
                self._improvements.append(lf.price - sf.price)
            else:
                self._improvements.append(sf.price - lf.price)

    def execution_quality(self) -> ExecutionQuality:
        """Aggregate slippage vs mid and mean shadow price improvement vs paired live fills."""
        n_sh = len(self._shadow_slips)
        n_lv = len(self._live_slips)
        avg_sh = (
            sum(self._shadow_slips, start=Decimal(0)) / Decimal(n_sh) if n_sh else Decimal(0)
        )
        avg_lv = (
            sum(self._live_slips, start=Decimal(0)) / Decimal(n_lv) if n_lv else None
        )
        avg_imp = (
            sum(self._improvements, start=Decimal(0)) / Decimal(len(self._improvements))
            if self._improvements
            else None
        )
        return ExecutionQuality(
            sample_count=len(self._improvements),
            shadow_avg_slippage_vs_mid=avg_sh,
            live_avg_slippage_vs_mid=avg_lv,
            shadow_total_fills=n_sh,
            live_total_fills=n_lv,
            avg_price_improvement_vs_live=avg_imp,
        )
