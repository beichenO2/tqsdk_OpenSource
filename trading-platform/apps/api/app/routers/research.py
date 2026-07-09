"""Research workbench API — create, execute, and manage strategy research runs."""

from __future__ import annotations

import asyncio
import json as _json
import logging
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "packages"))

from experiment.research_run import ResearchRun, RunStatus, RunStore

logger = logging.getLogger(__name__)

RESEARCH_DIR = REPO_ROOT / "output" / "research"
_store = RunStore(RESEARCH_DIR)

# SSE event bus — lightweight in-process pub/sub for research run events
_sse_subscribers: dict[str, list[asyncio.Queue]] = {}


def _publish_sse(run_id: str, event: str, data: dict[str, Any]) -> None:
    """Push an event to all SSE subscribers watching this run_id."""
    payload = {"event": event, "run_id": run_id, "ts": time.time(), **data}
    for q in _sse_subscribers.get(run_id, []):
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            pass

router = APIRouter(prefix="/research", tags=["research"])


class CreateRunRequest(BaseModel):
    prompt: str
    strategy_name: str = ""
    symbols: list[str] = []
    timeframe: str = "5m"
    config: dict[str, Any] = {}
    tags: list[str] = []


class UpdateRunRequest(BaseModel):
    status: str | None = None
    notes: str | None = None
    tags: list[str] | None = None
    metrics: dict[str, float] | None = None


class DiagnosticRequest(BaseModel):
    category: str
    code: str
    message: str
    severity: str = "warning"


class IterationRequest(BaseModel):
    prompt: str
    changes: str
    metrics_before: dict[str, float] = {}
    metrics_after: dict[str, float] = {}


class ValidationRequest(BaseModel):
    gate: str
    passed: bool
    metrics: dict[str, float] = {}
    thresholds: dict[str, float] = {}


@router.get("/runs")
async def list_runs(limit: int = 50):
    return {"runs": _store.list_runs(limit=limit)}


@router.post("/runs")
async def create_run(req: CreateRunRequest):
    run = ResearchRun(
        prompt=req.prompt,
        strategy_name=req.strategy_name,
        symbols=req.symbols,
        timeframe=req.timeframe,
        config=req.config,
        tags=req.tags,
    )
    _store.save(run)
    return {"run_id": run.run_id, "status": run.status.value}


@router.get("/runs/{run_id}")
async def get_run(run_id: str):
    run = _store.load(run_id)
    if not run:
        raise HTTPException(404, f"Run {run_id} not found")
    return run.to_dict()


@router.patch("/runs/{run_id}")
async def update_run(run_id: str, req: UpdateRunRequest):
    run = _store.load(run_id)
    if not run:
        raise HTTPException(404, f"Run {run_id} not found")
    if req.status:
        run.status = RunStatus(req.status)
    if req.notes is not None:
        run.notes = req.notes
    if req.tags is not None:
        run.tags = req.tags
    if req.metrics:
        run.metrics.update(req.metrics)
    _store.save(run)
    return {"ok": True, "run_id": run_id}


@router.delete("/runs/{run_id}")
async def delete_run(run_id: str):
    if _store.delete(run_id):
        return {"ok": True}
    raise HTTPException(404, f"Run {run_id} not found")


@router.post("/runs/{run_id}/diagnostics")
async def add_diagnostic(run_id: str, req: DiagnosticRequest):
    run = _store.load(run_id)
    if not run:
        raise HTTPException(404, f"Run {run_id} not found")
    run.add_diagnostic(req.category, req.code, req.message, req.severity)
    _store.save(run)
    return {"ok": True, "diagnostics_count": len(run.diagnostics)}


@router.post("/runs/{run_id}/iterations")
async def add_iteration(run_id: str, req: IterationRequest):
    run = _store.load(run_id)
    if not run:
        raise HTTPException(404, f"Run {run_id} not found")
    run.add_iteration(req.prompt, req.changes, req.metrics_before, req.metrics_after)
    _store.save(run)
    return {"ok": True, "iteration": len(run.iterations)}


@router.post("/runs/{run_id}/validation")
async def add_validation(run_id: str, req: ValidationRequest):
    run = _store.load(run_id)
    if not run:
        raise HTTPException(404, f"Run {run_id} not found")
    run.add_validation(req.gate, req.passed, req.metrics, req.thresholds)
    _store.save(run)
    return {"ok": True, "validation_count": len(run.validation)}


@router.get("/runs/{run_id}/pipeline")
async def get_pipeline(run_id: str):
    """8-step research pipeline status (Idea→…→Record)."""
    run = _store.load(run_id)
    if not run:
        raise HTTPException(404, f"Run {run_id} not found")
    return run.derive_pipeline()


class PromoteRequest(BaseModel):
    target: str  # research | backtest | paper | live
    note: str = ""
    force: bool = False


@router.post("/runs/{run_id}/promote")
async def promote_run(run_id: str, req: PromoteRequest):
    """Promote run along research→backtest→paper→live. Live never auto-enabled."""
    run = _store.load(run_id)
    if not run:
        raise HTTPException(404, f"Run {run_id} not found")

    target = req.target.strip().lower()
    ok, msg = run.can_promote_to(target)
    if not ok and not req.force:
        raise HTTPException(400, msg)

    prev = run.promotion
    run.promotion = target
    if target == "backtest":
        run.pipeline_stage = "backtest"
    elif target == "paper":
        run.pipeline_stage = "deploy"
        run.artifact = {**(run.artifact or {}), "deployed": True, "deploy_target": "paper"}
    elif target == "live":
        run.pipeline_stage = "deploy"
        run.artifact = {
            **(run.artifact or {}),
            "deployed": True,
            "deploy_target": "live",
            "live_note": "Promotion recorded only — LIVE_TRADING_ENABLED still required",
        }
    if req.note:
        run.notes = (run.notes + "\n" if run.notes else "") + f"[promote {prev}→{target}] {req.note}"
    _store.save(run)
    _publish_sse(run_id, "promotion", {"from": prev, "to": target})
    return {
        "ok": True,
        "run_id": run_id,
        "from": prev,
        "to": target,
        "forced": bool(req.force and not ok),
        "pipeline": run.derive_pipeline(),
        "warning": None if target != "live" else "Live promotion is a record only; RiskGate + LIVE_TRADING_ENABLED still gate real orders",
    }


class FactorSnapshotRequest(BaseModel):
    ic: dict[str, Any] = {}
    dedupe: dict[str, Any] = {}
    factor_names: list[str] = []
    extra: dict[str, Any] = {}


@router.post("/runs/{run_id}/factor-snapshot")
async def set_factor_snapshot(run_id: str, req: FactorSnapshotRequest):
    """Attach factor analysis snapshot to advance Factor / Validate-F steps."""
    run = _store.load(run_id)
    if not run:
        raise HTTPException(404, f"Run {run_id} not found")
    run.factor_snapshot = {
        "ic": req.ic,
        "dedupe": req.dedupe,
        "factor_names": req.factor_names,
        **req.extra,
        "updated_at": time.time(),
    }
    if "factor" not in run.tags:
        run.tags = list(run.tags) + ["factor"]
    run.pipeline_stage = "validate_factor" if (req.ic or req.dedupe) else "factor"
    _store.save(run)
    return {"ok": True, "pipeline": run.derive_pipeline()}


class AgenticCycleRequest(BaseModel):
    topics: list[str] = []


@router.post("/agentic-cycle")
async def run_agentic_cycle(req: AgenticCycleRequest = AgenticCycleRequest()):
    """Idea → Factor → Eval loop: DiGist signals, deployed params, overfit gates."""
    sys.path.insert(0, str(REPO_ROOT / "eternal-optimizer"))
    from agentic_loop import run_agentic_cycle as _cycle

    topics = req.topics or None
    return _cycle(topics)


_UPLOAD_MAX_BYTES = 50 * 1024 * 1024  # 50 MB
_UPLOAD_ALLOWED_EXT = {".csv", ".json", ".parquet", ".py", ".yaml", ".yml", ".txt", ".md"}


@router.post("/upload")
async def upload_file(request: Request):
    """Upload a data file for research use with size + extension validation.

    Reference: Vibe-Trading /upload pattern — chunk-aware, extension allowlist.
    """
    content_type = request.headers.get("content-type", "")

    if "multipart/form-data" in content_type:
        from fastapi import UploadFile
        form = await request.form()
        upload: UploadFile | None = form.get("file")  # type: ignore[assignment]
        if upload is None:
            raise HTTPException(400, "No file field in multipart form")
        filename = upload.filename or "unnamed"
        body = await upload.read()
    else:
        filename = request.headers.get("x-filename", "unnamed")
        body = await request.body()

    ext = Path(filename).suffix.lower()
    if ext not in _UPLOAD_ALLOWED_EXT:
        raise HTTPException(
            415,
            f"Extension '{ext}' not allowed. Accepted: {sorted(_UPLOAD_ALLOWED_EXT)}",
        )

    if len(body) > _UPLOAD_MAX_BYTES:
        raise HTTPException(
            413,
            f"File too large ({len(body)} bytes). Max: {_UPLOAD_MAX_BYTES} bytes ({_UPLOAD_MAX_BYTES // 1024 // 1024} MB)",
        )

    if len(body) == 0:
        raise HTTPException(400, "Empty file")

    upload_dir = RESEARCH_DIR / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_name = f"{uuid.uuid4().hex[:12]}_{Path(filename).name}"
    dest = upload_dir / safe_name
    dest.write_bytes(body)

    logger.info("Research upload: %s (%d bytes) → %s", filename, len(body), dest)
    return {
        "ok": True,
        "filename": filename,
        "stored_as": safe_name,
        "size_bytes": len(body),
        "path": str(dest.relative_to(REPO_ROOT)),
    }


@router.post("/runs/{run_id}/execute")
async def execute_run(run_id: str, background_tasks: BackgroundTasks):
    """Start a backtest for this research run using existing tqsdk engine."""
    run = _store.load(run_id)
    if not run:
        raise HTTPException(404, f"Run {run_id} not found")
    if run.status == RunStatus.RUNNING:
        raise HTTPException(409, "Run already executing")

    run.status = RunStatus.RUNNING
    _store.save(run)
    background_tasks.add_task(_execute_backtest, run_id)
    return {"ok": True, "status": "running"}


@router.get("/runs/{run_id}/artifact")
async def get_artifact(run_id: str):
    """Export the run's research artifact for KnowLever/AutoOffice consumption."""
    run = _store.load(run_id)
    if not run:
        raise HTTPException(404, f"Run {run_id} not found")
    if run.status != RunStatus.COMPLETED:
        raise HTTPException(400, "Run not completed — cannot export artifact")
    from dataclasses import asdict
    artifact = run.build_artifact()
    return asdict(artifact)


@router.get("/runs/{run_id}/events")
async def stream_run_events(run_id: str, request: Request, events: str | None = None):
    """SSE endpoint — stream real-time events for a research run.

    Event types: status_change, backtest_progress, diagnostic, validation_result, completed.
    Reference: contracts.md §7 — `/sessions/{id}/events` SSE pattern.

    Query params:
        events: comma-separated event types to subscribe to (e.g. ?events=completed,status_change).
                If omitted, all events are delivered.
    """
    run = _store.load(run_id)
    if not run:
        raise HTTPException(404, f"Run {run_id} not found")

    event_filter: set[str] | None = None
    if events:
        event_filter = {e.strip() for e in events.split(",") if e.strip()}

    queue: asyncio.Queue = asyncio.Queue(maxsize=256)
    _sse_subscribers.setdefault(run_id, []).append(queue)

    async def event_generator():
        try:
            yield f"event: connected\ndata: {_json.dumps({'run_id': run_id, 'status': run.status.value, 'filter': list(event_filter) if event_filter else 'all'})}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=30.0)
                    event_type = msg.pop("event", "message")
                    if event_filter and event_type not in event_filter:
                        continue
                    yield f"event: {event_type}\ndata: {_json.dumps(msg, default=str)}\n\n"
                except asyncio.TimeoutError:
                    yield f"event: heartbeat\ndata: {_json.dumps({'ts': time.time()})}\n\n"
        finally:
            subs = _sse_subscribers.get(run_id, [])
            if queue in subs:
                subs.remove(queue)
            if not subs:
                _sse_subscribers.pop(run_id, None)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/runs/{run_id}/artifact/markdown")
async def get_artifact_markdown(run_id: str):
    """Export as Markdown for KnowLever consumption."""
    run = _store.load(run_id)
    if not run:
        raise HTTPException(404, f"Run {run_id} not found")
    md = _render_artifact_markdown(run)
    return StreamingResponse(
        iter([md]), media_type="text/markdown",
        headers={"Content-Disposition": f"inline; filename=research_{run_id}.md"},
    )


async def _execute_backtest(run_id: str) -> None:
    """Background task: run backtest using existing tqsdk engine."""
    run = _store.load(run_id)
    if not run:
        return
    try:
        from backtest.engine import BacktestEngine
        from experiment.validation import MethodValidator

        _publish_sse(run_id, "status_change", {"status": "running"})

        engine = BacktestEngine()
        symbols = run.symbols or ["rb"]
        results = {}

        for i, sym in enumerate(symbols):
            _publish_sse(run_id, "backtest_progress", {
                "symbol": sym, "index": i + 1, "total": len(symbols),
            })
            result = engine.run(
                strategy_name=run.strategy_name,
                symbol=sym,
                timeframe=run.timeframe,
                params=run.config,
            )
            results[sym] = result

        agg_metrics = _aggregate_metrics(results)
        run.backtest_results = {sym: _safe_serialize(r) for sym, r in results.items()}
        run.metrics = agg_metrics

        run.add_diagnostic("runtime", "backtest_complete",
                           f"Backtest completed for {len(symbols)} symbols", severity="info")
        _publish_sse(run_id, "diagnostic", {"code": "backtest_complete", "symbols": len(symbols)})

        validator = MethodValidator()
        for gate_name, gate_fn in [
            ("OOS", lambda: validator.oos_check(results)),
            ("WF", lambda: validator.walk_forward(results)),
        ]:
            try:
                gate_result = gate_fn()
                passed = gate_result.get("passed", False)
                run.add_validation(gate_name, passed,
                                   gate_result.get("metrics", {}), gate_result.get("thresholds", {}))
                _publish_sse(run_id, "validation_result", {
                    "gate": gate_name, "passed": passed,
                })
            except Exception as e:
                run.add_diagnostic("runtime", f"{gate_name}_error", str(e))

        run.status = RunStatus.COMPLETED
        run.artifact = run.build_artifact().__dict__ if hasattr(run.build_artifact(), '__dict__') else {}
        _publish_sse(run_id, "completed", {"status": "completed", "metrics": agg_metrics})

    except ImportError as e:
        run.add_diagnostic("runtime", "import_error", str(e), severity="error")
        run.status = RunStatus.FAILED
        _publish_sse(run_id, "completed", {"status": "failed", "error": str(e)})
    except Exception as e:
        run.add_diagnostic("runtime", "execution_error",
                           f"{type(e).__name__}: {e}", severity="error",
                           traceback=traceback.format_exc())
        run.status = RunStatus.FAILED
        _publish_sse(run_id, "completed", {"status": "failed", "error": str(e)})

    _store.save(run)


def _aggregate_metrics(results: dict) -> dict[str, float]:
    if not results:
        return {}
    all_sharpes = []
    all_returns = []
    all_dds = []
    for r in results.values():
        if isinstance(r, dict):
            if "sharpe" in r:
                all_sharpes.append(r["sharpe"])
            if "total_return" in r:
                all_returns.append(r["total_return"])
            if "max_dd" in r:
                all_dds.append(r["max_dd"])
    return {
        "avg_sharpe": sum(all_sharpes) / len(all_sharpes) if all_sharpes else 0,
        "avg_return": sum(all_returns) / len(all_returns) if all_returns else 0,
        "avg_max_dd": sum(all_dds) / len(all_dds) if all_dds else 0,
        "symbols_tested": len(results),
    }


def _safe_serialize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(v) for v in obj]
    if isinstance(obj, float):
        if obj != obj:
            return None
        return round(obj, 6)
    return obj


def _render_artifact_markdown(run: ResearchRun) -> str:
    lines = [
        f"# Research Artifact: {run.run_id}",
        "",
        f"**Strategy**: {run.strategy_name}",
        f"**Status**: {run.status.value}",
        f"**Symbols**: {', '.join(run.symbols)}",
        f"**Timeframe**: {run.timeframe}",
        f"**Data Range**: {run.data_range}",
        "",
        "## Metrics",
        "",
    ]
    for k, v in run.metrics.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append("## Validation Gates")
    lines.append("")
    for v in run.validation:
        status = "PASS" if v.get("passed") else "FAIL"
        lines.append(f"- **{v.get('gate', '?')}**: {status}")
        for mk, mv in v.get("metrics", {}).items():
            lines.append(f"  - {mk}: {mv}")
    lines.append("")
    lines.append("## Diagnostics")
    lines.append("")
    for d in run.diagnostics:
        lines.append(f"- [{d.get('severity', '?').upper()}] {d.get('category', '?')}/{d.get('code', '?')}: {d.get('message', '')}")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append("```json")
    lines.append(_json.dumps(run.config, indent=2, default=str))
    lines.append("```")
    lines.append("")
    if run.iterations:
        lines.append("## Iterations")
        lines.append("")
        for it in run.iterations:
            lines.append(f"### Iteration {it.get('iteration', '?')}")
            lines.append(f"- Prompt: {it.get('prompt', '')}")
            lines.append(f"- Changes: {it.get('changes', '')}")
            lines.append("")
    return "\n".join(lines)
