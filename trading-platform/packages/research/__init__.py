"""Research workbench package — facade over experiment.research_run.

This package is 4号位's main battlefield. It re-exports the core research
run types and adds MCP/SSE integration helpers.

Core logic: packages/experiment/research_run.py
API layer:  apps/api/app/routers/research.py (REST + SSE)
MCP layer:  apps/api/app/routers/mcp.py (tool protocol)
"""

from experiment.research_run import ResearchRun, RunStatus, RunStore

__all__ = ["ResearchRun", "RunStatus", "RunStore"]
