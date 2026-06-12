"""Repo-root shim next to the installable `core` package (see pyproject).

Prefer `from core....` against an installed `trading-core` distribution.
Flat `enums.py` / `type_aliases.py` here are legacy helpers for local scripts only.
"""

__all__: list[str] = []
