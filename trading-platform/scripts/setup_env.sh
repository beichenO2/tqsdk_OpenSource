#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating venv at $VENV_DIR ..."
  uv venv "$VENV_DIR" --python 3.11
fi

source "$VENV_DIR/bin/activate"

echo "Installing core dependencies ..."
uv pip install -q \
  tqsdk fastapi 'uvicorn[standard]' pydantic pydantic-settings \
  sqlalchemy sqlmodel httpx websockets python-dotenv loguru tenacity orjson \
  polars pyarrow duckdb alembic asyncpg gymnasium optuna \
  cryptography keyring pytest pytest-asyncio

echo ""
echo "Verifying tqsdk ..."
python3 -c "import tqsdk; print('  tqsdk', tqsdk.__version__, '✓')"

echo ""
echo "venv ready: source $VENV_DIR/bin/activate"
