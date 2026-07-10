#!/usr/bin/env bash
# Create an isolated conda-forge ML environment (tq-ml) with a single OpenMP runtime.
# Do NOT import lightgbm in trading/API/gateway/strategy-worker processes.
#
# Usage:
#   ./Start/setup-ml-env.sh
#
# After creation, point the ML training worker at this interpreter:
#   ML_PYTHON_BIN=~/miniforge3/envs/tq-ml/bin/python
set -euo pipefail

ENV_NAME="tq-ml"
CHANNELS=(-c conda-forge)
PACKAGES=(
    python=3.12
    pytorch
    lightgbm
    xgboost
    scikit-learn
    pandas
    pyarrow
)

resolve_conda() {
    if command -v mamba >/dev/null 2>&1; then
        echo "mamba"
        return 0
    fi
    if command -v conda >/dev/null 2>&1; then
        echo "conda"
        return 0
    fi
    return 1
}

CONDA_CMD="$(resolve_conda)" || {
    echo "ERROR: neither mamba nor conda found in PATH." >&2
    echo "Install Miniforge/Mambaforge from https://github.com/conda-forge/miniforge" >&2
    exit 1
}

echo "Using $CONDA_CMD to manage environment '$ENV_NAME'"

if "$CONDA_CMD" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Environment '$ENV_NAME' already exists."
    echo "To recreate: $CONDA_CMD env remove -n $ENV_NAME && $0"
    exit 0
fi

echo "Creating '$ENV_NAME' (conda-forge strict, single OpenMP stack)..."
"$CONDA_CMD" create -y -n "$ENV_NAME" "${CHANNELS[@]}" --strict-channel-priority "${PACKAGES[@]}"

CONDA_PREFIX="$("$CONDA_CMD" info --base)"
ML_PYTHON_BIN="${CONDA_PREFIX}/envs/${ENV_NAME}/bin/python"

echo ""
echo "Environment '$ENV_NAME' created successfully."
echo ""
echo "Usage — point the ML training worker at this interpreter:"
echo "  export ML_PYTHON_BIN=${ML_PYTHON_BIN}"
echo ""
echo "Verify:"
echo "  ${ML_PYTHON_BIN} -c \"import lightgbm, torch; print('ok')\""
