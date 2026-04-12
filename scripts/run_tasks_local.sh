#!/bin/bash
set -euo pipefail
export PYTHONNOUSERSITE=1

mkdir -p logs
echo "Starting job on $(hostname) at $(date)"

cd "$(git rev-parse --show-toplevel)"

set -a
. ./.env
set +a

: "${OMP_NUM_THREADS:=2}"
: "${MKL_NUM_THREADS:=2}"
: "${OPENBLAS_NUM_THREADS:=2}"
: "${NUMEXPR_NUM_THREADS:=2}"
export OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS

uv sync --locked

echo "Using python: $(uv run which python)"

export HYDRA_FULL_ERROR=1

uv run python src/every_query/tasks.py "$@"

echo "Finished at $(date)"
