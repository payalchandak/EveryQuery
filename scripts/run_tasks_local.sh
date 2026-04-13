#!/bin/bash
set -euo pipefail
export PYTHONNOUSERSITE=1

mkdir -p logs
echo "Starting job on $(hostname) at $(date)"

cd "$(git rev-parse --show-toplevel)"

set -a
# shellcheck source=/dev/null
. ./.env
set +a


export POLARS_MAX_THREADS=32
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

uv sync --locked

echo "Using python: $(uv run which python)"

export HYDRA_FULL_ERROR=1

uv run taskset -c 0-55 python src/every_query/tasks.py "$@"

echo "Finished at $(date)"
