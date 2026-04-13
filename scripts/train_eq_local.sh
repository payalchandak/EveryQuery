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

: "${OMP_NUM_THREADS:=2}"
: "${MKL_NUM_THREADS:=2}"
: "${OPENBLAS_NUM_THREADS:=2}"
: "${NUMEXPR_NUM_THREADS:=2}"
export OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS
export POLARS_MAX_THREADS=32

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

uv sync --locked

echo "Using python: $(uv run which python)"
uv run python -c "import sys; print('Executable:', sys.executable)"

export HYDRA_FULL_ERROR=1

uv run numactl --cpunodebind=0 --membind=0 python src/every_query/train.py "$@"

echo "Finished at $(date)"
