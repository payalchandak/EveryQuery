#!/bin/bash
# sbatch worker: run EQ_evaluate for one tokenization.
#
# Inputs via env:
#   SWEEP_TOK                    tokenization name (e.g. msc_1000)
#   SWEEP_PREDICTIONS_PARQUET    input PredictionSchema parquet from EQ_predict
#   SWEEP_METRICS_PARQUET        output per-(query, duration_days) metrics parquet
#
# Output: a single metrics parquet at $SWEEP_METRICS_PARQUET with one row per
# (query, duration_days) and columns n_rows, n_occurs_labeled, n_positive,
# occurs_auroc, censor_auroc.

#SBATCH --job-name=eq_eval_evaluate
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-user=gbk2114@cumc.columbia.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail
export PYTHONNOUSERSITE=1

mkdir -p logs
echo "Starting evaluate on $(hostname) at $(date)"
cd "${SLURM_SUBMIT_DIR:-$PWD}"

set -a
# shellcheck source=/dev/null
. ./.env
set +a

: "${SWEEP_TOK:?SWEEP_TOK env var required}"
: "${SWEEP_PREDICTIONS_PARQUET:?SWEEP_PREDICTIONS_PARQUET env var required}"
: "${SWEEP_METRICS_PARQUET:?SWEEP_METRICS_PARQUET env var required}"

if [[ ! -f "$SWEEP_PREDICTIONS_PARQUET" ]]; then
    echo "ERROR: predictions parquet missing: $SWEEP_PREDICTIONS_PARQUET" >&2
    exit 1
fi
mkdir -p "$(dirname "$SWEEP_METRICS_PARQUET")"

echo "Tok=$SWEEP_TOK"
echo "Predictions=$SWEEP_PREDICTIONS_PARQUET"
echo "Metrics out=$SWEEP_METRICS_PARQUET"

uv sync --locked
export HYDRA_FULL_ERROR=1

srun uv run EQ_evaluate \
    predictions_parquet="$SWEEP_PREDICTIONS_PARQUET" \
    metrics_parquet="$SWEEP_METRICS_PARQUET"

echo "Finished evaluate for $SWEEP_TOK at $(date)"
