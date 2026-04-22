#!/usr/bin/env bash
# Launch EQ_evaluate across all 10 sweep tokenizations.
#
# Reads the PredictionSchema parquets written by launch_predict.sh and writes
# per-(query, duration_days) metrics parquets (one row per task).
#
# Usage:
#   bash scripts/eval_sweep/launch_evaluate.sh                    # all 10 toks
#   TOKS="msc_1000 msc_10000" bash scripts/eval_sweep/...         # subset
#   SWEEP_SPLIT=held_out bash scripts/eval_sweep/...              # override split
#
# Run after launch_predict.sh has finished and predictions parquets are on
# disk. On success prints a colon-joined jobid list to stdout.
set -euo pipefail

SWEEP_TASK_ROOT="${SWEEP_TASK_ROOT:-/groups/mm6677_gp/gbk2114/eic_stuff/tokenization_sweep_tasks}"
SWEEP_SPLIT="${SWEEP_SPLIT:-tuning}"
export SWEEP_TASK_ROOT

DEFAULT_TOKS=(
    meps_10
    meps_32
    meps_100
    meps_316
    meps_1000
    msc_100
    msc_316
    msc_1000
    msc_3162
    msc_10000
)
if [[ -n "${TOKS:-}" ]]; then
    # shellcheck disable=SC2206
    TOK_LIST=($TOKS)
else
    TOK_LIST=("${DEFAULT_TOKS[@]}")
fi

mkdir -p logs
JOBIDS=()
echo "Launching evaluate for ${#TOK_LIST[@]} tokenizations (split=$SWEEP_SPLIT)" >&2

for tok in "${TOK_LIST[@]}"; do
    predictions_parquet="$SWEEP_TASK_ROOT/$tok/predictions/$SWEEP_SPLIT.parquet"
    metrics_parquet="$SWEEP_TASK_ROOT/$tok/metrics/$SWEEP_SPLIT.parquet"

    if [[ ! -f "$predictions_parquet" ]]; then
        echo "-- $tok: SKIP (predictions parquet missing: $predictions_parquet)" >&2
        continue
    fi

    echo "-- $tok (predictions=$predictions_parquet) --" >&2

    jobid=$(sbatch --parsable \
        --job-name="eq_eval_evaluate_${tok}" \
        --export=ALL,SWEEP_TOK="$tok",SWEEP_PREDICTIONS_PARQUET="$predictions_parquet",SWEEP_METRICS_PARQUET="$metrics_parquet" \
        scripts/eval_sweep/evaluate_worker.sh)
    echo "   sbatch -> $jobid" >&2
    JOBIDS+=("$jobid")
done

IFS=: ; echo "${JOBIDS[*]}"
