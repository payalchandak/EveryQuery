#!/bin/bash
#SBATCH --job-name=eq_upload
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-user=gbk2114@cumc.columbia.edu
#SBATCH --mail-type=END,FAIL

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00

set -euo pipefail
export PYTHONNOUSERSITE=1

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <runs-file> [--dry-run]" >&2
    echo "  <runs-file>  text file with one run directory per line" >&2
    echo "  Bucket + path layout come from scripts/sync_config.yaml (outputs mapping)" >&2
    exit 2
fi

RUNS_FILE="$1"
shift

mkdir -p logs
echo "Starting upload job on $(hostname) at $(date)"
cd "${SLURM_SUBMIT_DIR:-$PWD}"

set -a
# shellcheck source=/dev/null
. ./.env
set +a

uv sync --locked

uv run python src/every_query/upload_models.py \
    --runs-file "${RUNS_FILE}" \
    "$@"

echo "Finished at $(date)"
