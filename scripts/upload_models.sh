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

mkdir -p logs
echo "Starting upload job on $(hostname) at $(date)"
cd "${SLURM_SUBMIT_DIR:-$PWD}"

set -a
. ./.env
set +a

uv sync --locked

uv run python src/every_query/upload_models.py "$@"

echo "Finished at $(date)"
