#!/bin/bash
#SBATCH --job-name=eq_fast
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-user=gbk2114@cumc.columbia.edu
#SBATCH --mail-type=END,FAIL

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=00:15:00

echo "Allocated GPUs:"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

set -euo pipefail
export PYTHONNOUSERSITE=1

mkdir -p logs
echo "Starting job on $(hostname) at $(date)"
echo "SLURM CPU PER TASK: $SLURM_CPUS_PER_TASK"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

set -a
# shellcheck source=/dev/null
. ./.env
set +a

uv sync --locked

echo "Using python: $(uv run which python)"
uv run python -c "import sys; print('Executable:', sys.executable)"

export HYDRA_FULL_ERROR=1

srun uv run python src/every_query/train.py --config-name=fast_config "$@"

echo "Finished at $(date)"
