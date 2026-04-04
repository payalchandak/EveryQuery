#!/bin/bash
#SBATCH --job-name=eq_sweep
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-user=gbk2114@cumc.columbia.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=256G
#SBATCH --time=300:00:00

# This script is submitted by launch_sweep.sh — not called directly.
# Each job runs exactly 1 W&B sweep trial then exits.

set -euo pipefail

SWEEP_ID="$1"

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p logs
echo "Sweep agent on $(hostname) at $(date)"

UVENV="$HOME/eq_stuff/eq"

cd ~/EveryQuery

set -a
. ./.env
set +a

export HYDRA_FULL_ERROR=1

srun $UVENV/bin/python -m wandb agent --count 1 "$SWEEP_ID"

echo "Sweep agent finished at $(date)"
