#!/bin/bash
#SBATCH --job-name=eq_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-user=gbk2114@cumc.columbia.edu
#SBATCH --mail-type=END,FAIL

#SBATCH --nodes=1               # Explicitly define 1 node
#SBATCH --ntasks-per-node=1     # Must match Lightning devices count for DDP
# #SBATCH --gres=gpu:2            # Total GPUs per node
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=20
# #SBATCH --cpus-per-task=2
#SBATCH --mem=512G
#SBATCH --time=100:00:00

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
# Path to uv env
UVENV="$HOME/eq_stuff/eq"

# Debug — should output eq_uv Python
echo "Using python: $UVENV/bin/python"
$UVENV/bin/python -c "import sys; print('Executable:', sys.executable)"

cd ~/EveryQuery

# Load .env variables
set -a
. ./.env
set +a

export HYDRA_FULL_ERROR=1

srun $UVENV/bin/python src/every_query/train.py

echo "Finished at $(date)"
