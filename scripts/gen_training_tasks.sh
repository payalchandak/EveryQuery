#!/bin/bash
#SBATCH --job-name=eq-gen-train
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-291%20
#SBATCH --output=logs/gen_train_%A_%a.out
#SBATCH --mail-user=gbk2114@cumc.columbia.edu
#SBATCH --mail-type=END,FAIL
#
# Parallelizes the training-tasks sweep across input shards via a SLURM job array.
# Each array task runs its 16 task_shards as a Hydra multirun.
#   input_shard=0..291 (array tasks), task_shard=0..15 (per-task multirun)
#
# Usage:
#   sbatch scripts/gen_training_tasks.sh

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

set -a
source .env
set +a

mkdir -p logs

uv run EQ_generate_training_tasks -m \
    split=train \
    input_shard=${SLURM_ARRAY_TASK_ID} \
    task_shard='range(0,16)'
