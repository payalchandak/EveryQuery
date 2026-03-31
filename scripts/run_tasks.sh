#!/bin/bash
#SBATCH --job-name=eq-tasks
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/tasks_%A_%a.out
#
# Usage:
#   # First, count total shards:
#   N=$(python -c "
#   import os
#   from dotenv import load_dotenv
#   from meds import held_out_split, train_split, tuning_split
#   load_dotenv()
#   read_dir = os.environ['INTERMEDIATE']
#   n = 0
#   for split in [train_split, tuning_split, held_out_split]:
#       for f in os.listdir(f'{read_dir}/data/{split}'):
#           if f.endswith('.parquet'): n += 1
#   print(n - 1)
#   ")
#
#   # Then submit:
#   sbatch --array=0-$N scripts/run_tasks.sh

uv run python src/every_query/tasks.py --shard-index "$SLURM_ARRAY_TASK_ID"
