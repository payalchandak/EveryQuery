# shellcheck shell=bash
# EveryQuery path/config vars.
#
# Copy this file and fill in values for your machine, then `source` it before running
# (and from your SLURM scripts) so the vars below expand into the Hydra args the CLIs take:
#     cp env.example.sh env.sh   # edit env.sh for your machine
#     source env.sh
#     EQ_generate_training_tasks data_dir=$TOKENIZED_EVENTS_DIR out_dir=$TRAINING_TASKS_DIR ...
#
# Nothing in the Python package reads these directly (no dotenv) — they are plain shell
# vars expanded into `key=value` Hydra overrides. `source`-ing one file is all that's
# required to move the project to a new machine.

# --- Paths -------------------------------------------------------------------

# Root of the raw MEDS cohort, read by preprocessing via `input_dir=`. (Preprocessing also sets
# RAW_MEDS_DIR internally for its subprocess — there is no user-facing RAW env var.)
export DATA_DIR="/path/to/MIMIC_MEDS/MEDS_cohort"

# Preprocessing writes both dirs below: TOKENIZED_EVENTS_DIR (intermediate event shards the
# samplers read) and TENSORIZED_COHORT_DIR (the final tensorized cohort, which also holds the
# query-code universe at metadata/codes.parquet).
export TOKENIZED_EVENTS_DIR="${DATA_DIR}/intermediate"        # data_dir= for the samplers
export TENSORIZED_COHORT_DIR="${DATA_DIR}/processed"          # output_dir= (preprocess); tensorized_cohort_dir= (train); query_codes= (samplers)
export EVAL_TASKS_DIR="/path/to/eq_stuff/tasks"              # out_dir= for evaluation tasks
export TRAINING_TASKS_DIR="/path/to/eq_stuff/training_tasks" # out_dir= for training tasks
export TRAINING_OUTPUT_DIR="/path/to/EveryQuery/results"     # pass as EQ_train output_dir= base; Hydra appends <date>/<time>

# Only for the `aces_to_eq` conversion pipeline (predict/external_tasks), read via
# ${oc.env:ACES_SHARDS_DIR}. Leave unset/commented otherwise.
# export ACES_SHARDS_DIR="/path/to/eic_stuff/make_index_dfs/task_configs"

# --- Weights & Biases (read natively by wandb) -------------------------------
export WANDB_ENTITY="your-wandb-entity"
export WANDB_PROJECT="EveryQuery"
