# `generate_tasks/`

Task-label generation stage of the EveryQuery pipeline. Two console scripts
live here, each producing a `TaskQuerySchema`-conformant parquet but with
different row distributions:

- **`EQ_generate_training_tasks`** — scattered shape: `N` independent
    `(query, duration_days)` tasks × `M` contexts, for pretraining.
- **`EQ_generate_evaluation_tasks`** — dense-grid shape: sampled prediction
    times × `(codes × durations)`, for feeding `EQ_predict` → `EQ_evaluate`.

## What lives here

- **`sample_tasks.py`** — sampling-first pretraining-task generator. Draws `N`
    `(code, duration_days)` tasks from a uniform × log-uniform distribution, `N × M`
    `(subject_id, prediction_time)` contexts iid with replacement, zips them, and
    writes a long-format labels parquet per worker via a single-pass `join_asof`.
    Registered as `EQ_generate_training_tasks` and runnable as
    `python -m every_query.generate_tasks.sample_tasks`.
- **`sample_evaluation_tasks.py`** — dense-grid evaluation-task generator.
    Samples `K` prediction times per subject, cross-joins with the full
    `(codes × durations)` grid the caller specifies, labels via the same
    `evaluate_index_df` primitive from `sample_tasks.py`. Registered as
    `EQ_generate_evaluation_tasks` and runnable as
    `python -m every_query.generate_tasks.sample_evaluation_tasks`.
- **`configs/sample_tasks_config.yaml`** / **`configs/sample_evaluation_tasks_config.yaml`**
    — shipped Hydra configs. Path fallbacks resolve via the repo's `.env`-based
    env-var convention (`$INTERMEDIATE`, `$PROCESSED`, `$TASK_DIR`); everything
    else is a Hydra override.

## Pipeline position

```
preprocessing/     →  generate_tasks/                   →  train/      →  predict/   →  evaluate/
EQ_process_data       EQ_generate_training_tasks           EQ_train       EQ_predict     EQ_evaluate
                      EQ_generate_evaluation_tasks ────────────────────►  (inference input)
```

Both endpoints consume:

1. Event shards at `$INTERMEDIATE/data/{split}/*.parquet` (from
    [`preprocessing/`](../preprocessing/)).
2. The query-code universe at `$PROCESSED/metadata/codes.parquet` — or a CLI override:
    `query_codes=...` for training, `codes=...` for evaluation.

Training-task outputs land at `$TASK_DIR/{split}/*.parquet`; evaluation-task
outputs land at `$TASK_DIR/eval/{split}/*.parquet`. The separate `eval/`
subdirectory keeps the two row distributions from colliding in one directory.

## Sweeping across shards

```
# Pretraining tasks (random tasks × random contexts):
python -m every_query.generate_tasks.sample_tasks -m \
    input_shard=0,1,2,... task_shard=range(0,K)

# Restrict sampled training queries to a YAML code list:
python -m every_query.generate_tasks.sample_tasks -m \
    input_shard=0,1,2,... task_shard=range(0,K) \
    query_codes=/path/to/train_query_codes.yaml

# `train_query_codes.yaml` may be either a flat list or `{codes: [...]}`.

# Evaluation tasks (dense grid over the held-out cohort):
python -m every_query.generate_tasks.sample_evaluation_tasks -m \
    input_shard=0,1,2,... split=held_out
```

The pretraining generator's seed derivation (`utils.seeds.derive_seed`)
separates task-axis and context-axis randomness so fixing `task_shard` across
`input_shard` values evaluates the *same* tasks on *different* patients; the
evaluation generator only has a prediction-time axis (codes and durations are
caller-specified), so its seed derives on `(seed, split, input_shard)`. Each
worker writes idempotently; re-running is a no-op.

## Related

- Parent refactor umbrella: [#54](https://github.com/payalchandak/EveryQuery/issues/54).
- Phase 2.1 — cross-stage `TaskQuerySchema`: [#80](https://github.com/payalchandak/EveryQuery/issues/80) (closed, merged in #96).
- Phase 2.2 — `EQ_predict` (consumes `sample_evaluation_tasks` output):
    [#81](https://github.com/payalchandak/EveryQuery/issues/81) (closed, merged in #99).
