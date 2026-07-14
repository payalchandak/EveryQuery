# `generate_tasks/`

Task-label generation stage of the EveryQuery pipeline. Two console scripts
live here, each producing a `TaskQuerySchema`-conformant parquet but with
different row distributions:

- **`EQ_generate_training_tasks`** — scattered shape: `num_queries` independent
    `(code, duration_days)` queries × `num_contexts_per_query` patient contexts,
    for pretraining. Runs a single-process 5-stage pipeline (see below).
- **`EQ_generate_evaluation_tasks`** — dense-grid shape: sampled prediction
    times × `(codes × durations)`, for feeding `EQ_predict` → `EQ_evaluate`.

## What lives here

- **`sample_tasks.py`** — the 5-stage training-task sampler. Registered as
    `EQ_generate_training_tasks` and runnable as
    `python -m every_query.generate_tasks.sample_tasks`. One console invocation runs
    the **whole** pipeline in one process on one node — Stages 0–3 sequentially in the
    driver, then Stage 4 fans out one labeling worker per shard via a
    `ProcessPoolExecutor`:

    | Stage | Where             | What                                                                                                                          |
    | ----- | ----------------- | ----------------------------------------------------------------------------------------------------------------------------- |
    | 0     | driver            | Scan shards once, dedup to distinct `(subject_id, time)`, dense-rank → prediction-time map; filter eligible subjects. Cached. |
    | 1     | driver            | Sample `num_queries` `(code, duration_days)` queries (uniform code × uniform/log-uniform duration).                           |
    | 2     | driver            | Sample `N = num_queries × num_contexts_per_query` `(subject_idx, prediction_time_index)` contexts, iid with replacement.      |
    | 3     | driver            | Resolve each context's `prediction_time` against the Stage 0 map, zip with queries, write the partitioned index.              |
    | 4     | per-shard workers | Label each index shard independently via `join_asof`; write the final per-shard parquet atomically.                           |

    See [`redesign-spec.md`](redesign-spec.md) for the full stage contract, invariants,
    and determinism model — this README does not restate them.

- **`sample_evaluation_tasks.py`** — dense-grid evaluation-task generator.
    Samples `K` prediction times per subject, cross-joins with the full
    `(codes × durations)` grid the caller specifies, labels via the same
    `evaluate_index_df` primitive from `sample_tasks.py`. Registered as
    `EQ_generate_evaluation_tasks` and runnable as
    `python -m every_query.generate_tasks.sample_evaluation_tasks`.

- **`sample_task_tracking_pairs.py`** — compact per-task AUROC-tracking sampler.
    Consumes `EQ_generate_evaluation_tasks`'s labeled output (typically `split=tuning`)
    and, per `(query, duration_days)` task, samples exactly one positive + one negative
    row into a single small parquet (2 rows/task). Registered as
    `EQ_sample_task_tracking_pairs` and runnable as
    `python -m every_query.generate_tasks.sample_task_tracking_pairs`. Feeds
    the optional in-training `TaskAurocTrackingCallback` — see
    `trainer.callbacks.task_auroc_tracking` in `train/configs/config.yaml`.

- **`configs/sample_training_tasks_config.yaml`** / **`configs/sample_evaluation_tasks_config.yaml`**
    / **`configs/sample_task_tracking_pairs_config.yaml`**
    — shipped Hydra configs. Path roots (`data_dir`, `out_dir`, and `query_codes` — pass the
    metadata root dir to load the full universe) are **required Hydra args** — no `.env`/env-var
    fallback (see [#235](https://github.com/payalchandak/EveryQuery/issues/235)). Pass them as
    shell-expanded vars (`data_dir=$TOKENIZED_EVENTS_DIR out_dir=$TRAINING_TASKS_DIR query_codes=$TENSORIZED_COHORT_DIR`);
    everything else is a Hydra override.

## Pipeline position

```
preprocessing/     →  generate_tasks/                   →  train/      →  predict/   →  evaluate/
EQ_process_data       EQ_generate_training_tasks           EQ_train       EQ_predict     EQ_evaluate
                      EQ_generate_evaluation_tasks ────────────────────►  (inference input)
                      EQ_generate_evaluation_tasks(split=tuning)
                        → EQ_sample_task_tracking_pairs ──►  (in-training tracking input)
```

Both endpoints consume:

1. Event shards at `$TOKENIZED_EVENTS_DIR/data/{split}/*.parquet` (from
    [`preprocessing/`](../preprocessing/)).
2. The query-code universe — pass `query_codes=$TENSORIZED_COHORT_DIR` (a metadata root dir resolves to
    `$TENSORIZED_COHORT_DIR/metadata/codes.parquet`), or override with an explicit list / file path. Same
    `query_codes` knob for both training and evaluation.

**Training-task outputs** land at `$TRAINING_TASKS_DIR/{split}/{shard}.parquet` — the final
dataset is the union of the per-shard files, and that directory holds *nothing else* at rest.
All intermediates (the Stage 0 prediction-time map, Stage 3 index) live in the **sibling**
`*_artifacts` root (`{parent}/{name}_artifacts` of `$TRAINING_TASKS_DIR`, no env var of its
own), so the two trees never nest and cleanup is a single `rm -rf` of the artifacts dir.

**Evaluation-task outputs** land at `$EVAL_TASKS_DIR/eval/{split}/*.parquet`. The separate `eval/`
subdirectory keeps the two row distributions from colliding in one directory.

## Running it

### Pretraining tasks — whole pipeline, one command

```bash
# Stages 0–3 run inline in the driver; Stage 4 fans out across shards in-process.
EQ_generate_training_tasks \
	split=train \
	num_queries=1024 \
	num_contexts_per_query=1

# Restrict sampled training queries to a YAML code list (flat list or `{codes: [...]}`):
EQ_generate_training_tasks \
	split=train \
	query_codes=/path/to/train_query_codes.yaml
```

There is **no** `task_shard`/`input_shard` axis and no Hydra `-m` sweep — the single driver
process samples globally and fans Stage-4 labeling out itself. Key knobs (full list in
`configs/sample_training_tasks_config.yaml`):

- `num_queries`, `num_contexts_per_query` — sampling budget; output is
    `num_queries × num_contexts_per_query` rows.
- `min_prediction_times_per_subject` (default 50) — minimum prior **prediction times**
    (distinct `(subject_id, time)` rows, not events) before a prediction time is eligible.
- `min_duration`, `max_duration`, `duration_distribution` (`uniform | log-uniform`) — the
    Stage 1 duration draw; `duration_days` is a float (no day-rounding).
- `query_codes` — **required**: `query_codes=$TENSORIZED_COHORT_DIR` (a metadata root dir) loads the full
    vocabulary from `$TENSORIZED_COHORT_DIR/metadata/codes.parquet`; or pass an inline Hydra list
    (`query_codes=[HR,TEMP]`) or a YAML/parquet path.
- `max_workers` — optional cap on the Stage 4 pool; `null` uses cores-on-node, an int caps that
    downward only. Set it when a run OOMs.
- `seed`, `overwrite`, and optional path overrides `data_dir` / `out_dir`.

### Evaluation tasks — dense grid, swept per shard

```bash
EQ_generate_evaluation_tasks -m \
	input_shard=0,1,2,... split=held_out
```

### Task-tracking pairs — one compact parquet for in-training AUROC monitoring

```bash
# First, generate dense tuning-split labels the same way as above (note split=tuning):
EQ_generate_evaluation_tasks -m \
	input_shard=0,1,2,... split=tuning \
	query_codes=$TENSORIZED_COHORT_DIR durations=[30,90,180,365,731]

# Then sample one pos/neg pair per task from that output:
EQ_sample_task_tracking_pairs \
	eval_labels_dir=$EVAL_TASKS_DIR/eval out_dir=$TASK_TRACKING_DIR split=tuning
```

This is a one-time offline step; the resulting `$TASK_TRACKING_DIR/tuning/0.parquet` is
small (2 rows per task) and fixed for the duration of a training run — point
`trainer.callbacks.task_auroc_tracking.config.task_labels_dir` at `$TASK_TRACKING_DIR` to
have `EQ_train` log a macro-averaged, per-task-sampled AUROC (`tuning/occurs_auroc_macro_sampled`)
every validation pass, without paying the cost of scoring the full tuning split.

## Determinism & restartability

All training draws derive from `seed` via `utils.seeds.derive_seed`, splitting the **query
axis** (`derive_seed(seed, "queries")`) and **context axis** (`derive_seed(seed, "contexts")`)
so they reproduce independently for a fixed seed. Stage 4 writes each shard via a temp file +
`os.replace`, so a present `{shard}.parquet` is always complete; reruns skip finished shards
(idempotent), and `overwrite=true` forces relabeling. The evaluation generator has only a
prediction-time axis (codes and durations are caller-specified), so its seed derives on
`(seed, split, input_shard)` and each worker writes idempotently. The task-tracking
sampler derives its per-(task, class) sample key on `(seed, "task_tracking_pairs", split)`
and writes its single output file atomically the same way; it skips work if the output
already exists unless `overwrite=true`.

## Related

- Architecture reference for the training sampler: [`redesign-spec.md`](redesign-spec.md).
- Parent refactor umbrella: [#54](https://github.com/payalchandak/EveryQuery/issues/54).
- Phase 2.1 — cross-stage `TaskQuerySchema`: [#80](https://github.com/payalchandak/EveryQuery/issues/80) (closed, merged in #96).
- Phase 2.2 — `EQ_predict` (consumes `sample_evaluation_tasks` output):
    [#81](https://github.com/payalchandak/EveryQuery/issues/81) (closed, merged in #99).
