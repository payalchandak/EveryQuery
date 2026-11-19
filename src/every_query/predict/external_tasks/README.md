# `predict/external_tasks/`

Utilities for running EQ predictions against tasks that aren't in the model's native
single-code vocabulary — i.e. **ACES composite tasks** and disjunction-style aggregations.

Groups code previously scattered across two top-level submodules (`aces_to_eq/` and
`process_composite/`) under one conceptual umbrella: *predict a task outside of EQ's
vocabulary*.

## What lives here

- **`aces_to_eq.py`** — converts an [ACES](https://github.com/justin13601/ACES) task parquet
    into EQ's task-df shape by matching `(subject_id, prediction_time)` tuples and copying the
    ACES `boolean_value` as the target label.
- **`process_composite.py`** — aggregates per-code prediction probabilities into composite
    probabilities for disjunction tasks (e.g. "readmission = any of {code1, code2, code3}"),
    with `max` / `or` / `sum` aggregation modes.
- **`get_per_code_from_composite.py`** — the inverse: given a composite-task eval dataframe,
    extract per-code tasks and evaluate the trained model on each individually.
- **`configs/`** — the Hydra configs for all three scripts.
- **`task_configs/`**: the ACES task definitions themselves, one YAML per task with the clinical
    question in a header comment. These are the *inputs* to an ACES extraction, whose output
    parquets `aces_to_eq.py` then converts. The file stem is the `task_name` that
    `configs/aces_to_eq.yaml` interpolates into `${ACES_SHARDS_DIR}/${task_name}/held_out`.

## Entry points

Currently invocable only via `python -m every_query.predict.external_tasks.<script>`.
Registering as `EQ_aces_to_eq` / `EQ_process_composite` console scripts is tracked in
[#62](https://github.com/payalchandak/EveryQuery/issues/62).
