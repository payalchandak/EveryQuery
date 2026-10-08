# `predict/`

Inference stage of the EveryQuery pipeline — everything that consumes a trained model
checkpoint and produces per-`(subject_id, prediction_time, query, duration_days)`
probabilities.

## Layout

```python
>>> from pretty_print_directory import PrintConfig
>>> print_directory("src/every_query/predict", config=PrintConfig(ignore_regex="__pycache__"))
├── README.md
├── __init__.py
├── configs
│   ├── predict.yaml
│   └── predict_multitask.yaml
├── external_tasks
│   ├── README.md
│   ├── __init__.py
│   ├── aces_to_eq.py
│   ├── configs
│   │   ├── aces_to_eq.yaml
│   │   ├── get_per_code_from_composite_config.yaml
│   │   └── process_composite_config.yaml
│   ├── get_per_code_from_composite.py
│   └── process_composite.py
├── predict.py
├── predict_multitask.py
└── schema.py

```

Key files:

- `predict.py` — `EQ_predict` (inference-only Hydra main).
- `predict_multitask.py` — `EQ_predict_multitask`: scores the
    `EQ_generate_evaluation_query_sequences` grid with a `ConditionalMultitaskLightningModule`
    checkpoint. Per grid row, `queries[:-1]` / `answers[:-1]` are the teacher-forced conditioning
    pairs and the final query is scored target-only at its last window (no all-vocabulary logits,
    packed labels, manifest or sidecar), emitting one row per grid row (`subject_id`,
    `prediction_time`, the window lists incl. `start_durations` / `start_events`, `target_code`,
    `label`, `prob`). This is the only inference path that consumes active window starts. A
    checkpoint trained with `ontology_dir` scores ancestor queries too: the grid's ancestor names
    resolve to the ontology's `[V, V_ext)` indices, and a grid whose provenance sidecar records a
    *different* closure than the checkpoint's ontology is refused before scoring. The cohort on the
    inference machine must match the checkpoint's by width *and*, when the checkpoint recorded it,
    by vocabulary fingerprint (`cohort_vocab_fingerprint`), and the evaluation adapter requires the
    ontology's leaves to be that cohort's `codes.parquet` rows code for code. Inference
    is `Trainer.predict` over a `ConditionalMultitaskDataModule` built from the checkpoint's cohort
    settings with only the label root swapped for the grid (#30), on exactly one device in exactly
    one process (`device=` picks the accelerator; a multi-device trainer or a `torchrun` / `srun   --ntasks>1` launch is refused so rows stay aligned with the grid, and the collated labels and
    scored codes are re-checked row by row against the grid before writing).
- `schema.py` — `PredictionSchema` (`TaskQuerySchema` + `censor_prob` + `occurs_prob`).
- `configs/predict_multitask.yaml` — the same required trio as `predict.yaml`
    (`model_run_dir`, `tasks_dir`, `output_parquet`) for a multitask run; `tasks_dir` is the
    QuerySeq grid's `eval/` root. Optional: `ckpt_name`, `split`, `overwrite`, `batch_size`,
    `num_workers`, `device` (`null` | `cpu` | `cuda` | `cuda:N` | `mps`, always one device),
    `precision` (`bf16-mixed`, the training / sibling-CLI precision; `32-true` for fp32),
    `enable_progress_bar`.
- `configs/predict.yaml` — required: `model_run_dir`, `tasks_dir`, `output_parquet`; optional: `ckpt_name`, `split` (`held_out` | `tuning`), `overwrite` (default `false` — refuses to clobber an existing `output_parquet`; pass `overwrite=true` to replace).
- `external_tasks/` — convert + aggregate tasks outside EQ's native vocabulary (`aces_to_eq.py`, `process_composite.py`, `get_per_code_from_composite.py`).

## Pipeline position

```
generate_tasks/  +  train/ best_model.ckpt
       │                     │
       ▼                     ▼
     tasks_dir/    ──►  predict/  ──►  predictions.parquet  ──►  evaluate/
     *.parquet                         (PredictionSchema)
     (TaskQuerySchema)
```

`EQ_predict` takes a directory of `TaskQuerySchema`-conformant parquet files — rows of
`(subject_id, prediction_time, query, duration_days)` plus optional inherited label
columns — and writes a `PredictionSchema`-conformant parquet adding the model's two-head
probabilities per row: `censor_prob` (P(row is censored)) and `occurs_prob`
(P(event occurred | not censored)). No AUCs, no model selection — that's
`evaluate/` (Phase 2.4, #83).

See [#129](https://github.com/payalchandak/EveryQuery/issues/129) for the post-refactor
discussion on generalizing `occurs_prob` → `label_prob` for non-occurrence task types.

## External tasks

See [`external_tasks/README.md`](external_tasks/README.md) for the ACES / composite-code
aggregation utilities.

## Related

- Parent refactor umbrella: [#54](https://github.com/payalchandak/EveryQuery/issues/54)
- Phase 2.1 — task-query schema: [#80](https://github.com/payalchandak/EveryQuery/issues/80)
- Phase 2.2 — `EQ_predict` implementation: [#81](https://github.com/payalchandak/EveryQuery/issues/81) (this PR)
- Phase 3 — external-tasks promotion: [#62](https://github.com/payalchandak/EveryQuery/issues/62)
