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
│   └── predict.yaml
├── external_tasks
│   ├── README.md
│   ├── __init__.py
│   ├── aces_to_eq.py
│   ├── configs
│   │   ├── aces_to_eq.yaml
│   │   ├── get_per_code_from_composite_config.yaml
│   │   └── process_composite_config.yaml
│   ├── get_per_code_from_composite.py
│   ├── process_composite.py
│   └── task_configs
│       ├── README.md
│       └── icu
│           ├── acute_deterioration_event.yaml
│           ├── consecutive_low_map.yaml
│           ├── discharge_to_facility_ventilated.yaml
│           ├── extubation_before_tracheostomy.yaml
│           ├── icu_bounceback.yaml
│           ├── imminent_hypoglycemia.yaml
│           ├── imminent_icu_mortality.yaml
│           ├── mcs_on_vasopressors.yaml
│           ├── mortality_90d.yaml
│           ├── new_onset_atrial_fibrillation.yaml
│           ├── prolonged_ventilation_past_day14.yaml
│           ├── prolonged_ventilation_past_day21.yaml
│           ├── reintubation_after_extubation.yaml
│           ├── second_vasopressor_added.yaml
│           └── vasopressor_reinitiation_after_weaning.yaml
├── predict.py
└── schema.py

```

Key files:

- `predict.py` — `EQ_predict` (inference-only Hydra main).
- `schema.py` — `PredictionSchema` (`TaskQuerySchema` + `censor_prob` + `occurs_prob`).
- `configs/predict.yaml` — required: `model_run_dir`, `tasks_dir`, `output_parquet`; optional: `ckpt_name`, `split` (`held_out` | `tuning`), `overwrite` (default `false` — refuses to clobber an existing `output_parquet`; pass `overwrite=true` to replace).
- `external_tasks/` — convert + aggregate tasks outside EQ's native vocabulary (`aces_to_eq.py`, `process_composite.py`, `get_per_code_from_composite.py`).
- `external_tasks/task_configs/`: the ACES task-definition YAMLs those conversions start from, one file per task ([`task_configs/README.md`](external_tasks/task_configs/README.md)).

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
