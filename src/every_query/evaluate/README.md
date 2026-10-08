# `evaluate/`

Evaluation stage of the EveryQuery pipeline. Two evaluators ship here, one per pipeline:
`EQ_evaluate` for the single-query model and `EQ_evaluate_multitask` for the multitask one.
They consume different prediction schemas and do not substitute for each other.

`EQ_evaluate` was rewired in Phase 2.5 ([#131](https://github.com/payalchandak/EveryQuery/pull/131))
to point at the consolidated `evaluate.py` (single-stage, no model instantiation). The
legacy four-stage evaluator (`eval.py`, `gen_index_times.py`, `gen_task.py`,
`select_model.py`) has been deleted; recover from git history if needed. Cross-model
comparison (what the old `EQ_select_model` did) lives in the `EveryQueryExperiments` repo
— tracked on [#83](https://github.com/payalchandak/EveryQuery/issues/83).

## Consolidated pipeline (`evaluate/evaluate.py`)

```
predict/ predictions.parquet  ──►  EQ_evaluate  ──►  metrics.parquet
(PredictionSchema)                                  (per-(query, duration_days): n_rows,
                                                     n_occurs_labeled, n_positive,
                                                     occurs_auroc, censor_auroc,
                                                     prevalence)
```

```bash
EQ_evaluate \
	predictions_parquet="$TRAINING_OUTPUT_DIR/predictions.parquet" \
	metrics_parquet="$TRAINING_OUTPUT_DIR/metrics.parquet"
```

One Hydra main. No model instantiation, no trainer loop, no multi-model orchestration.

## Multitask pipeline (`evaluate/evaluate_multitask.py`)

`EQ_predict_multitask` writes one row per evaluation-grid row, not per
`(query, duration_days)` pair, so `EQ_evaluate` cannot read it. `EQ_evaluate_multitask`
groups those rows by the query **specification** — the five window list columns
`(queries, durations, start_durations, start_events, bound_events)` — which is what
`EQ_generate_evaluation_query_sequences` resolves once and labels at every context — plus
`prior_answers` (`answers[:-1]`), the teacher-forced answers the final query was conditioned
on. Each group is therefore one task — one spec under one fixed conditioning — and every
`(subject_id, prediction_time)` row in it is one prediction. AUROC is computed within the
task, never pooled across tasks, which would measure cross-query (or cross-conditioning)
base-rate separation instead. With one query per spec, `prior_answers` is always `[]` and
the tasks are exactly the specs. `forced_answers` is in the key too: a designed spec that forces
an answer is written only at the contexts where that answer is true, so it holds the same rows as
the matching `prior_answers` task of the unforced spec, and keying on it keeps the two from being
pooled (and double-counted) when a grid carries both.

```
predict/ predictions.parquet  ──►  EQ_evaluate_multitask  ──►  <stem>.by_task.parquet
(one row per grid row)                                          (one row per task: AUROC + 95% CI)
```

```bash
EQ_evaluate_multitask \
	predictions_parquet="$TRAINING_OUTPUT_DIR/predictions.parquet" \
	metrics_stem="$TRAINING_OUTPUT_DIR/metrics"
```

The output is the per-task AUROC and its 95% interval, nothing else — no macro, no cross-task
intervals. The interval is a row bootstrap within the task: draw the task's rows with
replacement, recompute the AUROC, repeat `n_resamples` times (default 1000, seeded by
`bootstrap_seed`), and take the 2.5th / 97.5th percentiles. A single-class task has a null
`auroc` and a null interval. Rows are treated as independent, so when
`prediction_times_per_subject` exceeds 1 the interval runs a little narrow; `n_subjects` is
reported beside `n_rows` so that case is visible.

## Related

- Parent refactor umbrella: [#54](https://github.com/payalchandak/EveryQuery/issues/54)
- Phase 2.2 — `EQ_predict` (the producer for the new pipeline): [#81](https://github.com/payalchandak/EveryQuery/issues/81) (closed, merged in [#99](https://github.com/payalchandak/EveryQuery/pull/99))
- Phase 2.4 — consolidated `evaluate.py` landed: [#100](https://github.com/payalchandak/EveryQuery/pull/100)
- Phase 2.5 — `EQ_evaluate` rewired to new main: [#131](https://github.com/payalchandak/EveryQuery/pull/131)
- Cross-model leaderboard (lives in `EveryQueryExperiments`): [#83](https://github.com/payalchandak/EveryQuery/issues/83)
