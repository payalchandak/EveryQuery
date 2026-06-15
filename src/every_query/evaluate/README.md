# `evaluate/`

Evaluation stage of the EveryQuery pipeline. Consumes a `PredictionSchema` parquet
written by `EQ_predict`, produces a per-`(query, duration_days)` metrics parquet.

`EQ_evaluate` was rewired in Phase 2.5 ([#131](https://github.com/payalchandak/EveryQuery/pull/131))
to point at the consolidated `evaluate.py` (single-stage, no model instantiation). The
legacy four-stage evaluator (`eval.py`, `gen_index_times.py`, `gen_task.py`,
`select_model.py`) has been deleted; recover from git history if needed. Cross-model
comparison (what the old `EQ_select_model` did) moves to `experiments/leaderboard/`
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
	predictions_parquet="$OUTPUT_DIR/predictions.parquet" \
	metrics_parquet="$OUTPUT_DIR/metrics.parquet"
```

One Hydra main. No model instantiation, no trainer loop, no multi-model orchestration.

## Related

- Parent refactor umbrella: [#54](https://github.com/payalchandak/EveryQuery/issues/54)
- Phase 2.2 — `EQ_predict` (the producer for the new pipeline): [#81](https://github.com/payalchandak/EveryQuery/issues/81) (closed, merged in [#99](https://github.com/payalchandak/EveryQuery/pull/99))
- Phase 2.4 — consolidated `evaluate.py` landed: [#100](https://github.com/payalchandak/EveryQuery/pull/100)
- Phase 2.5 — `EQ_evaluate` rewired to new main: [#131](https://github.com/payalchandak/EveryQuery/pull/131)
- Leaderboard relocation (tracks `experiments/leaderboard/`): [#83](https://github.com/payalchandak/EveryQuery/issues/83)
