# `tests/training_validity/`

End-to-end test that trains an `EveryQueryModel` on a designed-signal synthetic dataset
and asserts the trained model's predictions reflect the signal across both prediction
heads. The EQ analog of [`MEDS_EIC_AR`'s `test_pattern_generation.py`][meicar-gen]:
instead of asserting "the pipeline runs," it asserts "the model *learned* from training."

## What this test covers that the rest of the E2E suite does not

The other integration tests — `tests/test_process_data.py`, `tests/test_generate_tasks.py`,
`tests/test_train.py`, `tests/test_e2e_foundation.py` — cover:

- The pipeline runs end-to-end without raising
- CLI knobs are honored (differentials on `n_tasks`, `MIN_SUBJECTS_PER_CODE`, etc.)
- Resume advances `global_step`
- Sampler output has the right schema + label semantics

They all train for **2 optimizer steps on random weights** and never observe the training
dynamics. So silent regressions in the gradient path, label alignment, loss-mask
wiring, or duration-input propagation would pass every one of them.

This test trains for **2000 steps** on a tiny model (~5-7 min wall time) against a dataset
where the ground-truth labels are a deterministic function of two per-subject marker
tokens. It asserts:

| #   | Assertion                                                                                           | Regression it catches                                                                  |
| --- | --------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| 1   | **Censor-head AUROC ≥ 0.9**                                                                         | Broken censor-head wiring, flag-token tokenisation failure, censor-loss mask inverted  |
| 2   | **Per-(code, duration) occurs AUROC ≥ 0.8** on every non-degenerate cell                            | Broken occurs-head wiring, label flip, pattern-marker tokenisation failure             |
| 3   | **Duration monotonicity** — mean predicted probability strictly increases across `d=1 < d=7 < d=30` | Duration-input feature path detached from gradient, model ignoring the duration scalar |

A gradient-path regression (no parameter updates), label flip, constant-output head, or
collapsed query-token embedding all drop at least one AUROC below threshold.

## Dataset design

See [#123][issue-123] for the full design-space comparison; four candidate approaches were
evaluated against the unified criteria and **Design 2 — oracle markers + fire-time
patterns** was chosen.

### Per-subject markers (emitted at day 0)

Each subject gets exactly one of each marker pair:

- **Fire-time marker** — one of `P_FIRE_D05`, `P_FIRE_D5`, `P_FIRE_D15`, `P_FIRE_D50`,
    `P_NEVER`. Determines whether (and when) the single `TARGET` event fires relative to
    the prediction time.
- **End marker** — one of `P_END_D20`, `P_END_D100`. Determines the subject's
    observation window (20 days or 100 days), which drives censoring at long durations.

### Event stream

- `TARGET` fires once on day `prediction_time + fire_offset` for firing-type subjects
    whose observation window includes that day.
- 5 `NOISE_0..NOISE_4` codes are Poisson-drawn at 0.5/day each to fill the sequence.
    The model has to attend to the two markers rather than treating every history as
    identical.

### Query task

- Query code: `TARGET`
- Query durations: `[1, 7, 30]` days
- Prediction time: day 10

Combined with the fire-time and end markers, the windowing logic produces:

| fire_marker                 | max_time  | d=1 occurs | d=7 occurs | d=30 occurs       | d=30 censored?     |
| --------------------------- | --------- | ---------- | ---------- | ----------------- | ------------------ |
| P_FIRE_D05 (fires day 10.5) | 20 or 100 | ✓          | ✓          | varies on censor  | END_D20 → censored |
| P_FIRE_D5 (fires day 15)    | 20 or 100 | ✗          | ✓          | varies on censor  | END_D20 → censored |
| P_FIRE_D15 (fires day 25)   | 20 or 100 | ✗          | ✗          | if not censored   | END_D20 → censored |
| P_FIRE_D50 (fires day 60)   | 20 or 100 | ✗          | ✗          | event past window | END_D20 → censored |
| P_NEVER (no fire)           | 20 or 100 | ✗          | ✗          | ✗                 | END_D20 → censored |

This produces ~20%/40%/70% `occurs=True` at `d=1/7/30` among non-censored subjects
(hence duration monotonicity), and ~55% censored at `d=30` (driven by `END_D20`).

### Sample data

All snapshots below read from the *actual* training shard the test fixture builds —
`tests/training_validity/conftest.py` calls the same `_synthesize_meds` +
`_compute_labels` helpers the test uses and injects the resulting `events` / `labels`
DataFrames into the doctest namespace. The synthesis seed is pinned to `_DATASET_SEED=1`,
so these numbers are byte-identical across CI and local runs — when any of them drift,
something in the synthesis path changed.

At a glance: 100 training subjects, ~14.8k events — 200 day-0 markers (two per
subject), 54 `TARGET` events (one per firing-type subject whose observation window
reaches the fire offset), and ~14.5k Poisson-drawn noise events filling the sequence.

```python
>>> events.shape
(14806, 4)
>>> events.group_by("code").len().sort("code")
shape: (13, 2)
┌────────────┬──────┐
│ code       ┆ len  │
│ ---        ┆ ---  │
│ str        ┆ u32  │
╞════════════╪══════╡
│ NOISE_0    ┆ 2872 │
│ NOISE_1    ┆ 2966 │
│ NOISE_2    ┆ 2909 │
│ NOISE_3    ┆ 2877 │
│ NOISE_4    ┆ 2928 │
│ P_END_D100 ┆ 48   │
│ P_END_D20  ┆ 52   │
│ P_FIRE_D05 ┆ 17   │
│ P_FIRE_D15 ┆ 18   │
│ P_FIRE_D5  ┆ 16   │
│ P_FIRE_D50 ┆ 24   │
│ P_NEVER    ┆ 25   │
│ TARGET     ┆ 54   │
└────────────┴──────┘

```

Each subject draws exactly one fire marker and one end marker at day 0; this is the
distribution across the 10 possible pairs after the seeded RNG assigns them:

```python
>>> markers = events.filter(pl.col("code").str.starts_with("P_"))
>>> pair_per_subject = (
...     markers.group_by("subject_id")
...     .agg(pl.col("code").sort())
...     .with_columns(pl.col("code").list.join(" + ").alias("pair"))
... )
>>> pair_per_subject.group_by("pair").len().sort("pair")
shape: (10, 2)
┌─────────────────────────┬─────┐
│ pair                    ┆ len │
│ ---                     ┆ --- │
│ str                     ┆ u32 │
╞═════════════════════════╪═════╡
│ P_END_D100 + P_FIRE_D05 ┆ 6   │
│ P_END_D100 + P_FIRE_D15 ┆ 10  │
│ P_END_D100 + P_FIRE_D5  ┆ 10  │
│ P_END_D100 + P_FIRE_D50 ┆ 11  │
│ P_END_D100 + P_NEVER    ┆ 11  │
│ P_END_D20 + P_FIRE_D05  ┆ 11  │
│ P_END_D20 + P_FIRE_D15  ┆ 8   │
│ P_END_D20 + P_FIRE_D5   ┆ 6   │
│ P_END_D20 + P_FIRE_D50  ┆ 13  │
│ P_END_D20 + P_NEVER     ┆ 14  │
└─────────────────────────┴─────┘

```

The head of the full events DataFrame — both markers for subject 1000 at day 0,
followed by Poisson noise over the next few days. Subject 1000 drew `P_FIRE_D15`, so
its `TARGET` event lands on day 25 (past this 10-row head):

```python
>>> events.head(10)
shape: (10, 4)
┌────────────┬────────────────────────────────┬────────────┬───────────────┐
│ subject_id ┆ time                           ┆ code       ┆ numeric_value │
│ ---        ┆ ---                            ┆ ---        ┆ ---           │
│ i64        ┆ datetime[μs, UTC]              ┆ str        ┆ f64           │
╞════════════╪════════════════════════════════╪════════════╪═══════════════╡
│ 1000       ┆ 2020-01-01 00:00:00 UTC        ┆ P_END_D100 ┆ null          │
│ 1000       ┆ 2020-01-01 00:00:00 UTC        ┆ P_FIRE_D15 ┆ null          │
│ 1000       ┆ 2020-01-01 14:13:03.469283 UTC ┆ NOISE_1    ┆ null          │
│ 1000       ┆ 2020-01-01 17:15:31.271791 UTC ┆ NOISE_0    ┆ null          │
│ 1000       ┆ 2020-01-02 16:22:14.730412 UTC ┆ NOISE_2    ┆ null          │
│ 1000       ┆ 2020-01-02 16:37:17.200863 UTC ┆ NOISE_3    ┆ null          │
│ 1000       ┆ 2020-01-02 23:50:13.880159 UTC ┆ NOISE_4    ┆ null          │
│ 1000       ┆ 2020-01-03 03:44:53.346246 UTC ┆ NOISE_4    ┆ null          │
│ 1000       ┆ 2020-01-03 06:31:01.635866 UTC ┆ NOISE_4    ┆ null          │
│ 1000       ┆ 2020-01-03 11:00:42.293597 UTC ┆ NOISE_1    ┆ null          │
└────────────┴────────────────────────────────┴────────────┴───────────────┘

```

The head of the labels DataFrame shows how the marker pair maps onto
(censored, occurs) across the three durations — three rows per subject, one per
queried duration:

```python
>>> labels.head(9)
shape: (9, 6)
┌────────────┬─────────────────────┬───────────────┬────────┬────────┬───────────────┐
│ subject_id ┆ prediction_time     ┆ boolean_value ┆ occurs ┆ query  ┆ duration_days │
│ ---        ┆ ---                 ┆ ---           ┆ ---    ┆ ---    ┆ ---           │
│ i64        ┆ datetime[μs]        ┆ bool          ┆ bool   ┆ str    ┆ i64           │
╞════════════╪═════════════════════╪═══════════════╪════════╪════════╪═══════════════╡
│ 1000       ┆ 2020-01-11 00:00:00 ┆ false         ┆ false  ┆ TARGET ┆ 1             │
│ 1000       ┆ 2020-01-11 00:00:00 ┆ false         ┆ false  ┆ TARGET ┆ 7             │
│ 1000       ┆ 2020-01-11 00:00:00 ┆ false         ┆ true   ┆ TARGET ┆ 30            │
│ 1001       ┆ 2020-01-11 00:00:00 ┆ false         ┆ false  ┆ TARGET ┆ 1             │
│ 1001       ┆ 2020-01-11 00:00:00 ┆ false         ┆ false  ┆ TARGET ┆ 7             │
│ 1001       ┆ 2020-01-11 00:00:00 ┆ false         ┆ true   ┆ TARGET ┆ 30            │
│ 1002       ┆ 2020-01-11 00:00:00 ┆ false         ┆ false  ┆ TARGET ┆ 1             │
│ 1002       ┆ 2020-01-11 00:00:00 ┆ false         ┆ false  ┆ TARGET ┆ 7             │
│ 1002       ┆ 2020-01-11 00:00:00 ┆ true          ┆ false  ┆ TARGET ┆ 30            │
└────────────┴─────────────────────┴───────────────┴────────┴────────┴───────────────┘

```

Aggregate positive / censored rates across the full 100-subject training split —
17/33/26 `occurs=True` at d=1/7/30 (monotone, as the duration-monotonicity check
requires) and 52/100 censored at d=30 (driven by the `P_END_D20` end marker, which
ends observation at day 20 — inside the 30-day query window):

```python
>>> labels.group_by("duration_days").agg(
...     pl.len().alias("n"),
...     pl.col("boolean_value").sum().alias("n_censored"),
...     pl.col("occurs").sum().alias("n_occurs"),
... ).sort("duration_days")
shape: (3, 4)
┌───────────────┬─────┬────────────┬──────────┐
│ duration_days ┆ n   ┆ n_censored ┆ n_occurs │
│ ---           ┆ --- ┆ ---        ┆ ---      │
│ i64           ┆ u32 ┆ u32        ┆ u32      │
╞═══════════════╪═════╪════════════╪══════════╡
│ 1             ┆ 100 ┆ 0          ┆ 17       │
│ 7             ┆ 100 ┆ 0          ┆ 33       │
│ 30            ┆ 100 ┆ 52         ┆ 26       │
└───────────────┴─────┴────────────┴──────────┘

```

Finally, two subject-level zooms. `subject_id=1000` drew `P_FIRE_D15 + P_END_D100` —
`TARGET` fires on day 25 (inside only the d=30 window, since prediction_time=10) and
observation extends to day 100, so nothing is censored. The model's view of history
ends at prediction_time=day 10, so it sees the two markers plus whichever noise
events fell in the first 10 days, and must predict future `TARGET` firing from the
markers alone:

```python
>>> subj_1000 = events.filter(pl.col("subject_id") == 1000)
>>> markers_and_target = ["P_FIRE_D15", "P_END_D100", "TARGET"]
>>> subj_1000.filter(pl.col("code").is_in(markers_and_target)).sort(["time", "code"])
shape: (3, 4)
┌────────────┬─────────────────────────┬────────────┬───────────────┐
│ subject_id ┆ time                    ┆ code       ┆ numeric_value │
│ ---        ┆ ---                     ┆ ---        ┆ ---           │
│ i64        ┆ datetime[μs, UTC]       ┆ str        ┆ f64           │
╞════════════╪═════════════════════════╪════════════╪═══════════════╡
│ 1000       ┆ 2020-01-01 00:00:00 UTC ┆ P_END_D100 ┆ null          │
│ 1000       ┆ 2020-01-01 00:00:00 UTC ┆ P_FIRE_D15 ┆ null          │
│ 1000       ┆ 2020-01-26 00:00:00 UTC ┆ TARGET     ┆ null          │
└────────────┴─────────────────────────┴────────────┴───────────────┘
>>> labels.filter(pl.col("subject_id") == 1000).select("duration_days", "boolean_value", "occurs")
shape: (3, 3)
┌───────────────┬───────────────┬────────┐
│ duration_days ┆ boolean_value ┆ occurs │
│ ---           ┆ ---           ┆ ---    │
│ i64           ┆ bool          ┆ bool   │
╞═══════════════╪═══════════════╪════════╡
│ 1             ┆ false         ┆ false  │
│ 7             ┆ false         ┆ false  │
│ 30            ┆ false         ┆ true   │
└───────────────┴───────────────┴────────┘

```

Contrast: `subject_id=1003` drew `P_FIRE_D05 + P_END_D20` — `TARGET` fires on day
10.5 (inside every duration window) but observation ends at day 20, censoring the
d=30 window:

```python
>>> labels.filter(pl.col("subject_id") == 1003).select("duration_days", "boolean_value", "occurs")
shape: (3, 3)
┌───────────────┬───────────────┬────────┐
│ duration_days ┆ boolean_value ┆ occurs │
│ ---           ┆ ---           ┆ ---    │
│ i64           ┆ bool          ┆ bool   │
╞═══════════════╪═══════════════╪════════╡
│ 1             ┆ false         ┆ true   │
│ 7             ┆ false         ┆ true   │
│ 30            ┆ true          ┆ false  │
└───────────────┴───────────────┴────────┘

```

## Why this particular design

Three of the four #123 candidates were evaluated:

- **Design 1** (pure Poisson + `SUBJECT_FLAG` for censoring) was attempted during
    implementation; per-cell occurs AUROC is inherently capped near chance because the
    Poisson process is memoryless — there's no per-subject signal in history to predict
    the next interval at a fixed rate.
- **Design 1a** (per-subject log-normal latent intensity + flag) got *close* — censor
    AUROC 0.994 in 18m42s of training, but occurs AUROC on the HOT-at-d=1 cell only
    reached 0.697. The Bayes-optimal classifier on that cell is bounded below 1.0 by
    the Bernoulli-sampling-noise floor, and the model's latent-intensity inference from
    a 30-day history didn't approach it in the CPU budget.
- **Design 2** (this test) clears every threshold on first try with perfect AUROC
    (1.000) on every cell at 2000 steps. The deterministic-marker → label mapping
    sidesteps the sampling-noise ceiling entirely.

Design 2 is the right trade-off for a *training-validity* test: the model isn't being
asked to estimate a latent variable, it's being asked to attend to a couple of marker
tokens and compose them with the duration input. That's the minimum bar for "the
architecture works end-to-end" — if it can't do this, it can't do anything more
complex.

If we want a stricter test that exercises rate-estimation from history, the Design 1a
branch [`test/e2e-training-validity-d1a`][d1a] remains available as a follow-up.

## Runtime

Target was ≤ 10 minutes CPU per [#123][issue-123]. Runs at `trainer.max_steps=2000`:

| Environment                    | Test wall time |
| ------------------------------ | -------------- |
| Laptop-class CPU               | ~6-7 min       |
| GitHub Actions `ubuntu-latest` | ~5 min         |

(Test wall time = the single `test_trained_model_learns_occurs_and_censor` test end to
end, which is what counts toward the #123 budget — includes MEDS preprocessing +
training + tuning-set inference, not just training. Full CI test-session wall time
including the other 199 tests runs ~10-11 min.)

Subprocess timeout is set to 1800s (30 min) as a safety ceiling; the workflow's job
`timeout-minutes` is 45 (see `.github/workflows/tests.yaml`) for slack beyond that.

An earlier iteration of this test used `max_steps=4000` to brute-force past an
unlucky weight-init trajectory observed on one Python-3.12 CI run (censor AUROC
under-converged to 0.765 / flat duration means). That was rooted in `train.py`
calling `seed_everything` *after* `hydra.utils.instantiate(cfg.lightning_module)`,
so model init was sampled from an unseeded RNG and varied across platforms. Fixed
in #124 (`fix(train): seed RNG before instantiate()`); once that landed, `max_steps`
was dropped back to 2000.

## Gotchas baked into the test (for future readers)

Three label-semantics and dataset-construction details worth calling out; all are
documented inline in the test module too.

1. **`boolean_value` = *censored*, not *occurs***. The EQ sampler overloads MEDS's
    `boolean_value` label column to mean "censored" (observation ended before we could
    observe the outcome), and uses a separate `occurs` column for the real positive-class
    label. The dataset derives `batch.censor = boolean_value` and the model's
    occurs-loss is masked to `~batch.censor`. Swapping the two labels silently inverts
    training. (See also [#122][issue-122] for the ongoing discussion of collapsing these
    into one nullable column.)
2. **`MEDSDataset.write` treats `data_shards` keys as path stems**, so the key
    `"train/0"` produces `data/train/0.parquet` but `"train"` produces
    `data/train.parquet`. MEDS-transforms expects the sharded layout; flat layout
    breaks the fit_normalization stage's subject_splits lookup.
3. **Query-code attribution uses `dataset.query[i]`**, not a separately-sorted label
    parquet — iteration order is the dataset's internal schema_df, not our sort of the
    source parquet. Re-indexing from a sorted parquet mis-aligns rows.

## Related

- Parent umbrella: [#104][issue-104]
- Design issue: [#123][issue-123]
- PR: [#119](https://github.com/payalchandak/EveryQuery/pull/119)

[d1a]: https://github.com/payalchandak/EveryQuery/tree/test/e2e-training-validity-d1a
[issue-104]: https://github.com/payalchandak/EveryQuery/issues/104
[issue-122]: https://github.com/payalchandak/EveryQuery/issues/122
[issue-123]: https://github.com/payalchandak/EveryQuery/issues/123
[meicar-gen]: https://github.com/mmcdermott/MEDS_EIC_AR/blob/main/tests/test_pattern_generation.py
