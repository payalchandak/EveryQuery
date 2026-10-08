"""Tests for ``EQ_evaluate_multitask``'s task grouping, AUROCs and bootstrap intervals.

Two halves:

- **The defensive ``group_by`` suite.**  ``compute_multitask_metrics`` groups directly on five
  *list* columns rather than hashing them to a ``task_id``, which keeps the spec legible in the
  output but leans on a polars path the project does not otherwise exercise.  ``pyproject`` allows
  ``polars>=1.35,<2``, so these tests pin the behaviour across that range: distinct specs must
  partition, nulls inside lists must stay significant, the ``-1.0`` event-bound sentinel must not
  merge with a real horizon, the partition must round-trip against the grid it came from, and the
  result must not depend on input row order.  If a future polars breaks any of them the fallback is
  a derived ``pl.struct(TASK_KEY).hash()`` key.
- **End-to-end metric behaviour** on small synthetic frames: a separable task scores 1.0, a
  single-class task comes back null with no interval, the conditioning answers split a spec into
  tasks, and each task's row-bootstrap interval brackets its AUROC, is reproducible per seed, and
  resamples rows rather than subjects.

Every fixture here is synthetic — no cohort data is read.
"""

from __future__ import annotations

import polars as pl
import pytest

from every_query.evaluate.evaluate_multitask import (
    TASK_KEY,
    _with_prior_answers,
    compute_multitask_metrics,
)

# Small enough to keep the suite fast, large enough that the percentile bounds are stable.
_N_RESAMPLES = 64

_GRID_SCHEMA = {
    "subject_id": pl.Int64,
    "queries": pl.List(pl.Utf8),
    "durations": pl.List(pl.Float32),
    "start_durations": pl.List(pl.Float32),
    "start_events": pl.List(pl.Utf8),
    "bound_events": pl.List(pl.Utf8),
    "forced_answers": pl.List(pl.Boolean),
    "answers": pl.List(pl.Boolean),
    "label": pl.Boolean,
    "prob": pl.Float32,
}


def _frame(rows: list[dict]) -> pl.DataFrame:
    """Rows to a prediction frame; a row without ``answers`` gets all-``False`` prior answers.

    All-``False`` priors keep every spec in one cell, so a test that is not about the conditioning
    answers sees exactly the spec-level partition.
    """
    for row in rows:
        row.setdefault("answers", [False] * (len(row["queries"]) - 1) + [row["label"]])
    return pl.DataFrame(rows, schema=_GRID_SCHEMA)


def _grid(specs: list[dict], n_subjects: int = 4) -> pl.DataFrame:
    """Cross every spec with ``n_subjects`` subjects, exactly as the dense evaluation grid does.

    Each spec dict supplies the five ``TASK_KEY`` columns.  Labels alternate and probabilities are
    perfectly separable, so every cell is scorable unless a test overrides it.
    """
    rows = []
    for spec in specs:
        for s in range(n_subjects):
            positive = s < n_subjects // 2
            rows.append(
                {
                    "subject_id": s,
                    **spec,
                    "label": positive,
                    "prob": 0.9 - 0.1 * s,
                }
            )
    return _frame(rows)


def _spec(
    queries: list[str],
    durations: list[float],
    start_durations: list[float] | None = None,
    start_events: list[str | None] | None = None,
    bound_events: list[str | None] | None = None,
    forced_answers: list[bool | None] | None = None,
) -> dict:
    """One ``SequenceSpec``-shaped set of the five list columns, defaults filled in as
    ``EQ_predict_multitask`` normalises them (``0.0`` starts, null start/bound events)."""
    k = len(queries)
    return {
        "queries": queries,
        "durations": durations,
        "start_durations": start_durations if start_durations is not None else [0.0] * k,
        "start_events": start_events if start_events is not None else [None] * k,
        "bound_events": bound_events if bound_events is not None else [None] * k,
        "forced_answers": forced_answers if forced_answers is not None else [None] * k,
    }


def _tuple_set(series: pl.Series) -> set[tuple]:
    """Hashable view of a list column's values — ``sorted`` cannot order lists holding ``None``."""
    return {tuple(v) for v in series.to_list()}


def _cells(predictions: pl.DataFrame) -> pl.DataFrame:
    return compute_multitask_metrics(predictions, n_resamples=_N_RESAMPLES)


# --- 1. no collision --------------------------------------------------------------------------


def test_same_codes_different_durations_do_not_collide() -> None:
    """Identical query codes over different horizons are different questions, so different cells."""
    grid = _grid([_spec(["A", "B"], [1.0, 30.0]), _spec(["A", "B"], [1.0, 7.0])])

    by_task = _cells(grid)

    assert by_task.height == 2
    assert sorted(by_task["durations"].to_list()) == [[1.0, 7.0], [1.0, 30.0]]
    # Same final code, so `target_code` alone would have merged them.
    assert by_task["target_code"].to_list() == ["B", "B"]


def test_event_bounded_and_horizon_bounded_do_not_collide() -> None:
    """One position bounded by an event, the other by a horizon, splits the cell."""
    grid = _grid(
        [
            _spec(["A", "B"], [1.0, 30.0]),
            _spec(["A", "B"], [1.0, -1.0], bound_events=[None, "DISCHARGE"]),
        ]
    )

    by_task = _cells(grid)

    assert by_task.height == 2
    assert sorted(by_task["duration_bucket"].to_list()) == ["8-30d", "event-bound"]


# --- 2. nulls inside lists are significant ----------------------------------------------------


def test_nulls_inside_bound_events_stay_significant() -> None:
    """``[None, None]`` and ``[None, "DISCHARGE"]`` must not group together.

    This is the failure mode most likely to appear from a polars change: a list comparison that
    treats a null element as "absent" rather than as a value would silently pool an event-bounded
    spec with its horizon-bounded twin.
    """
    grid = _grid(
        [
            _spec(["A", "B"], [1.0, -1.0], bound_events=[None, None]),
            _spec(["A", "B"], [1.0, -1.0], bound_events=[None, "DISCHARGE"]),
        ]
    )

    by_task = _cells(grid)

    assert by_task.height == 2
    assert _tuple_set(by_task["bound_events"]) == {(None, None), (None, "DISCHARGE")}


def test_nulls_inside_start_events_stay_significant() -> None:
    """The same guarantee on the other nullable list column, ``start_events``."""
    grid = _grid(
        [
            _spec(["A"], [30.0], start_durations=[-1.0], start_events=["ADMISSION"]),
            _spec(["A"], [30.0], start_durations=[0.0], start_events=[None]),
        ]
    )

    by_task = _cells(grid)

    assert by_task.height == 2
    assert _tuple_set(by_task["start_events"]) == {("ADMISSION",), (None,)}


# --- 3. the -1.0 sentinel groups by exact equality --------------------------------------------


def test_event_bound_sentinel_never_merges_with_a_real_horizon() -> None:
    """``-1.0`` is a sentinel, not a duration: it must never merge with a short real horizon.

    A float comparison that rounded, clamped at zero, or bucketed before grouping would file the sentinel next
    to a sub-day horizon and average two different questions into one AUROC.
    """
    grid = _grid(
        [
            _spec(["A"], [-1.0], bound_events=["DISCHARGE"]),
            _spec(["A"], [0.5]),
            _spec(["A"], [1.0]),
        ]
    )

    by_task = _cells(grid)

    assert by_task.height == 3
    assert sorted(by_task["durations"].to_list()) == [[-1.0], [0.5], [1.0]]
    # Three distinct horizons in, three distinct cells out, each holding the whole cohort.
    assert by_task["n_rows"].to_list() == [4, 4, 4]


def test_event_bound_sentinel_bucket_is_separate() -> None:
    """The sentinel's descriptive bucket is ``event-bound``, not the shortest horizon."""
    grid = _grid([_spec(["A"], [-1.0], bound_events=["DISCHARGE"]), _spec(["A"], [0.5])])

    by_task = _cells(grid).sort("duration_bucket")

    assert by_task["duration_bucket"].to_list() == ["1d", "event-bound"]


def test_start_duration_sentinel_never_merges_with_a_real_delay() -> None:
    """Same exact-equality guarantee on ``start_durations``' sentinel."""
    grid = _grid(
        [
            _spec(["A"], [30.0], start_durations=[-1.0], start_events=["ADMISSION"]),
            _spec(["A"], [30.0], start_durations=[7.0]),
        ]
    )

    by_task = _cells(grid)

    assert by_task.height == 2
    assert sorted(by_task["start_durations"].to_list()) == [[-1.0], [7.0]]


def test_forced_answers_split_an_otherwise_identical_spec() -> None:
    """``P(B | A forced YES)``, ``P(B | A forced NO)`` and teacher-forced ``P(B | A)`` are three tasks over
    the same five window columns; keyed on the windows alone they would pool into one cell."""
    specs = [
        _spec(["A", "B"], [1.0, 30.0], forced_answers=[True, None]),
        _spec(["A", "B"], [1.0, 30.0], forced_answers=[False, None]),
        _spec(["A", "B"], [1.0, 30.0]),
    ]
    by_task = _cells(_grid(specs))
    assert by_task.height == 3
    assert by_task["n_rows"].to_list() == [4, 4, 4]
    assert _tuple_set(by_task["forced_answers"]) == {(True, None), (False, None), (None, None)}


def test_predictions_written_before_forced_answers_existed_still_evaluate() -> None:
    """No ``forced_answers`` column reads as nothing forced: same cells, same numbers."""
    grid = _grid([_spec(["A"], [30.0]), _spec(["A", "B"], [1.0, 30.0])])
    assert _cells(grid.drop("forced_answers")).equals(_cells(grid))


# --- 4. round-trip against the grid -----------------------------------------------------------


def test_cells_round_trip_against_the_grid() -> None:
    """Cell count equals the distinct spec count and ``sum(n_rows)`` equals the input height.

    The strongest of the five: it catches collision (too few cells, or rows double-counted) and
    fragmentation (too many cells) in one assertion, over a grid that exercises every list column.
    """
    specs = [
        _spec(["A"], [30.0]),
        _spec(["A"], [7.0]),
        _spec(["A", "B"], [1.0, 30.0]),
        _spec(["A", "B"], [1.0, -1.0], bound_events=[None, "DISCHARGE"]),
        _spec(["A", "B"], [1.0, -1.0], bound_events=[None, None]),
        _spec(["B"], [30.0], start_durations=[-1.0], start_events=["ADMISSION"]),
        _spec(["B"], [30.0], start_durations=[7.0]),
        _spec(["A", "B"], [1.0, 30.0], forced_answers=[True, None]),
    ]
    n_subjects = 6
    grid = _grid(specs, n_subjects=n_subjects)

    by_task = _cells(grid)

    assert by_task.height == len(specs)
    assert by_task["n_rows"].sum() == grid.height
    assert by_task["n_rows"].to_list() == [n_subjects] * len(specs)
    assert by_task["n_subjects"].to_list() == [n_subjects] * len(specs)
    # The emitted keys are exactly the input's distinct specs, nothing invented or dropped.
    expected = _with_prior_answers(grid).select(TASK_KEY).unique().sort(TASK_KEY)
    assert by_task.select(TASK_KEY).sort(TASK_KEY).equals(expected)


def _conditioned_grid(n_subjects: int = 40) -> pl.DataFrame:
    """One two-query spec whose conditioning answer all but determines the label.

    Even subjects were told ``Q1=YES`` and are mostly positive; odd subjects were told ``Q1=NO`` and
    are mostly negative.  The score is a function of the conditioning answer alone, so it carries
    **no** signal about the label within either group — a model leaning entirely on what it was told.
    """
    spec = _spec(["Q1", "Q2"], [1.0, 30.0])
    rows = []
    for s in range(n_subjects):
        told_yes = s % 2 == 0
        label = told_yes if s % 8 >= 2 else not told_yes  # a quarter of each group breaks the rule
        prob = 0.7 if told_yes else 0.2
        rows.append({"subject_id": s, **spec, "answers": [told_yes, label], "label": label, "prob": prob})
    return _frame(rows)


def test_prior_answers_split_a_spec_into_cells() -> None:
    """One spec, two conditioning prefixes: two cells that partition the spec's rows and subjects."""
    grid = _conditioned_grid()

    by_task = _cells(grid)

    assert by_task["prior_answers"].to_list() == [[False], [True]]
    assert by_task["n_rows"].to_list() == [20, 20]
    # A cell holds only the subjects whose conditioning matches — not the whole cohort.
    assert by_task["n_subjects"].to_list() == [20, 20]
    # The description is still the final query's.
    assert by_task["target_code"].to_list() == ["Q2", "Q2"]
    assert by_task["n_queries"].to_list() == [2, 2]


def test_a_model_that_only_echoes_its_conditioning_does_not_score() -> None:
    """The reason the key carries the prior answers.

    Pooled over the spec, the conditioning answer separates the classes on its own (AUROC well above chance);
    within one conditioning it carries no information, and the cells say so.
    """
    from sklearn.metrics import roc_auc_score

    grid = _conditioned_grid()
    pooled = roc_auc_score(grid["label"].to_list(), grid["prob"].to_list())
    assert pooled == pytest.approx(0.75)

    by_task = _cells(grid)

    assert by_task["auroc"].to_list() == [0.5, 0.5]
    for row in by_task.iter_rows(named=True):
        assert row["auroc_ci_lo"] <= row["auroc"] <= row["auroc_ci_hi"]


# --- 5. order independence --------------------------------------------------------------------


def test_shuffling_input_rows_changes_nothing() -> None:
    """Shuffled input yields identical tasks, AUROCs *and* intervals.

    Guards the grouping and the bootstrap together: rows are put in a canonical order before any
    resample index is drawn, so the intervals cannot depend on how the parquet happened to be
    written.  Uses noisy scores — on a separable grid every interval is ``[1, 1]`` regardless.
    """
    grid = _spread_grid()

    by_task = _cells(grid)
    by_task_shuffled = _cells(grid.sample(fraction=1.0, shuffle=True, seed=17))

    assert by_task["auroc_ci_lo"].n_unique() > 1
    assert by_task.equals(by_task_shuffled)


# --- end-to-end metric behaviour --------------------------------------------------------------


def _separable_and_degenerate_grid() -> pl.DataFrame:
    """Three cells over six subjects: separable, anti-separable, and single-class (unscorable)."""
    separable = _spec(["A"], [30.0])
    inverted = _spec(["A"], [7.0])
    single_class = _spec(["A"], [1.0])
    rows = []
    for s in range(6):
        positive = s < 3
        rows.append({"subject_id": s, **separable, "label": positive, "prob": 0.9 if positive else 0.1})
        rows.append({"subject_id": s, **inverted, "label": positive, "prob": 0.1 if positive else 0.9})
        rows.append({"subject_id": s, **single_class, "label": True, "prob": 0.5 + 0.01 * s})
    return _frame(rows)


def test_separable_cell_scores_one_and_inverted_cell_scores_zero() -> None:
    """A perfectly ordered cell is AUROC 1.0; a perfectly reversed one is 0.0."""
    by_task = _cells(_separable_and_degenerate_grid())

    aurocs = {
        tuple(d): a for d, a in zip(by_task["durations"].to_list(), by_task["auroc"].to_list(), strict=True)
    }
    assert aurocs[(30.0,)] == 1.0
    assert aurocs[(7.0,)] == 0.0


def test_single_class_cell_is_null() -> None:
    """AUROC is undefined on a single-class task: null, with no fabricated interval either."""
    by_task = _cells(_separable_and_degenerate_grid())

    single = by_task.filter(pl.col("durations").list.first() == 1.0)
    assert single.height == 1
    assert single["auroc"].to_list() == [None]
    assert single["auroc_ci_lo"].to_list() == [None]
    assert single["auroc_ci_hi"].to_list() == [None]
    assert single["n_degenerate_replicates"].to_list() == [None]
    assert by_task["auroc"].null_count() == 1


def test_empty_predictions_give_an_empty_table_with_the_full_schema() -> None:
    """No rows in, no tasks out — but the same columns and dtypes a populated run writes."""
    grid = _grid([_spec(["A"], [30.0])])

    assert _cells(grid.clear()).schema == _cells(grid).schema
    assert _cells(grid.clear()).height == 0


def _spread_grid(n_subjects: int = 40) -> pl.DataFrame:
    """A grid with genuinely imperfect, per-cell-varying discrimination.

    Perfectly separable cells give degenerate intervals ``[1.0, 1.0]``, which cannot distinguish a wide
    interval from a narrow one.  Here each cell's scores are a noisy, cell-dependent function of the label, so
    the bootstrap has something to vary.
    """
    specs = [_spec(["A"], [float(d)]) for d in (2, 7, 30, 90, 180)]
    rows = []
    for c, spec in enumerate(specs):
        for s in range(n_subjects):
            positive = s % 2 == 0
            # Overlapping score distributions; the amount of overlap differs per cell.
            base = 0.5 + (0.30 - 0.05 * c) * (1 if positive else -1)
            rows.append(
                {
                    "subject_id": s,
                    **spec,
                    "label": positive,
                    "prob": base + 0.9 * ((s * 7919 % 97) / 97.0 - 0.5),
                }
            )
    return _frame(rows)


def test_per_task_intervals_bracket_their_point_estimates() -> None:
    """Every scored task's row bootstrap brackets that task's own AUROC, with real width."""
    by_task = compute_multitask_metrics(_spread_grid(), n_resamples=200)

    for row in by_task.iter_rows(named=True):
        assert row["auroc"] is not None
        assert row["auroc_ci_lo"] < row["auroc"] < row["auroc_ci_hi"]
        # A task with plenty of both classes should almost never resample single-class.
        assert row["n_degenerate_replicates"] == 0


def test_bootstrap_is_reproducible_and_seed_dependent() -> None:
    """A fixed seed gives fixed intervals; a different seed moves them but not the point AUROC."""
    grid = _spread_grid()
    a = compute_multitask_metrics(grid, n_resamples=200, bootstrap_seed=0)
    b = compute_multitask_metrics(grid, n_resamples=200, bootstrap_seed=0)
    c = compute_multitask_metrics(grid, n_resamples=200, bootstrap_seed=1)

    assert a.equals(b)
    assert a["auroc"].to_list() == c["auroc"].to_list()
    assert a["auroc_ci_lo"].to_list() != c["auroc_ci_lo"].to_list()


def test_rows_are_the_resampling_unit() -> None:
    """Each ``(subject_id, prediction_time)`` row is one prediction, resampled on its own.

    Duplicating every row (a second prediction time per subject with identical labels and scores)
    leaves the point AUROC alone and doubles ``n_rows`` but not ``n_subjects``; the row bootstrap
    sees twice the data and the interval narrows toward ``1/sqrt(2)`` of its width.  That is the
    documented behaviour, pinned here so a change to subject-level resampling is a deliberate one.
    """
    grid = _spread_grid(n_subjects=40)
    single = compute_multitask_metrics(grid, n_resamples=400)
    double = compute_multitask_metrics(pl.concat([grid, grid]), n_resamples=400)

    assert double["auroc"].to_list() == pytest.approx(single["auroc"].to_list())
    assert double["n_rows"].to_list() == [80] * 5
    assert double["n_subjects"].to_list() == [40] * 5

    def mean_width(by_task: pl.DataFrame) -> float:
        return (by_task["auroc_ci_hi"] - by_task["auroc_ci_lo"]).mean()

    assert mean_width(double) == pytest.approx(mean_width(single) / 2**0.5, rel=0.2)


def test_missing_columns_fail_with_the_column_names() -> None:
    """A frame that is not an ``EQ_predict_multitask`` output fails on the schema, not mid-loop."""
    grid = _grid([_spec(["A"], [30.0])]).drop("bound_events")

    with pytest.raises(ValueError, match="bound_events"):
        compute_multitask_metrics(grid, n_resamples=_N_RESAMPLES)


def test_null_label_is_rejected() -> None:
    """The evaluation grid's answers are binary and never null, so a null label is malformed input."""
    grid = _grid([_spec(["A"], [30.0])]).with_columns(
        pl.when(pl.col("subject_id") == 0).then(None).otherwise(pl.col("label")).alias("label")
    )

    with pytest.raises(ValueError, match="null 'label'"):
        compute_multitask_metrics(grid, n_resamples=_N_RESAMPLES)


def test_zero_resamples_is_rejected() -> None:
    """``n_resamples`` trades runtime for resolution; it is not a way to switch intervals off."""
    with pytest.raises(ValueError, match="n_resamples"):
        compute_multitask_metrics(_grid([_spec(["A"], [30.0])]), n_resamples=0)
