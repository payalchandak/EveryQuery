"""Stage 4: per-shard labeling — ``evaluate_index_df`` (pure core) + ``label_one_shard`` (worker).

Covers the three-valued label (True / False / null-censored), strict-``>`` at prediction_time,
inclusive window-end, missing-subject censoring, dtype normalization on the event-shard read path,
the worker's skip/overwrite/atomicity behavior, and stale-temp cleanup.  Gap tests pin the
forward-only asof direction (past events don't count) and query-code isolation.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from every_query.generate_tasks import sample_tasks as st
from every_query.generate_tasks.sample_tasks import (
    _clean_stale_temps,
    evaluate_index_df,
    label_one_shard,
)


class TestEvaluateIndexDfEdgeCases:
    """Hand-crafted edge-case tests for ``evaluate_index_df``."""

    def test_event_exactly_at_prediction_time_is_excluded(self):
        """``prediction_time == event_time`` must not count as an occurrence (strict ``>``).

        Under the collapsed ``TaskQuerySchema`` label: non-censored row with an event in
        the window → ``boolean_value = True``; non-censored row without → ``False``.

        When printed, the ``events`` DataFrame looks like::

            shape: (5, 3)
            ┌────────────┬─────────────────────┬──────┐
            │ subject_id ┆ time                ┆ code │
            │ ---        ┆ ---                 ┆ ---  │
            │ i64        ┆ datetime[μs]        ┆ str  │
            ╞════════════╪═════════════════════╪══════╡
            │ 1          ┆ 2020-01-01 00:00:00 ┆ A    │
            │ 1          ┆ 2020-01-02 00:00:00 ┆ A    │
            │ 1          ┆ 2020-01-03 00:00:00 ┆ A    │
            │ 1          ┆ 2020-01-04 00:00:00 ┆ A    │
            │ 1          ┆ 2021-01-01 00:00:00 ┆ A    │
            └────────────┴─────────────────────┴──────┘
        """
        events = pl.DataFrame(
            {
                "subject_id": [1, 1, 1, 1, 1],
                "time": [
                    datetime(2020, 1, 1),
                    datetime(2020, 1, 2),
                    datetime(2020, 1, 3),
                    datetime(2020, 1, 4),
                    datetime(2021, 1, 1),  # record_end_time far enough that days=10 is uncensored
                ],
                "code": ["A", "A", "A", "A", "A"],
            }
        ).sort(["subject_id", "time"])

        # Construct one index row at prediction_time = 2020-01-03. The event at 2020-01-03
        # should NOT count. The next event is 2020-01-04, which is within 10d → True.
        index_df = pl.DataFrame(
            {
                "subject_id": [1],
                "prediction_time": [datetime(2020, 1, 3)],
                "query": ["A"],
                "duration_days": [10],
            }
        ).with_columns(pl.col("prediction_time").cast(pl.Datetime("us")))
        result = evaluate_index_df(index_df, events)
        assert result["boolean_value"].to_list() == [True]

        # Now with duration=0: no window → False (and not censored because 2020-01-03
        # + 0 days = 2020-01-03 which is <= 2021-01-01).
        index_df_zero = index_df.with_columns(pl.lit(0, dtype=pl.Int64).alias("duration_days"))
        result_zero = evaluate_index_df(index_df_zero, events)
        assert result_zero["boolean_value"].to_list() == [False]

    def test_event_exactly_at_window_end_is_included(self):
        """An event at exactly ``prediction_time + duration_days`` counts as an occurrence.

        The upper window bound is **inclusive** (``<=``), matching upstream
        ``MEDS_trajectory_evaluation`` (``tte <= evaluation_window_end``).  This test hardcodes
        the expected label rather than re-deriving it, so it pins the inclusive boundary
        independent of the implementation's own comparison (issue #223).

        - subject 1: only matching event lands exactly on ``window_end`` → ``True``.
        - subject 2: only matching event lands 1µs **past** ``window_end`` → ``False``
          (uncensored, so the bound — not censoring — is what excludes it).
        """
        # window_end for prediction_time 2020-01-01 + 10 days == 2020-01-11 00:00:00 exactly.
        window_end = datetime(2020, 1, 11)
        events = (
            pl.DataFrame(
                {
                    "subject_id": [1, 1, 2, 2],
                    "time": [
                        window_end,  # subj 1: exactly on the boundary → included
                        datetime(2021, 1, 1),  # pushes max_time well past window_end (uncensored)
                        window_end + timedelta(microseconds=1),  # subj 2: just past boundary → excluded
                        datetime(2021, 1, 1),  # uncensored
                    ],
                    "code": ["A", "A", "A", "A"],
                }
            )
            .with_columns(pl.col("time").cast(pl.Datetime("us")))
            .sort(["subject_id", "time"])
        )

        index_df = pl.DataFrame(
            {
                "subject_id": [1, 2],
                "prediction_time": [datetime(2020, 1, 1), datetime(2020, 1, 1)],
                "query": ["A", "A"],
                "duration_days": [10, 10],
            }
        ).with_columns(pl.col("prediction_time").cast(pl.Datetime("us")))

        result = evaluate_index_df(index_df, events)
        labels = {row["subject_id"]: row["boolean_value"] for row in result.iter_rows(named=True)}
        assert labels == {1: True, 2: False}

    def test_unknown_subject_is_treated_as_censored(self, caplog):
        """An index_df row referencing a subject absent from events_df resolves to censored (``boolean_value =
        null``) under the collapsed schema, and emits a warning."""
        events = pl.DataFrame(
            {
                "subject_id": [1],
                "time": [datetime(2020, 1, 1)],
                "code": ["A"],
            }
        ).with_columns(pl.col("time").cast(pl.Datetime("us")))

        index_df = pl.DataFrame(
            {
                # Subject 1 is present; subject 2 is not.
                "subject_id": [1, 2],
                "prediction_time": [datetime(2020, 1, 1), datetime(2020, 1, 1)],
                "query": ["A", "A"],
                "duration_days": [10, 10],
            }
        ).with_columns(pl.col("prediction_time").cast(pl.Datetime("us")))

        with caplog.at_level("WARNING", logger="every_query.generate_tasks.sample_tasks"):
            result = evaluate_index_df(index_df, events)

        # Subject 2 (unknown) → censored → null boolean_value.
        unknown = result.filter(pl.col("subject_id") == 2)
        assert unknown["boolean_value"].to_list() == [None]

        # Warning was emitted with a count.
        assert any("not present in events_df" in record.message for record in caplog.records), (
            f"expected unknown-subject warning, got: {[r.message for r in caplog.records]}"
        )

    # -- Gap tests: asof direction + query-code isolation -----------------------------------------

    def test_event_before_prediction_time_is_not_an_occurrence(self):
        """A matching event strictly *before* prediction_time must not count (forward-only asof).

        subject 1's only in-code "A" events are at day 0 (before the day-5 prediction_time) and day
        100 (well past the day-12 window end), with day 100 keeping the row uncensored → ``False``.
        """
        base = datetime(2020, 1, 1)
        events = pl.DataFrame(
            {
                "subject_id": [1, 1],
                "time": [base, base + timedelta(days=100)],
                "code": ["A", "A"],
            }
        ).with_columns(pl.col("time").cast(pl.Datetime("us")))

        index_df = pl.DataFrame(
            {
                "subject_id": [1],
                "prediction_time": [base + timedelta(days=5)],
                "query": ["A"],
                "duration_days": [7.0],
            }
        ).with_columns(pl.col("prediction_time").cast(pl.Datetime("us")))

        result = evaluate_index_df(index_df, events)
        assert result["boolean_value"].to_list() == [False]

    def test_nonmatching_code_in_window_is_ignored(self):
        """An in-window event with a *different* query code must not satisfy the query.

        subject 1 has a "B" event inside the window and the matching "A" code only outside it. The
        ``A`` query must be ``False`` (the in-window "B" is invisible to it), while a control ``B``
        query on the same window is ``True`` — proving the asof ``by=["subject_id","query"]``
        isolation.
        """
        base = datetime(2020, 1, 1)
        events = pl.DataFrame(
            {
                "subject_id": [1, 1],
                "time": [base + timedelta(days=6), base + timedelta(days=100)],
                "code": ["B", "A"],  # B in window (day 6), A only at day 100 (uncensored, out of window)
            }
        ).with_columns(pl.col("time").cast(pl.Datetime("us")))

        index_df = pl.DataFrame(
            {
                "subject_id": [1, 1],
                "prediction_time": [base + timedelta(days=5), base + timedelta(days=5)],
                "query": ["A", "B"],
                "duration_days": [7.0, 7.0],
            }
        ).with_columns(pl.col("prediction_time").cast(pl.Datetime("us")))

        result = evaluate_index_df(index_df, events)
        labels = {row["query"]: row["boolean_value"] for row in result.iter_rows(named=True)}
        assert labels == {"A": False, "B": True}

    def test_multiple_matching_events_in_window_still_true(self):
        """≥2 matching events inside the window resolve to a single ``True`` (OR semantics)."""
        base = datetime(2020, 1, 1)
        events = pl.DataFrame(
            {
                "subject_id": [1, 1, 1],
                "time": [base + timedelta(days=6), base + timedelta(days=8), base + timedelta(days=100)],
                "code": ["A", "A", "A"],
            }
        ).with_columns(pl.col("time").cast(pl.Datetime("us")))

        index_df = pl.DataFrame(
            {
                "subject_id": [1],
                "prediction_time": [base + timedelta(days=5)],
                "query": ["A"],
                "duration_days": [7.0],
            }
        ).with_columns(pl.col("prediction_time").cast(pl.Datetime("us")))

        result = evaluate_index_df(index_df, events)
        assert result.height == 1
        assert result["boolean_value"].to_list() == [True]


class TestBooleanValueTruthTable:
    """One test per row of the spec's labeling truth table (redesign-spec.md §Stage 4).

    For each ``(subject_id, prediction_time)`` row the observed window is
    ``(prediction_time, min(prediction_time + duration_days, max_time[subject_id])]``.
    Occurrence is resolved first; censoring applies only when the event did **not** occur:

    | occurs in observed window | censored | ``boolean_value`` |
    | ------------------------- | -------- | ----------------- |
    | yes                       | —        | True              |
    | no                        | yes      | null              |
    | no                        | no       | False             |

    Each case is a single index row over a single subject so the resolved label is
    unambiguously the table row under test, pinned independently of the other two.
    """

    BASE = datetime(2020, 1, 1)

    def _evaluate_one(self, events: pl.DataFrame, duration_days: float) -> bool | None:
        """Label a single index row (subject 1, code ``A``, ``prediction_time = BASE``) and return its
        ``boolean_value``."""
        events = events.with_columns(pl.col("time").cast(pl.Datetime("us"))).sort(["subject_id", "time"])
        index_df = pl.DataFrame(
            {
                "subject_id": [1],
                "prediction_time": [self.BASE],
                "query": ["A"],
                "duration_days": [duration_days],
            }
        ).with_columns(pl.col("prediction_time").cast(pl.Datetime("us")))
        result = evaluate_index_df(index_df, events)
        assert result.height == 1
        return result["boolean_value"].to_list()[0]

    def test_occurs_in_observed_window_is_true(self):
        """Row 1: a matching event falls in the observed window → ``True``.

        Event ``A`` at day 3 is strictly within ``(day 0, day 7]``; the later event at day 100
        pushes ``max_time`` well past the window so occurrence — not censoring — decides the label.
        """
        events = pl.DataFrame(
            {
                "subject_id": [1, 1],
                "time": [self.BASE + timedelta(days=3), self.BASE + timedelta(days=100)],
                "code": ["A", "A"],
            }
        )
        assert self._evaluate_one(events, duration_days=7.0) is True

    def test_occurs_takes_priority_over_censoring_is_true(self):
        """Row 1, censored column ``—``: occurrence wins even when the requested window runs past ``max_time``
        → ``True``, never ``null``.

        The subject's last (and only matching) event is ``A`` at day 3, so ``max_time = day 3``.
        The requested window end (day 30) exceeds ``max_time`` — the censoring predicate
        ``prediction_time + duration_days > max_time`` is **true** — but the ``A`` at day 3 falls
        in the observed window ``(day 0, day 3]``.  Per the spec, occurrence is resolved first, so
        the label is ``True`` and censoring never applies.
        """
        events = pl.DataFrame(
            {
                "subject_id": [1],
                "time": [self.BASE + timedelta(days=3)],
                "code": ["A"],
            }
        )
        assert self._evaluate_one(events, duration_days=30.0) is True

    def test_no_occurrence_and_censored_is_null(self):
        """Row 2: no matching event in the observed window **and** the window runs past
        ``max_time`` → ``null`` (censored).

        The subject's only event is at day 10 (so ``max_time = day 10``) and carries a
        non-matching code, so no ``A`` occurs in ``(day 0, day 10]``.  The requested window
        end (day 30) exceeds ``max_time``, so the unobserved tail is unknown → censored.
        """
        events = pl.DataFrame(
            {
                "subject_id": [1],
                "time": [self.BASE + timedelta(days=10)],
                "code": ["B"],  # non-matching: no "A" anywhere
            }
        )
        assert self._evaluate_one(events, duration_days=30.0) is None

    def test_no_occurrence_and_fully_observed_is_false(self):
        """Row 3: no matching event in the window and the full window is observed → ``False``.

        The matching event ``A`` lands at day 100 — outside ``(day 0, day 7]`` — and keeps
        ``max_time`` (day 100) past the window end (day 7), so the window is fully observed
        with no in-window occurrence.
        """
        events = pl.DataFrame(
            {
                "subject_id": [1],
                "time": [self.BASE + timedelta(days=100)],
                "code": ["A"],
            }
        )
        assert self._evaluate_one(events, duration_days=7.0) is False


class TestReadEventShardDtypeNormalization:
    """``_read_event_shard`` must normalize ``code`` → Utf8 and ``time`` → Datetime(us) regardless of how the
    source parquet encoded them, so ``evaluate_index_df``'s joins stay type-stable."""

    def test_categorical_code_is_normalized_to_utf8(self, tmp_path):
        fp = tmp_path / "0.parquet"
        df = pl.DataFrame(
            {
                "subject_id": [1, 2],
                "time": [datetime(2020, 1, 1), datetime(2020, 2, 1)],
                "code": pl.Series(["A", "B"], dtype=pl.Categorical),
                "numeric_value": [1.0, 2.0],  # extra column to test `.select` doesn't blow up
            }
        )
        df.write_parquet(fp)

        out = st._read_event_shard(fp)
        assert out.schema["code"] == pl.Utf8
        assert out.schema["time"] == pl.Datetime("us")
        assert set(out.columns) == {"subject_id", "time", "code"}
        assert sorted(out["code"].to_list()) == ["A", "B"]

    def test_millisecond_time_is_normalized_to_microseconds(self, tmp_path):
        fp = tmp_path / "0.parquet"
        df = pl.DataFrame(
            {
                "subject_id": [1],
                "time": pl.Series([datetime(2020, 1, 1)], dtype=pl.Datetime("ms")),
                "code": ["A"],
            }
        )
        df.write_parquet(fp)

        out = st._read_event_shard(fp)
        assert out.schema["time"] == pl.Datetime("us")

    def test_normalized_shard_joins_correctly_in_evaluate(self, tmp_path):
        """End-to-end: a Categorical-coded shard must produce correct labels when fed through
        ``_read_event_shard`` + ``evaluate_index_df``."""
        fp = tmp_path / "0.parquet"
        pl.DataFrame(
            {
                "subject_id": [1, 1, 1],
                "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2021, 1, 1)],
                "code": pl.Series(["A", "A", "B"], dtype=pl.Categorical),
            }
        ).write_parquet(fp)

        events = st._read_event_shard(fp)
        index_df = pl.DataFrame(
            {
                "subject_id": [1],
                "prediction_time": [datetime(2020, 1, 1)],
                "query": ["A"],
                "duration_days": [10],
            }
        ).with_columns(pl.col("prediction_time").cast(pl.Datetime("us")))

        result = evaluate_index_df(index_df, events)
        # Uncensored (max_time = 2021-01-01 is way past prediction + 10d) and the next "A"
        # event is 2020-01-02, which is strictly within the window → boolean_value=True.
        assert result["boolean_value"].to_list() == [True]


def _make_shard_fixture(
    tmp_path: Path,
    events: pl.DataFrame,
    index_df: pl.DataFrame,
    shard: str = "0",
) -> tuple[Path, Path, Path]:
    """Write an events shard and an index partition to disk; return ``(index_dir, data_dir, out_dir)``."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    events.write_parquet(data_dir / f"{shard}.parquet")

    index_dir = tmp_path / "_index"
    index_dir.mkdir(parents=True, exist_ok=True)
    index_df.write_parquet(index_dir / f"{shard}.parquet")

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    return index_dir, data_dir, out_dir


class TestLabelOneShard:
    """Stage 4 per-shard labeling worker."""

    BASE = datetime(2020, 1, 1, tzinfo=UTC)

    def _events(self) -> pl.DataFrame:
        """3 subjects, each with events at day 0, 5, 10, 15, 20."""
        rows = [
            {"subject_id": s, "time": self.BASE + timedelta(days=d), "code": "ICD//A01"}
            for s in [1, 2, 3]
            for d in [0, 5, 10, 15, 20]
        ]
        return pl.DataFrame(rows).with_columns(pl.col("time").cast(pl.Datetime("us")))

    def _index(self, duration_days: float = 7.0) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "subject_id": [1, 2, 3],
                "prediction_time": [
                    self.BASE + timedelta(days=2),
                    self.BASE + timedelta(days=2),
                    self.BASE + timedelta(days=2),
                ],
                "query": ["ICD//A01", "ICD//A01", "ICD//A01"],
                "duration_days": [duration_days, duration_days, duration_days],
            }
        ).with_columns(
            pl.col("prediction_time").cast(pl.Datetime("us")),
            pl.col("duration_days").cast(pl.Float32),
        )

    def test_basic_labeling(self, tmp_path):
        events = self._events()
        index_df = self._index(duration_days=7.0)
        index_dir, data_dir, out_dir = _make_shard_fixture(tmp_path, events, index_df)

        _shard, status = label_one_shard("0", index_dir, data_dir, out_dir)
        assert status == "labeled"

        result = pl.read_parquet(out_dir / "0.parquet")
        assert result.height == 3
        expected_cols = {"subject_id", "prediction_time", "query", "duration_days", "boolean_value"}
        assert set(result.columns) == expected_cols
        # prediction_time=day2, event at day5 is in (day2, day2+7=day9] → True for all
        assert result["boolean_value"].to_list() == [True, True, True]

    def test_skip_on_matching_fingerprint(self, tmp_path):
        """A second run over the *same* index skips (fingerprint matches), leaving output untouched."""
        events = self._events()
        index_df = self._index()
        index_dir, data_dir, out_dir = _make_shard_fixture(tmp_path, events, index_df)

        _shard, status = label_one_shard("0", index_dir, data_dir, out_dir, overwrite=False)
        assert status == "labeled"
        first_bytes = (out_dir / "0.parquet").read_bytes()

        _shard, status = label_one_shard("0", index_dir, data_dir, out_dir, overwrite=False)
        assert status == "skipped"
        assert (out_dir / "0.parquet").read_bytes() == first_bytes

    def test_relabels_when_fingerprint_missing(self, tmp_path):
        """An existing output with no provenance sidecar is treated as stale ⇒ relabel (safe default).

        Guards against the old existence-only skip, which would silently keep a pre-fingerprint (or half-
        written) output.
        """
        events = self._events()
        index_df = self._index()
        index_dir, data_dir, out_dir = _make_shard_fixture(tmp_path, events, index_df)

        (out_dir / "0.parquet").write_bytes(b"stale-no-fingerprint")

        _shard, status = label_one_shard("0", index_dir, data_dir, out_dir, overwrite=False)
        assert status == "labeled"
        assert pl.read_parquet(out_dir / "0.parquet").height == 3

    def test_relabels_when_index_changed(self, tmp_path):
        """When the Stage 3 index is rewritten with different content, overwrite=False still relabels.

        This is bug #2: Stage 3 always rebuilds the index, so an existence-only skip would keep stale labels
        after a sampling-config change. The fingerprint mismatch must force a relabel.
        """
        events = self._events()
        index_dir, data_dir, out_dir = _make_shard_fixture(tmp_path, events, self._index(duration_days=7.0))

        _shard, status = label_one_shard("0", index_dir, data_dir, out_dir, overwrite=False)
        assert status == "labeled"

        # Rewrite the index partition with a duration that flips the labels (window day2->day3 excludes
        # the day5 event), mimicking Stage 3 rebuilding under a changed config.
        self._index(duration_days=1.0).write_parquet(index_dir / "0.parquet")

        _shard, status = label_one_shard("0", index_dir, data_dir, out_dir, overwrite=False)
        assert status == "labeled"
        # day2 + 1d = day3 window; the next "A01" event is day5 (outside) → False, not the prior True.
        assert pl.read_parquet(out_dir / "0.parquet")["boolean_value"].to_list() == [False, False, False]

    def test_overwrite(self, tmp_path):
        events = self._events()
        index_df = self._index()
        index_dir, data_dir, out_dir = _make_shard_fixture(tmp_path, events, index_df)

        (out_dir / "0.parquet").write_bytes(b"sentinel")

        _shard, status = label_one_shard("0", index_dir, data_dir, out_dir, overwrite=True)
        assert status == "labeled"
        result = pl.read_parquet(out_dir / "0.parquet")
        assert result.height == 3

    def test_float_duration_labeling(self, tmp_path):
        """Float duration_days must not be truncated to integer days."""
        events = pl.DataFrame(
            {
                "subject_id": [1, 1, 1],
                "time": [
                    self.BASE,
                    self.BASE + timedelta(days=1, hours=6),  # 1.25 days after base
                    self.BASE + timedelta(days=100),
                ],
                "code": ["ICD//A01", "ICD//A01", "ICD//X99"],
            }
        ).with_columns(pl.col("time").cast(pl.Datetime("us")))

        # prediction_time = base, duration = 1.5 days → window ends at day 1.5
        # event at day 1.25 is in (base, base+1.5d] → True
        index_true = pl.DataFrame(
            {
                "subject_id": [1],
                "prediction_time": [self.BASE],
                "query": ["ICD//A01"],
                "duration_days": [1.5],
            }
        ).with_columns(
            pl.col("prediction_time").cast(pl.Datetime("us")),
            pl.col("duration_days").cast(pl.Float32),
        )
        index_dir, data_dir, out_dir = _make_shard_fixture(tmp_path, events, index_true)
        label_one_shard("0", index_dir, data_dir, out_dir)
        result = pl.read_parquet(out_dir / "0.parquet")
        assert result["boolean_value"][0] is True

        # duration = 1.0 days → window ends at day 1.0; event at day 1.25 is outside → False
        out_dir2 = tmp_path / "out2"
        out_dir2.mkdir()
        index_false = index_true.with_columns(pl.lit(1.0).cast(pl.Float32).alias("duration_days"))
        index_false.write_parquet(index_dir / "0.parquet")
        label_one_shard("0", index_dir, data_dir, out_dir2)
        result2 = pl.read_parquet(out_dir2 / "0.parquet")
        assert result2["boolean_value"][0] is False

    def test_censoring_logic(self, tmp_path):
        """Three-valued label through the full worker: True (event in window), False (no event,
        fully observed), null (censored).  This exercises all three label values end-to-end through
        ``label_one_shard`` (not just ``evaluate_index_df`` in isolation)."""
        events = pl.DataFrame(
            {
                "subject_id": [1, 1, 2, 2, 3, 3],
                "time": [
                    self.BASE,
                    self.BASE + timedelta(days=5),  # event at day 5
                    self.BASE,
                    self.BASE + timedelta(days=10),  # max_time = day 10
                    self.BASE,
                    self.BASE + timedelta(days=10),  # max_time = day 10
                ],
                "code": ["ICD//X", "ICD//A01", "ICD//X", "ICD//X", "ICD//X", "ICD//X"],
            }
        ).with_columns(pl.col("time").cast(pl.Datetime("us")))

        index_df = pl.DataFrame(
            {
                "subject_id": [1, 2, 3],
                "prediction_time": [self.BASE, self.BASE, self.BASE],
                "query": ["ICD//A01", "ICD//A01", "ICD//A01"],
                # subject 1: window 7d, event at day 5 → True
                # subject 2: window 7d, no ICD//A01 event, max_time=10 ≥ 0+7 → False
                # subject 3: window 30d, no ICD//A01 event, max_time=10 < 0+30 → null (censored)
                "duration_days": [7.0, 7.0, 30.0],
            }
        ).with_columns(
            pl.col("prediction_time").cast(pl.Datetime("us")),
            pl.col("duration_days").cast(pl.Float32),
        )

        index_dir, data_dir, out_dir = _make_shard_fixture(tmp_path, events, index_df)
        label_one_shard("0", index_dir, data_dir, out_dir)

        result = pl.read_parquet(out_dir / "0.parquet").sort("subject_id")
        labels = result["boolean_value"].to_list()
        assert labels[0] is True
        assert labels[1] is False
        assert labels[2] is None

    def test_stale_temp_cleanup(self, tmp_path):
        events = self._events()
        index_df = self._index()
        index_dir, data_dir, out_dir = _make_shard_fixture(tmp_path, events, index_df)

        # Create orphan temp files
        (out_dir / ".0.parquet.tmp.12345").write_bytes(b"stale")
        (out_dir / ".0.parquet.tmp.67890").write_bytes(b"stale")
        assert len(list(out_dir.glob(".0.parquet.tmp.*"))) == 2

        label_one_shard("0", index_dir, data_dir, out_dir)
        assert len(list(out_dir.glob(".0.parquet.tmp.*"))) == 0
        assert (out_dir / "0.parquet").exists()

    def test_empty_index_partition(self, tmp_path):
        events = self._events()
        empty_index = pl.DataFrame(
            schema={
                "subject_id": pl.Int64,
                "prediction_time": pl.Datetime("us"),
                "query": pl.Utf8,
                "duration_days": pl.Float32,
            }
        )
        index_dir, data_dir, out_dir = _make_shard_fixture(tmp_path, events, empty_index)

        _shard, status = label_one_shard("0", index_dir, data_dir, out_dir)
        assert status == "labeled"

        result = pl.read_parquet(out_dir / "0.parquet")
        assert result.height == 0
        expected_cols = {"subject_id", "prediction_time", "query", "duration_days", "boolean_value"}
        assert set(result.columns) == expected_cols


class TestCleanStaleTemps:
    def test_removes_matching_temps(self, tmp_path):
        (tmp_path / ".0.parquet.tmp.111").write_bytes(b"x")
        (tmp_path / ".0.parquet.tmp.222").write_bytes(b"x")
        # A temp produced the *real* way (via _unique_tmp_path) must be cleaned too — guards against
        # the cleanup glob drifting from the actual mkstemp naming.
        real = st._unique_tmp_path(tmp_path / "0.parquet")
        (tmp_path / ".1.parquet.tmp.333").write_bytes(b"x")  # different shard

        removed = _clean_stale_temps(tmp_path, "0")
        assert removed == 3
        assert not (tmp_path / ".0.parquet.tmp.111").exists()
        assert not (tmp_path / ".0.parquet.tmp.222").exists()
        assert not real.exists()
        assert (tmp_path / ".1.parquet.tmp.333").exists()  # untouched

    def test_no_temps_returns_zero(self, tmp_path):
        assert _clean_stale_temps(tmp_path, "0") == 0
