"""Tests for ``every_query.sample_tasks`` (sampling-first PT task label generator).

Structure:

- ``TestPrimitives`` — unit tests for the pure building blocks.
- ``TestEvaluateAgainstReference`` — bit-identity cross-check of ``evaluate_index_df`` against the
  ``tasks_reference`` oracle on a shared synthetic fixture.
- ``TestRunWorkerPipeline`` — artifact-level tests for the three-stage idempotent pipeline:
  stage skip on rerun, overwrite, seed-axis independence, pre-seeded tasks.
- ``TestEndToEndWithDataset`` — smoke test that builds sampler-shaped labels referencing real
  subject IDs from ``tensorized_cohort_dir``, feeds them through ``EveryQueryPytorchDataset``, and
  pushes one collated batch through ``demo_model``.  Verifies schema compatibility with the
  downstream dataloader.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
import torch
from meds import train_split
from meds_torchdata.config import MEDSTorchDataConfig

from every_query import sample_tasks as st
from every_query import tasks_reference
from every_query.dataset import EveryQueryPytorchDataset
from every_query.sample_tasks import (
    TaskSpec,
    build_index_df,
    compute_max_time_per_subject,
    derive_seed,
    evaluate_index_df,
    run_worker,
    sample_contexts,
    sample_tasks,
)

# ---------------------------------------------------------------------------
# Shared synthetic fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_events() -> pl.DataFrame:
    """A deterministic events DataFrame: 3 subjects x 30 events x 5 codes, 10d spacing.

    Shape matches the ``test_tasks.py`` fixture so the same data exercises both the old
    wide-matrix pipeline and the new sampler.
    """
    codes = ["ICD//A01", "ICD//B02", "ICD//C03", "MED//D04", "MED//E05"]
    base = datetime(2020, 1, 1)
    rows = [
        {
            "subject_id": subj,
            "time": base + timedelta(days=i * 10 + subj),
            "code": codes[i % len(codes)],
        }
        for subj in range(1, 4)
        for i in range(30)
    ]
    return pl.DataFrame(rows).sort(["subject_id", "time"])


@pytest.fixture
def synthetic_query_codes() -> list[str]:
    return ["ICD//A01", "ICD//B02", "ICD//C03", "MED//D04", "MED//E05"]


# ---------------------------------------------------------------------------
# Pure-primitive unit tests
# ---------------------------------------------------------------------------


class TestPrimitives:
    def test_derive_seed_is_stable_across_calls(self):
        assert derive_seed(1, "tasks", 0) == derive_seed(1, "tasks", 0)
        assert derive_seed(1, "contexts", "shard_a", 7) == derive_seed(1, "contexts", "shard_a", 7)

    def test_derive_seed_separates_axes(self):
        assert derive_seed(1, "tasks", 0) != derive_seed(1, "tasks", 1)
        assert derive_seed(1, "contexts", "a", 0) != derive_seed(1, "contexts", "b", 0)
        assert derive_seed(1, "contexts", "a", 0) != derive_seed(1, "contexts", "a", 1)
        assert derive_seed(1, "tasks", 0) != derive_seed(2, "tasks", 0)

    def test_sample_tasks_determinism(self, synthetic_query_codes):
        a = sample_tasks(32, synthetic_query_codes, 1, 365, seed=42)
        b = sample_tasks(32, synthetic_query_codes, 1, 365, seed=42)
        assert a == b

    def test_sample_tasks_respects_bounds(self, synthetic_query_codes):
        tasks = sample_tasks(256, synthetic_query_codes, 10, 100, seed=7)
        assert all(isinstance(t, TaskSpec) for t in tasks)
        assert all(t.code in synthetic_query_codes for t in tasks)
        assert all(10 <= t.duration_days <= 100 for t in tasks)

    def test_sample_tasks_rejects_invalid_inputs(self, synthetic_query_codes):
        with pytest.raises(ValueError):
            sample_tasks(-1, synthetic_query_codes, 1, 365, seed=0)
        with pytest.raises(ValueError):
            sample_tasks(5, [], 1, 365, seed=0)
        with pytest.raises(ValueError):
            sample_tasks(5, synthetic_query_codes, 0, 365, seed=0)
        with pytest.raises(ValueError):
            sample_tasks(5, synthetic_query_codes, 100, 50, seed=0)

    def test_sample_contexts_returns_exactly_n_rows(self, synthetic_events):
        # Each subject has 30 events, min_context=5 leaves 26 candidates per subject.
        contexts = sample_contexts(synthetic_events, n=50, min_context_per_subject=5, seed=1)
        assert contexts.height == 50
        assert set(contexts.columns) == {"subject_id", "prediction_time"}

    def test_sample_contexts_determinism(self, synthetic_events):
        a = sample_contexts(synthetic_events, n=20, min_context_per_subject=5, seed=3)
        b = sample_contexts(synthetic_events, n=20, min_context_per_subject=5, seed=3)
        assert a.equals(b)

    def test_sample_contexts_different_seeds_differ(self, synthetic_events):
        a = sample_contexts(synthetic_events, n=40, min_context_per_subject=5, seed=1)
        b = sample_contexts(synthetic_events, n=40, min_context_per_subject=5, seed=2)
        assert not a.equals(b)

    def test_sample_contexts_respects_min_context(self, synthetic_events):
        # Each subject has 30 events — asking for min_context=28 leaves 3 candidates per subject = 9 total.
        contexts = sample_contexts(synthetic_events, n=100, min_context_per_subject=28, seed=0)
        # With replacement, we still get exactly n rows; the underlying pool is the 9 candidates.
        assert contexts.height == 100
        # At most 9 distinct candidate rows (3 per subject x 3 subjects).
        distinct_pairs = contexts.unique().height
        assert distinct_pairs <= 9

    def test_sample_contexts_empty_candidate_pool(self, synthetic_events):
        """``min_context`` larger than any subject's event count yields a zero-row result."""
        contexts = sample_contexts(synthetic_events, n=10, min_context_per_subject=999, seed=0)
        assert contexts.height == 0
        assert set(contexts.columns) == {"subject_id", "prediction_time"}

    def test_sample_contexts_n_zero(self, synthetic_events):
        contexts = sample_contexts(synthetic_events, n=0, min_context_per_subject=5, seed=0)
        assert contexts.height == 0

    def test_build_index_df_zip_correctness(self, synthetic_events):
        """Each task's 2 contexts appear consecutively, labeled with that task's code/duration."""
        tasks = [TaskSpec("X", 30), TaskSpec("Y", 60), TaskSpec("Z", 90)]
        contexts = sample_contexts(synthetic_events, n=6, min_context_per_subject=5, seed=0)
        index_df = build_index_df(tasks, contexts)

        assert index_df.height == 6
        assert set(index_df.columns) == {
            "task_id",
            "subject_id",
            "prediction_time",
            "query",
            "duration_days",
        }
        assert index_df["task_id"].to_list() == [0, 0, 1, 1, 2, 2]
        assert index_df["query"].to_list() == ["X", "X", "Y", "Y", "Z", "Z"]
        assert index_df["duration_days"].to_list() == [30, 30, 60, 60, 90, 90]

    def test_build_index_df_m_equals_one(self, synthetic_events):
        tasks = [TaskSpec("A", 30), TaskSpec("B", 60)]
        contexts = sample_contexts(synthetic_events, n=2, min_context_per_subject=5, seed=0)
        index_df = build_index_df(tasks, contexts)
        assert index_df.height == 2
        assert index_df["task_id"].to_list() == [0, 1]

    def test_build_index_df_rejects_non_divisible(self, synthetic_events):
        tasks = [TaskSpec("A", 30), TaskSpec("B", 60)]
        contexts = sample_contexts(synthetic_events, n=3, min_context_per_subject=5, seed=0)
        with pytest.raises(ValueError, match="divisible"):
            build_index_df(tasks, contexts)

    def test_build_index_df_empty(self):
        assert build_index_df([], pl.DataFrame({"subject_id": [], "prediction_time": []})).height == 0

    def test_compute_max_time_per_subject(self, synthetic_events):
        max_df = compute_max_time_per_subject(synthetic_events)
        assert set(max_df.columns) == {"subject_id", "max_time"}
        # Every subject has 30 events; max_time is the 30th event time per subject.
        for subj in [1, 2, 3]:
            subj_max = synthetic_events.filter(pl.col("subject_id") == subj)["time"].max()
            row = max_df.filter(pl.col("subject_id") == subj).row(0, named=True)
            assert row["max_time"] == subj_max


# ---------------------------------------------------------------------------
# Cross-check: sampler vs reference oracle
# ---------------------------------------------------------------------------


def _reference_long(
    events_df: pl.DataFrame,
    query_codes: list[str],
    min_context: int,
    duration_days: int,
) -> pl.DataFrame:
    """Run the reference pipeline and unpivot the wide matrix to the sampler's long schema.

    Produces columns ``(subject_id, prediction_time, boolean_value, occurs, query, duration_days)``
    to match :func:`sample_tasks.evaluate_index_df` exactly.
    """
    duration = {"days": duration_days}
    ref_censor = tasks_reference.compute_censor_dataframe(events_df, min_context, duration)
    ref_wide = tasks_reference.build_task_label_matrix(events_df, ref_censor, query_codes, duration)
    # Wide columns: subject_id, prediction_time, censored, <code_1>, ..., <code_N>
    long = (
        ref_wide.unpivot(
            index=["subject_id", "prediction_time", "censored"],
            variable_name="query",
            value_name="occurs",
        )
        .with_columns(pl.col("occurs").fill_null(False))
        .with_columns(
            pl.col("censored").alias("boolean_value"),
            pl.lit(duration_days, dtype=pl.Int64).alias("duration_days"),
        )
        .select("subject_id", "prediction_time", "boolean_value", "occurs", "query", "duration_days")
    )
    return long


def _sampler_long(
    events_df: pl.DataFrame,
    query_codes: list[str],
    min_context: int,
    duration_days: int,
) -> pl.DataFrame:
    """Run the sampler evaluator over the full (subject x pt x code) cross-product for one duration.

    This bypasses :func:`sample_tasks.sample_tasks` / :func:`sample_tasks.sample_contexts` — we don't
    want the equivalence test to depend on sampler randomness.  The equivalence claim is that for
    *any* index_df row, the labels from ``evaluate_index_df`` agree with the reference.
    """
    from every_query import tasks as new_tasks

    base_df = new_tasks.compute_base_prediction_times(events_df, min_context).select(
        "subject_id", "prediction_time"
    )
    codes_df = pl.DataFrame({"query": query_codes}, schema={"query": pl.Utf8})
    index_df = base_df.join(codes_df, how="cross").with_columns(
        pl.lit(duration_days, dtype=pl.Int64).alias("duration_days")
    )
    max_time_df = compute_max_time_per_subject(events_df)
    return evaluate_index_df(index_df, events_df, max_time_df)


def _sort_long(df: pl.DataFrame) -> pl.DataFrame:
    return df.sort(["subject_id", "prediction_time", "query"]).select(
        "subject_id", "prediction_time", "boolean_value", "occurs", "query", "duration_days"
    )


class TestEvaluateAgainstReference:
    """Bit-identity cross-check of ``evaluate_index_df`` against ``tasks_reference``."""

    @pytest.mark.parametrize("duration_days", [30, 90, 180, 365])
    def test_labels_match_reference(self, synthetic_events, synthetic_query_codes, duration_days):
        ref = _sort_long(_reference_long(synthetic_events, synthetic_query_codes, 5, duration_days))
        new = _sort_long(_sampler_long(synthetic_events, synthetic_query_codes, 5, duration_days))

        # `boolean_value` and `occurs` must agree — the rest is schema/metadata.
        cols = ["subject_id", "prediction_time", "query", "boolean_value", "occurs"]
        assert ref.select(cols).equals(new.select(cols)), (
            f"Label mismatch at duration={duration_days}; "
            f"ref={ref.select(cols).head(5)}, new={new.select(cols).head(5)}"
        )

    def test_codes_not_in_shard_are_all_false(self, synthetic_events):
        """A query code with no events anywhere in the shard should produce ``occurs=False`` for every
        uncensored row (and ``occurs=False`` for censored rows, matching reference)."""
        query_codes = ["ICD//A01", "FAKE//Z99"]
        ref = _sort_long(_reference_long(synthetic_events, query_codes, 5, 90))
        new = _sort_long(_sampler_long(synthetic_events, query_codes, 5, 90))

        new_fake = new.filter(pl.col("query") == "FAKE//Z99")
        assert not new_fake["occurs"].any()
        cols = ["subject_id", "prediction_time", "query", "boolean_value", "occurs"]
        assert ref.select(cols).equals(new.select(cols))

    def test_event_exactly_at_prediction_time_is_excluded(self):
        """``prediction_time == event_time`` must not count as an occurrence (strict ``>``)."""
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
        # should NOT count. The next event is 2020-01-04, which is within 10d → occurs=True.
        index_df = pl.DataFrame(
            {
                "subject_id": [1],
                "prediction_time": [datetime(2020, 1, 3)],
                "query": ["A"],
                "duration_days": [10],
            }
        ).with_columns(pl.col("prediction_time").cast(pl.Datetime("us")))
        max_time_df = compute_max_time_per_subject(events)
        result = evaluate_index_df(index_df, events, max_time_df)
        assert result["boolean_value"].to_list() == [False]
        assert result["occurs"].to_list() == [True]

        # Now with duration=0: no window → occurs=False (and not censored because 2020-01-03
        # + 0 days = 2020-01-03 which is <= 2021-01-01).
        index_df_zero = index_df.with_columns(pl.lit(0, dtype=pl.Int64).alias("duration_days"))
        result_zero = evaluate_index_df(index_df_zero, events, max_time_df)
        assert result_zero["occurs"].to_list() == [False]

    def test_unknown_subject_is_treated_as_censored(self, caplog):
        """An index_df row referencing a subject absent from events_df must not produce null booleans — the
        torch collate can't consume nulls.

        Policy: treat missing-subject rows as
        censored (boolean_value=True, occurs=False) and emit a warning so the condition is
        visible to the user.
        """
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

        max_time_df = compute_max_time_per_subject(events)
        with caplog.at_level("WARNING", logger="every_query.sample_tasks"):
            result = evaluate_index_df(index_df, events, max_time_df)

        # Neither row produces a null label.
        assert result["boolean_value"].null_count() == 0
        assert result["occurs"].null_count() == 0

        # Subject 2 (unknown) → censored, not occurred.
        unknown = result.filter(pl.col("subject_id") == 2)
        assert unknown["boolean_value"].to_list() == [True]
        assert unknown["occurs"].to_list() == [False]

        # Warning was emitted with a count.
        assert any("not present in events_df" in record.message for record in caplog.records), (
            f"expected unknown-subject warning, got: {[r.message for r in caplog.records]}"
        )


# ---------------------------------------------------------------------------
# I/O helpers: dtype normalization and atomic writes
# ---------------------------------------------------------------------------


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

        result = evaluate_index_df(index_df, events, compute_max_time_per_subject(events))
        # Uncensored (max_time = 2021-01-01 is way past prediction + 10d) and the next "A"
        # event is 2020-01-02, which is strictly within the window → occurs=True.
        assert result["boolean_value"].to_list() == [False]
        assert result["occurs"].to_list() == [True]


class TestAtomicWriteConcurrency:
    """Atomic-write helpers must use unique tmp filenames so two concurrent writers to the same target don't
    clobber each other.

    We can't directly test a race condition, but we can assert the tmp filename differs across invocations
    with the same target.
    """

    def test_unique_tmp_path_changes_between_calls(self, tmp_path):
        fp = tmp_path / "target.parquet"
        a = st._unique_tmp_path(fp)
        b = st._unique_tmp_path(fp)
        assert a != b, "two _unique_tmp_path calls for the same target must return different names"
        assert a.parent == fp.parent
        assert b.parent == fp.parent
        # Both have the target basename as a prefix so they sit alongside the target.
        assert fp.name in a.name
        assert fp.name in b.name
        # Clean up the placeholder tmpfiles mkstemp creates.
        a.unlink(missing_ok=True)
        b.unlink(missing_ok=True)

    def test_atomic_write_text_round_trip(self, tmp_path):
        fp = tmp_path / "out.json"
        st._atomic_write_text(fp, '{"hello": "world"}')
        assert fp.read_text() == '{"hello": "world"}'
        # No orphan tmpfiles should linger after a successful write.
        orphans = [p for p in tmp_path.iterdir() if p.name != "out.json"]
        assert orphans == []

    def test_atomic_write_parquet_round_trip(self, tmp_path):
        fp = tmp_path / "out.parquet"
        df = pl.DataFrame({"a": [1, 2, 3]})
        st._atomic_write_parquet(df, fp)
        assert fp.exists()
        loaded = pl.read_parquet(fp)
        assert loaded.equals(df)
        orphans = [p for p in tmp_path.iterdir() if p.name != "out.parquet"]
        assert orphans == []


# ---------------------------------------------------------------------------
# run_worker: artifact-level pipeline tests
# ---------------------------------------------------------------------------


def _write_fake_cohort(
    tmp_path: Path,
    events: pl.DataFrame,
    query_codes: list[str],
    split: str = "train",
    shard_name: str = "0",
) -> tuple[Path, Path]:
    """Write a minimal MEDS-shaped cohort so ``run_worker`` can read from it.

    Mirrors the split layout of ``.env.example``: ``$INTERMEDIATE`` and ``$PROCESSED`` are sibling
    directories rather than the same root, so the cohort this helper writes uses:

        {data_dir}/data/{split}/{shard_name}.parquet     <- event shards ($INTERMEDIATE)
        {codes_dir}/metadata/codes.parquet               <- query universe ($PROCESSED)

    Returns ``(data_dir, codes_dir)`` as two distinct paths so the tests exercise the separation.
    """
    data_dir = tmp_path / "intermediate"
    split_dir = data_dir / "data" / split
    split_dir.mkdir(parents=True, exist_ok=True)
    events.write_parquet(split_dir / f"{shard_name}.parquet")

    codes_dir = tmp_path / "processed"
    metadata_dir = codes_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"code": query_codes}).write_parquet(metadata_dir / "codes.parquet")
    return data_dir, codes_dir


class TestRunWorkerPipeline:
    def test_writes_all_three_artifacts(self, tmp_path, synthetic_events, synthetic_query_codes):
        data_dir, codes_dir = _write_fake_cohort(tmp_path, synthetic_events, synthetic_query_codes)
        out_dir = tmp_path / "out"
        labels_fp = run_worker(
            data_dir=data_dir,
            codes_dir=codes_dir,
            out_dir=out_dir,
            split="train",
            input_shard="0",
            task_shard=0,
            seed=1,
            n_tasks=32,
            contexts_per_task=1,
            duration_min=10,
            duration_max=365,
            min_context_per_subject=5,
        )
        assert labels_fp is not None
        assert labels_fp.exists()
        assert (out_dir / "_artifacts" / "tasks" / "train" / "0__0000.json").exists()
        assert (out_dir / "_artifacts" / "index" / "train" / "0__0000.parquet").exists()

        labels = pl.read_parquet(labels_fp)
        assert labels.height == 32
        assert set(labels.columns) == {
            "subject_id",
            "prediction_time",
            "boolean_value",
            "occurs",
            "query",
            "duration_days",
        }

    def test_rerun_is_idempotent_noop(self, tmp_path, synthetic_events, synthetic_query_codes):
        """Second invocation with identical args returns ``None`` and leaves outputs untouched."""
        data_dir, codes_dir = _write_fake_cohort(tmp_path, synthetic_events, synthetic_query_codes)
        out_dir = tmp_path / "out"
        kwargs = {
            "data_dir": data_dir,
            "codes_dir": codes_dir,
            "out_dir": out_dir,
            "split": "train",
            "input_shard": "0",
            "task_shard": 0,
            "seed": 1,
            "n_tasks": 16,
            "contexts_per_task": 1,
            "duration_min": 10,
            "duration_max": 365,
            "min_context_per_subject": 5,
        }
        first = run_worker(**kwargs)
        assert first is not None
        first_bytes = first.read_bytes()

        second = run_worker(**kwargs)
        assert second is None  # skipped
        assert first.read_bytes() == first_bytes  # untouched

    def test_overwrite_regenerates(self, tmp_path, synthetic_events, synthetic_query_codes):
        """``overwrite=True`` must actually regenerate; ``overwrite=False`` must not.

        Captures bytes before and after each rerun and asserts the invariant directly — the earlier version of
        this test only checked that the tasks file existed and was non-None, which would pass even if
        overwrite were silently ignored (Copilot review on #41).
        """
        data_dir, codes_dir = _write_fake_cohort(tmp_path, synthetic_events, synthetic_query_codes)
        out_dir = tmp_path / "out"
        kwargs = {
            "data_dir": data_dir,
            "codes_dir": codes_dir,
            "out_dir": out_dir,
            "split": "train",
            "input_shard": "0",
            "task_shard": 0,
            "n_tasks": 16,
            "contexts_per_task": 1,
            "duration_min": 10,
            "duration_max": 365,
            "min_context_per_subject": 5,
        }
        first = run_worker(seed=1, **kwargs)
        assert first is not None

        tasks_path = out_dir / "_artifacts" / "tasks" / "train" / "0__0000.json"
        first_label_bytes = first.read_bytes()
        first_task_bytes = tasks_path.read_bytes()

        # No-op rerun with same seed: bytes untouched.
        skipped = run_worker(seed=1, **kwargs)
        assert skipped is None
        assert first.read_bytes() == first_label_bytes
        assert tasks_path.read_bytes() == first_task_bytes

        # Overwrite with a different seed: both labels and tasks must change.
        second = run_worker(seed=2, overwrite=True, **kwargs)
        assert second is not None
        assert first == second  # same path
        assert second.read_bytes() != first_label_bytes, (
            "overwrite=True with a new seed did not change the labels parquet"
        )
        assert tasks_path.read_bytes() != first_task_bytes, (
            "overwrite=True with a new seed did not change the tasks JSON"
        )

    def test_task_shard_axis_changes_tasks(self, tmp_path, synthetic_events, synthetic_query_codes):
        """Fixing ``input_shard`` and varying ``task_shard`` must change the sampled tasks."""
        data_dir, codes_dir = _write_fake_cohort(tmp_path, synthetic_events, synthetic_query_codes)
        out_dir = tmp_path / "out"
        kwargs = {
            "data_dir": data_dir,
            "codes_dir": codes_dir,
            "out_dir": out_dir,
            "split": "train",
            "input_shard": "0",
            "seed": 1,
            "n_tasks": 32,
            "contexts_per_task": 1,
            "duration_min": 10,
            "duration_max": 365,
            "min_context_per_subject": 5,
        }
        run_worker(task_shard=0, **kwargs)
        run_worker(task_shard=1, **kwargs)

        tasks_a = (out_dir / "_artifacts" / "tasks" / "train" / "0__0000.json").read_text()
        tasks_b = (out_dir / "_artifacts" / "tasks" / "train" / "0__0001.json").read_text()
        assert tasks_a != tasks_b

    def test_input_shard_axis_preserves_tasks(self, tmp_path, synthetic_events, synthetic_query_codes):
        """Fixing ``task_shard`` and varying ``input_shard`` must keep the same task list but still draw
        *different* contexts.

        This is what enables ``hydra -m input_shard=range(K)`` to evaluate *the same* sampled tasks
        across every shard — a key property for producing a coherent PT dataset — while ensuring
        each shard's worker draws independent contexts (otherwise parallelism across shards
        would be statistically wasteful).
        """
        data_dir, codes_dir = _write_fake_cohort(
            tmp_path, synthetic_events, synthetic_query_codes, shard_name="0"
        )
        # Write a second shard with the same events under a different basename so run_worker has
        # something to load.
        (data_dir / "data" / "train" / "1.parquet").write_bytes(
            (data_dir / "data" / "train" / "0.parquet").read_bytes()
        )

        out_dir = tmp_path / "out"
        kwargs = {
            "data_dir": data_dir,
            "codes_dir": codes_dir,
            "out_dir": out_dir,
            "split": "train",
            "task_shard": 0,
            "seed": 1,
            "n_tasks": 16,
            "contexts_per_task": 1,
            "duration_min": 10,
            "duration_max": 365,
            "min_context_per_subject": 5,
        }
        run_worker(input_shard="0", **kwargs)
        run_worker(input_shard="1", **kwargs)

        # Same tasks across input shards.
        tasks_0 = (out_dir / "_artifacts" / "tasks" / "train" / "0__0000.json").read_text()
        tasks_1 = (out_dir / "_artifacts" / "tasks" / "train" / "1__0000.json").read_text()
        assert tasks_0 == tasks_1, "tasks should be identical across input_shards at fixed task_shard"

        # ... but the sampled context *pairs* must differ, since the context seed depends on
        # input_shard.  Compare the (subject_id, prediction_time) multiset rather than the raw
        # parquet bytes to avoid flakiness from parquet metadata / row-order.
        index_0 = pl.read_parquet(out_dir / "_artifacts" / "index" / "train" / "0__0000.parquet")
        index_1 = pl.read_parquet(out_dir / "_artifacts" / "index" / "train" / "1__0000.parquet")
        pairs_0 = set(index_0.select("subject_id", "prediction_time").iter_rows())
        pairs_1 = set(index_1.select("subject_id", "prediction_time").iter_rows())
        assert pairs_0 != pairs_1, (
            "contexts should differ across input_shards at fixed task_shard "
            "(contexts_seed depends on both axes)"
        )

    def test_config_mismatch_raises_without_overwrite(
        self, tmp_path, synthetic_events, synthetic_query_codes
    ):
        """A second run with a different sampling config must not silently reuse stale artifacts.

        Catches the failure mode Copilot flagged on #41: without a meta sidecar check, the
        artifact filename only keys on ``(split, input_shard, task_shard)``, so changing
        ``seed``, ``n_tasks``, etc. would silently reuse the old output.
        """
        data_dir, codes_dir = _write_fake_cohort(tmp_path, synthetic_events, synthetic_query_codes)
        out_dir = tmp_path / "out"
        kwargs = {
            "data_dir": data_dir,
            "codes_dir": codes_dir,
            "out_dir": out_dir,
            "split": "train",
            "input_shard": "0",
            "task_shard": 0,
            "seed": 1,
            "contexts_per_task": 1,
            "duration_min": 10,
            "duration_max": 365,
            "min_context_per_subject": 5,
        }
        first = run_worker(n_tasks=16, **kwargs)
        assert first is not None

        # Changing n_tasks without overwrite must fail loudly.
        with pytest.raises(ValueError, match="worker config"):
            run_worker(n_tasks=32, **kwargs)

        # overwrite=True bypasses the check and actually regenerates.
        second = run_worker(n_tasks=32, overwrite=True, **kwargs)
        assert second is not None
        labels = pl.read_parquet(second)
        assert labels.height == 32, "overwrite with n_tasks=32 should produce 32 labels"

    def test_config_mismatch_raises_on_seed_change(self, tmp_path, synthetic_events, synthetic_query_codes):
        """Changing only ``seed`` is the other obvious way to produce silent staleness."""
        data_dir, codes_dir = _write_fake_cohort(tmp_path, synthetic_events, synthetic_query_codes)
        out_dir = tmp_path / "out"
        kwargs = {
            "data_dir": data_dir,
            "codes_dir": codes_dir,
            "out_dir": out_dir,
            "split": "train",
            "input_shard": "0",
            "task_shard": 0,
            "n_tasks": 8,
            "contexts_per_task": 1,
            "duration_min": 10,
            "duration_max": 365,
            "min_context_per_subject": 5,
        }
        run_worker(seed=1, **kwargs)
        with pytest.raises(ValueError, match="seed"):
            run_worker(seed=999, **kwargs)

    def test_missing_meta_is_tolerated(self, tmp_path, synthetic_events, synthetic_query_codes):
        """Legacy or hand-injected artifacts without a meta sidecar must still load.

        This preserves the pre-seeding workflow: a user drops tasks.json by hand (without any
        corresponding meta file) and reruns; the sampler loads the preseeded tasks verbatim.
        """
        data_dir, codes_dir = _write_fake_cohort(tmp_path, synthetic_events, synthetic_query_codes)
        out_dir = tmp_path / "out"
        kwargs = {
            "data_dir": data_dir,
            "codes_dir": codes_dir,
            "out_dir": out_dir,
            "split": "train",
            "input_shard": "0",
            "task_shard": 0,
            "seed": 1,
            "n_tasks": 8,
            "contexts_per_task": 1,
            "duration_min": 10,
            "duration_max": 365,
            "min_context_per_subject": 5,
        }
        # First run: creates artifacts and meta.
        first = run_worker(**kwargs)
        assert first is not None

        # Delete the meta sidecar to simulate a legacy / pre-meta artifact.
        run_meta_fp = out_dir / "_artifacts" / "runs" / "train" / "0__0000.json"
        assert run_meta_fp.exists()
        run_meta_fp.unlink()

        # Rerun with the same config: should succeed (labels already exist, no-op path).
        skipped = run_worker(**kwargs)
        assert skipped is None

        # Rerun with a changed config: should also succeed (no meta to validate against);
        # labels already exist so it's a no-op regardless.
        skipped2 = run_worker(**{**kwargs, "n_tasks": 99})
        assert skipped2 is None

    def test_codes_dir_is_read_from_a_distinct_root(self, tmp_path, synthetic_events, synthetic_query_codes):
        """Smoke test that codes_dir is actually read *from* codes_dir, not data_dir.

        The ``.env.example`` layout has ``$INTERMEDIATE`` (event shards) and ``$PROCESSED``
        (metadata/codes.parquet) as sibling directories; the sampler must not conflate them.
        ``_write_fake_cohort`` writes codes.parquet *only* under ``codes_dir`` — so if
        ``run_worker`` were still reading codes from ``data_dir`` it would raise ``FileNotFoundError``.
        """
        data_dir, codes_dir = _write_fake_cohort(tmp_path, synthetic_events, synthetic_query_codes)
        # Precondition: codes.parquet lives *only* under codes_dir, not data_dir.
        assert not (data_dir / "metadata" / "codes.parquet").exists()
        assert (codes_dir / "metadata" / "codes.parquet").exists()

        out_dir = tmp_path / "out"
        labels_fp = run_worker(
            data_dir=data_dir,
            codes_dir=codes_dir,
            out_dir=out_dir,
            split="train",
            input_shard="0",
            task_shard=0,
            seed=1,
            n_tasks=4,
            contexts_per_task=1,
            duration_min=10,
            duration_max=365,
            min_context_per_subject=5,
        )
        assert labels_fp is not None
        labels = pl.read_parquet(labels_fp)
        # The queries in the output must all come from codes.parquet under codes_dir.
        assert set(labels["query"].to_list()) <= set(synthetic_query_codes)

    def test_preseeded_tasks_are_honored(self, tmp_path, synthetic_events, synthetic_query_codes):
        """Dropping a ``tasks.json`` on disk before run → the sampler uses it verbatim."""
        data_dir, codes_dir = _write_fake_cohort(tmp_path, synthetic_events, synthetic_query_codes)
        out_dir = tmp_path / "out"

        preset = [TaskSpec("ICD//A01", 30), TaskSpec("MED//D04", 60)]
        tasks_fp = out_dir / "_artifacts" / "tasks" / "train" / "0__0000.json"
        st.save_tasks(preset, tasks_fp)

        labels_fp = run_worker(
            data_dir=data_dir,
            codes_dir=codes_dir,
            out_dir=out_dir,
            split="train",
            input_shard="0",
            task_shard=0,
            seed=999,  # would normally draw different tasks
            n_tasks=2,
            contexts_per_task=1,
            duration_min=10,
            duration_max=365,
            min_context_per_subject=5,
        )
        assert labels_fp is not None
        labels = pl.read_parquet(labels_fp)
        assert sorted(labels["query"].unique().to_list()) == sorted(["ICD//A01", "MED//D04"])
        assert sorted(labels["duration_days"].unique().to_list()) == [30, 60]


# ---------------------------------------------------------------------------
# End-to-end: sampler output → EveryQueryPytorchDataset → model forward
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# main() path resolution (env fallback + dotenv)
# ---------------------------------------------------------------------------


class TestResolvePath:
    """``_resolve_path`` is how ``main()`` threads ``cfg.data_dir`` / ``$INTERMEDIATE`` / required.

    The three path roots (``data_dir``, ``codes_dir``, ``out_dir``) share this resolution logic,
    which is why it was factored out.  Directly exercising the helper lets us pin the fallback
    matrix without spinning up a full Hydra run per case.
    """

    def test_explicit_cfg_value_wins(self, monkeypatch):
        monkeypatch.setenv("MY_VAR", "/from/env")
        result = st._resolve_path("/from/cfg", "MY_VAR", "data_dir")
        assert result == Path("/from/cfg")

    def test_env_fallback_when_cfg_is_none(self, monkeypatch):
        monkeypatch.setenv("MY_VAR", "/from/env")
        result = st._resolve_path(None, "MY_VAR", "data_dir")
        assert result == Path("/from/env")

    def test_raises_when_both_unset(self, monkeypatch):
        monkeypatch.delenv("MY_VAR", raising=False)
        with pytest.raises(ValueError, match="data_dir must be set"):
            st._resolve_path(None, "MY_VAR", "data_dir")

    def test_error_message_mentions_env_var_and_dotenv(self, monkeypatch):
        monkeypatch.delenv("INTERMEDIATE", raising=False)
        with pytest.raises(ValueError) as excinfo:
            st._resolve_path(None, "INTERMEDIATE", "data_dir")
        msg = str(excinfo.value)
        assert "INTERMEDIATE" in msg
        assert ".env" in msg  # dotenv hint

    def test_empty_env_var_is_treated_as_unset(self, monkeypatch):
        """An explicitly-empty env var should not be taken as a valid path."""
        monkeypatch.setenv("MY_VAR", "")
        with pytest.raises(ValueError, match="must be set"):
            st._resolve_path(None, "MY_VAR", "data_dir")


# Subject IDs and prediction times from conftest / simple_static_MEDS.  Duplicated here rather
# than imported to keep the test module self-contained.
_E2E_TRAIN_SUBJECTS = [239684, 1195293, 68729, 814703]
_E2E_PRED_TIMES: dict[int, datetime] = {
    239684: datetime(2010, 5, 11, 18, 0, tzinfo=UTC),
    1195293: datetime(2010, 6, 20, 20, 30, tzinfo=UTC),
    68729: datetime(2010, 5, 26, 3, 0, tzinfo=UTC),
    814703: datetime(2010, 2, 5, 6, 0, tzinfo=UTC),
}
_E2E_QUERIES = ["HR", "TEMP"]


class TestEndToEndWithDataset:
    """Sanity-check that sampler output is a drop-in replacement for ``collate_tasks`` output.

    We don't run ``run_worker`` here because the sampler's shard-reading path expects string-coded
    events from an early preprocessing stage that the test fixture does not expose (the tensorized
    cohort only materializes ``normalization/*.parquet`` with *integer-coded* events downstream of
    tokenization).  The ``TestRunWorkerPipeline`` class already exercises ``run_worker`` end-to-end
    on synthetic cohorts.

    Instead, we test the downstream half of the pipeline: hand-build an ``index_df`` referencing
    the real subject IDs in ``tensorized_cohort_dir``, call ``evaluate_index_df`` on it against a
    synthetic events table, write the result as sampler-shaped parquet, then instantiate
    ``EveryQueryPytorchDataset`` against the real tokenized shards + the sampler-shaped labels and
    push a collated batch through ``demo_model``.  This is the minimum sufficient test to catch
    schema drift between the sampler's output and the downstream dataset.
    """

    def test_sampler_output_is_drop_in_for_dataset(self, tensorized_cohort_dir, tmp_path_factory, demo_model):
        labels_dir = tmp_path_factory.mktemp("sampler_labels")

        # Build a tiny events df covering the real train subjects.  Each subject gets an "HR"
        # event at its prediction_time plus a "TEMP" event ~100 days later; the later event
        # pushes max_time comfortably past prediction_time + 30d so the resulting rows are
        # *uncensored* (otherwise the whole batch would be censored and the occurs-loss BCE on
        # an empty mask would NaN out — see issue #30).
        rows = []
        for subj in _E2E_TRAIN_SUBJECTS:
            pt = _E2E_PRED_TIMES[subj]
            rows.append({"subject_id": subj, "time": pt, "code": "HR"})
            rows.append({"subject_id": subj, "time": pt + timedelta(days=5), "code": "HR"})
            rows.append({"subject_id": subj, "time": pt + timedelta(days=100), "code": "TEMP"})
        events_df = pl.DataFrame(rows).cast({"time": pl.Datetime("us")})

        # Index_df: one row per (subject x query), fixed duration.  This is the shape sampler.py
        # would produce for n_tasks=len(_E2E_QUERIES), contexts_per_task=len(_E2E_TRAIN_SUBJECTS)
        # with specific draws; constructing it directly bypasses the sampler randomness.
        index_df = pl.DataFrame(
            [
                {
                    "subject_id": subj,
                    "prediction_time": _E2E_PRED_TIMES[subj],
                    "query": q,
                    "duration_days": 30,
                }
                for subj in _E2E_TRAIN_SUBJECTS
                for q in _E2E_QUERIES
            ]
        ).cast(
            {
                "subject_id": pl.Int64,
                "prediction_time": pl.Datetime("us"),
                "query": pl.Utf8,
                "duration_days": pl.Int64,
            }
        )

        max_time_df = compute_max_time_per_subject(events_df)
        labeled = evaluate_index_df(index_df, events_df, max_time_df)

        # Sanity: schema matches EveryQueryPytorchDataset's expectations exactly.
        assert set(labeled.columns) == {
            "subject_id",
            "prediction_time",
            "boolean_value",
            "occurs",
            "query",
            "duration_days",
        }
        assert labeled.schema["boolean_value"] == pl.Boolean
        assert labeled.schema["occurs"] == pl.Boolean

        labels_fp = Path(labels_dir) / "0.parquet"
        labeled.write_parquet(labels_fp)

        cfg = MEDSTorchDataConfig(
            tensorized_cohort_dir=str(tensorized_cohort_dir),
            task_labels_dir=str(labels_dir),
            max_seq_len=64,
            seq_sampling_strategy="to_end",
            static_inclusion_mode="omit",
            batch_mode="SM",
        )
        dataset = EveryQueryPytorchDataset(cfg, split=train_split)
        assert len(dataset) >= 2

        items = [dataset[0], dataset[1]]
        batch = dataset.collate(items)

        with torch.no_grad():
            loss, out = demo_model._forward(batch)

        assert torch.isfinite(loss).item(), f"Non-finite loss on sampler-produced batch: {loss}"
        assert out.query_embed.shape[0] == batch.batch_size
