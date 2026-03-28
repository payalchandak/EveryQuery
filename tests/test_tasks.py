"""Tests for tasks.py: regression tests (new vs reference) and unit tests for optimized functions."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from every_query import tasks, tasks_reference

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_events_df() -> pl.DataFrame:
    """Synthetic events spanning 3 subjects, ~5 codes, many timestamps."""
    codes = ["ICD//A01", "ICD//B02", "ICD//C03", "MED//D04", "MED//E05"]
    rows = []
    base = datetime(2020, 1, 1)
    for subj in range(1, 4):  # 3 subjects
        for i in range(30):  # 30 events each
            rows.append(
                {
                    "subject_id": subj,
                    "time": base + timedelta(days=i * 10 + subj),
                    "code": codes[i % len(codes)],
                }
            )
    df = pl.DataFrame(rows).sort(["subject_id", "time"])
    return df


@pytest.fixture
def query_codes() -> list[str]:
    return ["ICD//A01", "ICD//B02", "ICD//C03", "MED//D04", "MED//E05"]


@pytest.fixture
def min_context() -> int:
    return 5


@pytest.fixture
def test_durations() -> list[int]:
    return [30, 90, 180, 365]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sort_df(df: pl.DataFrame) -> pl.DataFrame:
    """Sort a task label DataFrame for deterministic comparison."""
    return df.sort(["subject_id", "prediction_time"])


def _align_columns(ref: pl.DataFrame, new: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Ensure both DataFrames have the same column order."""
    cols = ref.columns
    return ref.select(cols), new.select(cols)


# ---------------------------------------------------------------------------
# Regression tests: new pipeline vs reference
# ---------------------------------------------------------------------------


class TestCensorRegression:
    """Verify compute_base_prediction_times + derive_censor_for_duration matches reference."""

    def test_censor_matches_reference(self, sample_events_df, min_context, test_durations):
        base_df = tasks.compute_base_prediction_times(sample_events_df, min_context)

        for days in test_durations:
            duration = {"days": days}
            ref = tasks_reference.compute_censor_dataframe(sample_events_df, min_context, duration)
            new = tasks.derive_censor_for_duration(base_df, duration)

            ref_sorted = _sort_df(ref)
            new_sorted = _sort_df(new)
            ref_sorted, new_sorted = _align_columns(ref_sorted, new_sorted)
            assert ref_sorted.equals(new_sorted), f"Censor mismatch at duration={days}"


class TestTaskLabelRegression:
    """Verify full new pipeline matches reference build_task_label_matrix."""

    def test_task_labels_match_reference(self, sample_events_df, query_codes, min_context, test_durations):
        # New pipeline: precompute once
        base_df = tasks.compute_base_prediction_times(sample_events_df, min_context)
        min_deltas = tasks.precompute_min_deltas_wide(sample_events_df, base_df, query_codes)

        for days in test_durations:
            duration = {"days": days}

            # Reference path
            ref_censor = tasks_reference.compute_censor_dataframe(sample_events_df, min_context, duration)
            ref_labels = tasks_reference.build_task_label_matrix(
                sample_events_df, ref_censor, query_codes, duration
            )

            # New path
            new_labels = tasks.build_task_for_duration(min_deltas, query_codes, duration)

            ref_sorted = _sort_df(ref_labels)
            new_sorted = _sort_df(new_labels)
            ref_sorted, new_sorted = _align_columns(ref_sorted, new_sorted)
            assert ref_sorted.equals(new_sorted), f"Task label mismatch at duration={days}"

    def test_old_pipeline_matches_reference(self, sample_events_df, query_codes, min_context, test_durations):
        """Verify the kept old functions in tasks.py still match reference."""
        for days in test_durations:
            duration = {"days": days}

            ref_censor = tasks_reference.compute_censor_dataframe(sample_events_df, min_context, duration)
            ref_labels = tasks_reference.build_task_label_matrix(
                sample_events_df, ref_censor, query_codes, duration
            )

            old_censor = tasks.compute_censor_dataframe(sample_events_df, min_context, duration)
            old_labels = tasks.build_task_label_matrix(sample_events_df, old_censor, query_codes, duration)

            ref_sorted = _sort_df(ref_labels)
            old_sorted = _sort_df(old_labels)
            ref_sorted, old_sorted = _align_columns(ref_sorted, old_sorted)
            assert ref_sorted.equals(old_sorted), f"Old pipeline mismatch at duration={days}"


# ---------------------------------------------------------------------------
# Unit tests for new functions
# ---------------------------------------------------------------------------


class TestComputeBasePredictionTimes:
    def test_schema(self, sample_events_df, min_context):
        base_df = tasks.compute_base_prediction_times(sample_events_df, min_context)
        assert set(base_df.columns) == {"subject_id", "prediction_time", "future_duration"}
        assert base_df["future_duration"].dtype == pl.Duration("us")
        assert base_df.null_count().sum_horizontal().item() == 0

    def test_min_context_filter(self, sample_events_df):
        """With min_context higher than any subject's event count, result should be empty."""
        base_df = tasks.compute_base_prediction_times(sample_events_df, 999)
        assert len(base_df) == 0


class TestDeriveCensorForDuration:
    def test_monotonicity(self, sample_events_df, min_context):
        """Larger duration threshold -> fewer rows are censored (more future data required)."""
        base_df = tasks.compute_base_prediction_times(sample_events_df, min_context)
        counts = []
        for days in [30, 90, 180, 365]:
            censor_df = tasks.derive_censor_for_duration(base_df, {"days": days})
            n_censored = censor_df.filter(pl.col("censored")).height
            counts.append(n_censored)
        # More days -> stricter censoring -> more censored rows
        for i in range(len(counts) - 1):
            assert counts[i] <= counts[i + 1], f"Censoring should be monotonically non-decreasing: {counts}"


class TestPrecomputeMinDeltasWide:
    def test_schema(self, sample_events_df, query_codes, min_context):
        base_df = tasks.compute_base_prediction_times(sample_events_df, min_context)
        min_deltas = tasks.precompute_min_deltas_wide(sample_events_df, base_df, query_codes)

        expected_cols = {"subject_id", "prediction_time", "future_duration", *query_codes}
        assert set(min_deltas.columns) == expected_cols

        for code in query_codes:
            assert min_deltas[code].dtype == pl.Duration("us")

    def test_missing_codes_get_null_columns(self, sample_events_df, min_context):
        """Codes not present in events should still get a column (all nulls)."""
        base_df = tasks.compute_base_prediction_times(sample_events_df, min_context)
        fake_codes = ["FAKE//Z99"]
        min_deltas = tasks.precompute_min_deltas_wide(sample_events_df, base_df, fake_codes)
        assert "FAKE//Z99" in min_deltas.columns
        assert min_deltas["FAKE//Z99"].null_count() == min_deltas.height


class TestSampleDurations:
    def test_reproducibility(self):
        d1 = tasks.sample_durations(100, 1, 731, seed=42)
        d2 = tasks.sample_durations(100, 1, 731, seed=42)
        assert d1 == d2

    def test_different_seeds(self):
        d1 = tasks.sample_durations(100, 1, 731, seed=42)
        d2 = tasks.sample_durations(100, 1, 731, seed=99)
        assert d1 != d2

    def test_range(self):
        durations = tasks.sample_durations(500, 1, 731, seed=42)
        assert all(1 <= d <= 731 for d in durations)

    def test_sorted_unique(self):
        durations = tasks.sample_durations(200, 1, 731, seed=42)
        assert durations == sorted(set(durations))

    def test_log_distribution(self):
        """Log-uniform should produce more small values than large ones when sampled sparsely."""
        # Use a small n relative to range so deduplication doesn't saturate
        durations = tasks.sample_durations(50, 1, 731, seed=42)
        median = durations[len(durations) // 2]
        # Log-uniform median of [1, 731] is exp((ln1+ln731)/2) ~ 27
        # So the median of sampled values should be well below the arithmetic midpoint (366)
        assert median < 366 / 2, f"Median {median} should be well below arithmetic midpoint"
