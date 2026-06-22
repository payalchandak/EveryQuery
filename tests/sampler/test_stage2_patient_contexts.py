"""Stage 2: ``sample_patient_contexts`` — the vectorized patient-context draw.

Pure function tests: draw-range invariant, last-index reachability, the fixed draw-order
(invariant 5), eligibility/empty/negative guards, with-replacement, plus universe-coverage and
join-key dtype gap tests.
"""

import numpy as np
import polars as pl
import pytest

from every_query.generate_tasks.sample_tasks import sample_patient_contexts
from every_query.utils.seeds import derive_seed


class TestSamplePatientContexts:
    """Unit tests for the redesigned Stage 2 patient-context draw (``sample_patient_contexts``)."""

    def _rng(self, *parts):
        """Build an rng seeded on the ``contexts`` axis, mirroring the redesign caller."""
        return np.random.default_rng(derive_seed(*parts, "contexts"))

    def test_shape_and_columns(self, prediction_time_counts):
        ctx = sample_patient_contexts(
            prediction_time_counts, n=256, min_prediction_times_per_subject=50, rng=self._rng(0)
        )
        assert ctx.height == 256
        assert ctx.columns == ["subject_id", "shard", "prediction_time_index"]

    def test_determinism(self, prediction_time_counts):
        a = sample_patient_contexts(prediction_time_counts, 128, 50, self._rng(42))
        b = sample_patient_contexts(prediction_time_counts, 128, 50, self._rng(42))
        assert a.equals(b)

    def test_different_seeds_differ(self, prediction_time_counts):
        a = sample_patient_contexts(prediction_time_counts, 128, 50, self._rng(1))
        b = sample_patient_contexts(prediction_time_counts, 128, 50, self._rng(2))
        assert not a.equals(b)

    def test_draw_range_invariant(self, prediction_time_counts):
        """Every prediction_time_index lands in ``[min, n_prediction_times)`` for its subject."""
        ctx = sample_patient_contexts(prediction_time_counts, 5000, 50, self._rng(7))
        checked = ctx.join(prediction_time_counts, on="subject_id", how="left")
        assert (checked["prediction_time_index"] >= 50).all()
        assert (checked["prediction_time_index"] < checked["n_prediction_times"]).all()

    def test_last_prediction_time_is_reachable(self, prediction_time_counts):
        """The subject's last index (n_prediction_times - 1) is eligible (high is exclusive)."""
        # subject 20 has the tightest range: [50, 51) ⇒ only index 50 == n_prediction_times - 1.
        ctx = sample_patient_contexts(prediction_time_counts, 5000, 50, self._rng(3))
        sub20 = ctx.filter(pl.col("subject_id") == 20)
        assert sub20.height > 0
        assert (sub20["prediction_time_index"] == 50).all()
        # Across all subjects, some draw reaches the last eligible index.
        checked = ctx.join(prediction_time_counts, on="subject_id", how="left")
        assert (checked["prediction_time_index"] == checked["n_prediction_times"] - 1).any()

    def test_subject_shard_mapping_is_consistent(self, prediction_time_counts):
        """Every (subject_id, shard) pair in the output matches the fixture (no cross-wiring)."""
        ctx = sample_patient_contexts(prediction_time_counts, 2000, 50, self._rng(9))
        pairs = set(zip(ctx["subject_id"].to_list(), ctx["shard"].to_list(), strict=True))
        truth = set(
            zip(
                prediction_time_counts["subject_id"].to_list(),
                prediction_time_counts["shard"].to_list(),
                strict=True,
            )
        )
        # pairs is a subset of truth
        # i.e. all (subject_id,shard) tuple produced by sample_patient_contexts is consistent
        # with stage 0's prediction_time_counts
        assert pairs <= truth

    def test_with_replacement_allows_n_above_universe(self, prediction_time_counts):
        """N may far exceed the eligible universe; sampling is with replacement."""
        n = 10 * prediction_time_counts.height
        ctx = sample_patient_contexts(prediction_time_counts, n, 50, self._rng(0))
        assert ctx.height == n

    def test_n_zero_returns_typed_empty(self, prediction_time_counts):
        ctx = sample_patient_contexts(prediction_time_counts, 0, 50, self._rng(0))
        assert ctx.height == 0
        assert ctx.columns == ["subject_id", "shard", "prediction_time_index"]
        assert ctx.schema["subject_id"] == prediction_time_counts.schema["subject_id"]
        assert ctx.schema["shard"] == prediction_time_counts.schema["shard"]
        assert ctx.schema["prediction_time_index"] == pl.Int64

    def test_rejects_negative_n(self, prediction_time_counts):
        with pytest.raises(ValueError, match="n must be"):
            sample_patient_contexts(prediction_time_counts, -1, 50, self._rng(0))

    def test_rejects_empty_counts_when_n_positive(self):
        empty = pl.DataFrame(
            schema={"subject_id": pl.Int64, "shard": pl.Utf8, "n_prediction_times": pl.Int64}
        )
        with pytest.raises(ValueError, match="empty"):
            sample_patient_contexts(empty, 10, 50, self._rng(0))

    def test_rejects_ineligible_subject(self):
        """A counts row violating Stage 0 eligibility (n <= min) raises (stale-table guard)."""
        bad = pl.DataFrame({"subject_id": [1, 2], "shard": ["0", "0"], "n_prediction_times": [60, 50]})
        with pytest.raises(ValueError, match="stale or corrupt"):
            sample_patient_contexts(bad, 100, 50, self._rng(0))

    def test_subject_axis_consumed_before_time_axis(self, prediction_time_counts):
        """Invariant 5: Step A (subject_idx) is consumed before Step B (prediction_time_index).

        With the same seed, the subject_id/shard columns of an ``n`` draw are the prefix of an
        ``n + k`` draw — the subject axis is drawn fully first, so growing N only appends.
        """
        small = sample_patient_contexts(prediction_time_counts, 32, 50, self._rng(11))
        large = sample_patient_contexts(prediction_time_counts, 64, 50, self._rng(11))
        assert small.select(["subject_id", "shard"]).equals(large.head(32).select(["subject_id", "shard"]))

    # -- Gap tests --------------------------------------------------------------------------------

    def test_subject_draw_covers_eligible_universe(self, prediction_time_counts):
        """Over a large draw, every eligible subject is sampled at least once (with-replacement)."""
        ctx = sample_patient_contexts(prediction_time_counts, 2000, 50, self._rng(13))
        assert set(ctx["subject_id"].to_list()) == set(prediction_time_counts["subject_id"].to_list())

    def test_prediction_time_index_dtype_on_nonempty_path(self, prediction_time_counts):
        """The populated path must also yield Int64 indices so the Stage 3 join key aligns with the Stage 0
        map (the empty path is checked separately in ``test_n_zero_returns_typed_empty``)."""
        ctx = sample_patient_contexts(prediction_time_counts, 10, 50, self._rng(0))
        assert ctx.schema["prediction_time_index"] == pl.Int64
