"""Tests for ``every_query.generate_tasks`` (sampling-first PT task label generator).

Structure:

- ``TestPrimitives`` — unit tests for the pure building blocks.
- ``TestEvaluateIndexDfEdgeCases`` — hand-crafted edge cases for ``evaluate_index_df`` (strict-``>``
  semantics, missing-subject handling).
- ``TestResolveWorkers`` — the Stage 4 worker-pool sizing (``resolve_workers``): SLURM env
  precedence, ``os.cpu_count()`` fallback, and the downward-only ``max_workers`` cap.
- ``TestMainOrchestration`` — end-to-end ``main()`` over a synthetic cohort: runs Stages 0-4 inline,
  writes the per-shard final dataset, asserts the union row count equals the sampling budget, and is
  idempotent on rerun.
- ``TestEndToEndWithDataset`` — smoke test that builds sampler-shaped labels referencing real
  subject IDs from ``tensorized_cohort_dir``, feeds them through ``EveryQueryPytorchDataset``, and
  pushes one collated batch through ``demo_model``.  Verifies schema compatibility with the
  downstream dataloader.
"""

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
from meds import train_split
from meds_torchdata.config import MEDSTorchDataConfig
from omegaconf import OmegaConf

from every_query.data.dataset import EveryQueryPytorchDataset
from every_query.data.schema import TaskQuerySchema
from every_query.generate_tasks import sample_tasks as st
from every_query.generate_tasks.sample_tasks import (
    QueryDistribution,
    QuerySpec,
    _clean_stale_temps,
    build_index,
    build_prediction_times,
    compute_max_time_per_subject,
    default_artifacts_dir,
    evaluate_index_df,
    final_output_path,
    index_path,
    label_one_shard,
    prediction_time_counts_path,
    prediction_times_meta_path,
    prediction_times_path,
    resolve_training_task_paths,
    resolve_workers,
    sample_patient_contexts,
)
from every_query.utils.seeds import derive_seed

# ---------------------------------------------------------------------------
# Shared synthetic fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_events() -> pl.DataFrame:
    """A deterministic events DataFrame: 3 subjects x 30 events x 5 codes, 10d spacing."""
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

    def test_read_query_codes_uses_processed_fallback(self, monkeypatch, tmp_path, synthetic_query_codes):
        processed = tmp_path / "processed"
        metadata_dir = processed / "metadata"
        metadata_dir.mkdir(parents=True)
        pl.DataFrame({"code": synthetic_query_codes}).write_parquet(metadata_dir / "codes.parquet")

        monkeypatch.setenv("PROCESSED", str(processed))

        assert st.read_query_codes(None) == sorted(synthetic_query_codes)

    def test_read_query_codes_requires_processed_when_null(self, monkeypatch):
        monkeypatch.delenv("PROCESSED", raising=False)

        with pytest.raises(ValueError, match="query_codes is null"):
            st.read_query_codes(None)

    def test_read_query_codes_reads_codes_key_yaml(self, tmp_path):
        codes_yaml = tmp_path / "allowed_codes.yaml"
        codes_yaml.write_text("codes:\n  - A\n  - B\n  - A\n")

        assert st.read_query_codes(codes_yaml) == ["A", "B"]

    def test_read_query_codes_reads_flat_yaml_list(self, tmp_path):
        codes_yaml = tmp_path / "allowed_codes.yaml"
        codes_yaml.write_text("- A\n- B\n- A\n")

        assert st.read_query_codes(codes_yaml) == ["A", "B"]

    def test_read_query_codes_rejects_yaml_mapping_without_codes(self, tmp_path):
        codes_yaml = tmp_path / "allowed_codes.yaml"
        codes_yaml.write_text("query_codes:\n  - A\n  - B\n")

        with pytest.raises(ValueError, match="list of codes"):
            st.read_query_codes(codes_yaml)

    def test_read_query_codes_rejects_non_list_yaml(self, tmp_path):
        codes_yaml = tmp_path / "allowed_codes.yaml"
        codes_yaml.write_text("codes: A\n")

        with pytest.raises(ValueError, match="list of codes"):
            st.read_query_codes(codes_yaml)

    def test_read_query_codes_deduplicates_inline_list(self):
        assert st.read_query_codes(["A", "B", "A"]) == ["A", "B"]

    def test_compute_max_time_per_subject(self, synthetic_events):
        max_df = compute_max_time_per_subject(synthetic_events)
        assert set(max_df.columns) == {"subject_id", "max_time"}
        # Every subject has 30 events; max_time is the 30th event time per subject.
        for subj in [1, 2, 3]:
            subj_max = synthetic_events.filter(pl.col("subject_id") == subj)["time"].max()
            row = max_df.filter(pl.col("subject_id") == subj).row(0, named=True)
            assert row["max_time"] == subj_max


# ---------------------------------------------------------------------------
# QueryDistribution (Stage 1)
# ---------------------------------------------------------------------------


class TestQueryDistribution:
    """Unit tests for the redesigned Stage 1 query draw (``QueryDistribution``)."""

    def _rng(self, *parts):
        """Build an rng seeded on the ``queries`` axis, mirroring the redesign caller."""
        return np.random.default_rng(derive_seed(*parts, "queries"))

    def test_query_universe_size_is_derived(self, synthetic_query_codes):
        dist = QueryDistribution(synthetic_query_codes, 1.0, 365.0, "log-uniform")
        assert dist.query_universe_size == len(synthetic_query_codes)

    def test_from_config_builds_expected_dataclass(self, synthetic_query_codes):
        # query_codes is resolved inside from_config via read_query_codes(cfg.query_codes); an inline
        # list is returned order-preserving + deduped, so it matches synthetic_query_codes here.
        cfg = OmegaConf.create(
            {
                "query_codes": synthetic_query_codes,
                "min_duration": 1,
                "max_duration": 731,
                "duration_distribution": "log-uniform",
            }
        )
        dist = QueryDistribution.from_config(cfg)
        assert dist == QueryDistribution(synthetic_query_codes, 1.0, 731.0, "log-uniform")
        assert isinstance(dist.min_duration, float) and isinstance(dist.max_duration, float)

    def test_sample_determinism(self, synthetic_query_codes):
        dist = QueryDistribution(synthetic_query_codes, 1.0, 365.0, "log-uniform")
        a = dist.sample(32, self._rng(42))
        b = dist.sample(32, self._rng(42))
        assert a == b

    def test_sample_different_seeds_differ(self, synthetic_query_codes):
        dist = QueryDistribution(synthetic_query_codes, 1.0, 365.0, "log-uniform")
        a = dist.sample(32, self._rng(1))
        b = dist.sample(32, self._rng(2))
        assert a != b

    @pytest.mark.parametrize("distribution", ["uniform", "log-uniform"])
    def test_sample_respects_bounds(self, synthetic_query_codes, distribution):
        dist = QueryDistribution(synthetic_query_codes, 10.0, 100.0, distribution)
        specs = dist.sample(256, self._rng(7))
        assert all(isinstance(s, QuerySpec) for s in specs)
        assert all(s.code in synthetic_query_codes for s in specs)
        assert all(10.0 <= s.duration_days <= 100.0 for s in specs)

    @pytest.mark.parametrize("distribution", ["uniform", "log-uniform"])
    def test_durations_are_unrounded_floats(self, synthetic_query_codes, distribution):
        """Durations stay float — the intentional divergence from legacy whole-day quantization."""
        dist = QueryDistribution(synthetic_query_codes, 1.0, 731.0, distribution)
        specs = dist.sample(64, self._rng(3))
        assert all(isinstance(s.duration_days, float) for s in specs)
        assert any(s.duration_days != round(s.duration_days) for s in specs)

    def test_sample_zero_returns_empty(self, synthetic_query_codes):
        dist = QueryDistribution(synthetic_query_codes, 1.0, 365.0, "uniform")
        assert dist.sample(0, self._rng(0)) == []

    def test_sample_rejects_negative_num_queries(self, synthetic_query_codes):
        dist = QueryDistribution(synthetic_query_codes, 1.0, 365.0, "uniform")
        with pytest.raises(ValueError, match="num_queries"):
            dist.sample(-1, self._rng(0))

    def test_rejects_empty_query_codes(self):
        with pytest.raises(ValueError, match="query_codes"):
            QueryDistribution([], 1.0, 365.0, "uniform")

    def test_rejects_nonpositive_min_duration(self, synthetic_query_codes):
        with pytest.raises(ValueError, match="min_duration"):
            QueryDistribution(synthetic_query_codes, 0.0, 365.0, "uniform")

    def test_rejects_inverted_bounds(self, synthetic_query_codes):
        with pytest.raises(ValueError, match="max_duration"):
            QueryDistribution(synthetic_query_codes, 100.0, 50.0, "uniform")

    def test_rejects_unknown_distribution(self, synthetic_query_codes):
        with pytest.raises(ValueError, match="duration_distribution"):
            QueryDistribution(synthetic_query_codes, 1.0, 365.0, "normal")


# ---------------------------------------------------------------------------
# sample_patient_contexts (Stage 2)
# ---------------------------------------------------------------------------


@pytest.fixture
def prediction_time_counts() -> pl.DataFrame:
    """A small Stage 0 ``_prediction_time_counts`` table, sorted by ``subject_id``.

    Row position is ``subject_idx``; subjects span two shards with varying ``n_prediction_times``.
    """
    return pl.DataFrame(
        {
            "subject_id": [10, 20, 30, 40, 50],
            "shard": ["0", "0", "0", "1", "1"],
            "n_prediction_times": [60, 51, 200, 80, 120],
        }
    )


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


# ---------------------------------------------------------------------------
# evaluate_index_df edge cases
# ---------------------------------------------------------------------------


class TestEvaluateIndexDfEdgeCases:
    """Hand-crafted edge-case tests for ``evaluate_index_df``."""

    def test_event_exactly_at_prediction_time_is_excluded(self):
        """``prediction_time == event_time`` must not count as an occurrence (strict ``>``).

        Under the collapsed ``TaskQuerySchema`` label: non-censored row with an event in
        the window → ``boolean_value = True``; non-censored row without → ``False``.
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
        max_time_df = compute_max_time_per_subject(events)
        result = evaluate_index_df(index_df, events, max_time_df)
        assert result["boolean_value"].to_list() == [True]

        # Now with duration=0: no window → False (and not censored because 2020-01-03
        # + 0 days = 2020-01-03 which is <= 2021-01-01).
        index_df_zero = index_df.with_columns(pl.lit(0, dtype=pl.Int64).alias("duration_days"))
        result_zero = evaluate_index_df(index_df_zero, events, max_time_df)
        assert result_zero["boolean_value"].to_list() == [False]

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

        max_time_df = compute_max_time_per_subject(events)
        with caplog.at_level("WARNING", logger="every_query.generate_tasks.sample_tasks"):
            result = evaluate_index_df(index_df, events, max_time_df)

        # Subject 2 (unknown) → censored → null boolean_value.
        unknown = result.filter(pl.col("subject_id") == 2)
        assert unknown["boolean_value"].to_list() == [None]

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
        # event is 2020-01-02, which is strictly within the window → boolean_value=True.
        assert result["boolean_value"].to_list() == [True]


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
# Synthetic-cohort helper (shared by the end-to-end main() orchestration test)
# ---------------------------------------------------------------------------


def _write_fake_cohort(
    tmp_path: Path,
    events: pl.DataFrame,
    query_codes: list[str],
    split: str = "train",
    shard_name: str = "0",
) -> tuple[Path, Path]:
    """Write a minimal MEDS-shaped cohort the redesigned ``main()`` can read from.

    Mirrors the split layout of ``.env.example``: ``$INTERMEDIATE`` and ``$PROCESSED`` are sibling
    directories rather than the same root, so the cohort this helper writes uses:

        {data_dir}/data/{split}/{shard_name}.parquet     <- event shards ($INTERMEDIATE)
        {processed_dir}/metadata/codes.parquet           <- query universe ($PROCESSED)

    Returns ``(data_dir, processed_dir)`` as two distinct paths so tests can exercise the
    separation.
    """
    data_dir = tmp_path / "intermediate"
    split_dir = data_dir / "data" / split
    split_dir.mkdir(parents=True, exist_ok=True)
    events.write_parquet(split_dir / f"{shard_name}.parquet")

    processed_dir = tmp_path / "processed"
    metadata_dir = processed_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"code": query_codes}).write_parquet(metadata_dir / "codes.parquet")
    return data_dir, processed_dir


# End-to-end: sampler output → EveryQueryPytorchDataset → model forward
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# main() path resolution (env fallback + dotenv)
# ---------------------------------------------------------------------------


class TestResolvePath:
    """``_resolve_path`` is how ``main()`` threads explicit path roots / env fallbacks / required.

    Directly exercising the helper lets us pin the fallback matrix without spinning up a full Hydra run per
    case.
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
    """Sanity-check that sampler output is a drop-in replacement for the dataset's task-labels contract.

    We don't run the full ``main()`` pipeline here because its shard-reading path expects string-coded
    events from an early preprocessing stage that this fixture does not expose (the tensorized cohort
    only materializes ``normalization/*.parquet`` with *integer-coded* events downstream of
    tokenization).  ``TestMainOrchestration`` already exercises the full pipeline on a synthetic cohort.

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
                # Float32 to match TaskQuerySchema.duration_days.
                "duration_days": pl.Float32,
            }
        )

        max_time_df = compute_max_time_per_subject(events_df)
        labeled = evaluate_index_df(index_df, events_df, max_time_df)

        # Sanity: labeled output conforms to TaskQuerySchema — same check the Stage 4 write path
        # (``label_one_shard``) does, exercised here at the per-function boundary.
        TaskQuerySchema.validate(labeled.to_arrow())

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


# ---------------------------------------------------------------------------
# Redesign config & path resolution (issue #203)
# ---------------------------------------------------------------------------


class TestResolveTrainingTaskPaths:
    """Unit tests for the redesigned sampler's env-only path-resolution surface.

    The three roots are not config keys — they come from env vars only (``$INTERMEDIATE`` and
    ``$TRAINING_TASKS_DIR``, with the artifacts root always the sibling default).  Pure path
    functions (no file I/O, no dir creation), so they exercise the env-var matrix without a Hydra
    run.
    """

    @pytest.fixture(autouse=True)
    def _clear_env(self, monkeypatch):
        # Isolate from the developer's own .env / shell so assertions are deterministic.
        monkeypatch.delenv("INTERMEDIATE", raising=False)
        monkeypatch.delenv("TRAINING_TASKS_DIR", raising=False)

    def test_default_artifacts_dir_is_a_sibling(self):
        assert default_artifacts_dir(Path("/x/y/tasks")) == Path("/x/y/tasks_artifacts")
        # Sibling, never nested under the final-output root (invariant 7).
        assert default_artifacts_dir(Path("/x/y/tasks")).parent == Path("/x/y/tasks").parent

    def test_roots_resolve_from_env_vars(self, monkeypatch):
        monkeypatch.setenv("INTERMEDIATE", "/env/data")
        monkeypatch.setenv("TRAINING_TASKS_DIR", "/env/tasks")
        data, tasks, arts = resolve_training_task_paths()
        assert data == Path("/env/data")
        assert tasks == Path("/env/tasks")
        # Artifacts dir is NOT its own env var; it derives from training_tasks_dir.
        assert arts == Path("/env/tasks_artifacts")

    def test_artifacts_dir_has_no_env_var(self, monkeypatch):
        # Even if someone exports it, the resolver ignores it and uses the sibling default.
        monkeypatch.setenv("INTERMEDIATE", "/env/data")
        monkeypatch.setenv("TRAINING_TASKS_DIR", "/env/tasks")
        monkeypatch.setenv("TRAINING_TASK_ARTIFACTS_DIR", "/env/should_be_ignored")
        _, _, arts = resolve_training_task_paths()
        assert arts == Path("/env/tasks_artifacts")

    def test_missing_path_to_data_raises(self, monkeypatch):
        monkeypatch.delenv("INTERMEDIATE", raising=False)
        monkeypatch.setenv("TRAINING_TASKS_DIR", "/env/tasks")
        with pytest.raises(ValueError, match="data_dir"):
            resolve_training_task_paths()

    def test_missing_training_tasks_dir_raises(self, monkeypatch):
        monkeypatch.setenv("INTERMEDIATE", "/env/data")
        monkeypatch.delenv("TRAINING_TASKS_DIR", raising=False)
        with pytest.raises(ValueError, match="out_dir"):
            resolve_training_task_paths()

    def test_cfg_overrides_take_precedence_over_env(self, monkeypatch):
        # override > env: cfg.data_dir / cfg.out_dir win even when the env vars are set.
        monkeypatch.setenv("INTERMEDIATE", "/env/data")
        monkeypatch.setenv("TRAINING_TASKS_DIR", "/env/tasks")
        cfg = OmegaConf.create({"data_dir": "/cli/data", "out_dir": "/cli/tasks"})
        data, tasks, arts = resolve_training_task_paths(cfg)
        assert data == Path("/cli/data")
        assert tasks == Path("/cli/tasks")
        assert arts == Path("/cli/tasks_artifacts")

    def test_null_cfg_overrides_fall_back_to_env(self, monkeypatch):
        # null override => env fallback (the cluster/.env contract is preserved).
        monkeypatch.setenv("INTERMEDIATE", "/env/data")
        monkeypatch.setenv("TRAINING_TASKS_DIR", "/env/tasks")
        cfg = OmegaConf.create({"data_dir": None, "out_dir": None})
        data, tasks, _ = resolve_training_task_paths(cfg)
        assert data == Path("/env/data")
        assert tasks == Path("/env/tasks")


class TestRedesignConfigFile:
    """The new ``sample_training_tasks_config.yaml`` exposes the spec's keys with stated defaults."""

    @staticmethod
    def _load():
        from importlib.resources import files

        from omegaconf import OmegaConf

        path = files("every_query") / "generate_tasks" / "configs" / "sample_training_tasks_config.yaml"
        return OmegaConf.load(str(path))

    def test_defaults_match_spec(self):
        cfg = self._load()
        assert cfg.min_prediction_times_per_subject == 50
        assert cfg.max_workers is None
        assert cfg.duration_distribution == "log-uniform"

    def test_path_roots_are_optional_config_keys(self):
        cfg = self._load()
        # The two input roots are optional CLI overrides defaulting to null (override > env > raise;
        # see resolve_training_task_paths).
        assert "data_dir" in cfg and cfg.data_dir is None
        assert "out_dir" in cfg and cfg.out_dir is None
        # The internal artifacts root is still derived (sibling of out_dir), never a config key.
        for derived_key in ("path_to_data", "training_tasks_dir", "training_task_artifacts_dir"):
            assert derived_key not in cfg

    def test_legacy_keys_are_gone(self):
        cfg = self._load()
        # Superseded knobs must not leak into the redesign surface.
        for legacy in ("min_context_per_subject", "num_workers", "num_codes", "n_tasks"):
            assert legacy not in cfg


class TestArtifactLayout:
    """Issue #204: the two-root, never-nested on-disk layout every stage resolves against.

    Pure path functions (no I/O), so they map roots+split+shard to the spec's fixed locations.  The
    contract under test: the final-output root holds *only* ``{shard}.parquet`` while every
    intermediate lands under the disjoint artifacts root with its ``_``-prefixed name (invariant 7).
    """

    TASKS = Path("/x/tasks")
    ARTS = Path("/x/tasks_artifacts")

    def test_final_output_path(self):
        assert final_output_path(self.TASKS, "train", "0") == Path("/x/tasks/train/0.parquet")

    def test_prediction_time_counts_path(self):
        assert prediction_time_counts_path(self.ARTS, "train") == Path(
            "/x/tasks_artifacts/train/_prediction_time_counts.parquet"
        )

    def test_prediction_times_path(self):
        assert prediction_times_path(self.ARTS, "train", "0") == Path(
            "/x/tasks_artifacts/train/_prediction_times/0.parquet"
        )

    def test_index_path(self):
        assert index_path(self.ARTS, "train", "0") == Path("/x/tasks_artifacts/train/_index/0.parquet")

    def test_final_root_split_dir_holds_only_shard_parquets(self):
        # The final-output split dir must be directly glob-consumable as `{split}/*.parquet`: no
        # `_`-prefixed entries (those all live under the artifacts root). Asserting on the name keeps
        # the "nothing but {shard}.parquet" invariant auditable.
        name = final_output_path(self.TASKS, "train", "0").name
        assert name == "0.parquet"
        assert not name.startswith("_")

    def test_every_intermediate_is_under_the_artifacts_root_not_the_dataset(self):
        # Invariant 7: no intermediate may resolve inside the final-output root.
        intermediates = [
            prediction_time_counts_path(self.ARTS, "train"),
            prediction_times_path(self.ARTS, "train", "0"),
            index_path(self.ARTS, "train", "0"),
        ]
        for p in intermediates:
            assert self.ARTS in p.parents
            assert self.TASKS not in p.parents

    def test_default_artifacts_root_is_disjoint_so_rm_rf_cannot_touch_dataset(self):
        # The sibling default (from #203) is what makes `rm -rf {artifacts}` safe: the artifacts root
        # is neither equal to nor nested under the dataset root, and vice versa.
        arts = default_artifacts_dir(self.TASKS)
        assert arts != self.TASKS
        assert self.TASKS not in arts.parents
        assert arts not in self.TASKS.parents


def _subject_events(subject_id: int, n_times: int, *, base: datetime, dups: int = 1) -> pl.DataFrame:
    """``n_times`` distinct ``(subject_id, time)`` rows, each emitted ``dups`` times (same time, diff code) so
    Stage 0's distinct-time dedup is exercised.

    Times are 1 day apart starting at ``base``.
    """
    rows = [
        {"subject_id": subject_id, "time": base + timedelta(days=i), "code": f"ICD//{d:02d}"}
        for i in range(n_times)
        for d in range(dups)
    ]
    return pl.DataFrame(rows)


def _write_split_shards(
    tmp_path: Path, shard_to_events: dict[str, pl.DataFrame], split: str = "train"
) -> Path:
    """Write ``{shard: events}`` to ``tmp_path/intermediate/data/{split}/{shard}.parquet``; return the
    ``path_to_data`` root."""
    data_dir = tmp_path / "intermediate"
    split_dir = data_dir / "data" / split
    split_dir.mkdir(parents=True, exist_ok=True)
    for shard, df in shard_to_events.items():
        df.write_parquet(split_dir / f"{shard}.parquet")
    return data_dir


class TestStage0:
    """Issue #205: Stage 0 builds + caches the canonical prediction-time map and counts summary.

    ``min_prediction_times_per_subject=2`` ⇒ eligibility is ``n_prediction_times >= 3``. The fixture
    has subject 1 (5 distinct times, eligible), subject 2 (3, eligible at the boundary), and subject 3
    (2, dropped) — so the ``+1`` boundary and the eligibility filter are both exercised.
    """

    BASE = datetime(2020, 1, 1)
    MIN = 2

    @pytest.fixture
    def path_to_data(self, tmp_path: Path) -> Path:
        events = pl.concat(
            [
                _subject_events(1, 5, base=self.BASE, dups=2),  # duplicate (subject,time) rows
                _subject_events(2, 3, base=self.BASE),
                _subject_events(3, 2, base=self.BASE),
            ]
        )
        return _write_split_shards(tmp_path, {"0": events})

    @pytest.fixture
    def artifacts_dir(self, tmp_path: Path) -> Path:
        return tmp_path / "tasks_artifacts"

    def test_returns_patient_universe_size(self, path_to_data, artifacts_dir):
        # Subjects 1 and 2 are eligible (>= 3 distinct times); subject 3 is not.
        n = build_prediction_times(path_to_data, artifacts_dir, "train", self.MIN)
        assert n == 2

    def test_index_is_gapless_and_time_ordered_with_dedup(self, path_to_data, artifacts_dir):
        build_prediction_times(path_to_data, artifacts_dir, "train", self.MIN)
        pmap = pl.read_parquet(prediction_times_path(artifacts_dir, "train", "0"))
        assert pmap.columns == ["subject_id", "prediction_time_index", "time"]

        s1 = pmap.filter(pl.col("subject_id") == 1).sort("time")
        # 5 distinct times despite dups=2 duplicate rows; gapless [0, 5).
        assert s1["prediction_time_index"].to_list() == [0, 1, 2, 3, 4]
        assert s1["time"].to_list() == sorted(s1["time"].to_list())

    def test_eligibility_drops_subjects_below_min_plus_one(self, path_to_data, artifacts_dir):
        build_prediction_times(path_to_data, artifacts_dir, "train", self.MIN)
        counts = pl.read_parquet(prediction_time_counts_path(artifacts_dir, "train"))
        pmap = pl.read_parquet(prediction_times_path(artifacts_dir, "train", "0"))
        # Subject 3 (2 distinct times < MIN+1=3) excluded from both artifacts; subject 2 kept at boundary.
        assert set(counts["subject_id"].to_list()) == {1, 2}
        assert 3 not in pmap["subject_id"].to_list()

    def test_counts_schema_and_subject_idx_ordering(self, path_to_data, artifacts_dir):
        build_prediction_times(path_to_data, artifacts_dir, "train", self.MIN)
        counts = pl.read_parquet(prediction_time_counts_path(artifacts_dir, "train"))
        assert counts.columns == ["subject_id", "shard", "n_prediction_times"]
        # Row position is subject_idx, so the table must be ascending by subject_id.
        assert counts["subject_id"].to_list() == sorted(counts["subject_id"].to_list())
        assert dict(zip(counts["subject_id"], counts["n_prediction_times"], strict=True)) == {1: 5, 2: 3}

    def test_row_count_identity(self, path_to_data, artifacts_dir):
        n = build_prediction_times(path_to_data, artifacts_dir, "train", self.MIN)
        counts = pl.read_parquet(prediction_time_counts_path(artifacts_dir, "train"))
        pmap = pl.read_parquet(prediction_times_path(artifacts_dir, "train", "0"))
        assert int(counts["n_prediction_times"].sum()) == pmap.height
        assert counts.height == n == pmap["subject_id"].n_unique()

    def test_subject_spanning_shards_raises(self, tmp_path, artifacts_dir):
        # Subject 1's distinct times split across two shards ⇒ invariant 4 violation.
        path_to_data = _write_split_shards(
            tmp_path,
            {
                "0": _subject_events(1, 3, base=self.BASE),
                "1": _subject_events(1, 3, base=self.BASE + timedelta(days=100)),
            },
        )
        with pytest.raises(ValueError, match=r"span multiple shards.*1"):
            build_prediction_times(path_to_data, artifacts_dir, "train", self.MIN)

    def test_partitioned_by_shard(self, tmp_path, artifacts_dir):
        path_to_data = _write_split_shards(
            tmp_path,
            {
                "0": _subject_events(1, 4, base=self.BASE),
                "1": _subject_events(2, 4, base=self.BASE),
            },
        )
        build_prediction_times(path_to_data, artifacts_dir, "train", self.MIN)
        # One map partition per shard, each holding only its own subject.
        m0 = pl.read_parquet(prediction_times_path(artifacts_dir, "train", "0"))
        m1 = pl.read_parquet(prediction_times_path(artifacts_dir, "train", "1"))
        assert m0["subject_id"].unique().to_list() == [1]
        assert m1["subject_id"].unique().to_list() == [2]

    def test_no_shards_raises(self, tmp_path, artifacts_dir):
        (tmp_path / "intermediate" / "data" / "train").mkdir(parents=True)
        with pytest.raises(FileNotFoundError):
            build_prediction_times(tmp_path / "intermediate", artifacts_dir, "train", self.MIN)

    def test_cache_reused_without_rescanning(self, path_to_data, artifacts_dir, monkeypatch):
        build_prediction_times(path_to_data, artifacts_dir, "train", self.MIN)

        calls = {"n": 0}
        real = st._read_prediction_time_shard

        def _spy(*args, **kwargs):
            calls["n"] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(st, "_read_prediction_time_shard", _spy)
        n = build_prediction_times(path_to_data, artifacts_dir, "train", self.MIN)
        assert n == 2
        assert calls["n"] == 0  # cache hit: no shard scan

    def test_changing_min_invalidates_cache(self, path_to_data, artifacts_dir):
        build_prediction_times(path_to_data, artifacts_dir, "train", self.MIN)
        # MIN=4 ⇒ eligibility n>=5: subject 2 (3 times) now drops, only subject 1 remains.
        n = build_prediction_times(path_to_data, artifacts_dir, "train", 4)
        assert n == 1
        counts = pl.read_parquet(prediction_time_counts_path(artifacts_dir, "train"))
        assert counts["subject_id"].to_list() == [1]
        meta = json.loads(prediction_times_meta_path(artifacts_dir, "train").read_text())
        assert meta["min_prediction_times_per_subject"] == 4

    def test_overwrite_forces_rebuild(self, path_to_data, artifacts_dir, monkeypatch):
        build_prediction_times(path_to_data, artifacts_dir, "train", self.MIN)

        calls = {"n": 0}
        real = st._read_prediction_time_shard

        def _spy(*args, **kwargs):
            calls["n"] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(st, "_read_prediction_time_shard", _spy)
        build_prediction_times(path_to_data, artifacts_dir, "train", self.MIN, overwrite=True)
        assert calls["n"] == 1  # rebuilt despite a valid cache

    def test_null_time_rows_are_not_prediction_times(self, tmp_path, artifacts_dir):
        # Subject 1: 3 real distinct times + one null-time static row (e.g. MEDS demographics).
        # The null must be dropped, not claim prediction_time_index=0 nor inflate n_prediction_times.
        events = pl.concat(
            [
                _subject_events(1, 3, base=self.BASE),
                pl.DataFrame({"subject_id": [1], "time": [None], "code": ["DEMO//SEX"]}),
            ],
            how="diagonal",
        )
        path_to_data = _write_split_shards(tmp_path, {"0": events})
        build_prediction_times(path_to_data, artifacts_dir, "train", self.MIN)

        pmap = pl.read_parquet(prediction_times_path(artifacts_dir, "train", "0"))
        s1 = pmap.filter(pl.col("subject_id") == 1).sort("prediction_time_index")
        assert s1["prediction_time_index"].to_list() == [0, 1, 2]
        assert s1["time"].null_count() == 0

        counts = pl.read_parquet(prediction_time_counts_path(artifacts_dir, "train"))
        assert dict(zip(counts["subject_id"], counts["n_prediction_times"], strict=True)) == {1: 3}


class TestStage3:
    """Issue #208: Stage 3 zips queries with contexts, resolves prediction_time, writes _index/.

    Fixture: two shards — shard "0" has subjects 1 (5 times) and 2 (4 times), shard "1" has
    subject 3 (4 times).  ``min_prediction_times_per_subject=2`` so all three are eligible
    (``n_prediction_times >= 3``).
    """

    BASE = datetime(2020, 1, 1)
    MIN = 2

    @pytest.fixture
    def stage0_env(self, tmp_path: Path) -> tuple[Path, Path]:
        """Run Stage 0 and return ``(path_to_data, artifacts_dir)``."""
        path_to_data = _write_split_shards(
            tmp_path,
            {
                "0": pl.concat(
                    [
                        _subject_events(1, 5, base=self.BASE),
                        _subject_events(2, 4, base=self.BASE),
                    ]
                ),
                "1": _subject_events(3, 4, base=self.BASE),
            },
        )
        artifacts_dir = tmp_path / "tasks_artifacts"
        build_prediction_times(path_to_data, artifacts_dir, "train", self.MIN)
        return path_to_data, artifacts_dir

    def _make_contexts(self, subject_ids, shards, indices):
        return pl.DataFrame(
            {
                "subject_id": subject_ids,
                "shard": shards,
                "prediction_time_index": pl.Series(indices, dtype=pl.Int64),
            }
        )

    def test_basic_columns_and_row_count(self, stage0_env):
        _, artifacts_dir = stage0_env
        queries = [QuerySpec("ICD//A", 30.5), QuerySpec("ICD//B", 60.0)]
        contexts = self._make_contexts(
            [1, 1, 1, 1],
            ["0", "0", "0", "0"],
            [2, 3, 2, 4],
        )
        build_index(queries, contexts, artifacts_dir, "train", num_contexts_per_query=2)

        idx = pl.read_parquet(index_path(artifacts_dir, "train", "0"))
        assert idx.height == 4
        assert idx.columns == ["subject_id", "prediction_time", "query", "duration_days"]
        assert idx.schema["prediction_time"] == pl.Datetime("us")
        assert idx.schema["duration_days"] == pl.Float32
        assert idx["prediction_time"].null_count() == 0

    def test_multi_shard_partitioning(self, stage0_env):
        _, artifacts_dir = stage0_env
        queries = [QuerySpec("ICD//X", 10.0)]
        contexts = self._make_contexts(
            [1, 3],
            ["0", "1"],
            [3, 3],
        )
        build_index(queries, contexts, artifacts_dir, "train", num_contexts_per_query=2)

        idx0 = pl.read_parquet(index_path(artifacts_dir, "train", "0"))
        idx1 = pl.read_parquet(index_path(artifacts_dir, "train", "1"))
        assert idx0.height == 1
        assert idx1.height == 1
        assert idx0["subject_id"].to_list() == [1]
        assert idx1["subject_id"].to_list() == [3]

    def test_prediction_time_resolution_correctness(self, stage0_env):
        _, artifacts_dir = stage0_env
        pt_map = pl.read_parquet(prediction_times_path(artifacts_dir, "train", "0"))
        s1_times = pt_map.filter(pl.col("subject_id") == 1).sort("prediction_time_index")["time"].to_list()

        queries = [QuerySpec("ICD//A", 7.0)]
        contexts = self._make_contexts([1, 1], ["0", "0"], [2, 4])
        build_index(queries, contexts, artifacts_dir, "train", num_contexts_per_query=2)

        idx = pl.read_parquet(index_path(artifacts_dir, "train", "0"))
        resolved = idx.sort("prediction_time")["prediction_time"].to_list()
        assert resolved == sorted([s1_times[2], s1_times[4]])

    def test_query_zip_assignment(self, stage0_env):
        _, artifacts_dir = stage0_env
        queries = [QuerySpec("ICD//FIRST", 10.0), QuerySpec("ICD//SECOND", 20.0)]
        contexts = self._make_contexts(
            [1, 1, 2, 2],
            ["0", "0", "0", "0"],
            [3, 4, 3, 3],
        )
        build_index(queries, contexts, artifacts_dir, "train", num_contexts_per_query=2)

        idx = pl.read_parquet(index_path(artifacts_dir, "train", "0"))
        assert idx["query"].to_list() == ["ICD//FIRST", "ICD//FIRST", "ICD//SECOND", "ICD//SECOND"]
        assert idx["duration_days"].to_list() == pytest.approx([10.0, 10.0, 20.0, 20.0])

    def test_null_assertion_on_invalid_index(self, stage0_env):
        _, artifacts_dir = stage0_env
        queries = [QuerySpec("ICD//A", 5.0)]
        contexts = self._make_contexts([1], ["0"], [999])
        with pytest.raises(ValueError, match="null prediction_time"):
            build_index(queries, contexts, artifacts_dir, "train", num_contexts_per_query=1)

    def test_null_error_includes_offending_sample(self, stage0_env):
        _, artifacts_dir = stage0_env
        queries = [QuerySpec("ICD//A", 5.0)]
        contexts = self._make_contexts([1], ["0"], [999])
        with pytest.raises(ValueError, match=r"Sample of offending rows.*999"):
            build_index(queries, contexts, artifacts_dir, "train", num_contexts_per_query=1)

    def test_join_key_dtype_mismatch_raises(self, stage0_env):
        _, artifacts_dir = stage0_env
        queries = [QuerySpec("ICD//A", 5.0)]
        # ``prediction_time_index`` as UInt32 mismatches the Stage 0 map's Int64 — a silent
        # all-null join without the dtype guard.
        contexts = pl.DataFrame(
            {
                "subject_id": [1],
                "shard": ["0"],
                "prediction_time_index": pl.Series([3], dtype=pl.UInt32),
            }
        )
        with pytest.raises(ValueError, match="dtype mismatch"):
            build_index(queries, contexts, artifacts_dir, "train", num_contexts_per_query=1)

    def test_empty_queries(self, stage0_env):
        _, artifacts_dir = stage0_env
        contexts = pl.DataFrame(
            schema={"subject_id": pl.Int64, "shard": pl.Utf8, "prediction_time_index": pl.Int64}
        )
        build_index([], contexts, artifacts_dir, "train", num_contexts_per_query=3)
        assert not index_path(artifacts_dir, "train", "0").exists()

    def test_height_mismatch_raises(self, stage0_env):
        _, artifacts_dir = stage0_env
        queries = [QuerySpec("ICD//A", 5.0), QuerySpec("ICD//B", 10.0)]
        contexts = self._make_contexts([1, 1, 1], ["0", "0", "0"], [2, 3, 4])
        with pytest.raises(ValueError, match="num_contexts_per_query"):
            build_index(queries, contexts, artifacts_dir, "train", num_contexts_per_query=2)

    def test_stale_index_dir_cleaned(self, stage0_env):
        _, artifacts_dir = stage0_env
        stale = index_path(artifacts_dir, "train", "stale_shard")
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_bytes(b"garbage")

        queries = [QuerySpec("ICD//A", 5.0)]
        contexts = self._make_contexts([1], ["0"], [3])
        build_index(queries, contexts, artifacts_dir, "train", num_contexts_per_query=1)

        assert not index_path(artifacts_dir, "train", "stale_shard").exists()
        assert index_path(artifacts_dir, "train", "0").exists()

    def test_duration_days_is_float32(self, stage0_env):
        _, artifacts_dir = stage0_env
        queries = [QuerySpec("ICD//A", 30.123456789)]
        contexts = self._make_contexts([1], ["0"], [3])
        build_index(queries, contexts, artifacts_dir, "train", num_contexts_per_query=1)

        idx = pl.read_parquet(index_path(artifacts_dir, "train", "0"))
        assert idx.schema["duration_days"] == pl.Float32


# ---------------------------------------------------------------------------
# Stage 4: per-shard labeling worker
# ---------------------------------------------------------------------------


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
    """Issue #209: Stage 4 per-shard labeling worker."""

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

    def test_skip_on_success(self, tmp_path):
        events = self._events()
        index_df = self._index()
        index_dir, data_dir, out_dir = _make_shard_fixture(tmp_path, events, index_df)

        sentinel = b"sentinel"
        (out_dir / "0.parquet").write_bytes(sentinel)

        _shard, status = label_one_shard("0", index_dir, data_dir, out_dir, overwrite=False)
        assert status == "skipped"
        assert (out_dir / "0.parquet").read_bytes() == sentinel

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
        """Three-valued label: True (event in window), False (no event, fully observed), null (censored)."""
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
        (tmp_path / ".1.parquet.tmp.333").write_bytes(b"x")  # different shard

        removed = _clean_stale_temps(tmp_path, "0")
        assert removed == 2
        assert not (tmp_path / ".0.parquet.tmp.111").exists()
        assert not (tmp_path / ".0.parquet.tmp.222").exists()
        assert (tmp_path / ".1.parquet.tmp.333").exists()  # untouched

    def test_no_temps_returns_zero(self, tmp_path):
        assert _clean_stale_temps(tmp_path, "0") == 0


# ---------------------------------------------------------------------------
# Orchestration & parallelism (issue #210)
# ---------------------------------------------------------------------------


class TestResolveWorkers:
    """Stage 4 worker-pool sizing: SLURM env precedence, cpu_count fallback, downward-only cap."""

    @pytest.fixture(autouse=True)
    def _clear_slurm_env(self, monkeypatch):
        monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
        monkeypatch.delenv("SLURM_CPUS_ON_NODE", raising=False)

    def test_prefers_cpus_per_task_over_cpus_on_node(self, monkeypatch):
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
        monkeypatch.setenv("SLURM_CPUS_ON_NODE", "8")
        assert resolve_workers() == 4

    def test_falls_back_to_cpus_on_node(self, monkeypatch):
        monkeypatch.setenv("SLURM_CPUS_ON_NODE", "8")
        assert resolve_workers() == 8

    def test_falls_back_to_os_cpu_count_without_slurm(self):
        assert resolve_workers() == (os.cpu_count() or 1)

    def test_max_workers_caps_downward_only(self, monkeypatch):
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
        assert resolve_workers(2) == 2  # lower cap applies
        assert resolve_workers(16) == 8  # higher cap is ignored (never exceed cores)

    def test_max_workers_none_returns_cores(self, monkeypatch):
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "6")
        assert resolve_workers(None) == 6


class TestMainOrchestration:
    """End-to-end ``run()``: Stages 0-4 inline over a synthetic single-shard cohort.

    Exercises the issue #210 wiring — Stage 3's full arg set, the Stage 4 ProcessPoolExecutor
    fan-out (``max_workers=1`` keeps the pool light), the final row-count assertion, and
    idempotent reruns (atomic-write skip path).
    """

    def _cfg(self, query_codes, *, num_queries=8, num_contexts_per_query=2, overwrite=False):
        return OmegaConf.create(
            {
                "num_queries": num_queries,
                "num_contexts_per_query": num_contexts_per_query,
                "min_prediction_times_per_subject": 5,
                "max_workers": 1,
                "query_codes": list(query_codes),
                "min_duration": 1,
                "max_duration": 365,
                "duration_distribution": "log-uniform",
                "split": "train",
                "seed": 1,
                "overwrite": overwrite,
            }
        )

    def _run_env(self, monkeypatch, tmp_path, synthetic_events, synthetic_query_codes):
        """Write a synthetic cohort and point the env-only path roots at it."""
        data_dir, _processed = _write_fake_cohort(tmp_path, synthetic_events, synthetic_query_codes)
        tasks_dir = tmp_path / "training_tasks"
        monkeypatch.setenv("INTERMEDIATE", str(data_dir))
        monkeypatch.setenv("TRAINING_TASKS_DIR", str(tasks_dir))
        return tasks_dir

    def test_run_writes_final_dataset_with_expected_rows(
        self, monkeypatch, tmp_path, synthetic_events, synthetic_query_codes
    ):
        tasks_dir = self._run_env(monkeypatch, tmp_path, synthetic_events, synthetic_query_codes)
        cfg = self._cfg(synthetic_query_codes)

        st.run(cfg)

        # Final-output root holds only {shard}.parquet under the split (invariant 7).
        out_dir = tasks_dir / "train"
        shard_files = sorted(out_dir.glob("*.parquet"))
        assert shard_files, "Stage 4 wrote no shard outputs"

        union = pl.concat([pl.read_parquet(f) for f in shard_files])
        assert union.height == cfg.num_queries * cfg.num_contexts_per_query
        assert set(union.columns) == {
            "subject_id",
            "prediction_time",
            "query",
            "duration_days",
            "boolean_value",
        }
        # Output conforms to the downstream schema.
        TaskQuerySchema.validate(union.to_arrow())

    def test_run_rejects_row_count_mismatch_is_not_triggered(
        self, monkeypatch, tmp_path, synthetic_events, synthetic_query_codes
    ):
        """A clean run never trips the union-row-count guard (it equals the sampling budget)."""
        self._run_env(monkeypatch, tmp_path, synthetic_events, synthetic_query_codes)
        # Would raise ValueError if the wired row count diverged from num_queries * num_contexts.
        st.run(self._cfg(synthetic_query_codes, num_queries=4, num_contexts_per_query=3))

    def test_rerun_is_idempotent(self, monkeypatch, tmp_path, synthetic_events, synthetic_query_codes):
        tasks_dir = self._run_env(monkeypatch, tmp_path, synthetic_events, synthetic_query_codes)
        cfg = self._cfg(synthetic_query_codes)

        st.run(cfg)
        shard_fp = tasks_dir / "train" / "0.parquet"
        first_bytes = shard_fp.read_bytes()

        # Second run with overwrite=False skips the finished shard (atomic-write skip path),
        # leaving the output byte-for-byte identical.
        st.run(cfg)
        assert shard_fp.read_bytes() == first_bytes


def _union_final_output(tasks_dir: Path, split: str = "train") -> pl.DataFrame:
    """Read every Stage 4 shard parquet under ``{tasks_dir}/{split}`` into one sorted frame.

    Sorting on all columns makes the union order-insensitive so two runs that differ only in how rows were
    partitioned across worker processes compare value-equal.
    """
    shard_files = sorted((tasks_dir / split).glob("*.parquet"))
    assert shard_files, f"no shard outputs under {tasks_dir / split}"
    union = pl.concat([pl.read_parquet(f) for f in shard_files])
    return union.sort(by=union.columns)


def _two_shard_cohort(tmp_path: Path, query_codes: list[str], *, split: str = "train") -> Path:
    """Write a two-shard MEDS cohort (subjects do NOT span shards — invariant 4) and return its root.

    Mirrors ``_write_fake_cohort``'s ``$INTERMEDIATE`` layout but splits subjects across two shard
    parquets so Stage 4's ProcessPoolExecutor actually fans out (a single shard collapses N workers
    to 1).  Shard "0" holds subjects 1 & 2, shard "1" holds subject 3; each subject gets enough
    distinct prediction times to clear ``min_prediction_times_per_subject`` and cycles through
    ``query_codes`` so labels are non-trivial.
    """
    base = datetime(2020, 1, 1)

    def _subject(subj: int, n: int) -> pl.DataFrame:
        return pl.DataFrame(
            [
                {
                    "subject_id": subj,
                    "time": base + timedelta(days=i * 10 + subj),
                    "code": query_codes[i % len(query_codes)],
                }
                for i in range(n)
            ]
        )

    data_dir = tmp_path / "intermediate"
    split_dir = data_dir / "data" / split
    split_dir.mkdir(parents=True, exist_ok=True)
    pl.concat([_subject(1, 30), _subject(2, 30)]).write_parquet(split_dir / "0.parquet")
    _subject(3, 30).write_parquet(split_dir / "1.parquet")
    return data_dir


class TestCrossProcessDeterminism:
    """Issue #211: the parallel Stage 4 fan-out must produce the same labeled dataset as the serial path.

    Every other ``run()`` test pins ``max_workers=1``, so the spawn-based ``ProcessPoolExecutor`` added in
    4d39a24 (polars-fork deadlock fix) is never compared against the single-worker path.  This runs the
    *same* config (same seed, same two-shard cohort) once serially and once across >=2 workers, then
    asserts the unioned outputs are value-identical — guarding against any worker-count-dependent drift
    in RNG order, shard assignment, or asof labeling.
    """

    def _cfg(self, query_codes, *, max_workers):
        return OmegaConf.create(
            {
                "num_queries": 8,
                "num_contexts_per_query": 3,
                "min_prediction_times_per_subject": 5,
                "max_workers": max_workers,
                "query_codes": list(query_codes),
                "min_duration": 1,
                "max_duration": 365,
                "duration_distribution": "log-uniform",
                "split": "train",
                "seed": 1,
                "overwrite": False,
            }
        )

    def test_serial_and_parallel_outputs_are_identical(self, monkeypatch, tmp_path, synthetic_query_codes):
        data_dir = _two_shard_cohort(tmp_path, synthetic_query_codes)
        monkeypatch.setenv("INTERMEDIATE", str(data_dir))

        # Two disjoint output roots: TRAINING_TASKS_DIR drives both the final root and (via
        # default_artifacts_dir's sibling rule) the intermediate root, so the runs never share state.
        serial_dir = tmp_path / "tasks_serial"
        parallel_dir = tmp_path / "tasks_parallel"

        monkeypatch.setenv("TRAINING_TASKS_DIR", str(serial_dir))
        st.run(self._cfg(synthetic_query_codes, max_workers=1))

        monkeypatch.setenv("TRAINING_TASKS_DIR", str(parallel_dir))
        st.run(self._cfg(synthetic_query_codes, max_workers=2))

        serial = _union_final_output(serial_dir)
        parallel = _union_final_output(parallel_dir)
        # Both runs must spread their rows across the two shards (otherwise the >=2-worker run never
        # actually fanned out and the comparison is vacuous).
        assert len(sorted((parallel_dir / "train").glob("*.parquet"))) == 2
        assert serial.equals(parallel)


# Pinned expected rows for TestSnapshot (seed=1, num_queries=3, num_contexts_per_query=2 over the
# two-shard fixture).  ``prediction_time`` is an ISO string parsed to Datetime(us) in the test.
EXPECTED_SNAPSHOT_ROWS: list[dict] = [
    {
        "subject_id": 1,
        "prediction_time": "2020-02-21T00:00:00",
        "query": "ICD//B02",
        "duration_days": 1.145217776298523,
        "boolean_value": False,
    },
    {
        "subject_id": 1,
        "prediction_time": "2020-05-11T00:00:00",
        "query": "ICD//A01",
        "duration_days": 38.73188018798828,
        "boolean_value": True,
    },
    {
        "subject_id": 1,
        "prediction_time": "2020-06-30T00:00:00",
        "query": "MED//D04",
        "duration_days": 1.6647769212722778,
        "boolean_value": False,
    },
    {
        "subject_id": 2,
        "prediction_time": "2020-03-03T00:00:00",
        "query": "MED//D04",
        "duration_days": 1.6647769212722778,
        "boolean_value": False,
    },
    {
        "subject_id": 2,
        "prediction_time": "2020-06-11T00:00:00",
        "query": "ICD//A01",
        "duration_days": 38.73188018798828,
        "boolean_value": False,
    },
    {
        "subject_id": 3,
        "prediction_time": "2020-05-23T00:00:00",
        "query": "ICD//B02",
        "duration_days": 1.145217776298523,
        "boolean_value": False,
    },
]


class TestSnapshot:
    """Issue #211: pin actual output *values*, not just shape.

    Existing end-to-end tests assert row count / columns / schema but never the cell values, so a silent RNG-
    order or asof-window regression that preserves row count would slip through.  This snapshots the full
    sampled-and-labeled output of a tiny fixed-seed run against an inline expected frame; any change to query
    draws, context draws, prediction-time resolution, or asof labeling flips a value and trips it.
    """

    def test_output_matches_inline_snapshot(self, monkeypatch, tmp_path, synthetic_query_codes):
        data_dir = _two_shard_cohort(tmp_path, synthetic_query_codes)
        tasks_dir = tmp_path / "training_tasks"
        monkeypatch.setenv("INTERMEDIATE", str(data_dir))
        monkeypatch.setenv("TRAINING_TASKS_DIR", str(tasks_dir))

        cfg = OmegaConf.create(
            {
                "num_queries": 3,
                "num_contexts_per_query": 2,
                "min_prediction_times_per_subject": 5,
                "max_workers": 1,
                "query_codes": list(synthetic_query_codes),
                "min_duration": 1,
                "max_duration": 365,
                "duration_distribution": "log-uniform",
                "split": "train",
                "seed": 1,
                "overwrite": False,
            }
        )
        st.run(cfg)

        got = _union_final_output(tasks_dir).select(
            ["subject_id", "prediction_time", "query", "duration_days", "boolean_value"]
        )

        expected = pl.DataFrame(EXPECTED_SNAPSHOT_ROWS).select(got.columns)
        expected = expected.with_columns(
            pl.col("prediction_time").str.to_datetime(time_unit="us"),
            pl.col("duration_days").cast(pl.Float32),
        )
        expected = expected.sort(by=expected.columns)

        assert got.equals(expected), (
            "Stage 4 output drifted from the pinned snapshot. If this change is intentional, "
            f"update EXPECTED_SNAPSHOT_ROWS to:\n{got.to_dicts()}"
        )
