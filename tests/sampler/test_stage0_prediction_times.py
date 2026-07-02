"""Stage 0: ``build_prediction_times`` — build + cache the canonical prediction-time map and counts.

``min_prediction_times_per_subject=2`` ⇒ eligibility is ``n_prediction_times >= 3``. The fixture
has subject 1 (5 distinct times, eligible), subject 2 (3, eligible at the boundary), and subject 3
(2, dropped) — so the ``+1`` boundary and the eligibility filter are both exercised.

Covers the gapless+dedup index (invariant 2), eligibility, invariant 4 (no subject spans shards),
shard partitioning, null-time dropping, death truncation (#265), and the cache reuse /
invalidation / overwrite paths.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from meds import death_code

from every_query.generate_tasks import sample_tasks as st
from every_query.generate_tasks.sample_tasks import (
    build_prediction_times,
    prediction_time_counts_path,
    prediction_times_meta_path,
    prediction_times_path,
)


class TestStage0:
    BASE = datetime(2020, 1, 1)
    MIN = 2

    @pytest.fixture
    def path_to_data(self, tmp_path: Path, subject_events, write_split_shards) -> Path:
        """Single-shard MEDS dataset; returns the ``path_to_data`` root.

        The concatenated ``events`` written to shard ``"0"`` look like::

            shape: (15, 3)
            ┌────────────┬─────────────────────┬─────────┐
            │ subject_id ┆ time                ┆ code    │
            │ ---        ┆ ---                 ┆ ---     │
            │ i64        ┆ datetime[μs]        ┆ str     │
            ╞════════════╪═════════════════════╪═════════╡
            │ 1          ┆ 2020-01-01 00:00:00 ┆ ICD//00 │
            │ 1          ┆ 2020-01-01 00:00:00 ┆ ICD//01 │
            │ 1          ┆ 2020-01-02 00:00:00 ┆ ICD//00 │
            │ 1          ┆ 2020-01-02 00:00:00 ┆ ICD//01 │
            │ 1          ┆ 2020-01-03 00:00:00 ┆ ICD//00 │
            │ 1          ┆ 2020-01-03 00:00:00 ┆ ICD//01 │
            │ 1          ┆ 2020-01-04 00:00:00 ┆ ICD//00 │
            │ 1          ┆ 2020-01-04 00:00:00 ┆ ICD//01 │
            │ 1          ┆ 2020-01-05 00:00:00 ┆ ICD//00 │
            │ 1          ┆ 2020-01-05 00:00:00 ┆ ICD//01 │
            │ 2          ┆ 2020-01-01 00:00:00 ┆ ICD//00 │
            │ 2          ┆ 2020-01-02 00:00:00 ┆ ICD//00 │
            │ 2          ┆ 2020-01-03 00:00:00 ┆ ICD//00 │
            │ 3          ┆ 2020-01-01 00:00:00 ┆ ICD//00 │
            │ 3          ┆ 2020-01-02 00:00:00 ┆ ICD//00 │
            └────────────┴─────────────────────┴─────────┘

        Subject 1 has ``dups=2`` (two codes per distinct time → 5 distinct times,
        10 rows); subjects 2 and 3 have 3 and 2 distinct times respectively.
        """
        events = pl.concat(
            [
                subject_events(1, 5, base=self.BASE, dups=2),  # duplicate (subject,time) rows
                subject_events(2, 3, base=self.BASE),
                subject_events(3, 2, base=self.BASE),
            ]
        )
        return write_split_shards(tmp_path, {"0": events})

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

        s2 = pmap.filter(pl.col("subject_id") == 2).sort("time")
        assert s2["prediction_time_index"].to_list() == [0, 1, 2]
        assert s2["time"].to_list() == sorted(s2["time"].to_list())

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
        # counts should contain a unique row per elible subject
        counts = pl.read_parquet(prediction_time_counts_path(artifacts_dir, "train"))
        # pmap has a row for each unique timestamp for each subject
        pmap = pl.read_parquet(prediction_times_path(artifacts_dir, "train", "0"))
        assert int(counts["n_prediction_times"].sum()) == pmap.height
        assert counts.height == n == pmap["subject_id"].n_unique()

    def test_subject_spanning_shards_raises(
        self, tmp_path, artifacts_dir, subject_events, write_split_shards
    ):
        # Subject 1's distinct times split across two shards ⇒ invariant 4 violation.
        path_to_data = write_split_shards(
            tmp_path,
            {
                "0": subject_events(1, 3, base=self.BASE),
                "1": subject_events(1, 3, base=self.BASE + timedelta(days=100)),
            },
        )
        with pytest.raises(ValueError, match=r"span multiple shards.*1"):
            build_prediction_times(path_to_data, artifacts_dir, "train", self.MIN)

    def test_partitioned_by_shard(self, tmp_path, artifacts_dir, subject_events, write_split_shards):
        path_to_data = write_split_shards(
            tmp_path,
            {
                "0": subject_events(1, 4, base=self.BASE),
                "1": subject_events(2, 4, base=self.BASE),
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

    def test_null_time_rows_are_not_prediction_times(
        self, tmp_path, artifacts_dir, subject_events, write_split_shards
    ):
        # Subject 1: 3 real distinct times + one null-time static row (e.g. MEDS demographics).
        # The null must be dropped, not claim prediction_time_index=0 nor inflate n_prediction_times.
        events = pl.concat(
            [
                subject_events(1, 3, base=self.BASE),
                pl.DataFrame({"subject_id": [1], "time": [None], "code": ["DEMO//SEX"]}),
            ],
            how="diagonal",
        )
        path_to_data = write_split_shards(tmp_path, {"0": events})
        build_prediction_times(path_to_data, artifacts_dir, "train", self.MIN)

        pmap = pl.read_parquet(prediction_times_path(artifacts_dir, "train", "0"))
        s1 = pmap.filter(pl.col("subject_id") == 1).sort("prediction_time_index")
        assert s1["prediction_time_index"].to_list() == [0, 1, 2]
        assert s1["time"].null_count() == 0

        counts = pl.read_parquet(prediction_time_counts_path(artifacts_dir, "train"))
        assert dict(zip(counts["subject_id"], counts["n_prediction_times"], strict=True)) == {1: 3}

    def test_post_death_times_are_dropped_and_dont_count_toward_eligibility(
        self, tmp_path, artifacts_dir, subject_events, write_split_shards
    ):
        # Subject 1: 5 distinct times, MEDS_DEATH at day 2 ⇒ days 3-4 truncated, 3 remain (still
        # eligible; the death timestamp itself is kept as a prediction time). Subject 2: 3 distinct
        # times, MEDS_DEATH at day 1 ⇒ 2 remain — without truncation it would clear the
        # eligibility bar on post-death timestamps alone (#265's inflation case).
        events = pl.concat(
            [
                subject_events(1, 5, base=self.BASE),
                pl.DataFrame(
                    {"subject_id": [1], "time": [self.BASE + timedelta(days=2)], "code": [death_code]}
                ),
                subject_events(2, 3, base=self.BASE),
                pl.DataFrame(
                    {"subject_id": [2], "time": [self.BASE + timedelta(days=1)], "code": [death_code]}
                ),
            ]
        )
        path_to_data = write_split_shards(tmp_path, {"0": events})
        n = build_prediction_times(path_to_data, artifacts_dir, "train", self.MIN)
        assert n == 1

        pmap = pl.read_parquet(prediction_times_path(artifacts_dir, "train", "0"))
        assert pmap["subject_id"].unique().to_list() == [1]
        s1 = pmap.filter(pl.col("subject_id") == 1).sort("prediction_time_index")
        assert s1["time"].to_list() == [self.BASE + timedelta(days=i) for i in range(3)]

        counts = pl.read_parquet(prediction_time_counts_path(artifacts_dir, "train"))
        assert dict(zip(counts["subject_id"], counts["n_prediction_times"], strict=True)) == {1: 3}

    def test_pre_death_truncation_cache_is_rebuilt(self, path_to_data, artifacts_dir, monkeypatch):
        build_prediction_times(path_to_data, artifacts_dir, "train", self.MIN)
        # Simulate a pre-#265 cache: sidecar without the death_truncation field.
        meta_fp = prediction_times_meta_path(artifacts_dir, "train")
        meta = json.loads(meta_fp.read_text())
        del meta["death_truncation"]
        meta_fp.write_text(json.dumps(meta))

        calls = {"n": 0}
        real = st._read_prediction_time_shard

        def _spy(*args, **kwargs):
            calls["n"] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(st, "_read_prediction_time_shard", _spy)
        n = build_prediction_times(path_to_data, artifacts_dir, "train", self.MIN)
        assert n == 2
        assert calls["n"] == 1  # stale sidecar ⇒ rebuild, not silent reuse
