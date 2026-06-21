"""Stage 3: ``build_index`` — zip queries with contexts, resolve prediction_time, write ``_index/``.

Fixture: two shards — shard "0" has subjects 1 (5 times) and 2 (4 times), shard "1" has
subject 3 (4 times).  ``min_prediction_times_per_subject=2`` so all three are eligible
(``n_prediction_times >= 3``).

Covers prediction-time resolution, the query<->context zip, dtype-mismatch / height-mismatch /
null-resolution guards, multi-shard partitioning, float32 durations, stale-dir cleanup, plus
artifact-level determinism and cross-shard zip alignment gap tests.
"""

from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from every_query.generate_tasks.sample_tasks import (
    QuerySpec,
    build_index,
    build_prediction_times,
    index_path,
    prediction_times_path,
)


class TestStage3:
    BASE = datetime(2020, 1, 1)
    MIN = 2

    @pytest.fixture
    def stage0_env(self, tmp_path: Path, subject_events, write_split_shards) -> tuple[Path, Path]:
        """Run Stage 0 and return ``(path_to_data, artifacts_dir)``."""
        path_to_data = write_split_shards(
            tmp_path,
            {
                "0": pl.concat(
                    [
                        subject_events(1, 5, base=self.BASE),
                        subject_events(2, 4, base=self.BASE),
                    ]
                ),
                "1": subject_events(3, 4, base=self.BASE),
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

    # -- Gap tests --------------------------------------------------------------------------------

    def test_build_index_is_deterministic(self, stage0_env):
        """Same queries + contexts + Stage 0 map ⇒ byte-identical index parquet on rerun.

        Artifact-level determinism *below* the end-to-end value snapshot: pins that Stage 3's zip + per-shard
        join is order-stable so a rebuild never reshuffles the index.
        """
        _, artifacts_dir = stage0_env
        queries = [QuerySpec("ICD//A", 30.5), QuerySpec("ICD//B", 60.0)]
        contexts = self._make_contexts([1, 1, 1, 1], ["0", "0", "0", "0"], [2, 3, 2, 4])

        build_index(queries, contexts, artifacts_dir, "train", num_contexts_per_query=2)
        first = pl.read_parquet(index_path(artifacts_dir, "train", "0"))
        build_index(queries, contexts, artifacts_dir, "train", num_contexts_per_query=2)
        second = pl.read_parquet(index_path(artifacts_dir, "train", "0"))

        assert first.equals(second)

    def test_query_zip_alignment_across_shards(self, stage0_env):
        """Each query's ``(code, duration)`` block stays glued to its contexts even when those contexts land
        on different shards.

        queries=[FIRST(10d), SECOND(20d)], num_contexts_per_query=2 ⇒ the np.repeat expansion glues FIRST to
        contexts[0:2] and SECOND to contexts[2:4].  Routing FIRST's contexts to shard "0" and SECOND's to
        shard "1" verifies the zip survives the per-shard partitioning.
        """
        _, artifacts_dir = stage0_env
        queries = [QuerySpec("ICD//FIRST", 10.0), QuerySpec("ICD//SECOND", 20.0)]
        contexts = self._make_contexts([1, 2, 3, 3], ["0", "0", "1", "1"], [3, 3, 3, 3])
        build_index(queries, contexts, artifacts_dir, "train", num_contexts_per_query=2)

        idx0 = pl.read_parquet(index_path(artifacts_dir, "train", "0"))
        idx1 = pl.read_parquet(index_path(artifacts_dir, "train", "1"))
        assert set(idx0["query"].to_list()) == {"ICD//FIRST"}
        assert idx0["duration_days"].to_list() == pytest.approx([10.0, 10.0])
        assert set(idx1["query"].to_list()) == {"ICD//SECOND"}
        assert idx1["duration_days"].to_list() == pytest.approx([20.0, 20.0])
