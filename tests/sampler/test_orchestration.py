"""End-to-end orchestration: ``run()`` wires Stages 0-4 together.

- ``TestMainOrchestration`` — a clean run over a synthetic single-shard cohort writes the per-shard
  final dataset, the union row count equals the sampling budget, and reruns are idempotent.
- ``TestCrossProcessDeterminism`` — the parallel Stage 4 fan-out matches the serial path value-for-value.
- ``TestSnapshot`` — pins the actual output *values* (not just shape) for a tiny fixed-seed run.
- Gap tests — the row-count guard actually raises on a budget mismatch, and ``overwrite=True`` forces
  a full rebuild.
"""

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from omegaconf import OmegaConf

from every_query.data.schema import TaskQuerySchema
from every_query.generate_tasks import sample_tasks as st


def _write_fake_cohort(
    tmp_path: Path,
    events: pl.DataFrame,
    query_codes: list[str],
    split: str = "train",
    shard_name: str = "0",
) -> tuple[Path, Path]:
    """Write a minimal MEDS-shaped cohort the redesigned ``run()`` can read from.

    Mirrors the split layout of ``.env.example``: ``$INTERMEDIATE`` and ``$PROCESSED`` are sibling
    directories rather than the same root, so the cohort this helper writes uses:

        {data_dir}/data/{split}/{shard_name}.parquet     <- event shards ($INTERMEDIATE)
        {processed_dir}/metadata/codes.parquet           <- query universe ($PROCESSED)

    Returns ``(data_dir, processed_dir)`` as two distinct paths so tests can exercise the separation.
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

    # -- Gap tests --------------------------------------------------------------------------------

    def test_run_raises_on_row_count_mismatch(
        self, monkeypatch, tmp_path, synthetic_events, synthetic_query_codes
    ):
        """The final-row-count guard must fire when Stage 4 writes fewer rows than the budget.

        Every other ``run()`` test only proves the *happy* path (guard not triggered). Here we patch
        ``build_index`` to drop the last query and its context block, so Stage 4 legitimately writes
        ``(num_queries-1) * num_contexts_per_query`` rows — fewer than the budget — and the guard
        raises.
        """
        self._run_env(monkeypatch, tmp_path, synthetic_events, synthetic_query_codes)
        cfg = self._cfg(synthetic_query_codes, num_queries=4, num_contexts_per_query=2)  # budget 8

        real_build_index = st.build_index

        def _short_build_index(queries, contexts, training_task_artifacts_dir, split, num_contexts_per_query):
            # Drop the last query and its trailing contexts block; the (queries, contexts) lengths
            # stay consistent so build_index's own height check passes — only the *total* shrinks.
            return real_build_index(
                queries[:-1],
                contexts.head(contexts.height - num_contexts_per_query),
                training_task_artifacts_dir,
                split,
                num_contexts_per_query,
            )

        monkeypatch.setattr(st, "build_index", _short_build_index)
        with pytest.raises(ValueError, match=r"expected 8"):
            st.run(cfg)

    def test_run_overwrite_forces_rebuild(
        self, monkeypatch, tmp_path, synthetic_events, synthetic_query_codes
    ):
        """``overwrite=True`` relabels shards even when an output already exists.

        Complements ``test_rerun_is_idempotent`` (the skip path): after a clean run we corrupt the
        shard output, rerun with ``overwrite=True``, and assert the dataset is rebuilt to the same
        deterministic value (proving the worker did NOT skip the existing file).
        """
        tasks_dir = self._run_env(monkeypatch, tmp_path, synthetic_events, synthetic_query_codes)

        st.run(self._cfg(synthetic_query_codes))
        clean = _union_final_output(tasks_dir)

        shard_fp = tasks_dir / "train" / "0.parquet"
        shard_fp.write_bytes(b"corrupted-not-a-parquet")

        st.run(self._cfg(synthetic_query_codes, overwrite=True))
        rebuilt = _union_final_output(tasks_dir)
        assert rebuilt.equals(clean)

    def test_changed_seed_relabels_without_overwrite(
        self, monkeypatch, tmp_path, synthetic_events, synthetic_query_codes
    ):
        """Bug #2: a sampling-config change must not leave stale labels under ``overwrite=False``.

        Stage 3 always rebuilds the index, but Stage 4's skip used to be existence-only — so rerunning
        with a new ``seed`` (same shard set) silently kept the previous run's labels. The per-shard
        index fingerprint must detect the mismatch and relabel even without ``overwrite=True``.
        """
        tasks_dir = self._run_env(monkeypatch, tmp_path, synthetic_events, synthetic_query_codes)

        st.run(self._cfg(synthetic_query_codes))  # seed=1
        seed1 = _union_final_output(tasks_dir)

        cfg2 = self._cfg(synthetic_query_codes)  # overwrite stays False
        cfg2.seed = 2
        st.run(cfg2)
        relabeled = _union_final_output(tasks_dir)

        # The stale seed-1 labels must be gone, and the output must match an overwrite=True rebuild at
        # seed 2 (the ground truth for that config) — proving the relabel happened in place.
        assert not relabeled.equals(seed1)
        cfg2_force = self._cfg(synthetic_query_codes, overwrite=True)
        cfg2_force.seed = 2
        st.run(cfg2_force)
        assert relabeled.equals(_union_final_output(tasks_dir))

    def test_num_queries_zero_writes_empty_dataset(
        self, monkeypatch, tmp_path, synthetic_events, synthetic_query_codes
    ):
        """Bug #3: ``num_queries=0`` (empty budget) must report 0 rows, not crash on an empty glob.

        With no queries, Stage 3 writes no index and Stage 4 writes no output; the final/summary scans must
        side-step polars' "no files found" error and the row-count guard must pass (0 == 0).
        """
        tasks_dir = self._run_env(monkeypatch, tmp_path, synthetic_events, synthetic_query_codes)

        st.run(self._cfg(synthetic_query_codes, num_queries=0))  # must not raise

        assert sorted((tasks_dir / "train").glob("*.parquet")) == []


class TestCrossProcessDeterminism:
    """The parallel Stage 4 fan-out must produce the same labeled dataset as the serial path.

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
    """Pin actual output *values*, not just shape.

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
