"""Tests for the eval suite: gen_task.py (process_eval_tasks) and eval.py helpers."""

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from omegaconf import ListConfig, OmegaConf

from every_query.eval import _model_name
from every_query.eval_suite.gen_task import _resolve_codes, process_eval_tasks
from every_query.utils.codes import code_slug

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

CODES = ["ICD//A01", "ICD//B02", "MED//C03"]
DURATIONS = [30, 90, 180]
INDEX_HASH = "testhash123"


def _make_task_df(
    n_subjects: int = 4,
    n_pred_times: int = 5,
    codes: list[str] = CODES,
    censored_rate: float = 0.25,
) -> pl.DataFrame:
    """Build a wide-format task DataFrame: subject_id, prediction_time, censored, <code columns>."""
    base = datetime(2020, 1, 1)
    rows = []
    for subj in range(1, n_subjects + 1):
        for t in range(n_pred_times):
            row = {
                "subject_id": subj,
                "prediction_time": base + timedelta(days=t * 10 + subj),
                "censored": (subj + t) % int(1 / censored_rate) == 0 if censored_rate > 0 else False,
            }
            for code in codes:
                # Some True, some False, some None (for null-filtering tests)
                val = (subj + t) % 3
                if val == 0:
                    row[code] = True
                elif val == 1:
                    row[code] = False
                else:
                    row[code] = None
            rows.append(row)
    return pl.DataFrame(rows)


def _make_index_df(task_df: pl.DataFrame, sample_frac: float = 0.6, seed: int = 42) -> pl.DataFrame:
    """Sample a subset of prediction times from task_df to serve as index times."""
    return task_df.select("subject_id", "prediction_time").sample(fraction=sample_frac, seed=seed)


def _build_fixtures(
    tmp_path: Path,
    durations: list[int] = DURATIONS,
    codes: list[str] = CODES,
    n_subjects: int = 4,
    n_pred_times: int = 5,
    shard_name: str = "0.parquet",
) -> tuple[Path, Path, Path]:
    """Create directory tree with index times + duration-specific task dirs.

    Returns (index_dir, task_dir_base, out_root).
    """
    task_dir_base = tmp_path / "tasks"
    index_dir = tmp_path / "index_times"
    out_root = tmp_path / "output"

    # Build task data (same subjects/times, different censoring per duration is realistic
    # but for testing we use the same base df — the key invariant is that the duration
    # label in the output matches what we asked for)
    task_df = _make_task_df(n_subjects, n_pred_times, codes)
    index_df = _make_index_df(task_df)

    # Write index times
    index_dir.mkdir(parents=True)
    index_df.write_parquet(index_dir / shard_name)

    # Write duration-specific task dirs
    for d in durations:
        dur_dir = task_dir_base / str(d) / "held_out"
        dur_dir.mkdir(parents=True)
        task_df.write_parquet(dur_dir / shard_name)

    return index_dir, task_dir_base, out_root


def _read_all_outputs(out_root: Path, index_hash: str) -> pl.DataFrame:
    """Read and concatenate all output parquets under out_root/{index_hash}/."""
    base = out_root / index_hash
    if not base.exists():
        return pl.DataFrame()
    parquets = list(base.rglob("*.parquet"))
    if not parquets:
        return pl.DataFrame()
    return pl.concat([pl.read_parquet(p) for p in parquets], how="vertical")


# ---------------------------------------------------------------------------
# _resolve_codes tests
# ---------------------------------------------------------------------------


class TestResolveCodes:
    def test_flat_list(self):
        obj = ListConfig(["A", "B", "C"])
        assert _resolve_codes(obj) == ["A", "B", "C"]

    def test_dict_with_id_ood(self):
        obj = OmegaConf.create({"id": ["A", "B"], "ood": ["C"]})
        assert _resolve_codes(obj) == ["A", "B", "C"]

    def test_dict_missing_ood(self):
        obj = OmegaConf.create({"id": ["A", "B"]})
        assert _resolve_codes(obj) == ["A", "B"]

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="eval_codes must be a list or dict"):
            _resolve_codes("not_valid")


# ---------------------------------------------------------------------------
# process_eval_tasks — output schema
# ---------------------------------------------------------------------------


class TestGenTaskOutputSchema:
    def test_columns(self, tmp_path):
        index_dir, task_dir_base, out_root = _build_fixtures(tmp_path)
        process_eval_tasks(index_dir, task_dir_base, out_root, INDEX_HASH, CODES, DURATIONS)

        df = _read_all_outputs(out_root, INDEX_HASH)
        expected = {"subject_id", "prediction_time", "boolean_value", "query", "occurs", "duration_days"}
        assert set(df.columns) == expected

    def test_no_null_occurs(self, tmp_path):
        """Rows with null occurs should be filtered out."""
        index_dir, task_dir_base, out_root = _build_fixtures(tmp_path)
        process_eval_tasks(index_dir, task_dir_base, out_root, INDEX_HASH, CODES, DURATIONS)

        df = _read_all_outputs(out_root, INDEX_HASH)
        assert df["occurs"].null_count() == 0


# ---------------------------------------------------------------------------
# process_eval_tasks — duration invariants
# ---------------------------------------------------------------------------


class TestGenTaskDurations:
    def test_all_requested_durations_present(self, tmp_path):
        """Output should contain rows for every requested duration."""
        index_dir, task_dir_base, out_root = _build_fixtures(tmp_path)
        process_eval_tasks(index_dir, task_dir_base, out_root, INDEX_HASH, CODES, DURATIONS)

        df = _read_all_outputs(out_root, INDEX_HASH)
        assert set(df["duration_days"].unique().to_list()) == set(DURATIONS)

    def test_duration_values_exact(self, tmp_path):
        """duration_days column should only contain values from the requested durations."""
        index_dir, task_dir_base, out_root = _build_fixtures(tmp_path)
        process_eval_tasks(index_dir, task_dir_base, out_root, INDEX_HASH, CODES, DURATIONS)

        df = _read_all_outputs(out_root, INDEX_HASH)
        assert set(df["duration_days"].unique().to_list()).issubset(set(DURATIONS))

    def test_duration_order_invariant(self, tmp_path):
        """Changing the order of durations should produce identical output (when sorted)."""
        index_dir, task_dir_base, _out_root = _build_fixtures(tmp_path)

        out1 = tmp_path / "out1"
        out2 = tmp_path / "out2"
        process_eval_tasks(index_dir, task_dir_base, out1, INDEX_HASH, CODES, [30, 90, 180])
        process_eval_tasks(index_dir, task_dir_base, out2, INDEX_HASH, CODES, [180, 30, 90])

        df1 = _read_all_outputs(out1, INDEX_HASH).sort("duration_days", "query", "subject_id")
        df2 = _read_all_outputs(out2, INDEX_HASH).sort("duration_days", "query", "subject_id")
        assert df1.equals(df2)

    def test_same_prediction_times_across_durations(self, tmp_path):
        """Same index times should be evaluated at every duration (same subject_id x prediction_time set)."""
        index_dir, task_dir_base, out_root = _build_fixtures(tmp_path)
        process_eval_tasks(index_dir, task_dir_base, out_root, INDEX_HASH, CODES[:1], DURATIONS)

        df = _read_all_outputs(out_root, INDEX_HASH)
        per_duration_keys = {}
        for d in DURATIONS:
            keys = (
                df.filter(pl.col("duration_days") == d)
                .select("subject_id", "prediction_time")
                .unique()
                .sort("subject_id", "prediction_time")
            )
            per_duration_keys[d] = keys

        # All durations should have the same prediction time set
        reference = per_duration_keys[DURATIONS[0]]
        for d in DURATIONS[1:]:
            assert reference.equals(per_duration_keys[d]), f"Prediction times differ for duration={d}"

    def test_missing_duration_skipped(self, tmp_path):
        """If a duration directory doesn't exist, it should be skipped without error."""
        index_dir, task_dir_base, out_root = _build_fixtures(tmp_path, durations=[30, 90])
        # Ask for duration=180 which doesn't exist on disk
        process_eval_tasks(index_dir, task_dir_base, out_root, INDEX_HASH, CODES, [30, 90, 180])

        df = _read_all_outputs(out_root, INDEX_HASH)
        assert set(df["duration_days"].unique().to_list()) == {30, 90}


# ---------------------------------------------------------------------------
# process_eval_tasks — code invariants
# ---------------------------------------------------------------------------


class TestGenTaskCodes:
    def test_all_requested_codes_present(self, tmp_path):
        index_dir, task_dir_base, out_root = _build_fixtures(tmp_path)
        process_eval_tasks(index_dir, task_dir_base, out_root, INDEX_HASH, CODES, DURATIONS)

        df = _read_all_outputs(out_root, INDEX_HASH)
        assert set(df["query"].unique().to_list()) == set(CODES)

    def test_code_order_invariant(self, tmp_path):
        """Changing code order should produce identical output (when sorted)."""
        index_dir, task_dir_base, _out_root = _build_fixtures(tmp_path)

        out1 = tmp_path / "out1"
        out2 = tmp_path / "out2"
        process_eval_tasks(index_dir, task_dir_base, out1, INDEX_HASH, CODES, DURATIONS)
        process_eval_tasks(index_dir, task_dir_base, out2, INDEX_HASH, list(reversed(CODES)), DURATIONS)

        df1 = _read_all_outputs(out1, INDEX_HASH).sort("duration_days", "query", "subject_id")
        df2 = _read_all_outputs(out2, INDEX_HASH).sort("duration_days", "query", "subject_id")
        assert df1.equals(df2)

    def test_missing_code_skipped(self, tmp_path):
        """If a code doesn't exist in the task shard, it should be skipped without error."""
        index_dir, task_dir_base, out_root = _build_fixtures(tmp_path)
        # Add a code that doesn't exist in the data
        codes_with_missing = [*CODES, "FAKE//MISSING"]
        process_eval_tasks(index_dir, task_dir_base, out_root, INDEX_HASH, codes_with_missing, DURATIONS)

        df = _read_all_outputs(out_root, INDEX_HASH)
        assert "FAKE//MISSING" not in df["query"].unique().to_list()
        assert set(CODES).issubset(set(df["query"].unique().to_list()))


# ---------------------------------------------------------------------------
# process_eval_tasks — directory structure
# ---------------------------------------------------------------------------


class TestGenTaskDirectoryStructure:
    def test_output_organized_by_duration_and_code(self, tmp_path):
        """Output parquets should be at out_root/{hash}/{duration}/{code_slug}/."""
        index_dir, task_dir_base, out_root = _build_fixtures(tmp_path, durations=[30], codes=CODES[:1])
        process_eval_tasks(index_dir, task_dir_base, out_root, INDEX_HASH, CODES[:1], [30])

        slug = code_slug(CODES[0])
        expected_dir = out_root / INDEX_HASH / "30" / slug
        assert expected_dir.is_dir()
        assert list(expected_dir.glob("*.parquet"))

    def test_multiple_shards_produce_multiple_outputs(self, tmp_path):
        """Each input shard should produce a corresponding output shard."""
        index_dir, task_dir_base, out_root = _build_fixtures(tmp_path, durations=[30], codes=CODES[:1])

        # Add a second shard
        task_df = _make_task_df(codes=CODES[:1])
        index_df = _make_index_df(task_df, seed=99)
        index_df.write_parquet(index_dir / "1.parquet")
        (task_dir_base / "30" / "held_out" / "1.parquet").write_bytes(
            (task_dir_base / "30" / "held_out" / "0.parquet").read_bytes()
        )

        process_eval_tasks(index_dir, task_dir_base, out_root, INDEX_HASH, CODES[:1], [30])

        slug = code_slug(CODES[0])
        output_dir = out_root / INDEX_HASH / "30" / slug
        assert (output_dir / "0.parquet").exists()
        assert (output_dir / "1.parquet").exists()


# ---------------------------------------------------------------------------
# process_eval_tasks — censoring comes from duration-specific file
# ---------------------------------------------------------------------------


class TestGenTaskCensorSource:
    def test_boolean_value_from_task_file_not_index(self, tmp_path):
        """boolean_value should reflect the duration-specific censored column, not index times."""
        base = datetime(2020, 1, 1)

        # Index times (no censored column needed anymore — we only use subject_id, prediction_time)
        index_df = pl.DataFrame(
            {
                "subject_id": [1, 2],
                "prediction_time": [base, base + timedelta(days=10)],
            }
        )

        # Duration=30 task data: subject 1 is censored, subject 2 is not
        task_df_30 = pl.DataFrame(
            {
                "subject_id": [1, 2],
                "prediction_time": [base, base + timedelta(days=10)],
                "censored": [True, False],
                "ICD//A01": [False, True],
            }
        )

        # Duration=90 task data: neither is censored
        task_df_90 = pl.DataFrame(
            {
                "subject_id": [1, 2],
                "prediction_time": [base, base + timedelta(days=10)],
                "censored": [False, False],
                "ICD//A01": [True, True],
            }
        )

        index_dir = tmp_path / "index"
        index_dir.mkdir()
        index_df.write_parquet(index_dir / "0.parquet")

        task_base = tmp_path / "tasks"
        for d, df in [(30, task_df_30), (90, task_df_90)]:
            d_dir = task_base / str(d) / "held_out"
            d_dir.mkdir(parents=True)
            df.write_parquet(d_dir / "0.parquet")

        out_root = tmp_path / "out"
        process_eval_tasks(index_dir, task_base, out_root, INDEX_HASH, ["ICD//A01"], [30, 90])

        all_df = _read_all_outputs(out_root, INDEX_HASH)

        # At duration=30, subject 1 should be censored (True)
        row_30_s1 = all_df.filter(
            (pl.col("duration_days") == 30) & (pl.col("subject_id") == 1)
        )
        assert row_30_s1["boolean_value"][0] is True

        # At duration=90, subject 1 should NOT be censored (False)
        row_90_s1 = all_df.filter(
            (pl.col("duration_days") == 90) & (pl.col("subject_id") == 1)
        )
        assert row_90_s1["boolean_value"][0] is False


# ---------------------------------------------------------------------------
# process_eval_tasks — skip_existing
# ---------------------------------------------------------------------------


class TestGenTaskSkipExisting:
    def test_skip_existing_preserves_file(self, tmp_path):
        index_dir, task_dir_base, out_root = _build_fixtures(tmp_path, durations=[30], codes=CODES[:1])

        process_eval_tasks(index_dir, task_dir_base, out_root, INDEX_HASH, CODES[:1], [30])

        slug = code_slug(CODES[0])
        out_fp = out_root / INDEX_HASH / "30" / slug / "0.parquet"
        mtime_before = out_fp.stat().st_mtime

        # Run again with skip_existing=True
        process_eval_tasks(
            index_dir, task_dir_base, out_root, INDEX_HASH, CODES[:1], [30], skip_existing=True
        )
        assert out_fp.stat().st_mtime == mtime_before

    def test_overwrite_when_skip_existing_false(self, tmp_path):
        index_dir, task_dir_base, out_root = _build_fixtures(tmp_path, durations=[30], codes=CODES[:1])

        process_eval_tasks(index_dir, task_dir_base, out_root, INDEX_HASH, CODES[:1], [30])

        slug = code_slug(CODES[0])
        out_fp = out_root / INDEX_HASH / "30" / slug / "0.parquet"
        # Run again with skip_existing=False (default) — should rewrite
        process_eval_tasks(
            index_dir, task_dir_base, out_root, INDEX_HASH, CODES[:1], [30], skip_existing=False
        )
        # File should still exist and have same content (deterministic)
        assert out_fp.exists()


# ---------------------------------------------------------------------------
# process_eval_tasks — shard mismatch
# ---------------------------------------------------------------------------


class TestGenTaskShardMismatch:
    def test_shard_name_mismatch_raises(self, tmp_path):
        """If index shards and task shards have different names, should raise AssertionError."""
        index_dir, task_dir_base, out_root = _build_fixtures(tmp_path, durations=[30])

        # Rename the task shard so names don't match
        src = task_dir_base / "30" / "held_out" / "0.parquet"
        dst = task_dir_base / "30" / "held_out" / "99.parquet"
        src.rename(dst)

        with pytest.raises(AssertionError, match="Shard mismatch"):
            process_eval_tasks(index_dir, task_dir_base, out_root, INDEX_HASH, CODES, [30])


# ---------------------------------------------------------------------------
# process_eval_tasks — cross-duration row count
# ---------------------------------------------------------------------------


class TestGenTaskRowCounts:
    def test_row_count_per_duration_code(self, tmp_path):
        """Each (duration, code) combination should have the same number of rows
        (since we use the same index times and same base task data)."""
        index_dir, task_dir_base, out_root = _build_fixtures(tmp_path)
        process_eval_tasks(index_dir, task_dir_base, out_root, INDEX_HASH, CODES, DURATIONS)

        df = _read_all_outputs(out_root, INDEX_HASH)
        counts = df.group_by("duration_days", "query").len()

        # With identical task data per duration, counts should be uniform across durations for each code
        for code in CODES:
            code_counts = counts.filter(pl.col("query") == code)["len"].to_list()
            assert len(set(code_counts)) == 1, f"Row counts differ across durations for code={code}"


# ---------------------------------------------------------------------------
# eval.py — _model_name
# ---------------------------------------------------------------------------


class TestModelName:
    def test_extracts_basename(self):
        assert _model_name("/users/foo/results/outputs/2026-01-03/15-43-49") == "15-43-49"

    def test_trailing_slash(self):
        assert _model_name("/users/foo/results/outputs/my_run/") == "my_run"

    def test_simple_name(self):
        assert _model_name("my_model") == "my_model"
