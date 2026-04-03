"""Tests for collate_tasks() and _collate_shard() in train.py."""

import json
import os
from datetime import datetime, timedelta

import polars as pl
from omegaconf import OmegaConf

from every_query.train import _collate_shard, collate_tasks

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CODES = ["ICD//A01", "ICD//B02", "MED//C03"]
DURATIONS = [30, 90, 180]


def _make_shard_df(n_subjects: int = 3, n_pred_times: int = 5) -> pl.DataFrame:
    """Build a minimal wide-format task-label DataFrame that _collate_shard expects."""
    base = datetime(2020, 1, 1)
    rows = []
    for subj in range(1, n_subjects + 1):
        for t in range(n_pred_times):
            row = {
                "subject_id": subj,
                "prediction_time": base + timedelta(days=t * 10 + subj),
                "censored": t % 3 == 0,  # some censored
            }
            for code in CODES:
                row[code] = bool((subj + t) % 2)
            rows.append(row)
    return pl.DataFrame(rows)


def _build_task_dir(
    tmp_path,
    durations=DURATIONS,
    codes=CODES,
    splits=("train", "tuning"),
    n_subjects=3,
    n_pred_times=5,
    shard_name="0.parquet",
):
    """Create a fake task directory tree with parquet shards for each duration/split."""
    task_dir = tmp_path / "tasks"
    shard_df = _make_shard_df(n_subjects, n_pred_times)
    for d in durations:
        for split in splits:
            split_dir = task_dir / str(d) / split
            split_dir.mkdir(parents=True, exist_ok=True)
            shard_df.write_parquet(str(split_dir / shard_name))
    return str(task_dir), shard_df


def _make_cfg(task_dir, codes=CODES, durations=None, seed=42, sample_times_per_subject=20):
    """Build a minimal OmegaConf config matching what collate_tasks expects."""
    cfg = {
        "query": {
            "task_dir": task_dir,
            "codes": list(codes),
            "sample_times_per_subject": sample_times_per_subject,
        },
        "seed": seed,
    }
    return OmegaConf.create(cfg)


# ---------------------------------------------------------------------------
# _collate_shard tests
# ---------------------------------------------------------------------------


class TestCollateShardReproducibility:
    """Same seed → bit-identical output; different seed → different output."""

    def test_same_seed_same_output(self, tmp_path):
        task_dir, _ = _build_task_dir(tmp_path)
        for run in ("run1", "run2"):
            write_dir = str(tmp_path / run)
            os.makedirs(f"{write_dir}/train", exist_ok=True)
            _collate_shard(
                "0.parquet",
                "train",
                write_dir,
                task_dir,
                DURATIONS,
                CODES,
                sample_times_per_subject=20,
                seed=42,
            )

        df1 = pl.read_parquet(str(tmp_path / "run1" / "train" / "0.parquet"))
        df2 = pl.read_parquet(str(tmp_path / "run2" / "train" / "0.parquet"))
        assert df1.equals(df2)

    def test_different_seed_different_output(self, tmp_path):
        task_dir, _ = _build_task_dir(tmp_path)
        for seed, run in [(42, "run1"), (99, "run2")]:
            write_dir = str(tmp_path / run)
            os.makedirs(f"{write_dir}/train", exist_ok=True)
            _collate_shard(
                "0.parquet",
                "train",
                write_dir,
                task_dir,
                DURATIONS,
                CODES,
                sample_times_per_subject=20,
                seed=seed,
            )

        df1 = pl.read_parquet(str(tmp_path / "run1" / "train" / "0.parquet"))
        df2 = pl.read_parquet(str(tmp_path / "run2" / "train" / "0.parquet"))
        sort_cols = ["subject_id", "prediction_time", "duration_days", "query"]
        assert not df1.sort(sort_cols).equals(df2.sort(sort_cols))


class TestCollateShardSamplingCap:
    """Output never exceeds sample_times_per_subject prediction times per subject."""

    def test_respects_cap(self, tmp_path):
        cap = 2
        task_dir, _ = _build_task_dir(tmp_path, n_subjects=3, n_pred_times=10)
        write_dir = str(tmp_path / "out")
        os.makedirs(f"{write_dir}/train", exist_ok=True)
        _collate_shard(
            "0.parquet", "train", write_dir, task_dir, DURATIONS, CODES, sample_times_per_subject=cap, seed=42
        )

        df = pl.read_parquet(str(tmp_path / "out" / "train" / "0.parquet"))
        # Each sampled prediction_time gets exactly one row (one duration+code),
        # so the number of unique prediction_times per subject is bounded by cap.
        per_subj = df.group_by("subject_id").agg(pl.col("prediction_time").n_unique().alias("n_times"))
        assert per_subj["n_times"].max() <= cap

    def test_cap_larger_than_available(self, tmp_path):
        """When cap exceeds available prediction times, use all of them."""
        n_pred = 3
        task_dir, _ = _build_task_dir(tmp_path, n_subjects=1, n_pred_times=n_pred)
        write_dir = str(tmp_path / "out")
        os.makedirs(f"{write_dir}/train", exist_ok=True)
        _collate_shard(
            "0.parquet", "train", write_dir, task_dir, DURATIONS, CODES, sample_times_per_subject=100, seed=42
        )

        df = pl.read_parquet(str(tmp_path / "out" / "train" / "0.parquet"))
        assert df.get_column("prediction_time").n_unique() <= n_pred


class TestCollateShardOutputSchema:
    """Output has expected columns and types."""

    def test_columns(self, tmp_path):
        task_dir, _ = _build_task_dir(tmp_path)
        write_dir = str(tmp_path / "out")
        os.makedirs(f"{write_dir}/train", exist_ok=True)
        _collate_shard(
            "0.parquet", "train", write_dir, task_dir, DURATIONS, CODES, sample_times_per_subject=20, seed=42
        )

        df = pl.read_parquet(str(tmp_path / "out" / "train" / "0.parquet"))
        expected_cols = {"subject_id", "prediction_time", "boolean_value", "duration_days", "query", "occurs"}
        assert set(df.columns) == expected_cols

    def test_no_null_in_occurs(self, tmp_path):
        task_dir, _ = _build_task_dir(tmp_path)
        write_dir = str(tmp_path / "out")
        os.makedirs(f"{write_dir}/train", exist_ok=True)
        _collate_shard(
            "0.parquet", "train", write_dir, task_dir, DURATIONS, CODES, sample_times_per_subject=20, seed=42
        )

        df = pl.read_parquet(str(tmp_path / "out" / "train" / "0.parquet"))
        assert df["occurs"].null_count() == 0


class TestCollateShardValueValidity:
    """All duration_days and query values come from the provided inputs."""

    def test_durations_valid(self, tmp_path):
        task_dir, _ = _build_task_dir(tmp_path)
        write_dir = str(tmp_path / "out")
        os.makedirs(f"{write_dir}/train", exist_ok=True)
        _collate_shard(
            "0.parquet", "train", write_dir, task_dir, DURATIONS, CODES, sample_times_per_subject=20, seed=42
        )

        df = pl.read_parquet(str(tmp_path / "out" / "train" / "0.parquet"))
        assert set(df["duration_days"].unique().to_list()).issubset(set(DURATIONS))

    def test_codes_valid(self, tmp_path):
        task_dir, _ = _build_task_dir(tmp_path)
        write_dir = str(tmp_path / "out")
        os.makedirs(f"{write_dir}/train", exist_ok=True)
        _collate_shard(
            "0.parquet", "train", write_dir, task_dir, DURATIONS, CODES, sample_times_per_subject=20, seed=42
        )

        df = pl.read_parquet(str(tmp_path / "out" / "train" / "0.parquet"))
        assert set(df["query"].unique().to_list()).issubset(set(CODES))


class TestCollateShardIdempotency:
    """If output file exists, _collate_shard skips without overwriting."""

    def test_skip_existing(self, tmp_path):
        task_dir, _ = _build_task_dir(tmp_path)
        write_dir = str(tmp_path / "out")
        os.makedirs(f"{write_dir}/train", exist_ok=True)

        _collate_shard(
            "0.parquet", "train", write_dir, task_dir, DURATIONS, CODES, sample_times_per_subject=20, seed=42
        )

        out_path = tmp_path / "out" / "train" / "0.parquet"
        mtime_before = out_path.stat().st_mtime

        # Run again — should skip
        _collate_shard(
            "0.parquet", "train", write_dir, task_dir, DURATIONS, CODES, sample_times_per_subject=20, seed=99
        )

        mtime_after = out_path.stat().st_mtime
        assert mtime_before == mtime_after


class TestCollateShardEdgeCases:
    """Edge cases: empty input, single subject, single duration/code."""

    def test_empty_shard(self, tmp_path):
        """Empty source parquet → no output file written."""
        task_dir = str(tmp_path / "tasks")
        empty_df = pl.DataFrame(
            {
                "subject_id": pl.Series([], dtype=pl.Int64),
                "prediction_time": pl.Series([], dtype=pl.Datetime),
                "censored": pl.Series([], dtype=pl.Boolean),
                "ICD//A01": pl.Series([], dtype=pl.Boolean),
            }
        )
        for d in DURATIONS:
            split_dir = tmp_path / "tasks" / str(d) / "train"
            split_dir.mkdir(parents=True)
            empty_df.write_parquet(str(split_dir / "0.parquet"))

        write_dir = str(tmp_path / "out")
        os.makedirs(f"{write_dir}/train", exist_ok=True)
        _collate_shard(
            "0.parquet",
            "train",
            write_dir,
            task_dir,
            DURATIONS,
            ["ICD//A01"],
            sample_times_per_subject=20,
            seed=42,
        )

        assert not os.path.exists(f"{write_dir}/train/0.parquet")

    def test_single_duration_single_code(self, tmp_path):
        """With one duration and one code, every row gets that duration and code."""
        durations = [30]
        codes = ["ICD//A01"]
        task_dir, _ = _build_task_dir(tmp_path, durations=durations, codes=["ICD//A01"])
        write_dir = str(tmp_path / "out")
        os.makedirs(f"{write_dir}/train", exist_ok=True)
        _collate_shard(
            "0.parquet", "train", write_dir, task_dir, durations, codes, sample_times_per_subject=20, seed=42
        )

        df = pl.read_parquet(str(tmp_path / "out" / "train" / "0.parquet"))
        assert df["duration_days"].unique().to_list() == [30]
        assert df["query"].unique().to_list() == ["ICD//A01"]

    def test_row_count_equals_sampled_times(self, tmp_path):
        """Each sampled prediction_time gets exactly one row (one duration+code assignment)."""
        task_dir, _ = _build_task_dir(tmp_path, n_subjects=2, n_pred_times=4)
        write_dir = str(tmp_path / "out")
        os.makedirs(f"{write_dir}/train", exist_ok=True)
        _collate_shard(
            "0.parquet", "train", write_dir, task_dir, DURATIONS, CODES, sample_times_per_subject=20, seed=42
        )

        df = pl.read_parquet(str(tmp_path / "out" / "train" / "0.parquet"))
        n_unique_keys = df.select("subject_id", "prediction_time").unique().height
        assert len(df) == n_unique_keys


# ---------------------------------------------------------------------------
# collate_tasks tests
# ---------------------------------------------------------------------------


class TestCollateTasksHashDeterminism:
    """Same inputs → same hash directory."""

    def test_same_inputs_same_hash(self, tmp_path):
        task_dir, _ = _build_task_dir(tmp_path)
        cfg = _make_cfg(task_dir)
        # Write sampled_durations.json so we control durations exactly
        with open(f"{task_dir}/sampled_durations.json", "w") as f:
            json.dump(DURATIONS, f)

        result1 = collate_tasks(cfg)
        result2 = collate_tasks(cfg)
        assert result1 == result2


class TestCollateTasksCodeOrderInvariance:
    """Code order should not affect the hash (codes are sorted internally)."""

    def test_code_order_invariant(self, tmp_path):
        task_dir, _ = _build_task_dir(tmp_path)
        with open(f"{task_dir}/sampled_durations.json", "w") as f:
            json.dump(DURATIONS, f)

        cfg_a = _make_cfg(task_dir, codes=["ICD//A01", "ICD//B02", "MED//C03"])
        cfg_b = _make_cfg(task_dir, codes=["MED//C03", "ICD//A01", "ICD//B02"])

        assert collate_tasks(cfg_a) == collate_tasks(cfg_b)


class TestCollateTasksDurationOrderInvariance:
    """Duration order should not affect the hash (durations are sorted internally)."""

    def test_duration_order_invariant(self, tmp_path):
        task_dir, _ = _build_task_dir(tmp_path)
        with open(f"{task_dir}/sampled_durations.json", "w") as f:
            json.dump([180, 30, 90], f)
        result1 = collate_tasks(_make_cfg(task_dir))

        with open(f"{task_dir}/sampled_durations.json", "w") as f:
            json.dump([30, 90, 180], f)
        result2 = collate_tasks(_make_cfg(task_dir))

        assert result1 == result2


class TestCollateTasksDirectoryStructure:
    """Creates expected output directories and processes all shards."""

    def test_creates_split_dirs(self, tmp_path):
        task_dir, _ = _build_task_dir(tmp_path)
        with open(f"{task_dir}/sampled_durations.json", "w") as f:
            json.dump(DURATIONS, f)

        write_dir = collate_tasks(_make_cfg(task_dir))
        assert os.path.isdir(f"{write_dir}/train")
        assert os.path.isdir(f"{write_dir}/tuning")

    def test_all_shards_processed(self, tmp_path):
        task_dir, _ = _build_task_dir(tmp_path, shard_name="0.parquet")
        # Add a second shard
        shard_df = _make_shard_df()
        for d in DURATIONS:
            for split in ("train", "tuning"):
                shard_df.write_parquet(f"{task_dir}/{d}/{split}/1.parquet")

        with open(f"{task_dir}/sampled_durations.json", "w") as f:
            json.dump(DURATIONS, f)

        write_dir = collate_tasks(_make_cfg(task_dir))
        for split in ("train", "tuning"):
            assert os.path.exists(f"{write_dir}/{split}/0.parquet")
            assert os.path.exists(f"{write_dir}/{split}/1.parquet")


class TestCollateTasksDurationSource:
    """Uses sampled_durations.json when present, otherwise falls back to range."""

    def test_reads_json_durations(self, tmp_path):
        custom_durations = [7, 14, 28]
        task_dir, _ = _build_task_dir(tmp_path, durations=custom_durations)
        with open(f"{task_dir}/sampled_durations.json", "w") as f:
            json.dump(custom_durations, f)

        cfg = _make_cfg(task_dir, codes=CODES)
        write_dir = collate_tasks(cfg)

        # The hash should incorporate these custom durations
        import hashlib

        task_str = f"{'|'.join(sorted(CODES))}_{'|'.join(str(d) for d in sorted(custom_durations))}"
        expected_hash = hashlib.md5(task_str.encode()).hexdigest()
        assert expected_hash in write_dir

    def test_returns_correct_path_format(self, tmp_path):
        task_dir, _ = _build_task_dir(tmp_path)
        with open(f"{task_dir}/sampled_durations.json", "w") as f:
            json.dump(DURATIONS, f)

        write_dir = collate_tasks(_make_cfg(task_dir))
        assert write_dir.startswith(f"{task_dir}/collated/")
        # Hash is 32 hex chars
        hash_part = write_dir.split("/collated/")[1]
        assert len(hash_part) == 32
        assert all(c in "0123456789abcdef" for c in hash_part)
