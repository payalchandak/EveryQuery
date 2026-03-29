"""CLI-level integration tests for ``every_query.train``.

Tests exercise ``collate_tasks`` (direct call) and the full ``train.main``
Hydra entry-point (via subprocess) using the ``_demo_train.yaml`` config.
"""

import os
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest
from meds import train_split, tuning_split
from omegaconf import DictConfig, OmegaConf

from every_query.train import collate_tasks

_VENV_BIN = str(Path(sys.executable).parent)


def _build_collate_cfg(task_parquet_dir: Path) -> DictConfig:
    """Minimal DictConfig for ``collate_tasks`` matching the ``task_parquet_dir`` fixture."""
    return OmegaConf.create(
        {
            "query": {
                "task_dir": str(task_parquet_dir),
                "codes": ["HR", "TEMP"],
                "duration_min": 30,
                "duration_max": 32,
                "sample_times_per_subject": 5,
            },
            "seed": 1,
        }
    )


def _run_train_subprocess(
    task_parquet_dir: Path,
    tensorized_cohort_dir: Path,
    output_dir: Path,
    *,
    do_resume: bool = False,
    do_overwrite: bool = True,
    extra_overrides: list[str] | None = None,
) -> subprocess.CompletedProcess:
    """Run ``python -m every_query.train`` as a subprocess with the demo config."""
    env = os.environ.copy()
    env["PATH"] = _VENV_BIN + os.pathsep + env.get("PATH", "")

    overrides = [
        f"output_dir={output_dir}",
        f"query.task_dir={task_parquet_dir}",
        f"datamodule.config.tensorized_cohort_dir={tensorized_cohort_dir}",
        f"do_resume={str(do_resume).lower()}",
        f"do_overwrite={str(do_overwrite).lower()}",
        f"hydra.run.dir={output_dir}/.hydra_run",
    ]
    if extra_overrides:
        overrides.extend(extra_overrides)

    cmd = [
        sys.executable,
        "-m",
        "every_query.train",
        "--config-name=_demo_train",
        *overrides,
    ]

    return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=180)


# ── test_collate_tasks ──────────────────────────────────────────────────


class TestCollateTasks:
    """``collate_tasks`` reads per-duration task parquets and writes collated shards."""

    @pytest.fixture()
    def collated_dir(self, task_parquet_dir) -> Path:
        cfg = _build_collate_cfg(task_parquet_dir)
        return Path(collate_tasks(cfg))

    def test_creates_collated_directory(self, collated_dir):
        """Output lives under ``{task_dir}/collated/{hash}/``."""
        assert collated_dir.is_dir()
        assert collated_dir.parent.name == "collated"

    def test_train_and_tuning_splits_written(self, collated_dir):
        """Both train and tuning splits contain at least one parquet shard."""
        for split in [train_split, tuning_split]:
            split_dir = collated_dir / split
            assert split_dir.is_dir(), f"Missing {split} split directory"
            parquets = list(split_dir.glob("*.parquet"))
            assert len(parquets) > 0, f"No parquets in {split} split"

    def test_output_parquet_columns(self, collated_dir):
        """Collated parquets have the expected EveryQuery task schema."""
        df = pl.read_parquet(collated_dir / train_split / "0.parquet")
        expected = {"subject_id", "prediction_time", "boolean_value", "occurs", "query", "duration_days"}
        assert expected == set(df.columns)


# ── test_train_cli_runs ─────────────────────────────────────────────────


class TestTrainCliRuns:
    """Full training run via subprocess with ``_demo_train.yaml``."""

    @pytest.fixture(scope="class")
    def training_output(self, task_parquet_dir, tensorized_cohort_dir, tmp_path_factory) -> Path:
        output_dir = tmp_path_factory.mktemp("cli_train")
        result = _run_train_subprocess(task_parquet_dir, tensorized_cohort_dir, output_dir)
        assert result.returncode == 0, (
            f"train.py failed (rc={result.returncode}).\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        return output_dir

    def test_config_yaml_written(self, training_output):
        assert (training_output / "config.yaml").is_file()

    def test_resolved_config_yaml_written(self, training_output):
        assert (training_output / "resolved_config.yaml").is_file()

    def test_checkpoints_directory_has_ckpt(self, training_output):
        ckpt_dir = training_output / "checkpoints"
        assert ckpt_dir.is_dir()
        ckpts = list(ckpt_dir.glob("*.ckpt"))
        assert len(ckpts) >= 1

    def test_best_model_ckpt_exists(self, training_output):
        assert (training_output / "best_model.ckpt").is_file()


# ── test_train_resume ───────────────────────────────────────────────────


class TestTrainResume:
    """Resuming from an existing checkpoint completes without error."""

    @pytest.fixture(scope="class")
    def resumed_output(self, task_parquet_dir, tensorized_cohort_dir, tmp_path_factory) -> Path:
        output_dir = tmp_path_factory.mktemp("cli_resume")

        initial = _run_train_subprocess(task_parquet_dir, tensorized_cohort_dir, output_dir)
        assert initial.returncode == 0, (
            f"Initial training failed (rc={initial.returncode}).\nstderr:\n{initial.stderr}"
        )

        resumed = _run_train_subprocess(
            task_parquet_dir,
            tensorized_cohort_dir,
            output_dir,
            do_resume=True,
            do_overwrite=False,
            extra_overrides=["trainer.max_steps=4"],
        )
        assert resumed.returncode == 0, (
            f"Resumed training failed (rc={resumed.returncode}).\nstderr:\n{resumed.stderr}"
        )
        return output_dir

    def test_checkpoints_present_after_resume(self, resumed_output):
        ckpt_dir = resumed_output / "checkpoints"
        assert ckpt_dir.is_dir()
        ckpts = list(ckpt_dir.glob("*.ckpt"))
        assert len(ckpts) >= 1

    def test_best_model_present_after_resume(self, resumed_output):
        assert (resumed_output / "best_model.ckpt").is_file()
