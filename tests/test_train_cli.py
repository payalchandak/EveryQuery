"""CLI-level integration tests for ``every_query.train``.

Tests exercise the full ``train.main`` Hydra entry-point (via subprocess) using the
``_demo_train.yaml`` config against sampler-shaped task labels.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

_VENV_BIN = str(Path(sys.executable).parent)


def _run_train_subprocess(
    task_labels_dir: Path,
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
    # Provide dummy env vars so ensure_env() passes in the subprocess.
    # Hydra CLI overrides control the actual paths used by the test.
    for var in (
        "PROJECT_DIR",
        "OUTPUT_DIR",
        "TASK_DIR",
        "PROCESSED",
        "INTERMEDIATE",
        "FINAL_DATA_DIR",
    ):
        env.setdefault(var, str(output_dir))
    env.setdefault("WANDB_ENTITY", "test")

    overrides = [
        f"output_dir={output_dir}",
        f"datamodule.config.task_labels_dir={task_labels_dir}",
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


class TestTrainCliRuns:
    """Full training run via subprocess with ``_demo_train.yaml``."""

    @pytest.fixture(scope="class")
    def training_output(self, task_labels_dir, tensorized_cohort_dir, tmp_path_factory) -> Path:
        output_dir = tmp_path_factory.mktemp("cli_train")
        result = _run_train_subprocess(task_labels_dir, tensorized_cohort_dir, output_dir)
        assert result.returncode == 0, (
            f"train.py failed (rc={result.returncode}).\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
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


class TestTrainResume:
    """Resuming from an existing checkpoint completes without error."""

    @pytest.fixture(scope="class")
    def resumed_output(self, task_labels_dir, tensorized_cohort_dir, tmp_path_factory) -> Path:
        output_dir = tmp_path_factory.mktemp("cli_resume")

        initial = _run_train_subprocess(task_labels_dir, tensorized_cohort_dir, output_dir)
        assert initial.returncode == 0, (
            f"Initial training failed (rc={initial.returncode}).\nstderr:\n{initial.stderr}"
        )

        resumed = _run_train_subprocess(
            task_labels_dir,
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
