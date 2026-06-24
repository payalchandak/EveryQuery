"""Unit tests for ``every_query.train.train.validate_training_config`` (the #184 post-compose gate).

These replace the old blind ``ensure_env()`` env-var presence check: validation now runs against the
*resolved* Hydra config, so an overridden node needs no backing env var, and a disabled logger needs
no ``WANDB_ENTITY``.
"""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from every_query.train.train import validate_training_config


def _make_cfg(
    *,
    tensorized_cohort_dir,
    task_labels_dir,
    output_dir,
    logger,
) -> "OmegaConf":
    return OmegaConf.create(
        {
            "output_dir": output_dir,
            "datamodule": {
                "config": {
                    "tensorized_cohort_dir": tensorized_cohort_dir,
                    "task_labels_dir": task_labels_dir,
                }
            },
            "trainer": {"logger": logger, "default_root_dir": output_dir},
        }
    )


@pytest.fixture
def existing_dirs(tmp_path: Path) -> tuple[Path, Path]:
    cohort = tmp_path / "cohort"
    tasks = tmp_path / "tasks"
    cohort.mkdir()
    tasks.mkdir()
    return cohort, tasks


def test_valid_config_no_logger_passes(existing_dirs, tmp_path):
    cohort, tasks = existing_dirs
    cfg = _make_cfg(
        tensorized_cohort_dir=str(cohort),
        task_labels_dir=str(tasks),
        output_dir=str(tmp_path / "does_not_exist_yet"),  # write target, need not pre-exist
        logger=False,
    )
    validate_training_config(cfg)  # should not raise


def test_missing_task_labels_dir_raises(existing_dirs, tmp_path):
    cohort, _ = existing_dirs
    cfg = _make_cfg(
        tensorized_cohort_dir=str(cohort),
        task_labels_dir=None,
        output_dir=str(tmp_path / "out"),
        logger=False,
    )
    with pytest.raises(ValueError, match="task_labels_dir"):
        validate_training_config(cfg)


def test_nonexistent_task_labels_dir_raises(existing_dirs, tmp_path):
    cohort, _ = existing_dirs
    cfg = _make_cfg(
        tensorized_cohort_dir=str(cohort),
        task_labels_dir=str(tmp_path / "nope"),
        output_dir=str(tmp_path / "out"),
        logger=False,
    )
    with pytest.raises(NotADirectoryError, match="task_labels_dir"):
        validate_training_config(cfg)


def test_missing_tensorized_cohort_dir_raises(existing_dirs, tmp_path):
    _, tasks = existing_dirs
    cfg = _make_cfg(
        tensorized_cohort_dir=None,
        task_labels_dir=str(tasks),
        output_dir=str(tmp_path / "out"),
        logger=False,
    )
    with pytest.raises(ValueError, match="tensorized_cohort_dir"):
        validate_training_config(cfg)


def test_missing_output_dir_raises(existing_dirs):
    cohort, tasks = existing_dirs
    cfg = _make_cfg(
        tensorized_cohort_dir=str(cohort),
        task_labels_dir=str(tasks),
        output_dir=None,
        logger=False,
    )
    with pytest.raises(ValueError, match="output_dir"):
        validate_training_config(cfg)


def test_wandb_logger_without_entity_raises(existing_dirs, tmp_path):
    cohort, tasks = existing_dirs
    cfg = _make_cfg(
        tensorized_cohort_dir=str(cohort),
        task_labels_dir=str(tasks),
        output_dir=str(tmp_path / "out"),
        logger={
            "_target_": "pytorch_lightning.loggers.wandb.WandbLogger",
            "entity": None,
        },
    )
    with pytest.raises(ValueError, match="entity"):
        validate_training_config(cfg)


def test_wandb_logger_with_entity_passes(existing_dirs, tmp_path):
    cohort, tasks = existing_dirs
    cfg = _make_cfg(
        tensorized_cohort_dir=str(cohort),
        task_labels_dir=str(tasks),
        output_dir=str(tmp_path / "out"),
        logger={
            "_target_": "pytorch_lightning.loggers.wandb.WandbLogger",
            "entity": "my-entity",
        },
    )
    validate_training_config(cfg)  # should not raise


def test_non_wandb_logger_without_entity_passes(existing_dirs, tmp_path):
    cohort, tasks = existing_dirs
    cfg = _make_cfg(
        tensorized_cohort_dir=str(cohort),
        task_labels_dir=str(tasks),
        output_dir=str(tmp_path / "out"),
        logger={"_target_": "lightning.pytorch.loggers.CSVLogger"},
    )
    validate_training_config(cfg)  # should not raise
