"""Foundation-level integration tests — just verify the CLI-chain fixtures wire up.

Each real per-stage test (``test_process_data.py``, ``test_generate_tasks.py``,
``test_train.py``, etc.) will land as its own PR under the #104 umbrella.  This file is
the minimal sanity check that the ``eq_preprocessed_dataset`` → ``eq_sampled_tasks_dir``
→ ``eq_trained_model_dir`` chain actually runs end-to-end on ``simple_static_MEDS``.

Kept small on purpose: the per-stage assertion logic lives in its own PR.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from pathlib import Path


def test_preprocess_produces_metadata(eq_preprocessed_dataset: Path) -> None:
    """EQ_process_data writes a vocabulary / codes metadata parquet."""
    assert (eq_preprocessed_dataset / "metadata" / "codes.parquet").exists()
    # Tokenization artifacts should be present — downstream MTD consumers need them.
    assert (eq_preprocessed_dataset / "tokenization" / "event_seqs").exists()


def test_generate_tasks_writes_both_splits(eq_sampled_tasks_dir: Path) -> None:
    """EQ_generate_tasks produces at least one labeled parquet per split."""
    for split in ("train", "tuning"):
        fps = list((eq_sampled_tasks_dir / split).glob("*.parquet"))
        assert fps, f"no labeled parquet found under {eq_sampled_tasks_dir / split}"
        cols = set(pl.scan_parquet(fps[0]).collect_schema().names())
        assert cols >= {
            "subject_id",
            "prediction_time",
            "boolean_value",
            "occurs",
            "query",
            "duration_days",
        }, f"unexpected columns in {fps[0]}: {cols}"


def test_generate_tasks_out_dir_contains_only_label_parquets(eq_sampled_tasks_dir: Path) -> None:
    """Regression guard for the sampler's ``out_dir`` / cache-dir separation.

    Every ``*.parquet`` under ``eq_sampled_tasks_dir`` (the value users pass to
    ``EQ_train``'s ``datamodule.config.task_labels_dir``) must conform to the label schema.
    If the sampler regresses and starts leaking its caching artifacts (especially the
    unlabeled 4-column index parquet) back under ``out_dir``, MTD's
    ``MEDSTorchDataConfig.task_labels_fps`` glob will pick them up and training fails with
    a confusing ``polars ShapeError``.  This test catches that regression directly.
    """
    required_label_cols = {
        "subject_id",
        "prediction_time",
        "boolean_value",
        "occurs",
        "query",
        "duration_days",
    }
    for fp in eq_sampled_tasks_dir.rglob("*.parquet"):
        cols = set(pl.scan_parquet(fp).collect_schema().names())
        assert required_label_cols.issubset(cols), (
            f"non-label parquet leaked under the sampler out_dir: {fp} has columns {cols}.  "
            f"MTD's task_labels_fps globs recursively, so non-label parquets here will break "
            f"downstream training.  Caching artifacts must live outside out_dir."
        )


def test_train_produces_checkpoint(eq_trained_model_dir: Path) -> None:
    """EQ_train --config-name=_demo_train produces a checkpoint and resolved config."""
    ckpts = list((eq_trained_model_dir / "checkpoints").glob("*.ckpt"))
    assert ckpts, f"no checkpoint under {eq_trained_model_dir / 'checkpoints'}"
