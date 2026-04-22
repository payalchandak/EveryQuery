"""Tests for ``PerTaskAurocCallback`` — disabled paths and metric-logging surface.

The full inference path (construct an MTD datamodule, iterate, run the model,
compute metrics) is exercised end-to-end by the CLI smoke tests for
``EQ_generate_tracking_tasks`` paired with ``EQ_train``; here we focus on the
guard rails (disabled when no labels) and the metric-logging mapping (per-task
scalars + aggregates wired through ``pl_module.log``).
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import polars as pl
import pytest

from every_query.train.callbacks.per_task_auroc import (
    PerTaskAurocCallback,
    _slug_for_metric,
)


def _trainer(global_step: int = 0, datamodule=None) -> SimpleNamespace:
    """Minimal stand-in for ``pl.Trainer`` — just the attrs the callback reads."""
    return SimpleNamespace(global_step=global_step, datamodule=datamodule, is_global_zero=True)


def _pl_module() -> MagicMock:
    m = MagicMock()
    m.log = MagicMock()
    return m


# ----------------------------------------------------------------------
# Slug helper
# ----------------------------------------------------------------------


class TestSlugForMetric:
    def test_basic_slug(self):
        assert _slug_for_metric("ICD//I10", 30.0) == "ICD__I10@30d"

    def test_no_slashes(self):
        assert _slug_for_metric("HR", 7.0) == "HR@7d"

    def test_truncates_fractional_duration(self):
        # WandB tags shouldn't carry float noise; durations are conceptually integer-day.
        assert _slug_for_metric("X", 30.7) == "X@30d"


# ----------------------------------------------------------------------
# Disabled paths
# ----------------------------------------------------------------------


class TestDisabledPaths:
    """The callback is opt-in.  When `tracking_dir` is unset or the labels dir is
    missing/empty, training must proceed exactly as if the callback weren't there.
    """

    def test_init_rejects_zero_step(self):
        with pytest.raises(ValueError, match="every_n_steps"):
            PerTaskAurocCallback(every_n_steps=0)

    def test_no_tracking_dir_disables(self):
        cb = PerTaskAurocCallback(tracking_dir=None)
        cb.setup(_trainer(), _pl_module(), stage="fit")
        assert cb._labels_dir is None

        cb.on_train_batch_end(_trainer(global_step=2000), _pl_module(), None, None, 0)
        # No exception, no datamodule access — successful no-op.

    def test_missing_labels_dir_disables(self, tmp_path: Path):
        # Pass a real dir but no `labels/` subdir inside it.
        cb = PerTaskAurocCallback(tracking_dir=str(tmp_path), every_n_steps=10)
        cb.setup(_trainer(), _pl_module(), stage="fit")
        assert cb._labels_dir is None

    def test_empty_labels_dir_disables(self, tmp_path: Path):
        (tmp_path / "labels").mkdir()
        cb = PerTaskAurocCallback(tracking_dir=str(tmp_path))
        cb.setup(_trainer(), _pl_module(), stage="fit")
        assert cb._labels_dir is None

    def test_setup_outside_fit_is_noop(self, tmp_path: Path):
        labels = tmp_path / "labels"
        labels.mkdir()
        (labels / "shard.parquet").touch()
        cb = PerTaskAurocCallback(tracking_dir=str(tmp_path))
        cb.setup(_trainer(), _pl_module(), stage="validate")
        assert cb._labels_dir is None  # only `fit` enables

    def test_step_zero_does_not_fire(self, tmp_path: Path, monkeypatch):
        labels = tmp_path / "labels"
        labels.mkdir()
        (labels / "shard.parquet").touch()
        cb = PerTaskAurocCallback(tracking_dir=str(tmp_path), every_n_steps=10)
        cb.setup(_trainer(), _pl_module(), stage="fit")
        assert cb._labels_dir == labels

        called = []
        monkeypatch.setattr(cb, "_run_per_task_pass", lambda *a, **kw: called.append(True))

        # Step 0 is treated as "model is still random" — skip it even though 0 % 10 == 0.
        cb.on_train_batch_end(_trainer(global_step=0), _pl_module(), None, None, 0)
        assert called == []

    def test_off_cadence_step_does_not_fire(self, tmp_path: Path, monkeypatch):
        labels = tmp_path / "labels"
        labels.mkdir()
        (labels / "shard.parquet").touch()
        cb = PerTaskAurocCallback(tracking_dir=str(tmp_path), every_n_steps=10)
        cb.setup(_trainer(), _pl_module(), stage="fit")

        called = []
        monkeypatch.setattr(cb, "_run_per_task_pass", lambda *a, **kw: called.append(True))

        cb.on_train_batch_end(_trainer(global_step=7), _pl_module(), None, None, 0)
        assert called == []

    def test_dedup_within_one_step(self, tmp_path: Path, monkeypatch):
        """Gradient accumulation calls ``on_train_batch_end`` once per micro-batch but
        ``trainer.global_step`` only advances on optimizer step.  Make sure we fire once
        per (firing) step regardless of how many times the hook is invoked at that step.
        """
        labels = tmp_path / "labels"
        labels.mkdir()
        (labels / "shard.parquet").touch()
        cb = PerTaskAurocCallback(tracking_dir=str(tmp_path), every_n_steps=10)
        cb.setup(_trainer(), _pl_module(), stage="fit")

        called = []
        monkeypatch.setattr(cb, "_run_per_task_pass", lambda *a, **kw: called.append(True))

        for _ in range(4):
            cb.on_train_batch_end(_trainer(global_step=10), _pl_module(), None, None, 0)
        assert len(called) == 1

    def test_non_rank_zero_skips(self, tmp_path: Path, monkeypatch):
        labels = tmp_path / "labels"
        labels.mkdir()
        (labels / "shard.parquet").touch()
        cb = PerTaskAurocCallback(tracking_dir=str(tmp_path), every_n_steps=10)
        cb.setup(_trainer(), _pl_module(), stage="fit")

        called = []
        monkeypatch.setattr(cb, "_run_per_task_pass", lambda *a, **kw: called.append(True))

        non_zero_trainer = SimpleNamespace(global_step=10, datamodule=None, is_global_zero=False)
        cb.on_train_batch_end(non_zero_trainer, _pl_module(), None, None, 0)
        assert called == []

    def test_incompatible_datamodule_returns_none(self, tmp_path: Path):
        """Tests built on a non-MTD datamodule should not crash the callback."""
        labels = tmp_path / "labels"
        labels.mkdir()
        (labels / "shard.parquet").touch()
        cb = PerTaskAurocCallback(tracking_dir=str(tmp_path), every_n_steps=10)
        cb.setup(_trainer(), _pl_module(), stage="fit")

        bare_trainer = SimpleNamespace(
            global_step=10,
            datamodule=SimpleNamespace(),  # no .config / .data_class
            is_global_zero=True,
        )
        # Returns None silently; doesn't raise.
        assert cb._build_dataloader(bare_trainer) is None


# ----------------------------------------------------------------------
# Metric-logging surface
# ----------------------------------------------------------------------


class TestLogMetrics:
    """``_log_metrics`` is the boundary between ``compute_metrics()``'s output frame
    and the trainer's logger.  Verify the per-task scalars + aggregates land with the
    right names and that ``None`` AUROCs (single-class groups) are skipped.
    """

    @staticmethod
    def _metrics_df() -> pl.DataFrame:
        return pl.DataFrame(
            {
                "query": ["ICD//I10", "HR", "TEMP"],
                "duration_days": pl.Series([30.0, 7.0, 365.0], dtype=pl.Float32),
                "n_rows": [100, 100, 100],
                "n_occurs_labeled": [80, 80, 0],  # third group fully censored
                "n_positive": [10, 40, 0],
                "occurs_auroc": [0.85, 0.60, None],  # third is None
                "censor_auroc": [0.92, None, 0.70],  # second is None
            }
        )

    def test_per_task_scalars_logged(self):
        cb = PerTaskAurocCallback(tracking_dir=None, prefix="tuning/per_task")
        pl_mod = _pl_module()
        cb._log_metrics(pl_mod, self._metrics_df())

        log_keys = {call.args[0] for call in pl_mod.log.call_args_list}

        # First row — both AUROCs present
        assert "tuning/per_task/ICD__I10@30d/occurs_auroc" in log_keys
        assert "tuning/per_task/ICD__I10@30d/censor_auroc" in log_keys
        # Second row — censor_auroc is None, must NOT appear
        assert "tuning/per_task/HR@7d/occurs_auroc" in log_keys
        assert "tuning/per_task/HR@7d/censor_auroc" not in log_keys
        # Third row — occurs_auroc is None, must NOT appear
        assert "tuning/per_task/TEMP@365d/occurs_auroc" not in log_keys
        assert "tuning/per_task/TEMP@365d/censor_auroc" in log_keys

    def test_aggregates_logged(self):
        cb = PerTaskAurocCallback(tracking_dir=None, prefix="tuning/per_task")
        pl_mod = _pl_module()
        cb._log_metrics(pl_mod, self._metrics_df())

        logged = {call.args[0]: call.args[1] for call in pl_mod.log.call_args_list}

        # Two non-null occurs_auroc values: 0.85, 0.60 → mean 0.725, median 0.725.
        assert logged["tuning/per_task/occurs_auroc_mean"] == pytest.approx(0.725)
        assert logged["tuning/per_task/occurs_auroc_median"] == pytest.approx(0.725)
        assert logged["tuning/per_task/n_groups_with_occurs_auroc"] == 2.0

    def test_per_task_log_values_match_input(self):
        cb = PerTaskAurocCallback(tracking_dir=None, prefix="p")
        pl_mod = _pl_module()
        cb._log_metrics(pl_mod, self._metrics_df())

        logged = {call.args[0]: call.args[1] for call in pl_mod.log.call_args_list}
        assert logged["p/ICD__I10@30d/occurs_auroc"] == pytest.approx(0.85)
        assert logged["p/HR@7d/occurs_auroc"] == pytest.approx(0.60)
        assert logged["p/ICD__I10@30d/censor_auroc"] == pytest.approx(0.92)
        assert logged["p/TEMP@365d/censor_auroc"] == pytest.approx(0.70)

    def test_empty_metrics_no_log(self):
        cb = PerTaskAurocCallback(tracking_dir=None)
        pl_mod = _pl_module()
        empty = pl.DataFrame(
            schema={
                "query": pl.Utf8,
                "duration_days": pl.Float32,
                "n_rows": pl.Int64,
                "n_occurs_labeled": pl.Int64,
                "n_positive": pl.Int64,
                "occurs_auroc": pl.Float64,
                "censor_auroc": pl.Float64,
            }
        )
        cb._log_metrics(pl_mod, empty)
        assert pl_mod.log.call_count == 0

    def test_all_null_aurocs_skips_aggregates(self):
        cb = PerTaskAurocCallback(tracking_dir=None, prefix="p")
        pl_mod = _pl_module()
        all_null = pl.DataFrame(
            {
                "query": ["A"],
                "duration_days": pl.Series([30.0], dtype=pl.Float32),
                "n_rows": [10],
                "n_occurs_labeled": [0],
                "n_positive": [0],
                "occurs_auroc": pl.Series([None], dtype=pl.Float64),
                "censor_auroc": pl.Series([None], dtype=pl.Float64),
            }
        )
        cb._log_metrics(pl_mod, all_null)
        # No per-task scalars (both None), no aggregates (no non-null occurs_auroc).
        assert pl_mod.log.call_count == 0
