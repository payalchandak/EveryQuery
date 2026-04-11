"""Tests for eval.py: _setup_model, _run_test, _run_predict."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
import torch
from omegaconf import OmegaConf

from every_query.eval import _run_predict, _run_test, _setup_model
from every_query.utils.codes import code_slug

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

CODES = ["ICD//A01", "ICD//B02", "MED//C03"]
DURATIONS = [30, 90]


def _make_model_run_dir(tmp_path: Path, ckpt_locations: list[str] | None = None) -> Path:
    """Create a fake model run directory with resolved_config.yaml and checkpoint files."""
    run_dir = tmp_path / "model_run"
    run_dir.mkdir()
    (run_dir / "checkpoints").mkdir()

    cfg = {
        "seed": 42,
        "trainer": {"_target_": "lightning.pytorch.Trainer", "accelerator": "cpu", "logger": ""},
        "datamodule": {
            "_target_": "unittest.mock.MagicMock",
            "config": {"task_labels_dir": "", "max_seq_len": 256},
        },
    }
    OmegaConf.save(OmegaConf.create(cfg), run_dir / "resolved_config.yaml")

    if ckpt_locations is None:
        ckpt_locations = ["best_model.ckpt"]
    for loc in ckpt_locations:
        path = run_dir / loc
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake_ckpt")

    return run_dir


def _make_task_dirs(tmp_path: Path, durations: list[int], codes: list[str]) -> Path:
    """Create task_set_dir/{duration}/{code_slug}/ directories."""
    task_set_dir = tmp_path / "task_set"
    task_set_dir.mkdir()
    for d in durations:
        for c in codes:
            (task_set_dir / str(d) / code_slug(c)).mkdir(parents=True)
    return task_set_dir


def _make_mock_model(num_hidden_layers: int = 4) -> MagicMock:
    m = MagicMock()
    m.model.HF_model_config.num_hidden_layers = num_hidden_layers
    return m


def _make_mock_trainer(
    test_return: list | None = None,
    validate_return: list | None = None,
) -> MagicMock:
    trainer = MagicMock()
    trainer.test.return_value = test_return if test_return is not None else [{}]
    trainer.validate.return_value = validate_return if validate_return is not None else [{}]
    return trainer


def _make_predict_batch(n: int, embed_dim: int = 4, start_id: int = 0) -> dict:
    return {
        "subject_id": torch.arange(start_id, start_id + n),
        "prediction_time": torch.arange(1000, 1000 + n),
        "occurs_probs": torch.rand(n),
        "query_embed": torch.rand(n, embed_dim),
    }


def _make_run_test_cfg(
    id_codes: list[str] | None = None,
    ood_codes: list[str] | None = None,
    manual_codes: list[str] | None = None,
    split: str = "held_out",
    ckpt_path: str | None = None,
) -> "DictConfig":  # noqa: F821
    return OmegaConf.create(
        {
            "id_codes": id_codes,
            "ood_codes": ood_codes,
            "manual_codes": manual_codes,
            "split": split,
            "ckpt_path": ckpt_path,
        }
    )


def _make_train_cfg() -> "DictConfig":  # noqa: F821
    return OmegaConf.create(
        {
            "datamodule": {
                "_target_": "unittest.mock.MagicMock",
                "config": {"task_labels_dir": "", "max_seq_len": 256},
            }
        }
    )


# ---------------------------------------------------------------------------
# _setup_model checkpoint resolution
# ---------------------------------------------------------------------------

_SETUP_PATCHES = {
    "load": "every_query.lightning_module.EveryQueryLightningModule.load_from_checkpoint",
    "instantiate": "every_query.eval.instantiate",
    "seed": "every_query.eval.seed_everything",
}


class TestSetupModelCheckpointResolution:
    def _call(self, run_dir, ckpt_name=None):
        with (
            patch(_SETUP_PATCHES["load"], return_value=MagicMock()) as mock_load,
            patch(_SETUP_PATCHES["instantiate"], return_value=MagicMock()),
            patch(_SETUP_PATCHES["seed"]),
        ):
            result = _setup_model(run_dir, ckpt_name=ckpt_name)
            return result, mock_load

    def test_best_model_preferred(self, tmp_path):
        run_dir = _make_model_run_dir(tmp_path, ["best_model.ckpt", "checkpoints/last.ckpt"])
        _, mock_load = self._call(run_dir)
        loaded_path = mock_load.call_args[0][0]
        assert loaded_path == str(run_dir / "best_model.ckpt")

    def test_fallback_to_last(self, tmp_path):
        run_dir = _make_model_run_dir(tmp_path, ["checkpoints/last.ckpt"])
        _, mock_load = self._call(run_dir)
        loaded_path = mock_load.call_args[0][0]
        assert loaded_path == str(run_dir / "checkpoints" / "last.ckpt")

    def test_ckpt_name_best_same_as_none(self, tmp_path):
        run_dir = _make_model_run_dir(tmp_path, ["best_model.ckpt"])
        _, mock_load = self._call(run_dir, ckpt_name="best")
        loaded_path = mock_load.call_args[0][0]
        assert loaded_path == str(run_dir / "best_model.ckpt")

    def test_explicit_ckpt_name(self, tmp_path):
        run_dir = _make_model_run_dir(tmp_path, ["checkpoints/epoch_5.ckpt"])
        _, mock_load = self._call(run_dir, ckpt_name="epoch_5")
        loaded_path = mock_load.call_args[0][0]
        assert loaded_path == str(run_dir / "checkpoints" / "epoch_5.ckpt")

    def test_missing_all_raises(self, tmp_path):
        run_dir = _make_model_run_dir(tmp_path, [])
        with pytest.raises(FileNotFoundError):
            self._call(run_dir)

    def test_missing_named_raises(self, tmp_path):
        run_dir = _make_model_run_dir(tmp_path, ["best_model.ckpt"])
        with pytest.raises(FileNotFoundError):
            self._call(run_dir, ckpt_name="nonexistent")

    def test_nonexistent_dir_raises(self, tmp_path):
        with pytest.raises(NotADirectoryError):
            self._call(tmp_path / "does_not_exist")

    def test_returns_tuple(self, tmp_path):
        run_dir = _make_model_run_dir(tmp_path)
        result, _ = self._call(run_dir)
        assert isinstance(result, tuple)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# _run_test
# ---------------------------------------------------------------------------


@pytest.fixture()
def run_test_deps(tmp_path):
    """Common dependencies for _run_test tests."""
    task_set_dir = _make_task_dirs(tmp_path, DURATIONS, CODES)
    M = _make_mock_model()
    train_cfg = _make_train_cfg()
    return task_set_dir, M, train_cfg


class TestRunTestOutputSchema:
    EXPECTED_COLUMNS = {
        "model",
        "duration_days",
        "code",
        "code_slug",
        "bucket",
        "occurs_auc",
        "censor_auc",
        "num_layers",
        "max_seq_len",
        "eval_time",
        "ckpt",
    }

    def test_output_columns(self, run_test_deps):
        task_set_dir, M, train_cfg = run_test_deps
        cfg = _make_run_test_cfg(id_codes=CODES[:1])
        trainer = _make_mock_trainer()
        with patch("every_query.eval.instantiate", return_value=MagicMock()):
            df = _run_test(cfg, train_cfg, M, trainer, task_set_dir, "test_model", DURATIONS)
        assert set(df.columns) == self.EXPECTED_COLUMNS

    def test_output_dtypes(self, run_test_deps):
        task_set_dir, M, train_cfg = run_test_deps
        cfg = _make_run_test_cfg(
            id_codes=CODES[:1],
            ood_codes=[],
        )
        metrics = {"held_out/occurs_auc": 0.85, "held_out/censor_auc": 0.90}
        trainer = _make_mock_trainer(test_return=[metrics])
        with patch("every_query.eval.instantiate", return_value=MagicMock()):
            df = _run_test(cfg, train_cfg, M, trainer, task_set_dir, "test_model", DURATIONS)
        assert df["duration_days"].dtype == pl.Int64
        assert df["occurs_auc"].dtype == pl.Float64
        assert df["censor_auc"].dtype == pl.Float64
        assert df["eval_time"].dtype == pl.Float64


class TestRunTestRowGeneration:
    def test_all_combinations_produce_rows(self, run_test_deps):
        task_set_dir, M, train_cfg = run_test_deps
        cfg = _make_run_test_cfg(id_codes=CODES, ood_codes=[])
        trainer = _make_mock_trainer()
        with patch("every_query.eval.instantiate", return_value=MagicMock()):
            df = _run_test(cfg, train_cfg, M, trainer, task_set_dir, "m", DURATIONS)
        assert len(df) == len(CODES) * len(DURATIONS)

    def test_missing_dir_skipped(self, tmp_path):
        # Only create dirs for 3 of 4 code/duration combos
        codes = CODES[:2]
        durations = [30, 90]
        task_set_dir = tmp_path / "task_set"
        task_set_dir.mkdir()
        created = 0
        for d in durations:
            for c in codes:
                if created < 3:
                    (task_set_dir / str(d) / code_slug(c)).mkdir(parents=True)
                    created += 1

        cfg = _make_run_test_cfg(id_codes=codes, ood_codes=[])
        trainer = _make_mock_trainer()
        M = _make_mock_model()
        train_cfg = _make_train_cfg()
        with patch("every_query.eval.instantiate", return_value=MagicMock()):
            df = _run_test(cfg, train_cfg, M, trainer, task_set_dir, "m", durations)
        assert len(df) == 3

    def test_all_missing_returns_empty(self, tmp_path):
        task_set_dir = tmp_path / "task_set"
        task_set_dir.mkdir()
        cfg = _make_run_test_cfg(id_codes=CODES, ood_codes=[])
        trainer = _make_mock_trainer()
        M = _make_mock_model()
        train_cfg = _make_train_cfg()
        with patch("every_query.eval.instantiate", return_value=MagicMock()):
            df = _run_test(cfg, train_cfg, M, trainer, task_set_dir, "m", DURATIONS)
        assert len(df) == 0

    def test_code_order_invariance(self, run_test_deps):
        task_set_dir, M, train_cfg = run_test_deps
        metrics = {"held_out/occurs_auc": 0.8, "held_out/censor_auc": 0.7}
        trainer = _make_mock_trainer(test_return=[metrics])

        cfg1 = _make_run_test_cfg(id_codes=CODES, ood_codes=[])
        cfg2 = _make_run_test_cfg(id_codes=list(reversed(CODES)), ood_codes=[])

        with patch("every_query.eval.instantiate", return_value=MagicMock()):
            df1 = _run_test(cfg1, train_cfg, M, trainer, task_set_dir, "m", DURATIONS)
            df2 = _run_test(cfg2, train_cfg, M, trainer, task_set_dir, "m", DURATIONS)

        df1 = df1.sort("code", "duration_days")
        df2 = df2.sort("code", "duration_days")
        assert df1.select("code", "duration_days", "occurs_auc", "censor_auc").equals(
            df2.select("code", "duration_days", "occurs_auc", "censor_auc")
        )

    def test_duration_order_invariance(self, run_test_deps):
        task_set_dir, M, train_cfg = run_test_deps
        metrics = {"held_out/occurs_auc": 0.8, "held_out/censor_auc": 0.7}
        trainer = _make_mock_trainer(test_return=[metrics])

        cfg = _make_run_test_cfg(id_codes=CODES[:1], ood_codes=[])
        with patch("every_query.eval.instantiate", return_value=MagicMock()):
            df1 = _run_test(cfg, train_cfg, M, trainer, task_set_dir, "m", [30, 90])
            df2 = _run_test(cfg, train_cfg, M, trainer, task_set_dir, "m", [90, 30])

        df1 = df1.sort("duration_days")
        df2 = df2.sort("duration_days")
        assert df1.select("duration_days", "occurs_auc", "censor_auc").equals(
            df2.select("duration_days", "occurs_auc", "censor_auc")
        )


class TestRunTestBucketLabeling:
    def _run(self, tmp_path, id_codes=None, ood_codes=None, manual_codes=None):
        all_codes = (id_codes or []) + (ood_codes or []) + (manual_codes or [])
        task_set_dir = _make_task_dirs(tmp_path, [30], all_codes)
        cfg = _make_run_test_cfg(id_codes=id_codes, ood_codes=ood_codes, manual_codes=manual_codes)
        trainer = _make_mock_trainer()
        M = _make_mock_model()
        train_cfg = _make_train_cfg()
        with patch("every_query.eval.instantiate", return_value=MagicMock()):
            return _run_test(cfg, train_cfg, M, trainer, task_set_dir, "m", [30])

    def test_ood_code_labeled_ood(self, tmp_path):
        df = self._run(tmp_path, ood_codes=["ICD//A01"])
        assert df["bucket"].to_list() == ["ood"]

    def test_id_code_labeled_id(self, tmp_path):
        df = self._run(tmp_path, id_codes=["ICD//A01"], ood_codes=[])
        assert df["bucket"].to_list() == ["id"]

    def test_manual_code_labeled_manual(self, tmp_path):
        df = self._run(tmp_path, manual_codes=["ICD//A01"], ood_codes=[])
        assert df["bucket"].to_list() == ["manual"]

    def test_ood_codes_none_handled(self, tmp_path):
        """After bug fix, ood_codes=None should not crash; all codes get bucket='id'."""
        df = self._run(tmp_path, id_codes=["ICD//A01"], ood_codes=None)
        assert df["bucket"].to_list() == ["id"]


class TestRunTestSplitRouting:
    def test_tuning_calls_validate(self, run_test_deps):
        task_set_dir, M, train_cfg = run_test_deps
        cfg = _make_run_test_cfg(id_codes=CODES[:1], ood_codes=[], split="tuning")
        trainer = _make_mock_trainer(validate_return=[{"tuning/occurs_auc": 0.8}])
        with patch("every_query.eval.instantiate", return_value=MagicMock()):
            _run_test(cfg, train_cfg, M, trainer, task_set_dir, "m", [30])
        trainer.validate.assert_called()
        trainer.test.assert_not_called()

    def test_held_out_calls_test(self, run_test_deps):
        task_set_dir, M, train_cfg = run_test_deps
        cfg = _make_run_test_cfg(id_codes=CODES[:1], ood_codes=[], split="held_out")
        trainer = _make_mock_trainer(test_return=[{"held_out/occurs_auc": 0.8}])
        with patch("every_query.eval.instantiate", return_value=MagicMock()):
            _run_test(cfg, train_cfg, M, trainer, task_set_dir, "m", [30])
        trainer.test.assert_called()
        trainer.validate.assert_not_called()


class TestRunTestMetricExtraction:
    def _run_with_metrics(self, run_test_deps, split, metrics):
        task_set_dir, M, train_cfg = run_test_deps
        cfg = _make_run_test_cfg(id_codes=CODES[:1], ood_codes=[], split=split)
        if split == "tuning":
            trainer = _make_mock_trainer(validate_return=[metrics])
        else:
            trainer = _make_mock_trainer(test_return=[metrics])
        with patch("every_query.eval.instantiate", return_value=MagicMock()):
            return _run_test(cfg, train_cfg, M, trainer, task_set_dir, "m", [30])

    def test_held_out_prefix(self, run_test_deps):
        df = self._run_with_metrics(
            run_test_deps, "held_out", {"held_out/occurs_auc": 0.85, "held_out/censor_auc": 0.90}
        )
        assert df["occurs_auc"][0] == pytest.approx(0.85)
        assert df["censor_auc"][0] == pytest.approx(0.90)

    def test_tuning_prefix(self, run_test_deps):
        df = self._run_with_metrics(
            run_test_deps, "tuning", {"tuning/occurs_auc": 0.75, "tuning/censor_auc": 0.80}
        )
        assert df["occurs_auc"][0] == pytest.approx(0.75)
        assert df["censor_auc"][0] == pytest.approx(0.80)

    def test_missing_metric_becomes_none(self, run_test_deps):
        df = self._run_with_metrics(run_test_deps, "held_out", {})
        assert df["occurs_auc"][0] is None
        assert df["censor_auc"][0] is None

    def test_empty_trainer_output(self, run_test_deps):
        task_set_dir, M, train_cfg = run_test_deps
        cfg = _make_run_test_cfg(id_codes=CODES[:1], ood_codes=[], split="held_out")
        trainer = _make_mock_trainer(test_return=[])
        with patch("every_query.eval.instantiate", return_value=MagicMock()):
            df = _run_test(cfg, train_cfg, M, trainer, task_set_dir, "m", [30])
        assert df["occurs_auc"][0] is None
        assert df["censor_auc"][0] is None


# ---------------------------------------------------------------------------
# _run_predict
# ---------------------------------------------------------------------------


@pytest.fixture()
def run_predict_deps(tmp_path):
    """Common dependencies for _run_predict tests."""
    task_set_dir = _make_task_dirs(tmp_path, DURATIONS, CODES)
    M = _make_mock_model()
    train_cfg = _make_train_cfg()
    return task_set_dir, M, train_cfg


def _make_predict_cfg(
    id_codes: list[str] | None = None,
    ood_codes: list[str] | None = None,
    manual_codes: list[str] | None = None,
) -> "DictConfig":  # noqa: F821
    return OmegaConf.create(
        {
            "id_codes": id_codes,
            "ood_codes": ood_codes,
            "manual_codes": manual_codes,
        }
    )


class TestRunPredictOutputSchema:
    def test_pred_df_columns(self, run_predict_deps):
        task_set_dir, M, train_cfg = run_predict_deps
        cfg = _make_predict_cfg(id_codes=CODES[:1])
        trainer = MagicMock()
        trainer.predict.return_value = [_make_predict_batch(3)]
        with patch("every_query.eval.instantiate", return_value=MagicMock()):
            pred_df, _ = _run_predict(cfg, train_cfg, M, trainer, task_set_dir, "m", [30])
        assert set(pred_df.columns) == {
            "subject_id",
            "prediction_time",
            "occurs_probs",
            "code",
            "bucket",
            "duration_days",
            "model",
        }

    def test_embed_df_columns(self, run_predict_deps):
        task_set_dir, M, train_cfg = run_predict_deps
        cfg = _make_predict_cfg(id_codes=CODES[:1])
        trainer = MagicMock()
        trainer.predict.return_value = [_make_predict_batch(3)]
        with patch("every_query.eval.instantiate", return_value=MagicMock()):
            _, embed_df = _run_predict(cfg, train_cfg, M, trainer, task_set_dir, "m", [30])
        assert set(embed_df.columns) == {
            "subject_id",
            "prediction_time",
            "code",
            "embedding",
            "bucket",
            "duration_days",
            "model",
        }


class TestRunPredictRowGeneration:
    def test_missing_dir_skipped(self, tmp_path):
        task_set_dir = tmp_path / "task_set"
        task_set_dir.mkdir()
        # Only create dir for first code, first duration
        (task_set_dir / "30" / code_slug(CODES[0])).mkdir(parents=True)

        cfg = _make_predict_cfg(id_codes=CODES[:2])
        trainer = MagicMock()
        trainer.predict.return_value = [_make_predict_batch(5)]
        M = _make_mock_model()
        train_cfg = _make_train_cfg()
        with patch("every_query.eval.instantiate", return_value=MagicMock()):
            pred_df, embed_df = _run_predict(cfg, train_cfg, M, trainer, task_set_dir, "m", [30])
        # Only 1 code/duration combo has a dir
        assert len(pred_df) == 5
        assert len(embed_df) == 5

    def test_all_missing_returns_empty(self, tmp_path):
        task_set_dir = tmp_path / "task_set"
        task_set_dir.mkdir()
        cfg = _make_predict_cfg(id_codes=CODES)
        trainer = MagicMock()
        M = _make_mock_model()
        train_cfg = _make_train_cfg()
        with patch("every_query.eval.instantiate", return_value=MagicMock()):
            pred_df, embed_df = _run_predict(cfg, train_cfg, M, trainer, task_set_dir, "m", DURATIONS)
        assert pred_df.is_empty()
        assert embed_df.is_empty()

    def test_batch_concatenation_count(self, run_predict_deps):
        task_set_dir, M, train_cfg = run_predict_deps
        cfg = _make_predict_cfg(id_codes=CODES[:1])
        trainer = MagicMock()
        # 3 batches of 10 samples each
        trainer.predict.return_value = [_make_predict_batch(10, start_id=i * 10) for i in range(3)]
        with patch("every_query.eval.instantiate", return_value=MagicMock()):
            pred_df, _ = _run_predict(cfg, train_cfg, M, trainer, task_set_dir, "m", [30])
        assert len(pred_df) == 30

    def test_multiple_codes_durations(self, run_predict_deps):
        task_set_dir, M, train_cfg = run_predict_deps
        codes = CODES[:2]
        durations = [30, 90]
        cfg = _make_predict_cfg(id_codes=codes)
        trainer = MagicMock()
        trainer.predict.return_value = [_make_predict_batch(5)]
        with patch("every_query.eval.instantiate", return_value=MagicMock()):
            pred_df, embed_df = _run_predict(cfg, train_cfg, M, trainer, task_set_dir, "m", durations)
        # 2 codes * 2 durations * 5 samples = 20
        assert len(pred_df) == 20
        assert len(embed_df) == 20
        # Check labels are correct
        assert set(pred_df["code"].unique().to_list()) == set(codes)
        assert set(pred_df["duration_days"].unique().to_list()) == set(durations)

    def test_subject_ids_preserved(self, run_predict_deps):
        task_set_dir, M, train_cfg = run_predict_deps
        cfg = _make_predict_cfg(id_codes=CODES[:1])
        trainer = MagicMock()
        batch = _make_predict_batch(5, start_id=100)
        trainer.predict.return_value = [batch]
        with patch("every_query.eval.instantiate", return_value=MagicMock()):
            pred_df, _ = _run_predict(cfg, train_cfg, M, trainer, task_set_dir, "m", [30])
        expected_ids = list(range(100, 105))
        assert pred_df["subject_id"].to_list() == expected_ids


# ---------------------------------------------------------------------------
# Multi-checkpoint loop (ckpt_names feature)
# ---------------------------------------------------------------------------


class TestMultiCheckpointLoop:
    """Tests for evaluating multiple checkpoints from the same run directory."""

    CKPT_NAMES = ["epoch=0-step=100", "epoch=0-step=200", "epoch=0-step=300"]

    def _make_multi_ckpt_run_dir(self, tmp_path):
        ckpt_locations = [f"checkpoints/{name}.ckpt" for name in self.CKPT_NAMES]
        return _make_model_run_dir(tmp_path, ckpt_locations)

    def test_multiple_ckpts_produce_distinct_model_names(self, tmp_path):
        run_dir = self._make_multi_ckpt_run_dir(tmp_path)
        task_set_dir = _make_task_dirs(tmp_path, [30], CODES[:1])
        metrics = {"held_out/occurs_auc": 0.8, "held_out/censor_auc": 0.7}

        all_dfs = []
        for ckpt_name in self.CKPT_NAMES:
            model_name = f"{run_dir.name}/{ckpt_name}"
            with (
                patch(_SETUP_PATCHES["load"], return_value=MagicMock()),
                patch(_SETUP_PATCHES["instantiate"], return_value=MagicMock()),
                patch(_SETUP_PATCHES["seed"]),
            ):
                train_cfg, M, trainer = _setup_model(run_dir, ckpt_name=ckpt_name)

            trainer = _make_mock_trainer(test_return=[metrics])
            cfg = _make_run_test_cfg(id_codes=CODES[:1], ood_codes=[])
            with patch("every_query.eval.instantiate", return_value=MagicMock()):
                df = _run_test(cfg, _make_train_cfg(), M, trainer, task_set_dir, model_name, [30])
            all_dfs.append(df)

        combined = pl.concat(all_dfs, how="vertical")
        model_names = combined["model"].unique().to_list()
        assert len(model_names) == 3
        for ckpt_name in self.CKPT_NAMES:
            assert any(ckpt_name in m for m in model_names)

    def test_ckpt_names_fallback_to_ckpt_path(self):
        # Simulates the resolution logic from main()
        cfg_with_names = OmegaConf.create({"ckpt_names": ["a", "b"], "ckpt_path": "best"})
        result = list(cfg_with_names.ckpt_names) if cfg_with_names.get("ckpt_names") else [cfg_with_names.get("ckpt_path")]
        assert result == ["a", "b"]

        cfg_without_names = OmegaConf.create({"ckpt_names": None, "ckpt_path": "best"})
        result = list(cfg_without_names.ckpt_names) if cfg_without_names.get("ckpt_names") else [cfg_without_names.get("ckpt_path")]
        assert result == ["best"]

    def test_all_ckpts_results_in_single_dataframe(self, tmp_path):
        run_dir = self._make_multi_ckpt_run_dir(tmp_path)
        task_set_dir = _make_task_dirs(tmp_path, DURATIONS, CODES[:2])
        metrics = {"held_out/occurs_auc": 0.85, "held_out/censor_auc": 0.90}

        all_dfs = []
        for ckpt_name in self.CKPT_NAMES:
            model_name = f"{run_dir.name}/{ckpt_name}"
            with (
                patch(_SETUP_PATCHES["load"], return_value=MagicMock()),
                patch(_SETUP_PATCHES["instantiate"], return_value=MagicMock()),
                patch(_SETUP_PATCHES["seed"]),
            ):
                train_cfg, M, trainer = _setup_model(run_dir, ckpt_name=ckpt_name)

            trainer = _make_mock_trainer(test_return=[metrics])
            cfg = _make_run_test_cfg(id_codes=CODES[:2], ood_codes=[])
            with patch("every_query.eval.instantiate", return_value=MagicMock()):
                df = _run_test(cfg, _make_train_cfg(), M, trainer, task_set_dir, model_name, DURATIONS)
            all_dfs.append(df)

        combined = pl.concat(all_dfs, how="vertical")
        # 3 ckpts × 2 codes × 2 durations = 12 rows
        assert len(combined) == 3 * 2 * 2
