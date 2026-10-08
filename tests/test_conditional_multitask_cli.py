"""CLI integration for multitask sampling, training, checkpoint restoration and grid prediction."""

import filecmp
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
import yaml
from meds import train_split, tuning_split
from polars.testing import assert_frame_equal
from torch.utils.data import DataLoader

from conftest import run_and_check
from every_query.model.conditional_multitask_lightning import ConditionalMultitaskLightningModule
from every_query.predict.predict_multitask import build_eval_dataset
from every_query.utils.model_loader import setup_model
from tests.designed_specs import entry


def test_conditional_multitask_config_help():
    run_and_check(["EQ_train", "--config-name=_demo_train_conditional_multitask_ar", "--help"], timeout=60.0)


@pytest.fixture(scope="session")
def conditional_multitask_labels_dir(eq_preprocessed_dataset: Path, tmp_path_factory) -> Path:
    """Generate issue-#24 labels for both splits needed by Lightning's fit loop."""
    intermediate = eq_preprocessed_dataset.parent / "intermediate"
    out_dir = tmp_path_factory.mktemp("conditional_multitask_labels")
    for split in (train_split, tuning_split):
        run_and_check(
            [
                "EQ_generate_multitask_sequences",
                f"data_dir={intermediate!s}",
                f"out_dir={out_dir!s}",
                f"query_codes={eq_preprocessed_dataset!s}",
                f"split={split}",
                "num_training_examples=8",
                "num_bounds=5",
                "duration_min=0.01",
                "duration_max=2",
                "eventbound_fraction=0.5",
                "eventstart_fraction=0.25",
                "prediction_time_start_fraction=0.25",
                "start_duration_min=0.01",
                "start_duration_max=2",
                "min_prediction_times_per_subject=1",
                "max_workers=1",
                "label_chunk_rows=2",
                "seed=1",
            ],
            timeout=180.0,
        )
    return out_dir


@pytest.fixture(scope="session")
def conditional_multitask_trained_dir(
    eq_preprocessed_dataset: Path, conditional_multitask_labels_dir: Path, tmp_path_factory
) -> Path:
    output_dir = tmp_path_factory.mktemp("conditional_multitask_train")
    run_and_check(
        [
            "EQ_train",
            "--config-name=_demo_train_conditional_multitask_ar",
            f"output_dir={output_dir!s}",
            f"datamodule.config.tensorized_cohort_dir={eq_preprocessed_dataset!s}",
            f"datamodule.config.task_labels_dir={conditional_multitask_labels_dir!s}",
        ],
        timeout=300.0,
    )
    return output_dir


def test_conditional_multitask_train_checkpoint_and_reload(
    conditional_multitask_trained_dir: Path,
):
    checkpoint = conditional_multitask_trained_dir / "checkpoints" / "last.ckpt"
    resolved = conditional_multitask_trained_dir / "resolved_config.yaml"
    assert checkpoint.exists() and resolved.exists()

    cfg = yaml.safe_load(resolved.read_text())
    model_cfg = cfg["lightning_module"]["model"]
    assert model_cfg["_target_"].endswith("ConditionalMultitaskARModel")
    expected = cfg["datamodule"]["config"]["max_seq_len"] + 3 * model_cfg["max_windows"]
    assert model_cfg["config_overrides"]["max_position_embeddings"] == expected

    from every_query.model.conditional_multitask_ar_model import ConditionalMultitaskARModel
    from every_query.model.conditional_multitask_lightning import ConditionalMultitaskLightningModule
    from every_query.utils.model_loader import setup_model

    loaded_cfg, module, trainer = setup_model(
        conditional_multitask_trained_dir, module_cls=ConditionalMultitaskLightningModule
    )
    assert isinstance(module.model, ConditionalMultitaskARModel)
    assert loaded_cfg.lightning_module.model.max_windows == 5
    assert trainer is not None


def _train_cmd(cohort_dir: Path, labels_dir: Path, output_dir: Path, *overrides: str) -> list[str]:
    return [
        "EQ_train",
        "--config-name=_demo_train_conditional_multitask_ar",
        f"output_dir={output_dir!s}",
        f"datamodule.config.tensorized_cohort_dir={cohort_dir!s}",
        f"datamodule.config.task_labels_dir={labels_dir!s}",
        *overrides,
    ]


def test_logger_false_drops_the_lr_monitor(
    eq_preprocessed_dataset: Path, conditional_multitask_labels_dir: Path, tmp_path: Path
):
    """Every production config ships a ``LearningRateMonitor``; ``trainer.logger=false`` must not crash on it
    at train start (Lightning refuses the monitor without a logger)."""
    output_dir = tmp_path / "nologger"
    run_and_check(
        _train_cmd(
            eq_preprocessed_dataset,
            conditional_multitask_labels_dir,
            output_dir,
            "trainer.logger=false",
            "+trainer.callbacks.learning_rate_monitor="
            "{_target_: lightning.pytorch.callbacks.LearningRateMonitor}",
        ),
        timeout=300.0,
    )
    assert (output_dir / "best_model.ckpt").is_file()


@pytest.fixture(scope="module")
def max_steps_before_first_validation_dir(
    eq_preprocessed_dataset: Path, conditional_multitask_labels_dir: Path, tmp_path_factory
) -> Path:
    """One optimizer step under a fractional val cadence, i.e. training ends before the first validation ever
    records a best checkpoint; logged to CSV so hparams.yaml can be inspected."""
    output_dir = tmp_path_factory.mktemp("multitask_max_steps")
    run_and_check(
        _train_cmd(
            eq_preprocessed_dataset,
            conditional_multitask_labels_dir,
            output_dir,
            "~trainer.logger",
            "+trainer.logger={_target_: lightning.pytorch.loggers.CSVLogger, "
            "save_dir: ${trainer.default_root_dir}/loggers}",
            "trainer.check_val_every_n_epoch=1",
            "trainer.val_check_interval=0.5",
            "+trainer.max_steps=1",
        ),
        timeout=300.0,
    )
    return output_dir


def test_max_steps_before_first_validation_still_publishes_best_model(
    max_steps_before_first_validation_dir: Path,
):
    out = max_steps_before_first_validation_dir
    assert (out / "best_model.ckpt").is_file()
    # No validation ran, so there is no "best"; last.ckpt is what gets published.
    assert filecmp.cmp(out / "best_model.ckpt", out / "checkpoints" / "last.ckpt", shallow=False)


def test_csv_logger_logs_best_ckpt_path_as_a_plain_string(max_steps_before_first_validation_dir: Path):
    """``best_ckpt_path`` is logged as ``str``: a ``!!python/object`` PosixPath tag is not
    ``yaml.safe_load``-able.

    (The datamodule's own ``MEDSTorchDataConfig`` paths still land as
    tags in the same file; only the entry train.py writes is checked here.)
    """
    hparams = sorted(max_steps_before_first_validation_dir.rglob("hparams.yaml"))
    assert hparams, "CSVLogger wrote no hparams.yaml"
    lines = [ln for ln in hparams[-1].read_text().splitlines() if ln.startswith("best_ckpt_path:")]
    assert len(lines) == 1, lines
    loaded = yaml.safe_load(lines[0])
    assert isinstance(loaded["best_ckpt_path"], str)
    assert Path(loaded["best_ckpt_path"]).is_file()


# ---------------------------------------------------------------------------
# Issue #28: EQ_predict_multitask over a QuerySeqSchema grid with active starts
# ---------------------------------------------------------------------------

# Designed sequences with duration and event starts (issue #27), labeled at a supplied cohort.
_GRID_SPECS = {
    "post_admission": [entry("DISCHARGE", 30, start_event="ADMISSION//PULMONARY")],
    "delayed_then_bounded": [
        entry("HR//value_[119.8,inf)", 30, start_duration_days=1),
        entry("DISCHARGE", start_event="ADMISSION//PULMONARY", bound_event="TIMELINE//END"),
    ],
    "single": [entry("TIMELINE//END", 1)],
    # One query spec under both designed conditioning answers: same windows, and each keeps only the
    # contexts whose true first answer is the forced one, so together they partition the cohort.
    "forced_yes": [entry("TIMELINE//END", 30, forced_answer=True), entry("DISCHARGE", 30)],
    "forced_no": [entry("TIMELINE//END", 30, forced_answer=False), entry("DISCHARGE", 30)],
}


@pytest.fixture(scope="module")
def queryseq_grid(eq_preprocessed_dataset: Path, tmp_path_factory) -> tuple[Path, pl.DataFrame]:
    """``EQ_generate_evaluation_query_sequences`` over ``_GRID_SPECS`` on the tuning split: the grid's
    ``out_dir`` (score ``out_dir / "eval"``) and the cohort it was labeled at."""
    root = tmp_path_factory.mktemp("queryseq_grid")
    intermediate = eq_preprocessed_dataset.parent / "intermediate"
    shard = pl.read_parquet(next((intermediate / "data" / tuning_split).rglob("*.parquet")))
    # Prediction time = each subject's first timed event, so windows opening later have data to see.
    cohort = shard.group_by("subject_id").agg(pl.col("time").drop_nulls().min().alias("prediction_time"))
    cohort_fp = root / "cohort.parquet"
    cohort.write_parquet(cohort_fp)
    specs_fp = root / "specs.yaml"
    specs_fp.write_text(yaml.safe_dump(_GRID_SPECS))
    grid_dir = root / "grid"
    run_and_check(
        [
            "EQ_generate_evaluation_query_sequences",
            f"data_dir={intermediate!s}",
            f"out_dir={grid_dir!s}",
            f"query_codes={eq_preprocessed_dataset!s}",
            f"split={tuning_split}",
            f"contexts_path={cohort_fp!s}",
            f"sequences_path={specs_fp!s}",
            "min_task_positives=null",  # the production default (40) would drop every tiny-fixture task
        ],
        timeout=180.0,
    )
    return grid_dir, cohort


def _read_grid(grid_dir: Path) -> pl.DataFrame:
    return pl.concat(
        [pl.read_parquet(fp) for fp in sorted((grid_dir / "eval" / tuning_split).glob("*.parquet"))]
    )


def _predict_multitask(run_dir: Path, grid_dir: Path, output_parquet: Path, *overrides: str) -> pl.DataFrame:
    run_and_check(
        [
            "EQ_predict_multitask",
            f"model_run_dir={run_dir!s}",
            f"tasks_dir={grid_dir / 'eval'!s}",
            f"output_parquet={output_parquet!s}",
            f"split={tuning_split}",
            *overrides,
        ],
        timeout=300.0,
    )
    return pl.read_parquet(output_parquet)


def test_predict_multitask_scores_a_queryseq_grid_with_active_starts(
    conditional_multitask_trained_dir: Path, queryseq_grid: tuple[Path, pl.DataFrame], tmp_path: Path
):
    """End to end: EQ_generate_evaluation_query_sequences (designed sequences with duration and
    event starts, on a supplied cohort) -> EQ_predict_multitask, one scalar prediction per row,
    with no legacy sidecar anywhere."""
    grid_dir, cohort = queryseq_grid
    specs = _GRID_SPECS
    grid = pl.concat(
        [pl.read_parquet(fp) for fp in sorted((grid_dir / "eval" / tuning_split).glob("*.parquet"))]
    )
    # ``forced_yes`` + ``forced_no`` partition the cohort, so they add one cohort's worth of rows.
    assert grid.height == cohort.height * (len(specs) - 1)
    assert {"start_durations", "start_events"} <= set(grid.columns)
    for name in ("_multitask_manifest.json", "eval_meta", "eval_tasks.parquet"):
        assert not list(grid_dir.parent.rglob(name)), name
    assert not list(grid_dir.parent.rglob("*.labels.npy"))

    predictions_fp = tmp_path / "predictions.parquet"
    run_and_check(
        [
            "EQ_predict_multitask",
            f"model_run_dir={conditional_multitask_trained_dir!s}",
            f"tasks_dir={grid_dir / 'eval'!s}",
            f"output_parquet={predictions_fp!s}",
            f"split={tuning_split}",
        ],
        timeout=300.0,
    )
    preds = pl.read_parquet(predictions_fp)
    assert preds.height == grid.height
    assert preds.columns == [
        "subject_id",
        "prediction_time",
        "queries",
        "start_durations",
        "start_events",
        "durations",
        "bound_events",
        "answers",
        "forced_answers",
        "target_code",
        "label",
        "prob",
    ]
    # A designed conditioning answer selects its cohort: the two variants split the contexts by the
    # first query's true answer, so what the model is told is always what happened.
    by_forced = {
        forced: preds.filter(pl.col("forced_answers").list.first() == forced) for forced in (True, False)
    }
    assert by_forced[True].height + by_forced[False].height == cohort.height
    for forced, rows in by_forced.items():
        assert rows["answers"].list.first().to_list() == [forced] * rows.height
    assert preds["prob"].is_between(0.0, 1.0).all()
    assert preds["target_code"].to_list() == [q[-1] for q in preds["queries"].to_list()]
    assert preds["label"].to_list() == [a[-1] for a in preds["answers"].to_list()]
    # The grid rows come back verbatim, active starts included.
    # ``forced_first`` tells the two forced variants apart; they share every other key column.
    forced_first = pl.col("forced_answers").list.first().alias("forced_first")
    key = ["subject_id", "prediction_time", "queries", "forced_first"]
    joined = preds.with_columns(forced_first).join(
        grid.with_columns(forced_first), on=key, how="inner", suffix="_grid", nulls_equal=True
    )
    assert joined.height == grid.height
    for col in ("answers", "forced_answers", "durations", "bound_events", "start_durations", "start_events"):
        assert joined[col].to_list() == joined[f"{col}_grid"].to_list(), col
    starts = preds.explode("start_durations", "start_events")
    assert (starts["start_events"] == "ADMISSION//PULMONARY").sum() == 2 * cohort.height
    assert (starts["start_durations"] == 1.0).sum() == cohort.height
    # Both label values are represented, so the contract is exercised on a real positive and negative.
    assert preds["label"].any() and not preds["label"].all()
    # Nothing legacy was written by inference either.
    for name in ("_multitask_manifest.json", "eval_meta", "eval_tasks.parquet"):
        assert not list(tmp_path.rglob(name)), name
    assert not list(tmp_path.rglob("*.labels.npy"))


# ---------------------------------------------------------------------------
# Issue #30: prediction through Trainer.predict on ConditionalMultitaskDataModule
# ---------------------------------------------------------------------------


def test_trained_run_dir_carries_the_split_datamodule_shape(conditional_multitask_trained_dir: Path):
    """The run dir the end-to-end test above predicts from has the post-#30 ``datamodule`` node:

    ``ConditionalMultitaskDataModule`` with the grid root unset at training time.  That test is
    therefore the CLI-level check that such a run dir predicts fine.
    """
    cfg = yaml.safe_load((conditional_multitask_trained_dir / "resolved_config.yaml").read_text())
    node = cfg["datamodule"]
    assert node["_target_"].endswith(".conditional_multitask_datamodule.ConditionalMultitaskDataModule")
    assert node["eval_tasks_dir"] is None
    assert node["max_windows"] == cfg["lightning_module"]["model"]["max_windows"]
    assert "data_class" not in node
    assert (
        node["dataset_kwargs"]["expected_vocab_size"]
        == (cfg["lightning_module"]["model"]["config_overrides"]["vocab_size"])
    )


def _pre_issue_30_run_dir(run_dir: Path, dst: Path) -> Path:
    """Copy ``run_dir`` and rewrite its resolved ``datamodule`` node to the pre-#30 shape.

    Older runs recorded ``ResumableDatamodule`` with an explicit ``data_class`` and no grid keys; the
    checkpoint, its cohort settings and its loader settings are otherwise identical.
    """
    shutil.copytree(run_dir, dst)
    resolved = dst / "resolved_config.yaml"
    cfg = yaml.safe_load(resolved.read_text())
    node = cfg["datamodule"]
    cfg["datamodule"] = {
        "_target_": "every_query.data.datamodule.ResumableDatamodule",
        "data_class": "every_query.data.multitask_dataset.MultitaskBoundaryPytorchDataset",
        "config": node["config"],
        "dataset_kwargs": node["dataset_kwargs"],
        "batch_size": node["batch_size"],
        "num_workers": node["num_workers"],
        "pin_memory": node["pin_memory"],
    }
    resolved.write_text(yaml.safe_dump(cfg))
    return dst


def test_predict_multitask_accepts_a_pre_issue_30_run_dir(
    conditional_multitask_trained_dir: Path, queryseq_grid: tuple[Path, pl.DataFrame], tmp_path: Path
):
    """A checkpoint whose ``resolved_config.yaml`` predates the split datamodule predicts identically to the
    same checkpoint under the current shape: the predictor reads the node's cohort / loader settings,
    never its ``_target_``.  Both runs pin ``device=cpu`` so the comparison is like for like."""
    grid_dir, _ = queryseq_grid
    old_dir = _pre_issue_30_run_dir(conditional_multitask_trained_dir, tmp_path / "pre_issue_30_run")
    old_node = yaml.safe_load((old_dir / "resolved_config.yaml").read_text())["datamodule"]
    assert old_node["_target_"] == "every_query.data.datamodule.ResumableDatamodule"
    assert "eval_tasks_dir" not in old_node and "max_windows" not in old_node

    quiet = ("device=cpu", "enable_progress_bar=false")
    new = _predict_multitask(conditional_multitask_trained_dir, grid_dir, tmp_path / "new.parquet", *quiet)
    old = _predict_multitask(old_dir, grid_dir, tmp_path / "old.parquet", *quiet)

    grid = _read_grid(grid_dir)
    assert new.height == old.height == grid.height
    assert new["prob"].is_between(0.0, 1.0).all()
    assert_frame_equal(new, old)


def test_predict_multitask_probabilities_match_a_direct_score_final_query(
    conditional_multitask_trained_dir: Path, queryseq_grid: tuple[Path, pl.DataFrame], tmp_path: Path
):
    """The CLI's ``prob`` column is ``sigmoid(score_final_query)`` of the checkpoint over the grid, in grid
    order: ``precision=32-true`` on the CPU reproduces a direct fp32 call (to the tolerance of batch
    padding), and the default ``bf16-mixed`` agrees with it to bf16 rounding.  Everything but ``prob`` is
    identical between the two runs."""
    grid_dir, _ = queryseq_grid
    run_dir = conditional_multitask_trained_dir
    quiet = ("enable_progress_bar=false",)
    default = _predict_multitask(run_dir, grid_dir, tmp_path / "bf16.parquet", *quiet)
    fp32 = _predict_multitask(
        run_dir, grid_dir, tmp_path / "fp32.parquet", "precision=32-true", "device=cpu", *quiet
    )
    assert_frame_equal(default.drop("prob"), fp32.drop("prob"))

    train_cfg, module, _ = setup_model(run_dir, module_cls=ConditionalMultitaskLightningModule)
    model = module.model.cpu().eval()
    ds = build_eval_dataset(
        train_cfg,
        grid_dir / "eval",
        tuning_split,
        expected_vocab_size=model.vocab_size,
        use_rope_time=model.use_rope_time,
        max_windows=model.max_windows,
    )
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=ds.collate)
    with torch.no_grad():
        reference = torch.cat([torch.sigmoid(model.score_final_query(b, b.scored_codes)) for b in loader])
    reference = reference.float().numpy()
    assert reference.shape == (fp32.height,) and len(np.unique(np.round(reference, 5))) > 1
    np.testing.assert_allclose(fp32["prob"].to_numpy(), reference, atol=1e-4, rtol=0.0)
    np.testing.assert_allclose(default["prob"].to_numpy(), reference, atol=2e-2, rtol=0.0)


# ---------------------------------------------------------------------------
# Ontology: train with derived ancestor targets, score an ancestor query
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def multitask_ontology_dir(eq_preprocessed_dataset: Path, tmp_path_factory) -> Path:
    """``EQ_build_ontology`` over the demo cohort (module invocation, as in ``test_features_e2e_cli``)."""
    out = tmp_path_factory.mktemp("multitask_ontology")
    run_and_check(
        [
            sys.executable,
            "-m",
            "every_query.data.build_ontology",
            f"tensorized_cohort_dir={eq_preprocessed_dataset!s}",
            f"out_dir={out!s}",
            "decay=0.5",
        ],
        timeout=120.0,
    )
    return out


def _an_ancestor(ontology_dir: Path, prefer: str = "ADMISSION") -> str:
    nodes = pl.read_parquet(ontology_dir / "ontology_vocab.parquet").filter(~pl.col("is_observed_code"))
    names = sorted(nodes["node_name"].to_list())
    assert names, "the demo cohort's hierarchical names must mint ancestor nodes"
    return prefer if prefer in names else names[0]


@pytest.fixture(scope="module")
def conditional_multitask_ontology_trained_dir(
    eq_preprocessed_dataset: Path,
    conditional_multitask_labels_dir: Path,
    multitask_ontology_dir: Path,
    tmp_path_factory,
) -> Path:
    """The demo multitask config trained on the *same leaf-only labels* with ``ontology_dir`` set."""
    output_dir = tmp_path_factory.mktemp("conditional_multitask_train_ontology")
    run_and_check(
        _train_cmd(
            eq_preprocessed_dataset,
            conditional_multitask_labels_dir,
            output_dir,
            f"lightning_module.model.ontology_dir={multitask_ontology_dir!s}",
        ),
        timeout=300.0,
    )
    return output_dir


def test_ontology_training_records_both_widths_and_needs_no_new_labels(
    conditional_multitask_ontology_trained_dir: Path,
    conditional_multitask_labels_dir: Path,
    multitask_ontology_dir: Path,
    eq_preprocessed_dataset: Path,
):
    """Training with an ontology reads the leaf-only sidecars unchanged: ``resolved_config.yaml`` records the
    ontology's ``V_ext`` on the model, the cohort's ``V`` on the datamodule and the cohort's vocabulary
    fingerprint (the manifest's) on the model, and the reloaded model reports all three."""
    from every_query.data.ontology import cohort_code_map, extended_vocab_size, ontology_vocab_fingerprint
    from every_query.generate_tasks.sample_multitask_sequences import read_manifest
    from every_query.utils.digest import vocab_fingerprint

    cfg = yaml.safe_load((conditional_multitask_ontology_trained_dir / "resolved_config.yaml").read_text())
    v_ext = extended_vocab_size(multitask_ontology_dir)
    manifest = read_manifest(conditional_multitask_labels_dir / train_split)
    v = int(manifest["vocab_size"])
    codes_fp = eq_preprocessed_dataset / "metadata" / "codes.parquet"
    codes = pl.read_parquet(codes_fp)
    assert v == int(codes["code/vocab_index"].max()) + 1 < v_ext
    model_cfg = cfg["lightning_module"]["model"]
    assert model_cfg["ontology_dir"] == str(multitask_ontology_dir)
    assert model_cfg["config_overrides"]["vocab_size"] == v_ext
    # One digest, three artifacts: the cohort's codes.parquet, the leaf manifest and the ontology's leaves.
    fingerprint = vocab_fingerprint(cohort_code_map(codes_fp))
    assert model_cfg["cohort_vocab_fingerprint"] == fingerprint
    assert manifest["vocab_fingerprint"] == fingerprint == ontology_vocab_fingerprint(multitask_ontology_dir)
    dm_kwargs = cfg["datamodule"]["dataset_kwargs"]
    assert dm_kwargs["expected_vocab_size"] == v, (
        "the training datasets are checked against the leaf manifest"
    )
    assert dm_kwargs["ontology_dir"] == str(multitask_ontology_dir)

    _, module, _ = setup_model(
        conditional_multitask_ontology_trained_dir, module_cls=ConditionalMultitaskLightningModule
    )
    assert module.model.vocab_size == v_ext and module.model.base_vocab_size == v
    assert module.model.code_bias.shape == (v_ext,)
    assert module.model.cohort_vocab_fingerprint == fingerprint


def test_train_refuses_a_same_width_permuted_ontology(
    eq_preprocessed_dataset: Path, conditional_multitask_labels_dir: Path, tmp_path: Path
):
    """Regression for the PR #32 review: an ontology built from this cohort's codes with two indices swapped
    has the cohort's ``V`` and the genuine ontology's ``V_ext``, so every width check passes.

    ``EQ_train``
    must still refuse it before a single step, naming the renumbered codes.
    """
    from every_query.data.ontology import cohort_code_map
    from tests.multitask.conftest import write_cohort_ontology

    first, second = sorted(cohort_code_map(eq_preprocessed_dataset / "metadata" / "codes.parquet"))[:2]
    permuted = write_cohort_ontology(eq_preprocessed_dataset, tmp_path / "permuted", swap=(first, second))
    output_dir = tmp_path / "train_permuted"
    with pytest.raises(RuntimeError, match=r"different codes\.parquet than this cohort") as info:
        run_and_check(
            _train_cmd(
                eq_preprocessed_dataset,
                conditional_multitask_labels_dir,
                output_dir,
                f"lightning_module.model.ontology_dir={permuted!s}",
            ),
            timeout=300.0,
        )
    assert "2 code(s) sit at a different index" in str(info.value)
    assert not (output_dir / "resolved_config.yaml").exists(), "refused before the run directory was written"


@pytest.fixture(scope="module")
def ancestor_queryseq_grid(
    eq_preprocessed_dataset: Path, multitask_ontology_dir: Path, tmp_path_factory
) -> tuple[Path, str]:
    """An evaluation grid labeled *with* the ontology whose sequences ask an ancestor node."""
    ancestor = _an_ancestor(multitask_ontology_dir)
    root = tmp_path_factory.mktemp("ancestor_queryseq_grid")
    intermediate = eq_preprocessed_dataset.parent / "intermediate"
    shard = pl.read_parquet(next((intermediate / "data" / tuning_split).rglob("*.parquet")))
    cohort = shard.group_by("subject_id").agg(pl.col("time").drop_nulls().min().alias("prediction_time"))
    cohort_fp = root / "cohort.parquet"
    cohort.write_parquet(cohort_fp)
    specs_fp = root / "specs.yaml"
    specs_fp.write_text(
        yaml.safe_dump(
            {
                "family": [entry(ancestor, 30)],
                "leaf_then_family": [entry("TIMELINE//END", 1), entry(ancestor, 30)],
                "family_then_leaf": [entry(ancestor, 30), entry("DISCHARGE", 30)],
            }
        )
    )
    grid_dir = root / "grid"
    run_and_check(
        [
            "EQ_generate_evaluation_query_sequences",
            f"data_dir={intermediate!s}",
            f"out_dir={grid_dir!s}",
            f"query_codes={eq_preprocessed_dataset!s}",
            f"split={tuning_split}",
            f"contexts_path={cohort_fp!s}",
            f"sequences_path={specs_fp!s}",
            "min_task_positives=null",  # the production default (40) would drop every tiny-fixture task
            f"ontology_dir={multitask_ontology_dir!s}",
        ],
        timeout=180.0,
    )
    return grid_dir, ancestor


def test_predict_multitask_scores_an_ancestor_query(
    conditional_multitask_ontology_trained_dir: Path,
    conditional_multitask_trained_dir: Path,
    ancestor_queryseq_grid: tuple[Path, str],
    multitask_ontology_dir: Path,
    tmp_path: Path,
):
    """End to end: a grid asking an ancestor node (labeled with the ontology) is scored by the ontology
    checkpoint, one probability per row, matching a direct ``score_final_query``; the leaf checkpoint
    refuses the same grid because the ancestor name is not in its vocabulary."""
    grid_dir, ancestor = ancestor_queryseq_grid
    grid = _read_grid(grid_dir)
    assert grid.height == 3 * grid.select("subject_id").n_unique()
    assert any(ancestor in q for q in grid["queries"].to_list())
    # The grid's provenance names the ontology's closure, which is what the predictor checks.
    sidecars = sorted(
        (grid_dir.parent / "grid_artifacts" / "_labeled" / "eval" / tuning_split).glob("*.json")
    )
    assert sidecars

    quiet = ("device=cpu", "enable_progress_bar=false")
    preds = _predict_multitask(
        conditional_multitask_ontology_trained_dir, grid_dir, tmp_path / "ancestor.parquet", *quiet
    )
    assert preds.height == grid.height
    assert preds["prob"].is_between(0.0, 1.0).all()
    assert preds["target_code"].to_list() == [q[-1] for q in preds["queries"].to_list()]
    assert (preds["target_code"] == ancestor).sum() == 2 * grid.select("subject_id").n_unique()
    key = ["subject_id", "prediction_time", "queries"]
    joined = preds.join(grid, on=key, how="inner", suffix="_grid")
    assert joined.height == grid.height and joined["answers"].to_list() == joined["answers_grid"].to_list()

    train_cfg, module, _ = setup_model(
        conditional_multitask_ontology_trained_dir, module_cls=ConditionalMultitaskLightningModule
    )
    model = module.model.cpu().eval()
    ds = build_eval_dataset(
        train_cfg,
        grid_dir / "eval",
        tuning_split,
        expected_vocab_size=model.vocab_size,
        base_vocab_size=model.base_vocab_size,
        ontology_dir=model.ontology_dir,
        use_rope_time=model.use_rope_time,
        max_windows=model.max_windows,
    )
    assert ds.code_to_index[ancestor] >= model.base_vocab_size
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=ds.collate)
    with torch.no_grad():
        reference = torch.cat([torch.sigmoid(model.score_final_query(b, b.scored_codes)) for b in loader])
    np.testing.assert_allclose(preds["prob"].to_numpy(), reference.float().numpy(), atol=2e-2, rtol=0.0)

    # The leaf checkpoint has no row for the ancestor: refused at dataset construction, not scored.
    cmd = [
        "EQ_predict_multitask",
        f"model_run_dir={conditional_multitask_trained_dir!s}",
        f"tasks_dir={grid_dir / 'eval'!s}",
        f"output_parquet={tmp_path / 'leaf.parquet'!s}",
        f"split={tuning_split}",
        *quiet,
    ]
    from conftest import _VENV_BIN

    env = dict(os.environ, PATH=_VENV_BIN + os.pathsep + os.environ.get("PATH", ""))
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300.0)
    assert result.returncode != 0
    assert "not in this run's vocabulary" in result.stderr + result.stdout
    assert not (tmp_path / "leaf.parquet").exists()
