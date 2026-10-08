"""Logic of ``EQ_predict_multitask`` on a ``QuerySeqSchema`` grid (issue #28 / #30), without a checkpoint.

What is tested is the bookkeeping that decides *which* number gets compared to *which* label — get
that wrong and the metrics look plausible and mean nothing:

1.  missing start / bound columns normalize to zeros / ``NO_BOUND_INDEX``;
2.  active duration and event starts tensorize through the explicit multitask opt-in;
3.  query / start / bound codes map to the right vocabulary indices;
4.  unknown codes fail at dataset init, before any model runs;
5.  conditions are all and only the real queries before the final one;
6.  the final code / label is selected per row under variable-length padding;
7.  one-query sequences produce zero conditioning pairs;
8.  target-only logits equal gathered full-vocabulary logits (also in
    ``tests/test_conditional_multitask_ar_model.py``);
9.  the all-vocabulary training forward is unchanged (``tests/test_conditional_multitask_ar_model.py``);
10. prediction through the Lightning predict loop is row-aligned, one row per input row, and equals
    ``sigmoid(score_final_query)`` over the sequential loader; the collated labels *and* scored code
    indices are re-checked against the grid, so a same-label swap of rows scoring different codes
    cannot slip through;
11. a grid subject absent from the tensorized cohort is rejected, never silently dropped;
12. no manifest, packed labels or eval-meta sidecar is needed;
13. the prediction datamodule is built from either shape of ``resolved_config.yaml`` (pre- and
    post-#30) with the checkpoint-vs-cohort checks, and the inference trainer is single-device and
    single-process (a ``torchrun`` / ``srun --ntasks>1`` launch is refused even though Lightning's
    single-device strategy reports ``world_size == 1``).

The grid rows are written against the session fixture cohort (``tensorized_cohort_dir``), whose
subjects and codes are the real ones the dataset joins against.  Dataset-level tests use the train
subjects; predictor-level tests go through the prediction datamodule, which scores only the
evaluation splits (``train`` shuffles), so their grids sit on the tuning subject.
"""

import os
from datetime import datetime
from functools import partial
from pathlib import Path

import lightning.pytorch as L
import numpy as np
import polars as pl
import pytest
import torch
from lightning.pytorch.strategies import SingleDeviceStrategy
from meds import train_split, tuning_split
from meds_torchdata.config import MEDSTorchDataConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, SequentialSampler, Subset

from conftest import _PRED_TIMES, _TRAIN_SUBJECTS, _TUNING_SUBJECTS
from every_query.data.conditional_multitask_datamodule import ConditionalMultitaskDataModule
from every_query.data.multitask_dataset import MultitaskBoundaryBatch
from every_query.data.multitask_eval_dataset import MultitaskEvalBatch, QuerySeqMultitaskEvalDataset
from every_query.data.query_seq_dataset import (
    EVENT_BOUND_DURATION_SENTINEL as SENTINEL,
)
from every_query.data.query_seq_dataset import (
    NO_BOUND_INDEX,
    QuerySeqPytorchDataset,
)
from every_query.model.conditional_multitask_ar_model import ConditionalMultitaskARModel
from every_query.model.conditional_multitask_lightning import ConditionalMultitaskLightningModule
from every_query.predict.predict_multitask import (
    build_eval_dataset,
    build_predict_datamodule,
    build_predict_trainer,
    check_grid_ontology_provenance,
    check_single_device,
    check_single_process,
    launcher_world_size,
    predictions_to_df,
    resolve_accelerator,
    run_inference,
)
from tests.multitask.conftest import write_cohort_ontology

# Real codes of the fixture cohort.
Q1, Q2, Q3 = "HR", "TEMP", "DISCHARGE"
ADMIT = "ADMISSION//CARDIAC"
LEGACY_SIDECARS = ("_multitask_manifest.json", "eval_meta", "eval_tasks.parquet")
TUNING_SUBJECT = _TUNING_SUBJECTS[0]


def _write_grid(root: Path, rows: list[dict], *, split: str = train_split, shard: str = "0") -> Path:
    """Write ``rows`` as one ``QuerySeqSchema`` shard under ``{root}/{split}/``; return ``root``."""
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    cols = {
        "subject_id": pl.Int64,
        "prediction_time": pl.Datetime("us"),
        "queries": pl.List(pl.Utf8),
        "durations": pl.List(pl.Float32),
        "answers": pl.List(pl.Boolean),
    }
    if "bound_events" in rows[0]:
        cols["bound_events"] = pl.List(pl.Utf8)
    if "start_durations" in rows[0]:
        cols["start_durations"] = pl.List(pl.Float32)
        cols["start_events"] = pl.List(pl.Utf8)
    if "forced_answers" in rows[0]:
        cols["forced_answers"] = pl.List(pl.Boolean)
    pl.DataFrame(rows, schema=cols).write_parquet(split_dir / f"{shard}.parquet")
    return root


def _row(subject: int, queries: list[str], answers: list[bool], **extra) -> dict:
    return {
        "subject_id": subject,
        "prediction_time": _PRED_TIMES[subject],
        "queries": queries,
        "durations": [float(7 * (i + 1)) for i in range(len(queries))],
        "answers": answers,
        **extra,
    }


def _mixed_rows() -> list[dict]:
    """1-, 2-, 3- and 2-query rows: one per fixture subject, no optional columns."""
    a, b, c, d = _TRAIN_SUBJECTS
    return [
        _row(a, [Q1], [True]),
        _row(b, [Q1, Q2], [False, True]),
        _row(c, [Q2, Q1, Q3], [True, True, False]),
        _row(d, [Q3, Q2], [False, False]),
    ]


def _eval_rows() -> list[dict]:
    """The ``_mixed_rows`` shapes on the tuning subject, for grids scored through the datamodule."""
    s = TUNING_SUBJECT
    return [
        _row(s, [Q1], [True]),
        _row(s, [Q1, Q2], [False, True]),
        _row(s, [Q2, Q1, Q3], [True, True, False]),
        _row(s, [Q3, Q2], [False, False]),
    ]


def _data_config(cohort: Path, grid: Path) -> MEDSTorchDataConfig:
    return MEDSTorchDataConfig(
        tensorized_cohort_dir=str(cohort),
        task_labels_dir=str(grid),
        max_seq_len=64,
        seq_sampling_strategy="to_end",
        static_inclusion_mode="omit",
        batch_mode="SM",
    )


def _dataset(cohort: Path, grid: Path, **kw) -> QuerySeqMultitaskEvalDataset:
    return QuerySeqMultitaskEvalDataset(_data_config(cohort, grid), split=train_split, **kw)


def _collate_all(ds: QuerySeqMultitaskEvalDataset) -> MultitaskEvalBatch:
    return ds.collate([ds[i] for i in range(len(ds))])


def _tiny_model(vocab_size: int) -> ConditionalMultitaskARModel:
    """A seeded, CPU-sized ``ConditionalMultitaskARModel`` whose tied embedding spans the cohort
    vocabulary."""
    torch.manual_seed(0)
    return ConditionalMultitaskARModel(
        config_overrides={
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "intermediate_size": 32,
            "max_position_embeddings": 64 + 15,
            "vocab_size": vocab_size,
            "pad_token_id": 0,
            "attention_dropout": 0.0,
        },
        max_windows=5,
    ).eval()


# --- 1 / 12: defaults and no sidecars -----------------------------------------------------------


def test_missing_start_and_bound_columns_normalize_to_defaults(tensorized_cohort_dir, tmp_path):
    grid = _write_grid(tmp_path / "grid", _mixed_rows())
    for name in LEGACY_SIDECARS:
        assert not list(grid.rglob(name)), "the grid must carry no legacy sidecar"
    assert not list(grid.rglob("*.labels.npy"))
    ds = _dataset(tensorized_cohort_dir, grid)
    assert not ds.has_starts and not ds.has_bound_events and ds.allow_active_starts
    batch = _collate_all(ds)
    assert isinstance(batch, MultitaskEvalBatch)
    assert batch.num_bounds == 3
    assert torch.equal(batch.q_start_durations, torch.zeros(4, 3))
    assert torch.equal(batch.q_start_codes, torch.full((4, 3), NO_BOUND_INDEX))
    assert torch.equal(batch.q_bound_codes, torch.full((4, 3), NO_BOUND_INDEX))
    assert batch.q_start_durations.dtype == torch.float32 and batch.q_start_codes.dtype == torch.long


# --- 2 / 3: active starts, bounds, and the vocabulary indices ---------------------------------


def test_active_starts_and_bounds_tensorize_with_the_cohort_indices(tensorized_cohort_dir, tmp_path):
    a, b, c, d = _TRAIN_SUBJECTS
    rows = [
        _row(
            a,
            [Q1, Q2],
            [True, False],
            start_durations=[7.0, 0.0],
            start_events=[None, None],
            bound_events=[None, Q3],
        ),
        _row(b, [Q3], [True], start_durations=[SENTINEL], start_events=[ADMIT], bound_events=[None]),
        _row(
            c,
            [Q2, Q1, Q3],
            [True, True, False],
            start_durations=[0.0, SENTINEL, 3.0],
            start_events=[None, ADMIT, None],
            bound_events=[Q3, None, ADMIT],
        ),
        _row(d, [Q1], [False], start_durations=[0.0], start_events=[None], bound_events=[None]),
    ]
    grid = _write_grid(tmp_path / "grid", rows)
    # The ordinary sequence path must refuse this grid; the adapter is the opt-in.
    with pytest.raises(ValueError, match="active start"):
        QuerySeqPytorchDataset(_data_config(tensorized_cohort_dir, grid), split=train_split)

    ds = _dataset(tensorized_cohort_dir, grid)
    assert ds.has_starts and ds.has_bound_events
    idx = ds.code_to_index
    batch = _collate_all(ds)
    nb = NO_BOUND_INDEX
    assert batch.q_start_durations.tolist() == [
        [7.0, 0.0, 0.0],
        [SENTINEL, 0.0, 0.0],
        [0.0, SENTINEL, 3.0],
        [0.0, 0.0, 0.0],
    ]
    assert batch.q_start_codes.tolist() == [
        [nb, nb, nb],
        [idx[ADMIT], nb, nb],
        [nb, idx[ADMIT], nb],
        [nb, nb, nb],
    ]
    assert batch.q_bound_codes.tolist() == [
        [nb, idx[Q3], nb],
        [nb, nb, nb],
        [idx[Q3], nb, idx[ADMIT]],
        [nb, nb, nb],
    ]
    assert batch.scored_codes.tolist() == [idx[Q2], idx[Q3], idx[Q3], idx[Q1]]
    assert batch.condition_codes.tolist() == [[idx[Q1], 0], [0, 0], [idx[Q2], idx[Q1]], [0, 0]]


# --- 4: unknown codes fail before inference ---------------------------------------------------


@pytest.mark.parametrize(
    "column", ["queries", "start_events", "bound_events"], ids=["query", "start", "bound"]
)
def test_unknown_codes_fail_at_init(tensorized_cohort_dir, tmp_path, column):
    a = _TRAIN_SUBJECTS[0]
    row = _row(
        a,
        [Q1, Q2],
        [True, False],
        start_durations=[SENTINEL, 0.0],
        start_events=[ADMIT, None],
        bound_events=[None, Q3],
    )
    if column == "queries":
        row["queries"] = ["NOPE//X", Q2]
    elif column == "start_events":
        row["start_events"] = ["NOPE//X", None]
    else:
        row["bound_events"] = [None, "NOPE//X"]
    grid = _write_grid(tmp_path / "grid", [row])
    with pytest.raises(ValueError, match="not in this run's vocabulary"):
        _dataset(tensorized_cohort_dir, grid)


def test_codes_past_the_checkpoint_vocabulary_width_are_rejected(tensorized_cohort_dir, tmp_path):
    grid = _write_grid(tmp_path / "grid", _mixed_rows())
    ds = _dataset(tensorized_cohort_dir, grid)
    widest = int(ds._q_codes.max())
    _dataset(tensorized_cohort_dir, grid, expected_vocab_size=widest + 1)  # fits
    with pytest.raises(ValueError, match="outside the checkpoint's vocabulary"):
        _dataset(tensorized_cohort_dir, grid, expected_vocab_size=widest)


def test_rows_longer_than_the_checkpoints_window_budget_are_rejected_at_init(tensorized_cohort_dir, tmp_path):
    """A 3-query row fits ``max_windows=3`` and is rejected by ``max_windows=2`` before any model runs (the
    model itself would only fail when that row's batch reached the backbone)."""
    grid = _write_grid(tmp_path / "grid", _mixed_rows())
    ds = _dataset(tensorized_cohort_dir, grid, max_windows=3)
    assert ds.max_windows == 3
    with pytest.raises(ValueError, match="max_windows=2"):
        _dataset(tensorized_cohort_dir, grid, max_windows=2)


# --- 13: the prediction datamodule from a checkpoint's resolved config -------------------------


def _train_cfg(cohort: Path, *, strip_delta_tokens: bool = False, shape: str = "new") -> DictConfig:
    """The ``datamodule`` node of a checkpoint's ``resolved_config.yaml``, in either shape found on disk.

    ``"old"`` is a run trained before issue #30: ``ResumableDatamodule`` with an explicit ``data_class``
    and no grid keys.  ``"new"`` names ``ConditionalMultitaskDataModule`` with ``eval_tasks_dir`` unset
    (training never needs the grid).  Both record the TRAINING labels root, which does not exist here:
    prediction must not depend on it.
    """
    node = {
        "config": {
            "_target_": "meds_torchdata.MEDSTorchDataConfig",
            "tensorized_cohort_dir": str(cohort),
            "task_labels_dir": str(cohort / "not-on-this-machine" / "training_labels"),
            "max_seq_len": 64,
            "seq_sampling_strategy": "to_end",
            "static_inclusion_mode": "omit",
            "batch_mode": "SM",
        },
        "dataset_kwargs": {"strip_delta_tokens": strip_delta_tokens, "expected_vocab_size": 999},
        "batch_size": 2,
        "num_workers": 0,
        "pin_memory": False,
    }
    if shape == "old":
        node = {
            "_target_": "every_query.data.datamodule.ResumableDatamodule",
            "data_class": "every_query.data.multitask_dataset.MultitaskBoundaryPytorchDataset",
            **node,
        }
    elif shape == "new":
        node = {
            "_target_": "every_query.data.conditional_multitask_datamodule.ConditionalMultitaskDataModule",
            **node,
            "eval_tasks_dir": None,
            "max_windows": 5,
        }
    else:
        raise ValueError(shape)
    return OmegaConf.create({"datamodule": node})


def _vocab_size(cohort: Path, grid: Path) -> int:
    return int(_data_config(cohort, grid).vocab_size)


@pytest.mark.parametrize("shape", ["old", "new"])
def test_build_predict_datamodule_accepts_both_resolved_config_shapes(tensorized_cohort_dir, tmp_path, shape):
    """Only ``config`` / ``dataset_kwargs`` / loader settings are read from the checkpoint's datamodule node,
    never its ``_target_``, so pre- and post-#30 run directories predict alike."""
    grid = _write_grid(tmp_path / "grid", _eval_rows(), split=tuning_split)
    cfg = _train_cfg(tensorized_cohort_dir, shape=shape)
    v = _vocab_size(tensorized_cohort_dir, grid)
    assert not Path(cfg.datamodule.config.task_labels_dir).exists()

    dm = build_predict_datamodule(
        cfg, grid, tuning_split, expected_vocab_size=v, use_rope_time=False, max_windows=5
    )
    assert isinstance(dm, ConditionalMultitaskDataModule)
    assert dm.eval_tasks_dir == grid and dm.predict_split == tuning_split and dm.max_windows == 5
    assert dm.batch_size == 2 and dm.num_workers == 0 and dm.pin_memory is False
    # The checkpoint's own expected_vocab_size is replaced by the loaded model's tied-embedding width.
    assert dm.dataset_kwargs == {"strip_delta_tokens": False, "expected_vocab_size": v}
    # Only the label root differs from the checkpoint's config, and it is the grid on both sides:
    # the training labels are never read.
    assert dm.config.task_labels_dir == grid == dm.eval_config.task_labels_dir
    assert dm.config.tensorized_cohort_dir == tensorized_cohort_dir and dm.config.max_seq_len == 64

    ds = dm.predict_dataset
    assert isinstance(ds, QuerySeqMultitaskEvalDataset) and len(ds) == 4
    assert ds.split == tuning_split and ds.expected_vocab_size == v and ds.max_windows == 5
    loader = dm.predict_dataloader()
    assert isinstance(loader.sampler, SequentialSampler) and loader.batch_size == 2
    assert torch.cat([b.labels for b in loader]).tolist() == [r["answers"][-1] for r in _eval_rows()]

    # CLI overrides win over the checkpoint's loader settings.
    override = build_predict_datamodule(
        cfg, grid, tuning_split, expected_vocab_size=v, use_rope_time=False, batch_size=3, num_workers=0
    )
    assert override.predict_dataloader().batch_size == 3

    # ``build_eval_dataset`` is that datamodule's predict dataset.
    ds2 = build_eval_dataset(
        cfg, grid, tuning_split, expected_vocab_size=v, use_rope_time=False, max_windows=5
    )
    assert isinstance(ds2, QuerySeqMultitaskEvalDataset) and len(ds2) == 4 and ds2.max_windows == 5


def test_build_predict_datamodule_checks_the_checkpoint_against_the_cohort(tensorized_cohort_dir, tmp_path):
    """The predictor-level guards: the cohort's vocabulary width must equal the model's tied
    embedding width, the datamodule's delta-token strip must agree with ``use_rope_time``, the
    window budget flows through to the dataset, and ``train`` is not a grid split."""
    grid = _write_grid(tmp_path / "grid", _eval_rows(), split=tuning_split)
    cfg = _train_cfg(tensorized_cohort_dir)
    v = _vocab_size(tensorized_cohort_dir, grid)
    kw = {"expected_vocab_size": v, "use_rope_time": False}

    ds = build_eval_dataset(cfg, grid, tuning_split, **kw, max_windows=5)
    assert len(ds) == 4 and ds.expected_vocab_size == v and ds.max_windows == 5
    with pytest.raises(ValueError, match="tied embedding table"):
        build_predict_datamodule(cfg, grid, tuning_split, expected_vocab_size=v + 1, use_rope_time=False)
    with pytest.raises(ValueError, match=r"use_rope_time=True; resolved_config\.yaml is inconsistent"):
        build_predict_datamodule(cfg, grid, tuning_split, expected_vocab_size=v, use_rope_time=True)
    rope_cfg = _train_cfg(tensorized_cohort_dir, strip_delta_tokens=True)
    with pytest.raises(ValueError, match="strips delta tokens=True"):
        build_predict_datamodule(rope_cfg, grid, tuning_split, **kw)
    assert build_predict_datamodule(
        rope_cfg, grid, tuning_split, expected_vocab_size=v, use_rope_time=True
    ).predict_dataset.strip_delta_tokens
    with pytest.raises(ValueError, match="max_windows=2"):
        build_eval_dataset(cfg, grid, tuning_split, **kw, max_windows=2)
    with pytest.raises(ValueError, match="predict_split must be one of"):
        build_predict_datamodule(cfg, grid, train_split, **kw)


# --- 14: an ontology separates the cohort width from the table width ---------------------------------


@pytest.fixture(scope="module")
def cohort_ontology_dir(tensorized_cohort_dir: Path, tmp_path_factory) -> Path:
    """An ontology built from the fixture cohort's own ``codes.parquet``."""
    return write_cohort_ontology(tensorized_cohort_dir, tmp_path_factory.mktemp("cohort_ontology"))


def _an_ancestor(ontology_dir: Path, prefer: str = "ADMISSION") -> tuple[str, int]:
    """``(name, index)`` of one pure ancestor node of the ontology (``prefer`` when it is one)."""
    from every_query.data.ontology import load_nodes

    nodes = load_nodes(ontology_dir).filter(~pl.col("is_observed_code")).sort("token_id")
    names = nodes["node_name"].to_list()
    name = prefer if prefer in names else names[0]
    return name, int(nodes.filter(pl.col("node_name") == name)["token_id"][0])


def _tiny_ontology_model(v_ext: int, ontology_dir: Path) -> ConditionalMultitaskARModel:
    torch.manual_seed(0)
    return ConditionalMultitaskARModel(
        config_overrides={
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "intermediate_size": 32,
            "max_position_embeddings": 64 + 15,
            "vocab_size": v_ext,
            "pad_token_id": 0,
            "attention_dropout": 0.0,
        },
        max_windows=5,
        ontology_dir=str(ontology_dir),
    ).eval()


def test_build_predict_datamodule_separates_cohort_width_from_table_width(
    tensorized_cohort_dir, tmp_path, cohort_ontology_dir
):
    """With an ontology the cohort on this machine is checked against the checkpoint's **cohort** width and
    the ontology against its **table** width, with distinct messages; the adapter's ceiling is the table width
    and the training-side kwarg the cohort width."""
    from every_query.data.ontology import extended_vocab_size

    grid = _write_grid(tmp_path / "grid", _eval_rows(), split=tuning_split)
    cfg = _train_cfg(tensorized_cohort_dir)
    v = _vocab_size(tensorized_cohort_dir, grid)
    v_ext = extended_vocab_size(cohort_ontology_dir)
    assert v_ext > v
    kw = {"use_rope_time": False, "max_windows": 5}

    dm = build_predict_datamodule(
        cfg,
        grid,
        tuning_split,
        expected_vocab_size=v_ext,
        base_vocab_size=v,
        ontology_dir=cohort_ontology_dir,
        **kw,
    )
    assert dm.ontology_dir == cohort_ontology_dir and dm.dataset_kwargs["expected_vocab_size"] == v
    ds = dm.predict_dataset
    assert ds.expected_vocab_size == v_ext and ds.ontology_dir == str(cohort_ontology_dir) and len(ds) == 4
    assert _an_ancestor(cohort_ontology_dir)[0] in ds.code_to_index

    # The cohort is not the checkpoint's: the message names the cohort width, not the table.
    with pytest.raises(ValueError, match=rf"trained on a cohort of vocab_size={v + 1}"):
        build_predict_datamodule(
            cfg,
            grid,
            tuning_split,
            expected_vocab_size=v_ext,
            base_vocab_size=v + 1,
            ontology_dir=cohort_ontology_dir,
            **kw,
        )
    # The ontology is not the checkpoint's: the message names the ontology.
    with pytest.raises(ValueError, match="different ontology"):
        build_predict_datamodule(
            cfg,
            grid,
            tuning_split,
            expected_vocab_size=v_ext + 1,
            base_vocab_size=v,
            ontology_dir=cohort_ontology_dir,
            **kw,
        )
    # Two widths without an ontology to explain them is an inconsistent checkpoint.
    with pytest.raises(ValueError, match="no ontology to account"):
        build_predict_datamodule(cfg, grid, tuning_split, expected_vocab_size=v + 1, base_vocab_size=v, **kw)
    # The pre-ontology call shape keeps its meaning: one width, the cohort must match it.
    with pytest.raises(ValueError, match="tied embedding table"):
        build_predict_datamodule(cfg, grid, tuning_split, expected_vocab_size=v_ext, **kw)


def test_build_predict_datamodule_checks_the_cohort_identity_not_just_its_width(
    tensorized_cohort_dir, tmp_path, cohort_ontology_dir
):
    """Regression for the PR #32 review.

    A checkpoint that recorded its training cohort's vocabulary
    fingerprint refuses a same-width cohort whose ``codes.parquet`` digests differently, with or without an
    ontology; and an ontology of this cohort's codes at two swapped indices (same ``V``, same ``V_ext``)
    passes every width check here and is refused by the evaluation adapter, code by code.
    """
    from every_query.data.ontology import cohort_code_map, extended_vocab_size
    from every_query.utils.digest import vocab_fingerprint

    grid = _write_grid(tmp_path / "grid", _eval_rows(), split=tuning_split)
    cfg = _train_cfg(tensorized_cohort_dir)
    v = _vocab_size(tensorized_cohort_dir, grid)
    v_ext = extended_vocab_size(cohort_ontology_dir)
    cohort = cohort_code_map(_data_config(tensorized_cohort_dir, grid).code_metadata_fp)
    fingerprint = vocab_fingerprint(cohort)
    kw = {"use_rope_time": False, "max_windows": 5}

    # The cohort on this machine is the checkpoint's: accepted with and without an ontology.
    with_onto = build_predict_datamodule(
        cfg,
        grid,
        tuning_split,
        expected_vocab_size=v_ext,
        base_vocab_size=v,
        ontology_dir=cohort_ontology_dir,
        cohort_vocab_fingerprint=fingerprint,
        **kw,
    )
    assert len(with_onto.predict_dataset) == 4
    plain = build_predict_datamodule(
        cfg, grid, tuning_split, expected_vocab_size=v, cohort_vocab_fingerprint=fingerprint, **kw
    )
    assert len(plain.predict_dataset) == 4
    # Same width, different codes.parquet (one code moved to the free PAD slot): the width check passes,
    # the fingerprint check does not.
    other = vocab_fingerprint({**cohort, Q1: 0})
    for onto_kw in ({}, {"ontology_dir": cohort_ontology_dir, "base_vocab_size": v}):
        table = v_ext if onto_kw else v
        with pytest.raises(ValueError, match=r"same width, other codes or a different numbering"):
            build_predict_datamodule(
                cfg,
                grid,
                tuning_split,
                expected_vocab_size=table,
                cohort_vocab_fingerprint=other,
                **onto_kw,
                **kw,
            )
    # A pre-fingerprint checkpoint (``None``) keeps the width-only behaviour.
    legacy = build_predict_datamodule(cfg, grid, tuning_split, expected_vocab_size=v, **kw)
    assert len(legacy.predict_dataset) == 4

    # A permuted ontology of this very cohort: widths agree everywhere, the adapter refuses it by name.
    permuted = write_cohort_ontology(tensorized_cohort_dir, tmp_path / "permuted", swap=(Q1, Q2))
    assert extended_vocab_size(permuted) == v_ext
    dm = build_predict_datamodule(
        cfg, grid, tuning_split, expected_vocab_size=v_ext, base_vocab_size=v, ontology_dir=permuted, **kw
    )
    by_name = rf"different codes\.parquet.*'{Q1}': ontology {cohort[Q2]} vs cohort {cohort[Q1]}.*'{Q2}'"
    with pytest.raises(ValueError, match=by_name):
        _ = dm.predict_dataset
    # ...and the model itself refuses to be built on it once it knows the cohort.
    with pytest.raises(ValueError, match=r"different codes\.parquet than this cohort"):
        ConditionalMultitaskARModel(
            config_overrides={
                "hidden_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "intermediate_size": 32,
                "max_position_embeddings": 64 + 15,
                "vocab_size": v_ext,
                "pad_token_id": 0,
            },
            max_windows=5,
            ontology_dir=str(permuted),
            cohort_vocab_fingerprint=fingerprint,
        )


def test_ancestor_query_scores_through_the_predict_datamodule(
    tensorized_cohort_dir, tmp_path, cohort_ontology_dir
):
    """A grid naming an ancestor node scores under an ontology checkpoint - through the same sequential
    loader, ``score_final_query`` and ``predictions_to_df`` as a leaf grid - and cannot even be tensorized
    for a leaf-width checkpoint (the name is an unknown code there)."""
    from every_query.data.ontology import extended_vocab_size

    name, index = _an_ancestor(cohort_ontology_dir)
    s = TUNING_SUBJECT
    rows = [_row(s, [name], [True]), _row(s, [Q1, name], [False, True]), _row(s, [name, Q2], [True, False])]
    grid = _write_grid(tmp_path / "grid", rows, split=tuning_split)
    cfg = _train_cfg(tensorized_cohort_dir)
    v = _vocab_size(tensorized_cohort_dir, grid)
    v_ext = extended_vocab_size(cohort_ontology_dir)
    assert index >= v

    with pytest.raises(ValueError, match="not in this run's vocabulary"):
        build_eval_dataset(cfg, grid, tuning_split, expected_vocab_size=v, use_rope_time=False)

    dm = build_predict_datamodule(
        cfg,
        grid,
        tuning_split,
        expected_vocab_size=v_ext,
        base_vocab_size=v,
        ontology_dir=cohort_ontology_dir,
        use_rope_time=False,
        max_windows=5,
        batch_size=2,
    )
    module = ConditionalMultitaskLightningModule(
        model=_tiny_ontology_model(v_ext, cohort_ontology_dir), optimizer=partial(torch.optim.AdamW, lr=1e-4)
    ).eval()
    probs, labels, codes = run_inference(module, dm, _cpu_trainer())
    ds = dm.predict_dataset
    assert codes.tolist() == [index, index, ds.code_to_index[Q2]]
    assert labels.tolist() == [True, True, False]
    assert probs.shape == (3,) and (probs > 0.0).all() and (probs < 1.0).all()
    with torch.no_grad():
        expected = torch.cat(
            [
                torch.sigmoid(module.model.score_final_query(b, b.scored_codes))
                for b in dm.predict_dataloader()
            ]
        )
    torch.testing.assert_close(torch.from_numpy(probs), expected.float())
    out = predictions_to_df(ds, probs, labels, codes)
    assert out["target_code"].to_list() == [name, name, Q2]
    assert out["queries"].to_list() == [r["queries"] for r in rows]


def test_grid_provenance_gate_compares_the_closure_halves(
    tmp_path, cohort_ontology_dir, tensorized_cohort_dir, caplog
):
    """``check_grid_ontology_provenance``: only a grid labeled under a *different* closure than the model's
    ontology is refused; a missing sidecar, a leaf-only grid and an ontology grid scored by a leaf model only
    warn."""
    import json
    import logging

    from every_query.data.ontology import closure_fingerprint
    from every_query.generate_tasks.sample_evaluation_query_sequences import _provenance_path

    root = tmp_path / "grid"
    tasks = root / "eval"
    _write_grid(tasks, _eval_rows(), split=tuning_split)
    fp = tasks / tuning_split / "0.parquet"

    def record(fingerprint: str | None) -> None:
        sidecar = _provenance_path(root, fp)
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        payload = {"ontology_fingerprint": fingerprint, "specs_fingerprint": "s", "cohort_fingerprint": "c"}
        sidecar.write_text(json.dumps(payload))

    # A second ontology, built from a cohort missing one code: same universe shape, different closure.
    codes = pl.read_parquet(tensorized_cohort_dir / "metadata" / "codes.parquet")
    hierarchical = codes.filter(pl.col("code").str.contains("//"))["code"].to_list()
    stale_cohort = tmp_path / "stale_cohort"
    (stale_cohort / "metadata").mkdir(parents=True)
    codes.filter(pl.col("code") != hierarchical[0]).write_parquet(stale_cohort / "metadata" / "codes.parquet")
    stale = write_cohort_ontology(stale_cohort, tmp_path / "stale_ontology")
    closure, stale_closure = closure_fingerprint(cohort_ontology_dir), closure_fingerprint(stale)
    assert closure != stale_closure

    with caplog.at_level(logging.WARNING):
        check_grid_ontology_provenance(tasks, tuning_split, cohort_ontology_dir)
    assert "no provenance sidecar" in caplog.text

    record(None)  # a leaf-only grid: nothing to compare, whatever the model has
    caplog.clear()
    check_grid_ontology_provenance(tasks, tuning_split, cohort_ontology_dir)
    check_grid_ontology_provenance(tasks, tuning_split, None)
    assert not caplog.text

    record(f"{closure}|universe")  # the model's own closure
    check_grid_ontology_provenance(tasks, tuning_split, cohort_ontology_dir)
    assert not caplog.text

    record(f"{stale_closure}|universe")  # a different closure: refused
    with pytest.raises(ValueError, match="differs from the checkpoint's"):
        check_grid_ontology_provenance(tasks, tuning_split, cohort_ontology_dir)
    # ...but a leaf model only warns: the grid's leaf labels are unaffected and its ancestor names
    # are rejected as unknown codes by the dataset.
    with caplog.at_level(logging.WARNING):
        check_grid_ontology_provenance(tasks, tuning_split, None)
    assert "checkpoint has none" in caplog.text


# --- 5 / 6 / 7: conditions, final code and label under padding -------------------------------


def test_conditions_and_final_query_follow_each_real_row(tensorized_cohort_dir, tmp_path):
    rows = _mixed_rows()
    ds = _dataset(tensorized_cohort_dir, _write_grid(tmp_path / "grid", rows))
    idx = ds.code_to_index
    batch = _collate_all(ds)
    assert batch.num_bounds == 3  # padded to the longest row, not to a fixed K
    assert batch.q_mask.tolist() == [
        [True, False, False],
        [True, True, False],
        [True, True, True],
        [True, True, False],
    ]
    assert batch.n_queries.tolist() == [1, 2, 3, 2]
    for i, row in enumerate(rows):
        n = len(row["queries"])
        # Conditions: all and only the real queries before the final one, PAD / False beyond.
        assert batch.condition_codes[i, : n - 1].tolist() == [idx[c] for c in row["queries"][:-1]]
        assert batch.condition_answers[i, : n - 1].tolist() == row["answers"][:-1]
        assert (batch.condition_codes[i, n - 1 :] == 0).all()
        assert not batch.condition_answers[i, n - 1 :].any()
        # The final query of the row, never the padded width.
        assert int(batch.scored_codes[i]) == idx[row["queries"][-1]]
        assert bool(batch.labels[i]) == row["answers"][-1]
        assert batch.q_durations[i, :n].tolist() == row["durations"]
        assert (batch.q_durations[i, n:] == 0).all()


def test_one_query_rows_have_no_conditioning_pairs_and_k_may_be_one(tensorized_cohort_dir, tmp_path):
    a, b = _TRAIN_SUBJECTS[:2]
    ds = _dataset(
        tensorized_cohort_dir, _write_grid(tmp_path / "grid", [_row(a, [Q1], [True]), _row(b, [Q3], [False])])
    )
    batch = _collate_all(ds)
    assert batch.num_bounds == 1
    assert batch.condition_codes.shape == (2, 0) and batch.condition_answers.shape == (2, 0)
    assert batch.q_mask.tolist() == [[True], [True]]
    assert batch.scored_codes.tolist() == [ds.code_to_index[Q1], ds.code_to_index[Q3]]
    assert batch.labels.tolist() == [True, False]


def _forced_rows() -> list[dict]:
    """``_mixed_rows`` with a designed conditioning answer wherever a row has one to give.

    Every forced value is the *opposite* of the labeled answer, so a collate that ignored the column could not
    pass by coincidence.
    """
    a, b, c, d = _TRAIN_SUBJECTS
    return [
        _row(a, [Q1], [True], forced_answers=[None]),
        _row(b, [Q1, Q2], [False, True], forced_answers=[True, None]),
        _row(c, [Q2, Q1, Q3], [True, True, False], forced_answers=[None, False, None]),
        _row(d, [Q3, Q2], [False, False], forced_answers=[None, None]),
    ]


def test_forced_answers_replace_the_conditioning_answer_and_never_the_label(tensorized_cohort_dir, tmp_path):
    rows = _forced_rows()
    forced_ds = _dataset(tensorized_cohort_dir, _write_grid(tmp_path / "forced", rows))
    forced = _collate_all(forced_ds)
    truth = _collate_all(_dataset(tensorized_cohort_dir, _write_grid(tmp_path / "truth", _mixed_rows())))

    # Forced where given, the labeled truth where null.
    assert forced.condition_answers.tolist() == [
        [False, False],
        [True, False],
        [True, False],
        [False, False],
    ]
    assert truth.condition_answers.tolist() == [
        [False, False],
        [False, False],
        [True, True],
        [False, False],
    ]
    # Everything that is not the conditioning answer is untouched: the label above all.
    assert forced.labels.tolist() == truth.labels.tolist() == [r["answers"][-1] for r in rows]
    assert torch.equal(forced.condition_codes, truth.condition_codes)
    assert torch.equal(forced.scored_codes, truth.scored_codes)

    # And the model is actually conditioned on it: only the two rows with a forced answer move.
    model = _tiny_model(max(forced_ds.code_to_index.values()) + 1)
    with torch.no_grad():
        moved = model.score_final_query(forced, forced.scored_codes) != model.score_final_query(
            truth, truth.scored_codes
        )
    assert moved.tolist() == [False, True, True, False]


def test_an_all_null_forced_answers_column_is_the_unforced_grid(tensorized_cohort_dir, tmp_path):
    rows = [{**r, "forced_answers": [None] * len(r["queries"])} for r in _mixed_rows()]
    nulls = _collate_all(_dataset(tensorized_cohort_dir, _write_grid(tmp_path / "nulls", rows)))
    plain = _collate_all(_dataset(tensorized_cohort_dir, _write_grid(tmp_path / "plain", _mixed_rows())))
    assert torch.equal(nulls.condition_answers, plain.condition_answers)


@pytest.mark.parametrize(
    "queries, answers, forced",
    [([Q1, Q2], [False, True], [None, True]), ([Q1], [True], [False])],
    ids=["final-of-two", "only-query"],
)
def test_a_forced_answer_on_the_final_query_is_rejected(
    tensorized_cohort_dir, tmp_path, queries, answers, forced
):
    """The generator enforces this on specs; a hand-built grid must hit the same wall at load."""
    a, b = _TRAIN_SUBJECTS[:2]
    rows = [
        _row(a, [Q1, Q2], [True, True], forced_answers=[False, None]),
        _row(b, queries, answers, forced_answers=forced),
    ]
    with pytest.raises(ValueError, match="labels row 1 forces the answer of its final query"):
        _dataset(tensorized_cohort_dir, _write_grid(tmp_path / "grid", rows))


def test_ragged_forced_answers_are_rejected(tensorized_cohort_dir, tmp_path):
    a = _TRAIN_SUBJECTS[0]
    grid = _write_grid(tmp_path / "grid", [_row(a, [Q1, Q2], [True, True], forced_answers=[None])])
    with pytest.raises(ValueError, match="forced_answers list lengths disagree"):
        _dataset(tensorized_cohort_dir, grid)


def test_a_model_that_cannot_honor_forced_answers_refuses_the_grid(tensorized_cohort_dir, tmp_path):
    """The plain sequence dataset uses one answers tensor as input *and* target, so it must not silently drop
    a designed conditioning answer; an all-null column is still fine."""
    forced = _write_grid(tmp_path / "forced", _forced_rows())
    with pytest.raises(ValueError, match=r"are forced.*EQ_predict_multitask"):
        QuerySeqPytorchDataset(_data_config(tensorized_cohort_dir, forced), split=train_split)
    rows = [{**r, "forced_answers": [None] * len(r["queries"])} for r in _mixed_rows()]
    nulls = _write_grid(tmp_path / "nulls", rows)
    assert len(QuerySeqPytorchDataset(_data_config(tensorized_cohort_dir, nulls), split=train_split)) == 4


def test_empty_query_lists_are_rejected(tensorized_cohort_dir, tmp_path):
    a = _TRAIN_SUBJECTS[0]
    grid = _write_grid(tmp_path / "grid", [_row(a, [], [])])
    with pytest.raises(ValueError, match="at least one query"):
        _dataset(tensorized_cohort_dir, grid)


# --- 8: the adapter's batch drives the same scoring path as the training batch ----------------


def test_adapter_batch_scores_like_a_training_batch_with_the_same_windows(tensorized_cohort_dir, tmp_path):
    """Feed the adapter's batch and a ``MultitaskBoundaryBatch`` carrying identical windows to the
    model: the target-only logit equals the gathered full-vocabulary logit of the training path."""
    ds = _dataset(tensorized_cohort_dir, _write_grid(tmp_path / "grid", _mixed_rows()))
    batch = _collate_all(ds)
    vocab = max(ds.code_to_index.values()) + 1
    model = _tiny_model(vocab)
    target_only = model.score_final_query(batch, batch.scored_codes)

    training_like = MultitaskBoundaryBatch(
        code=batch.code,
        numeric_value=batch.numeric_value,
        numeric_value_mask=batch.numeric_value_mask,
        time_delta_days=batch.time_delta_days,
        q_start_durations=batch.q_start_durations,
        q_start_codes=batch.q_start_codes,
        q_durations=batch.q_durations,
        q_bound_codes=batch.q_bound_codes,
        q_mask=batch.q_mask,
        targets=torch.zeros(batch.batch_size, batch.num_bounds, vocab, dtype=torch.bool),
        condition_codes=batch.condition_codes,
        condition_answers=batch.condition_answers,
    )
    _, out = model(training_like)
    last = batch.q_mask.sum(1) - 1
    gathered = out.logits[torch.arange(batch.batch_size), last, batch.scored_codes]
    torch.testing.assert_close(target_only, gathered, atol=1e-5, rtol=1e-5)


# --- 10 / 11: row alignment through the Lightning predict loop, and dropped subjects ------------


def _predict_setup(
    cohort: Path, tmp_path: Path, *, batch_size: int = 3
) -> tuple[ConditionalMultitaskDataModule, ConditionalMultitaskLightningModule]:
    """A prediction datamodule over the tuning-subject grid and a tiny Lightning module sized to the
    cohort."""
    grid = _write_grid(tmp_path / "grid", _eval_rows(), split=tuning_split)
    v = _vocab_size(cohort, grid)
    dm = build_predict_datamodule(
        _train_cfg(cohort),
        grid,
        tuning_split,
        expected_vocab_size=v,
        use_rope_time=False,
        max_windows=5,
        batch_size=batch_size,
    )
    module = ConditionalMultitaskLightningModule(
        model=_tiny_model(v), optimizer=partial(torch.optim.AdamW, lr=1e-4)
    ).eval()
    return dm, module


def _cpu_trainer(precision: str = "32-true") -> L.Trainer:
    """Full precision by default so the exactness tests below compare like with like."""
    return build_predict_trainer("cpu", precision=precision, enable_progress_bar=False)


def test_predict_trainer_defaults_to_bf16_mixed_and_agrees_with_fp32_to_bf16_rounding(
    tensorized_cohort_dir, tmp_path
):
    """The inference trainer defaults to the production training precision, ``bf16-mixed``; its probabilities
    match a full-precision run up to bf16 rounding, and ``precision=32-true`` opts out."""
    default = build_predict_trainer("cpu", enable_progress_bar=False)
    assert default.precision == "bf16-mixed"
    assert _cpu_trainer().precision == "32-true"

    dm, module = _predict_setup(tensorized_cohort_dir, tmp_path)
    probs_fp32, labels_fp32, codes_fp32 = run_inference(module, dm, _cpu_trainer())
    probs_bf16, labels_bf16, codes_bf16 = run_inference(module, dm, _cpu_trainer("bf16-mixed"))
    assert probs_bf16.dtype == np.float32 and probs_bf16.shape == probs_fp32.shape
    np.testing.assert_array_equal(labels_bf16, labels_fp32)
    np.testing.assert_array_equal(codes_bf16, codes_fp32)
    np.testing.assert_allclose(probs_bf16, probs_fp32, atol=2e-2, rtol=0.0)


def test_run_inference_matches_score_final_query_over_the_sequential_loader(tensorized_cohort_dir, tmp_path):
    """Through a real CPU ``Trainer``: ``probs`` is ``sigmoid(score_final_query)`` batch by batch over the
    datamodule's sequential loader (four rows in batches of three and one), ``labels`` / ``scored_codes`` the
    collated final answers / final query indices, all in loader order."""
    dm, module = _predict_setup(tensorized_cohort_dir, tmp_path)
    ds = dm.predict_dataset
    probs, labels, codes = run_inference(module, dm, _cpu_trainer())

    assert isinstance(probs, np.ndarray) and probs.dtype == np.float32 and probs.shape == (4,)
    assert isinstance(labels, np.ndarray) and labels.dtype == np.bool_ and labels.shape == (4,)
    assert isinstance(codes, np.ndarray) and codes.dtype == np.int64 and codes.shape == (4,)
    with torch.no_grad():
        batches = list(dm.predict_dataloader())
        assert [b.batch_size for b in batches] == [3, 1]
        expected = torch.cat(
            [torch.sigmoid(module.model.score_final_query(b, b.scored_codes)) for b in batches]
        )
        expected_labels = torch.cat([b.labels for b in batches])
        expected_codes = torch.cat([b.scored_codes for b in batches])
    torch.testing.assert_close(torch.from_numpy(probs), expected.float())
    assert labels.tolist() == expected_labels.tolist() == [r["answers"][-1] for r in _eval_rows()]
    assert (
        codes.tolist()
        == expected_codes.tolist()
        == [ds.code_to_index[r["queries"][-1]] for r in _eval_rows()]
    )
    assert len({round(p, 5) for p in probs.tolist()}) > 1, "rows must not collapse to one value"


def test_predictions_are_row_aligned_one_per_grid_row(tensorized_cohort_dir, tmp_path):
    rows = _eval_rows()
    dm, module = _predict_setup(tensorized_cohort_dir, tmp_path)
    ds = dm.predict_dataset
    probs, labels, codes = run_inference(module, dm, _cpu_trainer())
    out = predictions_to_df(ds, probs, labels, codes)

    assert out.height == len(rows) == len(ds)
    assert out.columns == [
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
    assert out["forced_answers"].to_list() == [[None] * len(r["queries"]) for r in rows]
    assert out["subject_id"].to_list() == [r["subject_id"] for r in rows]
    assert out["queries"].to_list() == [r["queries"] for r in rows]
    assert out["target_code"].to_list() == [r["queries"][-1] for r in rows]
    assert out["label"].to_list() == [r["answers"][-1] for r in rows]
    # Normalized defaults are written even though the grid lacked the optional columns.
    assert out["start_durations"].to_list() == [[0.0] * len(r["queries"]) for r in rows]
    assert out["start_events"].to_list() == [[None] * len(r["queries"]) for r in rows]
    assert out["bound_events"].to_list() == [[None] * len(r["queries"]) for r in rows]
    # Each prob is the one computed for *that* row: scoring row i on its own (no padding from its
    # batch mates) gives the same number.
    with torch.no_grad():
        alone = []
        for i in range(len(ds)):
            single = ds.collate([ds[i]])
            alone.append(torch.sigmoid(module.model.score_final_query(single, single.scored_codes)).item())
    assert out["prob"].to_numpy() == pytest.approx(alone, abs=1e-4)
    assert out["prob"].dtype == pl.Float32


def test_run_inference_requires_every_row(tensorized_cohort_dir, tmp_path, monkeypatch):
    """A loader that yields fewer rows than the grid has is an error, not a shorter output."""
    dm, module = _predict_setup(tensorized_cohort_dir, tmp_path)
    ds = dm.predict_dataset

    def truncated() -> DataLoader:
        return DataLoader(Subset(ds, range(3)), batch_size=2, shuffle=False, collate_fn=ds.collate)

    monkeypatch.setattr(dm, "predict_dataloader", truncated)
    with pytest.raises(RuntimeError, match=r"yielded 3 prediction\(s\) but the grid has 4 row\(s\)"):
        run_inference(module, dm, _cpu_trainer())


def test_run_inference_never_calls_the_dense_forward_or_projects_onto_the_vocabulary(
    tensorized_cohort_dir, tmp_path, monkeypatch
):
    """The predict loop scores through ``score_final_query`` only: the dense ``forward`` is never entered and
    no ``(.., V)``-wide matmul happens."""
    dm, module = _predict_setup(tensorized_cohort_dir, tmp_path)
    vocab = module.model.vocab_size

    def boom(self, batch):
        raise AssertionError("dense forward called during prediction")

    monkeypatch.setattr(ConditionalMultitaskARModel, "forward", boom)
    calls = []
    real_matmul = torch.Tensor.__matmul__

    def spy(a, b):
        calls.append((tuple(a.shape), tuple(b.shape)))
        return real_matmul(a, b)

    monkeypatch.setattr(torch.Tensor, "__matmul__", spy)

    probs, labels, codes = run_inference(module, dm, _cpu_trainer())
    assert probs.shape == labels.shape == codes.shape == (4,)
    assert not any(shape[-1] == vocab for _, shape in calls), "no (.., V) projection may be built"


def test_predictions_to_df_rejects_misaligned_labels_and_scored_codes(tensorized_cohort_dir, tmp_path):
    """Rows 0 and 1 share a final label but score different codes: swapping them keeps the labels aligned
    and is caught by the scored-code check, which the label check alone would miss."""
    rows = _mixed_rows()
    ds = _dataset(tensorized_cohort_dir, _write_grid(tmp_path / "grid", rows))
    probs = np.zeros(len(ds), dtype=np.float32)
    labels = np.array([r["answers"][-1] for r in rows])
    codes = np.array([ds.code_to_index[r["queries"][-1]] for r in rows], dtype=np.int64)
    predictions_to_df(ds, probs, labels, codes)
    with pytest.raises(RuntimeError, match="disagrees with answers\\[-1\\]"):
        predictions_to_df(ds, probs, ~labels, codes)

    assert labels[0] == labels[1] and codes[0] != codes[1]
    swapped = codes.copy()
    swapped[[0, 1]] = codes[[1, 0]]
    with pytest.raises(RuntimeError, match="scored code disagrees with queries\\[-1\\] at grid row 0"):
        predictions_to_df(ds, probs, labels, swapped)
    with pytest.raises(RuntimeError, match="for 4 grid row"):
        predictions_to_df(ds, probs[:-1], labels[:-1], codes[:-1])


def test_a_same_label_row_swap_in_the_loader_is_caught_by_the_scored_codes(
    tensorized_cohort_dir, tmp_path, monkeypatch
):
    """End to end through ``run_inference``: a loader that yields grid rows 0 and 1 in the wrong order (same
    final label, different final query) passes the label check and is stopped by the scored codes, so a
    probability can never be attached to the wrong query."""
    rows = _eval_rows()
    assert (
        rows[0]["answers"][-1] == rows[1]["answers"][-1] and rows[0]["queries"][-1] != rows[1]["queries"][-1]
    )
    dm, module = _predict_setup(tensorized_cohort_dir, tmp_path)
    ds = dm.predict_dataset

    def swapped() -> DataLoader:
        return DataLoader(Subset(ds, [1, 0, 2, 3]), batch_size=3, shuffle=False, collate_fn=ds.collate)

    monkeypatch.setattr(dm, "predict_dataloader", swapped)
    probs, labels, codes = run_inference(module, dm, _cpu_trainer())
    assert labels.tolist() == [r["answers"][-1] for r in rows], "the label check alone cannot see the swap"
    with pytest.raises(RuntimeError, match="scored code disagrees with queries\\[-1\\] at grid row 0"):
        predictions_to_df(ds, probs, labels, codes)


def test_a_grid_subject_absent_from_the_cohort_is_rejected(tensorized_cohort_dir, tmp_path):
    rows = [
        *_mixed_rows(),
        {
            "subject_id": 999_999_999,
            "prediction_time": datetime(2010, 1, 1),
            "queries": [Q1],
            "durations": [7.0],
            "answers": [True],
        },
    ]
    grid = _write_grid(tmp_path / "grid", rows)
    with pytest.raises(RuntimeError, match="1 grid row\\(s\\) were dropped"):
        _dataset(tensorized_cohort_dir, grid)


def test_only_this_splits_shards_are_read(tensorized_cohort_dir, tmp_path):
    """A grid root holding another split's parquets (or several shards) is read per split."""
    grid = _write_grid(tmp_path / "grid", _mixed_rows()[:2], shard="0")
    _write_grid(grid, _mixed_rows()[2:], shard="1")
    _write_grid(grid, [_row(_TRAIN_SUBJECTS[0], [Q1], [True])], split="tuning", shard="0")
    ds = _dataset(tensorized_cohort_dir, grid)
    assert ds.n_grid_rows == 4 and len(ds) == 4


def test_empty_grid_is_rejected(tensorized_cohort_dir, tmp_path):
    grid = tmp_path / "grid"
    (grid / train_split).mkdir(parents=True)
    pl.DataFrame(
        schema={
            "subject_id": pl.Int64,
            "prediction_time": pl.Datetime("us"),
            "queries": pl.List(pl.Utf8),
            "durations": pl.List(pl.Float32),
            "answers": pl.List(pl.Boolean),
        }
    ).write_parquet(grid / train_split / "0.parquet")
    with pytest.raises(ValueError, match="has no rows"):
        _dataset(tensorized_cohort_dir, grid)
    # A grid root that exists but holds no parquet for *this* split.
    other = _write_grid(tmp_path / "other", [_row(_TRAIN_SUBJECTS[0], [Q1], [True])], split="tuning")
    with pytest.raises(FileNotFoundError, match="no QuerySeqSchema parquets"):
        _dataset(tensorized_cohort_dir, other)


# --- 13: device -> single-device Lightning trainer ------------------------------------------------


@pytest.mark.parametrize(
    "device, expected",
    [
        (None, ("auto", 1)),
        ("cpu", ("cpu", 1)),
        ("cpu:0", ("cpu", 1)),
        ("cuda", ("gpu", 1)),
        ("cuda:3", ("gpu", [3])),
        ("mps", ("mps", 1)),
    ],
    ids=["null", "cpu", "cpu:0", "cuda", "cuda:3", "mps"],
)
def test_device_strings_map_to_one_lightning_device(device, expected):
    assert resolve_accelerator(device) == expected


@pytest.mark.parametrize("device", ["xpu", "tpu", "not-a-device", "2"])
def test_unsupported_device_strings_are_rejected(device):
    with pytest.raises(ValueError, match="device must be null, cpu, cuda, cuda:N or mps"):
        resolve_accelerator(device)


def test_predict_trainer_is_single_device_and_multi_device_trainers_are_rejected():
    trainer = _cpu_trainer()
    assert isinstance(trainer.strategy, SingleDeviceStrategy)
    assert trainer.strategy.root_device.type == "cpu"
    assert trainer.num_devices == 1 and trainer.num_nodes == 1 and trainer.world_size == 1
    assert trainer.loggers == [] and trainer.checkpoint_callbacks == []
    check_single_device(trainer)  # no raise

    quiet = {"logger": False, "enable_checkpointing": False, "enable_progress_bar": False}
    for kwargs in ({"devices": 2}, {"devices": 1, "num_nodes": 2}, {"devices": 1, "strategy": "ddp"}):
        multi = L.Trainer(accelerator="cpu", enable_model_summary=False, **quiet, **kwargs)
        with pytest.raises(ValueError, match="exactly one device") as err:
            check_single_device(multi)
        assert "row-aligned with dataset.schema_df" in str(err.value) and "issue #30" in str(err.value)


# What ``torchrun --nproc_per_node=2`` and ``srun --ntasks=2`` put in rank 0's environment (the
# rendezvous / rank variables ``torch.distributed`` and ``slurmstepd`` set per task).
_TORCHRUN_RANK0 = {
    "TORCHELASTIC_RUN_ID": "eq-predict-test",
    "WORLD_SIZE": "2",
    "LOCAL_WORLD_SIZE": "2",
    "RANK": "0",
    "LOCAL_RANK": "0",
    "GROUP_RANK": "0",
    "MASTER_ADDR": "127.0.0.1",
    "MASTER_PORT": "29500",
}
_SRUN_RANK0 = {
    "SLURM_JOB_ID": "12345",
    "SLURM_JOB_NAME": "eq_predict_multitask",
    "SLURM_NTASKS": "2",
    "SLURM_STEP_ID": "0",
    "SLURM_STEP_NUM_TASKS": "2",
    "SLURM_PROCID": "0",
    "SLURM_LOCALID": "0",
    "SLURM_NODEID": "0",
}
# The batch step of ``sbatch --ntasks=2`` running the CLI directly (no ``srun``): one real process,
# but SLURM says two tasks and nothing distinguishes it from rank 0 of two, so it is refused too
# (the message says to use ``--ntasks=1``).
_SBATCH_NTASKS2_BATCH_STEP = {
    "SLURM_JOB_ID": "12345",
    "SLURM_JOB_NAME": "eq_predict_multitask",
    "SLURM_NTASKS": "2",
    "SLURM_PROCID": "0",
    "SLURM_LOCALID": "0",
    "SLURM_NODEID": "0",
}
# An ``salloc -n 2`` shell before any ``srun``: the allocation's variables, no task-level ones.
_SALLOC_SHELL = {"SLURM_JOB_ID": "12345", "SLURM_JOB_NAME": "interactive", "SLURM_NTASKS": "2"}
_LAUNCHER_VARS = {
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "RANK",
    "LOCAL_RANK",
    "GROUP_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
}


def _simulate_launcher(monkeypatch, env: dict[str, str]) -> None:
    """Replace whatever launcher variables this test process inherited with ``env``."""
    for key in list(os.environ):
        if key in _LAUNCHER_VARS or key.startswith(("SLURM_", "TORCHELASTIC_")):
            monkeypatch.delenv(key)
    for key, value in env.items():
        monkeypatch.setenv(key, value)


@pytest.mark.parametrize(
    "env, expected",
    [
        (_TORCHRUN_RANK0, 2),
        (_SRUN_RANK0, 2),
        ({**_SRUN_RANK0, "SLURM_JOB_NAME": "bash"}, 2),
        (_SBATCH_NTASKS2_BATCH_STEP, 2),
        ({**_TORCHRUN_RANK0, "WORLD_SIZE": "1", "LOCAL_WORLD_SIZE": "1"}, 1),
        ({**_SRUN_RANK0, "SLURM_NTASKS": "1", "SLURM_STEP_NUM_TASKS": "1"}, 1),
        ({**_SBATCH_NTASKS2_BATCH_STEP, "SLURM_NTASKS": "1"}, 1),
        (_SALLOC_SHELL, 1),
        ({}, 1),
    ],
    ids=[
        "torchrun-2",
        "srun-2",
        "srun-2-inside-a-bash-named-allocation",
        "sbatch-ntasks-2-batch-step",
        "torchrun-1",
        "srun-1",
        "sbatch-ntasks-1",
        "salloc-shell-no-srun",
        "plain",
    ],
)
def test_multi_process_launchers_are_refused_though_the_trainer_looks_single_device(
    monkeypatch, env, expected, tensorized_cohort_dir, tmp_path
):
    """``devices=1`` resolves to a ``SingleDeviceStrategy`` whose ``world_size`` is 1 even inside a
    ``torchrun`` / ``srun --ntasks=2`` job, where every rank would score the whole grid and write the same
    parquet.

    The guard counts the launcher's processes instead - before the trainer exists
    (:func:`check_single_process`, as ``main`` calls it) and again inside :func:`check_single_device`
    - and refuses before ``trainer.predict``.  One-task launches, an ``salloc`` shell that has not
    run ``srun`` and a plain launch pass; a ``bash``-named allocation is no exemption.
    """
    _simulate_launcher(monkeypatch, env)
    assert launcher_world_size() == expected
    if expected == 1:
        check_single_process()  # no raise
        trainer = _cpu_trainer()  # built under the launcher's environment, as the CLI's would be
        assert isinstance(trainer.strategy, SingleDeviceStrategy) and trainer.world_size == 1
        check_single_device(trainer)  # no raise
        return

    with pytest.raises(ValueError, match=r"the launcher started 2 process\(es\)") as err:
        check_single_process()
    assert "--ntasks=1" in str(err.value)

    # Lightning's own SLURM detection may refuse to build a Trainer under some of these environments
    # (it rejects ``--ntasks=N`` without ``--ntasks-per-node``); ``main`` therefore checks first.
    try:
        trainer = _cpu_trainer()
    except RuntimeError as e:
        assert "ntasks" in str(e), e
        return
    assert isinstance(trainer.strategy, SingleDeviceStrategy) and trainer.world_size == 1
    with pytest.raises(ValueError, match=r"the launcher started 2 process\(es\)"):
        check_single_device(trainer)

    dm, module = _predict_setup(tensorized_cohort_dir, tmp_path)

    def never(*args, **kwargs):
        raise AssertionError("trainer.predict ran under a multi-process launcher")

    monkeypatch.setattr(trainer, "predict", never)
    with pytest.raises(ValueError, match="exactly one process"):
        run_inference(module, dm, trainer)
