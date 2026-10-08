"""``ConditionalMultitaskDataModule`` (issue #30): two roots, one datamodule.

The fit side reads the multitask sampler's packed training labels (``config.task_labels_dir``); the
test / predict side reads a ``QuerySeqSchema`` evaluation grid (``eval_tasks_dir``).  What is checked:

1. ``train_dataloader`` is the resumable ``StatefulDataLoader`` and both fit loaders yield
   ``MultitaskBoundaryBatch``;
2. ``test_dataloader`` / ``predict_dataloader`` yield ``MultitaskEvalBatch`` sequentially, in grid order;
3. ``eval_config`` is ``config`` with ``task_labels_dir`` alone replaced;
4. nothing grid-related is touched by construction or the fit loaders, and a missing grid root is
   refused with an error naming ``eval_tasks_dir``;
5. ``predict_split`` selects the grid split and ``train`` is rejected;
6. ``strip_delta_tokens`` / ``expected_vocab_size`` / ``max_windows`` reach the evaluation adapter, and
   ``ontology_dir`` reaches it *and* the training datasets (which need it to read ancestor-valued
   start / bound / conditioning codes) while their target width stays the leaf manifest's ``V``;
7. the demo train config's ``datamodule`` node instantiates this class through Hydra.

The training labels are generated in-process by the multitask sampler against the session fixture
cohort's vocabulary, so the manifest fingerprint matches the cohort the datasets join against.
"""

import dataclasses
import json
import shutil
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path

import polars as pl
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from meds import held_out_split, train_split, tuning_split
from meds_torchdata.config import MEDSTorchDataConfig
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from conftest import _PRED_TIMES, _TUNING_SUBJECTS
from every_query.data.conditional_multitask_datamodule import (
    EVAL_SPLITS,
    MISSING_EVAL_TASKS_DIR_MSG,
    ConditionalMultitaskDataModule,
)
from every_query.data.multitask_dataset import (
    MANIFEST_NAME,
    MultitaskBoundaryBatch,
    MultitaskBoundaryPytorchDataset,
)
from every_query.data.multitask_eval_dataset import MultitaskEvalBatch, QuerySeqMultitaskEvalDataset
from every_query.generate_tasks import sample_multitask_sequences as sms
from tests.multitask.conftest import base_cfg, write_cohort_ontology

TRAIN_CONFIGS = str(files("every_query") / "train" / "configs")

# Real codes and subjects of the static sample cohort behind ``tensorized_cohort_dir``.
Q1, Q2, Q3 = "HR", "TEMP", "DISCHARGE"
HELD_OUT_SUBJECT = 1500733
HELD_OUT_PRED_TIME = datetime(2010, 6, 3, 15, 0, tzinfo=UTC)  # after the subject's first timed events
TUNING_SUBJECT = _TUNING_SUBJECTS[0]


def _grid_row(subject: int, pred_time: datetime, queries: list[str], answers: list[bool]) -> dict:
    return {
        "subject_id": subject,
        "prediction_time": pred_time,
        "queries": queries,
        "durations": [float(7 * (i + 1)) for i in range(len(queries))],
        "answers": answers,
    }


# 1-, 2-, 3-, 2- and 1-query rows with mixed final labels, so batches of two straddle rows of
# different lengths and a shuffled or re-sorted loader would be caught by the label order.
HELD_OUT_ROWS = [
    _grid_row(HELD_OUT_SUBJECT, HELD_OUT_PRED_TIME, [Q1], [True]),
    _grid_row(HELD_OUT_SUBJECT, HELD_OUT_PRED_TIME, [Q1, Q2], [False, False]),
    _grid_row(HELD_OUT_SUBJECT, HELD_OUT_PRED_TIME, [Q2, Q1, Q3], [True, False, True]),
    _grid_row(HELD_OUT_SUBJECT, HELD_OUT_PRED_TIME, [Q3, Q2], [True, False]),
    _grid_row(HELD_OUT_SUBJECT, HELD_OUT_PRED_TIME, [Q2], [True]),
]
TUNING_ROWS = [
    _grid_row(TUNING_SUBJECT, _PRED_TIMES[TUNING_SUBJECT], [Q1, Q3], [True, False]),
    _grid_row(TUNING_SUBJECT, _PRED_TIMES[TUNING_SUBJECT], [Q2], [True]),
]


def _write_grid(root: Path, rows: list[dict], split: str) -> Path:
    """Write ``rows`` as one ``QuerySeqSchema`` shard under ``{root}/{split}/``; return ``root``."""
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    schema = {
        "subject_id": pl.Int64,
        "prediction_time": pl.Datetime("us"),
        "queries": pl.List(pl.Utf8),
        "durations": pl.List(pl.Float32),
        "answers": pl.List(pl.Boolean),
    }
    pl.DataFrame(rows, schema=schema).write_parquet(split_dir / "0.parquet")
    return root


@pytest.fixture(scope="module")
def grid_dir(tmp_path_factory) -> Path:
    """A QuerySeq evaluation grid ``eval/`` root holding a held-out and a tuning split."""
    root = tmp_path_factory.mktemp("queryseq_grid") / "eval"
    _write_grid(root, HELD_OUT_ROWS, held_out_split)
    _write_grid(root, TUNING_ROWS, tuning_split)
    return root


@pytest.fixture(scope="module")
def multitask_labels_dir(simple_static_MEDS: Path, tensorized_cohort_dir: Path, tmp_path_factory) -> Path:
    """``EQ_generate_multitask_sequences`` output for train + tuning, run in-process.

    ``query_codes`` is the tensorized cohort, so the manifest's vocabulary fingerprint is the one
    ``MultitaskBoundaryPytorchDataset`` recomputes from ``config.code_metadata_fp``.
    """
    out_dir = tmp_path_factory.mktemp("multitask_labels")
    for split in (train_split, tuning_split):
        cfg = OmegaConf.create(
            base_cfg(
                simple_static_MEDS,
                out_dir,
                query_codes=str(tensorized_cohort_dir),
                split=split,
                num_training_examples=8,
                duration_min=0.01,
                duration_max=2,
                eventstart_fraction=0.25,
                prediction_time_start_fraction=0.25,
                start_duration_min=0.01,
                start_duration_max=2,
                min_prediction_times_per_subject=1,
                max_workers=1,
                label_chunk_rows=2,
                seed=1,
            )
        )
        sms.run(cfg)
    return out_dir


@pytest.fixture(scope="module")
def data_config(tensorized_cohort_dir: Path, multitask_labels_dir: Path) -> MEDSTorchDataConfig:
    """The checkpoint-side config: cohort settings plus the TRAINING labels root."""
    return MEDSTorchDataConfig(
        tensorized_cohort_dir=str(tensorized_cohort_dir),
        task_labels_dir=str(multitask_labels_dir),
        max_seq_len=64,
        seq_sampling_strategy="to_end",
        static_inclusion_mode="omit",
        batch_mode="SM",
    )


def _datamodule(config: MEDSTorchDataConfig, **kwargs) -> ConditionalMultitaskDataModule:
    kwargs.setdefault("batch_size", 2)
    kwargs.setdefault("dataset_kwargs", {"expected_vocab_size": config.vocab_size})
    return ConditionalMultitaskDataModule(config, **kwargs)


# --- 1: the fit side ---------------------------------------------------------------------------------


def test_fit_loaders_are_the_resumable_training_loaders(data_config):
    dm = _datamodule(data_config)
    assert dm.data_class is MultitaskBoundaryPytorchDataset

    train = dm.train_dataloader()
    assert isinstance(train, StatefulDataLoader)
    assert isinstance(dm.train_dataset, MultitaskBoundaryPytorchDataset)
    assert dm.train_dataset.split == train_split
    assert isinstance(next(iter(train)), MultitaskBoundaryBatch)

    val = dm.val_dataloader()
    assert isinstance(val, DataLoader)
    assert isinstance(dm.val_dataset, MultitaskBoundaryPytorchDataset)
    assert dm.val_dataset.split == tuning_split
    assert isinstance(next(iter(val)), MultitaskBoundaryBatch)


# --- 2: the evaluation side, in grid order -------------------------------------------------------


@pytest.mark.parametrize("which", ["test", "predict"])
def test_eval_loaders_yield_the_grid_sequentially(data_config, grid_dir, which):
    dm = _datamodule(data_config, eval_tasks_dir=grid_dir, max_windows=5)
    loader = dm.test_dataloader() if which == "test" else dm.predict_dataloader()
    dataset = dm.test_dataset if which == "test" else dm.predict_dataset

    assert isinstance(dataset, QuerySeqMultitaskEvalDataset)
    assert dataset.split == held_out_split
    assert dm.predict_dataset is dm.test_dataset  # the default predict split is held_out
    assert isinstance(loader.sampler, SequentialSampler)
    assert loader.batch_size == 2

    batches = list(loader)
    assert len(batches) == 3  # five rows, two per batch
    assert all(isinstance(b, MultitaskEvalBatch) for b in batches)
    labels = torch.cat([b.labels for b in batches]).tolist()
    assert labels == dataset.schema_df["answers"].list.last().to_list()
    assert labels == [r["answers"][-1] for r in HELD_OUT_ROWS]
    n_queries = torch.cat([b.n_queries for b in batches]).tolist()
    assert n_queries == [len(r["queries"]) for r in HELD_OUT_ROWS]


# --- 3: eval_config ------------------------------------------------------------------------------


def test_eval_config_replaces_only_the_labels_root(data_config, grid_dir, multitask_labels_dir):
    dm = _datamodule(data_config, eval_tasks_dir=grid_dir)
    a, b = dataclasses.asdict(dm.config), dataclasses.asdict(dm.eval_config)
    assert a.pop("task_labels_dir") == Path(multitask_labels_dir)
    assert b.pop("task_labels_dir") == Path(grid_dir)
    assert a == b
    # The training config is shared, never mutated.
    assert dm.config is data_config
    assert data_config.task_labels_dir == Path(multitask_labels_dir)
    assert dm.eval_config is dm.eval_config


# --- 4: no grid root -----------------------------------------------------------------------------


def test_no_grid_root_is_fine_for_fit_and_refused_for_eval(data_config):
    dm = _datamodule(data_config)
    assert dm.eval_tasks_dir is None
    assert isinstance(dm.train_dataloader(), StatefulDataLoader)
    assert isinstance(dm.val_dataloader(), DataLoader)

    for method in ("test_dataloader", "predict_dataloader"):
        with pytest.raises(ValueError, match="eval_tasks_dir") as err:
            getattr(dm, method)()
        assert str(err.value) == MISSING_EVAL_TASKS_DIR_MSG
    for attr in ("test_dataset", "predict_dataset", "eval_config"):
        with pytest.raises(ValueError, match="eval_tasks_dir"):
            getattr(dm, attr)


def test_construction_and_fit_loaders_never_touch_the_grid(data_config, tmp_path):
    """A grid root that does not exist yet must not break construction or fit; only test / predict read it,
    and they get the config's own directory error."""
    missing = tmp_path / "not-generated-yet"
    dm = _datamodule(data_config, eval_tasks_dir=missing)
    assert dm.eval_tasks_dir == missing
    assert isinstance(dm.train_dataloader(), StatefulDataLoader)
    assert isinstance(next(iter(dm.val_dataloader())), MultitaskBoundaryBatch)
    assert "eval_config" not in dm.__dict__ and "test_dataset" not in dm.__dict__
    with pytest.raises(FileNotFoundError, match="task_labels_dir"):
        _ = dm.test_dataset
    with pytest.raises(FileNotFoundError, match="task_labels_dir"):
        dm.predict_dataloader()


# --- 5: predict_split ----------------------------------------------------------------------------


def test_predict_split_selects_the_grid_split_and_train_is_rejected(data_config, grid_dir):
    assert train_split not in EVAL_SPLITS
    with pytest.raises(ValueError, match="predict_split must be one of"):
        _datamodule(data_config, eval_tasks_dir=grid_dir, predict_split=train_split)

    dm = _datamodule(data_config, eval_tasks_dir=grid_dir, predict_split=tuning_split)
    assert dm.predict_split == tuning_split
    assert dm.predict_dataset.split == tuning_split
    assert dm.predict_dataset.schema_df["subject_id"].to_list() == [TUNING_SUBJECT] * len(TUNING_ROWS)
    assert dm.test_dataset.split == held_out_split
    assert dm.test_dataset is not dm.predict_dataset

    loader = dm.predict_dataloader()
    assert isinstance(loader.sampler, SequentialSampler)
    labels = torch.cat([b.labels for b in loader]).tolist()
    assert labels == [r["answers"][-1] for r in TUNING_ROWS]


# --- 6: what reaches the adapter -----------------------------------------------------------------


def test_dataset_kwargs_and_max_windows_reach_the_eval_adapter(data_config, grid_dir):
    dm = _datamodule(data_config, eval_tasks_dir=grid_dir, max_windows=3, dataset_kwargs={})
    ds = dm.test_dataset
    assert ds.max_windows == 3 and ds.expected_vocab_size is None and ds.strip_delta_tokens is False

    # max_windows: the longest held-out row has three queries; a two-window budget is refused at build.
    with pytest.raises(ValueError, match="max_windows=2"):
        _ = _datamodule(data_config, eval_tasks_dir=grid_dir, max_windows=2).predict_dataset

    # expected_vocab_size: a grid code at or past the checkpoint's embedding width is refused.
    widest = int(ds._q_codes.max())
    fits = _datamodule(
        data_config, eval_tasks_dir=grid_dir, dataset_kwargs={"expected_vocab_size": widest + 1}
    )
    assert fits.test_dataset.expected_vocab_size == widest + 1
    too_narrow = _datamodule(
        data_config, eval_tasks_dir=grid_dir, dataset_kwargs={"expected_vocab_size": widest}
    )
    with pytest.raises(ValueError, match="outside the checkpoint's vocabulary"):
        _ = too_narrow.test_dataset

    # strip_delta_tokens: the RoPE-time toggle reaches the adapter as it reaches the training datasets.
    rope = _datamodule(data_config, eval_tasks_dir=grid_dir, dataset_kwargs={"strip_delta_tokens": True})
    assert rope.test_dataset.strip_delta_tokens is True


def test_hyperparameters_record_the_grid_settings(data_config, grid_dir):
    dm = _datamodule(data_config, eval_tasks_dir=grid_dir, max_windows=5, predict_split=tuning_split)
    hp = dm.hparams
    assert hp["eval_tasks_dir"] == str(grid_dir)
    assert hp["max_windows"] == 5
    assert hp["predict_split"] == tuning_split
    assert hp["dataset_kwargs"] == {"expected_vocab_size": data_config.vocab_size}
    assert hp["ontology_dir"] is None
    # The parent's entries survive the merge.
    assert hp["batch_size"] == 2
    assert hp["config"]["max_seq_len"] == 64
    assert dict(dm.hparams_initial) == dict(hp)
    assert _datamodule(data_config).hparams["eval_tasks_dir"] is None


# --- 6b: an ontology reaches both sides; only the evaluation adapter widens ---------------------------


@pytest.fixture(scope="module")
def cohort_ontology_dir(tensorized_cohort_dir: Path, tmp_path_factory) -> Path:
    """An ontology built from the fixture cohort's own ``codes.parquet`` (so leaf ids match its manifest)."""
    return write_cohort_ontology(tensorized_cohort_dir, tmp_path_factory.mktemp("cohort_ontology"))


def test_the_training_dataset_gets_the_ontology_but_is_still_checked_against_the_leaf_manifest(
    data_config, grid_dir, cohort_ontology_dir, multitask_labels_dir
):
    """``dataset_kwargs["ontology_dir"]`` reaches *both* sides; only the evaluation adapter widens.

    The ontology is lifted out of ``dataset_kwargs`` so it can be handed to the evaluation adapter
    explicitly - there it makes ancestor query / start / bound names resolvable and raises the width
    ceiling from ``V`` to ``V_ext`` - and then put back, because a labels directory sampled with
    ancestor-valued start / bound / conditioning codes can only be read through the ontology that
    numbered them (see
    :func:`test_an_ancestor_bearing_labels_dir_needs_the_ontology_in_dataset_kwargs`).  Against these
    leaf-only labels the training datasets are handed it and simply do not use it.

    What must **not** move is the training width.  The packed target bits are leaf-only in every
    ontology mode, so the manifest here records ``vocab_size == boundary_vocab_size == V`` and
    ``expected_vocab_size`` stays the cohort's ``V``.  ``V_ext`` is the model's *embedding* width and
    passing it here is the mistake this pins - it is refused, loudly, at dataset build.
    """
    from every_query.data.ontology import extend_code_map, extended_vocab_size, load_nodes

    v = data_config.vocab_size
    v_ext = extended_vocab_size(cohort_ontology_dir)
    assert v_ext > v, "the fixture cohort's hierarchical names must mint ancestor nodes"
    expected_kwargs = {"expected_vocab_size": v, "ontology_dir": str(cohort_ontology_dir)}
    dm = _datamodule(
        data_config, eval_tasks_dir=grid_dir, max_windows=5, dataset_kwargs=dict(expected_kwargs)
    )
    assert dm.ontology_dir == cohort_ontology_dir
    assert dm.dataset_kwargs == expected_kwargs, (
        "the ontology must be put back into dataset_kwargs after being lifted for the eval adapter"
    )
    assert dm.hparams["ontology_dir"] == str(cohort_ontology_dir)
    assert dm.hparams["dataset_kwargs"] == expected_kwargs

    # Training side: the ontology is actually handed to the training dataset's constructor.  Recorded
    # rather than inferred from ``dataset_kwargs``, since the leaf manifest makes it a no-op here and
    # nothing on the built dataset would show it had arrived.
    seen: list[dict] = []

    class _RecordingDataset(MultitaskBoundaryPytorchDataset):
        def __init__(self, cfg, **kwargs):
            seen.append(dict(kwargs))
            super().__init__(cfg, **kwargs)

    dm.data_class = _RecordingDataset
    train = dm.train_dataset
    assert seen == [{"split": train_split, **expected_kwargs}], (
        f"the training dataset was not built with the ontology: {seen}"
    )

    # ...and the leaf manifest it reads is accepted against the cohort width, unwidened.
    assert isinstance(train, MultitaskBoundaryPytorchDataset)
    assert (train.vocab_size, train.boundary_vocab_size) == (v, v)
    assert train.manifest["vocab_size"] == v and train.manifest["boundary_vocab_size"] == v
    assert train.ontology_mode == "none", "these labels were sampled leaf-only"
    batch = next(iter(dm.train_dataloader()))
    assert isinstance(batch, MultitaskBoundaryBatch) and batch.targets.shape[-1] == v
    # ...and would be rejected against V_ext, which is why train.py pins the cohort width explicitly.
    widened = _datamodule(
        data_config, dataset_kwargs={"expected_vocab_size": v_ext, "ontology_dir": str(cohort_ontology_dir)}
    )
    with pytest.raises(ValueError, match="vocab_size mismatch"):
        _ = widened.train_dataset

    # Evaluation side: V_ext is the ceiling and every ancestor name is addressable.
    assert dm.eval_vocab_size == v_ext
    ds = dm.test_dataset
    assert isinstance(ds, QuerySeqMultitaskEvalDataset)
    assert ds.expected_vocab_size == v_ext and ds.ontology_dir == str(cohort_ontology_dir)
    nodes = load_nodes(cohort_ontology_dir)
    ancestors = dict(zip(nodes["node_name"], nodes["token_id"], strict=True))
    ancestors = {n: int(i) for n, i in ancestors.items() if int(i) >= v}
    assert ancestors and all(ds.code_to_index[n] == i for n, i in ancestors.items())
    assert ds.code_to_index == extend_code_map(
        _datamodule(data_config, eval_tasks_dir=grid_dir).test_dataset.code_to_index, cohort_ontology_dir
    )
    # Without an ontology the ceiling is the cohort width and no ancestor name resolves.
    plain = _datamodule(data_config, eval_tasks_dir=grid_dir).test_dataset
    assert plain.expected_vocab_size == v and not (set(ancestors) & set(plain.code_to_index))


def test_evaluation_adapter_refuses_a_same_width_permuted_ontology(
    data_config, grid_dir, tensorized_cohort_dir, cohort_ontology_dir, tmp_path
):
    """Regression for the PR #32 review.  The adapter grafts the ontology's ancestor indices onto the cohort's
    code map, so the ontology's leaves must *be* this cohort's ``codes.parquet`` rows.  One built.

    from the same codes with two indices swapped has the cohort's ``V`` and the genuine ontology's ``V_ext``
    - every width check passes, and the datamodule's width view is unchanged - and is refused when the
    dataset is built, naming the renumbered codes.
    """
    from every_query.data.ontology import extended_vocab_size

    permuted = write_cohort_ontology(tensorized_cohort_dir, tmp_path / "permuted", swap=("HR", "TEMP"))
    v, v_ext = data_config.vocab_size, extended_vocab_size(cohort_ontology_dir)
    assert extended_vocab_size(permuted) == v_ext
    dm = _datamodule(
        data_config,
        eval_tasks_dir=grid_dir,
        max_windows=5,
        dataset_kwargs={"expected_vocab_size": v, "ontology_dir": str(permuted)},
    )
    assert dm.eval_vocab_size == v_ext, "widths alone cannot see the permutation"
    by_name = (
        r"different codes\.parquet.*2 code\(s\) sit at a different index "
        r"\('HR': ontology \d+ vs cohort \d+, 'TEMP'"
    )
    with pytest.raises(ValueError, match=by_name):
        _ = dm.test_dataset
    # The training side is handed the same permuted ontology, but these labels are leaf-only
    # (``ontology_mode: "none"``), so it never resolves a name through it and is unaffected.
    assert dm.train_dataset.vocab_size == v


def test_an_ancestor_bearing_labels_dir_needs_the_ontology_in_dataset_kwargs(
    data_config, multitask_labels_dir, cohort_ontology_dir, tmp_path
):
    """Labels sampled in an ontology mode are unreadable without the ontology that numbered them.

    This is the reason ``ontology_dir`` is put *back* into ``dataset_kwargs``: when the sampler drew
    ancestor-valued start / bound / conditioning codes, the ids in the sidecars reach into
    ``[V, V_ext)`` and only that ontology says which node each one is.  The dataset refuses such a
    labels directory outright rather than reading ancestor ids as leaves, so a datamodule that kept
    the ontology to itself could not build a training dataset at all.

    The manifest is rewritten by hand rather than resampled: the three keys patched here are exactly
    what the dataset's check reads, and the labels themselves stay leaf-valued, so a pass cannot come
    from the ancestor path being skipped for want of an ancestor id.
    """
    from every_query.data.ontology import closure_fingerprint, extended_vocab_size

    labels = tmp_path / "ancestor_labels"
    shutil.copytree(multitask_labels_dir, labels)
    v, v_ext = data_config.vocab_size, extended_vocab_size(cohort_ontology_dir)
    manifests = sorted(labels.rglob(MANIFEST_NAME))
    assert manifests, f"no manifest under {labels}; the sampler fixture changed shape"
    for fp in manifests:
        manifest = json.loads(fp.read_text())
        assert manifest["ontology_mode"] == "none", "fixture drift: the sampler run must be leaf-only"
        fp.write_text(
            json.dumps(
                {
                    **manifest,
                    "ontology_mode": "boundaries",
                    "ontology_fingerprint": closure_fingerprint(cohort_ontology_dir),
                    "boundary_vocab_size": v_ext,
                }
            )
        )
    config = dataclasses.replace(data_config, task_labels_dir=str(labels))

    without = _datamodule(config, dataset_kwargs={"expected_vocab_size": v})
    assert "ontology_dir" not in without.dataset_kwargs
    with pytest.raises(ValueError, match=r"ontology_mode 'boundaries'.*no ontology_dir was given"):
        _ = without.train_dataset

    # With it in ``dataset_kwargs`` the same directory loads: the ontology is forwarded, its ancestor
    # names are grafted onto the cohort's code map, and the targets stay leaf-wide.
    with_ontology = _datamodule(
        config, dataset_kwargs={"expected_vocab_size": v, "ontology_dir": str(cohort_ontology_dir)}
    )
    assert with_ontology.dataset_kwargs["ontology_dir"] == str(cohort_ontology_dir)
    train = with_ontology.train_dataset
    assert (train.vocab_size, train.boundary_vocab_size) == (v, v_ext)
    ancestors = {n: i for n, i in train.code_to_index.items() if i >= v}
    assert ancestors, "the ontology's ancestor nodes were not grafted onto the training code map"
    batch = next(iter(with_ontology.train_dataloader()))
    assert isinstance(batch, MultitaskBoundaryBatch) and batch.targets.shape[-1] == v


# --- 7: Hydra ------------------------------------------------------------------------------------


def _compose_demo(overrides: list[str]):
    with initialize_config_dir(config_dir=TRAIN_CONFIGS, version_base=None):
        return compose(config_name="_demo_train_conditional_multitask_ar", overrides=overrides)


def test_demo_train_config_instantiates_the_datamodule(
    tensorized_cohort_dir, multitask_labels_dir, data_config, grid_dir
):
    overrides = [
        "output_dir=/unused",
        f"datamodule.config.tensorized_cohort_dir={tensorized_cohort_dir}",
        f"datamodule.config.task_labels_dir={multitask_labels_dir}",
        f"lightning_module.model.config_overrides.vocab_size={data_config.vocab_size}",
        "lightning_module.model.config_overrides.max_position_embeddings=79",
    ]
    cfg = _compose_demo(overrides)
    assert "data_class" not in cfg.datamodule
    assert cfg.datamodule.eval_tasks_dir is None

    dm = instantiate(cfg.datamodule)
    assert isinstance(dm, ConditionalMultitaskDataModule)
    assert dm.eval_tasks_dir is None
    assert dm.max_windows == 5 == cfg.lightning_module.model.max_windows
    assert dm.predict_split == held_out_split
    assert dm.dataset_kwargs == {"strip_delta_tokens": False, "expected_vocab_size": data_config.vocab_size}
    assert dm.batch_size == 2
    assert isinstance(next(iter(dm.train_dataloader())), MultitaskBoundaryBatch)
    with pytest.raises(ValueError, match="eval_tasks_dir"):
        dm.test_dataloader()

    # The grid root is an ordinary override that lands in eval_config untouched.
    dm = instantiate(_compose_demo([*overrides, f"datamodule.eval_tasks_dir={grid_dir}"]).datamodule)
    assert dm.eval_config.task_labels_dir == grid_dir
    assert dm.eval_config.tensorized_cohort_dir == tensorized_cohort_dir
    assert isinstance(next(iter(dm.test_dataloader())), MultitaskEvalBatch)


def test_production_config_keeps_the_grid_root_out_of_the_run_dir_name():
    with initialize_config_dir(config_dir=TRAIN_CONFIGS, version_base=None):
        cfg = compose(config_name="conditional_multitask_ar_config", return_hydra_config=True)
    excluded = set(cfg.hydra.job.config.override_dirname.exclude_keys)
    assert {"datamodule.config.task_labels_dir", "datamodule.eval_tasks_dir"} <= excluded
    assert cfg.datamodule._target_.endswith("ConditionalMultitaskDataModule")
    assert "data_class" not in cfg.datamodule
    assert cfg.datamodule.eval_tasks_dir is None
    assert cfg.datamodule.max_windows == cfg.lightning_module.model.max_windows
