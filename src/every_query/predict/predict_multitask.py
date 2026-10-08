"""Inference over a ``QuerySeqSchema`` evaluation grid with the multitask model — ``EQ_predict_multitask``.

The evaluation flow is::

    EQ_generate_evaluation_query_sequences -> QuerySeqSchema eval grid -> EQ_predict_multitask

Takes a trained :class:`~every_query.model.conditional_multitask_ar_model.ConditionalMultitaskARModel`
run directory and the ``eval/`` root written by ``EQ_generate_evaluation_query_sequences``
(including the explicit window starts only this model can read),
and writes **one scalar prediction per grid row**: the probability that the row's *final* query
occurs in its window, conditioned on the patient and on the earlier queries with their true answers
(or, where the grid's designed spec dictates one, with that ``forced_answers`` value instead).

Per real grid row the :class:`~every_query.data.multitask_eval_dataset.QuerySeqMultitaskEvalDataset`
adapter maps::

    q_durations        <- durations
    q_bound_codes      <- bound_events        (no-bound index where null / absent)
    q_start_durations  <- start_durations     (0.0 where absent)
    q_start_codes      <- start_events        (no-bound index where null / absent)
    condition_codes    <- queries[:-1]
    condition_answers  <- answers[:-1]        (forced_answers[:-1] wherever it is non-null)
    scored_code        <- queries[-1]
    label              <- answers[-1]

``[:-1]`` / ``[-1]`` are the row's own real queries, never the padded batch width, and the final
query is not teacher-forced into its own prediction.  The model scores only that code at only that
window (``ConditionalMultitaskARModel.score_final_query``): the same hidden state, tied embedding row
and bias the training forward uses, without ever materializing ``(B, K, V)`` logits or dense targets.

Nothing here needs, reads or writes a ``.labels.npy`` sidecar, a multitask manifest, ``eval_meta``
or ``eval_tasks.parquet``.  The prediction datamodule
(:class:`~every_query.data.conditional_multitask_datamodule.ConditionalMultitaskDataModule`) is
built from the checkpoint's cohort / sequence settings (``tensorized_cohort_dir``, ``max_seq_len``,
``seq_sampling_strategy``, ...) as recorded in its ``resolved_config.yaml``, with only the label
root swapped for the grid's ``eval/`` root; the training labels are never read, and the run
directory's own datamodule node is never instantiated (its ``config`` names the training labels
root, which ``MEDSTorchDataConfig`` requires to exist and which need not be on the inference
machine; older nodes also have no predict loader).  The Lightning predict loop owns device placement
and batching: ``Trainer.predict`` moves each batch to the accelerator and calls
:meth:`~every_query.model.conditional_multitask_lightning.ConditionalMultitaskLightningModule.predict_step`,
which scores through ``score_final_query`` and never touches the dense forward or ``batch.targets``.

Prediction is **single-device, single-process** by construction.  Row ``i`` of the output is grid
row ``i`` because the loader is sequential and the per-batch outputs are concatenated in loader
order; a multi-device or distributed strategy would shard and interleave rows across ranks and break
that alignment, so the inference trainer is pinned to one device (``device`` on the command line
picks which) and refuses anything else (issue #30 non-goal).  The guard also looks past the trainer
at the launcher: ``torchrun`` / ``srun --ntasks>1`` start several copies of this script, each of
which would score the whole grid and write the same ``output_parquet``, and Lightning's
single-device strategy cannot see them (:func:`launcher_world_size`).  Alignment is then verified
on the way out: the collated final-query labels and scored code indices are compared, row by row,
against ``answers[-1]`` / ``queries[-1]`` read back from the grid itself.

Output columns, one row per input sequence, in dataset (= dataloader = input) order::

    subject_id, prediction_time,
    queries, start_durations, start_events, durations, bound_events, answers, forced_answers,
    target_code, label, prob

The complete lists identify the conditional task (``target_code == queries[-1]``,
``label == answers[-1]``).  ``answers`` is always the labeled truth, never a forced value.
``start_durations`` / ``start_events`` / ``bound_events`` / ``forced_answers`` are always written,
normalized to their defaults (``0.0`` / null / null / null) when the grid lacked the column.  The
old evaluation-only ``task_id``, ``task_group``, ``start_resolved``, ``end_resolved`` and
``window_days`` sidecar fields are intentionally not recreated.
"""

from __future__ import annotations

import logging
import os
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import numpy as np
import polars as pl
import torch
from hydra.utils import instantiate
from lightning.fabric.plugins.environments import TorchElasticEnvironment
from lightning.pytorch import Trainer
from lightning.pytorch.strategies import SingleDeviceStrategy
from meds import held_out_split
from omegaconf import DictConfig  # noqa: TC002 - Hydra resolves this at runtime

from every_query.data.conditional_multitask_datamodule import EVAL_SPLITS, ConditionalMultitaskDataModule
from every_query.data.query_seq_dataset import (
    ANSWERS_COL,
    BOUND_EVENTS_COL,
    DURATIONS_COL,
    FORCED_ANSWERS_COL,
    QUERIES_COL,
    START_DURATIONS_COL,
    START_EVENTS_COL,
)
from every_query.model.conditional_multitask_lightning import ConditionalMultitaskLightningModule
from every_query.utils.model_loader import setup_model

if TYPE_CHECKING:
    from every_query.data.multitask_eval_dataset import QuerySeqMultitaskEvalDataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CONFIGS = str(files("every_query") / "predict" / "configs")

OUTPUT_COLUMNS = [
    "subject_id",
    "prediction_time",
    QUERIES_COL,
    START_DURATIONS_COL,
    START_EVENTS_COL,
    DURATIONS_COL,
    BOUND_EVENTS_COL,
    ANSWERS_COL,
    FORCED_ANSWERS_COL,
    "target_code",
    "label",
    "prob",
]


# --- data ------------------------------------------------------------------------------------------


def build_predict_datamodule(
    train_cfg: DictConfig,
    tasks_dir: Path,
    split: str,
    *,
    expected_vocab_size: int,
    use_rope_time: bool,
    max_windows: int | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    base_vocab_size: int | None = None,
    ontology_dir: str | Path | None = None,
    cohort_vocab_fingerprint: str | None = None,
) -> ConditionalMultitaskDataModule:
    """The prediction datamodule: the checkpoint's cohort settings, pointed at the grid.

    Reads ``train_cfg.datamodule`` - the ``datamodule`` node of the run's ``resolved_config.yaml`` -
    for its ``config`` (a ``MEDSTorchDataConfig``), ``dataset_kwargs``, ``batch_size``,
    ``num_workers`` and ``pin_memory``.  Only those keys are read, never the node's ``_target_``, so
    a run trained before issue #30 (whose node names ``ResumableDatamodule`` + a ``data_class``) and
    a current one (``ConditionalMultitaskDataModule`` with ``eval_tasks_dir: null``) are handled
    alike; the node itself is deliberately not instantiated (its ``config`` would require the training
    labels root to exist on this machine, and the former has no predict loader at all).

    ``config`` is instantiated with its ``task_labels_dir`` replaced by ``tasks_dir``: the training
    labels root recorded in the checkpoint need not exist on the machine running inference, and
    nothing here reads it.  Everything else - ``tensorized_cohort_dir``, ``max_seq_len``,
    ``seq_sampling_strategy``, ``static_inclusion_mode``, ``batch_mode`` - is the checkpoint's.

    Four consistency checks guard the checkpoint against the cohort and the ontology.
    ``strip_delta_tokens`` (from ``dataset_kwargs``) must agree with the model's ``use_rope_time``:
    a mismatch would feed a RoPE-time model token-index positions (or vice versa) and score garbage
    without an error.  The cohort's vocabulary width must equal the model's **cohort** width
    (``base_vocab_size``; ``train.py`` sizes the model from the cohort), so a checkpoint pointed at
    a different cohort fails here rather than scoring codes through the wrong embedding rows.  With
    an ontology, the ontology's extended width must equal the model's tied-embedding width
    (``expected_vocab_size``), so the two widths are checked separately and fail with distinct
    messages: the first names the cohort, the second the ontology.  And when the checkpoint recorded
    the training cohort's vocabulary fingerprint (``cohort_vocab_fingerprint``), the cohort on this
    machine must digest to it: a width cannot tell two cohorts apart, a fingerprint can.  (The
    ontology itself is checked against that fingerprint by the model at load, and against this
    cohort's ``codes.parquet`` row for row by the evaluation adapter.)

    Args:
        train_cfg: The OmegaConf of the run's ``resolved_config.yaml`` (``setup_model`` returns it).
        tasks_dir: The grid's ``eval/`` root.
        split: The grid split to score; one of ``EVAL_SPLITS`` (the datamodule refuses ``train``).
        expected_vocab_size: The model's tied-embedding width (``model.vocab_size``; ``V_ext`` under
            an ontology).  The evaluation adapter rejects grid codes at or past it.
        use_rope_time: The model's ``use_rope_time``.
        max_windows: The model's window budget, forwarded so an over-long grid row is rejected when
            the dataset is built rather than inside a forward pass.
        batch_size: Overrides the checkpoint's ``datamodule.batch_size`` when given.
        num_workers: Overrides the checkpoint's ``datamodule.num_workers`` when given.
        base_vocab_size: The model's cohort width (``model.base_vocab_size``), which the cohort on
            this machine must match.  Defaults to ``expected_vocab_size`` (no ontology).
        ontology_dir: The model's ``ontology_dir``; forwarded to the evaluation adapter so ancestor
            query / start / bound names resolve, and checked against ``expected_vocab_size``.
        cohort_vocab_fingerprint: The model's ``cohort_vocab_fingerprint`` (the training cohort's
            :func:`~every_query.utils.digest.vocab_fingerprint`), which this machine's cohort must
            digest to.  ``None`` (a checkpoint from before it was recorded) skips the check.
    """
    dm_cfg = train_cfg.datamodule
    dataset_kwargs = dm_cfg.get("dataset_kwargs") or {}
    strip_delta_tokens = bool(dataset_kwargs.get("strip_delta_tokens", False))
    if strip_delta_tokens != bool(use_rope_time):
        raise ValueError(
            f"the checkpoint's datamodule strips delta tokens={strip_delta_tokens} but its model has "
            f"use_rope_time={use_rope_time}; resolved_config.yaml is inconsistent."
        )
    if base_vocab_size is None:
        base_vocab_size = expected_vocab_size
    if ontology_dir is None and int(base_vocab_size) != int(expected_vocab_size):
        raise ValueError(
            f"the checkpoint's tied embedding table is {expected_vocab_size} wide but its cohort width is "
            f"{base_vocab_size} and it has no ontology to account for the difference."
        )
    if ontology_dir is not None:
        from every_query.data.ontology import extended_vocab_size

        v_ext = extended_vocab_size(ontology_dir)
        if v_ext != int(expected_vocab_size):
            raise ValueError(
                f"the ontology at {ontology_dir} extends the vocabulary to V_ext={v_ext} but the "
                f"checkpoint's tied embedding table is {expected_vocab_size} wide; the model was trained "
                "under a different ontology."
            )
    # ``task_labels_dir`` here is a placeholder that must merely exist: the checkpoint records the
    # TRAINING labels root, which need not be present on this machine, and ``__post_init__`` raises
    # FileNotFoundError on a missing one.  Nothing on this path reads it - the grid is reached through
    # the datamodule's ``eval_tasks_dir`` below - but pointing it at the grid (rather than ``None``)
    # keeps ``__post_init__``'s ``seq_sampling_strategy == to_end`` check firing here, where the error
    # names the checkpoint, instead of later inside ``eval_config``.
    data_cfg = instantiate(dm_cfg.config, task_labels_dir=str(tasks_dir))
    if int(data_cfg.vocab_size) != int(base_vocab_size):
        detail = (
            f"the checkpoint's tied embedding table is {expected_vocab_size} wide"
            if ontology_dir is None
            else f"the checkpoint was trained on a cohort of vocab_size={base_vocab_size} (its tied "
            f"embedding table is {expected_vocab_size} wide with the ontology's ancestor rows)"
        )
        raise ValueError(
            f"the cohort at {data_cfg.tensorized_cohort_dir} has vocab_size={data_cfg.vocab_size} but "
            f"{detail}; the model was trained on a different codes.parquet."
        )
    if cohort_vocab_fingerprint is not None:
        from every_query.data.ontology import cohort_code_map
        from every_query.utils.digest import vocab_fingerprint

        actual = vocab_fingerprint(cohort_code_map(data_cfg.code_metadata_fp))
        if actual != cohort_vocab_fingerprint:
            raise ValueError(
                f"the cohort at {data_cfg.tensorized_cohort_dir} has the checkpoint's vocab_size="
                f"{base_vocab_size} but its vocabulary digests to {actual[:12]}... where the checkpoint "
                f"recorded {cohort_vocab_fingerprint[:12]}...; the model was trained on a different "
                "codes.parquet (same width, other codes or a different numbering)."
            )
    if batch_size is None:
        batch_size = dm_cfg.batch_size
    if num_workers is None:
        num_workers = dm_cfg.get("num_workers", 0) or 0
    return ConditionalMultitaskDataModule(
        data_cfg,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        pin_memory=dm_cfg.get("pin_memory"),
        dataset_kwargs={
            "strip_delta_tokens": strip_delta_tokens,
            "expected_vocab_size": int(base_vocab_size),
            "ontology_dir": None if ontology_dir is None else str(ontology_dir),
        },
        eval_tasks_dir=tasks_dir,
        max_windows=max_windows,
        predict_split=split,
    )


def check_grid_ontology_provenance(tasks_dir: Path, split: str, ontology_dir: str | Path | None) -> None:
    """Refuse a grid whose provenance sidecars were labeled under a *different* closure than the model's.

    ``EQ_generate_evaluation_query_sequences`` writes, per output shard, a sidecar under
    ``{out_dir}_artifacts/_labeled/`` whose ``ontology_fingerprint`` is
    ``"{closure}|{universe}"`` (``None`` for a leaf-only run).  Only the closure half decides what an
    ancestor query's label means, so that is what is compared against
    :func:`~every_query.data.ontology.closure_fingerprint` of the checkpoint's ontology.  The
    combinations:

    - both present and equal: fine;
    - both present and different: **error** - the grid's ancestor labels do not mean what this
      model was trained to predict;
    - grid labeled without an ontology: fine whatever the model has (leaf labels are ontology
      independent), nothing to compare;
    - grid labeled with an ontology, model without: a warning only; any ancestor *name* in the grid
      is then rejected by the dataset as an unknown code, and a leaf-only grid labels identically;
    - no sidecar (a grid copied without its ``_artifacts`` sibling, or written by an older sampler):
      a warning; there is nothing to check against.

    Args:
        tasks_dir: The grid's ``eval/`` root (the sampler's ``out_dir / "eval"``).
        split: The split being scored.
        ontology_dir: The model's ``ontology_dir``, or ``None``.
    """
    # Lazy: the sampler module is the owner of the sidecar layout and its parsing.
    from every_query.generate_tasks.sample_evaluation_query_sequences import (
        _recorded_fingerprint,
        split_ontology_fingerprint,
    )

    model_closure = None
    if ontology_dir is not None:
        from every_query.data.ontology import closure_fingerprint

        model_closure = closure_fingerprint(ontology_dir)

    out_dir = Path(tasks_dir).parent
    # ``rglob``, not ``glob``: the dataset selects every shard with the split directory anywhere in
    # its parents, so a hand-merged grid with nested shards would otherwise be scored while this
    # gate silently inspected nothing.
    shards = sorted((Path(tasks_dir) / split).rglob("*.parquet"))
    without_sidecar: list[str] = []
    foreign: list[str] = []
    for fp in shards:
        recorded = _recorded_fingerprint(out_dir, fp)
        if recorded is None:
            without_sidecar.append(fp.name)
            continue
        grid_closure, _ = split_ontology_fingerprint(recorded.get("ontology_fingerprint"))
        if grid_closure is None:
            continue
        if model_closure is None:
            foreign.append(fp.name)
        elif grid_closure != model_closure:
            raise ValueError(
                f"the grid shard {fp} was labeled under an ontology whose closure ({grid_closure}) differs "
                f"from the checkpoint's ({model_closure}, {ontology_dir}); its ancestor labels do not mean "
                "what this model predicts. Regenerate the grid with the checkpoint's ontology_dir."
            )
    if without_sidecar:
        logger.warning(
            "%d grid shard(s) under %s carry no provenance sidecar (%s); the ontology they were labeled "
            "under cannot be checked against the checkpoint's.",
            len(without_sidecar),
            tasks_dir / split,
            without_sidecar[:3],
        )
    if foreign:
        logger.warning(
            "%d grid shard(s) under %s were labeled with an ontology but the checkpoint has none (%s); "
            "leaf labels are unaffected, and any ancestor query name is rejected as an unknown code.",
            len(foreign),
            tasks_dir / split,
            foreign[:3],
        )


def build_eval_dataset(
    train_cfg: DictConfig,
    tasks_dir: Path,
    split: str,
    *,
    expected_vocab_size: int,
    use_rope_time: bool,
    max_windows: int | None = None,
    base_vocab_size: int | None = None,
    ontology_dir: str | Path | None = None,
) -> QuerySeqMultitaskEvalDataset:
    """The prediction datamodule's ``predict_dataset``; see :func:`build_predict_datamodule`."""
    return build_predict_datamodule(
        train_cfg,
        tasks_dir,
        split,
        expected_vocab_size=expected_vocab_size,
        use_rope_time=use_rope_time,
        max_windows=max_windows,
        base_vocab_size=base_vocab_size,
        ontology_dir=ontology_dir,
    ).predict_dataset


# --- trainer ---------------------------------------------------------------------------------------


def resolve_accelerator(device: str | None) -> tuple[str, int | list[int]]:
    """Map the CLI ``device`` string onto Lightning's ``(accelerator, devices)`` for exactly one device.

    ``None`` lets Lightning pick (``"auto"``, one device: the first GPU when there is one, else the
    CPU).  A torch device string names the accelerator - ``cpu``, ``cuda`` (the default GPU),
    ``cuda:N`` (GPU ``N``) or ``mps`` - and always resolves to a single device.

    Examples:
        >>> resolve_accelerator(None)
        ('auto', 1)
        >>> resolve_accelerator("cpu"), resolve_accelerator("cuda"), resolve_accelerator("mps")
        (('cpu', 1), ('gpu', 1), ('mps', 1))
        >>> resolve_accelerator("cuda:2")
        ('gpu', [2])
        >>> resolve_accelerator("xpu")
        Traceback (most recent call last):
            ...
        ValueError: device must be null, cpu, cuda, cuda:N or mps; got 'xpu'
    """
    if device is None:
        return "auto", 1
    try:
        parsed = torch.device(str(device))
    except (RuntimeError, TypeError) as e:
        raise ValueError(f"device must be null, cpu, cuda, cuda:N or mps; got {device!r}") from e
    match parsed.type:
        case "cpu":
            return "cpu", 1
        case "cuda":
            return "gpu", (1 if parsed.index is None else [int(parsed.index)])
        case "mps":
            return "mps", 1
        case _:
            raise ValueError(f"device must be null, cpu, cuda, cuda:N or mps; got {device!r}")


def launcher_world_size() -> int:
    """How many processes the launcher that started this one created; ``1`` when it was started alone.

    ``devices=1`` always resolves to a ``SingleDeviceStrategy`` whose ``world_size`` is hard-wired to
    1, even when this process is one of several that ``torchrun`` or ``srun --ntasks>1`` started, so
    the trainer alone cannot tell.  Three witnesses are asked, in order: an already-initialized
    ``torch.distributed`` process group; ``torchrun`` (Lightning's own detector, checked before SLURM
    as Lightning does, since ``torchrun`` can run inside a SLURM job); and a SLURM job step, read
    from the task-level variables ``slurmstepd`` sets in every task (``SLURM_PROCID`` marks a step,
    ``SLURM_STEP_NUM_TASKS`` counts an ``srun`` step, ``SLURM_NTASKS`` a batch step).  An
    ``salloc`` shell that has not run ``srun`` sets neither ``SLURM_PROCID`` nor a step count, so a
    plain command there counts as one process.  Lightning's ``SLURMEnvironment`` is deliberately
    not used: its constructor rejects ``--ntasks=N`` without ``--ntasks-per-node`` with its own
    error, and its ``detect()`` skips allocations named ``bash`` / ``interactive`` - a hatch that
    would let ``srun -n 2`` from such a shell through.  (Only the SLURM variables, not a cluster,
    were available when this was written; validate on the cluster if it ever refuses a run you
    consider single-process.)
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_world_size())
    if TorchElasticEnvironment.detect():
        return int(os.environ.get("WORLD_SIZE", 1))
    if "SLURM_PROCID" in os.environ:
        return int(os.environ.get("SLURM_STEP_NUM_TASKS") or os.environ.get("SLURM_NTASKS") or 1)
    return 1


def check_single_process() -> None:
    """Refuse to run as one of several launcher processes (``torchrun`` / ``srun`` / ``sbatch --ntasks>1``).

    Called by ``main`` before the trainer is built and by :func:`check_single_device`: every such
    process would otherwise score the whole grid and write the same ``output_parquet``.

    Raises:
        ValueError: If :func:`launcher_world_size` is not 1.
    """
    launcher = launcher_world_size()
    if launcher != 1:
        raise ValueError(
            "EQ_predict_multitask prediction must run in exactly one process, but the launcher started "
            f"{launcher} process(es): each would score the whole grid and write the same output_parquet "
            "(issue #30 non-goal). Start exactly one process: no torchrun, and srun / sbatch with "
            "--ntasks=1."
        )


def check_single_device(trainer: Trainer) -> None:
    """Refuse a trainer that would run prediction on more than one device or in more than one process.

    Both the launcher (:func:`check_single_process`) and the trainer's shape (strategy, devices,
    nodes, world size) are checked: under ``torchrun`` / ``srun --ntasks>1`` every rank would pass
    the second alone.

    Raises:
        ValueError: If this process was started by a multi-process launcher, or the trainer's
            strategy is not a ``SingleDeviceStrategy``, or it spans more than one device, node or
            process.
    """
    check_single_process()
    single = (
        isinstance(trainer.strategy, SingleDeviceStrategy)
        and trainer.num_devices == 1
        and trainer.num_nodes == 1
        and trainer.world_size == 1
    )
    if not single:
        raise ValueError(
            "EQ_predict_multitask prediction must run on exactly one device: the inference trainer "
            f"resolved to strategy={type(trainer.strategy).__name__}, num_devices={trainer.num_devices}, "
            f"num_nodes={trainer.num_nodes}, world_size={trainer.world_size}. Outputs are concatenated in "
            "loader order and must stay row-aligned with dataset.schema_df; multi-device / distributed "
            "prediction would shard rows across ranks and is not supported (issue #30 non-goal). Pass "
            "device=cpu, device=cuda:N or device=mps (one device)."
        )


DEFAULT_PREDICT_PRECISION = "bf16-mixed"


def build_predict_trainer(
    device: str | None = None,
    *,
    precision: str = DEFAULT_PREDICT_PRECISION,
    enable_progress_bar: bool = True,
) -> Trainer:
    """A single-device inference ``Trainer``: no logger, no checkpointing, inference mode.

    This is not the training trainer of the run's ``resolved_config.yaml`` (callbacks, checkpoint
    directories, a logger); it is built from scratch on the device :func:`resolve_accelerator` maps
    ``device`` to.  The single-device guard (:func:`check_single_device`) lives in
    :func:`run_inference`, which applies it to whatever trainer it is handed.

    ``precision`` is a Lightning precision string and defaults to ``bf16-mixed``, the production
    training precision, so the backbone pass runs under the same autocast the checkpoint was trained
    with - and under the same numerics as ``EQ_predict``, which predicts
    through the trainer of the run's ``resolved_config.yaml`` and so inherits the ``bf16-mixed`` every
    shipped training config records; a cross-run comparison is therefore
    like for like (a run trained under another precision would need the matching override here).
    (The pre-#30 manual loop ran the model in fp32; ``bf16-mixed`` is a deliberate change.)  Pass
    ``32-true`` for full-precision scoring (e.g. to compare against a hand-computed
    ``score_final_query``); the two agree to bf16 rounding (~1e-2 on the probabilities), not exactly.
    """
    accelerator, devices = resolve_accelerator(device)
    return Trainer(
        accelerator=accelerator,
        devices=devices,
        num_nodes=1,
        strategy="auto",
        precision=precision,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=bool(enable_progress_bar),
        enable_model_summary=False,
        inference_mode=True,
    )


# --- inference -------------------------------------------------------------------------------------


def run_inference(
    model: ConditionalMultitaskLightningModule,
    datamodule: ConditionalMultitaskDataModule,
    trainer: Trainer,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``trainer.predict`` over the predict loader; ``(probs, labels, scored_codes)``, one per grid row.

    The Lightning predict loop moves each batch to the trainer's device and collects the
    ``predict_step`` dictionaries (``probs`` / ``labels`` / ``scored_codes``, CPU tensors of shape
    ``(B,)``), which are concatenated here in loader order into a float32, a bool and an int64
    array.  Each batch costs one backbone pass plus ``B`` dot products - nothing of shape
    ``(B, K, V)`` exists at any point.  The loader must yield exactly
    ``len(datamodule.predict_dataset)`` rows, or the output could not be row-aligned with the grid;
    :func:`predictions_to_df` then checks the labels and scored codes row by row against the grid.

    Raises:
        ValueError: If ``trainer`` is not single-device / single-process (:func:`check_single_device`).
        RuntimeError: If the loader yielded a different number of rows than the grid has.
    """
    check_single_device(trainer)
    n_rows = len(datamodule.predict_dataset)
    outputs = trainer.predict(model, datamodule=datamodule, return_predictions=True) or []
    seen = sum(int(out["probs"].shape[0]) for out in outputs)
    if seen != n_rows:
        raise RuntimeError(f"dataloader yielded {seen} prediction(s) but the grid has {n_rows} row(s)")
    if not outputs:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=bool), np.empty(0, dtype=np.int64)
    probs = torch.cat([out["probs"] for out in outputs]).float().cpu().numpy().astype(np.float32, copy=False)
    labels = torch.cat([out["labels"] for out in outputs]).cpu().numpy().astype(bool)
    scored_codes = torch.cat([out["scored_codes"] for out in outputs]).cpu().numpy().astype(np.int64)
    return probs, labels, scored_codes


def predictions_to_df(
    dataset: QuerySeqMultitaskEvalDataset, probs: np.ndarray, labels: np.ndarray, scored_codes: np.ndarray
) -> pl.DataFrame:
    """One output row per grid row, in dataset order, with the normalized window lists.

    ``labels`` and ``scored_codes`` (collated by the loader, carried through the predict loop) must
    equal ``answers[-1]`` and ``code_to_index[queries[-1]]`` read back from the dataset's own rows:
    the two sides are produced by different code paths, so disagreement means the loader's row order
    drifted from the dataset's, and the probabilities would be attached to the wrong rows.  The
    label alone would miss a swap of two same-label rows; the code index catches every swap of rows
    that score different codes.  (Two rows sharing both final label and final query remain
    interchangeable to these checks; nothing in the sequential single-process loader this CLI
    enforces can produce that swap.)
    """
    schema_df = dataset.schema_df
    n = schema_df.height
    if probs.shape != (n,) or labels.shape != (n,) or scored_codes.shape != (n,):
        raise RuntimeError(
            f"got {probs.shape[0]} prediction(s) / {labels.shape[0]} label(s) / {scored_codes.shape[0]} "
            f"scored code(s) for {n} grid row(s)"
        )
    queries = schema_df[QUERIES_COL]
    answers = schema_df[ANSWERS_COL]
    durations = schema_df[DURATIONS_COL]
    lengths = queries.list.len()
    if n and (lengths == 0).any():
        raise ValueError("every grid row needs at least one query")

    target_code = queries.list.last()
    label = answers.list.last()
    if n and not np.array_equal(label.to_numpy().astype(bool), labels):
        bad = int(np.flatnonzero(label.to_numpy().astype(bool) != labels)[0])
        raise RuntimeError(
            f"the collated final-query label disagrees with answers[-1] at grid row {bad}; the loader's "
            "row order does not match the dataset's."
        )
    expected_codes = np.array([dataset.code_to_index[c] for c in target_code.to_list()], dtype=np.int64)
    if n and not np.array_equal(expected_codes, scored_codes):
        bad = int(np.flatnonzero(expected_codes != scored_codes)[0])
        raise RuntimeError(
            f"the collated scored code disagrees with queries[-1] at grid row {bad} (index "
            f"{int(scored_codes[bad])} vs {int(expected_codes[bad])} for {target_code[bad]!r}); the "
            "loader's row order does not match the dataset's."
        )

    def default_list(fill, dtype: pl.DataType) -> pl.Series:
        return pl.Series([[fill] * int(k) for k in lengths.to_list()], dtype=pl.List(dtype))

    start_durations = schema_df[START_DURATIONS_COL] if dataset.has_starts else default_list(0.0, pl.Float32)
    start_events = schema_df[START_EVENTS_COL] if dataset.has_starts else default_list(None, pl.Utf8)
    bound_events = schema_df[BOUND_EVENTS_COL] if dataset.has_bound_events else default_list(None, pl.Utf8)
    # ``answers`` stays the labeled truth; this records which conditioning answers were dictated
    # instead, which is what tells a forced-YES row from a forced-NO row of the same query spec.
    forced_answers = (
        schema_df[FORCED_ANSWERS_COL] if dataset.has_forced_answers else default_list(None, pl.Boolean)
    )

    return pl.DataFrame(
        {
            "subject_id": schema_df["subject_id"],
            "prediction_time": schema_df["prediction_time"],
            QUERIES_COL: queries,
            START_DURATIONS_COL: start_durations.cast(pl.List(pl.Float32)).alias(START_DURATIONS_COL),
            START_EVENTS_COL: start_events.cast(pl.List(pl.Utf8)).alias(START_EVENTS_COL),
            DURATIONS_COL: durations.cast(pl.List(pl.Float32)),
            BOUND_EVENTS_COL: bound_events.cast(pl.List(pl.Utf8)).alias(BOUND_EVENTS_COL),
            ANSWERS_COL: answers,
            FORCED_ANSWERS_COL: forced_answers.cast(pl.List(pl.Boolean)).alias(FORCED_ANSWERS_COL),
            "target_code": target_code,
            "label": label.cast(pl.Boolean),
            "prob": pl.Series(probs, dtype=pl.Float32),
        }
    ).select(OUTPUT_COLUMNS)


@hydra.main(version_base="1.3", config_path=CONFIGS, config_name="predict_multitask")
def main(cfg: DictConfig) -> None:
    model_run_dir = Path(cfg.model_run_dir)
    tasks_dir = Path(cfg.tasks_dir)
    output_parquet = Path(cfg.output_parquet)
    split = cfg.get("split", held_out_split)
    overwrite = bool(cfg.get("overwrite", False))

    if split not in EVAL_SPLITS:
        raise ValueError(f"split must be one of {sorted(EVAL_SPLITS)}, got {split!r}.")
    if output_parquet.exists() and not overwrite:
        raise FileExistsError(f"output_parquet {output_parquet} already exists; pass overwrite=true.")
    if not tasks_dir.is_dir():
        raise NotADirectoryError(f"tasks_dir must be the grid's eval/ root, got {tasks_dir}")

    train_cfg, model, _ = setup_model(
        model_run_dir, ckpt_name=cfg.get("ckpt_name"), module_cls=ConditionalMultitaskLightningModule
    )

    # The widths and the ontology come from the loaded model itself (``vocab_size`` is the table,
    # ``base_vocab_size`` the cohort, both read from its ontology at construction), so a run dir
    # from before the multitask model knew about ontologies is handled exactly like a current one.
    check_grid_ontology_provenance(tasks_dir, split, model.model.ontology_dir)
    datamodule = build_predict_datamodule(
        train_cfg,
        tasks_dir,
        split,
        expected_vocab_size=model.model.vocab_size,
        use_rope_time=model.model.use_rope_time,
        max_windows=model.model.max_windows,
        batch_size=cfg.get("batch_size"),
        num_workers=cfg.get("num_workers"),
        base_vocab_size=model.model.base_vocab_size,
        ontology_dir=model.model.ontology_dir,
        cohort_vocab_fingerprint=model.model.cohort_vocab_fingerprint,
    )
    dataset = datamodule.predict_dataset
    logger.info(f"Loaded {len(dataset)} grid rows from {tasks_dir} (split={split})")

    # Before the trainer: Lightning's own SLURM detection can fail a multi-task launch with a less
    # useful message of its own while the Trainer is being built.
    check_single_process()
    trainer = build_predict_trainer(
        cfg.get("device"),
        precision=str(cfg.get("precision") or DEFAULT_PREDICT_PRECISION),
        enable_progress_bar=bool(cfg.get("enable_progress_bar", True)),
    )
    logger.info(f"Scoring the final query of {len(dataset)} rows on {trainer.strategy.root_device}")
    probs, labels, scored_codes = run_inference(model, datamodule, trainer)

    out = predictions_to_df(dataset, probs, labels, scored_codes)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(output_parquet)
    logger.info(f"Wrote {out.height} final-query predictions to {output_parquet}")


if __name__ == "__main__":
    main()
