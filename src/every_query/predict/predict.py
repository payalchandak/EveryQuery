"""Inference-only entry point — ``EQ_predict``.

Takes a trained model run directory and a directory of task-query parquets
(:class:`TaskQuerySchema`), runs per-row inference, writes a
:class:`PredictionSchema`-conformant parquet.  No AUCs, no model selection, no
multi-model orchestration — all of that is ``evaluate/`` or ``paper_experiments/``
territory.

Implementation is a single pass through :class:`EveryQueryPytorchDataset`:

1. Validate ``tasks_dir`` (must be a directory of ``.parquet`` files; warn if more
   than one file is present) and point MTD's ``task_labels_dir`` at it directly.
   No rewrite, no tmpdir, no input-side type coercion — if the parquets are
   malformed the dataset will surface that, and the output gets a canonical dtype
   pass via ``PredictionSchema.align`` at write.
2. ``trainer.predict(dataloaders=D.test_dataloader())`` — the dataset handles mixed
   ``(query, duration_days)`` rows natively (``_seeded_getitem`` prepends the
   row's own query token; ``collate`` builds per-item tensors).
3. Build the output directly from ``D.test_dataset.schema_df`` + the predict
   probabilities.  Schema_df is guaranteed by MTD to preserve the input labels
   frame's order + length (see
   ``MEDSPytorchDataset.get_task_seq_bounds_and_labels`` docstring), so no
   separate re-read of the input parquets is needed.  The ``null``/``True``/
   ``False`` ``boolean_value`` from the input is reconstructed from the dataset's
   collapsed ``(censor, occurs)`` pair — inverse of the collapse in
   ``EveryQueryPytorchDataset.__init__``.

Scope note: inherited ``LabelSchema`` columns beyond ``boolean_value`` (e.g.
``integer_value``, ``float_value``, ``categorical_value``) are not preserved on
output — the dataset doesn't carry them through ``schema_df``.  If a caller needs
those pass-through, a future ``--carry-columns`` opt-in can re-read the input
parquets for them.
"""

import logging
import time
from importlib.resources import files
from pathlib import Path

import hydra
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from hydra.utils import instantiate
from meds import held_out_split, tuning_split
from omegaconf import DictConfig

from every_query.data.schema import TaskQuerySchema
from every_query.predict.schema import PredictionSchema
from every_query.utils.model_loader import setup_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CONFIGS = str(files("every_query") / "predict" / "configs")


def _validate_tasks_dir(tasks_dir: Path) -> None:
    """Validate ``tasks_dir`` before handing it to MTD.

    Raises ``NotADirectoryError`` if it isn't a directory, ``ValueError`` if it
    contains any non-parquet files or is empty, and warns if more than one parquet
    is present — the typical inference use case is a single flat parquet; multiple
    files usually indicates the caller pointed us at the training task-labels
    directory by mistake.

    Examples:
        A directory with a single parquet file is the happy path:

        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     d = Path(tmpdir)
        ...     (d / "tasks.parquet").touch()
        ...     _validate_tasks_dir(d)  # no output, no error

        Non-directory argument raises ``NotADirectoryError``:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     f = Path(tmpdir) / "tasks.parquet"
        ...     f.touch()
        ...     _validate_tasks_dir(f)
        Traceback (most recent call last):
            ...
        NotADirectoryError: tasks_dir must be a directory, got .../tasks.parquet

        Empty directory raises ``ValueError``:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     _validate_tasks_dir(Path(tmpdir))
        Traceback (most recent call last):
            ...
        ValueError: tasks_dir ... contains no parquet files

        Any non-parquet file in the tree raises ``ValueError``:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     d = Path(tmpdir)
        ...     (d / "tasks.parquet").touch()
        ...     (d / "README.md").touch()
        ...     _validate_tasks_dir(d)
        Traceback (most recent call last):
            ...
        ValueError: tasks_dir ... contains non-parquet files: ['README.md']...

        Multiple parquet files pass but log a warning (caller can inspect logs):

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     d = Path(tmpdir)
        ...     (d / "a.parquet").touch()
        ...     (d / "b.parquet").touch()
        ...     _validate_tasks_dir(d)  # succeeds, warning is logged not raised
    """
    if not tasks_dir.is_dir():
        raise NotADirectoryError(f"tasks_dir must be a directory, got {tasks_dir}")

    all_files = sorted(p for p in tasks_dir.rglob("*") if p.is_file())
    non_parquet = [p for p in all_files if p.suffix != ".parquet"]
    if non_parquet:
        raise ValueError(
            f"tasks_dir {tasks_dir} contains non-parquet files: "
            f"{[str(p.relative_to(tasks_dir)) for p in non_parquet[:5]]}"
            f"{'...' if len(non_parquet) > 5 else ''}.  Point EQ_predict at a directory "
            f"containing only TaskQuerySchema-conformant parquet files."
        )
    parquets = [p for p in all_files if p.suffix == ".parquet"]
    if not parquets:
        raise ValueError(f"tasks_dir {tasks_dir} contains no parquet files")
    if len(parquets) > 1:
        logger.warning(
            f"tasks_dir {tasks_dir} contains {len(parquets)} parquet files; all will be "
            f"concatenated.  The typical inference use case is a single flat parquet."
        )


def _check_vocab(task_codes: set[str], train_cfg: DictConfig) -> None:
    """Raise if any task-query codes are missing from the trained model's vocabulary.

    Out-of-vocab query codes survive the predict loop (``encode_query`` silently falls
    back to ``PAD_INDEX``) but produce effectively-uniform (garbage) predictions — so
    the caller's ``predictions.parquet`` would silently contain rows whose probabilities
    have no relationship to the model.  Raise at startup rather than write misleading
    output.  Missing metadata is also a hard error: without the training vocab we can't
    validate inputs at all.

    Examples:
        Happy path — every task code is present in the training vocab:

        >>> import tempfile
        >>> from omegaconf import OmegaConf
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     meta_dir = Path(tmpdir) / "metadata"
        ...     meta_dir.mkdir()
        ...     pl.DataFrame({"code": ["A", "B", "C"]}).write_parquet(meta_dir / "codes.parquet")
        ...     cfg = OmegaConf.create({"datamodule": {"config": {"tensorized_cohort_dir": tmpdir}}})
        ...     _check_vocab({"A", "B"}, cfg)  # no raise

        Out-of-vocab codes raise ``ValueError``:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     meta_dir = Path(tmpdir) / "metadata"
        ...     meta_dir.mkdir()
        ...     pl.DataFrame({"code": ["A"]}).write_parquet(meta_dir / "codes.parquet")
        ...     cfg = OmegaConf.create({"datamodule": {"config": {"tensorized_cohort_dir": tmpdir}}})
        ...     try:
        ...         _check_vocab({"A", "MISSING"}, cfg)
        ...     except ValueError as e:
        ...         print(str(e).split(".  ")[0])
        1 of 2 task-query codes are not in the model's training vocabulary

        Missing metadata file also raises — we can't run inference without being
        able to confirm the inputs match the model's training vocab:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     cfg = OmegaConf.create({"datamodule": {"config": {"tensorized_cohort_dir": tmpdir}}})
        ...     try:
        ...         _check_vocab({"A"}, cfg)
        ...     except FileNotFoundError as e:
        ...         print("FileNotFoundError")
        FileNotFoundError
    """
    metadata_fp = Path(train_cfg.datamodule.config.tensorized_cohort_dir) / "metadata" / "codes.parquet"
    if not metadata_fp.is_file():
        raise FileNotFoundError(
            f"Cannot resolve training vocabulary — no codes metadata at {metadata_fp}.  "
            f"EQ_predict needs this to validate task codes against the model's vocab."
        )
    training_vocab = set(pl.read_parquet(metadata_fp, columns=["code"])["code"].to_list())
    missing = task_codes - training_vocab
    if missing:
        raise ValueError(
            f"{len(missing)} of {len(task_codes)} task-query codes are not in the model's training "
            f"vocabulary.  Out-of-vocab codes would be PAD-encoded and produce near-uniform "
            f"probabilities; refuse rather than write misleading predictions.  Missing codes: "
            f"{sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}"
        )


def _gather_probabilities(pred_batches: list[dict[str, torch.Tensor]]) -> pl.DataFrame:
    """Flatten Lightning's per-batch ``predict_step`` dicts into a two-column probabilities frame.

    Returns ``(censor_prob, occurs_prob)`` in dataset iteration order.  Column names
    come from :class:`PredictionSchema` so a rename on the schema flows through.

    ``logits_to_probs`` does a trailing ``.squeeze()``, so a single-item batch emits a
    0-d scalar tensor that ``torch.cat`` refuses to stack — ``reshape(-1)`` on every
    per-batch tensor makes the concat well-defined regardless of batch size.
    """
    if not pred_batches:
        return pl.DataFrame(
            schema={
                PredictionSchema.censor_prob_name: pl.Float32,
                PredictionSchema.occurs_prob_name: pl.Float32,
            }
        )

    def cat(key: str) -> pl.Series:
        # ``PredictionSchema.{censor,occurs}_prob`` are ``pa.float32()`` — cast here
        # so the final ``PredictionSchema.align`` doesn't have to coerce f64 → f32.
        return pl.Series(torch.cat([b[key].reshape(-1) for b in pred_batches]).numpy()).cast(pl.Float32)

    return pl.DataFrame(
        {
            PredictionSchema.censor_prob_name: cat("censor_probs"),
            PredictionSchema.occurs_prob_name: cat("occurs_probs"),
        }
    )


_EMBEDDING_COLUMN = "embedding"


def _gather_embeddings(pred_batches: list[dict[str, torch.Tensor]], hidden_size: int) -> pl.DataFrame:
    """Flatten Lightning's per-batch ``predict_step`` ``query_embed`` tensors into a one-column frame.

    Mirrors :func:`_gather_probabilities`: concatenates each batch's
    ``query_embed`` along dim 0 and returns a single-column polars frame whose
    ``embedding`` column is backed by an arrow ``fixed_size_list<float32>[hidden_size]``.
    Going through ``pa.FixedSizeListArray.from_arrays`` and ``pl.from_arrow``
    preserves the fixed-width invariant onto the eventual parquet write.

    ``hidden_size`` is required so the empty-batch case (degenerate cohort that
    produces zero predictions) still emits a fixed-size-list-typed column on
    disk — falling back to a variable-length ``list<float32>`` would silently
    diverge from the documented sidecar schema.

    Examples:
        Two batches of three rows each with hidden_size=4:

        >>> b1 = {"query_embed": torch.arange(12, dtype=torch.float32).reshape(3, 4)}
        >>> b2 = {"query_embed": torch.arange(12, 20, dtype=torch.float32).reshape(2, 4)}
        >>> df = _gather_embeddings([b1, b2], hidden_size=4)
        >>> df.height
        5
        >>> df.columns
        ['embedding']
        >>> df.to_arrow().schema.field("embedding").type
        FixedSizeListType(fixed_size_list<item: float>[4])

        Empty input still produces a fixed-size-list-typed frame so the
        on-disk sidecar schema matches the documented contract regardless of
        cohort size:

        >>> empty = _gather_embeddings([], hidden_size=4)
        >>> empty.height
        0
        >>> empty.columns
        ['embedding']
        >>> empty.to_arrow().schema.field("embedding").type
        FixedSizeListType(fixed_size_list<item: float>[4])
    """
    if pred_batches:
        embeddings = torch.cat([b["query_embed"] for b in pred_batches], dim=0)
        flat = pa.array(embeddings.reshape(-1).numpy().astype("float32"), type=pa.float32())
    else:
        flat = pa.array([], type=pa.float32())
    fixed = pa.FixedSizeListArray.from_arrays(flat, hidden_size)
    return pl.from_arrow(pa.table({_EMBEDDING_COLUMN: fixed}))


def _identifiers_from_schema_df(schema_df: pl.DataFrame) -> pl.DataFrame:
    """Build an output identifier frame from the dataset's post-``__init__`` schema_df.

    Reconstructs the original nullable ``boolean_value`` from the collapsed
    ``(censor, occurs)`` pair (inverse of the collapse in
    ``EveryQueryPytorchDataset.__init__``) so the output matches the
    ``TaskQuerySchema`` ``null``/``True``/``False`` convention.
    """
    out = schema_df.select(
        TaskQuerySchema.subject_id_name,
        TaskQuerySchema.prediction_time_name,
        TaskQuerySchema.query_name,
        TaskQuerySchema.duration_days_name,
    )
    # The dataset's collapse only runs when the input parquet had ``boolean_value``;
    # otherwise the output won't carry it either (``PredictionSchema`` treats
    # ``boolean_value`` as Optional).
    if "boolean_value" in schema_df.columns and "occurs" in schema_df.columns:
        # ``schema_df["boolean_value"]`` is the censor indicator; flip back:
        #   censor=True   → null   (ground truth unobserved)
        #   censor=False  → occurs (the true outcome)
        out = out.with_columns(
            pl.when(schema_df["boolean_value"])
            .then(pl.lit(None, dtype=pl.Boolean))
            .otherwise(schema_df["occurs"])
            .alias(TaskQuerySchema.boolean_value_name)
        )
    return out


def _derive_embeddings_path(output_parquet: Path) -> Path:
    """Derive the embeddings sibling path from ``output_parquet``.

    Inserts ``.embeddings`` immediately before the trailing suffix so the
    sibling lives next to the predictions file and is easy to glob.

    Examples:
        Standard ``.parquet`` predictions path:

        >>> _derive_embeddings_path(Path("/tmp/predictions.parquet"))
        PosixPath('/tmp/predictions.embeddings.parquet')

        Nested directories are preserved:

        >>> _derive_embeddings_path(Path("/a/b/c/run42.parquet"))
        PosixPath('/a/b/c/run42.embeddings.parquet')

        Non-``.parquet`` suffix is preserved (defensive — the EQ_predict input
        is always ``.parquet`` in practice, but keep the rule generic):

        >>> _derive_embeddings_path(Path("predictions.pq"))
        PosixPath('predictions.embeddings.pq')
    """
    return output_parquet.with_suffix(".embeddings" + output_parquet.suffix)


def _derive_timing_path(output_parquet: Path) -> Path:
    """Derive the timing sibling path from ``output_parquet``.

    Mirrors :func:`_derive_embeddings_path` so the timing artifact lives next
    to the predictions file and is easy to glob.

    Examples:
        >>> _derive_timing_path(Path("/tmp/predictions.parquet"))
        PosixPath('/tmp/predictions.timing.parquet')
    """
    return output_parquet.with_suffix(".timing" + output_parquet.suffix)


_SPLIT_TO_DATAMODULE_ATTRS: dict[str, tuple[str, str]] = {
    tuning_split: ("val_dataset", "val_dataloader"),
    held_out_split: ("test_dataset", "test_dataloader"),
}


@hydra.main(version_base="1.3", config_path=CONFIGS, config_name="predict")
def main(cfg: DictConfig) -> None:
    """Run inference and write :class:`PredictionSchema`-conformant output."""
    model_run_dir = Path(cfg.model_run_dir)
    tasks_dir = Path(cfg.tasks_dir)
    output_parquet = Path(cfg.output_parquet)
    ckpt_name = cfg.get("ckpt_name")
    split = cfg.get("split", held_out_split)
    overwrite = bool(cfg.get("overwrite", False))
    save_embeddings = bool(cfg.get("save_embeddings", False))
    embeddings_output_parquet = _derive_embeddings_path(output_parquet) if save_embeddings else None
    timing_output_parquet = _derive_timing_path(output_parquet)

    if split not in _SPLIT_TO_DATAMODULE_ATTRS:
        raise ValueError(
            f"split must be one of {sorted(_SPLIT_TO_DATAMODULE_ATTRS)}, got {split!r}.  "
            f"The 'train' split is disallowed because MTD's train_dataloader shuffles, "
            f"which would break the order-preserving hstack."
        )

    if output_parquet.exists() and not overwrite:
        raise FileExistsError(
            f"output_parquet {output_parquet} already exists.  Pass overwrite=true to replace, "
            f"or point at a new path — EQ_predict refuses to silently clobber existing output."
        )

    if embeddings_output_parquet is not None and embeddings_output_parquet.exists() and not overwrite:
        raise FileExistsError(
            f"embeddings sibling {embeddings_output_parquet} already exists.  Pass overwrite=true to "
            f"replace, or point output_parquet at a new path — EQ_predict refuses to silently clobber "
            f"existing output."
        )

    if timing_output_parquet.exists() and not overwrite:
        raise FileExistsError(
            f"timing sibling {timing_output_parquet} already exists.  Pass overwrite=true to "
            f"replace, or point output_parquet at a new path — EQ_predict refuses to silently clobber "
            f"existing output."
        )

    _validate_tasks_dir(tasks_dir)
    logger.info(f"Loading tasks from {tasks_dir} (split={split})")

    train_cfg, model, trainer = setup_model(model_run_dir, ckpt_name=ckpt_name)
    batch_size_override = cfg.get("batch_size")
    if batch_size_override is not None:
        train_cfg.datamodule.batch_size = int(batch_size_override)
    train_cfg.datamodule.config.task_labels_dir = str(tasks_dir)
    D = instantiate(train_cfg.datamodule)

    dataset_attr, dataloader_attr = _SPLIT_TO_DATAMODULE_ATTRS[split]
    dataset = getattr(D, dataset_attr)
    dataloader = getattr(D, dataloader_attr)()

    # Enforce order-preserving iteration — the ``hstack`` below assumes the dataloader
    # yields rows in ``schema_df`` order.  MTD's val/test dataloaders use
    # ``SequentialSampler`` by default, but asserting here turns a silent correctness
    # bug (shuffled probs stitched to the wrong identifiers) into a loud startup error
    # if that ever changes.
    sampler = getattr(dataloader, "sampler", None)
    sampler_cls = type(sampler).__name__ if sampler is not None else None
    if sampler_cls != "SequentialSampler":
        raise RuntimeError(
            f"{dataloader_attr} must use SequentialSampler to preserve row order for the "
            f"predictions hstack; got {sampler_cls!r}.  Re-check the MTD datamodule config."
        )

    _check_vocab(set(dataset.query.unique().to_list()), train_cfg)
    logger.info(f"Loaded {len(dataset)} tasks across {dataset.query.n_unique()} query codes")

    # Pass the split's dataloader directly — MTD's ``Datamodule`` has no
    # ``predict_dataloader``, so ``trainer.predict(datamodule=D)`` would hit the base
    # class's ``MisconfigurationException``.  The SequentialSampler check above
    # guarantees order preservation.
    t0 = time.perf_counter()
    pred_batches = trainer.predict(model=model, dataloaders=dataloader, ckpt_path=None)
    total_seconds = time.perf_counter() - t0

    probs = _gather_probabilities(pred_batches)
    identifiers = _identifiers_from_schema_df(dataset.schema_df)

    # MTD guarantees ``schema_df`` preserves the input labels frame's length + order
    # (see ``get_task_seq_bounds_and_labels`` docstring), and the dataloader is
    # ``shuffle=False`` — so ``probs`` comes back 1:1 matched with ``identifiers``.
    # A row-count mismatch would indicate a silent invariant violation; fail loudly.
    if probs.height != identifiers.height:
        raise RuntimeError(
            f"Prediction row count ({probs.height}) doesn't match dataset row count "
            f"({identifiers.height}).  This is an MTD invariant violation — "
            f"the dataloader should have yielded one prediction per schema_df row."
        )

    out = identifiers.hstack(probs)

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    # Final canonicalization pass — ``align`` casts / reorders columns to match the
    # schema's canonical arrow layout, and we write the aligned arrow table directly
    # via pyarrow so the schema-coerced dtypes actually land on disk (a polars
    # round-trip would re-infer types from the aligned data, which defeats the point).
    aligned = PredictionSchema.align(out.to_arrow())
    pq.write_table(aligned, output_parquet)
    logger.info(f"Wrote {out.height} predictions to {output_parquet}")

    n_rows = dataset.schema_df.height
    n_tasks = dataset.schema_df.select(["query", "duration_days"]).unique().height
    timing_table = pa.table(
        {
            "total_seconds": pa.array([total_seconds], type=pa.float64()),
            "n_rows": pa.array([n_rows], type=pa.int64()),
            "n_tasks": pa.array([n_tasks], type=pa.int64()),
            "seconds_per_row": pa.array([total_seconds / n_rows if n_rows else 0.0], type=pa.float64()),
            "seconds_per_task": pa.array([total_seconds / n_tasks if n_tasks else 0.0], type=pa.float64()),
        }
    )
    timing_output_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(timing_table, timing_output_parquet)
    logger.info(f"Wrote timing ({total_seconds:.2f}s, {n_tasks} tasks) to {timing_output_parquet}")

    if embeddings_output_parquet is not None:
        # ``hidden_size`` comes from the model's own config so the sidecar's
        # ``fixed_size_list<float32>[hidden_size]`` schema is well-defined even
        # for a degenerate empty cohort that produces zero predict batches.
        # The model object is already owned here (returned by ``setup_model``
        # and passed into ``trainer.predict``), so reading the property is not
        # additional coupling.
        hidden_size = model.model.HF_model.config.hidden_size
        embeddings_out = identifiers.hstack(_gather_embeddings(pred_batches, hidden_size))
        embeddings_output_parquet.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(embeddings_out.to_arrow(), embeddings_output_parquet)
        logger.info(f"Wrote {embeddings_out.height} embeddings to {embeddings_output_parquet}")


if __name__ == "__main__":
    main()
