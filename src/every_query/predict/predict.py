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
from importlib.resources import files
from pathlib import Path

import hydra
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from hydra.utils import instantiate
from lightning.pytorch.callbacks import BasePredictionWriter
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


class _StreamingPredictionWriter(BasePredictionWriter):
    """Lightning ``BasePredictionWriter`` that streams per-batch results to a single parquet.

    Each ``write_on_batch_end`` call appends one row group to ``output_parquet`` so
    EQ_predict's CPU-memory footprint doesn't scale with cohort size — only one batch's
    ``predict_step`` dict is alive at a time (Lightning drops it after the writer hook
    returns when ``return_predictions=False``).

    Identifiers come from slicing the dataset's pre-loaded ``schema_df`` with a running
    offset; correctness depends on the dataloader iterating in ``schema_df`` row order,
    which ``main()`` enforces via the ``SequentialSampler`` guard.

    Partial-write semantics: ``close()`` is idempotent and is invoked from ``main()``'s
    ``try/finally`` block, so a Python-level exception mid-stream still flushes the parquet
    footer and leaves a valid ``PredictionSchema`` parquet covering exactly the batches that
    completed.  Hard kills (SIGKILL) cannot be recovered — parquet has no
    append-after-close, so single-file streaming has no defense against them.
    """

    def __init__(self, output_parquet: Path, schema_df: pl.DataFrame) -> None:
        super().__init__(write_interval="batch")
        self._output_parquet = output_parquet
        self._schema_df = schema_df
        self._writer: pq.ParquetWriter | None = None
        self._offset: int = 0
        self._arrow_schema: pa.Schema | None = None

    def setup(self, trainer, pl_module, stage: str) -> None:
        if stage != "predict":
            return
        # Pre-build the canonical arrow schema from a zero-row aligned table.  Doing
        # this up front (rather than on the first batch) means an empty cohort still
        # produces a valid PredictionSchema parquet — matches the pre-streaming
        # behavior where ``_gather_probabilities`` returned an empty frame and the
        # final ``pq.write_table`` still ran.
        empty_ids = _identifiers_from_schema_df(self._schema_df.head(0))
        empty_probs = pl.DataFrame(
            schema={
                PredictionSchema.censor_prob_name: pl.Float32,
                PredictionSchema.occurs_prob_name: pl.Float32,
            }
        )
        empty_aligned = PredictionSchema.align(empty_ids.hstack(empty_probs).to_arrow())
        self._arrow_schema = empty_aligned.schema

        self._output_parquet.parent.mkdir(parents=True, exist_ok=True)
        self._writer = pq.ParquetWriter(self._output_parquet, schema=self._arrow_schema)

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction: dict[str, torch.Tensor],
        batch_indices,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self._writer is None:
            raise RuntimeError(
                "_StreamingPredictionWriter received a batch before setup() opened the "
                "parquet writer.  This shouldn't happen in normal predict flow."
            )

        n = prediction["occurs_probs"].reshape(-1).numel()
        if n == 0:
            return

        identifiers = _identifiers_from_schema_df(self._schema_df.slice(self._offset, n))
        # Cast to Float32 here so the per-batch ``PredictionSchema.align`` doesn't have
        # to coerce f64 → f32 every row group (mirrors the old ``_gather_probabilities``).
        probs = pl.DataFrame(
            {
                PredictionSchema.censor_prob_name: pl.Series(
                    prediction["censor_probs"].reshape(-1).numpy()
                ).cast(pl.Float32),
                PredictionSchema.occurs_prob_name: pl.Series(
                    prediction["occurs_probs"].reshape(-1).numpy()
                ).cast(pl.Float32),
            }
        )
        aligned = PredictionSchema.align(identifiers.hstack(probs).to_arrow())
        self._writer.write_table(aligned)
        self._offset += n

    def close(self) -> None:
        """Idempotent close — safe to call from both Lightning's teardown and main()'s finally."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def teardown(self, trainer, pl_module, stage: str) -> None:
        # Defense-in-depth: ``main()``'s ``try/finally`` is the primary closer.
        if stage == "predict":
            self.close()


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

    _validate_tasks_dir(tasks_dir)
    logger.info(f"Loading tasks from {tasks_dir} (split={split})")

    train_cfg, model, trainer = setup_model(model_run_dir, ckpt_name=ckpt_name)
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

    # Stream per-batch predictions to ``output_parquet`` so memory doesn't scale with
    # cohort size.  The callback opens the parquet writer in ``setup``, appends one row
    # group per batch in ``write_on_batch_end``, and is closed by the ``finally`` below
    # (idempotent — Lightning's ``teardown`` also calls ``close()``).  Pairing this with
    # ``return_predictions=False`` means Lightning drops each batch's ``predict_step``
    # dict as soon as the writer hook returns.
    writer = _StreamingPredictionWriter(output_parquet, dataset.schema_df)
    trainer.callbacks.append(writer)

    # Pass the split's dataloader directly — MTD's ``Datamodule`` has no
    # ``predict_dataloader``, so ``trainer.predict(datamodule=D)`` would hit the base
    # class's ``MisconfigurationException``.  The SequentialSampler check above
    # guarantees order preservation.
    try:
        trainer.predict(model=model, dataloaders=dataloader, ckpt_path=None, return_predictions=False)
        # Success-path invariant — MTD guarantees ``schema_df`` preserves the input
        # labels frame's length + order, so the writer should have streamed one
        # prediction per schema_df row.  A mismatch is a silent invariant violation;
        # fail loudly.  On an exception path, ``writer._offset`` will legitimately be
        # < ``schema_df.height`` (partial write) so we skip the check there.
        if writer._offset != dataset.schema_df.height:
            raise RuntimeError(
                f"Prediction row count ({writer._offset}) doesn't match dataset row count "
                f"({dataset.schema_df.height}).  This is an MTD invariant violation — "
                f"the dataloader should have yielded one prediction per schema_df row."
            )
    finally:
        writer.close()

    logger.info(f"Wrote {writer._offset} predictions to {output_parquet}")


if __name__ == "__main__":
    main()
