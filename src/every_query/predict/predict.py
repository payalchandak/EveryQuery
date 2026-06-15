"""Inference-only entry point — ``EQ_predict``.

Takes a trained model run directory and a directory of task-query parquets
(:class:`TaskQuerySchema`), runs per-row inference, writes a
:class:`PredictionSchema`-conformant parquet.  No AUCs, no model selection, no
multi-model orchestration — all of that is ``evaluate/`` or ``experiments/``
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


_EMBEDDING_COLUMN = "embedding"


def _embedding_batch_to_arrow(query_embed: torch.Tensor, hidden_size: int) -> pa.FixedSizeListArray:
    """Build a single-batch ``fixed_size_list<float32>[hidden_size]`` array from ``query_embed``.

    The streaming writer calls this once per ``write_on_batch_end`` to produce the
    per-batch embedding column.  Pre-building a fixed-size-list arrow array (rather
    than letting polars infer a variable-length list) preserves the documented
    sidecar schema across every row group of the streamed parquet — including the
    empty-cohort case, where the writer's pre-built schema in :meth:`setup`
    pins the same fixed-size-list type with no rows.

    Examples:
        Three rows with hidden_size=4:

        >>> arr = _embedding_batch_to_arrow(torch.arange(12, dtype=torch.float32).reshape(3, 4), 4)
        >>> arr.type
        FixedSizeListType(fixed_size_list<item: float>[4])
        >>> len(arr)
        3
    """
    flat = pa.array(query_embed.reshape(-1).numpy().astype("float32"), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, hidden_size)


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
    """Lightning ``BasePredictionWriter`` that streams per-batch results to single parquets.

    Writes probabilities to ``output_parquet`` and (when ``embeddings_output_parquet``
    is set) pooled query embeddings to that sibling.  Each ``write_on_batch_end``
    call appends one row group to each open writer so EQ_predict's CPU-memory
    footprint doesn't scale with cohort size — only one batch's ``predict_step``
    dict is alive at a time (Lightning drops it after the writer hook returns when
    ``return_predictions=False``).

    Identifiers come from slicing the dataset's pre-loaded ``schema_df`` with a running
    offset; correctness depends on the dataloader iterating in ``schema_df`` row order,
    which ``main()`` enforces via the ``SequentialSampler`` guard.

    Partial-write semantics: ``close()`` is idempotent and is invoked from ``main()``'s
    ``try/finally`` block, so a Python-level exception mid-stream still flushes both
    parquet footers and leaves valid files covering exactly the batches that completed.
    Hard kills (SIGKILL) cannot be recovered — parquet has no append-after-close, so
    single-file streaming has no defense against them.
    """

    def __init__(
        self,
        output_parquet: Path,
        schema_df: pl.DataFrame,
        embeddings_output_parquet: Path | None = None,
        hidden_size: int | None = None,
    ) -> None:
        super().__init__(write_interval="batch")
        if embeddings_output_parquet is not None and hidden_size is None:
            raise ValueError(
                "embeddings_output_parquet requires hidden_size — needed to build the "
                "fixed-size-list arrow schema for the streamed embeddings sibling."
            )
        self._output_parquet = output_parquet
        self._schema_df = schema_df
        self._embeddings_output_parquet = embeddings_output_parquet
        self._hidden_size = hidden_size
        self._writer: pq.ParquetWriter | None = None
        self._emb_writer: pq.ParquetWriter | None = None
        self._offset: int = 0
        self._arrow_schema: pa.Schema | None = None
        self._emb_arrow_schema: pa.Schema | None = None

    def setup(self, trainer, pl_module, stage: str) -> None:
        if stage != "predict":
            return
        # Pre-build the canonical arrow schema from a zero-row aligned table.  Doing
        # this up front (rather than on the first batch) means an empty cohort still
        # produces a valid PredictionSchema parquet — matches the pre-streaming
        # behavior where the gather helpers returned an empty frame and the
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

        if self._embeddings_output_parquet is not None:
            empty_ids_arrow = empty_ids.to_arrow()
            empty_emb = pa.FixedSizeListArray.from_arrays(pa.array([], type=pa.float32()), self._hidden_size)
            empty_emb_table = empty_ids_arrow.append_column(_EMBEDDING_COLUMN, empty_emb)
            self._emb_arrow_schema = empty_emb_table.schema
            self._embeddings_output_parquet.parent.mkdir(parents=True, exist_ok=True)
            self._emb_writer = pq.ParquetWriter(
                self._embeddings_output_parquet, schema=self._emb_arrow_schema
            )

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
        # to coerce f64 → f32 every row group.
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

        if self._emb_writer is not None:
            emb_arr = _embedding_batch_to_arrow(prediction["query_embed"], self._hidden_size)
            emb_table = identifiers.to_arrow().append_column(_EMBEDDING_COLUMN, emb_arr)
            self._emb_writer.write_table(emb_table)

        self._offset += n

    def close(self) -> None:
        """Idempotent close — safe to call from both Lightning's teardown and main()'s finally."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self._emb_writer is not None:
            self._emb_writer.close()
            self._emb_writer = None

    def teardown(self, trainer, pl_module, stage: str) -> None:
        # Defense-in-depth: ``main()``'s ``try/finally`` is the primary closer.
        if stage == "predict":
            self.close()


def _check_single_process_trainer(trainer) -> None:
    """Refuse multi-process predict — :class:`_StreamingPredictionWriter` assumes one rank.

    The writer opens a single ``output_parquet`` (and optionally an embeddings sibling)
    and tracks a process-local offset into ``schema_df``.  Under multi-process predict
    (DDP / multi-device / multi-node), every rank would open the same paths and start
    slicing ``schema_df`` from offset 0 — corrupting the parquets and mispairing
    probabilities with identifiers.  ``setup_model`` instantiates the trainer from the
    saved training config, so a model trained with ``trainer.devices > 1`` carries that
    strategy into predict.  Predict is single-pass by design (see module docstring);
    refuse loudly rather than write corrupted output.
    """
    if trainer.world_size > 1:
        raise RuntimeError(
            f"EQ_predict requires single-process prediction, but the trainer instantiated from "
            f"the training config has world_size={trainer.world_size} "
            f"(num_devices={trainer.num_devices}, num_nodes={trainer.num_nodes}).  "
            f"Override the run dir's resolved_config.yaml to set "
            f"trainer.devices=1, trainer.strategy=auto, trainer.num_nodes=1 before running predict."
        )


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


def _tmp_path(p: Path) -> Path:
    """Return a sibling ``.tmp`` path used as the streaming target before atomic rename.

    Appending ``.tmp`` to the full file name (rather than swapping the suffix)
    keeps the original parquet suffix in the temp name — useful when grepping
    for stray temp files left by hard kills, and avoids any tooling that keys
    off the ``.parquet`` extension misinterpreting the partial file.

    Examples:
        >>> _tmp_path(Path("/tmp/predictions.parquet"))
        PosixPath('/tmp/predictions.parquet.tmp')

        >>> _tmp_path(Path("/a/predictions.embeddings.parquet"))
        PosixPath('/a/predictions.embeddings.parquet.tmp')
    """
    return p.with_name(p.name + ".tmp")


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
    _check_single_process_trainer(trainer)

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

    # Stream per-batch predictions to a ``.tmp`` sibling of ``output_parquet`` (and
    # the optional embeddings sibling) so memory doesn't scale with cohort size.
    # The callback opens the parquet writers in ``setup``, appends one row group per
    # batch in ``write_on_batch_end``, and is closed by the ``finally`` below
    # (idempotent — Lightning's ``teardown`` also calls ``close()``).  Pairing this
    # with ``return_predictions=False`` means Lightning drops each batch's
    # ``predict_step`` dict as soon as the writer hook returns.
    #
    # On full success (predict completes, row-count invariant holds, timing
    # parquet flushed) all temp files are atomically renamed onto the canonical
    # paths via ``Path.replace`` so the canonical paths are guaranteed complete.
    # On a soft failure (Python exception / KeyboardInterrupt) the writer is
    # still ``close()``'d so the partial parquet's footer is flushed — leaving a
    # viewable forensic ``.tmp`` on disk that downstream consumers won't pick up
    # (canonical path stays absent).  A subsequent successful run truncates the
    # stale ``.tmp`` via the new ``ParquetWriter``; manual ``rm`` clears it
    # otherwise.  Hard kills (SIGKILL) leave a footer-less ``.tmp`` that needs
    # recovery tooling to read — unavoidable property of streaming parquet.
    output_tmp = _tmp_path(output_parquet)
    embeddings_tmp = _tmp_path(embeddings_output_parquet) if embeddings_output_parquet is not None else None
    timing_tmp = _tmp_path(timing_output_parquet)

    hidden_size = model.model.HF_model.config.hidden_size if save_embeddings else None
    writer = _StreamingPredictionWriter(
        output_tmp,
        dataset.schema_df,
        embeddings_output_parquet=embeddings_tmp,
        hidden_size=hidden_size,
    )
    trainer.callbacks.append(writer)

    # Pass the split's dataloader directly — MTD's ``Datamodule`` has no
    # ``predict_dataloader``, so ``trainer.predict(datamodule=D)`` would hit the base
    # class's ``MisconfigurationException``.  The SequentialSampler check above
    # guarantees order preservation.
    t0 = time.perf_counter()
    try:
        trainer.predict(model=model, dataloaders=dataloader, ckpt_path=None, return_predictions=False)
        # Success-path invariant — MTD guarantees ``schema_df`` preserves the input
        # labels frame's length + order, so the writer should have streamed one
        # prediction per schema_df row.  A mismatch is a silent invariant violation;
        # fail loudly so the rename below never runs.
        if writer._offset != dataset.schema_df.height:
            raise RuntimeError(
                f"Prediction row count ({writer._offset}) doesn't match dataset row count "
                f"({dataset.schema_df.height}).  This is an MTD invariant violation — "
                f"the dataloader should have yielded one prediction per schema_df row."
            )
        writer.close()
        total_seconds = time.perf_counter() - t0

        n_rows = dataset.schema_df.height
        n_tasks = dataset.schema_df.select(["query", "duration_days"]).unique().height
        timing_table = pa.table(
            {
                "total_seconds": pa.array([total_seconds], type=pa.float64()),
                "n_rows": pa.array([n_rows], type=pa.int64()),
                "n_tasks": pa.array([n_tasks], type=pa.int64()),
                "seconds_per_row": pa.array([total_seconds / n_rows if n_rows else 0.0], type=pa.float64()),
                "seconds_per_task": pa.array(
                    [total_seconds / n_tasks if n_tasks else 0.0], type=pa.float64()
                ),
            }
        )
        timing_tmp.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(timing_table, timing_tmp)

        # Atomic rename phase.  ``Path.replace`` is per-file atomic on POSIX; the
        # multi-file window between renames is microseconds vs. the minutes-long
        # streaming window, so a crash here is vastly less likely to leave a
        # partial set than streaming directly to the final paths.
        output_tmp.replace(output_parquet)
        if embeddings_tmp is not None and embeddings_output_parquet is not None:
            embeddings_tmp.replace(embeddings_output_parquet)
        timing_tmp.replace(timing_output_parquet)
    except BaseException:
        # Flush parquet footers so the partial ``.tmp`` files are viewable for
        # forensics; do NOT unlink — canonical paths stay absent because the
        # rename block never ran, so no consumer will mistake a partial for a
        # finished run.  Stale ``.tmp`` files are overwritten by the next
        # successful run or removed manually.
        writer.close()
        raise

    logger.info(f"Wrote {writer._offset} predictions to {output_parquet}")
    if embeddings_output_parquet is not None:
        logger.info(f"Wrote {writer._offset} embeddings to {embeddings_output_parquet}")
    logger.info(f"Wrote timing ({total_seconds:.2f}s, {n_tasks} tasks) to {timing_output_parquet}")


if __name__ == "__main__":
    main()
