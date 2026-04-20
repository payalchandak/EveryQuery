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


def _warn_out_of_vocab(task_codes: set[str], train_cfg: DictConfig) -> None:
    """Warn if any task-query codes are missing from the trained model's vocabulary.

    Out-of-vocab query codes survive the predict loop (``encode_query`` silently falls
    back to ``PAD_INDEX``) but produce effectively-uniform predictions.  Log a warning
    at startup so the caller notices rather than silently getting garbage probabilities.

    Examples:
        Happy path — every task code is present in the training vocab, no warning:

        >>> import tempfile
        >>> from omegaconf import OmegaConf
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     meta_dir = Path(tmpdir) / "metadata"
        ...     meta_dir.mkdir()
        ...     pl.DataFrame({"code": ["A", "B", "C"]}).write_parquet(meta_dir / "codes.parquet")
        ...     cfg = OmegaConf.create({"datamodule": {"config": {"tensorized_cohort_dir": tmpdir}}})
        ...     _warn_out_of_vocab({"A", "B"}, cfg)  # no raise, no logged warning

        Out-of-vocab codes trigger a warning (raised-vs-logged distinction: this is a
        soft signal, not an error — malformed inputs get visible-but-survivable
        degradation rather than a hard stop):

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     meta_dir = Path(tmpdir) / "metadata"
        ...     meta_dir.mkdir()
        ...     pl.DataFrame({"code": ["A"]}).write_parquet(meta_dir / "codes.parquet")
        ...     cfg = OmegaConf.create({"datamodule": {"config": {"tensorized_cohort_dir": tmpdir}}})
        ...     _warn_out_of_vocab({"A", "MISSING"}, cfg)  # logs warning, no raise

        Missing metadata file also warns rather than erroring — predict should still
        be able to run against a training dir that happens to lack the metadata:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     cfg = OmegaConf.create({"datamodule": {"config": {"tensorized_cohort_dir": tmpdir}}})
        ...     _warn_out_of_vocab({"A"}, cfg)  # logs warning, no raise
    """
    metadata_fp = Path(train_cfg.datamodule.config.tensorized_cohort_dir) / "metadata" / "codes.parquet"
    if not metadata_fp.is_file():
        logger.warning(f"Cannot resolve training vocabulary — no codes metadata at {metadata_fp}")
        return
    training_vocab = set(pl.read_parquet(metadata_fp, columns=["code"])["code"].to_list())
    missing = task_codes - training_vocab
    if missing:
        logger.warning(
            f"{len(missing)} of {len(task_codes)} task-query codes are not in the model's training "
            f"vocabulary and will be PAD-encoded (predictions will be near-uniform for these rows): "
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

    if split not in _SPLIT_TO_DATAMODULE_ATTRS:
        raise ValueError(
            f"split must be one of {sorted(_SPLIT_TO_DATAMODULE_ATTRS)}, got {split!r}.  "
            f"The 'train' split is disallowed because MTD's train_dataloader shuffles, "
            f"which would break the order-preserving hstack."
        )

    _validate_tasks_dir(tasks_dir)
    logger.info(f"Loading tasks from {tasks_dir} (split={split})")

    train_cfg, model, trainer = setup_model(model_run_dir, ckpt_name=ckpt_name)
    train_cfg.datamodule.config.task_labels_dir = str(tasks_dir)
    D = instantiate(train_cfg.datamodule)

    dataset_attr, dataloader_attr = _SPLIT_TO_DATAMODULE_ATTRS[split]
    dataset = getattr(D, dataset_attr)
    dataloader = getattr(D, dataloader_attr)()

    _warn_out_of_vocab(set(dataset.query.unique().to_list()), train_cfg)
    logger.info(f"Loaded {len(dataset)} tasks across {dataset.query.n_unique()} query codes")

    # Pass the split's dataloader directly — MTD's ``Datamodule`` has no
    # ``predict_dataloader``, so ``trainer.predict(datamodule=D)`` would hit the base
    # class's ``MisconfigurationException``.  ``val_dataloader`` / ``test_dataloader``
    # both use ``shuffle=False``.
    pred_batches = trainer.predict(model=model, dataloaders=dataloader, ckpt_path=None)

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


if __name__ == "__main__":
    main()
