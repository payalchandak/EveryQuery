"""Shared helper for loading a trained EveryQuery checkpoint + trainer.

Called by both ``predict/predict.py`` (inference) and ``evaluate/eval.py`` (the legacy
eval pipeline that still drives metrics today).  Lives under ``utils/`` rather than
inside either stage so it survives the #82 consolidation wave's deletion of
``evaluate/eval.py`` without breaking ``predict``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl
import torch
from hydra.utils import instantiate
from lightning.pytorch import seed_everything
from omegaconf import DictConfig, OmegaConf

from every_query.model.lightning_module import EveryQueryLightningModule

logger = logging.getLogger(__name__)


def _check_vocab_size(ckpt_vocab_size: int, train_cfg: DictConfig) -> None:
    """Raise if the checkpoint's embedding table doesn't match the cohort's code vocabulary.

    Since #283 the embedding table is sized from `MEDSTorchDataConfig.vocab_size`, so a
    checkpoint's table size is a fingerprint of the code metadata it was trained against.
    `code/vocab_index` is assigned lexicographically over the whole code set, so re-tensorizing
    a cohort with even one extra code renumbers *every* index — the checkpoint would still load
    (the state dict shape check only fails if the count changed) but every embedding lookup
    would return another code's vector.  Refuse instead of silently predicting from a shuffled
    vocabulary.

    Args:
        ckpt_vocab_size: The loaded model's embedding table size.
        train_cfg: The run's `resolved_config.yaml`.

    Raises:
        ValueError: if the cohort's current vocabulary size differs from the checkpoint's.

    Examples:
        Sizes agree — no raise (`code/vocab_index` is 1-based, so `max + 1` is 4 here):

        >>> import tempfile
        >>> def cfg_for(tmpdir, vocab_size=4):
        ...     return OmegaConf.create({
        ...         "datamodule": {"config": {"tensorized_cohort_dir": tmpdir}},
        ...         "lightning_module": {"model": {"vocab_size": vocab_size}},
        ...     })
        >>> def write_codes(tmpdir, indices):
        ...     meta_dir = Path(tmpdir) / "metadata"
        ...     meta_dir.mkdir(exist_ok=True)
        ...     pl.DataFrame({"code/vocab_index": indices}).write_parquet(meta_dir / "codes.parquet")
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     write_codes(tmpdir, [1, 2, 3])
        ...     _check_vocab_size(4, cfg_for(tmpdir))  # no output, no error

        A cohort that has since gained a code raises:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     write_codes(tmpdir, [1, 2, 3, 4])
        ...     _check_vocab_size(4, cfg_for(tmpdir))
        Traceback (most recent call last):
            ...
        ValueError: Checkpoint vocabulary size (4) does not match the cohort at ...

        Checkpoints predating #283 carry ModernBERT's text vocabulary rather than a
        data-derived one, so there is nothing meaningful to compare and the check is skipped:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     write_codes(tmpdir, [1, 2, 3, 4])
        ...     _check_vocab_size(50368, cfg_for(tmpdir, vocab_size=None))  # no error

        Missing metadata warns rather than raises — the checkpoint is still usable, we just
        can't verify it:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     _check_vocab_size(4, cfg_for(tmpdir))  # no error
    """
    if OmegaConf.select(train_cfg, "lightning_module.model.vocab_size") is None:
        return

    cohort_dir = Path(train_cfg.datamodule.config.tensorized_cohort_dir)
    metadata_fp = cohort_dir / "metadata" / "codes.parquet"
    if not metadata_fp.is_file():
        logger.warning(
            f"No code metadata at {metadata_fp}; cannot verify that the checkpoint's vocabulary "
            f"({ckpt_vocab_size}) still matches the cohort it was trained on."
        )
        return

    data_vocab_size = (
        pl.read_parquet(metadata_fp, columns=["code/vocab_index"], use_pyarrow=True)["code/vocab_index"].max()
        + 1
    )
    if data_vocab_size != ckpt_vocab_size:
        raise ValueError(
            f"Checkpoint vocabulary size ({ckpt_vocab_size}) does not match the cohort at "
            f"{cohort_dir} ({data_vocab_size}).  `code/vocab_index` is assigned lexicographically "
            f"over the full code set, so a cohort whose vocabulary has changed size has also "
            f"renumbered the codes this model was trained on — every embedding lookup would "
            f"return the wrong code's vector.  Re-train against the current cohort, or point "
            f"`datamodule.config.tensorized_cohort_dir` at the cohort the model was trained on."
        )


def setup_model(model_run_dir: str | Path, ckpt_name: str | None = None):
    """Resolve a ``model_run_dir`` into a ``(train_cfg, lightning_module, trainer)`` triple.

    Reads ``resolved_config.yaml`` from the run dir, seeds RNG, picks a checkpoint
    (explicit ``ckpt_name`` → ``best_model.ckpt`` → ``checkpoints/last.ckpt``), loads the
    Lightning module, and instantiates a trainer from the config.

    Args:
        model_run_dir: Path to a training run directory.  Must contain
            ``resolved_config.yaml`` and at least one of ``best_model.ckpt`` or
            ``checkpoints/last.ckpt`` (unless ``ckpt_name`` is given).
        ckpt_name: Optional explicit checkpoint stem under ``checkpoints/``.  ``None``
            or the literal string ``"best"`` falls back to ``best_model.ckpt``.

    Returns:
        Tuple ``(train_cfg, lightning_module, trainer)``.  ``train_cfg`` is the OmegaConf
        loaded from ``resolved_config.yaml`` (with ``trainer.logger`` blanked out so the
        trainer instantiation stays offline).  The trainer is freshly constructed from
        ``train_cfg.trainer`` — callers that need different trainer kwargs should
        override the config before calling.

    Raises:
        NotADirectoryError: if ``model_run_dir`` isn't a directory.
        FileNotFoundError: if no checkpoint can be resolved.
        ValueError: if the checkpoint's embedding table no longer matches the cohort's code
            vocabulary (see :func:`_check_vocab_size`).
    """
    model_run_dir = Path(model_run_dir)
    if not model_run_dir.is_dir():
        raise NotADirectoryError(f"{model_run_dir} is not a directory")

    train_cfg = OmegaConf.load(model_run_dir / "resolved_config.yaml")
    # Disable the training-time logger for inference: ``Trainer(logger=...)`` expects
    # ``bool | Logger | Iterable[Logger] | None`` per the Lightning contract.  Leaving
    # the training-time config in place would try to re-init a wandb run during
    # predict/evaluate; ``False`` turns it off cleanly.
    train_cfg.trainer.logger = False

    seed = train_cfg.get("seed", 42)
    if seed is not None:
        logger.info(f"Seeding with seed={seed}")
        seed_everything(seed, workers=True)

    logger.info("Setting torch float32 matmul precision to 'medium'.")
    torch.set_float32_matmul_precision("medium")

    # Resolve checkpoint: explicit name → best_model.ckpt → last.ckpt
    candidates = (
        [model_run_dir / "checkpoints" / f"{ckpt_name}.ckpt"]
        if ckpt_name is not None and ckpt_name != "best"
        else [
            model_run_dir / "best_model.ckpt",
            model_run_dir / "checkpoints" / "last.ckpt",
        ]
    )
    ckpt_path = next((p for p in candidates if p.is_file()), None)
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No checkpoint found in {model_run_dir} (tried {[str(p) for p in candidates]})"
        )
    if ckpt_path != candidates[0]:
        logger.warning(f"{candidates[0].name} not found, falling back to {ckpt_path}")

    logger.info(f"Loading lightning module from checkpoint: {ckpt_path}")
    M = EveryQueryLightningModule.load_from_checkpoint(str(ckpt_path))

    _check_vocab_size(M.model.vocab_size, train_cfg)

    logger.info("Instantiating trainer...")
    trainer = instantiate(train_cfg.trainer)

    return train_cfg, M, trainer
