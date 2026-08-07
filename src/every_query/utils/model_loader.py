"""Shared helper for loading a trained EveryQuery checkpoint + trainer.

Called by both ``predict/predict.py`` (inference) and ``evaluate/eval.py`` (the legacy
eval pipeline that still drives metrics today).  Lives under ``utils/`` rather than
inside either stage so it survives the #82 consolidation wave's deletion of
``evaluate/eval.py`` without breaking ``predict``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from hydra.utils import instantiate
from lightning.pytorch import seed_everything
from omegaconf import DictConfig, OmegaConf

from every_query.model.lightning_module import EveryQueryLightningModule

logger = logging.getLogger(__name__)


def _check_vocab_size(ckpt_vocab_size: int | None, train_cfg: DictConfig) -> None:
    """Raise if the checkpoint's embedding table is a different size than the cohort's vocabulary.

    Since #283 the embedding table is sized from `MEDSTorchDataConfig.vocab_size`, so its size is
    a partial fingerprint of the code metadata the model was trained against.
    `code/vocab_index` is assigned lexicographically over the whole code set, so re-tensorizing a
    cohort with even one extra code renumbers *every* index — the checkpoint would still load
    (the state-dict shape check only fails if the count changed) but every embedding lookup would
    return another code's vector.  Refuse instead of silently predicting from a shuffled
    vocabulary.

    This is a size check, not an identity check: a cohort that gained one code and dropped
    another, or had one renamed, has the same `max(code/vocab_index) + 1` and is renumbered just
    as thoroughly.  Catching that needs a hash of the code set recorded at train time; see #283
    for the follow-up.  What's here catches the common case at the cost of one parquet read.

    Args:
        ckpt_vocab_size: The checkpoint's ``hparams["model"]["vocab_size"]``.  `None` for
            checkpoints predating #283, which carry ModernBERT's 50368-entry text vocabulary
            rather than a data-derived one — nothing meaningful to compare, so the check is
            skipped.  Read off the checkpoint rather than the run's ``resolved_config.yaml``
            because ``setup_model`` will load any checkpoint under ``checkpoints/``, including
            one that predates the config sitting next to it.
        train_cfg: The run's ``resolved_config.yaml``.

    Raises:
        ValueError: if the cohort's current vocabulary size differs from the checkpoint's.
        FileNotFoundError: if the cohort has no code metadata to check against — the same
            policy `predict._check_vocab` applies to the same file.

    Examples:
        >>> import tempfile
        >>> import polars as pl
        >>> def cfg_for(tmpdir):
        ...     return OmegaConf.create({"datamodule": {"config": {
        ...         "tensorized_cohort_dir": str(tmpdir), "max_seq_len": 10,
        ...         "_target_": "meds_torchdata.config.MEDSTorchDataConfig",
        ...     }}})
        >>> tmp = Path(tempfile.mkdtemp())
        >>> (tmp / "metadata").mkdir()
        >>> pl.DataFrame({"code/vocab_index": [1, 2, 3]}).write_parquet(
        ...     tmp / "metadata" / "codes.parquet")

        Sizes agree — no raise (`code/vocab_index` is 1-based, so `max + 1` is 4 here):

        >>> _check_vocab_size(4, cfg_for(tmp))

        A cohort that has since gained a code raises:

        >>> _check_vocab_size(3, cfg_for(tmp))
        Traceback (most recent call last):
            ...
        ValueError: Checkpoint vocabulary size (3) does not match the cohort at ...

        A checkpoint predating #283 is skipped without reading anything:

        >>> _check_vocab_size(None, cfg_for(tmp / "nonexistent"))
    """
    if ckpt_vocab_size is None:
        return

    # Same call `train.py` sizes the table with, so the two can't drift.
    data_vocab_size = instantiate(train_cfg.datamodule.config).vocab_size
    if data_vocab_size != ckpt_vocab_size:
        cohort_dir = train_cfg.datamodule.config.tensorized_cohort_dir
        raise ValueError(
            f"Checkpoint vocabulary size ({ckpt_vocab_size}) does not match the cohort at "
            f"{cohort_dir} ({data_vocab_size}).  `code/vocab_index` is assigned lexicographically "
            f"over the full code set, so a cohort whose vocabulary has changed size has also "
            f"renumbered the codes this model was trained on — every embedding lookup would "
            f"return the wrong code's vector.  Re-train against the current cohort, or edit "
            f"`datamodule.config.tensorized_cohort_dir` in the run's `resolved_config.yaml` to "
            f"point at the cohort the model was trained on."
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
        FileNotFoundError: if the cohort has no code metadata to check the checkpoint against.
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

    _check_vocab_size(M.model.hparams.get("vocab_size"), train_cfg)

    logger.info("Instantiating trainer...")
    trainer = instantiate(train_cfg.trainer)

    return train_cfg, M, trainer
