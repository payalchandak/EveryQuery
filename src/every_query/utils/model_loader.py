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
from omegaconf import OmegaConf

from every_query.model.lightning_module import EveryQueryLightningModule

logger = logging.getLogger(__name__)


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
    """
    model_run_dir = Path(model_run_dir)
    if not model_run_dir.is_dir():
        raise NotADirectoryError(f"{model_run_dir} is not a directory")

    train_cfg = OmegaConf.load(model_run_dir / "resolved_config.yaml")
    train_cfg.trainer.logger = ""

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

    logger.info("Instantiating trainer...")
    trainer = instantiate(train_cfg.trainer)

    return train_cfg, M, trainer
