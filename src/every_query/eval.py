import logging
from pathlib import Path
from typing import Any
import hashlib
import re

import hydra
import polars as pl
import torch
from hydra.utils import instantiate
from lightning.pytorch import seed_everything
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def values_as_list(**kwargs) -> list[Any]:
    return list(kwargs.values())


def code_slug(code: str, n_hash: int = 10, prefix_len: int = 24) -> str:
    h = hashlib.sha1(code.encode("utf-8")).hexdigest()[:n_hash]
    prefix = re.sub(r"[^A-Za-z0-9._-]+", "_", code).strip("_")[:prefix_len]
    return f"{prefix}__{h}" if prefix else h



@hydra.main(version_base="1.3", config_path="", config_name="eval_config.yaml")
def main(cfg: DictConfig) -> None:
    model_run_dir = Path(cfg.model_run_dir)
    if not model_run_dir.is_dir():
        raise NotADirectoryError(f"{model_run_dir} is not a directory")

    # Load training config
    train_cfg = OmegaConf.load(model_run_dir / "resolved_config.yaml")

    # Nuke the logger so wandb dashboard is clean
    train_cfg.trainer.logger = ""

    seed = train_cfg.get("seed", 42)
    if seed is not None:
        logger.info(f"Seeding with seed={seed}")
        seed_everything(seed, workers=True)

    logger.info("Setting torch float32 matmul precision to 'medium'.")
    torch.set_float32_matmul_precision("medium")

    logger.info("Instantiating lightning module (architecture only)...")
    M = instantiate(train_cfg.lightning_module)

    logger.info("Instantiating trainer...")
    trainer = instantiate(train_cfg.trainer)
    
    task_set_dir = Path(cfg.task_set_dir)
    if not task_set_dir.is_dir():
        raise NotADirectoryError(f"{task_set_dir} is not a directory")

    codes = list(map(str, cfg.eval_codes))
    if not codes:
        raise ValueError("cfg.eval_codes is empty")

    rows: list[dict[str, Any]] = []

    for code in codes:
        slug = code_slug(code)
        task_labels_dir = str(task_set_dir / slug)

        if not Path(task_labels_dir).is_dir():
            logger.warning(f"Missing task_labels_dir for code={code}: {task_labels_dir} (skipping)")
            continue

        # Point datamodule at this code’s task dfs
        train_cfg.datamodule.config.task_labels_dir = task_labels_dir
        D = instantiate(train_cfg.datamodule)

        out = trainer.test(model=M, datamodule=D, ckpt_path=cfg.ckpt_path)
        m = out[0] if out else {}

        rows.append(
            {
                "code": code,
                "code_slug": slug,
                "bucket": str(cfg.bucket) if cfg.get("bucket") is not None else None,
                "occurs_auc": float(m.get(f"{split}/occurs_auc")) if m.get(f"{split}/occurs_auc") is not None else None,
                "censor_auc": float(m.get(f"{split}/censor_auc")) if m.get(f"{split}/censor_auc") is not None else None,
            }
        )





    auc_df = pl.DataFrame(rows)

    # save
    out_dir = Path(cfg.output_root)
    out_fp = out_dir / "all_code_aucs.csv"

    if out_fp.exists() and not cfg.do_overwrite:
        logger.info(f"Output exists at {out_fp}. Set do_overwrite=true to overwrite.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    auc_df.write_csv(out_fp)


if __name__ == "__main__":
    main()
