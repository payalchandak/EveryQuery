import logging
from pathlib import Path
from typing import Any

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
    manifest_df = (
        pl.read_parquet(cfg.manifest_path)
        .select(["bucket", "code", "code_slug"])
        .unique()
        .sort(["bucket", "code"])
    )

    rows: list[dict[str, Any]] = []

    split = cfg.split

    for row in manifest_df.iter_rows(named=True):
        bucket = row["bucket"]
        code = row["code"]
        code_slug = row["code_slug"]

        task_labels_dir = str(task_set_dir / bucket / code_slug)

        train_cfg.datamodule.config.task_labels_dir = task_labels_dir
        D = instantiate(train_cfg.datamodule)

        out = trainer.test(model=M, datamodule=D, ckpt_path=cfg.ckpt_path)
        m = out[0] if out else {}

        rows.append(
            {
                "code": code,
                "bucket": bucket,
                "occurs_auc": float(m[f"{split}/occurs_auc"]) if f"{split}/occurs_auc" in m else None,
                "censor_auc": float(m[f"{split}/censor_auc"]) if f"{split}/censor_auc" in m else None,
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
