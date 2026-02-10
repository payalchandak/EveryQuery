import logging
from pathlib import Path
from typing import Any

import hydra
import polars as pl
import torch
from hydra.utils import instantiate
from lightning.pytorch import seed_everything
from omegaconf import DictConfig, OmegaConf

from every_query.utils.codes import code_slug

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def values_as_list(**kwargs) -> list[Any]:
    return list(kwargs.values())


@hydra.main(version_base="1.3", config_path="./eval_suite/conf", config_name="eval_composite_config.yaml")
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

    codes: list[str] = cfg.query_codes
    rows = []

    for code in codes:
        slug = code_slug(code)
        task_labels_dir = str(task_set_dir / slug)

        if not Path(task_labels_dir).is_dir():
            logger.warning(f"Missing task_labels_dir for code={code}: {task_labels_dir} (skipping)")
            continue

        # Point datamodule at this code task df
        train_cfg.datamodule.config.task_labels_dir = task_labels_dir
        D = instantiate(train_cfg.datamodule)

        pred_batches = trainer.predict(model=M, datamodule=D, ckpt_path=cfg.ckpt_path)

        subject_id = torch.cat([b["subject_id"] for b in pred_batches]).numpy()
        prediction_time = torch.cat([b["prediction_time"] for b in pred_batches]).numpy()
        occurs_probs = torch.cat([b["occurs_probs"] for b in pred_batches]).numpy()

        df = pl.DataFrame(
            {
                "subject_id": subject_id,
                "prediction_time": prediction_time,
                "occurs_probs": occurs_probs,
            }
        ).with_columns(pl.lit(code).alias("code"))

        rows.append(df)

    final_df = pl.concat(rows, how="vertical")

    # save
    out_dir = Path(cfg.output_root)
    out_fp = out_dir / f"{cfg.task_name}_all_preds.csv"

    if out_fp.exists() and not cfg.do_overwrite:
        logger.info(f"Output exists at {out_fp}. Set do_overwrite=true to overwrite.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    final_df.write_csv(out_fp)


if __name__ == "__main__":
    main()
