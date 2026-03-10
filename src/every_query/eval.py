import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
import polars as pl
import torch
from hydra.utils import instantiate
from lightning.pytorch import seed_everything
from omegaconf import DictConfig, OmegaConf

from every_query.utils.codes import (  # noqa: F401 (values_as_list used by config.yaml)
    code_slug,
    values_as_list,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _setup_eval(cfg: DictConfig):
    model_run_dir = Path(cfg.model_run_dir)
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

    logger.info("Instantiating lightning module (architecture only)...")
    M = instantiate(train_cfg.lightning_module)

    logger.info("Instantiating trainer...")
    trainer = instantiate(train_cfg.trainer)

    task_set_dir = Path(cfg.task_set_dir)
    if not task_set_dir.is_dir():
        raise NotADirectoryError(f"{task_set_dir} is not a directory")

    return train_cfg, M, trainer, task_set_dir


def _run_test(cfg: DictConfig, train_cfg, M, trainer, task_set_dir: Path) -> None:
    codes: list[str] = []
    if cfg.id_codes is not None:
        codes += cfg.id_codes
    if cfg.ood_codes is not None:
        codes += cfg.ood_codes
    if cfg.manual_codes is not None:
        codes += cfg.manual_codes

    rows: list[dict[str, Any]] = []

    for code in codes:
        slug = code_slug(code)
        task_labels_dir = str(task_set_dir / slug)

        if not Path(task_labels_dir).is_dir():
            logger.warning(f"Missing task_labels_dir for code={code}: {task_labels_dir} (skipping)")
            continue

        train_cfg.datamodule.config.task_labels_dir = task_labels_dir
        D = instantiate(train_cfg.datamodule)

        t0 = time.time()
        out = trainer.test(model=M, datamodule=D, ckpt_path=cfg.ckpt_path)
        eval_time = time.time() - t0
        m = out[0] if out else {}

        rows.append(
            {
                "code": code,
                "code_slug": slug,
                "bucket": "ood" if code in cfg.ood_codes else "id",
                "occurs_auc": float(m.get("held_out/occurs_auc"))
                if m.get("held_out/occurs_auc") is not None
                else None,
                "censor_auc": float(m.get("held_out/censor_auc"))
                if m.get("held_out/censor_auc") is not None
                else None,
                "eval_time": eval_time,
            }
        )

    auc_df = pl.DataFrame(rows)

    out_dir = Path(cfg.output_root)
    date_str = datetime.now(tz=UTC).strftime("%Y-%m-%d_%H%M%S")
    out_fp = out_dir / f"all_code_aucs_{date_str}.csv"

    if out_fp.exists() and not cfg.do_overwrite:
        logger.info(f"Output exists at {out_fp}. Set do_overwrite=true to overwrite.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    auc_df.write_csv(out_fp)


def _run_predict(cfg: DictConfig, train_cfg, M, trainer, task_set_dir: Path) -> None:
    codes: list[str] = cfg.query_codes

    rows = []
    embed_rows = []

    for code in codes:
        slug = code_slug(code)
        task_labels_dir = str(task_set_dir / slug)

        if not Path(task_labels_dir).is_dir():
            logger.warning(f"Missing task_labels_dir for code={code}: {task_labels_dir} (skipping)")
            continue

        train_cfg.datamodule.config.task_labels_dir = task_labels_dir
        D = instantiate(train_cfg.datamodule)

        pred_batches = trainer.predict(model=M, datamodule=D, ckpt_path=cfg.ckpt_path)
        
        s_ids, p_times, o_probs, q_embeds = [], [], [], []
        for b in pred_batches:
            s_ids.append(b["subject_id"])
            p_times.append(b["prediction_time"])
            o_probs.append(b["occurs_probs"])
            q_embeds.append(b["query_embed"])
        subject_id = torch.cat(s_ids).numpy()
        prediction_time = torch.cat(p_times).numpy()
        occurs_probs = torch.cat(o_probs).numpy()
        query_embeds = torch.cat(q_embeds).numpy()


        rows.append(
            pl.DataFrame(
                {
                    "subject_id": subject_id,
                    "prediction_time": prediction_time,
                    "occurs_probs": occurs_probs,
                }
            ).with_columns(pl.lit(code).alias("code"))
        )
        embed_rows.append(
            pl.DataFrame(
                {
                    "subject_id": subject_id,
                    "prediction_time": prediction_time,
                    "code": [code] * len(subject_id),
                }
            ).with_columns(pl.Series("embedding", query_embeds))
        )

    if not rows:
        logger.warning("No predictions were generated — all codes were skipped.")
        return

    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    
    hc = HydraConfig.get()
    eval_codes_choice_str = hc.runtime.choices["eval_codes"]

    out_dir = Path(cfg.output_root)
    out_fp = out_dir / f"all_preds_{eval_codes_choice_str}_{timestamp}.csv"
    embed_fp = out_dir / f"query_embeds_{eval_codes_choice_str}_{timestamp}.parquet"

    out_dir.mkdir(parents=True, exist_ok=True)
    pl.concat(rows, how="vertical").write_csv(out_fp)
    pl.concat(embed_rows, how="vertical").write_parquet(embed_fp)
    logger.info(f"Saved predictions to {out_fp}")
    logger.info(f"Saved embeddings to {embed_fp}")


@hydra.main(version_base="1.3", config_path="./eval_suite/conf", config_name="eval_config.yaml")
def main(cfg: DictConfig) -> None:
    train_cfg, M, trainer, task_set_dir = _setup_eval(cfg)

    if cfg.mode == "predict":
        _run_predict(cfg, train_cfg, M, trainer, task_set_dir)
    else:
        _run_test(cfg, train_cfg, M, trainer, task_set_dir)


if __name__ == "__main__":
    main()
