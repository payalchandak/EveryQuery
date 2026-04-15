import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import hydra
import polars as pl
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from lightning.pytorch import seed_everything
from omegaconf import DictConfig, OmegaConf

from every_query.lightning_module import EveryQueryLightningModule
from every_query.utils.codes import (  # noqa: F401 (values_as_list used by config.yaml)
    code_slug,
    values_as_list,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _setup_model(model_run_dir: str | Path, ckpt_name: str | None = None):
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


def _model_name(model_run_dir: str) -> str:
    """Derive a short model name from the run directory path."""
    return Path(model_run_dir).name


def _collect_codes(cfg: DictConfig) -> tuple[list[str], dict[str, str]]:
    """Combine id/ood/manual codes into a flat list plus a code→bucket map."""
    codes: list[str] = []
    bucket: dict[str, str] = {}
    for c in cfg.id_codes or []:
        if c not in bucket:
            codes.append(c)
            bucket[c] = "id"
    for c in cfg.ood_codes or []:
        if c not in bucket:
            codes.append(c)
            bucket[c] = "ood"
    for c in cfg.manual_codes or []:
        if c not in bucket:
            codes.append(c)
            bucket[c] = "manual"
    return codes, bucket


def _run_test(
    cfg: DictConfig,
    train_cfg,
    M,
    trainer,
    task_set_dir: Path,
    model_name: str,
    durations: list[int],
    ckpt_name: str | None = None,
) -> pl.DataFrame:
    codes, bucket = _collect_codes(cfg)

    rows: list[dict[str, Any]] = []

    for duration in durations:
        for code in codes:
            slug = code_slug(code)
            task_labels_dir = str(task_set_dir / str(duration) / slug)

            if not Path(task_labels_dir).is_dir():
                logger.warning(
                    f"Missing task_labels_dir for code={code}, duration={duration}: "
                    f"{task_labels_dir} (skipping)"
                )
                continue

            train_cfg.datamodule.config.task_labels_dir = task_labels_dir
            D = instantiate(train_cfg.datamodule)

            t0 = time.time()
            if cfg.split == "tuning":
                out = trainer.validate(model=M, datamodule=D, ckpt_path=None)
            else:
                out = trainer.test(model=M, datamodule=D, ckpt_path=None)
            eval_time = time.time() - t0
            m = out[0] if out else {}

            metric_prefix = "tuning" if cfg.split == "tuning" else "held_out"
            rows.append(
                {
                    "model": model_name,
                    "duration_days": duration,
                    "code": code,
                    "code_slug": slug,
                    "bucket": bucket[code],
                    "occurs_auc": float(m.get(f"{metric_prefix}/occurs_auc"))
                    if m.get(f"{metric_prefix}/occurs_auc") is not None
                    else None,
                    "censor_auc": float(m.get(f"{metric_prefix}/censor_auc"))
                    if m.get(f"{metric_prefix}/censor_auc") is not None
                    else None,
                    "num_layers": M.model.HF_model_config.num_hidden_layers,
                    "max_seq_len": train_cfg.datamodule.config.max_seq_len,
                    "eval_time": eval_time,
                    "ckpt": ckpt_name,
                }
            )

    return pl.DataFrame(rows)


def _run_predict(
    cfg: DictConfig, train_cfg, M, trainer, task_set_dir: Path, model_name: str, durations: list[int]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    codes, bucket = _collect_codes(cfg)

    pred_rows = []
    embed_rows = []

    for duration in durations:
        for code in codes:
            slug = code_slug(code)
            task_labels_dir = str(task_set_dir / str(duration) / slug)

            if not Path(task_labels_dir).is_dir():
                logger.warning(
                    f"Missing task_labels_dir for code={code}, duration={duration}: "
                    f"{task_labels_dir} (skipping)"
                )
                continue

            train_cfg.datamodule.config.task_labels_dir = task_labels_dir
            D = instantiate(train_cfg.datamodule)

            pred_batches = trainer.predict(model=M, datamodule=D, ckpt_path=None)

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

            pred_rows.append(
                pl.DataFrame(
                    {
                        "subject_id": subject_id,
                        "prediction_time": prediction_time,
                        "occurs_probs": occurs_probs,
                    }
                ).with_columns(
                    pl.lit(code).alias("code"),
                    pl.lit(bucket[code]).alias("bucket"),
                    pl.lit(duration).alias("duration_days"),
                    pl.lit(model_name).alias("model"),
                )
            )
            embed_rows.append(
                pl.DataFrame(
                    {
                        "subject_id": subject_id,
                        "prediction_time": prediction_time,
                        "code": [code] * len(subject_id),
                    }
                ).with_columns(
                    pl.Series("embedding", query_embeds),
                    pl.lit(bucket[code]).alias("bucket"),
                    pl.lit(duration).alias("duration_days"),
                    pl.lit(model_name).alias("model"),
                )
            )

    pred_df = pl.concat(pred_rows, how="vertical") if pred_rows else pl.DataFrame()
    embed_df = pl.concat(embed_rows, how="vertical") if embed_rows else pl.DataFrame()
    return pred_df, embed_df


@hydra.main(version_base="1.3", config_path="./eval_suite/conf", config_name="eval_config.yaml")
def main(cfg: DictConfig) -> None:
    model_run_dirs = list(cfg.model_run_dirs) if cfg.get("model_run_dirs") else [cfg.model_run_dir]
    durations = list(cfg.durations)
    task_set_dir = Path(cfg.task_set_dir)

    if not task_set_dir.is_dir():
        raise NotADirectoryError(f"{task_set_dir} is not a directory")

    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    hc = HydraConfig.get()
    eval_codes_choice_str = hc.runtime.choices["eval_codes"]

    ckpt_names = list(cfg.ckpt_names) if cfg.get("ckpt_names") else [cfg.get("ckpt_path")]

    for model_run_dir in model_run_dirs:
        for ckpt_name in ckpt_names:
            model_name = f"{_model_name(model_run_dir)}/{ckpt_name}"
            logger.info(f"=== Evaluating model: {model_name} ({model_run_dir}) ===")

            out_dir = Path(model_run_dir) / "eval" / ckpt_name
            out_dir.mkdir(parents=True, exist_ok=True)

            train_cfg, M, trainer = _setup_model(model_run_dir, ckpt_name=ckpt_name)

            if cfg.mode == "predict":
                pred_df, embed_df = _run_predict(
                    cfg, train_cfg, M, trainer, task_set_dir, model_name, durations
                )
                if pred_df.is_empty():
                    logger.warning(f"No predictions generated for {model_name} — all codes were skipped.")
                    continue
                out_fp = out_dir / f"eval_preds_{eval_codes_choice_str}_{timestamp}.parquet"
                embed_fp = out_dir / f"eval_embeds_{eval_codes_choice_str}_{timestamp}.parquet"
                pred_df.write_parquet(out_fp)
                embed_df.write_parquet(embed_fp)
                logger.info(f"Saved predictions to {out_fp}")
                logger.info(f"Saved embeddings to {embed_fp}")
            else:
                test_df = _run_test(
                    cfg, train_cfg, M, trainer, task_set_dir, model_name, durations, ckpt_name
                )
                if test_df.is_empty():
                    logger.warning(f"No test results generated for {model_name}.")
                    continue
                out_fp = out_dir / f"eval_aucs_{cfg.split}_{timestamp}.parquet"
                if out_fp.exists() and not cfg.do_overwrite:
                    logger.info(f"Output exists at {out_fp}. Set do_overwrite=true to overwrite.")
                    continue
                test_df.write_parquet(out_fp)
                logger.info(f"Saved test results to {out_fp}")


if __name__ == "__main__":
    main()
