import builtins
import hashlib
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import polars as pl
import torch
from hydra.utils import instantiate
from lightning.pytorch import seed_everything
from meds import train_split, tuning_split
from MEDS_transforms.configs.utils import OmegaConfResolver
from omegaconf import DictConfig, ListConfig, OmegaConf

logger = logging.getLogger(__name__)


@OmegaConfResolver
def list_len(x):
    return builtins.len(x)


@OmegaConfResolver
def int_prod(x: int, y: int) -> int:
    """Returns the closest integer to the product of x and y (available as an OmegaConf resolver).

    Examples:
        >>> int_prod(2, 3)
        6
        >>> int_prod(2, 3.5)
        7
        >>> int_prod(2.49, 3)
        7
    """
    return round(x * y)


def values_as_list(**kwargs) -> list[Any]:
    return list(kwargs.values())


def save_resolved_config(cfg: DictConfig, fp: Path) -> bool:
    try:
        # Create a copy and resolve all interpolations
        resolved_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        OmegaConf.save(resolved_cfg, fp)
        return True
    except Exception as e:
        logger.warning(f"Could not save resolved config: {e}")
        return False


def find_checkpoint_path(output_dir: Path) -> Path | None:
    checkpoints_dir = output_dir / "checkpoints"

    if checkpoints_dir.is_file():
        raise NotADirectoryError(f"Checkpoints directory {checkpoints_dir} is a file, not a directory.")
    elif not checkpoints_dir.exists():
        return None

    last_ckpt = checkpoints_dir / "last.ckpt"
    if last_ckpt.is_file():
        return last_ckpt

    checkpoint_fps = list(checkpoints_dir.glob("epoch=*-step=*.ckpt"))
    if not checkpoint_fps:
        return None

    def get_epoch(fp: Path) -> int:
        return int(fp.stem.split("-")[0].split("=")[1])

    def get_step(fp: Path) -> int:
        return int(fp.stem.split("-")[1].split("=")[1])

    sorted_checkpoints = sorted(checkpoint_fps, key=lambda fp: (get_epoch(fp), get_step(fp)))

    return sorted_checkpoints[-1] if sorted_checkpoints else None


def collate_tasks(cfg: DictConfig) -> str:
    """Build training task rows with geometric query generation.

    For each shard, pre-generates ``len(cfg.query.codes)`` query definitions
    (matching the row count that the old unpivot approach produced).  Each
    definition draws ``k ~ Geometric(geometric_p)`` codes, caps at
    ``max_query_codes``, picks a random ANY/ALL quantifier, and computes
    ``occurs`` via ``pl.any_horizontal`` / ``pl.all_horizontal``.
    """
    task_dir = cfg.query.task_dir
    durations = list(range(cfg.query.duration_min, cfg.query.duration_max))

    geometric_p = cfg.query.geometric_p
    max_query_codes = cfg.query.max_query_codes
    seed = cfg.get("seed", 1)
    num_queries = cfg.query.queries_per_shard

    task_str = (
        f"{'|'.join(sorted(cfg.query.codes))}_{'|'.join(str(d) for d in sorted(durations))}"
        f"_gp{geometric_p}_mqc{max_query_codes}_qps{num_queries}_seed{seed}"
    )
    hash_hex = hashlib.md5(task_str.encode()).hexdigest()
    write_dir = f"{task_dir}/collated/{hash_hex}"

    first_duration = durations[0]
    all_codes = list(cfg.query.codes)
    rng = np.random.default_rng(seed)

    for split in [train_split, tuning_split]:
        os.makedirs(f"{write_dir}/{split}", exist_ok=True)

        for file_name in os.listdir(f"{task_dir}/{first_duration}/{split}"):
            f = f"{write_dir}/{split}/{file_name}"
            logger.info(f"Collating {f}")

            if os.path.exists(f):
                logger.info(f"Skipping shard. Already collated at {f}.")
                continue

            query_defs: list[tuple[str, list[str]]] = []
            for _ in range(num_queries):
                k = int(min(rng.geometric(geometric_p), max_query_codes, len(all_codes)))
                codes_subset = rng.choice(all_codes, size=k, replace=False).tolist()
                quantifier = str(rng.choice(["ANY", "ALL"]))
                query_defs.append((quantifier, codes_subset))

            duration_shards = []
            for duration in durations:
                wide_df = pl.read_parquet(
                    source=f"{task_dir}/{duration}/{split}/{file_name}",
                    columns=["subject_id", "prediction_time", "censored", *all_codes],
                ).with_columns(pl.lit(duration).alias("duration_days"))

                query_frames = []
                for quantifier, codes_subset in query_defs:
                    code_exprs = [pl.col(c) for c in codes_subset]
                    if quantifier == "ANY":
                        occurs_expr = pl.any_horizontal(*code_exprs)
                    else:
                        occurs_expr = pl.all_horizontal(*code_exprs)

                    query_frame = wide_df.select(
                        "subject_id",
                        "prediction_time",
                        "censored",
                        "duration_days",
                        occurs_expr.alias("occurs"),
                        pl.lit(quantifier).alias("quantifier"),
                        # Broadcasts a single list value (e.g. ["A", "B"]) as a constant column across
                        # all rows, tagging each row with the codes that define this query.
                        # Wrapping a Python list in a one-element Series of dtype List, then calling
                        # .first() collapses the length-1 Series to a scalar so Polars can broadcast it.
                        # This is brittle: future Polars versions may change lit(Series) semantics or
                        # alignment rules, and the .first() trick silently masks shape mismatches.
                        pl.lit(pl.Series("query_codes", [codes_subset])).first().alias("query_codes"),
                    )
                    query_frames.append(query_frame)

                duration_shards.append(pl.concat(query_frames))

            shard = (
                pl.concat(duration_shards)
                .rename({"censored": "boolean_value"})
                .with_columns(pl.col("occurs").fill_null(False))
                .sample(fraction=1, shuffle=True, seed=seed)
                .group_by(["subject_id"])
                .head(cfg.query.sample_times_per_subject)
            )
            shard.write_parquet(f)
        logger.info(f"Tasks collated for {split} and written to {hash_hex}.")

    return write_dir


@hydra.main(version_base="1.3", config_path="", config_name="config.yaml")
def main(cfg: DictConfig) -> float | None:
    if not isinstance(cfg.query.codes, ListConfig):
        raise ValueError("query.codes must be a list")

    task_dir = collate_tasks(cfg)
    if cfg.only_preprocess:
        print("Collate tasks complete. Exiting.")
        sys.exit(0)

    cfg.datamodule.config.task_labels_dir = task_dir

    if cfg.do_overwrite and cfg.do_resume:
        logger.warning(
            "Both `do_overwrite` and `do_resume` are set to True. "
            "Only `do_overwrite` will be used, and the output directory will be cleared."
        )

    output_dir = Path(cfg.output_dir)
    if output_dir.is_file():
        raise NotADirectoryError(f"Output directory {output_dir} is a file, not a directory.")
    os.makedirs(output_dir, exist_ok=True)

    cfg_path = output_dir / "config.yaml"
    ckpt_path = None
    if cfg_path.exists():
        if cfg.do_overwrite:
            logger.info(f"Overwriting existing output directory {output_dir}.")
            shutil.rmtree(output_dir, ignore_errors=True)
        elif cfg.do_resume:
            logger.info(f"Resuming training in existing output directory {output_dir}.")
            ckpt_path = find_checkpoint_path(output_dir)
        else:
            raise FileExistsError(
                f"Output directory {output_dir} already exists and is populated. "
                "Use `do_overwrite` or `do_resume` to proceed."
            )
    else:
        OmegaConf.save(cfg, output_dir / "config.yaml")
        save_resolved_config(cfg, output_dir / "resolved_config.yaml")

    logger.info("Setting torch float32 matmul precision to 'medium'.")
    torch.set_float32_matmul_precision("medium")

    D = instantiate(cfg.datamodule)
    logger.info(f"Train dataset contains {len(D.train_dataloader().dataset)} datapoints")

    M = hydra.utils.instantiate(cfg.lightning_module)

    if M.model.do_demo or cfg.get("seed", None):
        seed_everything(cfg.get("seed", 1), workers=True)

    trainer = instantiate(cfg.trainer)

    trainer_kwargs = {"model": M, "datamodule": D}
    if ckpt_path:
        logger.info(f"Trying to resume training from checkpoint {ckpt_path}.")
        trainer_kwargs["ckpt_path"] = ckpt_path
    print("fitting model")
    trainer.fit(**trainer_kwargs)

    best_ckpt_path = Path(trainer.checkpoint_callback.best_model_path)
    if not best_ckpt_path.is_file():
        raise ValueError("No best checkpoint reported.")
    else:
        for log in trainer.loggers:
            log.log_hyperparams({"best_ckpt_path": best_ckpt_path})

    output_fp = Path(cfg.output_dir) / "best_model.ckpt"
    shutil.copyfile(best_ckpt_path, output_fp)

    best_score = trainer.checkpoint_callback.best_model_score

    logger.info(f"Best checkpoint (with score {best_score:.2f}) copied to {output_fp!s}.")


if __name__ == "__main__":
    main()
