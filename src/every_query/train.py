import builtins
import hashlib
import json
import logging
import os

NUM_CPUS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
FILES_AT_ONCE = 10
THREADS_PER_FILE = max(1, NUM_CPUS // FILES_AT_ONCE)
os.environ["POLARS_MAX_THREADS"] = str(THREADS_PER_FILE)
os.environ["OMP_NUM_THREADS"] = str(THREADS_PER_FILE)
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _collate_shard(
    file_name: str,
    split: str,
    write_dir: str,
    task_dir: str,
    durations: list,
    codes: list,
    sample_times_per_subject: int,
    seed: int,
) -> None:
    out_path = f"{write_dir}/{split}/{file_name}"
    if os.path.exists(out_path):
        logger.info(f"Skipping shard. Already collated at {out_path}.")
        return

    logger.info(f"Collating {out_path}")
    n_codes = len(codes)
    rng = np.random.default_rng(seed)

    # --- PASS 1: Sample prediction times per subject ---
    # Prediction times are identical across durations, so read only the first.
    d0 = durations[0]
    base_idx = (
        pl.scan_parquet(f"{task_dir}/{d0}/{split}/{file_name}")
        .select("subject_id", "prediction_time")
        .collect()
    )

    if base_idx.is_empty():
        return

    # Sample up to sample_times_per_subject rows per subject.
    # Sort after group_by to ensure deterministic row order for reproducible RNG assignment.
    sampled = (
        base_idx.sample(fraction=1, shuffle=True, seed=seed)
        .group_by("subject_id")
        .head(sample_times_per_subject)
        .sort("subject_id", "prediction_time")
    )

    n_sampled = len(sampled)

    # --- PASS 2: Assign random duration and code to each sampled row ---
    sampled_labels = sampled.with_columns(
        pl.Series("duration_days", rng.choice(durations, size=n_sampled)),
        pl.Series("query", [codes[i] for i in rng.integers(0, n_codes, size=n_sampled)]),
    )
    del base_idx, sampled

    # --- PASS 3: Selective Extraction ---
    # We only load the columns and rows we actually selected in Pass 2.
    batch_results = []
    for d in durations:
        kept_for_duration = sampled_labels.filter(pl.col("duration_days") == d)
        if kept_for_duration.is_empty():
            continue

        unique_times = kept_for_duration.select("subject_id", "prediction_time").unique()
        needed_codes = kept_for_duration.get_column("query").unique().to_list()

        # Only read the subset of 10k columns that were actually sampled
        load_cols = ["subject_id", "prediction_time", "censored", *needed_codes]

        wide_df = (
            pl.scan_parquet(f"{task_dir}/{d}/{split}/{file_name}")
            .select(load_cols)
            .join(unique_times.lazy(), on=["subject_id", "prediction_time"], how="inner")
            .collect()
        )

        narrow_df = (
            wide_df.with_columns(pl.lit(d).alias("duration_days"))
            .unpivot(
                index=["subject_id", "prediction_time", "censored", "duration_days"],
                variable_name="query",
                value_name="occurs",
            )
            # Inner join ensures we only keep the specific (Time, Code) pairs from Pass 2
            .join(
                kept_for_duration, on=["subject_id", "prediction_time", "duration_days", "query"], how="inner"
            )
            .rename({"censored": "boolean_value"})
            .with_columns(pl.col("occurs").fill_null(False))
        )
        batch_results.append(narrow_df)

    # --- PASS 4: Final Write ---
    if batch_results:
        shard = pl.concat(batch_results)
        # Sort before shuffle for reproducibility (joins/unpivot don't guarantee order)
        shard = shard.sort("subject_id", "prediction_time", "duration_days", "query")
        # Final shuffle so the file isn't ordered by duration
        shard = shard.sample(fraction=1, shuffle=True, seed=seed)
        shard.write_parquet(out_path)


def collate_tasks(cfg: DictConfig) -> str:
    task_dir = cfg.query.task_dir
    durations_path = f"{task_dir}/sampled_durations.json"
    with open(durations_path) as f:
        durations = json.load(f)

    task_str = f"{'|'.join(sorted(cfg.query.codes))}_{'|'.join(str(d) for d in sorted(durations))}"
    hash_hex = hashlib.md5(task_str.encode()).hexdigest()
    write_dir = f"{task_dir}/collated/{hash_hex}"

    first_duration = durations[0]
    codes = list(cfg.query.codes)
    seed = cfg.get("seed", 1)
    sample_times_per_subject = cfg.query.sample_times_per_subject

    # Eval tasks generated in separate file
    for split in [train_split, tuning_split]:
        os.makedirs(f"{write_dir}/{split}", exist_ok=True)
        file_names = os.listdir(f"{task_dir}/{first_duration}/{split}")

        with ThreadPoolExecutor(max_workers=FILES_AT_ONCE) as executor:
            futures = [
                executor.submit(
                    _collate_shard,
                    file_name,
                    split,
                    write_dir,
                    task_dir,
                    durations,
                    codes,
                    sample_times_per_subject,
                    seed,
                )
                for file_name in file_names
            ]
            for future in as_completed(futures):
                future.result()

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
