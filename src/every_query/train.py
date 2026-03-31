import builtins
import hashlib
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


@OmegaConfResolver(replace=True)
def list_len(x):
    return builtins.len(x)


@OmegaConfResolver(replace=True)
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
    """Resolve all interpolations in *cfg* and write the result to *fp*.

    Returns ``True`` on success, ``False`` (with a warning) on failure.

    Examples:
        >>> with tempfile.TemporaryDirectory() as d:
        ...     cfg = OmegaConf.create({"a": 1, "b": "${a}"})
        ...     save_resolved_config(cfg, Path(d) / "out.yaml")
        True

        Interpolations are fully expanded in the saved file:

        >>> with tempfile.TemporaryDirectory() as d:
        ...     cfg = OmegaConf.create({"a": 1, "b": "${a}"})
        ...     fp = Path(d) / "out.yaml"
        ...     _ = save_resolved_config(cfg, fp)
        ...     OmegaConf.load(fp).b
        1

        Unresolvable interpolation returns ``False``:

        >>> with tempfile.TemporaryDirectory() as d:
        ...     cfg = OmegaConf.create({"a": "${missing}"})
        ...     save_resolved_config(cfg, Path(d) / "out.yaml")
        False
    """
    try:
        # Create a copy and resolve all interpolations
        resolved_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        OmegaConf.save(resolved_cfg, fp)
        return True
    except Exception as e:
        logger.warning(f"Could not save resolved config: {e}")
        return False


def find_checkpoint_path(output_dir: Path) -> Path | None:
    """Return the latest checkpoint under ``output_dir/checkpoints``, or ``None``.

    Prefers ``last.ckpt``; otherwise picks the file with the highest
    ``(epoch, step)`` pair.

    Raises:
        NotADirectoryError: If the checkpoints path is a regular file.

    Examples:
        No checkpoints directory:

        >>> with tempfile.TemporaryDirectory() as d:
        ...     find_checkpoint_path(Path(d)) is None
        True

        Empty checkpoints directory:

        >>> with tempfile.TemporaryDirectory() as d:
        ...     (Path(d) / "checkpoints").mkdir()
        ...     find_checkpoint_path(Path(d)) is None
        True

        ``last.ckpt`` is preferred when present:

        >>> with tempfile.TemporaryDirectory() as d:
        ...     ckpt = Path(d) / "checkpoints"; ckpt.mkdir()
        ...     (ckpt / "last.ckpt").touch()
        ...     find_checkpoint_path(Path(d)) == ckpt / "last.ckpt"
        True

        ``last.ckpt`` takes priority even when epoch checkpoints exist:

        >>> with tempfile.TemporaryDirectory() as d:
        ...     ckpt = Path(d) / "checkpoints"; ckpt.mkdir()
        ...     (ckpt / "last.ckpt").touch()
        ...     (ckpt / "epoch=5-step=999.ckpt").touch()
        ...     find_checkpoint_path(Path(d)) == ckpt / "last.ckpt"
        True

        Falls back to the latest ``epoch=*-step=*.ckpt``:

        >>> with tempfile.TemporaryDirectory() as d:
        ...     ckpt = Path(d) / "checkpoints"; ckpt.mkdir()
        ...     (ckpt / "epoch=0-step=100.ckpt").touch()
        ...     (ckpt / "epoch=1-step=50.ckpt").touch()
        ...     (ckpt / "epoch=1-step=200.ckpt").touch()
        ...     find_checkpoint_path(Path(d)) == ckpt / "epoch=1-step=200.ckpt"
        True

        Non-matching files in the directory are ignored:

        >>> with tempfile.TemporaryDirectory() as d:
        ...     ckpt = Path(d) / "checkpoints"; ckpt.mkdir()
        ...     (ckpt / "some_other_file.txt").touch()
        ...     find_checkpoint_path(Path(d)) is None
        True

        Raises when the checkpoints path is a file:

        >>> with tempfile.TemporaryDirectory() as d:
        ...     (Path(d) / "checkpoints").touch()
        ...     find_checkpoint_path(Path(d))
        Traceback (most recent call last):
            ...
        NotADirectoryError: ...
    """
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

    # --- PASS 1: Generate the Index Pool ---
    # We collect all available timestamps across all durations.
    index_frames = []
    for d in durations:
        idx = (
            pl.scan_parquet(f"{task_dir}/{d}/{split}/{file_name}")
            .select("subject_id", "prediction_time")
            .with_columns(pl.lit(d).alias("duration_days"))
            .collect()
        )
        index_frames.append(idx)

    if not index_frames:
        return

    all_indices = pl.concat(index_frames)
    del index_frames

    # --- PASS 2: Sample Labels Statistically Identical to Original ---
    # To match the original distribution, we must sample from (Time * Code) space.
    # Instead of exploding 10k codes (memory heavy), we assign random code indices.

    # We repeat the indices enough times to ensure we can hit the sample target
    # even if a subject has very few timestamps.
    oversample_factor = min(n_codes, 10)

    sampled_labels = (
        all_indices.select(pl.all(), pl.lit(list(range(oversample_factor))).alias("_rep"))
        .explode("_rep")
        .with_columns(
            # Randomly assign a code index to every "slot"
            pl.lit(rng.integers(0, n_codes, size=len(all_indices) * oversample_factor)).alias("code_idx")
        )
        # Global shuffle across all times/durations/codes for this shard
        .sample(fraction=1, shuffle=True, seed=seed)
        .group_by("subject_id")
        .head(sample_times_per_subject)
        .with_columns(
            # Map index back to the actual column name
            pl.col("code_idx").map_elements(lambda i: codes[i], return_dtype=pl.String).alias("query")
        )
        .drop("_rep", "code_idx")
    )
    del all_indices

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
        # Final shuffle so the file isn't ordered by duration
        shard = shard.sample(fraction=1, shuffle=True, seed=seed)
        shard.write_parquet(out_path)


def collate_tasks(cfg: DictConfig) -> str:
    task_dir = cfg.query.task_dir
    durations_path = f"{task_dir}/sampled_durations.json"
    if os.path.exists(durations_path):
        import json

        with open(durations_path) as f:
            durations = json.load(f)
    else:
        durations = list(range(cfg.query.duration_min, cfg.query.duration_max))

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
