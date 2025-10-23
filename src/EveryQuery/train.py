from meds import DataSchema
from dataset import EveryQueryPytorchDataset
from meds_torchdata import MEDSTorchDataConfig
from omegaconf import DictConfig, OmegaConf, ListConfig
import hydra, ipdb
from model import EveryQueryModel
from lightning_module import EveryQueryLightningModule
from typing import Any
from pathlib import Path
import shutil
import hashlib
import polars as pl
import os
from meds import train_split, tuning_split, held_out_split
import torch
from hydra.utils import instantiate
import logging 
from lightning.pytorch import seed_everything

logger = logging.getLogger(__name__)

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

def validate_resume_directory(output_dir: Path, cfg: DictConfig):

    old_cfg_fp = output_dir / "config.yaml"
    if not old_cfg_fp.is_file():
        raise FileNotFoundError(f"Configuration file {old_cfg_fp} does not exist in the output directory.")

    old_cfg = OmegaConf.load(old_cfg_fp)

    old_cfg = OmegaConf.to_container(old_cfg, resolve=True)
    new_cfg = OmegaConf.to_container(cfg, resolve=True)

    differences = diff_configs(new_cfg, old_cfg)

    err_lines = []
    for key, diff in differences.items():
        if key in ALLOWED_DIFFERENCE_KEYS:
            continue
        err_lines.append(f"  - key '{key}' {diff}")

    if err_lines:
        err_lines_str = "\n".join(err_lines)
        raise ValueError(
            f"The configuration in the output directory does not match the input:\n{err_lines_str}"
        )

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

def collate_tasks(cfg: DictConfig) -> None:

    read_dir = f"{cfg.query.task_dir}/all"

    task_str = f"{"|".join(sorted(cfg.query.codes))}_{cfg.query.sample_times_per_subject}"
    hash_hex = hashlib.md5(task_str.encode()).hexdigest()
    write_dir = f"{cfg.query.task_dir}/collated/{hash_hex}"
    
    for split in [train_split, tuning_split, held_out_split]:
        os.makedirs(f"{write_dir}/{split}", exist_ok=True)
        for file_name in os.listdir(f"{read_dir}/{split}"):
            f = f"{write_dir}/{split}/{file_name}"
            if os.path.exists(f):
                logger.info(f"Skipping shard. Already collated at {f}.")
            shard = (
                pl.read_parquet(source=f"{read_dir}/{split}/{file_name}", columns=['subject_id', 'prediction_time', 'censored'] + cfg.query.codes)
                .unpivot(index=['subject_id', 'prediction_time', 'censored'], variable_name="query", value_name="occurs")
                .rename({'censored':'boolean_value'})
                .with_columns(pl.col('occurs').fill_null(False)) 
                .sample(fraction=1, shuffle=True, seed=cfg.get("seed", 1))
                .group_by('subject_id')
                .head(cfg.query.sample_times_per_subject)
            )
            shard.write_parquet(f)
        logger.info(f"Tasks collated for {split} and written to {hash_hex}.")

    return write_dir    


@hydra.main(version_base="1.3", config_path='', config_name='config.yaml')
def main(cfg: DictConfig) -> float | None:

    if not isinstance(cfg.query.codes, ListConfig):
        raise ValueError("query.codes must be a list")
    
    task_dir = collate_tasks(cfg)
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
            validate_resume_directory(output_dir, cfg)
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

    trainer.fit(**trainer_kwargs)

    best_ckpt_path = Path(trainer.checkpoint_callback.best_model_path)
    if not best_ckpt_path.is_file():
        raise ValueError("No best checkpoint reported.")
    else:
        for log in trainer.loggers:
            log.log_hyperparams({'best_ckpt_path':best_ckpt_path})

    output_fp = Path(cfg.output_dir) / "best_model.ckpt"
    shutil.copyfile(best_ckpt_path, output_fp)

    best_score = trainer.checkpoint_callback.best_model_score

    logger.info(f"Best checkpoint (with score {best_score:.2f}) copied to {output_fp!s}.")

if __name__ == "__main__":
    main()

