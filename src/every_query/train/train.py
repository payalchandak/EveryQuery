import builtins
import logging
import os
import shutil
from importlib.resources import files
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.utils import instantiate
from lightning.pytorch import seed_everything
from MEDS_transforms.configs.utils import OmegaConfResolver
from omegaconf import DictConfig, OmegaConf, open_dict

from every_query.train.resume_check import validate_resume_directory

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
    # Drop None so an optional callback can be toggled off with `<name>: null` instead of
    # deleting/commenting its config block.
    return [v for v in kwargs.values() if v is not None]


def save_resolved_config(cfg: DictConfig, fp: Path) -> bool:
    """Resolve all interpolations in *cfg* and write the result to *fp*.

    Returns ``True`` on success, ``False`` (with a warning) on failure.

    Examples:
        >>> import tempfile
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

        >>> import tempfile
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


def _is_wandb_logger(logger_cfg: Any) -> bool:
    """Return ``True`` if *logger_cfg* is a wandb-shaped logger node.

    A disabled (``false`` / ``null``) or non-wandb logger returns ``False`` so that
    ``WANDB_ENTITY`` is only required when a wandb logger is actually instantiated.

    Examples:
        >>> _is_wandb_logger(False)
        False
        >>> _is_wandb_logger(None)
        False
        >>> _is_wandb_logger(OmegaConf.create({"_target_": "lightning.pytorch.loggers.CSVLogger"}))
        False
        >>> _is_wandb_logger(
        ...     OmegaConf.create({"_target_": "pytorch_lightning.loggers.wandb.WandbLogger"})
        ... )
        True
    """
    if not logger_cfg or not isinstance(logger_cfg, DictConfig):
        return False
    return "WandbLogger" in str(logger_cfg.get("_target_", ""))


def validate_training_config(cfg: DictConfig) -> None:
    """Validate the *resolved* training config, raising a clear error on a missing/bad value.

    Replaces the old blind env-var presence gate (#184).  Because this runs after Hydra has
    composed the config, a CLI override of a node (e.g. ``datamodule.config.task_labels_dir=/p``)
    means the backing ``${oc.env:...}`` interpolation never evaluates and the env var is not
    required.  Each error message names both the config node and the env var that backs it.

    Checks:
      * ``datamodule.config.tensorized_cohort_dir`` / ``datamodule.config.task_labels_dir`` —
        must resolve to an existing directory (these are read inputs).
      * ``output_dir`` — must resolve to a non-empty path (write target; created later, so it
        need not pre-exist).
      * wandb ``entity`` — required only when ``trainer.logger`` is wandb-shaped.

    Raises:
        ValueError: If a required path/value is missing or empty.
        NotADirectoryError: If a required input path does not exist or is not a directory.
    """
    ds_cfg = cfg.datamodule.config
    for node, env_var in (
        ("tensorized_cohort_dir", "TENSORIZED_COHORT_DIR"),
        ("task_labels_dir", "TRAINING_TASKS_DIR"),
    ):
        value = ds_cfg.get(node)
        if not value:
            raise ValueError(
                f"datamodule.config.{node} is unset. Pass it as a CLI override "
                f"(datamodule.config.{node}=/path, typically =${env_var})."
            )
        if not Path(value).is_dir():
            raise NotADirectoryError(
                f"datamodule.config.{node} ({value!r}, from ${env_var}) is not an existing directory."
            )

    # ``output_dir`` is a required base (``???``); supply it with ``output_dir=/path``.  An unset
    # value surfaces as Hydra's "Missing mandatory value" error on access, but guard explicitly too
    # for callers that build the config without Hydra (e.g. the tests).
    if not cfg.get("output_dir"):
        raise ValueError("output_dir is unset. Pass output_dir=/path.")

    # main() writes to trainer.default_root_dir (the Hydra-resolved per-run/per-job dir), not
    # output_dir directly.  Validate the dir actually used so a stray default_root_dir= override
    # can't pass this gate while artifacts land somewhere unintended.
    if not cfg.trainer.get("default_root_dir"):
        raise ValueError("trainer.default_root_dir is unset.")

    if _is_wandb_logger(cfg.trainer.get("logger")) and not cfg.trainer.logger.get("entity"):
        raise ValueError(
            "trainer.logger.entity is unset for a wandb logger. Pass "
            "trainer.logger.entity=<entity> or export $WANDB_ENTITY "
            "(or disable the logger with trainer.logger=false)."
        )


def resolve_vocab_size(cfg: DictConfig, run_cfg_path: Path) -> None:
    """Set ``lightning_module.model.vocab_size`` in-place to the cohort's code vocabulary.

    Sizes the embedding table to the dataset rather than to ModernBERT-base's 50368-entry text
    vocabulary, which is what `EveryQueryModel` inherits when nobody tells it otherwise (#283).
    `MEDSTorchDataConfig.vocab_size` reads ``max(code/vocab_index) + 1`` off the code metadata
    parquet, so instantiating the data config alone is enough — no dataset build needed.

    Three things suppress the derivation:

    - An explicit ``lightning_module.model.vocab_size``.  That's the escape hatch for training
      against a vocabulary that isn't this cohort's; since the key isn't declared in
      ``config.yaml`` it takes Hydra's append form, ``+lightning_module.model.vocab_size=N``.
    - An explicit ``config_overrides.vocab_size`` — the pre-#283 idiom.  Honoured (with a
      warning) rather than silently replaced, so existing user yaml keeps training the table it
      asked for.
    - Resuming a run whose saved config predates #283.  Such a run's checkpoint holds a
      ``[50368, H]`` table; deriving would rebuild the model at the cohort's width and
      ``trainer.fit(ckpt_path=...)`` would die on a state-dict shape mismatch.  Leaving the key
      unset reproduces the table the checkpoint actually contains *and* keeps the config
      diffable against the old one, so `validate_resume_directory` still passes.  A run started
      after #283 has the key saved and it is read back verbatim.

    Args:
        cfg: The training config, mutated in place.
        run_cfg_path: ``<run_dir>/config.yaml`` — the config of the run being resumed, if any.

    Examples:
        >>> import tempfile
        >>> import polars as pl
        >>> def cfg_for(cohort_dir, **model):
        ...     return OmegaConf.create({
        ...         "do_resume": True, "do_overwrite": False,
        ...         "datamodule": {"config": {
        ...             "tensorized_cohort_dir": str(cohort_dir), "max_seq_len": 10,
        ...             "_target_": "meds_torchdata.config.MEDSTorchDataConfig",
        ...         }},
        ...         "lightning_module": {"model": model},
        ...     })
        >>> tmp = Path(tempfile.mkdtemp())
        >>> (tmp / "metadata").mkdir()
        >>> pl.DataFrame({"code/vocab_index": [1, 2, 3]}).write_parquet(
        ...     tmp / "metadata" / "codes.parquet")

        With no run dir to resume, the value comes from the cohort:

        >>> cfg = cfg_for(tmp)
        >>> resolve_vocab_size(cfg, tmp / "nonexistent.yaml")
        >>> cfg.lightning_module.model.vocab_size
        4

        Resuming a run whose saved config predates #283, the key stays absent:

        >>> old_cfg_path = tmp / "config.yaml"
        >>> OmegaConf.save(OmegaConf.create({"lightning_module": {"model": {}}}), old_cfg_path)
        >>> cfg = cfg_for(tmp)
        >>> resolve_vocab_size(cfg, old_cfg_path)
        >>> "vocab_size" in cfg.lightning_module.model
        False

        Resuming a post-#283 run, the saved value is read back rather than re-derived:

        >>> OmegaConf.save(
        ...     OmegaConf.create({"lightning_module": {"model": {"vocab_size": 9}}}), old_cfg_path)
        >>> cfg = cfg_for(tmp)
        >>> resolve_vocab_size(cfg, old_cfg_path)
        >>> cfg.lightning_module.model.vocab_size
        9

        An explicit value is left alone, whichever way it was set:

        >>> cfg = cfg_for(tmp, vocab_size=61312)
        >>> resolve_vocab_size(cfg, tmp / "nonexistent.yaml")
        >>> cfg.lightning_module.model.vocab_size
        61312
        >>> cfg = cfg_for(tmp, config_overrides={"vocab_size": 100})
        >>> resolve_vocab_size(cfg, tmp / "nonexistent.yaml")
        >>> "vocab_size" in cfg.lightning_module.model
        False
    """
    model_cfg = cfg.lightning_module.model

    if model_cfg.get("vocab_size", None) is not None:
        return

    overridden = (model_cfg.get("config_overrides") or {}).get("vocab_size")
    if overridden is not None:
        logger.warning(
            f"Using config_overrides.vocab_size={overridden} rather than the cohort's code "
            f"vocabulary.  Prefer `+lightning_module.model.vocab_size={overridden}`, which also "
            f"pins the embedding's padding_idx to the index batches are padded with."
        )
        return

    if cfg.do_resume and not cfg.do_overwrite and run_cfg_path.is_file():
        resumed = OmegaConf.select(OmegaConf.load(run_cfg_path), "lightning_module.model.vocab_size")
        if resumed is None:
            logger.info(
                f"Resuming a run started before #283 ({run_cfg_path} sets no vocab_size); keeping "
                f"the checkpoint's ModernBERT-sized embedding table rather than deriving one."
            )
            return
        vocab_size = resumed
    else:
        vocab_size = instantiate(cfg.datamodule.config).vocab_size
        logger.info(f"Setting model vocab_size to the dataset's code vocabulary: {vocab_size}")

    with open_dict(cfg):
        model_cfg.vocab_size = vocab_size


def _init_env() -> None:
    """Configure thread counts for polars/OMP from the SLURM/system environment."""
    num_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    threads_per_file = max(1, num_cpus // 10)
    os.environ["POLARS_MAX_THREADS"] = str(threads_per_file)
    os.environ["OMP_NUM_THREADS"] = str(threads_per_file)


CONFIGS = str(files("every_query") / "train" / "configs")


@hydra.main(version_base="1.3", config_path=CONFIGS, config_name="config.yaml")
def main(cfg: DictConfig) -> float | None:
    _init_env()
    validate_training_config(cfg)

    if cfg.do_overwrite and cfg.do_resume:
        logger.warning(
            "Both `do_overwrite` and `do_resume` are set to True. "
            "Only `do_overwrite` will be used, and the output directory will be cleared."
        )

    # The per-run/per-job dir Hydra resolved (run.dir for a single run, sweep.dir/subdir for a sweep
    # job) — *not* cfg.output_dir, which is only the shared base.  Reading the resolved dir keeps
    # sweep jobs from rmtree-ing/writing to the common base and colliding.
    run_dir = Path(cfg.trainer.default_root_dir)
    if run_dir.is_file():
        raise NotADirectoryError(f"Run directory {run_dir} is a file, not a directory.")

    cfg_path = run_dir / "config.yaml"

    # Must land before both `validate_resume_directory` and the config write below, so the
    # resume diff compares like against like and ``resolved_config.yaml`` records the size the
    # run actually used.  Needs `cfg_path` to tell a pre-#283 run dir from a post-#283 one.
    resolve_vocab_size(cfg, cfg_path)

    ckpt_path = None
    if cfg_path.exists():
        if cfg.do_overwrite:
            logger.info(f"Overwriting existing run directory {run_dir}.")
            shutil.rmtree(run_dir, ignore_errors=True)
        elif cfg.do_resume:
            logger.info(f"Resuming training in existing run directory {run_dir}.")
            validate_resume_directory(run_dir, cfg)
            ckpt_path = find_checkpoint_path(run_dir)
        else:
            raise FileExistsError(
                f"Run directory {run_dir} already exists and is populated. "
                "Use `do_overwrite` or `do_resume` to proceed."
            )

    # Ensure run_dir exists *after* any overwrite rmtree above, then write the config for this
    # run.  On resume (without overwrite) we keep the original run's config untouched so the
    # resumed run stays bit-identical to the first.  On overwrite the previous rmtree wiped the
    # old config; writing it here restores reproducibility for downstream tools that load
    # ``resolved_config.yaml`` from the run dir.  Fixes #31.
    os.makedirs(run_dir, exist_ok=True)
    if not cfg.do_resume or cfg.do_overwrite:
        OmegaConf.save(cfg, run_dir / "config.yaml")
        save_resolved_config(cfg, run_dir / "resolved_config.yaml")

    logger.info("Setting torch float32 matmul precision to 'medium'.")
    torch.set_float32_matmul_precision("medium")

    # Seed *before* any `instantiate(...)` call so that model weight init, DataLoader
    # generator construction, and any other RNG-consuming work happen under the seeded
    # RNG state.  Previously this block ran after `instantiate(cfg.lightning_module)`,
    # which meant the starting weights were sampled from whatever torch's RNG happened
    # to be at process startup — a state that varies across Python versions and
    # platforms (PYTHONHASHSEED, module import order, etc.), so two runners with the
    # same `cfg.seed` still produced different initial weights and different training
    # trajectories.  Reading `do_demo` off the config (rather than the instantiated
    # `M.model.do_demo`) lets us keep the gate without needing `M` yet.
    do_demo = cfg.lightning_module.model.get("do_demo", False)
    if do_demo or cfg.get("seed", None):
        seed_everything(cfg.get("seed", 1), workers=True)

    D = instantiate(cfg.datamodule)
    logger.info(f"Train dataset contains {len(D.train_dataloader().dataset)} datapoints")

    M = hydra.utils.instantiate(cfg.lightning_module)

    trainer = instantiate(cfg.trainer)

    # Log the run dir up front so every run (even crashed/in-flight) is matchable from the wandb UI
    # back to its folder on disk — best_ckpt_path below is only logged after fit() completes.
    for log in trainer.loggers:
        log.log_hyperparams({"run_dir": str(run_dir)})

    trainer_kwargs = {"model": M, "datamodule": D}
    if ckpt_path:
        logger.info(f"Trying to resume training from checkpoint {ckpt_path}.")
        trainer_kwargs["ckpt_path"] = ckpt_path
    if not ckpt_path:
        # Baseline val metrics at step 0 so tuning curves start from the untrained model.
        # The sanity check can't do this: it doesn't write to loggers.
        logger.info("Running baseline validation")
        trainer.validate(M, datamodule=D)

    logger.info("Fitting model")
    trainer.fit(**trainer_kwargs)

    best_ckpt_path = Path(trainer.checkpoint_callback.best_model_path)
    if not best_ckpt_path.is_file():
        raise ValueError("No best checkpoint reported.")
    else:
        for log in trainer.loggers:
            log.log_hyperparams({"best_ckpt_path": best_ckpt_path})

    output_fp = run_dir / "best_model.ckpt"
    shutil.copyfile(best_ckpt_path, output_fp)

    best_score = trainer.checkpoint_callback.best_model_score

    # ``best_model_score`` is scoped to the current ``fit`` call's validation events: on a
    # no-op resume (``max_steps`` already reached) no validation runs, so it stays None
    # even though ``best_model_path`` still points at a real checkpoint inherited from the
    # prior run.  Guarding the format here keeps that path from crashing on
    # ``NoneType.__format__``.
    score_str = f" (with score {best_score:.2f})" if best_score is not None else ""
    logger.info(f"Best checkpoint{score_str} copied to {output_fp!s}.")


if __name__ == "__main__":
    main()
