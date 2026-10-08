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
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor
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


def required_position_embeddings(model_cfg: DictConfig, max_seq_len: int) -> int:
    """The ``max_position_embeddings`` a model config needs for a ``max_seq_len`` data window.

    The decoder-only
    :class:`~every_query.model.conditional_multitask_ar_model.ConditionalMultitaskARModel` runs
    patient history and window stream through **one** backbone, so its position budget must
    cover both: ``max_seq_len`` patient tokens plus three tokens per window block up to
    ``max_windows``.  Every other model only ever feeds the backbone the patient window plus the
    two tokens the single-query model splices in (query + duration), which is the fallback below.

    Examples:
        >>> multitask = OmegaConf.create(
        ...     {"_target_":
        ...      "every_query.model.conditional_multitask_ar_model.ConditionalMultitaskARModel",
        ...      "max_windows": 8}
        ... )
        >>> required_position_embeddings(multitask, 256)
        280

        Anything else gets the patient window plus the single-query model's two spliced tokens:

        >>> single = OmegaConf.create({"_target_": "every_query.model.EveryQueryModel"})
        >>> required_position_embeddings(single, 256)
        258
    """
    target = str(model_cfg.get("_target_", ""))
    target_name = target.rsplit(".", 1)[-1]
    if target_name == "ConditionalMultitaskARModel":
        return max_seq_len + 3 * int(model_cfg.max_windows)
    return max_seq_len + 2


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


def resolve_seed(do_demo: bool, seed: int | None) -> int | None:
    """The seed to pass to ``seed_everything``, or ``None`` when the run should not be seeded.

    ``seed=0`` is a legitimate seed; the old ``if do_demo or cfg.get("seed")`` truthiness gate
    silently skipped seeding for it.

    Examples:
        >>> resolve_seed(False, 0)
        0
        >>> resolve_seed(False, 7)
        7
        >>> resolve_seed(False, None) is None
        True
        >>> resolve_seed(True, None)
        1
    """
    if seed is not None:
        return int(seed)
    return 1 if do_demo else None


def drop_lr_monitor_without_logger(trainer: Trainer) -> int:
    """Remove every ``LearningRateMonitor`` from a trainer that has no logger; returns the count.

    ``LearningRateMonitor.on_train_start`` raises ``MisconfigurationException`` without a logger,
    *after* the dataset has been loaded and the baseline validation has run.  Every production
    config ships the monitor, so ``trainer.logger=false`` has to be able to shed it on its own.

    Examples:
        >>> t = Trainer(logger=False, callbacks=[LearningRateMonitor()], enable_progress_bar=False)
        >>> drop_lr_monitor_without_logger(t)
        1
        >>> any(isinstance(cb, LearningRateMonitor) for cb in t.callbacks)
        False

        With a logger the callbacks are left alone:

        >>> import tempfile
        >>> from lightning.pytorch.loggers import CSVLogger
        >>> with tempfile.TemporaryDirectory() as d:
        ...     t = Trainer(logger=CSVLogger(d), callbacks=[LearningRateMonitor()], enable_progress_bar=False)
        ...     drop_lr_monitor_without_logger(t)
        0
    """
    if trainer.loggers:
        return 0
    kept = [cb for cb in trainer.callbacks if not isinstance(cb, LearningRateMonitor)]
    dropped = len(trainer.callbacks) - len(kept)
    if dropped:
        logger.warning(
            "trainer.logger is disabled; dropping %d LearningRateMonitor callback(s) that would "
            "otherwise raise at train start.",
            dropped,
        )
        trainer.callbacks = kept
    return dropped


def completed_epochs(ckpt: dict) -> int | None:
    """Number of training epochs a Lightning checkpoint has fully completed, or ``None`` if unknown.

    ``ckpt["epoch"]`` is the epoch *index* and ``fit_loop.epoch_progress.current.processed`` is
    only bumped in ``on_train_epoch_end`` -- a ``last.ckpt`` written by the end-of-epoch
    validation (the shipped configs' cadence) therefore still reads ``processed=0`` after a full
    epoch.  Lightning infers the finished epoch from the batch loop on restore; this mirrors
    that inference so the decision can be made *before* ``trainer.fit`` touches the checkpoint.

    Examples:
        >>> def ckpt(processed, completed, started, ready_b, processed_b, last):
        ...     return {"loops": {"fit_loop": {
        ...         "epoch_progress": {"current": {"ready": started, "started": started,
        ...                                        "processed": processed, "completed": completed}},
        ...         "epoch_loop.batch_progress": {"is_last_batch": last,
        ...             "current": {"ready": ready_b, "started": ready_b,
        ...                         "processed": processed_b, "completed": processed_b}},
        ...     }}}

        Saved by ``on_train_epoch_end``: the epoch is already counted.

        >>> completed_epochs(ckpt(1, 0, 1, 4, 4, True))
        1

        Saved by the validation after the last batch of the first epoch: not yet counted.

        >>> completed_epochs(ckpt(0, 0, 1, 4, 4, True))
        1

        Saved mid-epoch:

        >>> completed_epochs(ckpt(0, 0, 1, 2, 2, False))
        0
        >>> completed_epochs(ckpt(1, 1, 2, 2, 2, False))
        1

        A checkpoint without loop state:

        >>> completed_epochs({"epoch": 0}) is None
        True
    """
    try:
        fit_loop = ckpt["loops"]["fit_loop"]
        epoch = fit_loop["epoch_progress"]["current"]
        batch = fit_loop["epoch_loop.batch_progress"]
        processed = int(epoch["processed"])
        at_last_batch = bool(batch["is_last_batch"]) and int(batch["current"]["ready"]) == int(
            batch["current"]["processed"]
        )
        uncounted = int(epoch["started"]) > processed and int(epoch["completed"]) == processed
    except (KeyError, TypeError, ValueError):
        return None
    return processed + 1 if (at_last_batch and uncounted) else processed


def resume_budget_spent(ckpt: dict, max_steps: int | None, max_epochs: int | None) -> str | None:
    """Why resuming *ckpt* under this trainer budget would train zero steps, or ``None`` if it would train.

    A no-op ``trainer.fit`` is not harmless: Lightning restores the loop state, bumps the epoch
    counters, and ``ModelCheckpoint.on_train_end`` rewrites ``last.ckpt`` with them, after which
    a later ``trainer.max_epochs=2`` extension resumes at the end of its batch loop and also
    trains zero steps.

    Examples:
        >>> ck = {"global_step": 4, "loops": {"fit_loop": {
        ...     "epoch_progress": {"current": {"ready": 1, "started": 1, "processed": 0, "completed": 0}},
        ...     "epoch_loop.batch_progress": {"is_last_batch": True,
        ...         "current": {"ready": 4, "started": 4, "processed": 4, "completed": 4}}}}}
        >>> resume_budget_spent(ck, 4, 5)
        'global_step=4 >= max_steps=4'
        >>> resume_budget_spent(ck, -1, 1)
        'completed_epochs=1 >= max_epochs=1'
        >>> resume_budget_spent(ck, 6, 2) is None
        True
        >>> resume_budget_spent({"global_step": 4}, None, 1) is None
        True
    """
    global_step = int(ckpt.get("global_step", 0))
    if max_steps is not None and max_steps > 0 and global_step >= max_steps:
        return f"global_step={global_step} >= max_steps={max_steps}"
    epochs = completed_epochs(ckpt)
    if max_epochs is not None and max_epochs > 0 and epochs is not None and epochs >= max_epochs:
        return f"completed_epochs={epochs} >= max_epochs={max_epochs}"
    return None


def resolve_best_checkpoint(reported: str | Path | None, run_dir: Path) -> Path:
    """The checkpoint to publish as ``best_model.ckpt``: the reported best, else the latest one.

    ``ModelCheckpoint`` only records a best path in ``on_validation_end``, so a run whose
    ``max_steps`` ends before the first validation (or a resume that trained zero steps) reports
    none even though training succeeded.  Falling back to ``last.ckpt`` (via
    :func:`find_checkpoint_path`) keeps such runs usable; only a run with no checkpoint at all
    is an error.  A reported path that no longer exists (the run dir was moved) is retried by
    basename under ``run_dir/checkpoints``.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as d:
        ...     ckpts = Path(d) / "checkpoints"; ckpts.mkdir()
        ...     (ckpts / "epoch=0-step=3.ckpt").touch(); (ckpts / "last.ckpt").touch()
        ...     resolve_best_checkpoint(ckpts / "epoch=0-step=3.ckpt", Path(d)).name
        ...     resolve_best_checkpoint("/moved/checkpoints/epoch=0-step=3.ckpt", Path(d)).name
        ...     resolve_best_checkpoint("", Path(d)).name
        'epoch=0-step=3.ckpt'
        'epoch=0-step=3.ckpt'
        'last.ckpt'
        >>> with tempfile.TemporaryDirectory() as d:
        ...     resolve_best_checkpoint("", Path(d))
        Traceback (most recent call last):
            ...
        ValueError: No best checkpoint reported and no checkpoint found under ...
    """
    if reported:
        reported = Path(reported)
        for candidate in (reported, run_dir / "checkpoints" / reported.name):
            if candidate.is_file():
                return candidate
    fallback = find_checkpoint_path(run_dir)
    if fallback is None:
        raise ValueError(
            f"No best checkpoint reported and no checkpoint found under {run_dir / 'checkpoints'}."
        )
    logger.warning(
        "No best checkpoint reported (reported=%r); publishing %s as best_model.ckpt instead. "
        "This happens when max_steps ends before the first validation.",
        str(reported) if reported else "",
        fallback,
    )
    return fallback


def _best_model_path_from_checkpoint(ckpt: dict) -> str:
    """``ModelCheckpoint``'s recorded ``best_model_path`` in a checkpoint's callback states, or ``""``."""
    for state in (ckpt.get("callbacks") or {}).values():
        if isinstance(state, dict) and state.get("best_model_path"):
            return str(state["best_model_path"])
    return ""


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

    # Size the model from the data: vocab from metadata/codes.parquet, positions from the
    # datamodule window plus what the selected architecture adds on top of it (two spliced
    # tokens for the single-query model; three tokens per query block for the decoder-only
    # conditional model — see ``required_position_embeddings``).  Done here, before the config
    # is saved and before
    # ``validate_resume_directory`` diffs it, so the run dir records the real numbers and a
    # resumed run compares like with like.
    ds_cfg = instantiate(cfg.datamodule.config)
    cohort_vocab_size = ds_cfg.vocab_size
    vocab_size = cohort_vocab_size

    # With an ontology, the embedding table must cover the ancestor nodes too: they are appended
    # above the highest leaf index, so the cohort's own vocab_size would leave every ancestor
    # index out of range.  Read it from the ontology rather than making the user keep a hardcoded
    # V_ext in the config in sync with a rebuilt DAG.
    ontology_dir = cfg.lightning_module.model.get("ontology_dir")
    if ontology_dir:
        from every_query.data.ontology import check_ontology_cohort, cohort_code_map, extended_vocab_size

        v_ext = extended_vocab_size(ontology_dir)
        if v_ext < vocab_size:
            raise ValueError(
                f"Ontology at {ontology_dir} declares V_ext={v_ext}, smaller than the cohort's "
                f"vocab_size={vocab_size}.  It was almost certainly built from a different "
                f"codes.parquet than this cohort."
            )
        # Widths agree; now require the ontology's observed nodes to be this cohort's codes.parquet
        # rows, code for code and index for index.  Same-width ontologies of another cohort, or of
        # this one with renumbered codes, would otherwise pair every leaf embedding and target
        # column with the wrong ancestor rows without any error.
        check_ontology_cohort(ontology_dir, code_to_index=cohort_code_map(ds_cfg.code_metadata_fp))
        logger.info("Ontology: sizing the encoder to V_ext=%d (cohort vocab %d).", v_ext, vocab_size)
        vocab_size = v_ext

    cfg.lightning_module.model.config_overrides.vocab_size = vocab_size
    # Models that persist the cohort's vocabulary identity (the multitask model's
    # ``cohort_vocab_fingerprint``; the shipped configs carry the key as ``null``) get it from the
    # same codes.parquet the widths came from, so every later load of the checkpoint can re-verify
    # its ontology - and EQ_predict_multitask the inference cohort - against the cohort it was
    # trained on rather than against a width.
    if "cohort_vocab_fingerprint" in cfg.lightning_module.model:
        from every_query.data.ontology import cohort_code_map
        from every_query.utils.digest import vocab_fingerprint

        cfg.lightning_module.model.cohort_vocab_fingerprint = vocab_fingerprint(
            cohort_code_map(ds_cfg.code_metadata_fp)
        )
    # The multitask datamodule's ``expected_vocab_size`` is the width its training sidecars are
    # checked against, and those are leaf-only: it must stay the *cohort's* V even when the model
    # above was just widened to V_ext.  The shipped configs interpolate it from
    # ``config_overrides.vocab_size``, which would now resolve to V_ext, so pin the cohort width
    # explicitly; ``resolved_config.yaml`` then records both numbers (``EQ_predict_multitask``
    # reads them).  Configs whose datasets do not take the key (the scalar models') are left alone.
    dataset_kwargs = cfg.datamodule.get("dataset_kwargs")
    if dataset_kwargs is not None and "expected_vocab_size" in dataset_kwargs:
        cfg.datamodule.dataset_kwargs.expected_vocab_size = cohort_vocab_size
    # The datasets' ``ontology_dir`` interpolates from the model's in every shipped config, but an
    # override can desync them - and then a labels directory sampled with ancestor-valued start /
    # bound / conditioning codes hands ids in [V, V_ext) to an embedding table the model sized for a
    # different (or no) ontology.  Nothing downstream can attribute that: on CUDA it is an
    # asynchronous device-side assert.  One comparison here turns it into a sentence.
    if dataset_kwargs is not None and "ontology_dir" in dataset_kwargs:
        data_onto = dataset_kwargs.get("ontology_dir")
        if (data_onto or None) != (ontology_dir or None):
            raise ValueError(
                f"datamodule.dataset_kwargs.ontology_dir ({data_onto!r}) and "
                f"lightning_module.model.ontology_dir ({ontology_dir!r}) must name the same ontology: "
                "the datasets resolve ancestor start / bound / conditioning codes through the first "
                "and the model embeds them from a table sized by the second."
            )
    cfg.lightning_module.model.config_overrides.max_position_embeddings = required_position_embeddings(
        cfg.lightning_module.model, ds_cfg.max_seq_len
    )

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
    ckpt_path = None
    if cfg_path.exists():
        if cfg.do_overwrite:
            logger.info(f"Overwriting existing run directory {run_dir}.")
            shutil.rmtree(run_dir, ignore_errors=True)
        elif cfg.do_resume:
            logger.info(f"Resuming training in existing run directory {run_dir}.")
            validate_resume_directory(run_dir, cfg)
            ckpt_path = find_checkpoint_path(run_dir)
            # Reuse the original run's wandb id so the resumed run continues the same curve
            # instead of starting a new wandb run.  Mutated *after* validate_resume_directory
            # so the config diff still compares like with like.
            wandb_id_fp = run_dir / "wandb_run_id"
            if ckpt_path and _is_wandb_logger(cfg.trainer.get("logger")) and wandb_id_fp.is_file():
                with open_dict(cfg.trainer.logger):
                    cfg.trainer.logger.id = wandb_id_fp.read_text().strip()
                    cfg.trainer.logger.resume = "allow"
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

    # Kept in step with ``utils/model_loader.py`` so training and scoring use the same matmuls.
    logger.info("Setting torch float32 matmul precision to 'high'.")
    torch.set_float32_matmul_precision("high")

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
    seed = resolve_seed(do_demo, cfg.get("seed", None))
    if seed is not None:
        seed_everything(seed, workers=True)

    D = instantiate(cfg.datamodule)
    logger.info(f"Train dataset contains {len(D.train_dataloader().dataset)} datapoints")

    M = hydra.utils.instantiate(cfg.lightning_module)

    trainer = instantiate(cfg.trainer)
    drop_lr_monitor_without_logger(trainer)

    # Log the run dir up front so every run (even crashed/in-flight) is matchable from the wandb UI
    # back to its folder on disk — best_ckpt_path below is only logged after fit() completes.
    for log in trainer.loggers:
        log.log_hyperparams({"run_dir": str(run_dir)})

    # Persist the wandb run id so a later `do_resume` continues the same wandb curve.
    # log_hyperparams above already created the experiment, so `version` is the real run id
    # on rank zero (non-zero ranks see None and skip).
    if _is_wandb_logger(cfg.trainer.get("logger")) and not (run_dir / "wandb_run_id").is_file():
        run_id = trainer.logger.version
        if trainer.is_global_zero and run_id:
            (run_dir / "wandb_run_id").write_text(str(run_id))

    trainer_kwargs = {"model": M, "datamodule": D}
    if ckpt_path:
        logger.info(f"Trying to resume training from checkpoint {ckpt_path}.")
        trainer_kwargs["ckpt_path"] = ckpt_path
    if not ckpt_path:
        # Baseline val metrics at step 0 so tuning curves start from the untrained model.
        # The sanity check can't do this: it doesn't write to loggers.
        logger.info("Running baseline validation")
        trainer.validate(M, datamodule=D)

    # A resume whose budget is already spent must not enter ``trainer.fit``: Lightning would
    # restore the loop state, count the finished epoch, and let ``ModelCheckpoint.on_train_end``
    # rewrite ``last.ckpt`` with the bumped counters -- after which a later ``max_epochs``
    # extension resumes at the end of its batch loop and trains zero steps too.
    skip_reason = None
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        skip_reason = resume_budget_spent(ckpt, cfg.trainer.get("max_steps"), cfg.trainer.get("max_epochs"))

    if skip_reason:
        logger.warning(
            f"Resume checkpoint {ckpt_path} already meets the training budget ({skip_reason}); "
            "skipping trainer.fit so last.ckpt is left untouched. Raise max_steps/max_epochs to train."
        )
        reported_best = _best_model_path_from_checkpoint(ckpt)
        best_score = None
    else:
        logger.info("Fitting model")
        trainer.fit(**trainer_kwargs)
        reported_best = trainer.checkpoint_callback.best_model_path
        best_score = trainer.checkpoint_callback.best_model_score

    best_ckpt_path = resolve_best_checkpoint(reported_best, run_dir)
    for log in trainer.loggers:
        # ``str``: a PosixPath in hparams.yaml is not ``yaml.safe_load``-able.
        log.log_hyperparams({"best_ckpt_path": str(best_ckpt_path)})

    output_fp = run_dir / "best_model.ckpt"
    shutil.copyfile(best_ckpt_path, output_fp)

    # ``best_model_score`` is scoped to the current ``fit`` call's validation events: on a
    # no-op resume (``max_steps`` already reached) no validation runs, so it stays None
    # even though ``best_model_path`` still points at a real checkpoint inherited from the
    # prior run.  Guarding the format here keeps that path from crashing on
    # ``NoneType.__format__``.
    score_str = f" (with score {best_score:.2f})" if best_score is not None else ""
    logger.info(f"Best checkpoint{score_str} copied to {output_fp!s}.")


if __name__ == "__main__":
    main()
