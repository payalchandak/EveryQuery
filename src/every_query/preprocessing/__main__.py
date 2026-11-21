import logging
import os
import subprocess
from importlib.resources import files
from pathlib import Path

import hydra
import polars as pl
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

CONFIGS = files("every_query") / "preprocessing" / "configs"

TEXT_VALUE_FUSION_STAGE = "fuse_text_values"
TEXT_VALUE_COLUMN = "text_value"


def _input_has_text_value(input_dir: Path) -> bool:
    """Whether the input cohort's data shards carry a ``text_value`` column, which MEDS leaves optional."""
    shards = sorted((input_dir / "data").rglob("*.parquet"))
    if not shards:
        return False
    return TEXT_VALUE_COLUMN in pl.scan_parquet(shards[0]).collect_schema().names()


def _pipeline_config(
    template_fp: Path, input_dir: Path, out_fp: Path, do_fuse_text_values: bool = True
) -> Path:
    """Write the pipeline config this run uses next to its outputs and return its path.

    The ``fuse_text_values`` stage reads ``text_value``, and MEDS-transforms refuses an expression over a
    column the data does not have, so the stage is dropped when the input lacks that column; it is also
    dropped when ``do_fuse_text_values`` is false. Writing the config out records which stages produced a
    given cohort.
    """
    cfg = OmegaConf.load(template_fp)
    skip_reason = None
    if not do_fuse_text_values:
        skip_reason = "do_fuse_text_values=False"
    elif not _input_has_text_value(input_dir):
        skip_reason = f"the input has no {TEXT_VALUE_COLUMN} column"
    if skip_reason is not None:
        logger.info(f"Skipping the {TEXT_VALUE_FUSION_STAGE} stage: {skip_reason}")
        cfg.stages = [
            s for s in cfg.stages if not (isinstance(s, DictConfig) and TEXT_VALUE_FUSION_STAGE in s)
        ]
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_fp)
    return out_fp


def _run_stage(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    """Run a preprocessing child (Hydra-based CLI) with live stdout/stderr.

    stdout/stderr inherit from the parent (not captured) so progress streams live and large
    outputs don't pile up in RAM.  Both wrapped tools (``MEDS_transform-pipeline`` and
    ``MTD_preprocess``) write their own Hydra log files under their respective run dirs; on
    failure the error message names the command + return code and tells the user to check those.
    """
    logger.info(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, check=False)
    if result.returncode != 0:  # pragma: no cover
        raise RuntimeError(
            f"{cmd[0]} failed with rc={result.returncode}. "
            f"Command: {' '.join(cmd)}. "
            f"See the child tool's Hydra log directory for details."
        )


@hydra.main(version_base=None, config_path=str(CONFIGS), config_name="_process_data")
def main(cfg: DictConfig):
    input_dir = Path(cfg.input_dir)
    intermediate_dir = Path(cfg.intermediate_dir)
    output_dir = Path(cfg.output_dir)
    do_demo = cfg.do_demo
    stage_runner_fp = cfg.stage_runner_fp  # optional MEDS-Transforms stage runner (e.g. parallelize block)

    # 0. Pre-MTD pre-processing
    logger.info("Pre-MTD pre-processing")
    done_fp = intermediate_dir / ".done"
    if done_fp.exists():  # pragma: no cover
        logger.info("Pre-MTD pre-processing already done, skipping")
    else:
        env = os.environ.copy()
        env["RAW_MEDS_DIR"] = str(input_dir)
        env["MTD_INPUT_DIR"] = str(intermediate_dir)

        if do_demo:
            env["MIN_SUBJECTS_PER_CODE"] = "2"
            env["MIN_EVENTS_PER_SUBJECT"] = "1"

        template_fp = (CONFIGS / "_reshard_data.yaml") if cfg.do_reshard else (CONFIGS / "_data.yaml")
        pipeline_config_fp = _pipeline_config(
            Path(str(template_fp)),
            input_dir,
            intermediate_dir / ".logs" / "pipeline_config.yaml",
            do_fuse_text_values=bool(cfg.get("do_fuse_text_values", True)),
        )
        cmd = ["MEDS_transform-pipeline", str(pipeline_config_fp)]
        if stage_runner_fp:
            cmd.extend(["--stage_runner_fp", str(stage_runner_fp)])
        _run_stage(cmd, env=env)

        logger.info("Pre-MTD pre-processing done")
        done_fp.touch()

    # 1. Run MTD pre-processing
    logger.info("Running MTD pre-processing")
    done_fp = output_dir / ".done"

    if done_fp.exists():  # pragma: no cover
        logger.info("MTD pre-processing already done, skipping")
    else:
        # MTD_preprocess takes everything it needs via CLI args; no env mutation needed, so we let
        # it inherit the parent environment instead of building a throwaway copy.
        cmd = [
            "MTD_preprocess",
            f"MEDS_dataset_dir={intermediate_dir!s}",
            f"output_dir={output_dir!s}",
        ]
        if stage_runner_fp:
            cmd.append(f"stage_runner_fp={stage_runner_fp!s}")
        _run_stage(cmd)

        logger.info("MTD pre-processing done")
        done_fp.touch()

    return


if __name__ == "__main__":
    main()
