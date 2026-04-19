import logging
import os
from importlib.resources import files
from pathlib import Path

import hydra
import polars as pl
from omegaconf import DictConfig
from tqdm import tqdm

from every_query.data.schema import TaskQuerySchema
from every_query.utils.codes import code_slug

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def shard_id(p: Path) -> int:
    return int(p.stem)


def create_eq_task_df(
    eq_shard_fp: str,
    aces_shard_fp: str,
    codes: list[str],
    output_root: str,
    target_rows: int,
    duration_days: float,
) -> int:
    """Convert an ACES task shard + matching EQ all-tasks shard into per-code ``TaskQuerySchema``-conformant
    parquets, one per code.

    Args:
        duration_days: Horizon in days to write into the ``duration_days`` column of the
            output.  Constant across every row emitted by a single ACES conversion —
            ACES-style tasks carry their window in the task config rather than per-row.
    """
    # boolean value in aces_shard_df is the real label for the task that will be used for eval
    # Note: every query expected this to eventually be called the "occurs" column as boolean_value
    # is for censoring
    aces_shard_df = pl.read_parquet(aces_shard_fp, columns=["subject_id", "prediction_time", "boolean_value"])
    aces_shard_df = aces_shard_df.rename({"boolean_value": "task_label"})
    eq_shard_all_tasks_df = pl.read_parquet(
        eq_shard_fp, columns=["subject_id", "prediction_time", "censored", *codes]
    )
    shard_name = Path(eq_shard_fp).name

    joined_df = aces_shard_df.join(eq_shard_all_tasks_df, on=["subject_id", "prediction_time"], how="left")

    assert joined_df.select(pl.col("censored").null_count()).item() == 0

    base_cols = ["subject_id", "prediction_time", "task_label", "censored"]

    # Output columns conform to TaskQuerySchema + one extra ``task_label`` column the
    # external-tasks pipeline carries for its own bookkeeping.  TaskQuerySchema allows
    # extra columns by default (``allow_extra_columns`` inherited from the Schema
    # base), so the extra passes validation.
    for code in codes:
        slug = code_slug(code)
        output_dir = Path(output_root) / slug
        os.makedirs(output_dir, exist_ok=True)

        # per-code EQ df conforming to TaskQuerySchema's nullable boolean_value:
        #   null  → censored (input `censored=True`)
        #   True  → event occurred (input `<code>=True`)
        #   False → no event, not censored
        boolean_value = (
            pl.when(pl.col("censored")).then(pl.lit(None, dtype=pl.Boolean)).otherwise(pl.col(code))
        )
        eq_df = (
            joined_df.select([*base_cols, code])
            .with_columns(
                boolean_value.alias(TaskQuerySchema.boolean_value_name),
                pl.lit(code).alias(TaskQuerySchema.query_name),
                pl.lit(duration_days).cast(pl.Float32).alias(TaskQuerySchema.duration_days_name),
            )
            .select(
                [
                    TaskQuerySchema.subject_id_name,
                    TaskQuerySchema.prediction_time_name,
                    TaskQuerySchema.query_name,
                    TaskQuerySchema.duration_days_name,
                    TaskQuerySchema.boolean_value_name,
                    "task_label",
                ]
            )
        )

        # Align to TaskQuerySchema at the write boundary — same check the sampler does.
        aligned = TaskQuerySchema.align(eq_df.to_arrow())

        out_fp = output_dir / shard_name  # same shard id as original
        pl.from_arrow(aligned).write_parquet(out_fp)

        logger.info(f"Wrote to {out_fp}")


ACES_CONFIGS = str(files("every_query") / "predict" / "external_tasks" / "configs")


@hydra.main(version_base="1.3", config_path=ACES_CONFIGS, config_name="aces_to_eq.yaml")
def main(cfg: DictConfig) -> None:
    all_eq_shards = sorted(
        Path(cfg.eq_tasks_all_dir).glob("*.parquet"),
        key=shard_id,
    )

    all_aces_shards = sorted(
        Path(cfg.aces_shards_dir).glob("*.parquet"),
        key=shard_id,
    )

    for eq_shard_fp, aces_shard_fp in tqdm(
        zip(all_eq_shards, all_aces_shards, strict=True), total=len(all_eq_shards)
    ):
        create_eq_task_df(
            eq_shard_fp=str(eq_shard_fp),
            aces_shard_fp=str(aces_shard_fp),
            codes=cfg.queries,
            output_root=cfg.output_dir,
            target_rows=cfg.target_rows_per_shard,
            duration_days=float(cfg.duration_days),
        )


if __name__ == "__main__":
    main()
