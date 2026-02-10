import logging
import os
from pathlib import Path

import hydra
import polars as pl
from omegaconf import DictConfig
from tqdm import tqdm

from every_query.utils.codes import code_slug

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def shard_id(p: Path) -> int:
    return int(p.stem)


def create_eq_task_df(
    eq_shard_fp: str, aces_shard_fp: str, codes: list[str], output_root: str, target_rows: int
) -> int:
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

    for code in codes:
        slug = code_slug(code)
        output_dir = Path(output_root) / slug
        os.makedirs(output_dir, exist_ok=True)

        # per-code EQ df (no unpivot; one file per input shard)
        eq_df = (
            joined_df.select([*base_cols + code])
            .rename({code: "occurs", "censored": "boolean_value"})
            .with_columns(pl.lit(code).alias("query"))
            .select(["subject_id", "prediction_time", "query", "boolean_value", "occurs", "task_label"])
        )

        out_fp = output_dir / shard_name  # same shard id as original
        eq_df.write_parquet(out_fp)

        logger.info(f"Wrote to {out_fp}")


@hydra.main(version_base="1.3", config_path=".", config_name="config.yaml")
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
        )


if __name__ == "__main__":
    main()
