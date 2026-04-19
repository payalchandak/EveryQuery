from importlib.resources import files
from pathlib import Path

import hydra
import polars as pl
from omegaconf import DictConfig, ListConfig
from tqdm import tqdm

from every_query.utils.codes import code_slug


def list_parquets(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.suffix == ".parquet")


def _resolve_codes(eval_codes_obj) -> list[str]:
    if isinstance(eval_codes_obj, ListConfig):
        return list(eval_codes_obj)
    elif isinstance(eval_codes_obj, DictConfig):
        id_codes = list(eval_codes_obj.get("id", []))
        ood_codes = list(eval_codes_obj.get("ood", []))
        return id_codes + ood_codes
    else:
        raise ValueError(
            f"eval_codes must be a list or dict with id/ood subfields, got {type(eval_codes_obj)}"
        )


def process_eval_tasks(
    index_dir: Path,
    task_dir_base: Path,
    out_root: Path,
    index_hash: str,
    codes: list[str],
    durations: list[int],
    split: str,
    skip_existing: bool = False,
) -> None:
    """Generate per-code, per-duration eval task parquets from index times and duration-specific task data."""
    index_shards = list_parquets(index_dir)

    for duration in durations:
        dur_task_dir = task_dir_base / str(duration) / split
        if not dur_task_dir.is_dir():
            print(
                f"WARNING: task dir missing for duration={duration}, split={split}: {dur_task_dir}, skipping"
            )
            continue

        dur_shards = list_parquets(dur_task_dir)
        assert [p.name for p in index_shards] == [p.name for p in dur_shards], (
            f"Shard mismatch for duration={duration}, split={split}: index has {len(index_shards)}, "
            f"tasks has {len(dur_shards)}"
        )

        for shard_idx, (idx_fp, task_fp) in tqdm(
            enumerate(zip(index_shards, dur_shards, strict=True)),
            total=len(index_shards),
            desc=f"split={split}, duration={duration}",
        ):
            # Index times: just the (subject_id, prediction_time) pairs to evaluate
            index_df = pl.read_parquet(idx_fp).select(["subject_id", "prediction_time"])

            # Duration-specific task data: has censored + per-code columns
            shard_task_df = pl.read_parquet(task_fp)

            for code in codes:
                slug = code_slug(code)
                code_dir = out_root / index_hash / split / str(duration) / slug
                code_dir.mkdir(parents=True, exist_ok=True)

                out_fp = code_dir / f"{shard_idx}.parquet"
                if out_fp.exists() and skip_existing:
                    print(f"Skipping {out_fp}, already exists")
                    continue

                if code not in shard_task_df.collect_schema().names():
                    print(f"WARNING: code {code} not in shard {task_fp.name} for duration={duration}")
                    continue

                # Collapsed nullable boolean_value per TaskQuerySchema:
                #   null  → censored
                #   True  → event occurred (input `<code>=True`)
                #   False → no event, not censored
                # Filter step drops rows whose underlying per-code value was null in the
                # wide input AND weren't censored (a null per-code cell means the matrix
                # didn't have a value for this (subject, prediction_time) pair — distinct
                # from "censored" which is the meaningful null).
                boolean_value = (
                    pl.when(pl.col("censored")).then(pl.lit(None, dtype=pl.Boolean)).otherwise(pl.col(code))
                )
                df = (
                    index_df.join(
                        shard_task_df.select(["subject_id", "prediction_time", "censored", code]),
                        on=["subject_id", "prediction_time"],
                        how="inner",
                    )
                    .filter(pl.col("censored") | pl.col(code).is_not_null())
                    .with_columns(
                        boolean_value.alias("boolean_value"),
                        pl.lit(code).alias("query"),
                        # Float32 to match ``TaskQuerySchema.duration_days`` so the
                        # output aligns to the schema natively.
                        pl.lit(duration).cast(pl.Float32).alias("duration_days"),
                    )
                    .select(
                        "subject_id",
                        "prediction_time",
                        "boolean_value",
                        "query",
                        "duration_days",
                    )
                )

                df.write_parquet(out_fp)


EVAL_CONFIGS = str(files("every_query") / "evaluate" / "conf")


@hydra.main(config_path=EVAL_CONFIGS, config_name="gen_tasks_config", version_base=None)
def main(cfg: DictConfig) -> None:
    codes = _resolve_codes(cfg.eval_codes)
    extra = list(cfg.get("extra_codes", []))
    if extra:
        codes = codes + [c for c in extra if c not in codes]
    durations = list(cfg.durations)
    splits = list(cfg.splits)
    index_times_base = Path(cfg.paths.index_times_base)

    for split in splits:
        print(f"\n=== Processing split: {split} ===")
        process_eval_tasks(
            index_dir=index_times_base / split,
            task_dir_base=Path(cfg.paths.task_dir_base),
            out_root=Path(cfg.paths.out_root_dir),
            index_hash=str(cfg.index_hash),
            codes=codes,
            durations=durations,
            split=split,
            skip_existing=bool(cfg.skip_existing),
        )


if __name__ == "__main__":
    main()
