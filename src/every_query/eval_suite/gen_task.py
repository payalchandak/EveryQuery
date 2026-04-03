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
    skip_existing: bool = False,
) -> None:
    """Generate per-code, per-duration eval task parquets from index times and duration-specific task data."""
    index_shards = list_parquets(index_dir)

    for duration in durations:
        dur_task_dir = task_dir_base / str(duration) / "held_out"
        if not dur_task_dir.is_dir():
            print(f"WARNING: task dir missing for duration={duration}: {dur_task_dir}, skipping")
            continue

        dur_shards = list_parquets(dur_task_dir)
        assert [p.name for p in index_shards] == [p.name for p in dur_shards], (
            f"Shard mismatch for duration={duration}: index has {len(index_shards)}, "
            f"tasks has {len(dur_shards)}"
        )

        for shard_idx, (idx_fp, task_fp) in tqdm(
            enumerate(zip(index_shards, dur_shards, strict=True)),
            total=len(index_shards),
            desc=f"duration={duration}",
        ):
            # Index times: just the (subject_id, prediction_time) pairs to evaluate
            index_df = pl.read_parquet(idx_fp).select(["subject_id", "prediction_time"])

            # Duration-specific task data: has censored + per-code columns
            shard_task_df = pl.read_parquet(task_fp)

            for code in codes:
                slug = code_slug(code)
                code_dir = out_root / index_hash / str(duration) / slug
                code_dir.mkdir(parents=True, exist_ok=True)

                out_fp = code_dir / f"{shard_idx}.parquet"
                if out_fp.exists() and skip_existing:
                    print(f"Skipping {out_fp}, already exists")
                    continue

                if code not in shard_task_df.collect_schema().names():
                    print(f"WARNING: code {code} not in shard {task_fp.name} for duration={duration}")
                    continue

                df = (
                    index_df.join(
                        shard_task_df.select(["subject_id", "prediction_time", "censored", code]),
                        on=["subject_id", "prediction_time"],
                        how="inner",
                    )
                    .rename({"censored": "boolean_value", code: "occurs"})
                    .with_columns(
                        pl.lit(code).alias("query"),
                        pl.lit(duration).alias("duration_days"),
                    )
                    .select(
                        "subject_id",
                        "prediction_time",
                        "boolean_value",
                        "query",
                        "occurs",
                        "duration_days",
                    )
                    .filter(pl.col("occurs").is_not_null())
                )

                df.write_parquet(out_fp)


@hydra.main(config_path="./conf", config_name="gen_tasks_config", version_base=None)
def main(cfg: DictConfig) -> None:
    process_eval_tasks(
        index_dir=Path(cfg.paths.index_times_dir),
        task_dir_base=Path(cfg.paths.task_dir_base),
        out_root=Path(cfg.paths.out_root_dir),
        index_hash=str(cfg.index_hash),
        codes=_resolve_codes(cfg.eval_codes),
        durations=list(cfg.durations),
        skip_existing=bool(cfg.skip_existing),
    )


if __name__ == "__main__":
    main()
