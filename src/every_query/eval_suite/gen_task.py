from pathlib import Path

import hydra
import polars as pl
from omegaconf import DictConfig, ListConfig
from tqdm import tqdm

from every_query.utils.codes import code_slug


def list_parquets(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.suffix == ".parquet")


@hydra.main(config_path="./conf", config_name="gen_tasks_config", version_base=None)
def main(cfg: DictConfig) -> None:
    index_dir = Path(cfg.paths.index_times_dir)
    all_dir = Path(cfg.paths.all_tasks_dir)
    out_root = Path(cfg.paths.out_root_dir)
    index_hash = str(cfg.index_hash)
    eval_codes_obj = cfg.eval_codes

    if isinstance(eval_codes_obj, ListConfig):
        # Flat list case: just read in all codes
        codes = list(eval_codes_obj)
    elif isinstance(eval_codes_obj, DictConfig):
        # Structured case: expect id/ood subfields
        id_codes = list(eval_codes_obj.get("id", []))
        ood_codes = list(eval_codes_obj.get("ood", []))
        codes = id_codes + ood_codes
    else:
        raise ValueError(
            f"eval_codes must be a list or dict with id/ood subfields, got {type(eval_codes_obj)}"
        )
    skip_existing = bool(cfg.skip_existing)

    out_base = out_root / index_hash
    out_base.mkdir(parents=True, exist_ok=True)

    index_shards = list_parquets(index_dir)
    all_shards = list_parquets(all_dir)

    assert [p.name for p in index_shards] == [p.name for p in all_shards]

    for shard_idx, (idx_fp, all_fp) in tqdm(
        enumerate(zip(index_shards, all_shards, strict=True)), total=len(index_shards)
    ):
        index_df = (
            pl.read_parquet(idx_fp)
            .select(["subject_id", "prediction_time", "censored"])
            .rename({"censored": "boolean_value"})
        )

        shard_all_df = pl.read_parquet(all_fp)

        for code in codes:
            slug = code_slug(code)
            code_dir = out_base / slug
            code_dir.mkdir(parents=True, exist_ok=True)

            out_fp = code_dir / f"{shard_idx}.parquet"
            if out_fp.exists() and skip_existing:
                print(f"Skipping {out_fp}, already exists")
                continue

            df = (
                index_df.join(
                    shard_all_df.select(["subject_id", "prediction_time", code]),
                    on=["subject_id", "prediction_time"],
                    how="left",
                )
                .rename({code: "occurs"})
                .with_columns(pl.lit(code).alias("query"))
                .select(
                    "subject_id",
                    "prediction_time",
                    "boolean_value",
                    "query",
                    "occurs",
                )
                .filter(pl.col("occurs").is_not_null())
            )

            df.write_parquet(out_fp)


if __name__ == "__main__":
    main()
