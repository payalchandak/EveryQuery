import hashlib
import json
from pathlib import Path

import hydra
import polars as pl
from meds import held_out_split
from omegaconf import DictConfig, OmegaConf


def list_parquets(d: Path) -> list[Path]:
    return sorted([p for p in d.iterdir() if p.is_file() and p.suffix == ".parquet"])


def build_time_hash(cfg: DictConfig, read_dir: Path) -> str:
    k = int(cfg.sampling.sample_times_per_subject)
    seed = int(cfg.sampling.seed)
    key = f"{read_dir.resolve()}|K={k}|seed={seed}"
    hash_hex = hashlib.md5(key.encode()).hexdigest()
    return hash_hex


@hydra.main(config_path="./conf", config_name="gen_index_times_config", version_base=None)
def main(cfg: DictConfig) -> None:
    task_dir = Path(cfg.io.task_dir)
    read_dir = task_dir / "all"
    out_root = Path(cfg.io.out_root)

    split = held_out_split
    time_hash = build_time_hash(cfg, read_dir)
    write_root = out_root / "index_times" / time_hash
    write_root.mkdir(parents=True, exist_ok=True)

    meta = {
        "time_hash": time_hash,
        "task_dir": str(task_dir.resolve()),
        "read_dir": str(read_dir.resolve()),
        "sample_times_per_subject": int(cfg.sampling.sample_times_per_subject),
        "seed": int(cfg.sampling.seed),
    }

    in_dir = read_dir / split
    write_root.mkdir(parents=True, exist_ok=True)
    write_shards = write_root / split
    write_shards.mkdir(parents=True, exist_ok=True)

    for shard_path in list_parquets(in_dir):
        out_path = write_shards / shard_path.name
        if out_path.exists() and bool(cfg.behavior.skip_existing):
            print(f"Skipping {out_path}, already exists")
            continue

        df = pl.read_parquet(
            shard_path,
            columns=["subject_id", "prediction_time", "censored"],
        )

        sampled = (
            df.sample(fraction=1.0, shuffle=True, seed=int(cfg.sampling.seed))
            .group_by("subject_id")
            .head(int(cfg.sampling.sample_times_per_subject))
        )

        sampled.write_parquet(out_path)

    (write_root / "meta.json").write_text(json.dumps(meta, indent=2))
    (write_root / "config_resolved.yaml").write_text(OmegaConf.to_yaml(cfg))

    print(str(write_root))


if __name__ == "__main__":
    main()
