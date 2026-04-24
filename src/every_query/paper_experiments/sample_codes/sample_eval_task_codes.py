import hashlib
import os
import random
from pathlib import Path

import polars as pl
import yaml


def stable_hash_list(items: list[str]) -> str:
    h = hashlib.sha256()
    for x in items:
        h.update(x.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:12]

PARQUET_PATH = "/users/gbk2114/data/MIMIC_MEDS/MEDS_cohort/processed/metadata/codes.parquet"
N_SAMPLES = 100
SEED = 42
OUT_DIR = Path("/users/gbk2114/EveryQuery/src/every_query/paper_experiments/sample_codes/outputs")

random.seed(SEED)

if __name__ == "__main__":
    df = pl.read_parquet(PARQUET_PATH)
    codes = df["code"].unique().sort().to_list()

    filtered = [c for c in codes if "TIME" not in c]

    sampled = random.sample(filtered, N_SAMPLES)
    code_hash = stable_hash_list(sampled)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_fp = OUT_DIR / f"eval__N{N_SAMPLES}__seed{SEED}__{code_hash}.yaml"

    with open(out_fp, "x") as f:
        yaml.safe_dump({"codes": sampled}, f)

    print(f"Written to {out_fp}")
