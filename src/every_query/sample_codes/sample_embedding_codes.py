import hashlib
import os
import random

import polars as pl

# -------------------
# Config
# -------------------
PARQUET_PATH = "/users/gbk2114/data/MIMIC_MEDS/MEDS_cohort/processed/metadata/codes.parquet"
N_SAMPLES = 200
N_REPEATS = 1
OUT_DIR = "../eval_suite/conf/eval_codes"
SEED = 42

random.seed(SEED)


def stable_hash_list(items: list[str]) -> str:
    """Order-sensitive, deterministic hash for a list of strings."""
    h = hashlib.sha256()
    for x in items:
        h.update(x.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:12]


# -------------------
# Load + filter codes
# -------------------
df = pl.read_parquet(PARQUET_PATH)
codes = df["code"].unique().sort().to_list()
print(f"num all codes {len(codes)}")

time_codes = [code for code in codes if "TIME" in code]
print(f"{len(time_codes)} TIME Codes removed:")
print(time_codes)

# Filter out time codes
filtered_codes = [code for code in codes if "TIME" not in code]
print(f"num codes after filtering: {len(filtered_codes)}")

os.makedirs(OUT_DIR, exist_ok=True)


sampled_embed_queries = random.sample(filtered_codes, N_SAMPLES)
hash = stable_hash_list(sampled_embed_queries)

# ---- write file ----
out_path = f"{OUT_DIR}/embed_{N_SAMPLES}_{hash}.yaml"
with open(out_path, "x") as f:
    for code in sampled_embed_queries:
        f.write(f'- "{code}"\n')


print(f"Done sampling {N_SAMPLES} queries for embedding plots")
print(f"Saved @ {out_path}")
