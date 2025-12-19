import os
import random
import polars as pl
import yaml

# -------------------
# Config
# -------------------
PARQUET_PATH = "/users/gbk2114/data/MIMIC_MEDS/MEDS_cohort/processed/metadata/codes.parquet"

N_SAMPLES = 10000
N_REPEATS = 5          # how many ID/OOD pairs to create
N_EVAL_REPEATS = 3     # how many pairs to sample eval codes from (your script used range(3))
N_PER_GROUP = 20       # sample this many from ID and from OOD

QUERY_DIR = f"{N_SAMPLES}_sampled_codes"  # or "query_codes" if you want to keep old naming
EVAL_DIR = "eval_codes"

# -------------------
# IMPORTANT: separate RNG streams to match "two scripts with seed(42) each"
# -------------------
rng_pre = random.Random(42)   # matches script 1's RNG stream
rng_eval = random.Random(42)  # matches script 2's RNG stream

# -------------------
# Load + filter codes (keep ordering logic identical)
# -------------------
df = pl.read_parquet(PARQUET_PATH)
codes = df["code"].unique().sort().to_list()

print(f"num all codes {len(codes)}")

time_codes = [code for code in codes if "TIME" in code]
print(f"{len(time_codes)} TIME Codes removed:")
print(time_codes)

filtered_codes = [code for code in codes if "TIME" not in code]
print(f"num codes after filtering: {len(filtered_codes)}")

# -------------------
# Stage 1: sample ID sets + OOD complements (same as before)
# -------------------
os.makedirs(QUERY_DIR, exist_ok=True)

id_sets = []
ood_sets = []

for i in range(N_REPEATS):
    id_codes = rng_pre.sample(filtered_codes, N_SAMPLES)

    id_set = set(id_codes)
    ood_codes = [c for c in filtered_codes if c not in id_set]

    id_path = f"{QUERY_DIR}/{i}_{N_SAMPLES}_ID.yaml"
    ood_path = f"{QUERY_DIR}/{i}_{N_SAMPLES}_OOD.yaml"

    with open(id_path, "w") as f:
        f.write("codes:\n")
        for code in id_codes:
            f.write(f'  - "{code}"\n')

    with open(ood_path, "w") as f:
        f.write("codes:\n")
        for code in ood_codes:
            f.write(f'  - "{code}"\n')

    id_sets.append(id_codes)
    ood_sets.append(ood_codes)

print(f"Done sampling {N_SAMPLES} ID sets and OOD complements for {N_REPEATS} repeats.")

# -------------------
# Stage 2: sample eval codes from the saved pairs (same as script 2)
# -------------------
os.makedirs(EVAL_DIR, exist_ok=True)

for i in range(N_EVAL_REPEATS):

    with open(f"{QUERY_DIR}/{i}_{N_SAMPLES}_ID.yaml", "r") as f:
        id_codes = yaml.safe_load(f)["codes"]
    with open(f"{QUERY_DIR}/{i}_{N_SAMPLES}_OOD.yaml", "r") as f:
        ood_codes = yaml.safe_load(f)["codes"]

    id_sampled = rng_eval.sample(id_codes, N_PER_GROUP)
    ood_sampled = rng_eval.sample(ood_codes, N_PER_GROUP)

    out_codes = {"id": id_sampled, "ood": ood_sampled}

    with open(f"{EVAL_DIR}/{i}.yaml", "w") as f:
        yaml.safe_dump(out_codes, f)

print(f"Done sampling eval codes into {EVAL_DIR}/ for i=0..{N_EVAL_REPEATS-1}.")

