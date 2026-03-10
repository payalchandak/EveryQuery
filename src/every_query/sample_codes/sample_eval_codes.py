import os
import random

import polars as pl
import yaml

# goal is to sample 20 random codes from each ID and OOD pair to get a
# total of 40 codes to calculate auroc for
PARQUET_PATH = "/users/gbk2114/data/MIMIC_MEDS/MEDS_cohort/processed/metadata/codes.parquet"
SEED = 42
NUM_CODES = 5000

random.seed(SEED)

# -------------------
# Load + filter codes
# -------------------
df = pl.read_parquet(PARQUET_PATH)
all_codes = set(df["code"].unique().to_list())

training_sets = ["10000_ID__8db2be6fadf8"]

for training_set in training_sets:
    with open(f"../train_codes/{training_set}.yaml") as f:
        id_codes = set(yaml.safe_load(f)["codes"])

    ood_codes = all_codes - id_codes

    id_codes = sorted(id_codes)
    ood_codes = sorted(ood_codes)

    id_sampled = random.sample(id_codes, NUM_CODES)
    ood_sampled = random.sample(ood_codes, 500)

    out_codes = {"id": id_sampled, "ood": ood_sampled}
    os.makedirs("../eval_suite/conf/eval_codes", exist_ok=True)

    out_fp = f"../eval_suite/conf/eval_codes/{training_set}_{NUM_CODES}.yaml"

    with open(out_fp, "w") as f:
        yaml.safe_dump(out_codes, f)

    print(f"Writtten to {out_fp}")
