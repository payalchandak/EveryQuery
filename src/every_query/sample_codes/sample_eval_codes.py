import polars as pl
import random
from glob import glob
from pathlib import Path
import yaml
import os

# goal is to sample 20 random codes from each ID and OOD pair to get a 
# total of 40 codes to calculate auroc for
PARQUET_PATH = "/users/gbk2114/data/MIMIC_MEDS/MEDS_cohort/processed/metadata/codes.parquet"
SEED = 42

random.seed(SEED)

# -------------------
# Load + filter codes
# -------------------
df = pl.read_parquet(PARQUET_PATH)
all_codes = set(df["code"].unique().to_list())

training_sets = ["10000_ID__8db2be6fadf8","10000_ID__fef969fa50be","10000_ID__e267abdbc547"]

for training_set in training_sets:
    with open(f"../train_codes/{training_set}.yaml", "r") as f:
        id_codes = set(yaml.safe_load(f)["codes"])

    ood_codes = all_codes - id_codes

    id_codes = sorted(id_codes)
    ood_codes = sorted(ood_codes)

    id_sampled = random.sample(id_codes, 20)
    ood_sampled = random.sample(ood_codes, 20)

    out_codes = {
            "id": id_sampled,
            "ood": ood_sampled
            }
    os.makedirs("../eval_suite/conf/eval_codes",exist_ok=True)

    with open(f"../eval_suite/conf/eval_codes/{training_set}.yaml", "w") as f:
        yaml.safe_dump(out_codes, f)


