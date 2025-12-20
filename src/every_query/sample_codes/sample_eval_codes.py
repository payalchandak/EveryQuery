import polars as pl
import random
from glob import glob
from pathlib import Path
import yaml
import os
random.seed(42)

# goal is to sample 20 random codes from each ID and OOD pair to get a 
# total of 40 codes to calculate auroc for

for i in range(3):
    with open(f"query_codes/{i}_10000_ID.yaml", "r") as f:
        id_codes = yaml.safe_load(f)["codes"]
    with open(f"query_codes/{i}_10000_OOD.yaml","r") as f:
        ood_codes = yaml.safe_load(f)["codes"]

    id_sampled = random.sample(id_codes, 20)
    ood_sampled = random.sample(ood_codes, 20)

    out_codes = {
            "id": id_sampled,
            "ood": ood_sampled
            }
    os.makedirs("eval_codes",exist_ok=True)

    with open(f"eval_codes/{i}.yaml", "w") as f:
        yaml.safe_dump(out_codes, f)


