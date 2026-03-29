import hashlib
import os
import random

import polars as pl

# -------------------
# Config
# -------------------
PARQUET_PATH = "/users/gbk2114/data/MIMIC_MEDS/MEDS_cohort/processed/metadata/codes.parquet"
N_SAMPLES = 10000
N_REPEATS = 5
OUT_DIR = "../train_codes"
SEED = 42

random.seed(SEED)


def stable_hash_list(items: list[str]) -> str:
    """Order-sensitive, deterministic hash for a list of strings."""
    h = hashlib.sha256()
    for x in items:
        h.update(x.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:12]


if __name__ == "__main__":
    # -------------------
    # Load + filter codes
    # -------------------
    df = pl.read_parquet(PARQUET_PATH)
    codes = df["code"].unique().sort().to_list()
    print(f"num all codes {len(codes)}")

    time_codes = [code for code in codes if "TIME" in code]
    print(f"{len(time_codes)} TIME Codes removed:")
    print(time_codes)

    filtered_codes = [code for code in codes if "TIME" not in code]
    print(f"num codes after filtering: {len(filtered_codes)}")

    os.makedirs(OUT_DIR, exist_ok=True)

    # -------------------
    # Sample ID + OOD sets
    # -------------------
    for _ in range(N_REPEATS):
        # ID = sampled codes
        id_codes = random.sample(filtered_codes, N_SAMPLES)

        # OOD = everything NOT in ID
        id_set = set(id_codes)
        ood_codes = [c for c in filtered_codes if c not in id_set]

        # Hashes define identity of the code universes
        id_hash = stable_hash_list(id_codes)
        ood_hash = stable_hash_list(ood_codes)

        # ---- write ID file ----
        id_path = f"{OUT_DIR}/{N_SAMPLES}_ID__{id_hash}.yaml"
        with open(id_path, "x") as f:
            f.write("codes:\n")
            for code in id_codes:
                f.write(f'  - "{code}"\n')

        # # ---- write OOD file ----
        # ood_path = f"{OUT_DIR}/{N_SAMPLES}_OOD__{ood_hash}.yaml"
        # with open(ood_path, "x") as f:
        #     f.write("codes:\n")
        #     for code in ood_codes:
        #         f.write(f'  - "{code}"\n')
        #
        # # Optional manifest line (highly recommended)
        # with open(f"{OUT_DIR}/MANIFEST.txt", "a") as f:
        #     f.write(
        #         f"ID {id_hash}  -> {os.path.basename(id_path)}\n"
        #         f"OOD {ood_hash} -> {os.path.basename(ood_path)}\n"
        #     )

    print(f"Done sampling {N_SAMPLES} ID sets and OOD complements for {N_REPEATS} repeats.")
