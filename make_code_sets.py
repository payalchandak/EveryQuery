import re, yaml, random, pathlib, os, polars as pl, numpy as np

# ------------ user‑configurable parameters ------------------------
SIZES      = [1, 3, 5, 8, 10, 100, 1000, 10000]        # grow this list any time
SEED       = 140799                          # keep constant for reproducibility
OUT_DIR    = "src/configs/data/codes"        # root output directory
PROCESSED  = os.getenv("PROCESSED", "/n/data1/hms/dbmi/zaklab/payal/mimic/processed")
# ------------------------------------------------------------------

np_rng       = np.random.default_rng(SEED)
out_dir      = pathlib.Path(OUT_DIR)
out_dir.mkdir(parents=True, exist_ok=True)

# 1️⃣  Load the universe of codes ------------------------------------------------
metadata   = pl.read_parquet(f"{PROCESSED}/metadata/codes.parquet")
all_codes  = sorted(metadata.select("code").drop_nulls().unique().to_series().to_list()) # must sorted to gaurantee consistent shuffle
N          = len(all_codes)

if N < max(SIZES):
    raise ValueError(f"Need at least {max(SIZES)} unique codes in metadata, but only {N} found.")

# 2️⃣  Canonical shuffled order for this SEED ------------------------------------
shuffle_path = out_dir / f"shuffled_seed_{SEED}.yaml"

if shuffle_path.exists():
    # Re‑use the canonical order
    with open(shuffle_path) as fh:
        shuffled = yaml.safe_load(fh)
    if set(shuffled) != set(all_codes):
        raise RuntimeError(
            "The universe of codes has changed since the shuffle file was created.\n"
            "Either regenerate everything in a fresh folder or update PROCESSED."
        )
else:
    shuffled = all_codes.copy()
    np_rng.shuffle(shuffled)
    print(shuffled[14])
    print(shuffled[7])
    print(shuffled[1999])
    with open(shuffle_path, "w") as fh:
        yaml.safe_dump(shuffled, fh, width=float("inf"))
    print(f"Saved canonical shuffle of {N} codes → {shuffle_path}")

# 3️⃣  Generate / validate stage files ------------------------------------------
for size in sorted(SIZES):
    stage_path = out_dir / f"N_{size}.yaml"
    expected = shuffled[:size]

    if stage_path.exists():
        # Verify consistency
        with open(stage_path) as fh:
            actual = yaml.safe_load(fh)
        if set(actual) != set(expected):
            raise RuntimeError(
                f"{stage_path} already exists but does not match the expected "
                f"first {size} codes for seed {SEED}. Aborting to protect data."
            )
        print(f"[OK] N={size}: existing file is consistent.")
    else:
        with open(stage_path, "w") as fh:
            yaml.safe_dump(expected, fh, width=float("inf"))
        print(f"Created traing set with {size} codes → {stage_path}")

# 4️⃣  Generate / update hold‑out -----------------------------------------------
largest_stage = max(SIZES)
hold_out  = shuffled[largest_stage:]
hold_path     = out_dir / "hold_out.yaml"

def write_holdout():
    with open(hold_path, "w") as fh:
        yaml.safe_dump(hold_out, fh, width=float("inf"))

if hold_path.exists():
    with open(hold_path) as fh:
        existing = yaml.safe_load(fh)
    if set(existing) == set(hold_out):
        print(f"[OK] Hold‑out: existing file is consistent.")
    else:
        print("Updating hold‑out to account for new largest train set size …")
        write_holdout()
else:
    write_holdout()
    print(f"Created hold‑out with {len(hold_out)} codes → {hold_path}")

# 5️⃣  Embedded sanity tests -----------------------------------------------------
if __name__ == "__main__":
    # Same assertions as before, updated to use the deterministic shuffled list
    assert all(
        (out_dir / f"N_{s}.yaml").exists() for s in SIZES
    ), "Some train set files are missing!"

    for i in range(1, len(SIZES)):
        prev = set(shuffled[:SIZES[i - 1]])
        curr = set(shuffled[:SIZES[i]])
        assert prev <= curr, f"Nestedness failed: {SIZES[i-1]} ⊄ {SIZES[i]}"

    assert set(hold_out).isdisjoint(shuffled[:largest_stage]), "Hold‑out overlaps."
    assert len(shuffled[:largest_stage]) + len(hold_out) == N, "Coverage error."
    print("All sanity tests passed ✔")