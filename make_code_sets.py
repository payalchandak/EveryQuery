import re, yaml, pathlib, os, polars as pl, numpy as np

sizes = [10, 100, 1000, 10000]  

OUT_DIR = "src/configs/data/codes"
out_dir = pathlib.Path(OUT_DIR)
out_dir.mkdir(parents=True, exist_ok=True)

np_rng = np.random.default_rng(140799)

# PROCESSED = os.environ.get("PROCESSED") # doesnt work for some reason 
PROCESSED="/n/data1/hms/dbmi/zaklab/payal/mimic/processed"
metadata = pl.read_parquet(f'{PROCESSED}/metadata/codes.parquet')
all_codes = metadata.select("code").drop_nulls().unique().to_series().to_list()

MAX_SIZE = sizes[-1]
if len(all_codes) < MAX_SIZE:
    raise ValueError(f"Need at least {MAX_SIZE} unique codes; only found {len(all_codes)}.")

cumulative = set()
prev_size = 0
for i in range(len(sizes)):
    num_to_add = sizes[i] - prev_size
    if num_to_add < 0:
        raise ValueError(f"Stage {i} has smaller total ({sizes[i]}) than previous stage ({prev_size})")
    pool = np.array(sorted(set(all_codes) - cumulative))
    new_codes = np_rng.choice(pool, size=num_to_add, replace=False).tolist()
    cumulative.update(new_codes)
    out_path = out_dir / f"stage_{i}.yaml"
    with open(out_path, "w") as f:
        f.write(f"# {len(cumulative)} codes\n")
        yaml.safe_dump(sorted(cumulative), f)
    print(f"Saved stage {i} with {len(cumulative)} codes to {out_path}")
    prev_size = sizes[i]

hold_out = sorted(set(all_codes) - cumulative)
out_path = out_dir / f"hold_out.yaml"
with open(out_path, "w") as f:
    f.write(f"# {len(hold_out)} codes\n")
    yaml.safe_dump(hold_out, f)
print(f"Saved hold out codes with {len(hold_out)} codes to {out_path}")

# tests 

stage_files = sorted(
    out_dir.glob("stage_*.yaml"),
    key=lambda p: int(re.search(r"\d+$", p.stem).group()),
)

stage_sets = []
for p in stage_files:
    with open(p) as fh:
        # Strip comment lines that begin with '#'
        data = yaml.safe_load("".join(line for line in fh if not line.lstrip().startswith("#")))
        stage_sets.append(set(data))

with open(out_dir / "hold_out.yaml") as fh:
    hold_out_set = set(
        yaml.safe_load("".join(line for line in fh if not line.lstrip().startswith("#")))
    )

actual_sizes = [len(s) for s in stage_sets]
assert actual_sizes == sizes, f"Stage sizes mismatch: {actual_sizes} ≠ {sizes}"

for i in range(1, len(stage_sets)):
    assert stage_sets[i - 1] <= stage_sets[i], (
        f"stage with {sizes[i-1]} codes is *not* a subset of stage with {sizes[i]} codes"
    )

assert hold_out_set.isdisjoint(stage_sets[-1]), "Hold‑out overlaps with staged codes"

assert stage_sets[-1] | hold_out_set == set(all_codes), (
    "Union of largest stage and hold‑out does not equal the full code universe"
)

print("All sanity tests passed ✔")
