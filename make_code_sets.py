import yaml, pathlib, os, polars as pl, numpy as np

sizes = [10, 50, 100, 500, 1000, 5000, 10000]  

OUT_DIR = "src/configs/data/codes"
out_dir = pathlib.Path(OUT_DIR)
out_dir.mkdir(parents=True, exist_ok=True)

np_rng = np.random.default_rng(140799)

# PROCESSED = os.environ.get("PROCESSED") # doesnt work for some reason 
PROCESSED="/n/data1/hms/dbmi/zaklab/payal/mimic/processed"
metadata = pl.read_parquet(f'{PROCESSED}/metadata/codes.parquet')
all_codes = metadata.select("code").drop_nulls().unique().to_series().to_list()

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