import re
import os
import polars as pl 
from omegaconf import DictConfig

def minutes(x):
    if isinstance(x, int): return x
    assert isinstance(x, str)
    x = x.lower().strip()
    if x.isdigit() or x.endswith("min"):
        return int(x.replace("min", ""))
    match = re.match(r"^(\d+)([ymwdh])$", x)
    if not match:
        raise ValueError(f"Invalid time format: {x}")
    num, unit = int(match.group(1)), match.group(2)
    units = {'y': 525600, 'm': 43800, 'w': 10080, 'd': 1440, 'h': 60}
    assert unit in units.keys()
    return num * units.get(unit) 

def resolve_random_codes(cfg): 
    prefix = 'RANDOM//'
    final_codes = None  # Cache to ensure codes are consistent across train/val/test
    for k, dataset in cfg.data.items():
        if not isinstance(dataset, DictConfig): continue 
        if dataset.get('_target_','') != 'dataset.EveryQueryDataset.initialize': continue
        if not any([x.startswith(prefix) for x in dataset.codes]): break 
        if final_codes is None:
            codes = [x for x in dataset.codes if not x.startswith(prefix)]
            for x in dataset.codes: 
                if x.startswith(prefix): 
                    # (TODO) cfg.seed later for reproducibility
                    codes.extend(sample_codes(n=int(x.replace(prefix,'')),seed=None))
            final_codes = codes 
            # (TODO) what if you sample a code that's manually provided 
        cfg.data[k].codes = final_codes
    return cfg

def sample_codes(n=1, seed=None):
    codes = pl.read_parquet(f"{os.getenv('PROCESSED')}/metadata/codes.parquet")
    return codes.sample(n=n, shuffle=True, with_replacement=False, seed=seed)["code"].to_list()
