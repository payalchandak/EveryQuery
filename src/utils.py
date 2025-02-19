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
    final_codes = None # only create updated codes once 
    for k in cfg.data.keys(): 
        dataset = cfg.data.get(k)
        if not isinstance(dataset, DictConfig): continue 
        if dataset.get('_target_','') == 'dataset.EveryQueryDataset.initialize': 
            if not any([x.startswith(prefix) for x in dataset.codes]): continue 
            if final_codes is None:
                codes = []
                for x in dataset.codes: 
                    if x.startswith(prefix): 
                        random = sample_codes(
                            n=int(x.replace(prefix,'')),
                            seed=None # (TODO) cfg.seed later for reproducibility
                        )
                        codes.extend(random)
                    else: 
                        codes.append(x)
                final_codes = codes
            cfg.data[k].codes = final_codes
    return cfg 
    

def sample_codes(n=1, seed=None):
    codes = pl.read_parquet(f"{os.getenv('PROCESSED')}/metadata/codes.parquet")
    sample = codes.sample(
        n=n, 
        shuffle=True,
        with_replacement=False,
        seed=seed,
    )
    return sample.get_column('code').to_list()