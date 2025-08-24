import json

import pandas as pd
import yaml
from omegaconf import DictConfig

from aces import config, predicates, query

import ipdb

config_path = "/home/pac4279/EveryQuery/src/configs/aces_imminent_mortality.yaml"
data_path = "/n/data1/hms/dbmi/zaklab/payal/mimic/processed/normalization"

with open(config_path) as stream:
    data_loaded = yaml.safe_load(stream)
    print(json.dumps(data_loaded, indent=4))

cfg = config.TaskExtractorConfig.load(config_path=config_path)

data_config = DictConfig({"path": data_path, "standard": "meds", "data": "sharded"})

predicates_df = predicates.get_predicates_df(cfg=cfg, data_config=data_config)

df_result = query.query(cfg=cfg, predicates_df=predicates_df)

ipdb.set_trace()