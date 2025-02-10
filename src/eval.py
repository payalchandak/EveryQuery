from importlib.resources import files

import hydra
from omegaconf import DictConfig

from meds_torch.utils import (
    RankedLogger,
    configure_logging,
)
from meds_torch.utils.resolvers import setup_resolvers

from meds_torch.eval import evaluate
from dataset import EveryQueryDataset
from models.everyquery import EveryQueryModule
from models.mlp import MLP 
import ipdb
import os 
import torch 
from tqdm import tqdm 
import polars as pl 
from pathlib import Path
from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict
import numpy as np 
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt 

setup_resolvers()
log = RankedLogger(__name__, rank_zero_only=True)
config_yaml = files("src").joinpath("configs/eval.yaml")

@hydra.main(version_base="1.3", config_path=str(config_yaml.parent.resolve()), config_name=config_yaml.stem)
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    Args:
        cfg (DictConfig): configuration composed by Hydra.
    """
    pt_0 = set(pl.read_parquet('/storage2/payal/dropbox/private/data/processed_ecg_pt/ecg_0.parquet').select(pl.col('empi')).unique().to_numpy().reshape(-1).tolist())
    pt_1 = set(pl.read_parquet('/storage2/payal/dropbox/private/data/processed_ecg_pt/ecg_1.parquet').select(pl.col('empi')).unique().to_numpy().reshape(-1).tolist())
    
    cfg.trainer.devices = [0]
    for query_code in ['ECG_0','ECG_1']: 
        cfg.data.test.codes = [query_code]
        configure_logging(cfg)
        metrics, obj = evaluate(cfg)
        predictions = obj['trainer'].predict(model=obj['model'], dataloaders=obj['datamodule'].test_dataloader(), ckpt_path=cfg.ckpt_path)
        for cohort_name, cohort in [('ECG_0',pt_0),('ECG_1',pt_1)]: 
            true, pred = [], []
            for x in predictions: 
                subj = x['batch']['context']['subject_id'].numpy().tolist()
                mask = ~x['batch']['answer']['censored'].squeeze(1)
                idx = torch.Tensor([x in cohort for x in subj])[mask].bool()
                true.extend(x['occurs_target'].reshape(-1)[idx].long().tolist())
                pred.extend(torch.sigmoid(x['occurs_logits'].reshape(-1)[idx]).tolist())
            title = f"Query {query_code} Cohort {cohort_name}"
            plt.hist(true, bins=25)
            plt.hist(pred, bins=25)
            plt.xlim(0,1)
            plt.title(title)
            plt.savefig(f"{title}.png")
            plt.close()

    # results = {}
    # for code_num in range(9): 
    #     cfg.data.test.codes = [f'LVEF_{code_num}']
    #     cfg.data.test.values_manual = {f'LVEF_{code_num}':[[0,40]]}
    #     configure_logging(cfg)
    #     metrics, objects = evaluate(cfg)
    #     results[code_num] = metrics
    # for x in range(9): 
    #     print(x, results[x]['test/occurs_auc'].item())
    # for x in range(9): 
    #     print(x, results[x]['test/censor_auc'].item())

    # eval duration over 1-12 months 
    # cfg.data.test.duration_sampling_strategy = 'fixed'
    # results = {}
    # for month in range(1,13): 
    #     cfg.data.test.fixed_duration = month * 43800
    #     configure_logging(cfg)
    #     metrics, objects = evaluate(cfg)
    #     results[month] = metrics
    # for month in range(1,13): 
    #     print(month, results[month]['test/censor_auc'].item())
    # for month in range(1,13): 
    #     print(month, results[month]['test/occurs_auc'].item())
    # ipdb.set_trace()


if __name__ == "__main__":
    main()
