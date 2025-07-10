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
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import expit
from experiment import Query

setup_resolvers()
log = RankedLogger(__name__, rank_zero_only=True)
config_yaml = files("src").joinpath("configs/eval.yaml")

@hydra.main(version_base="1.3", config_path=str(config_yaml.parent.resolve()), config_name=config_yaml.stem)
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    Args:
        cfg (DictConfig): configuration composed by Hydra.
    """

    prevalence = {}
    for code in cfg.data.dataset.codes: 
        q = Query(code=code, duration=cfg.data.dataset.fixed_duration, offset=cfg.data.dataset.fixed_offset)
        cfg.data.dataset.codes = [code]
        cfg.data.train.codes = [code]
        cfg.data.val.codes = [code]
        cfg.data.test.codes = [code]
        configure_logging(cfg)
        metrics, obj = evaluate(cfg)
        answers= []
        for batch in obj['datamodule'].train_dataloader(): 
            mask = ~batch['answer']['censored']
            answers.append(batch['answer']['occurs'][mask])
        prevalence[q] = torch.mean(torch.cat(answers)).item()

    ipdb.set_trace()

if __name__ == "__main__":
    main()