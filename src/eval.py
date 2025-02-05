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

setup_resolvers()
log = RankedLogger(__name__, rank_zero_only=True)
config_yaml = files("src").joinpath("configs/eval.yaml")

@hydra.main(version_base="1.3", config_path=str(config_yaml.parent.resolve()), config_name=config_yaml.stem)
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    Args:
        cfg (DictConfig): configuration composed by Hydra.
    """
    cfg.trainer.devices = [0]

    results = {}
    for code_num in range(9): 
        cfg.data.test.codes = [f'LVEF_{code_num}']
        cfg.data.test.values_manual = {f'LVEF_{code_num}':[[0,40]]}
        configure_logging(cfg)
        metrics, objects = evaluate(cfg)
        results[code_num] = metrics
    for x in range(9): 
        print(x, results[x]['test/occurs_auc'].item())
    for x in range(9): 
        print(x, results[x]['test/censor_auc'].item())
    ipdb.set_trace()

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
