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

setup_resolvers()
log = RankedLogger(__name__, rank_zero_only=True)
config_yaml = files("src").joinpath("configs/eval.yaml")

@hydra.main(version_base="1.3", config_path=str(config_yaml.parent.resolve()), config_name=config_yaml.stem)
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    Args:
        cfg (DictConfig): configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    configure_logging(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
