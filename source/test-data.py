from meds import DataSchema
from dataset import EveryQueryPytorchDataset
from meds_torchdata import MEDSTorchDataConfig
from omegaconf import DictConfig, OmegaConf
import hydra, ipdb
from model import EveryQueryModel
from lightning_module import EveryQueryLightningModule
from typing import Any

def values_as_list(**kwargs) -> list[Any]:
    return list(kwargs.values())

@hydra.main(version_base="1.3", config_path='', config_name='config.yaml')
def main(cfg: DictConfig) -> float | None:
    # keep the run tiny for testing
    cfg.trainer.max_steps = 3
    cfg.trainer.limit_val_batches = 1

    dm = hydra.utils.instantiate(cfg.datamodule)
    module = hydra.utils.instantiate(cfg.lightning_module)
    trainer = hydra.utils.instantiate(cfg.trainer)

    trainer.fit(model=module, datamodule=dm)
    print('Success')

if __name__ == "__main__":
    main()
