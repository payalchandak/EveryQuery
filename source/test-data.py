from meds import DataSchema
from dataset import EveryQueryPytorchDataset
from meds_torchdata import MEDSPytorchDataset, MEDSTorchBatch, MEDSTorchDataConfig
from omegaconf import DictConfig
import hydra, ipdb
from model import EveryQueryModel

model = EveryQueryModel()

@hydra.main(version_base="1.3", config_path='', config_name='config.yaml')
def main(cfg: DictConfig) -> float | None:
    dm = hydra.utils.instantiate(cfg.datamodule)
    loader = dm.train_dataloader()
    for batch in loader:
        print('\n', batch)
        loss, outputs = model(batch)
        print('\n', loss)
        print('\n', outputs)
        print('Success')
        break

if __name__ == "__main__":
    main()
