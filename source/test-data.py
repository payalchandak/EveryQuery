from meds import DataSchema
from dataset import EveryQueryPytorchDataset
from meds_torchdata import MEDSPytorchDataset, MEDSTorchBatch, MEDSTorchDataConfig
from omegaconf import DictConfig
import hydra, ipdb
from model import EveryQueryModel

model = EveryQueryModel()

@hydra.main(version_base="1.3", config_path='', config_name='config.yaml')
def main(cfg: DictConfig) -> float | None:
    print(cfg)
    dm = hydra.utils.instantiate(cfg.datamodule)
    loader = dm.train_dataloader()
    for batch in loader:
        print(batch)
        loss, outputs = model(batch)
        print(loss)
        print(outputs)
        print('Success')
        break

if __name__ == "__main__":
    main()
