from meds import DataSchema
from dataset import EveryQueryPytorchDataset
from meds_torchdata import MEDSPytorchDataset, MEDSTorchBatch, MEDSTorchDataConfig
from omegaconf import DictConfig
import hydra, ipdb
from model import Model

model = Model()

@hydra.main(version_base="1.3", config_path='', config_name='config.yaml')
def main(cfg: DictConfig) -> float | None:
    print(cfg)
    dm = hydra.utils.instantiate(cfg.datamodule)
    loader = dm.train_dataloader()
    for batch in loader:
        hf_inputs = {
            "input_ids": batch.code,
            "attention_mask": (batch.code != batch.PAD_INDEX),
        }
        #outputs = model(**hf_inputs)
        outputs = model(batch)
        print(outputs)
        print('Success')
        # ipdb.set_trace()
        break

if __name__ == "__main__":
    main()
