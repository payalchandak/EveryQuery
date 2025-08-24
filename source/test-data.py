from meds import DataSchema
from meds_torchdata import MEDSPytorchDataset, MEDSTorchBatch, MEDSTorchDataConfig
from omegaconf import DictConfig
import hydra, ipdb

from transformers import AutoConfig, ModernBertModel

config = AutoConfig.from_pretrained("answerdotai/ModernBERT-large")
model = ModernBertModel._from_config(config)

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
        outputs = model(**hf_inputs)
        print(outputs.last_hidden_state.shape)
        ipdb.set_trace()
        break

if __name__ == "__main__":
    main()