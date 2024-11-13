from omegaconf import DictConfig
import torch
from meds_torch.models import BACKBONE_TOKENS_KEY, MODEL_BATCH_LOSS_KEY as LOSS
from meds_torch.models.base_model import BaseModule

class EveryQueryModule(BaseModule): 
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def forward(self, batch):
        batch = self.input_encoder(batch)
        model_output = self.model(batch)
        batch = self.get_loss(batch)
        return batch
    
    def get_loss(self, batch): 
        batch[LOSS] = 0
        return batch 

    def _log(self, batch, split):
        self.log(split + "/loss", batch[LOSS])

    def training_step(self, batch):
        batch = self(batch)
        assert not torch.isnan(batch[LOSS]), "Loss is NaN"
        self._log(batch, "train")
        return batch[LOSS]

    def validation_step(self, batch):
        batch = self(batch)
        assert not torch.isnan(batch[LOSS]), "Loss is NaN"
        self._log(batch, "val")
        return batch[LOSS]

    def test_step(self, batch):
        batch = self(batch)
        assert not torch.isnan(batch[LOSS]), "Loss is NaN"
        self._log(batch, "test")
        return batch[LOSS]