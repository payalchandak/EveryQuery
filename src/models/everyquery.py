from omegaconf import DictConfig
import torch
from meds_torch.models import BACKBONE_EMBEDDINGS_KEY as EMBED, MODEL_BATCH_LOSS_KEY as LOSS
from meds_torch.models.base_model import BaseModule
from models.mlp import MLP


class EveryQueryModule(BaseModule): 
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        assert cfg.mode in [
            'supervised_context',
            'supervised_query',
        ]

        if cfg.mode == 'supervised_context':
            self.forward_fuction = self.supervised_context
            self.proj_censor = MLP(layers=[cfg.token_dim, 1], dropout_prob=0)
            self.proj_occurs = MLP(layers=[cfg.token_dim, 1], dropout_prob=0)
            self.bce = torch.nn.BCEWithLogitsLoss()

        if cfg.mode == 'supervised_query': 
            self.forward_fuction = self.supervised_query
            self.proj_query = MLP(layers=[7, cfg.token_dim], dropout_prob=0)
            self.proj_censor = MLP(layers=[cfg.token_dim*2, 1], dropout_prob=0)
            self.proj_occurs = MLP(layers=[cfg.token_dim*2, 1], dropout_prob=0)
            self.bce = torch.nn.BCEWithLogitsLoss()


    def supervised_query(self, batch): 
        context = self.model(self.input_encoder(batch['context']))

        query = self.proj_query(torch.vstack([
            batch['query']['offset'].float(),
            batch['query']['duration'].float(),
            batch['query']['code'].float(),
            batch['query']['has_value'].float(),
            batch['query']['use_value'].float(),
            batch['query']['range_lower'].float(),
            batch['query']['range_upper'].float(),
        ]).T)

        context_query_embed = torch.concat([query,context[EMBED]], dim=1)

        censor_loss = self.bce(
            input=self.proj_censor(context_query_embed),
            target=batch['answer']['censored'].float(),
        )

        mask = ~batch['answer']['censored'].squeeze(1)
        occurs_loss = self.bce(
            input=self.proj_occurs(context_query_embed[mask]),
            target=batch['answer']['occurs'][mask],
        )

        batch[LOSS] = censor_loss + occurs_loss
        return batch 
            

    def supervised_context(self, batch): 
        context = self.model(self.input_encoder(batch['context']))

        censor_loss = self.bce(
            input=self.proj_censor(context[EMBED]), 
            target=batch['answer']['censored'].float(),
        )

        mask = ~batch['answer']['censored'].squeeze(1)
        occurs_loss = self.bce(
            input=self.proj_occurs(context[EMBED][mask]),
            target=batch['answer']['occurs'][mask],
        )

        batch[LOSS] = censor_loss + occurs_loss
        return batch


    def _step(self, batch, split):

        batch = self.forward_fuction(batch)
        
        assert not torch.isnan(batch[LOSS]), f"{split} loss is NaN"
        self._log(batch, split)

        if split in ['train','val','test']: 
            return batch[LOSS]
        else:
            return batch

    def _log(self, batch, split):
        self.log(split + "/loss", batch[LOSS])

    def training_step(self, batch):
        return self._step(batch, 'train')

    def validation_step(self, batch):
        return self._step(batch, 'val')

    def test_step(self, batch):
        return self._step(batch, 'test')