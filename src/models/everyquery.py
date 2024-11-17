from omegaconf import DictConfig
import torch
from torcheval.metrics.functional import binary_auroc
from meds_torch.models import BACKBONE_EMBEDDINGS_KEY as EMBED, MODEL_BATCH_LOSS_KEY as LOSS
from meds_torch.models.base_model import BaseModule
from models.mlp import MLP


class EveryQueryModule(BaseModule): 
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        self.bce = torch.nn.BCEWithLogitsLoss()

        assert cfg.mode in [
            'supervised_context',
            'supervised_query',
        ]

        if cfg.mode == 'supervised_context':
            self.forward_fuction = self.supervised_context
            self.proj_censor = MLP(layers=[cfg.token_dim, 1], dropout_prob=0)
            self.proj_occurs = MLP(layers=[cfg.token_dim, 1], dropout_prob=0)
            
        if cfg.mode == 'supervised_query': 
            self.forward_fuction = self.supervised_query
            self.proj_query = MLP(layers=[7, cfg.token_dim], dropout_prob=0)
            self.proj_censor = MLP(layers=[cfg.token_dim*2, 1], dropout_prob=0)
            self.proj_occurs = MLP(layers=[cfg.token_dim*2, 1], dropout_prob=0)

    def _apply_bce(self, logits, target, mask=None): 
        if mask is not None: 
            logits = logits[mask]
            target = target[mask]
        loss = self.bce(logits, target)
        auc = binary_auroc(logits.squeeze(1), target.squeeze(1))
        return loss, auc


    def supervised_query(self, batch): 
        context = self.model(self.input_encoder(batch['context']))

        query = self.proj_query(
            torch.vstack([batch['query'][k].float() for k in batch['query'].keys()]).T
        )

        context_query_embed = torch.concat([context[EMBED],query], dim=1)

        batch['censor_loss'], batch['censor_auc'] = self._apply_bce(
            logits=self.proj_censor(context_query_embed),
            target=batch['answer']['censored'].float(),
        )

        batch['occurs_loss'], batch['occurs_auc'] = self._apply_bce(
            logits=self.proj_occurs(context_query_embed),
            target=batch['answer']['occurs'],
            mask=~batch['answer']['censored'].squeeze(1),
        )

        batch[LOSS] = batch['censor_loss'] + batch['occurs_loss']
        return batch 
            

    def supervised_context(self, batch): 
        context = self.model(self.input_encoder(batch['context']))

        batch['censor_loss'], batch['censor_auc'] = self._apply_bce(
            logits=self.proj_censor(context[EMBED]), 
            target=batch['answer']['censored'].float(),
        )

        batch['occurs_loss'], batch['occurs_auc'] = self._apply_bce(
            logits=self.proj_occurs(context[EMBED]),
            target=batch['answer']['occurs'],
            mask=~batch['answer']['censored'].squeeze(1),
        )

        batch[LOSS] = batch['censor_loss'] + batch['occurs_loss']
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
        for k in batch.keys(): 
            if k.endswith('_loss') or k.endswith('_auc'): 
                self.log(split + f"/{k}", batch[k])

    def training_step(self, batch):
        return self._step(batch, 'train')

    def validation_step(self, batch):
        return self._step(batch, 'val')

    def test_step(self, batch):
        return self._step(batch, 'test')