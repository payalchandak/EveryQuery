from omegaconf import DictConfig
import torch
from torchmetrics.classification import BinaryAUROC
from meds_torch.models import BACKBONE_EMBEDDINGS_KEY
from meds_torch.models.base_model import BaseModule
from models.mlp import MLP

class EveryQueryModule(BaseModule): 
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.criterion = torch.nn.BCEWithLogitsLoss()

        assert cfg.projector.mode in [
            'supervised_context',
            'supervised_query',
        ]

        if cfg.projector.mode == 'supervised_context':
            self.embed_function = self.supervised_context
            self.proj_censor = MLP(layers=[cfg.token_dim, 1], dropout_prob=cfg.projector.dropout)
            self.proj_occurs = MLP(layers=[cfg.token_dim, 1], dropout_prob=cfg.projector.dropout)
            
        if cfg.projector.mode == 'supervised_query': 
            self.embed_function = self.supervised_query
            self.proj_query = MLP(layers=[7, cfg.token_dim], dropout_prob=cfg.projector.dropout)
            self.proj_censor = MLP(layers=[cfg.token_dim*2, 1], dropout_prob=cfg.projector.dropout)
            self.proj_occurs = MLP(layers=[cfg.token_dim*2, 1], dropout_prob=cfg.projector.dropout)

        self.metrics = {
            'train': {
                'censor_auc': BinaryAUROC(),
                'occurs_auc': BinaryAUROC(),
            },
            'val': {
                'censor_auc': BinaryAUROC(),
                'occurs_auc': BinaryAUROC(),
            },
            'test': {
                'censor_auc': BinaryAUROC(),
                'occurs_auc': BinaryAUROC(),
            }
        }

    def update_metric(self, name, split, **kwargs): 
        assert name in self.metrics[split], f"Metric '{name}' not found in {split} metrics."
        self.metrics[split][name].update(**kwargs)
        
    def get_loss(self, embed, answer, split): 
        censor_logits = self.proj_censor(embed)
        censor_target = answer['censored'].float()
        censor_loss = self.criterion(censor_logits, censor_target)
        self.update_metric(name='censor_auc', split=split, preds=censor_logits.squeeze(1).sigmoid(), target=censor_target.squeeze(1).int())

        mask = ~answer['censored'].squeeze(1)
        occurs_logits = self.proj_occurs(embed[mask])
        occurs_target = answer['occurs'][mask]
        occurs_loss = self.criterion(occurs_logits, occurs_target)
        self.update_metric(name='occurs_auc', split=split, preds=occurs_logits.squeeze(1).sigmoid(), target=occurs_target.squeeze(1).int())

        loss = censor_loss + occurs_loss

        if split in ['train','val','test']: 
            self.log(f'{split}/loss', loss)
            self.log(f'{split}/censor_loss', censor_loss)
            self.log(f'{split}/occurs_loss', occurs_loss)

        return loss 

    def supervised_query(self, batch): 
        context = self.model(self.input_encoder(batch['context']))
        query = self.proj_query(
            torch.vstack([batch['query'][k].float() for k in batch['query'].keys()]).T
        )
        embed = torch.concat([context[BACKBONE_EMBEDDINGS_KEY], query], dim=1)
        return embed

    def supervised_context(self, batch): 
        context = self.model(self.input_encoder(batch['context']))
        embed = context[BACKBONE_EMBEDDINGS_KEY]
        return embed 

    def _step(self, batch, split):
        embed = self.embed_function(batch)
        loss = self.get_loss(embed, batch['answer'], split)
        assert not torch.isnan(loss), f"{split} loss is NaN"
        if split in ['train','val','test']: 
            return loss
        else:
            return embed

    def on_epoch_end(self, split):
        for metric_name, metric in self.metrics[split].items():
            self.log(f'{split}/{metric_name}', metric.compute(), sync_dist=True)
            metric.reset()
        # self.log(f'{split}/censor_auc', self.metrics[split]['censor_auc'].compute())
        # self.log(f'{split}/occurs_auc', self.metrics[split]['occurs_auc'].compute())
        # self.metrics[split]['censor_auc'].reset()
        # self.metrics[split]['occurs_auc'].reset()

    def on_train_epoch_end(self):
        self.on_epoch_end('train')
    
    def on_validation_epoch_end(self):
        self.on_epoch_end('val')

    def on_test_epoch_end(self):
        self.on_epoch_end('test')

    def training_step(self, batch):
        return self._step(batch, 'train')

    def validation_step(self, batch):
        return self._step(batch, 'val')

    def test_step(self, batch):
        return self._step(batch, 'test')
