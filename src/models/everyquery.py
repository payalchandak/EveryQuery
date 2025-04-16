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

        assert self.cfg.projector.mode in [
            'supervised_context',
            'supervised_query',
        ]

        query_encoding_mode = {
            'stack':self.cfg.query.embed_dim,
            'triplet':self.cfg.token_dim,
        }
        assert self.cfg.query.mode in query_encoding_mode.keys()
        self.cfg.query.encod_dim = query_encoding_mode[self.cfg.query.mode]

        if self.cfg.projector.mode == 'supervised_context':
            self.embed_function = self.supervised_context
            self.proj_censor = MLP(layers=[self.cfg.token_dim, 128, 1], dropout_prob=self.cfg.projector.dropout)
            self.proj_occurs = MLP(layers=[self.cfg.token_dim, 128, 1], dropout_prob=self.cfg.projector.dropout)
            
        if self.cfg.projector.mode == 'supervised_query': 
            self.embed_function = self.supervised_query
            self.proj_query = MLP(layers=[self.cfg.query.encod_dim, self.cfg.query.embed_dim], dropout_prob=self.cfg.projector.dropout).append(torch.nn.ReLU())
            self.proj_censor = MLP(layers=[self.cfg.token_dim+self.cfg.query.embed_dim, 128, 1], dropout_prob=self.cfg.projector.dropout)
            self.proj_occurs = MLP(layers=[self.cfg.token_dim+self.cfg.query.embed_dim, 128, 1], dropout_prob=self.cfg.projector.dropout)

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
            },
            'predict': {
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
            data = {} 
        else: 
            data = {
                'censor_logits':censor_logits,
                'censor_target':censor_target,
                'occurs_logits':occurs_logits,
                'occurs_target':occurs_target,
            }

        return loss, data

    def query_encoder(self, query): 
        match self.cfg.query.mode: 
            case 'stack': 
                encoding = torch.vstack([query[k].float() for k in query.keys()]).T
            case 'triplet':
                code = self.input_encoder.code_embedder.forward(query['code']) 
                range_mask = query['has_value'].unsqueeze(1) * query['use_value'].unsqueeze(1)
                range_lower = self.input_encoder.numeric_value_embedder.forward(query['range_lower'].unsqueeze(1).float()) * range_mask 
                range_upper = self.input_encoder.numeric_value_embedder.forward(query['range_upper'].unsqueeze(1).float()) * range_mask 
                # can change to time delta days and use input encoder
                duration = query['duration'].unsqueeze(1).repeat(1,self.cfg.token_dim)
                offset = query['offset'].unsqueeze(1).repeat(1,self.cfg.token_dim)
                encoding = code + range_lower + range_upper + duration + offset 
        assert encoding.shape[1] == self.cfg.query.encod_dim
        encoding = encoding.float()
        return encoding

    def supervised_query(self, batch): 
        context = self.model(self.input_encoder(batch['context']))
        query = self.proj_query(self.query_encoder(batch['query']))
        embed = torch.concat([context[BACKBONE_EMBEDDINGS_KEY], query], dim=1)
        return embed

    def supervised_context(self, batch): 
        context = self.model(self.input_encoder(batch['context']))
        embed = context[BACKBONE_EMBEDDINGS_KEY]
        return embed 

    def _step(self, batch, split):
        embed = self.embed_function(batch)
        loss, data = self.get_loss(embed, batch['answer'], split)
        # assert not torch.isnan(loss), f"{split} loss is NaN"
        if split in ['train','val','test']: 
            return loss
        else:
            data['batch'] = batch
            data['embed'] = embed
            return data

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
    
    def predict_step(self, batch):
        return self._step(batch, 'predict')
