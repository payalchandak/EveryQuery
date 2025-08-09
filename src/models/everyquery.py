from omegaconf import DictConfig
import torch
from torchmetrics.classification import BinaryAUROC
from meds_torch.models import BACKBONE_EMBEDDINGS_KEY, BACKBONE_TOKENS_KEY
from meds_torch.input_encoder import INPUT_ENCODER_MASK_KEY
from meds_torch.models.base_model import BaseModule
from models.mlp import MLP

class EveryQueryModule(BaseModule): 
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.criterion = torch.nn.BCEWithLogitsLoss()

        assert self.cfg.projector.mode in [
            'supervised_context',
            'supervised_query',
            'separate_censor_occurs',
            # New fusion modes that use cross-attention from query→context tokens
            'ca_shared',               # one fused embedding used for both heads
            'ca_separate',             # separate fused embeddings for censor/occurs (future vs query)
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

        if self.cfg.projector.mode == 'separate_censor_occurs': 
            self.embed_function = self.separate_censor_occurs
            self.proj_query = MLP(layers=[self.cfg.query.encod_dim, self.cfg.query.embed_dim], dropout_prob=self.cfg.projector.dropout).append(torch.nn.ReLU())
            self.proj_censor = MLP(layers=[self.cfg.token_dim+self.cfg.query.embed_dim, 128, 1], dropout_prob=self.cfg.projector.dropout)
            self.proj_occurs = MLP(layers=[self.cfg.token_dim+self.cfg.query.embed_dim, 128, 1], dropout_prob=self.cfg.projector.dropout)

        # Cross-attention fusion modes
        if self.cfg.projector.mode in ['ca_shared', 'ca_separate']:
            self.embed_function = self.cross_attention_fusion if self.cfg.projector.mode == 'ca_shared' else self.cross_attention_fusion_separate
            # Project query (or future) encoding up to token_dim to act as attention query
            self.proj_query = MLP(layers=[self.cfg.query.encod_dim, self.cfg.token_dim], dropout_prob=self.cfg.projector.dropout).append(torch.nn.ReLU())
            # Multihead attention: query (1 token) attends over context tokens
            nheads = getattr(self.cfg.backbone, 'nheads', 4)
            self.cross_attn = torch.nn.MultiheadAttention(embed_dim=self.cfg.token_dim, num_heads=nheads, batch_first=True)
            # Heads operate on fused token_dim embeddings
            self.proj_censor = MLP(layers=[self.cfg.token_dim, 128, 1], dropout_prob=self.cfg.projector.dropout)
            self.proj_occurs = MLP(layers=[self.cfg.token_dim, 128, 1], dropout_prob=self.cfg.projector.dropout)
        
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
        if self.cfg.projector.mode in ['separate_censor_occurs', 'ca_separate']:
            censor_embed, occurs_embed = embed
        else: 
            occurs_embed = embed
            censor_embed = embed

        censor_logits = self.proj_censor(censor_embed)
        censor_target = answer['censored'].float()
        censor_loss = self.criterion(censor_logits, censor_target)
        self.update_metric(name='censor_auc', split=split, preds=censor_logits.squeeze(1).sigmoid(), target=censor_target.squeeze(1).int())

        mask = ~answer['censored'].squeeze(1)
        occurs_logits = self.proj_occurs(occurs_embed[mask])
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
    
    def future_encoder(self, query): 
        match self.cfg.query.mode: 
            case 'stack': 
                future = query['duration'] + query['offset']
                encoding = torch.vstack([future.float() for _ in query.keys()]).T
            case 'triplet':
                assert self.cfg.token_dim % 2 == 0, "token_dim must be even"
                duration = query['duration'].unsqueeze(1).repeat(1,self.cfg.token_dim//2)
                offset = query['offset'].unsqueeze(1).repeat(1,self.cfg.token_dim//2)
                encoding = torch.concat([duration, offset], dim=1)
        assert encoding.shape[1] == self.cfg.query.encod_dim
        encoding = encoding.float()
        return encoding
    
    def separate_censor_occurs(self, batch): 
        encoded_context = self.input_encoder(batch['context'])
        context = self.model(encoded_context)
        future = self.proj_query(self.future_encoder(batch['query'])) # reuse query encoder
        censor_embed = torch.concat([context[BACKBONE_EMBEDDINGS_KEY], future], dim=1)
        query = self.proj_query(self.query_encoder(batch['query']))
        occurs_embed = torch.concat([context[BACKBONE_EMBEDDINGS_KEY], query], dim=1)
        return (censor_embed, occurs_embed)

    def supervised_query(self, batch): 
        encoded_context = self.input_encoder(batch['context'])
        context = self.model(encoded_context)
        query = self.proj_query(self.query_encoder(batch['query']))
        embed = torch.concat([context[BACKBONE_EMBEDDINGS_KEY], query], dim=1)
        return embed

    def supervised_context(self, batch): 
        encoded_context = self.input_encoder(batch['context'])
        context = self.model(encoded_context)
        embed = context[BACKBONE_EMBEDDINGS_KEY]
        return embed 

    def _ca(self, query_vec: torch.Tensor, context_tokens: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        """Run single-token cross-attention from query_vec over context_tokens.

        Shapes:
          - query_vec: (B, D)
          - context_tokens: (B, S, D)
          - key_padding_mask: (B, S) with True for masked (padded) positions
        Returns:
          - fused: (B, D)
        """
        q = query_vec.unsqueeze(1)  # (B, 1, D)
        fused, _ = self.cross_attn(q, context_tokens, context_tokens, key_padding_mask=key_padding_mask)
        return fused.squeeze(1)

    def cross_attention_fusion(self, batch):
        """Shared fused embedding for both heads using the query encoding."""
        encoded_context = self.input_encoder(batch['context'])
        # meds-torch sets INPUT_ENCODER_MASK_KEY to True for valid tokens; MultiheadAttention expects True for padding -> invert
        key_padding_mask = ~encoded_context[INPUT_ENCODER_MASK_KEY]
        context = self.model(encoded_context)
        context_tokens = context[BACKBONE_TOKENS_KEY]  # (B, S, D)

        query = self.proj_query(self.query_encoder(batch['query']))  # (B, D)
        fused = self._ca(query, context_tokens, key_padding_mask)
        return fused

    def cross_attention_fusion_separate(self, batch):
        """Separate fused embeddings for censor and occurs heads.

        - Censor: fuse with future window (offset+duration)
        - Occurs: fuse with full query (code ± value, duration, offset)
        """
        encoded_context = self.input_encoder(batch['context'])
        # meds-torch mask True=valid; MHA needs True=pad
        key_padding_mask = ~encoded_context[INPUT_ENCODER_MASK_KEY]
        context = self.model(encoded_context)
        context_tokens = context[BACKBONE_TOKENS_KEY]  # (B, S, D)

        future = self.proj_query(self.future_encoder(batch['query']))  # (B, D)
        query = self.proj_query(self.query_encoder(batch['query']))    # (B, D)

        censor_embed = self._ca(future, context_tokens, key_padding_mask)
        occurs_embed = self._ca(query, context_tokens, key_padding_mask)
        return (censor_embed, occurs_embed)

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
