from importlib.resources import files

import hydra
from omegaconf import DictConfig

from meds_torch.utils import (
    RankedLogger,
    configure_logging,
)
from meds_torch.utils.resolvers import setup_resolvers

from meds_torch.eval import evaluate
from dataset import EveryQueryDataset
from models.everyquery import EveryQueryModule
from models.mlp import MLP 
import ipdb
import os 
import torch 
from tqdm import tqdm 
import polars as pl 
from pathlib import Path
from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict
import numpy as np 
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import expit

setup_resolvers()
log = RankedLogger(__name__, rank_zero_only=True)
config_yaml = files("src").joinpath("configs/eval.yaml")

@hydra.main(version_base="1.3", config_path=str(config_yaml.parent.resolve()), config_name=config_yaml.stem)
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    Args:
        cfg (DictConfig): configuration composed by Hydra.
    """
    pt_0 = set(pl.read_parquet('/storage2/payal/dropbox/private/data/processed_ecg_pt/ecg_0.parquet').select(pl.col('empi')).unique().to_numpy().reshape(-1).tolist())
    pt_1 = set(pl.read_parquet('/storage2/payal/dropbox/private/data/processed_ecg_pt/ecg_1.parquet').select(pl.col('empi')).unique().to_numpy().reshape(-1).tolist())

    
    code_0 = 'ECG_0'
    code_1 = 'ECG_1'
    metadata_df = pl.read_parquet('/storage2/payal/EveryQuery/data/mgb_ecg_pt/processed/metadata/codes.parquet')
    vocab_code_0 = metadata_df.filter(pl.col('code')==code_0).select('code/vocab_index').item()
    vocab_code_1 = metadata_df.filter(pl.col('code')==code_1).select('code/vocab_index').item()

    cfg.trainer.devices = [2]
    results = []
    for query_code in [code_0,code_1]: 
        cfg.data.dataset.codes = [query_code]
        cfg.data.train.codes = [query_code]
        cfg.data.val.codes = [query_code]
        cfg.data.test.codes = [query_code]
        configure_logging(cfg)
        metrics, obj = evaluate(cfg)
        predictions = obj['trainer'].predict(model=obj['model'], dataloaders=obj['datamodule'].test_dataloader(), ckpt_path=cfg.ckpt_path)
        for x in predictions: # expects batch size of 1
            r = {
                'query_name':query_code,
                'query_code':x['batch']['query']['code'].item(),
                'censor_logits':x['censor_logits'].item(),
                'censor_target':x['censor_target'].item(),
                'subject_id':x['batch']['context']['subject_id'].item(),      
                'context_has_code_0': vocab_code_0 in x['batch']['context']['code'].reshape(-1).tolist(),
                'context_has_code_1': vocab_code_1 in x['batch']['context']['code'].reshape(-1).tolist(),
                'query_has_code_0': x['batch']['query_has_code_0'].item(),
                'query_has_code_1': x['batch']['query_has_code_1'].item(),
            }
            if x['occurs_logits'].numel(): 
                r['occurs_logits'] = x['occurs_logits'].item()
                r['occurs_target'] = x['occurs_target'].item()
            results.append(r)

    df = pl.DataFrame(results)
    df = df.with_columns(
        (pl.col('context_has_code_0') | pl.col('context_has_code_1')).alias('context_has_history'), 
        pl.when(
            ((pl.col('query_name')=='ECG_0') & (pl.col('subject_id').is_in(pt_0))) 
            | 
            ((pl.col('query_name')=='ECG_1') & (pl.col('subject_id').is_in(pt_1))) 
        ).then(True).otherwise(False).alias('query_aligned_pt')
    )

    def occurs_auc(df):
        df = df.filter(pl.col('occurs_logits').is_not_null())
        labels = df.select(['occurs_target']).to_numpy().reshape(-1)
        logits = df.select(['occurs_logits']).to_numpy().reshape(-1)
        try:
            auc = roc_auc_score(labels, logits)
        except: 
            auc = None 
        return auc
    
    for query_name, pt_cohort in [('ECG_0',pt_0),('ECG_1',pt_1)]: 
        sub = df.filter(pl.col('query_name')==query_name)
        print('All patients', occurs_auc(sub))
        print('context has ECG0', occurs_auc(sub.filter(pl.col('context_has_code_0'))) )
        print('context does not have ECG0', occurs_auc(sub.filter(~pl.col('context_has_code_0'))) )
        print('context has ECG1', occurs_auc(sub.filter(pl.col('context_has_code_1'))) )
        print('context has either ECG10 or ECG1', occurs_auc(sub.filter((pl.col('context_has_code_0')) | (pl.col('context_has_code_1')))) )
        print('context does not have ECG1', occurs_auc(sub.filter(~pl.col('context_has_code_1'))) )
        print('context has both ECG0 & ECG1', occurs_auc(sub.filter(pl.col('context_has_code_0')).filter(pl.col('context_has_code_1'))) )
        print('context has neither ECG0 & ECG1', occurs_auc(sub.filter(~pl.col('context_has_code_0')).filter(~pl.col('context_has_code_1'))) )
        sub = df.filter(pl.col('query_name')==query_name).filter(pl.col('subject_id').is_in(pt_cohort))
        print()
        print(f'{query_name} patients', occurs_auc(sub))
        print('context has ECG0', occurs_auc(sub.filter(pl.col('context_has_code_0'))) )
        print('context does not have ECG0', occurs_auc(sub.filter(~pl.col('context_has_code_0'))) )
        print('context has ECG1', occurs_auc(sub.filter(pl.col('context_has_code_1'))) )
        print('context does not have ECG1', occurs_auc(sub.filter(~pl.col('context_has_code_1'))) )
        print('\n\n')

    def plot_occurs_dist(df, path='occurs_dist'):
        data = df.select(['occurs_logits', 'occurs_target']).drop_nulls().to_pandas()
        import seaborn as sns
        from scipy.special import expit
        import matplotlib.pyplot as plt
        data['occurs_probs'] = expit(data['occurs_logits'])
        p = data['occurs_target'].mean()
        plt.figure(figsize=(8, 6))
        sns.histplot(
            data.loc[data['occurs_target'] == 1, 'occurs_probs'], 
            bins=20, stat='density', color='blue', label='target=1', alpha=0.6
        )
        sns.histplot(
            data.loc[data['occurs_target'] == 0, 'occurs_probs'], 
            bins=20, stat='density', color='orange', label='target=0', alpha=0.6
        )
        plt.title(f'Occurs prob distribution (target p={p:.2f})')
        plt.xlim(0, 1)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{path}.png')

    for query_num in [0,1]:
        not_query_num = int(not bool(query_num))
        sub = df.filter(pl.col('query_name')==f'ECG_{query_num}')
        plot_occurs_dist(sub.filter(pl.col('context_has_history')), f'q{query_num}_all_w_hx')
        plot_occurs_dist(sub.filter(~pl.col('context_has_history')), f'q{query_num}_all_no_hx')
        plot_occurs_dist(sub.filter(~pl.col('context_has_history')).filter(pl.col(f'query_has_code_{query_num}')), f'q{query_num}_all_no_hx_q_aligned')
        plot_occurs_dist(sub.filter(~pl.col('context_has_history')).filter(pl.col(f'query_has_code_{not_query_num}')), f'q{query_num}_all_no_hx_q_misaligned')
        plot_occurs_dist(sub.filter(~pl.col('context_has_history')).filter(~pl.col(f'query_has_code_{query_num}')), f'q{query_num}_all_no_hx_q_no_aligned')
        plot_occurs_dist(sub.filter(~pl.col('context_has_history')).filter(~pl.col(f'query_has_code_{not_query_num}')), f'q{query_num}_all_no_hx_q_no_misaligned')
        plot_occurs_dist(sub.filter(~pl.col('context_has_history')).filter(~pl.col(f'query_has_code_{not_query_num}')).filter(~pl.col(f'query_has_code_{query_num}')), f'q{query_num}_all_no_hx_q_none')
        plot_occurs_dist(sub.filter(pl.col('query_aligned_pt')).filter(pl.col('context_has_history')), f'q{query_num}_aligned_w_hx')
        plot_occurs_dist(sub.filter(pl.col('query_aligned_pt')).filter(~pl.col('context_has_history')), f'q{query_num}_aligned_no_hx')
        plot_occurs_dist(sub.filter(~pl.col('query_aligned_pt')).filter(pl.col('context_has_history')), f'q{query_num}_misaligned_w_hx')
        plot_occurs_dist(sub.filter(~pl.col('query_aligned_pt')).filter(~pl.col('context_has_history')), f'q{query_num}_misaligned_no_hx')

    ipdb.set_trace()

if __name__ == "__main__":
    main()