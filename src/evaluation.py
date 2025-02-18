from dataclasses import dataclass
import hydra
from omegaconf import DictConfig
import ipdb
from meds_torch.utils.resolvers import setup_resolvers
from lightning import LightningDataModule, LightningModule, Trainer
from hydra import compose, initialize
import os 
import torch 
from sklearn.metrics import roc_auc_score
import pickle
from collections import defaultdict
from pathlib import Path

def minutes(x):
    if isinstance(x, int): return x
    assert isinstance(x, str)
    if x.isdigit() or x.endswith("min"):
        return int(x.replace("min", ""))
    units = {'y': 525600, 'm': 43800, 'w': 10080, 'd': 1440, 'h': 60,
             'Y': 525600, 'M': 43800, 'W': 10080, 'D': 1440, 'H': 60}
    if x == "0": return 0
    num, unit = int(x[:-1]), x[-1]
    assert unit in units.keys()
    return num * units.get(unit) 

@dataclass(frozen=True)
class Query: 
    code: str
    duration: int 
    offset: int 
    range: tuple[float, float] | None = None

class Run: 
    def __init__(self, name, dir): 
        self.dir = dir 
        self.name = name
        self.cfg = self.load_cfg()
        self.model = self.load_model()
        self.trainer = self.load_trainer()

    @property
    def relative_dir(self): 
        script_dir = os.path.dirname(os.path.abspath(__file__)) 
        return os.path.relpath(self.dir, start=script_dir)

    @property
    def best_model_ckpt(self): 
        return f"{self.dir}/checkpoints/best_model.ckpt"
    
    @property
    def training_queries(self) -> set[Query]: 
        cfg = self.cfg.data.train
        codes = cfg.codes
        match cfg.duration_sampling_strategy: 
            case 'fixed': 
                durations = [cfg.fixed_duration]
            case 'categorical': 
                durations = cfg.categorical_duration
            # within_record, random
        match cfg.offset_sampling_strategy: 
            case 'fixed': 
                offsets = [cfg.fixed_offset]
            case 'categorical': 
                offsets = cfg.categorical_offset
            # within_record, random
        queries = set()
        for code in codes: 
            for duration in durations: 
                for offset in offsets: 
                    q = Query(code=code, duration=duration, offset=offset)
                    queries.add(q)
        return queries 
    
    def load_cfg(self): 
        setup_resolvers()
        with initialize(version_base='1.3', config_path=self.relative_dir):
            cfg = compose(config_name='hydra_config.yaml')
        return cfg 
    
    def load_model(self): 
        model: LightningModule = hydra.utils.instantiate(self.cfg.model)
        model.load_state_dict(torch.load(self.best_model_ckpt)["state_dict"])
        return model

    def load_trainer(self): 
        trainer: Trainer = hydra.utils.instantiate(self.cfg.trainer)
        return trainer

class ExperimentRegistry:
    def __init__(self, registry_path='experiment_registry.pkl'):
        self.runs = {}
        self.runs_by_name = defaultdict(list)  
        self.metrics = defaultdict(dict)  # {(run.dir, query) -> metrics dictionary}
        self.registry_path = Path(registry_path)
        self.load()

    def add_run(self, run: Run):
        self.runs[run.dir] = run
        self.runs_by_name[run.name].append(run)

    def get_run(self, dir: str):
        self.runs.get(dir, None)

    def get_runs(self, name: str): 
        return self.runs_by_name.get(name, [])
    
    def store_metrics(self, run: Run, query: Query, metrics: dict):
        self.metrics[(run.dir, query)] = metrics
        self.save()

    def get_metrics(self, run: Run, query: Query):
        return self.metrics.get((run.dir, query))
    
    def save(self):
        with open(self.registry_path, 'wb') as f:
            pickle.dump({'runs': self.runs, 'runs_by_name': self.runs_by_name, 'metrics': self.metrics}, f)
    
    def load(self):
        if self.registry_path.exists():
            with open(self.registry_path, 'rb') as f:
                data = pickle.load(f)
                self.runs = data.get('runs', {})
                self.runs_by_name = data.get('runs_by_name', defaultdict(list))
                self.metrics = data.get('metrics', defaultdict(dict))
    
    def predict(self, run, query):
        cfg = run.cfg
        cfg.data.test.codes = [query.code]
        cfg.data.test.duration_sampling_strategy = 'fixed'
        cfg.data.test.fixed_duration = query.duration
        cfg.data.test.offset_sampling_strategy = 'fixed'
        cfg.data.test.fixed_offset = query.offset
        if query.range is None:
            cfg.data.test.default_value_sampling_strategy = 'ignore'
        datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
        pred = run.trainer.predict(model=run.model, dataloaders=datamodule.test_dataloader())
        assert pred, f"Prediction output is empty for run {run} and query {query}"
        pred = {k: [x[k] for x in pred] for k in pred[0].keys()}
        for k in ['censor_target', 'occurs_target']:
            pred[k] = torch.cat(pred[k]).reshape(-1).long().tolist()
        for k in ['censor_logits', 'occurs_logits']:
            data = torch.cat(pred[k]).reshape(-1)
            pred[k] = data.tolist()
            pred[k.replace('logits','score')] = torch.sigmoid(data).tolist()
        return pred
    
    def evaluate(self, run, query):
        stored_metrics = self.get_metrics(run, query)
        if stored_metrics: 
            return stored_metrics
        pred = self.predict(run, query)
        metrics = {
            'censor_auc': roc_auc_score(pred['censor_target'], pred['censor_score']),
            'occurs_auc': roc_auc_score(pred['occurs_target'], pred['occurs_score'])
        }
        self.store_metrics(run, query, metrics)
        return metrics

exp = ExperimentRegistry()

obj = Run('test', '/storage2/payal/EveryQuery/results/2025-02-11_20-24-06_414169')
exp.add_run(obj)

query = Query(code='ECG', duration=minutes('5y'), offset=0)
metrics = exp.evaluate(obj, query)
print(metrics)

metrics = exp.evaluate(obj, query)
print(metrics)

ipdb.set_trace()
