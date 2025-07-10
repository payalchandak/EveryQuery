import os
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import hydra
from hydra import compose, initialize
from omegaconf import DictConfig
import numpy as np
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from sklearn.metrics import roc_auc_score
from meds_torch.utils.resolvers import setup_resolvers
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

@dataclass(frozen=True)
class Query: 
    code: str
    duration: int 
    offset: int 
    range: tuple[float, float] | None = None

class Run: 
    def __init__(self, dir):
        assert isinstance(dir, str)
        self.dir = dir 
        self._cfg = None  # Lazy loading
        self._model = None  # Lazy loading
        self._trainer = None  # Lazy loading

    @property
    def relative_dir(self): 
        script_dir = Path(__file__).resolve().parent
        return os.path.relpath(self.dir, start=script_dir)

    @property
    def best_model_ckpt(self): 
        return f"{self.dir}/checkpoints/best_model.ckpt"
    

    @property
    def training_queries(self) -> set[Query]: 
        cfg = self.cfg.data.train
        codes = cfg.codes

        match cfg.duration_sampling_strategy:
            case 'fixed': durations = [cfg.fixed_duration]
            case 'categorical': durations = cfg.categorical_duration
            case 'within_record': raise NotImplementedError
            case 'random': raise NotImplementedError
            case _: raise ValueError(f"Unknown duration sampling strategy: {cfg.duration_sampling_strategy}")

        match cfg.offset_sampling_strategy:
            case 'fixed': offsets = [cfg.fixed_offset]
            case 'categorical': offsets = cfg.categorical_offset
            case 'within_record': raise NotImplementedError
            case 'random': raise NotImplementedError
            case _: raise ValueError(f"Unknown offset sampling strategy: {cfg.offset_sampling_strategy}")

        return {Query(code, duration, offset) for code in codes for duration in durations for offset in offsets}
    
    @property
    def cfg(self):
        if self._cfg is None:
            setup_resolvers()
            with initialize(version_base='1.3', config_path=self.relative_dir):
                self._cfg = compose(config_name='hydra_config.yaml')
        return self._cfg
    
    @property
    def model(self):
        if self._model is None:
            self._model: LightningModule = hydra.utils.instantiate(self.cfg.model)
            self._model.load_state_dict(torch.load(self.best_model_ckpt)["state_dict"])
        return self._model

    @property
    def trainer(self):
        if self._trainer is None:
            self._trainer: Trainer = hydra.utils.instantiate(self.cfg.trainer)
        return self._trainer

class ExperimentRegistry:
    def __init__(self, registry_path='experiment_registry.pkl'):
        self.runs_by_dir = {}  # Only names and paths, instantiate on demand
        self.runs_by_name = defaultdict(set)
        self.metrics = defaultdict(dict)  # {(dir, query) -> metrics dictionary}
        self.registry_path = Path(registry_path)
        self.load()

    def add_run(self, name, dir):
        if dir in self.runs_by_dir:
            existing_name, _ = self.runs_by_dir[dir]
            if existing_name == name:
                return  # Run is already registered with the same name, do nothing
            raise ValueError(f"Cannot register '{name}'. Run at {dir} is already registered under the name '{existing_name}'. Remove it first before re-adding with a new name.")
        self.runs_by_dir[dir] = (name, dir)
        self.runs_by_name[name].add(dir)
        self.save()
        
    def remove_run(self, dir):
        if dir in self.runs_by_dir:
            name, _ = self.runs_by_dir.pop(dir)
            self.runs_by_name[name].remove(dir)
            if not self.runs_by_name[name]:
                del self.runs_by_name[name]
            self.save()
        else:
            raise ValueError(f"Run '{dir}' not found in the registry; cannot be removed.")

    def get_run(self, dir):
        return Run(dir)
    
    def get_one_run(self, name): 
        dirs = self.get_run_dirs(name)
        return self.get_run(next(iter(dirs)))

    def get_run_dirs(self, name):
        return self.runs_by_name.get(name, set())

    def store_metrics(self, dir, query, metrics):
        self.metrics[(dir, query)] = metrics
        self.save()

    def get_metrics(self, dir, query):
        return self.metrics.get((dir, query))

    def save(self):
        with open(self.registry_path, 'wb') as f:
            pickle.dump({'runs_by_dir': self.runs_by_dir, 'runs_by_name': self.runs_by_name, 'metrics': self.metrics}, f)

    def load(self):
        if self.registry_path.exists():
            with open(self.registry_path, 'rb') as f:
                data = pickle.load(f)
                self.runs_by_dir = data.get('runs_by_dir', {})
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
            pred[k.replace('logits', 'score')] = torch.sigmoid(data).tolist()
        return pred

    def _evaluate(self, dir, query):
        stored_metrics = self.get_metrics(dir, query)
        if stored_metrics:
            return stored_metrics
        run = self.get_run(dir)
        pred = self.predict(run, query)
        metrics = {
            'censor_auc': roc_auc_score(pred['censor_target'], pred['censor_score']),
            'occurs_auc': roc_auc_score(pred['occurs_target'], pred['occurs_score'])
        }
        self.store_metrics(dir, query, metrics)
        return metrics

    def evaluate(self, ids, queries):
        if isinstance(ids, str): ids = [ids]
        if isinstance(queries, Query): queries = [queries]
        assert all(isinstance(query, Query) for query in queries)
        dirs = [] # all ids are expected to be seed variations of the same model
        for id in ids:
            if id in self.runs_by_dir.keys():
                dirs.append(id)
            elif id in self.runs_by_name.keys():
                dirs.extend(self.get_run_dirs(id))
            else: 
                raise ValueError(f"Run {id} not found in names or dirs")
        metrics = [self._evaluate(dir, query) for dir in dirs for query in queries]
        censor_auc = [m['censor_auc'] for m in metrics]
        occurs_auc = [m['occurs_auc'] for m in metrics]
        return {
            'censor_auc': (round(float(np.mean(censor_auc)),3), round(float(np.std(censor_auc)),3), len(censor_auc)),
            'occurs_auc': (round(float(np.mean(occurs_auc)),3), round(float(np.std(occurs_auc)),3), len(occurs_auc)),
        }
    
    def compare(self, id1, id2, queries): 
        # (TODO) overload function to handle names
        '''
        average rank of Run
        percent of time Run is first 
        '''
        assert id1 in self.runs_by_dir.keys()
        assert id2 in self.runs_by_dir.keys()
        if isinstance(queries, Query): queries = [queries]
        assert all(isinstance(query, Query) for query in queries)

        comparisons = {'censor_auc': None, 'occurs_auc': None}
        for metric in comparisons.keys():
            margins = {id1: [], id2: []}
            counts = {id1: 0, id2: 0}
            for query in queries: 
                m1 = self._evaluate(id1, query).get(metric)
                m2 = self._evaluate(id2, query).get(metric)
                if m1 is None or m2 is None: continue
                if m1 > m2: # model 1 wins
                    counts[id1] += 1
                    margins[id1].append(m1 - m2)
                elif m2 > m1: # model 2 wins
                    counts[id2] += 1
                    margins[id2].append(m2 - m1)
                else: # tied 
                    counts[id1] += 1
                    counts[id2] += 1
                    margins[id1].append(0)
                    margins[id2].append(0)

            def summary(margins, count):
                return (round(count / len(queries), 3), len(margins), 
                        round(float(np.mean(margins)), 3) if margins else None, 
                        round(float(np.std(margins)), 3) if margins else None)

            comparisons[metric] = (summary(margins[id1], counts[id1]), summary(margins[id2], counts[id2]))

        return comparisons

    def compare_printer(self, name1, name2, queries): 
        id1 = self.get_one_run(name1).dir
        id2 = self.get_one_run(name2).dir

        comparisons = self.compare(id1, id2, queries)
        
        print()
        for metric in comparisons.keys(): 
            print(f"For metric {metric},")
            summary = comparisons[metric]
            s1, s2 = summary[0], summary[1]
            print(f"\t{name1} wins {s1[0]*100:.1f}% of the time with a margin of {s1[2]*100:.1f} ± {s1[3]*100:.1f}%")
            print(f"\t{name2} wins {s2[0]*100:.1f}% of the time with a margin of {s2[2]*100:.1f} ± {s2[3]*100:.1f}%")
            print()
        print()

    # ────────────────────────────────────────────────────────────────
    # Visualization helpers
    # ────────────────────────────────────────────────────────────────
    def plot_auroc_comparison(
        self,
        run_a: str,
        run_b: str,
        queries: list[Query] | set[Query],
        *,
        marker_size: int = 100,
        xlim: tuple[float, float] = (0, 1.0),
        ylim: tuple[float, float] = (0, 1.0),
        figsize: tuple[int, int] = (8, 8),
        show: bool = True,
        ax=None,
    ):
        """
        Scatter‑plot occurs‑AUC for two runs across a set of queries.

        Parameters
        ----------
        run_a, run_b : str
            Run *names* or *dirs* accepted by `self.evaluate`.
            `run_a` values appear on the x‑axis; `run_b` on the y‑axis.
        queries : list[Query] | set[Query]
            Queries to evaluate.
        marker_size : int, default 100
            Size of each scatter marker.
        xlim, ylim : tuple[float, float]
            Axis limits.
        figsize : tuple[int, int]
            Figure size if ``ax`` is not supplied.
        show : bool, default True
            Call ``plt.show()`` automatically.
        ax : matplotlib.axes.Axes | None
            Plot into an existing axes object if supplied.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes containing the scatter plot.
        """
        # Lazy import so non‑plotting workflows avoid the dependency
        import matplotlib.pyplot as plt

        if isinstance(queries, Query):
            queries = [queries]

        # Collect AUROC values
        aurocs_a, aurocs_b = [], []
        for q in queries:
            try:
                aurocs_a.append(self.evaluate(run_a, q)["occurs_auc"][0])
                aurocs_b.append(self.evaluate(run_b, q)["occurs_auc"][0])
            except Exception as exc:
                print(f"[plot_auroc_comparison] Skipping {q} – {exc}")

        # Create figure/axes if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        ax.scatter(aurocs_a, aurocs_b, s=marker_size)
        # Reference parity line
        ax.plot(xlim, ylim, "r--", linewidth=1)

        ax.set_xlabel(f"{run_a}")
        ax.set_ylabel(f"{run_b}")
        ax.set_title(f"AUROC Comparison")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(True)

        if show:
            plt.tight_layout()
            plt.show()

        return ax

    def plot_auroc_heatmap(
        self,
        runs: list[str],
        queries=None,
        *,
        stage_to_y: dict[str, int] | None = None,
        cmap: str = "Greys",
        vmin: float = 0.0,
        vmax: float = 1.0,
        highlight_thresh: float = 0.6,
        figsize: tuple[int, int] = (10, 1.2),
        show: bool = False,
        ax=None,
    ):
        """
        Heatmap of occurs-AUC for multiple runs × queries.

        Parameters
        ----------
        exp : ExperimentRegistry
            Registry with runs added.
        runs : list[str]
            List of run names or directories (each becomes a row).
        queries : list[Query] | None
            If None, use training queries from the first run.
        stage_to_y : dict[str, int] | None
            Optional mapping of run -> numeric label (e.g., {'stage_0': 10}).
            If not provided, defaults to number of training queries.
        cmap : str
            Matplotlib colormap name (default: "Greys").
        vmin, vmax : float
            Fixed color scale limits.
        highlight_thresh : float
            AUROC values above this get white text annotation.
        figsize : tuple
            Figure size.
        show : bool
            Whether to call plt.show().
        ax : matplotlib.axes.Axes | None
            If provided, plot into this axes.
        """
        if not runs:
            raise ValueError("At least one run must be provided.")

        # Default query list from first run if not specified
        if queries is None:
            queries = self.get_one_run(runs[0]).training_queries
        queries = list(queries)

        # Set default y-tick labels to number of training queries in each run
        if stage_to_y is None:
            stage_to_y = {
                run: len(self.get_one_run(run).training_queries) for run in runs
            }

        # Build AUROC matrix (rows = runs, cols = queries)
        data = np.zeros((len(runs), len(queries)))
        for i, run in enumerate(runs):
            for j, q in enumerate(queries):
                data[i, j] = self.evaluate(run, q)["occurs_auc"][0]

        # Create figure and axes
        if ax is None:
            fig, ax = plt.subplots(figsize=(figsize[0], figsize[1] * len(runs)))

        # Show the heatmap
        im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

        # Tick labels
        ax.set_xticks(np.arange(len(queries)))
        ax.set_xticklabels([q.code for q in queries], rotation=90, ha="right", fontsize=6)

        ax.set_yticks(np.arange(len(runs)))
        ax.set_yticklabels([stage_to_y[run] for run in runs])

        # Remove axis labels and grid
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(False)

        # Colorbar without label
        cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label("")

        # Annotate AUROC values in each cell
        for i in range(len(runs)):
            for j in range(len(queries)):
                val = data[i, j]
                color = "white" if val > highlight_thresh else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=color)

        if show:
            plt.tight_layout()
            plt.show()

        return ax

    def plot_auroc_clustermap(
        self,
        runs: list[str],
        queries=None,
        *,
        stage_to_y: dict[str, int] | None = None,
        cmap: str = "Greys",
        vmin: float = 0.5,
        vmax: float = 1.0,
        highlight_thresh: float = 0.8,
        figsize: tuple[int, int] = (10, 4),
        row_cluster: bool = False,
        col_cluster: bool = True,
        annot: bool = True,
    ):
        """
        Clustered heatmap of occurs-AUC for multiple runs × queries using seaborn.clustermap.

        Parameters
        ----------
        exp : ExperimentRegistry
            Experiment registry with runs added.
        runs : list[str]
            List of run names or dirs (each becomes a row).
        queries : list[Query] | None
            If None, use training queries from the first run.
        stage_to_y : dict[str, int] | None
            Maps run name to numeric y-tick label. If None, uses len(training_queries).
        cmap : str
            Colormap name (e.g., "Greys").
        vmin, vmax : float
            Fixed color limits.
        highlight_thresh : float
            AUROC values > threshold are annotated in white if `annot=True`.
        figsize : tuple[int, int]
            Size of the figure.
        row_cluster : bool
            Whether to cluster runs.
        col_cluster : bool
            Whether to cluster queries.
        annot : bool
            Whether to annotate values.
        """
        if not runs:
            raise ValueError("At least one run must be provided.")

        if queries is None:
            queries = self.get_one_run(runs[0]).training_queries
        queries = list(queries)

        if stage_to_y is None:
            stage_to_y = {
                run: len(self.get_one_run(run).training_queries) for run in runs
            }

        # Construct AUROC matrix
        data = np.zeros((len(runs), len(queries)))
        for i, run in enumerate(runs):
            for j, q in enumerate(queries):
                data[i, j] = self.evaluate(run, q)["occurs_auc"][0]

        row_labels = [stage_to_y[run] for run in runs]
        col_labels = [q.code for q in queries]

        df = pd.DataFrame(data, index=row_labels, columns=col_labels)

        # Create the clustermap
        cluster = sns.clustermap(
            df,
            row_cluster=row_cluster,
            col_cluster=col_cluster,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            figsize=figsize,
            linewidths=0.2,
            cbar_kws={"label": ""},
            annot=df.round(2) if annot else None,
            fmt=".2f",
            annot_kws={
                "size": 6,
                "color": "black"
            }
        )

        # Adjust annotation color based on threshold
        if annot and highlight_thresh is not None:
            for (i, j), val in np.ndenumerate(data):
                text = cluster.ax_heatmap.texts[i * len(queries) + j]
                if val > highlight_thresh:
                    text.set_color("white")

        cluster.ax_heatmap.set_xlabel("")
        cluster.ax_heatmap.set_ylabel("")

        return cluster
