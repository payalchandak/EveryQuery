import json
import os

import numpy as np
import pandas as pd
from omegaconf import OmegaConf


class Experiment:
    def __init__(self, dir, restart=False):
        self.dir = dir
        self.spec = self.load_spec()
        self.runs_path = os.path.join(self.dir, "runs.csv")
        if not restart and os.path.exists(self.runs_path):
            self.load_runs()
        else:
            self.build_runs()
            self.save_runs()

    def load_spec(self):
        pth = os.path.join(self.dir, "spec.json")
        with open(pth) as file:
            spec = json.load(file)
        if "id" not in spec.keys():
            spec["id"] = np.random.randint(10000000000)
            # (todo) assert that experiment ID is not repeated
            with open(pth, "w") as file:
                file.write(json.dumps(spec, indent=4))
        return spec

    def build_runs(self):
        rows = []
        for value in self.spec["variable_param"]["values"]:
            for _ in range(self.spec["seeds_per_run"]):
                seed = np.random.randint(1000000)
                row = {
                    "experiment_id": self.spec["id"],
                    "variable_parameter": self.spec["variable_param"]["name"],
                    "value": value,
                    "seed": seed,
                    "save_dir": None,
                    "wandb_run_id": None,
                    "training_initiated": False,
                    "training_finished": False,
                    "zeroshot_finished": False,
                }
                rows.append(row)
        self.runs = pd.DataFrame(rows).reset_index().rename({"index": "run_id"}, axis=1)

    def save_runs(self):
        self.runs.to_csv(self.runs_path, index=False)

    def load_runs(self):
        self.runs = pd.read_csv(self.runs_path)

    def build_run_config(self, run_idx):
        cfg_dir = "/storage2/payal/dropbox/private/EveryQueryGPT/configs/"
        run = self.runs.loc[run_idx, :]
        cfg_name = f"{run.experiment_id}_{run.run_id}.yaml"
        cfg = OmegaConf.load(cfg_dir + "config.yaml")
        cfg = self.update_config_param(cfg, "experiment.dir", self.dir)
        cfg = self.update_config_param(cfg, "experiment.id", run.experiment_id)
        cfg = self.update_config_param(cfg, "experiment.run", run.run_id)
        cfg = self.update_config_param(cfg, "seed", run.seed)
        cfg = self.update_config_param(cfg, run.variable_parameter, run.value)
        for param, value in self.spec["defaults"].items():
            cfg = self.update_config_param(cfg, param, value)
        with open(cfg_dir + cfg_name, "w") as file:
            OmegaConf.save(cfg, file)
        return cfg_name

    def update_config_param(self, cfg, parameter, value):
        if isinstance(value, np.generic):
            value = value.item()
        current = cfg
        keys = parameter.split(".")
        for key in keys[:-1]:
            current = current.setdefault(key, {})
        current[keys[-1]] = value
        return cfg

    def pretrain_commands(self, N=1, ignore_initiated=False):
        self.load_runs()
        commands = []
        for i in range(self.runs.shape[0]):
            if self.runs.loc[i, "training_finished"]:
                continue
            if ignore_initiated and self.runs.loc[i, "training_initiated"]:
                continue
            cfg_name = self.build_run_config(i)
            cmd = f'source .env && PYTHONPATH="$EVENT_STREAM_PATH:$PYTHON PATH" spython $EVENT_STREAM_PATH/scripts/pretrain.py --config-path=$(pwd)/configs --config-name={cfg_name} "hydra.searchpath=[$EVENT_STREAM_PATH/configs]"'
            commands.append(cmd)
            if len(commands) >= N:
                break
        [print(x) for x in commands]
        return commands

    def zeroshot_commands(self, N=1):
        self.load_runs()
        commands = []
        for i in range(self.runs.shape[0]):
            if not self.runs.loc[i, "training_finished"]:
                continue
            if self.runs.loc[i, "zeroshot_finished"]:
                continue
            save_dir = self.runs.loc[i, "save_dir"]
            cfg_name = f"{self.spec['id']}_{i}.yaml"
            cmd = f'source .env && PYTHONPATH="$EVENT_STREAM_PATH:$PYTHON PATH" python $EVENT_STREAM_PATH/scripts/zero_shot_queries.py --config-path=$(pwd)/configs --config-name={cfg_name} "hydra.searchpath=[$EVENT_STREAM_PATH/configs]" ++save_dir={save_dir}'
            commands.append(cmd)
            if len(commands) >= N:
                break
        [print(x) for x in commands]
        return commands

    def collate_results(self):
        # use save_dir to collect dumped predictions for finished runs
        # collate into performance metric figure (config parameter vs auroc)
        pass


exp = Experiment(
    dir="/storage2/payal/dropbox/private/EveryQueryGPT/experiments/num_hidden_layers/",
    restart=False,
)
exp.pretrain_commands(N=2, ignore_initiated=True)
exp.zeroshot_commands(N=2)
