"""Hydra adapter for package-owned random task sampling.

The sampling stages live in :mod:`meds_random_task_sampler.random_sample`.
"""

from importlib.resources import files

import hydra
from meds_random_task_sampler import RandomTaskSamplerConfig, sample_random_tasks
from meds_random_task_sampler.random_sample import (
    _require_path_arg,
)
from omegaconf import DictConfig, ListConfig

CONFIGS = str(files("every_query") / "generate_tasks" / "configs")


def run(cfg: DictConfig) -> None:
    """Translate the EveryQuery Hydra config and invoke the package sampler."""
    query_codes = list(cfg.query_codes) if isinstance(cfg.query_codes, ListConfig) else cfg.query_codes
    package_config = RandomTaskSamplerConfig(
        num_queries=int(cfg.num_queries),
        num_contexts_per_query=int(cfg.num_contexts_per_query),
        min_prediction_times_per_subject=int(cfg.min_prediction_times_per_subject),
        query_codes=query_codes,
        min_duration=float(cfg.min_duration),
        max_duration=float(cfg.max_duration),
        duration_distribution=str(cfg.duration_distribution),
        seed=int(cfg.seed),
        max_workers=None if cfg.max_workers is None else int(cfg.max_workers),
    )
    sample_random_tasks(
        data_dir=_require_path_arg(cfg.get("data_dir"), "data_dir"),
        output_dir=_require_path_arg(cfg.get("out_dir"), "out_dir"),
        split=str(cfg.split),
        config=package_config,
        overwrite=bool(cfg.overwrite),
    )


@hydra.main(version_base=None, config_path=CONFIGS, config_name="sample_training_tasks_config")
def main(cfg: DictConfig) -> None:
    """Generate random tasks from an EveryQuery Hydra configuration."""
    run(cfg)


if __name__ == "__main__":
    main()
