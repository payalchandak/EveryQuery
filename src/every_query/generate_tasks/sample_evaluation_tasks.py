"""Hydra adapter for package-owned dense task-grid generation.

The evaluation-shaped sampling stages live in
:mod:`meds_random_task_sampler.dense_grid`.
"""

from importlib.resources import files
from pathlib import Path

import hydra
from meds_random_task_sampler import TaskGridGeneratorConfig, generate_task_grid
from meds_random_task_sampler.random_sample import _require_path_arg, read_query_codes
from omegaconf import DictConfig, ListConfig

CONFIGS = str(files("every_query") / "generate_tasks" / "configs")


@hydra.main(version_base=None, config_path=CONFIGS, config_name="sample_evaluation_tasks_config")
def main(cfg: DictConfig) -> None:
    """Generate one dense task-grid shard from an EveryQuery Hydra configuration."""
    data_dir = _require_path_arg(cfg.get("data_dir"), "data_dir")
    out_dir = _require_path_arg(cfg.get("out_dir"), "out_dir")
    query_codes = cfg.get("query_codes")
    codes = read_query_codes(list(query_codes) if isinstance(query_codes, ListConfig) else query_codes)

    durations: list[int] = []
    for i, duration in enumerate(cfg.durations):
        if isinstance(duration, bool) or not isinstance(duration, int | float):
            raise TypeError(
                f"cfg.durations[{i}] must be a number, got {type(duration).__name__}: {duration!r}"
            )
        if not float(duration).is_integer():
            raise ValueError(
                f"cfg.durations[{i}] must be a whole-day integer, got {duration!r}. "
                "Fractional horizons aren't supported by this CLI yet."
            )
        durations.append(int(duration))

    fraction = cfg.get("subject_subsample_fraction")
    if isinstance(fraction, bool) or (fraction is not None and not isinstance(fraction, int | float)):
        raise TypeError(
            "cfg.subject_subsample_fraction must be a number in (0, 1] or null, "
            f"got {type(fraction).__name__}: {fraction!r}"
        )

    package_config = TaskGridGeneratorConfig(
        prediction_times_per_subject=int(cfg.prediction_times_per_subject),
        min_context_per_subject=int(cfg.min_context_per_subject),
        query_codes=codes,
        durations=durations,
        subject_subsample_fraction=None if fraction is None else float(fraction),
        write_unique_prediction_times=bool(cfg.get("write_unique_prediction_times", True)),
        censored_rows="drop",
        seed=int(cfg.seed),
    )
    generate_task_grid(
        data_dir=data_dir,
        output_dir=Path(out_dir) / "eval",
        split=str(cfg.split),
        input_shard=str(cfg.input_shard),
        config=package_config,
        overwrite=bool(cfg.get("overwrite", False)),
    )


if __name__ == "__main__":
    main()
