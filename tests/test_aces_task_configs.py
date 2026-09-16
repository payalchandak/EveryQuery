"""Check that the shipped ACES task configs actually load.

The configs under ``every_query/predict/external_tasks/task_configs/`` are consumed by
[ACES](https://github.com/justin13601/ACES), not by EveryQuery, so nothing in the normal pipeline
parses them and a typo would otherwise surface on a cluster, hours into an extraction run.

This runs ACES' own ``TaskExtractorConfig.load`` rather than re-implementing its rules: it resolves
the predicate DAG, builds the window tree, parses every time delta, and rejects undefined predicate
references, unknown config keys, and more than one labeled or index-timestamped window.
"""

from importlib.resources import files
from pathlib import Path

import pytest
from aces.config import TaskExtractorConfig

TASK_CONFIG_ROOT = Path(str(files("every_query") / "predict" / "external_tasks" / "task_configs"))
TASK_CONFIG_PATHS = sorted(TASK_CONFIG_ROOT.rglob("*.yaml"))


@pytest.mark.parametrize(
    "config_path",
    TASK_CONFIG_PATHS,
    ids=[str(p.relative_to(TASK_CONFIG_ROOT).with_suffix("")) for p in TASK_CONFIG_PATHS],
)
def test_parses_with_aces(config_path: Path):
    TaskExtractorConfig.load(config_path)
