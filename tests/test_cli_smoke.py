"""Subprocess smoke tests for the ``EQ_*`` console script entry points.

Each endpoint is invoked via its installed console script name (the contract
introduced in PR #61) with ``--help`` and must exit 0.  A successful exit
proves the ``[project.scripts]`` entry resolved, the package config
directory resolved via ``importlib.resources.files()``, and module-level
imports don't blow up in a fresh interpreter.

Three endpoints (``EQ_train``, ``EQ_evaluate``, ``EQ_gen_eval_tasks``) compose a
``{train,eval}_codes`` default group whose canonical file is generated out-of-band
(see ``src/every_query/sample_codes/``) and is not checked in.  For those we point
Hydra at a throwaway ``--config-dir`` that supplies an empty-codes smoke variant,
which preserves the rest of the compose path.

Child-process coverage is picked up automatically via
``[tool.coverage.run] patch = ["subprocess"]`` in ``pyproject.toml`` — no
per-subprocess env wiring required.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_VENV_BIN = str(Path(sys.executable).parent)

# (console_script, extra_args) — extras inject a smoke code-group for configs
# whose defaults pull an out-of-tree YAML.
_ENTRYPOINTS: list[tuple[str, list[str]]] = [
    ("EQ_process_data", []),
    ("EQ_train", ["train_codes=smoke"]),
    ("EQ_evaluate", ["eval_codes=smoke"]),
    ("EQ_generate_tasks", []),
    ("EQ_gen_eval_index", []),
    ("EQ_gen_eval_tasks", ["eval_codes=smoke"]),
    ("EQ_select_model", []),
]


@pytest.fixture(scope="module")
def smoke_config_dir(tmp_path_factory) -> Path:
    """Temp Hydra search dir supplying empty ``train_codes`` / ``eval_codes``."""
    d = tmp_path_factory.mktemp("eq_smoke_cfg")
    (d / "train_codes").mkdir()
    (d / "train_codes" / "smoke.yaml").write_text("codes: []\n")
    (d / "eval_codes").mkdir()
    (d / "eval_codes" / "smoke.yaml").write_text("id: []\nood: []\n")
    return d


@pytest.fixture(scope="module")
def cli_env() -> dict[str, str]:
    """Subprocess env with venv ``PATH`` prepended so console scripts resolve."""
    env = os.environ.copy()
    env["PATH"] = _VENV_BIN + os.pathsep + env.get("PATH", "")
    return env


@pytest.mark.parametrize(("script", "extra_args"), _ENTRYPOINTS, ids=[e[0] for e in _ENTRYPOINTS])
def test_entrypoint_help(script, extra_args, cli_env, smoke_config_dir):
    """``<script> --help`` exits 0."""
    cmd = [script, f"--config-dir={smoke_config_dir}", *extra_args, "--help"]
    result = subprocess.run(cmd, capture_output=True, text=True, env=cli_env, timeout=60)
    assert result.returncode == 0, (
        f"{script} --help failed (rc={result.returncode})\n"
        f"cmd: {cmd}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
