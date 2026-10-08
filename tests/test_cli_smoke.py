"""Smoke tests for the ``EQ_*`` console script entry points and the ``scripts/`` drivers.

Each endpoint registered under ``[project.scripts]`` in ``pyproject.toml`` is invoked via its
installed console script name (the contract introduced in PR #61) with ``--help`` and must exit 0.
A successful exit proves the ``[project.scripts]`` entry resolved, the package config directory
resolved via ``importlib.resources.files()``, and module-level imports don't blow up in a fresh
interpreter.

``test_script_imports`` covers the un-installed drivers under ``scripts/``.  CI runs ``pytest``
over the rootdir, where ``--doctest-modules`` imports them anyway, but a ``pytest tests/`` run
never reaches the directory — this keeps them covered either way.

Child-process coverage is picked up automatically via
``[tool.coverage.run] patch = ["subprocess"]`` in ``pyproject.toml`` — no
per-subprocess env wiring required.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

_VENV_BIN = str(Path(sys.executable).parent)
_REPO_ROOT = Path(__file__).parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"


def _project_scripts() -> dict[str, str]:
    """``[project.scripts]`` of ``pyproject.toml``: console-script name -> ``module:function``."""
    with (_REPO_ROOT / "pyproject.toml").open("rb") as f:
        return dict(tomllib.load(f)["project"]["scripts"])


# Every console script the package ships, read from ``pyproject.toml`` rather than listed by hand,
# so a newly registered ``EQ_*`` is smoke-tested the moment it exists and a deleted one cannot
# linger in a stale list.
_ENTRYPOINTS: list[str] = sorted(_project_scripts())

# The multitask evaluation generator was removed in #29: `EQ_generate_evaluation_query_sequences`
# is the one evaluation-grid generator and `EQ_predict_multitask` consumes its QuerySeqSchema
# output directly.  Neither the console script nor the module may come back under this name.
_DELETED_ENTRYPOINT = "EQ_generate_evaluation_multitask_sequences"
_DELETED_MODULE = "sample_evaluation_multitask_sequences"

# Third-party modules a ``scripts/`` driver needs that this project does not depend on.  Empty
# today — the report builders that needed ``reportlab`` are gone — but kept as the seam: a missing
# optional dep is not API drift, which is what this test exists to catch, so such a driver should
# skip rather than fail.
_OPTIONAL_SCRIPT_DEPS: dict[str, str] = {}


@pytest.fixture(scope="module")
def cli_env() -> dict[str, str]:
    """Subprocess env with venv ``PATH`` prepended so console scripts resolve."""
    env = os.environ.copy()
    env["PATH"] = _VENV_BIN + os.pathsep + env.get("PATH", "")
    return env


def test_every_entrypoint_is_an_eq_script():
    """The scripts table holds only ``EQ_*`` names, so the ``--help`` sweep below is the whole CLI."""
    assert _ENTRYPOINTS and all(name.startswith("EQ_") for name in _ENTRYPOINTS), _ENTRYPOINTS


def test_the_legacy_multitask_evaluation_generator_is_gone():
    """No console script, config or module remains for the deleted evaluation generator (#29)."""
    scripts = _project_scripts()
    assert _DELETED_ENTRYPOINT not in scripts
    assert not any(_DELETED_MODULE in target for target in scripts.values())
    pkg = _REPO_ROOT / "src" / "every_query" / "generate_tasks"
    assert not (pkg / f"{_DELETED_MODULE}.py").exists()
    assert not (pkg / "configs" / f"{_DELETED_MODULE}_config.yaml").exists()
    assert not (_REPO_ROOT / "tests" / "multitask" / "test_eval_multitask_grid.py").exists()


def test_no_live_reference_to_the_deleted_generator_module():
    """No source, script or test imports or names the deleted module.

    A plain text scan (not an import check) so a stale mention in a config comment, a docstring or a
    ``scripts/`` driver is caught too; the only allowed occurrences are this test's own constants.
    """
    # No trailing ``\b``: ``_`` is a word character, so it would let the deleted config stem
    # ``sample_evaluation_multitask_sequences_config`` (a ``--config-name`` / Hydra ``defaults``
    # entry) slip through.
    pattern = re.compile(rf"\b{_DELETED_MODULE}|\b{_DELETED_ENTRYPOINT}")
    offenders: list[str] = []
    for root in ("src", "tests", "scripts", "docs", ".github"):
        for fp in sorted((_REPO_ROOT / root).rglob("*")):
            if fp.suffix not in {".py", ".yaml", ".yml", ".md", ".toml", ".sh", ".txt", ".json"}:
                continue
            if fp.resolve() == Path(__file__).resolve() or "history" in fp.parts:
                continue
            if pattern.search(fp.read_text(errors="ignore")):
                offenders.append(str(fp.relative_to(_REPO_ROOT)))
    for name in ("pyproject.toml", "README.md", "conftest.py", "env.example.sh"):
        fp = _REPO_ROOT / name
        if fp.is_file() and pattern.search(fp.read_text()):
            offenders.append(fp.name)
    assert not offenders, f"stale references to the deleted evaluation generator: {offenders}"


@pytest.mark.parametrize("script", _ENTRYPOINTS)
def test_entrypoint_help(script, cli_env):
    """``<script> --help`` exits 0."""
    cmd = [script, "--help"]
    result = subprocess.run(cmd, capture_output=True, text=True, env=cli_env, timeout=60)
    assert result.returncode == 0, (
        f"{script} --help failed (rc={result.returncode})\n"
        f"cmd: {cmd}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


@pytest.mark.parametrize("script_path", sorted(_SCRIPTS_DIR.glob("*.py")), ids=lambda p: p.name)
def test_script_imports(script_path: Path):
    """Every ``scripts/*.py`` driver still imports against the current library API.

    ``pyproject.toml`` used to set ``--ignore=scripts``, which is what let several of these sit
    broken behind stale ``every_query`` imports across a whole sampler refactor, invisible to CI.
    The exclusion is gone, so a rootdir run now imports them under ``--doctest-modules`` too — but
    a ``pytest tests/`` run does not reach ``scripts/``, and this is what covers that case.
    Import only: no cohort, no checkpoint, no ``main()``.
    """
    optional = _OPTIONAL_SCRIPT_DEPS.get(script_path.name)
    if optional is not None:
        pytest.importorskip(optional, reason=f"{script_path.name} needs the optional {optional!r} dep")

    # A driver may import a sibling, which resolves only with scripts/ on the path — exactly how
    # they run (``python scripts/bench_multitask_dataset.py``).
    sys.path.insert(0, str(_SCRIPTS_DIR))
    try:
        spec = importlib.util.spec_from_file_location(f"_scripts_smoke_{script_path.stem}", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(_SCRIPTS_DIR))
