"""Structural validation for the shipped ACES task configs.

The configs under ``every_query/predict/external_tasks/task_configs/`` are consumed by
[ACES](https://github.com/justin13601/ACES), which is not an EveryQuery dependency, so nothing in this
repo parses them at runtime, so a typo would otherwise only surface on a cluster hours into an
extraction run.  These tests are a schema-free stand-in: they check the structure ACES requires
(predicates / trigger / windows, exactly one labeled window, no dangling predicate or window
reference) plus the EveryQuery-side conventions the ``aces_to_eq`` pipeline assumes (a single
``index_timestamp`` on the ``input`` window, a file stem usable as a ``task_name``).

They deliberately do *not* validate ACES semantics, only that every name resolves and every
constraint is well-formed.
"""

import re
from importlib.resources import files
from pathlib import Path

import pytest
import yaml

TASK_CONFIG_ROOT = Path(str(files("every_query") / "predict" / "external_tasks" / "task_configs"))

# ACES treats ``_ANY_EVENT`` as an implicitly-defined predicate matching every event.
SPECIAL_PREDICATES = frozenset({"_ANY_EVENT"})

# Top-level keys ACES recognises.  ``predicates`` / ``trigger`` / ``windows`` are required; the
# others are optional and unused by these tasks today, but allowed so adding one isn't a test edit.
REQUIRED_TOP_LEVEL = frozenset({"predicates", "trigger", "windows"})
ALLOWED_TOP_LEVEL = REQUIRED_TOP_LEVEL | {"patient_demographics", "metadata", "description"}

# ``(min, max)`` inclusion constraints: each bound is either ``None`` or a non-negative integer.
CONSTRAINT_RE = re.compile(r"^\(\s*(None|\d+)\s*,\s*(None|\d+)\s*\)$")

# ``end <- intubation`` / ``start -> map``: the token after the arrow names a predicate.
PREDICATE_ARROW_RE = re.compile(r"(?:<-|->)\s*([A-Za-z_][A-Za-z0-9_]*)\s*$")

# ``to_discharge.end`` / ``vent_context.start``: the token before the dot names another window.
WINDOW_REF_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\.(start|end)$")

# ``or(a, b, c)`` / ``and(a, b)``: every argument names another predicate.
EXPR_RE = re.compile(r"^(?:or|and)\(\s*([A-Za-z0-9_,\s]+?)\s*\)$")

# File stems double as ACES ``task_name`` path components (see configs/aces_to_eq.yaml).
TASK_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _task_config_paths() -> list[Path]:
    return sorted(TASK_CONFIG_ROOT.rglob("*.yaml"))


def _task_id(path: Path) -> str:
    return str(path.relative_to(TASK_CONFIG_ROOT).with_suffix(""))


TASK_CONFIG_PATHS = _task_config_paths()


@pytest.fixture(params=TASK_CONFIG_PATHS, ids=[_task_id(p) for p in TASK_CONFIG_PATHS])
def task_config_path(request) -> Path:
    return request.param


@pytest.fixture
def task_config(task_config_path: Path) -> dict:
    return yaml.safe_load(task_config_path.read_text())


def _referenced_predicates(cfg: dict) -> set[str]:
    """Every predicate name the config refers to, from any of the four reference sites."""
    refs: set[str] = {cfg["trigger"]}

    for predicate in cfg["predicates"].values():
        if "expr" in predicate:
            match = EXPR_RE.match(predicate["expr"])
            assert match is not None, f"unparsable expr: {predicate['expr']!r}"
            refs.update(arg.strip() for arg in match.group(1).split(","))

    for window in cfg["windows"].values():
        refs.update(window.get("has", {}).keys())
        if "label" in window:
            refs.add(window["label"])
        for endpoint in ("start", "end"):
            value = window.get(endpoint)
            if isinstance(value, str):
                arrow = PREDICATE_ARROW_RE.search(value)
                if arrow:
                    refs.add(arrow.group(1))

    return refs


def test_task_configs_are_discovered():
    """The configs must be found through ``importlib.resources``, i.e. they ship with the wheel.

    ``[tool.setuptools.package-data] every_query = ["**/*.yaml"]`` is what makes this true; this
    test fails if that glob is ever narrowed.
    """
    assert TASK_CONFIG_PATHS, f"no task configs found under {TASK_CONFIG_ROOT}"


def test_parses_as_yaml_mapping(task_config):
    assert isinstance(task_config, dict), "task config must be a YAML mapping"


def test_stem_is_a_usable_task_name(task_config_path: Path):
    """``aces_to_eq.yaml`` interpolates ``task_name`` into shard paths, so keep stems path-safe."""
    assert TASK_NAME_RE.match(task_config_path.stem), (
        f"{task_config_path.stem!r} is not a lowercase snake_case task name"
    )


def test_documents_its_clinical_question(task_config_path: Path):
    """Each config leads with a title comment and the clinical question it operationalises."""
    lines = task_config_path.read_text().splitlines()
    assert lines[0].startswith("# ") and len(lines[0]) > 2, "first line must be a '# Title' comment"
    assert any(line.startswith("# Clinical question:") for line in lines[:10]), (
        "header must carry a '# Clinical question:' block"
    )


def test_top_level_keys(task_config):
    keys = set(task_config)
    assert keys >= REQUIRED_TOP_LEVEL, f"missing required keys: {sorted(REQUIRED_TOP_LEVEL - keys)}"
    assert keys <= ALLOWED_TOP_LEVEL, f"unrecognised top-level keys: {sorted(keys - ALLOWED_TOP_LEVEL)}"


def test_predicates_are_code_or_expr(task_config):
    """A predicate is defined either by matching codes or by an expression over other predicates."""
    for name, predicate in task_config["predicates"].items():
        has_code, has_expr = "code" in predicate, "expr" in predicate
        assert has_code != has_expr, f"predicate {name!r} must have exactly one of 'code' / 'expr'"


def test_every_referenced_predicate_is_defined(task_config):
    """No dangling predicate names: the failure mode a typo in a `has:` key or `label:` produces."""
    defined = set(task_config["predicates"]) | SPECIAL_PREDICATES
    undefined = _referenced_predicates(task_config) - defined
    assert not undefined, f"referenced but undefined predicates: {sorted(undefined)}"


def test_every_referenced_window_is_defined(task_config):
    """``start: to_discharge.end`` must name a window that actually exists."""
    windows = task_config["windows"]
    for window_name, window in windows.items():
        for endpoint in ("start", "end"):
            value = window.get(endpoint)
            if not isinstance(value, str):
                continue
            match = WINDOW_REF_RE.match(value)
            if match and match.group(1) not in windows:
                pytest.fail(f"window {window_name!r}.{endpoint} references unknown window {value!r}")


def test_windows_have_both_endpoints(task_config):
    for name, window in task_config["windows"].items():
        assert "start" in window, f"window {name!r} is missing 'start'"
        assert "end" in window, f"window {name!r} is missing 'end'"


def test_has_constraints_are_well_formed(task_config):
    for window_name, window in task_config["windows"].items():
        for predicate_name, constraint in window.get("has", {}).items():
            assert isinstance(constraint, str) and CONSTRAINT_RE.match(constraint), (
                f"{window_name!r}.has[{predicate_name!r}] = {constraint!r} is not a '(min, max)' constraint"
            )


def test_exactly_one_labeled_window(task_config):
    """ACES emits one ``boolean_value`` per task, and ``aces_to_eq`` copies exactly that column."""
    labeled = [name for name, window in task_config["windows"].items() if "label" in window]
    assert len(labeled) == 1, f"expected exactly one labeled window, found {labeled}"


def test_input_window_carries_the_only_index_timestamp(task_config):
    """The prediction time is the end of ``input``, the tuple ``aces_to_eq`` joins EQ rows on."""
    indexed = [name for name, window in task_config["windows"].items() if "index_timestamp" in window]
    assert indexed == ["input"], f"expected only 'input' to set index_timestamp, found {indexed}"
    assert task_config["windows"]["input"]["index_timestamp"] == "end"
