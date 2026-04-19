"""Root conftest — session-scoped fixture DAG for the EveryQuery test suite.

In-process fixtures (objects live in the test interpreter)::

    demo_model ──────────► demo_lightning_module

    simple_static_MEDS ──► tensorized_cohort_dir
                                    │
                                    ▼
                               demo_dataset
                                    │
                                    ▼
                               sample_batch

CLI-chain fixtures (run real ``EQ_*`` console scripts via subprocess; each returns the
output directory so downstream tests / fixtures can chain off it)::

    simple_static_MEDS ──► eq_preprocessed_dataset
                                    │
                                    ├────► eq_sampled_tasks_dir
                                    │             │
                                    │             ▼
                                    └────► eq_trained_model_dir

All fixtures in this file are ``scope="session"`` — each runs at most once per pytest
invocation.
"""

import os
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import polars as pl
import pytest
import torch
from meds import train_split, tuning_split
from meds_torchdata.config import MEDSTorchDataConfig
from transformers import ModernBertConfig

from every_query.data.dataset import EveryQueryBatch, EveryQueryPytorchDataset
from every_query.model import EveryQueryModel
from every_query.model.lightning_module import EveryQueryLightningModule

# ── Constants derived from the canonical simple-static MEDS sample data ─

_TRAIN_SUBJECTS = [239684, 1195293, 68729, 814703]
_TUNING_SUBJECTS = [754281]

_PRED_TIMES: dict[int, datetime] = {
    239684: datetime(2010, 5, 11, 18, 0, tzinfo=UTC),
    1195293: datetime(2010, 6, 20, 20, 30, tzinfo=UTC),
    68729: datetime(2010, 5, 26, 3, 0, tzinfo=UTC),
    814703: datetime(2010, 2, 5, 6, 0, tzinfo=UTC),
    754281: datetime(2010, 1, 3, 7, 0, tzinfo=UTC),
}

_QUERY_CODES = ["HR", "TEMP"]

DEMO_CONFIG_OVERRIDES: dict[str, int] = {
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "intermediate_size": 128,
    "max_position_embeddings": 128,
    "vocab_size": 100,
    "pad_token_id": 0,
    "cls_token_id": 1,
    "bos_token_id": 1,
    "sep_token_id": 2,
    "eos_token_id": 2,
}


# ── Model fixtures ──────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def demo_model_config() -> ModernBertConfig:
    """Tiny ModernBERT config suitable for CPU-only testing."""
    return ModernBertConfig(**DEMO_CONFIG_OVERRIDES)


@pytest.fixture(scope="session")
def demo_model() -> EveryQueryModel:
    """Seeded, tiny EveryQueryModel for CPU-only testing."""
    torch.manual_seed(0)
    model = EveryQueryModel(
        config_overrides=DEMO_CONFIG_OVERRIDES,
        do_demo=True,
        precision="32-true",
    )
    model.eval()
    return model


@pytest.fixture(scope="session")
def demo_lightning_module(demo_model: EveryQueryModel) -> EveryQueryLightningModule:
    """Lightning wrapper around *demo_model* with an AdamW optimiser factory."""
    optimizer_factory = partial(torch.optim.AdamW, lr=1e-4)
    return EveryQueryLightningModule(model=demo_model, optimizer=optimizer_factory)


# ── Data fixtures ───────────────────────────────────────────────────────
# simple_static_MEDS  — provided by meds_testing_helpers (auto-registered pytest11 plugin).

_VENV_BIN = str(Path(sys.executable).parent)


@pytest.fixture(scope="session")
def tensorized_MEDS_dataset(simple_static_MEDS: Path) -> Path:
    """Override upstream fixture so ``MTD_preprocess`` (and its children) resolve from the active venv."""
    env = os.environ.copy()
    env["PATH"] = _VENV_BIN + os.pathsep + env.get("PATH", "")

    with tempfile.TemporaryDirectory() as cohort_dir:
        cohort_dir = Path(cohort_dir)
        cmd = f"MTD_preprocess MEDS_dataset_dir={simple_static_MEDS!s} output_dir={cohort_dir!s}"
        out = subprocess.run(cmd, shell=True, check=False, capture_output=True, env=env)
        assert out.returncode == 0, (
            f"MTD_preprocess failed (rc={out.returncode}).\n"
            f"stdout: {out.stdout.decode()}\nstderr: {out.stderr.decode()}"
        )
        yield cohort_dir


@pytest.fixture(scope="session")
def tensorized_cohort_dir(tensorized_MEDS_dataset: Path) -> Path:
    """Alias for the session-scoped tensorized MEDS directory."""
    return tensorized_MEDS_dataset


# ── CLI-chain fixtures ──────────────────────────────────────────────────
# These run the real ``EQ_*`` console scripts via subprocess and return the output
# directory.  Use these in integration tests that want to exercise the actual CLI
# surface; use the in-process ``demo_dataset`` / ``sample_batch`` fixtures above for
# narrower unit tests that just need a model forward pass.


def run_and_check(cmd: list[str], *, env: dict[str, str] | None = None, timeout: float = 180.0):
    """Run ``cmd`` in a subprocess with the venv bin on ``PATH``; raise on nonzero exit.

    The exception message includes captured stdout / stderr so a failing CI log points at the
    concrete error rather than just a return-code.  ``env`` merges into (rather than replaces)
    the parent env so PATH / PYTHONPATH etc. survive.
    """
    full_env = os.environ.copy()
    full_env["PATH"] = _VENV_BIN + os.pathsep + full_env.get("PATH", "")
    if env:
        full_env.update(env)

    result = subprocess.run(cmd, capture_output=True, text=True, env=full_env, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed (rc={result.returncode}):\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  stdout:\n{result.stdout}\n"
            f"  stderr:\n{result.stderr}"
        )
    return result


# Placeholder values for the ``ensure_env()`` gate in ``utils/_env.py``.  The demo Hydra configs
# do not interpolate any of these env vars, so the actual values are inert — they exist purely
# to let the gate pass on a clean CI machine with no ``.env`` file.
ENSURE_ENV_PLACEHOLDERS: dict[str, str] = {
    "PROJECT_DIR": "/tmp",
    "OUTPUT_DIR": "/tmp",
    "TASK_DIR": "/tmp",
    "FINAL_DATA_DIR": "/tmp",
    "WANDB_ENTITY": "test",
}


@pytest.fixture(scope="session")
def eq_preprocessed_dataset(simple_static_MEDS: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Runs ``EQ_process_data do_demo=True`` against ``simple_static_MEDS``.

    Returns the tensorized-cohort output dir (``final/``), ready to feed into
    ``EQ_generate_tasks`` and ``EQ_train`` as ``codes_dir`` and ``tensorized_cohort_dir``
    respectively.

    Sibling ``intermediate/`` dir is also produced (needed as ``data_dir`` for
    ``EQ_generate_tasks``) and stored alongside ``final/`` — callers can reach it via
    ``eq_preprocessed_dataset.parent / "intermediate"``.
    """
    root = tmp_path_factory.mktemp("eq_preprocessed")
    intermediate = root / "intermediate"
    final = root / "final"
    run_and_check(
        [
            "EQ_process_data",
            f"input_dir={simple_static_MEDS!s}",
            f"intermediate_dir={intermediate!s}",
            f"output_dir={final!s}",
            "do_demo=True",
        ],
        timeout=300.0,
    )
    assert (final / "metadata" / "codes.parquet").exists(), (
        f"EQ_process_data exited 0 but did not produce the expected metadata file under {final}"
    )
    return final


@pytest.fixture(scope="session")
def eq_sampled_tasks_dir(eq_preprocessed_dataset: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Runs ``EQ_generate_tasks`` for both ``train`` and ``tuning`` splits against
    ``eq_preprocessed_dataset``.

    Returns the sampler's ``out_dir`` directly — the contract with downstream tooling
    (MTD's ``task_labels_dir``, ``EQ_train``'s ``datamodule.config.task_labels_dir``) is
    that the sampler's output dir *is* the labels dir.  If the sampler leaks non-label
    parquets into this directory, MTD's ``rglob("*.parquet")`` picks them up, schemas
    don't match, and ``pl.concat`` on ``labels_df`` fails — which is exactly the
    production-facing failure mode the integration tests need to catch, not hide.
    """
    intermediate = eq_preprocessed_dataset.parent / "intermediate"
    assert intermediate.exists(), (
        "eq_sampled_tasks_dir expects eq_preprocessed_dataset.parent / 'intermediate' to exist "
        "(produced by EQ_process_data)"
    )

    out_dir = tmp_path_factory.mktemp("eq_tasks")
    for split in (train_split, tuning_split):
        # Seed pinned explicitly (not inherited from the config default) so tests that compare
        # against this fixture's output — notably the reproducibility test in
        # test_generate_tasks.py — stay anchored to a known value even if the config default
        # changes.
        run_and_check(
            [
                "EQ_generate_tasks",
                f"data_dir={intermediate!s}",
                f"codes_dir={eq_preprocessed_dataset!s}",
                f"out_dir={out_dir!s}",
                f"split={split}",
                "input_shard=0",
                "task_shard=0",
                "n_tasks=8",
                "contexts_per_task=2",
                "duration_min=1",
                "duration_max=30",
                "min_context_per_subject=1",
                "seed=1",
            ],
            timeout=120.0,
        )

    return out_dir


@pytest.fixture(scope="session")
def eq_trained_model_dir(
    eq_preprocessed_dataset: Path,
    eq_sampled_tasks_dir: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Runs ``EQ_train --config-name=_demo_train`` to produce a checkpoint.

    Returns the training ``output_dir``.  Callers should find a checkpoint under
    ``{output_dir}/checkpoints/last.ckpt`` and a resolved config under
    ``{output_dir}/resolved_config.yaml``.
    """
    output_dir = tmp_path_factory.mktemp("eq_train_out")
    run_and_check(
        [
            "EQ_train",
            "--config-name=_demo_train",
            f"output_dir={output_dir!s}",
            f"datamodule.config.tensorized_cohort_dir={eq_preprocessed_dataset!s}",
            f"datamodule.config.task_labels_dir={eq_sampled_tasks_dir!s}",
        ],
        env=ENSURE_ENV_PLACEHOLDERS,
        timeout=300.0,
    )
    return output_dir


@pytest.fixture(scope="session")
def task_labels_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Sampler-output-shaped task labels for both splits, in long format.

    Layout matches ``sample_tasks.run_worker`` output::

        {task_labels_dir}/{split}/{shard}.parquet

    Columns: ``subject_id, prediction_time, boolean_value, occurs, query, duration_days``.
    """
    task_dir = tmp_path_factory.mktemp("eq_task_labels")

    for split, subjects in [(train_split, _TRAIN_SUBJECTS), (tuning_split, _TUNING_SUBJECTS)]:
        split_dir = task_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for i, (subj, query_code) in enumerate((s, q) for s in subjects for q in _QUERY_CODES):
            rows.append(
                {
                    "subject_id": subj,
                    "prediction_time": _PRED_TIMES[subj],
                    "boolean_value": i % 2 == 0,
                    "occurs": i % 3 != 0,
                    "query": query_code,
                    "duration_days": 30,
                }
            )

        df = pl.DataFrame(rows).cast(
            {
                "prediction_time": pl.Datetime("us"),
                "subject_id": pl.Int64,
                "boolean_value": pl.Boolean,
                "occurs": pl.Boolean,
                "query": pl.Utf8,
                "duration_days": pl.Int64,
            }
        )
        df.write_parquet(split_dir / "0.parquet")

    return task_dir


@pytest.fixture(scope="session")
def demo_dataset(
    tensorized_cohort_dir: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> EveryQueryPytorchDataset:
    """``EveryQueryPytorchDataset`` for the *train* split backed by sampler-shaped task labels."""

    collated_dir = tmp_path_factory.mktemp("eq_collated")

    rows = []
    for i, (subj, query_code) in enumerate((s, q) for s in _TRAIN_SUBJECTS for q in _QUERY_CODES):
        rows.append(
            {
                "subject_id": subj,
                "prediction_time": _PRED_TIMES[subj],
                "boolean_value": i % 2 == 0,
                "occurs": i % 3 != 0,
                "query": query_code,
                "duration_days": 30.0,
            }
        )

    df = pl.DataFrame(rows).cast(
        {
            "prediction_time": pl.Datetime("us"),
            "subject_id": pl.Int64,
            "boolean_value": pl.Boolean,
            "occurs": pl.Boolean,
            "query": pl.Utf8,
            "duration_days": pl.Float64,
        }
    )
    df.write_parquet(collated_dir / "0.parquet")

    cfg = MEDSTorchDataConfig(
        tensorized_cohort_dir=str(tensorized_cohort_dir),
        task_labels_dir=str(collated_dir),
        max_seq_len=64,
        seq_sampling_strategy="to_end",
        static_inclusion_mode="omit",
        batch_mode="SM",
    )

    return EveryQueryPytorchDataset(cfg, split=train_split)


@pytest.fixture(scope="session")
def sample_batch(demo_dataset: EveryQueryPytorchDataset) -> EveryQueryBatch:
    """A two-sample ``EveryQueryBatch`` for unit / doctest use."""
    items = [demo_dataset[0], demo_dataset[1]]
    return demo_dataset.collate(items)


# ── Doctest namespace injection ─────────────────────────────────────────


@pytest.fixture(autouse=True)
def _setup_doctest_namespace(
    doctest_namespace,
    demo_model,
    demo_lightning_module,
    sample_batch,
    demo_model_config,
):
    """Makes common objects available as bare names inside ``>>>`` blocks."""
    doctest_namespace.update(
        {
            "torch": torch,
            "np": np,
            "demo_model": demo_model,
            "demo_lightning_module": demo_lightning_module,
            "sample_batch": sample_batch,
            "demo_model_config": demo_model_config,
            "Mock": Mock,
            "patch": patch,
            "tempfile": tempfile,
        }
    )
