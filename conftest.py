"""Root conftest — session-scoped fixture DAG for the EveryQuery test suite.

Fixture dependency graph (all session-scoped)::

    demo_model ──────────► demo_lightning_module

    simple_static_MEDS ──► tensorized_cohort_dir
                                    │
                                    ▼
                               demo_dataset
                                    │
                                    ▼
                               sample_batch
"""

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

from every_query.dataset import EveryQueryBatch, EveryQueryPytorchDataset
from every_query.lightning_module import EveryQueryLightningModule
from every_query.model import EveryQueryModel

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
    """Randomly-initialised EveryQueryModel with tiny dimensions."""
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
    import os

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
