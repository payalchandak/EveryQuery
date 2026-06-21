"""Shared fixtures and helpers for the 5-stage training-task sampler unit suite.

This package (``tests/sampler/``) is the per-stage unit layer for
``every_query.generate_tasks.sample_tasks``.  Every stage here is a pure data transform
(sampling + labeling + parquet I/O) with **no model dependency**, so this conftest overrides the
repo-root ``_setup_doctest_namespace`` autouse fixture (see the top-level ``conftest.py``) to drop
its dependency on ``demo_model`` — which builds a ModernBERT via ``AutoConfig.from_pretrained`` and
therefore needs HuggingFace/network access.  Decoupling keeps this layer fast and offline-runnable;
the dataset-integration test that genuinely needs the model lives at
``tests/test_sampler_dataset_integration.py`` (top level, where the root fixture still applies).

Value fixtures (``synthetic_events``, ``synthetic_query_codes``, ``prediction_time_counts``) are
shared verbatim across stage files.  Cohort-builder helpers that take arguments are exposed as
fixtures returning callables (``subject_events``, ``write_split_shards``) so they're auto-discovered
without cross-module imports.
"""

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest


@pytest.fixture(autouse=True)
def _setup_doctest_namespace():
    """Override the repo-root autouse fixture so sampler unit tests don't build the HF demo model.

    The root ``_setup_doctest_namespace`` pulls in ``demo_model`` (network/HF). No doctests live
    under ``tests/sampler/``, so a no-op override is safe and keeps the layer offline + fast.
    """
    yield


# ---------------------------------------------------------------------------
# Value fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_events() -> pl.DataFrame:
    """A deterministic events DataFrame: 3 subjects x 30 events x 5 codes, 10d spacing."""
    codes = ["ICD//A01", "ICD//B02", "ICD//C03", "MED//D04", "MED//E05"]
    base = datetime(2020, 1, 1)
    rows = [
        {
            "subject_id": subj,
            "time": base + timedelta(days=i * 10 + subj),
            "code": codes[i % len(codes)],
        }
        for subj in range(1, 4)
        for i in range(30)
    ]
    return pl.DataFrame(rows).sort(["subject_id", "time"])


@pytest.fixture
def synthetic_query_codes() -> list[str]:
    return ["ICD//A01", "ICD//B02", "ICD//C03", "MED//D04", "MED//E05"]


@pytest.fixture
def prediction_time_counts() -> pl.DataFrame:
    """A small Stage 0 ``_prediction_time_counts`` table, sorted by ``subject_id``.

    Row position is ``subject_idx``; subjects span two shards with varying ``n_prediction_times``.
    """
    return pl.DataFrame(
        {
            "subject_id": [10, 20, 30, 40, 50],
            "shard": ["0", "0", "0", "1", "1"],
            "n_prediction_times": [60, 51, 200, 80, 120],
        }
    )


# ---------------------------------------------------------------------------
# Cohort-builder helpers (exposed as fixtures returning callables)
# ---------------------------------------------------------------------------


@pytest.fixture
def subject_events():
    """Return a builder for ``n_times`` distinct ``(subject_id, time)`` rows, each emitted ``dups``
    times (same time, different code) so Stage 0's distinct-time dedup is exercised.  Times are 1
    day apart starting at ``base``."""

    def _subject_events(subject_id: int, n_times: int, *, base: datetime, dups: int = 1) -> pl.DataFrame:
        rows = [
            {"subject_id": subject_id, "time": base + timedelta(days=i), "code": f"ICD//{d:02d}"}
            for i in range(n_times)
            for d in range(dups)
        ]
        return pl.DataFrame(rows)

    return _subject_events


@pytest.fixture
def write_split_shards():
    """Return a writer for ``{shard: events}`` -> ``tmp_path/intermediate/data/{split}/{shard}.parquet``;
    the callable returns the ``path_to_data`` root."""

    def _write_split_shards(
        tmp_path: Path, shard_to_events: dict[str, pl.DataFrame], split: str = "train"
    ) -> Path:
        data_dir = tmp_path / "intermediate"
        split_dir = data_dir / "data" / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for shard, df in shard_to_events.items():
            df.write_parquet(split_dir / f"{shard}.parquet")
        return data_dir

    return _write_split_shards
