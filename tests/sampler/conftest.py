"""Fixtures for the EveryQuery task-sampler adapter tests."""

from datetime import datetime, timedelta

import polars as pl
import pytest


@pytest.fixture(autouse=True)
def _setup_doctest_namespace():
    """Avoid constructing the Hugging Face demo model for adapter-only tests."""
    yield


@pytest.fixture
def synthetic_query_codes() -> list[str]:
    """Return a small code universe for adapter tests."""
    return ["ICD//A01", "ICD//B02", "ICD//C03", "MED//D04", "MED//E05"]


@pytest.fixture
def synthetic_events() -> pl.DataFrame:
    """Return a deterministic MEDS event table for orchestration tests."""
    codes = ["ICD//A01", "ICD//B02", "ICD//C03", "MED//D04", "MED//E05"]
    base = datetime(2020, 1, 1)  # noqa: DTZ001 - MEDS test timestamps are intentionally naive.
    rows = [
        {
            "subject_id": subject_id,
            "time": base + timedelta(days=10 * event_index),
            "code": codes[event_index % len(codes)],
        }
        for subject_id in range(1, 4)
        for event_index in range(30)
    ]
    return pl.DataFrame(rows)
