"""Subprocess integration test for the consolidated ``EQ_evaluate`` entry point.

``EQ_evaluate`` now resolves to ``every_query.evaluate.evaluate:main``: consumes a
``PredictionSchema`` parquet written by ``EQ_predict`` and emits per-``(query,
duration_days)`` metrics.  The legacy four-stage evaluator has been deleted; recover
from git history if archival reference is needed.

This test exercises the full subprocess path: schema-aligned parquet read,
per-``(query, duration_days)`` metric computation, aligned parquet write.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from every_query.predict.schema import PredictionSchema

_VENV_BIN = str(Path(sys.executable).parent)


def _make_predictions_parquet(tmp_path: Path) -> Path:
    """Small synthetic PredictionSchema-conformant parquet with two queries."""
    rows = []
    for i in range(6):
        # Three rows per query: mix of censored (null boolean_value), positive, negative.
        query = "A" if i < 3 else "B"
        boolean_value = [True, False, None, True, True, None][i]
        censor_prob = [0.05, 0.10, 0.90, 0.02, 0.08, 0.80][i]
        occurs_prob = [0.9, 0.1, 0.5, 0.8, 0.7, 0.5][i]
        rows.append(
            {
                "subject_id": 1000 + i,
                "prediction_time": datetime(2023, 1, 1),
                "query": query,
                "duration_days": 30.0,
                "boolean_value": boolean_value,
                "censor_prob": censor_prob,
                "occurs_prob": occurs_prob,
            }
        )
    df = pl.DataFrame(rows).cast(
        {
            "subject_id": pl.Int64,
            "prediction_time": pl.Datetime("us"),
            "query": pl.Utf8,
            "duration_days": pl.Float32,
            "boolean_value": pl.Boolean,
            "censor_prob": pl.Float32,
            "occurs_prob": pl.Float32,
        }
    )
    out = tmp_path / "predictions.parquet"
    df.write_parquet(out)
    # Align at write to fail if dtype drift creeps in.
    PredictionSchema.align(pq.read_table(out))
    return out


def test_evaluate_end_to_end(tmp_path: Path) -> None:
    """``EQ_evaluate`` produces a per-group metrics parquet."""
    predictions_parquet = _make_predictions_parquet(tmp_path)
    metrics_parquet = tmp_path / "metrics.parquet"

    env = os.environ.copy()
    env["PATH"] = _VENV_BIN + os.pathsep + env.get("PATH", "")

    cmd = [
        "EQ_evaluate",
        f"predictions_parquet={predictions_parquet}",
        f"metrics_parquet={metrics_parquet}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=60)
    assert result.returncode == 0, (
        f"EQ_evaluate failed (rc={result.returncode})\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert metrics_parquet.exists(), "evaluate.py did not produce the metrics parquet"

    metrics = pl.read_parquet(metrics_parquet).sort("query")
    assert metrics.height == 2  # one row per (query, duration_days) group
    assert set(metrics.columns) >= {
        "query",
        "duration_days",
        "n_rows",
        "n_occurs_labeled",
        "n_positive",
        "occurs_auroc",
        "censor_auroc",
        "prevalence",
    }
    rows = metrics.to_dicts()
    # Query "A" has a mixed-label labeled subset ([True, False]), so occurs_auroc is defined (1.0).
    # Query "B" has a single-class labeled subset ([True, True]), so occurs_auroc is null.
    assert rows[0]["query"] == "A"
    assert rows[0]["occurs_auroc"] == 1.0
    assert rows[1]["query"] == "B"
    assert rows[1]["occurs_auroc"] is None
    # Censor AUROC is defined for both groups (censor_prob correctly ranks censored rows higher).
    assert rows[0]["censor_auroc"] == 1.0
    assert rows[1]["censor_auroc"] == 1.0
    # Prevalence is n_positive / n_occurs_labeled: A has 1 positive of 2 labeled (0.5),
    # B has 2 positives of 2 labeled (1.0).
    assert rows[0]["prevalence"] == 0.5
    assert rows[1]["prevalence"] == 1.0
