"""Subprocess integration test for ``EQ_predict``.

Acceptance criterion from #81: *"CPU-only integration test in ``tests/test_predict_cli.py``
that exercises the full subprocess path."*

Uses the session-scoped ``eq_trained_model_dir`` fixture (a real trained demo checkpoint +
``resolved_config.yaml``) and builds a ``TaskQuerySchema``-conformant tasks *directory*
with a single parquet, whose subjects live in the ``held_out`` split of the training
cohort.  Runs ``EQ_predict`` in a subprocess and verifies the output is
``PredictionSchema``-conformant with probabilities in ``[0, 1]`` and one row per input
task.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import pytest

from every_query.data.schema import TaskQuerySchema
from every_query.predict.schema import PredictionSchema

_VENV_BIN = str(Path(sys.executable).parent)

# Subject lives in the ``held_out`` split of the ``simple_static_sharded_by_split``
# testing dataset used by ``eq_preprocessed_dataset``.  Prediction time falls inside
# 1500733's event sequence (2010-06-03).
_HELD_OUT_SUBJECT = 1500733
_PRED_TIME = datetime(2010, 6, 3, 15, 0, 0)
_QUERY_CODES = ["HR", "TEMP"]
_DURATION_DAYS = 30.0


def _write_tasks_parquet(fp: Path, columns: dict[str, list]) -> None:
    """Write a TaskQuerySchema-aligned parquet from a column dict.

    Routes the frame through ``TaskQuerySchema.align`` for dtypes (so test sites
    don't restate per-column ``pl.Int64`` / ``pl.Float32`` casts) and writes the
    aligned arrow table directly via ``pyarrow.parquet`` — no extra polars
    round-trip on the way out.
    """
    aligned = TaskQuerySchema.align(pl.DataFrame(columns).to_arrow())
    pq.write_table(aligned, fp)


@pytest.fixture(scope="module")
def predict_tasks_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A directory containing a single TaskQuerySchema-conformant parquet for EQ_predict."""
    tasks_dir = tmp_path_factory.mktemp("eq_predict_tasks")
    _write_tasks_parquet(
        tasks_dir / "tasks.parquet",
        {
            "subject_id": [_HELD_OUT_SUBJECT] * len(_QUERY_CODES),
            "prediction_time": [_PRED_TIME] * len(_QUERY_CODES),
            "query": _QUERY_CODES,
            "duration_days": [_DURATION_DAYS] * len(_QUERY_CODES),
            "boolean_value": [None] * len(_QUERY_CODES),
        },
    )
    return tasks_dir


def test_eq_predict_end_to_end(
    eq_trained_model_dir: Path,
    predict_tasks_dir: Path,
    tmp_path: Path,
) -> None:
    """``EQ_predict`` runs end-to-end and produces a ``PredictionSchema``-conformant parquet.

    Exercises the full subprocess entry point — console-script resolution, Hydra config compose, model
    checkpoint load, single predict pass, row-order-preserved hstack with tasks_df, schema-aligned write.
    """
    output_parquet = tmp_path / "predictions.parquet"

    env = os.environ.copy()
    env["PATH"] = _VENV_BIN + os.pathsep + env.get("PATH", "")

    cmd = [
        "EQ_predict",
        f"model_run_dir={eq_trained_model_dir}",
        f"tasks_dir={predict_tasks_dir}",
        f"output_parquet={output_parquet}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
    assert result.returncode == 0, (
        f"EQ_predict failed (rc={result.returncode})\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert output_parquet.exists(), "EQ_predict did not produce the output parquet"

    # Schema conformance.
    table = pq.read_table(output_parquet)
    PredictionSchema.align(table)

    df = pl.from_arrow(table)
    assert df.height == len(_QUERY_CODES), (
        f"Expected {len(_QUERY_CODES)} prediction rows (one per query), got {df.height}"
    )

    # Probabilities bounded.
    for col in ("censor_prob", "occurs_prob"):
        col_min = float(df[col].min())
        col_max = float(df[col].max())
        assert 0.0 <= col_min <= col_max <= 1.0, f"{col} not in [0,1]: min={col_min} max={col_max}"

    # Every input query is represented in the output.
    assert set(df["query"].to_list()) == set(_QUERY_CODES)


def test_eq_predict_preserves_input_row_order(
    eq_trained_model_dir: Path,
    tmp_path: Path,
) -> None:
    """Multi-query/durations sanity-check: output row identifiers match the input parquet in order.

    Guards the implementation's load-bearing assumption that ``D.test_dataset.schema_df``
    preserves the input labels frame's row order (per MTD's
    ``get_task_seq_bounds_and_labels`` guarantee).  If a future MTD bump reorders
    schema_df rows, the output ``(query, duration_days)`` column would desync from the
    input parquet and this test would fail.
    """
    # Non-alphabetical query order, mixed durations — exercises the order check past
    # the alphabetical HR/TEMP happy path of the main integration test.
    expected_queries = ["TEMP", "HR", "TEMP"]
    expected_durations = [60.0, 30.0, 30.0]

    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    _write_tasks_parquet(
        tasks_dir / "tasks.parquet",
        {
            "subject_id": [_HELD_OUT_SUBJECT] * len(expected_queries),
            "prediction_time": [_PRED_TIME] * len(expected_queries),
            "query": expected_queries,
            "duration_days": expected_durations,
            "boolean_value": [None] * len(expected_queries),
        },
    )
    output_parquet = tmp_path / "predictions.parquet"

    env = os.environ.copy()
    env["PATH"] = _VENV_BIN + os.pathsep + env.get("PATH", "")
    result = subprocess.run(
        [
            "EQ_predict",
            f"model_run_dir={eq_trained_model_dir}",
            f"tasks_dir={tasks_dir}",
            f"output_parquet={output_parquet}",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"EQ_predict failed (rc={result.returncode})\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    df_out = pl.from_arrow(pq.read_table(output_parquet))
    assert df_out["query"].to_list() == expected_queries, (
        f"Output query order {df_out['query'].to_list()} doesn't match input {expected_queries}"
    )
    assert df_out["duration_days"].to_list() == expected_durations, (
        f"Output duration_days order {df_out['duration_days'].to_list()} doesn't match input "
        f"{expected_durations}"
    )
