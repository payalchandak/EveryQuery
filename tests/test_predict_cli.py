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
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from every_query.data.schema import TaskQuerySchema
from every_query.predict.schema import PredictionSchema

_VENV_BIN = str(Path(sys.executable).parent)

# Subject lives in the ``held_out`` split of the ``simple_static_sharded_by_split``
# testing dataset used by ``eq_preprocessed_dataset``.  Prediction time falls inside
# 1500733's event sequence (2010-06-03).
# Query codes must exist in the model's training vocab — ``EQ_predict`` now hard-errors
# on out-of-vocab codes (was: warn).  These are real codes in the
# ``simple_static_sharded_by_split`` testing dataset's vocab after preprocessing.
_HELD_OUT_SUBJECT = 1500733
_PRED_TIME = datetime(2010, 6, 3, 15, 0, 0)
_QUERY_CODES = ["DISCHARGE", "DOB"]
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
    expected_queries = ["DOB", "DISCHARGE", "DOB"]
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


def test_eq_predict_batch_size_override(
    eq_trained_model_dir: Path,
    predict_tasks_dir: Path,
    tmp_path: Path,
) -> None:
    """``batch_size=N`` overrides the inherited training ``datamodule.batch_size``.

    Runs predict with ``batch_size=1`` (forcing per-row batches) against the same
    fixture as the baseline test and asserts the output is still
    ``PredictionSchema``-conformant with the expected row count, schema, and
    bounded probabilities.  The override path mutates ``train_cfg`` in memory only;
    ``resolved_config.yaml`` on disk is untouched.
    """
    output_parquet = tmp_path / "predictions.parquet"

    env = os.environ.copy()
    env["PATH"] = _VENV_BIN + os.pathsep + env.get("PATH", "")

    cmd = [
        "EQ_predict",
        f"model_run_dir={eq_trained_model_dir}",
        f"tasks_dir={predict_tasks_dir}",
        f"output_parquet={output_parquet}",
        "batch_size=1",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
    assert result.returncode == 0, (
        f"EQ_predict failed (rc={result.returncode})\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert output_parquet.exists(), "EQ_predict did not produce the output parquet"

    table = pq.read_table(output_parquet)
    PredictionSchema.align(table)

    df = pl.from_arrow(table)
    assert df.height == len(_QUERY_CODES), (
        f"Expected {len(_QUERY_CODES)} prediction rows (one per query), got {df.height}"
    )
    for col in ("censor_prob", "occurs_prob"):
        col_min = float(df[col].min())
        col_max = float(df[col].max())
        assert 0.0 <= col_min <= col_max <= 1.0, f"{col} not in [0,1]: min={col_min} max={col_max}"
    assert set(df["query"].to_list()) == set(_QUERY_CODES)


def test_eq_predict_num_workers_override(
    eq_trained_model_dir: Path,
    predict_tasks_dir: Path,
    tmp_path: Path,
) -> None:
    """``num_workers=N`` overrides the inherited training ``datamodule.num_workers``.

    Runs predict with ``num_workers=2`` (multi-worker dataloading) against the same
    fixture as the baseline test and asserts the output is still
    ``PredictionSchema``-conformant with the expected row count and bounded
    probabilities.  The override path mutates ``train_cfg`` in memory only;
    ``resolved_config.yaml`` on disk is untouched.
    """
    output_parquet = tmp_path / "predictions.parquet"

    env = os.environ.copy()
    env["PATH"] = _VENV_BIN + os.pathsep + env.get("PATH", "")

    cmd = [
        "EQ_predict",
        f"model_run_dir={eq_trained_model_dir}",
        f"tasks_dir={predict_tasks_dir}",
        f"output_parquet={output_parquet}",
        "num_workers=2",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
    assert result.returncode == 0, (
        f"EQ_predict failed (rc={result.returncode})\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert output_parquet.exists(), "EQ_predict did not produce the output parquet"

    table = pq.read_table(output_parquet)
    PredictionSchema.align(table)

    df = pl.from_arrow(table)
    assert df.height == len(_QUERY_CODES), (
        f"Expected {len(_QUERY_CODES)} prediction rows (one per query), got {df.height}"
    )
    for col in ("censor_prob", "occurs_prob"):
        col_min = float(df[col].min())
        col_max = float(df[col].max())
        assert 0.0 <= col_min <= col_max <= 1.0, f"{col} not in [0,1]: min={col_min} max={col_max}"


def test_eq_predict_save_embeddings(
    eq_trained_model_dir: Path,
    predict_tasks_dir: Path,
    tmp_path: Path,
) -> None:
    """``save_embeddings=true`` writes a sibling parquet with row-aligned pooled embeddings.

    Covers the issue #169 acceptance: enabling the new flag produces a
    ``<output_parquet stem>.embeddings.parquet`` sibling carrying the
    ``TaskQuerySchema`` identifiers plus a fixed-size-list ``embedding`` column
    whose width matches the model's ``hidden_size``.  Also verifies that the
    sibling participates in the existing ``overwrite`` clobber-protection
    contract.
    """
    output_parquet = tmp_path / "predictions.parquet"
    embeddings_parquet = output_parquet.with_suffix(".embeddings" + output_parquet.suffix)

    env = os.environ.copy()
    env["PATH"] = _VENV_BIN + os.pathsep + env.get("PATH", "")

    cmd = [
        "EQ_predict",
        f"model_run_dir={eq_trained_model_dir}",
        f"tasks_dir={predict_tasks_dir}",
        f"output_parquet={output_parquet}",
        "save_embeddings=true",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
    assert result.returncode == 0, (
        f"EQ_predict failed (rc={result.returncode})\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    # Predictions branch is unchanged — sanity-check it still landed.
    assert output_parquet.exists(), "predictions parquet was not written"
    PredictionSchema.align(pq.read_table(output_parquet))

    # Embeddings sibling lives next to predictions at the derived path.
    assert embeddings_parquet.exists(), (
        f"embeddings sibling not found at {embeddings_parquet}; got dir contents: "
        f"{sorted(p.name for p in tmp_path.iterdir())}"
    )

    emb_table = pq.read_table(embeddings_parquet)
    schema = emb_table.schema

    # TaskQuerySchema identifier columns are present.
    expected_id_cols = {
        TaskQuerySchema.subject_id_name,
        TaskQuerySchema.prediction_time_name,
        TaskQuerySchema.query_name,
        TaskQuerySchema.duration_days_name,
    }
    assert expected_id_cols.issubset(set(schema.names)), (
        f"embeddings parquet missing identifier columns; got {schema.names}"
    )

    # The ``embedding`` column is a fixed-size-list of float32 with width > 0
    # (the demo model's hidden_size is implementation detail; assert the
    # invariant, not the constant).
    assert "embedding" in schema.names, f"embeddings parquet missing 'embedding' column; got {schema.names}"
    emb_type = schema.field("embedding").type
    assert pa.types.is_fixed_size_list(emb_type), (
        f"embedding column must be a fixed-size-list, got {emb_type!r}"
    )
    assert pa.types.is_float32(emb_type.value_type), (
        f"embedding column inner type must be float32, got {emb_type.value_type!r}"
    )
    assert emb_type.list_size > 0, f"embedding hidden_size must be > 0, got {emb_type.list_size}"

    # Row count matches the input task count, and identifier columns match the
    # predictions parquet row-for-row (same `_identifiers_from_schema_df` source).
    pred_df = pl.from_arrow(pq.read_table(output_parquet))
    emb_df = pl.from_arrow(emb_table)
    assert emb_df.height == pred_df.height, (
        f"embeddings row count {emb_df.height} != predictions row count {pred_df.height}"
    )
    for col in (
        TaskQuerySchema.subject_id_name,
        TaskQuerySchema.prediction_time_name,
        TaskQuerySchema.query_name,
        TaskQuerySchema.duration_days_name,
    ):
        assert emb_df[col].to_list() == pred_df[col].to_list(), (
            f"embeddings {col} ordering doesn't match predictions {col}"
        )

    # Embeddings sibling participates in the clobber-protection contract.  The
    # predictions check fires first when both files exist, so to isolate the
    # embeddings check we remove predictions and re-run — embeddings should now
    # be the file that blocks the run.
    output_parquet.unlink()
    assert embeddings_parquet.exists(), "embeddings sibling unexpectedly missing for clobber test"
    rerun = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
    assert rerun.returncode != 0, "expected re-run to fail (no overwrite=true) but it succeeded"
    assert "FileExistsError" in rerun.stderr, (
        f"expected FileExistsError in stderr, got:\nstdout:\n{rerun.stdout}\nstderr:\n{rerun.stderr}"
    )
    assert str(embeddings_parquet) in rerun.stderr, (
        f"expected embeddings path in stderr, got:\nstderr:\n{rerun.stderr}"
    )
