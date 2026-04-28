"""Unit tests for EQ_predict internals.

Complements ``test_predict_cli.py``'s subprocess integration coverage with direct,
in-process exercise of the streaming writer and the multi-process guard.  These cover
gaps the subprocess tests can't reach cheaply: multi-batch row-order across the writer's
offset boundary, the empty-cohort path, and the DDP / multi-device refusal path.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import lightning as L
import polars as pl
import pyarrow.parquet as pq
import pytest
import torch

from every_query.predict.predict import (
    _check_single_process_trainer,
    _StreamingPredictionWriter,
)
from every_query.predict.schema import PredictionSchema

if TYPE_CHECKING:
    from pathlib import Path


def _make_schema_df(n: int) -> pl.DataFrame:
    """Build a fake ``schema_df`` shaped like MTD's post-collapse output.

    Columns mirror :class:`every_query.data.dataset.EveryQueryPytorchDataset` after the
    ``boolean_value`` → ``(censor, occurs)`` collapse: identifier columns plus
    ``boolean_value`` (censor flag) and ``occurs`` (true outcome when not censored).

    Dtypes are pinned explicitly so the ``n=0`` fixture matches what MTD actually
    produces (parquet schema is preserved through ``head(0)`` in production); polars'
    default inference would assign ``null`` to empty Boolean columns and break the
    ``boolean_value`` reconstruction in ``_identifiers_from_schema_df``.
    """
    return pl.DataFrame(
        {
            "subject_id": list(range(10, 10 + n)),
            "prediction_time": [datetime(2024, 1, 1 + i % 28) for i in range(n)],
            "query": [["A", "B", "C"][i % 3] for i in range(n)],
            "duration_days": [30.0 * (1 + i % 3) for i in range(n)],
            # Alternate censor flags so boolean_value reconstruction exercises both branches.
            "boolean_value": [i % 2 == 1 for i in range(n)],
            "occurs": [i % 3 == 0 for i in range(n)],
        },
        schema_overrides={
            "subject_id": pl.Int64,
            "duration_days": pl.Float32,
            "boolean_value": pl.Boolean,
            "occurs": pl.Boolean,
        },
    )


def _drive_writer(writer: _StreamingPredictionWriter, batches: list[tuple[list[float], list[float]]]) -> None:
    """Mimic Lightning's predict lifecycle: setup → write_on_batch_end* → teardown."""
    writer.setup(trainer=None, pl_module=None, stage="predict")
    for batch_idx, (censor, occurs) in enumerate(batches):
        writer.write_on_batch_end(
            trainer=None,
            pl_module=None,
            prediction={
                "censor_probs": torch.tensor(censor),
                "occurs_probs": torch.tensor(occurs),
            },
            batch_indices=None,
            batch=None,
            batch_idx=batch_idx,
            dataloader_idx=0,
        )
    writer.teardown(trainer=None, pl_module=None, stage="predict")


def test_streaming_writer_single_batch_preserves_identifiers_and_probs(tmp_path: Path) -> None:
    """One full-cohort batch → one row group, identifiers + probs land 1:1 in input order."""
    schema_df = _make_schema_df(3)
    output = tmp_path / "predictions.parquet"
    writer = _StreamingPredictionWriter(output, schema_df)

    _drive_writer(writer, [([0.1, 0.2, 0.3], [0.7, 0.6, 0.5])])

    table = pq.read_table(output)
    PredictionSchema.align(table)  # raises if non-conformant
    df = pl.from_arrow(table)

    assert df.height == 3
    assert df["subject_id"].to_list() == [10, 11, 12]
    # boolean_value reconstruction: censor=True → null, censor=False → occurs
    # schema_df rows 0,1,2 have boolean_value=[F, T, F] and occurs=[T, F, F]
    assert df["boolean_value"].to_list() == [True, None, False]
    assert [round(x, 2) for x in df["censor_prob"].to_list()] == [0.10, 0.20, 0.30]
    assert [round(x, 2) for x in df["occurs_prob"].to_list()] == [0.70, 0.60, 0.50]
    assert pq.ParquetFile(output).num_row_groups == 1


def test_streaming_writer_multi_batch_preserves_row_order_across_offset(tmp_path: Path) -> None:
    """Multi-batch + partial-last-batch — covers the offset-tracking gap subprocess tests miss.

    The CLI fixture's ``tasks.parquet`` fits in a single batch under the demo model's
    batch_size, so the existing subprocess tests don't exercise multiple
    ``write_on_batch_end`` calls.  This test forces 3 batches of differing sizes and
    verifies the writer slices ``schema_df`` correctly across the boundaries.
    """
    schema_df = _make_schema_df(6)
    output = tmp_path / "predictions.parquet"
    writer = _StreamingPredictionWriter(output, schema_df)

    # Three batches: 3 rows, 2 rows, 1 row (partial last batch).
    _drive_writer(
        writer,
        [
            ([0.10, 0.20, 0.30], [0.90, 0.80, 0.70]),
            ([0.40, 0.50], [0.60, 0.55]),
            ([0.60], [0.50]),
        ],
    )

    df = pl.from_arrow(pq.read_table(output))
    assert df.height == 6
    assert df["subject_id"].to_list() == [10, 11, 12, 13, 14, 15]
    assert [round(x, 2) for x in df["censor_prob"].to_list()] == [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
    assert [round(x, 2) for x in df["occurs_prob"].to_list()] == [0.90, 0.80, 0.70, 0.60, 0.55, 0.50]
    # One row group per batch — confirms streaming, not buffer-then-write.
    assert pq.ParquetFile(output).num_row_groups == 3


def test_streaming_writer_empty_cohort_produces_valid_empty_parquet(tmp_path: Path) -> None:
    """Zero-row schema_df → valid PredictionSchema parquet with zero rows (matches BC)."""
    output = tmp_path / "predictions.parquet"
    writer = _StreamingPredictionWriter(output, _make_schema_df(0))

    _drive_writer(writer, [])

    assert output.exists()
    table = pq.read_table(output)
    PredictionSchema.align(table)
    assert table.num_rows == 0


def test_check_single_process_trainer_accepts_single_device() -> None:
    """The guard is a no-op for the canonical single-device trainer config."""
    trainer = L.Trainer(devices=1, accelerator="cpu", logger=False)
    _check_single_process_trainer(trainer)  # no raise


def test_check_single_process_trainer_rejects_ddp() -> None:
    """The guard refuses any trainer whose ``world_size > 1`` (DDP / multi-device)."""
    trainer = L.Trainer(devices=2, accelerator="cpu", strategy="ddp", logger=False)
    with pytest.raises(RuntimeError, match=r"single-process prediction.*world_size=2"):
        _check_single_process_trainer(trainer)
