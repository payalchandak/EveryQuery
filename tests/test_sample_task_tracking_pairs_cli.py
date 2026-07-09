"""Subprocess integration test for ``EQ_sample_task_tracking_pairs``.

Chains onto ``EQ_generate_evaluation_tasks(split=tuning)`` (already covered by
``test_generate_evaluation_tasks_cli.py``) to exercise the second, new stage that
compacts a dense per-shard label grid down to one positive + one negative row per
``(query, duration_days)`` task — the input ``EveryQueryLightningModule``'s optional
task-tracking dataloader reads.

Checks:
1. Invokes both CLIs in sequence against the session's preprocessed cohort.
2. Verifies the output parquet is ``TaskQuerySchema``-conformant.
3. Verifies the pairing invariant: every surviving task has exactly one ``True`` row
   and one ``False`` row (when the demo cohort produces any tasks at all — like the
   sibling evaluation-tasks test, this is not guaranteed to be non-empty).
4. Verifies determinism — two runs with the same seed produce identical output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pyarrow.parquet as pq

from conftest import run_and_check
from every_query.data.schema import TaskQuerySchema

if TYPE_CHECKING:
    from pathlib import Path


def _generate_tuning_eval_labels(intermediate: Path, out_dir: Path, seed: int) -> Path:
    run_and_check(
        [
            "EQ_generate_evaluation_tasks",
            f"data_dir={intermediate!s}",
            f"out_dir={out_dir!s}",
            "split=tuning",
            "input_shard=0",
            "prediction_times_per_subject=3",
            "min_context_per_subject=1",
            "query_codes=[HR,TEMP]",
            "durations=[1,7,30]",
            f"seed={seed}",
        ],
        timeout=120.0,
    )
    return out_dir / "eval"


def test_eq_sample_task_tracking_pairs_end_to_end(
    eq_preprocessed_dataset: Path,
    tmp_path: Path,
) -> None:
    """CLI writes a TaskQuerySchema parquet with exactly one pos + one neg row per surviving task."""
    intermediate = eq_preprocessed_dataset.parent / "intermediate"
    eval_out = tmp_path / "eval_tasks"
    tracking_out = tmp_path / "task_tracking"

    eval_labels_dir = _generate_tuning_eval_labels(intermediate, eval_out, seed=42)

    run_and_check(
        [
            "EQ_sample_task_tracking_pairs",
            f"eval_labels_dir={eval_labels_dir!s}",
            f"out_dir={tracking_out!s}",
            "split=tuning",
            "seed=1",
        ],
        timeout=60.0,
    )

    pairs_fp = tracking_out / "tuning" / "0.parquet"
    assert pairs_fp.is_file(), f"expected {pairs_fp} to exist"

    TaskQuerySchema.align(pq.read_table(pairs_fp))

    pairs = pl.read_parquet(pairs_fp)
    assert pairs[TaskQuerySchema.boolean_value_name].null_count() == 0, (
        "task-tracking pairs should never contain censored (null boolean_value) rows"
    )

    if pairs.height > 0:
        task_cols = [TaskQuerySchema.query_name, TaskQuerySchema.duration_days_name]
        per_task = pairs.group_by(task_cols).agg(
            n_rows=pl.len(),
            n_classes=pl.col(TaskQuerySchema.boolean_value_name).n_unique(),
        )
        assert (per_task["n_rows"] == 2).all(), (
            f"every tracked task should contribute exactly 2 rows, got: {per_task}"
        )
        assert (per_task["n_classes"] == 2).all(), (
            f"every tracked task should have one positive and one negative row, got: {per_task}"
        )


def test_eq_sample_task_tracking_pairs_deterministic(
    eq_preprocessed_dataset: Path,
    tmp_path: Path,
) -> None:
    """Two runs with the same seed produce equivalent parquet contents (row-for-row)."""
    intermediate = eq_preprocessed_dataset.parent / "intermediate"
    eval_out = tmp_path / "eval_tasks"
    eval_labels_dir = _generate_tuning_eval_labels(intermediate, eval_out, seed=7)

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    for out in (out_a, out_b):
        run_and_check(
            [
                "EQ_sample_task_tracking_pairs",
                f"eval_labels_dir={eval_labels_dir!s}",
                f"out_dir={out!s}",
                "split=tuning",
                "seed=3",
            ],
            timeout=60.0,
        )

    sort_cols = [
        TaskQuerySchema.subject_id_name,
        TaskQuerySchema.prediction_time_name,
        TaskQuerySchema.query_name,
        TaskQuerySchema.duration_days_name,
    ]
    df_a = pl.read_parquet(out_a / "tuning" / "0.parquet").sort(sort_cols)
    df_b = pl.read_parquet(out_b / "tuning" / "0.parquet").sort(sort_cols)
    assert df_a.equals(df_b), (
        "EQ_sample_task_tracking_pairs should be deterministic in (seed, split, eval_labels_dir)"
    )
