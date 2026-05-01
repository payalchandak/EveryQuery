"""Subprocess integration test for ``EQ_generate_evaluation_tasks``.

Complements ``test_generate_tasks.py`` (pretraining-shape, scattered tasks) by
exercising the new evaluation-shape endpoint: a dense grid of
``subjects x sampled_times x codes x durations``.  This is what ``EQ_predict``
consumes to produce held-out predictions that ``EQ_evaluate`` then scores.

Checks:
1. Invokes the CLI against the session's preprocessed cohort.
2. Verifies the output parquet is ``TaskQuerySchema``-conformant.
3. Verifies the dense-grid shape (row count = sampled_times x |codes| x |durations|
   subjects permitting).
4. Verifies determinism — two runs with the same seed produce identical output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pyarrow.parquet as pq
import pytest
from meds import DataSchema

from conftest import ENSURE_ENV_PLACEHOLDERS, run_and_check
from every_query.data.schema import TaskQuerySchema
from every_query.generate_tasks.sample_evaluation_tasks import subsample_subject_ids

if TYPE_CHECKING:
    from pathlib import Path


def test_eq_generate_evaluation_tasks_end_to_end(
    eq_preprocessed_dataset: Path,
    tmp_path: Path,
) -> None:
    """``EQ_generate_evaluation_tasks`` writes a TaskQuerySchema-conformant dense-grid parquet."""
    intermediate = eq_preprocessed_dataset.parent / "intermediate"
    out_dir = tmp_path / "eval_tasks"

    # Small but nontrivial: two codes, three durations, two prediction times per
    # subject → at most 2 * 2 * 3 = 12 rows per subject.
    codes = ["HR", "TEMP"]
    durations = [1, 7, 30]
    pt_per_subject = 2

    run_and_check(
        [
            "EQ_generate_evaluation_tasks",
            f"data_dir={intermediate!s}",
            f"codes_dir={eq_preprocessed_dataset!s}",
            f"out_dir={out_dir!s}",
            "split=tuning",
            "input_shard=0",
            f"prediction_times_per_subject={pt_per_subject}",
            "min_context_per_subject=1",
            f"codes=[{','.join(codes)}]",
            f"durations=[{','.join(str(d) for d in durations)}]",
            "seed=42",
        ],
        env=ENSURE_ENV_PLACEHOLDERS,
        timeout=120.0,
    )

    written = out_dir / "eval" / "tuning" / "0.parquet"
    assert written.is_file(), f"expected {written} to exist; got {list(out_dir.rglob('*.parquet'))}"

    # Schema conformance.
    TaskQuerySchema.align(pq.read_table(written))

    df = pl.read_parquet(written)
    assert df.height > 0, "dense grid should be non-empty for the tuning split of the demo cohort"

    # Dense-grid shape: for every sampled prediction_time, we emitted one row per
    # (code x duration) pair.  Count distinct (subject_id, prediction_time) pairs
    # in the output and confirm the grid expansion factor is exactly |codes| x |durations|.
    n_unique_contexts = (
        df.select(TaskQuerySchema.subject_id_name, TaskQuerySchema.prediction_time_name).unique().height
    )
    assert df.height == n_unique_contexts * len(codes) * len(durations), (
        f"expected dense grid height {n_unique_contexts * len(codes) * len(durations)} "
        f"({n_unique_contexts} contexts x {len(codes)} codes x {len(durations)} durations), "
        f"got {df.height}"
    )

    # At most ``prediction_times_per_subject`` prediction times per subject.
    per_subject_times = (
        df.select(TaskQuerySchema.subject_id_name, TaskQuerySchema.prediction_time_name)
        .unique()
        .group_by(TaskQuerySchema.subject_id_name)
        .len()
    )
    assert per_subject_times["len"].max() <= pt_per_subject, (
        f"per-subject prediction-time count exceeded cap: {per_subject_times['len'].to_list()}"
    )

    # Every (query, duration_days) cell covers the same set of (subject, time) pairs.
    cell_sizes = df.group_by(TaskQuerySchema.query_name, TaskQuerySchema.duration_days_name).len()
    assert cell_sizes["len"].n_unique() == 1, f"cell sizes should all be equal (dense grid); got {cell_sizes}"


def test_eq_generate_evaluation_tasks_deterministic(
    eq_preprocessed_dataset: Path,
    tmp_path: Path,
) -> None:
    """Two runs with the same seed produce equivalent parquet contents (row-for-row)."""
    intermediate = eq_preprocessed_dataset.parent / "intermediate"
    common_args: list[str] = [
        f"data_dir={intermediate!s}",
        f"codes_dir={eq_preprocessed_dataset!s}",
        "split=tuning",
        "input_shard=0",
        "prediction_times_per_subject=3",
        "min_context_per_subject=1",
        "codes=[HR,TEMP]",
        "durations=[1,7,30]",
        "seed=7",
    ]

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    for out in (out_a, out_b):
        run_and_check(
            ["EQ_generate_evaluation_tasks", f"out_dir={out!s}", *common_args],
            env=ENSURE_ENV_PLACEHOLDERS,
            timeout=120.0,
        )

    df_a = pl.read_parquet(out_a / "eval" / "tuning" / "0.parquet").sort(
        [
            TaskQuerySchema.subject_id_name,
            TaskQuerySchema.prediction_time_name,
            TaskQuerySchema.query_name,
            TaskQuerySchema.duration_days_name,
        ]
    )
    df_b = pl.read_parquet(out_b / "eval" / "tuning" / "0.parquet").sort(
        [
            TaskQuerySchema.subject_id_name,
            TaskQuerySchema.prediction_time_name,
            TaskQuerySchema.query_name,
            TaskQuerySchema.duration_days_name,
        ]
    )
    assert df_a.equals(df_b), (
        "EQ_generate_evaluation_tasks should be deterministic in (seed, split, input_shard)"
    )


def test_eq_generate_evaluation_tasks_subject_subsample_deterministic(
    eq_preprocessed_dataset: Path,
    tmp_path: Path,
) -> None:
    """Two CLI runs with the same seed and ``subject_subsample_fraction`` produce identical outputs.

    Guards against regressions in the per-subject hash-threshold sampler — both
    its determinism (cross-process xxhash3 stability) and the threading of
    ``subject_subsample_fraction`` through ``main`` -> ``run_worker`` ->
    ``subsample_subject_ids``.
    """
    intermediate = eq_preprocessed_dataset.parent / "intermediate"
    common_args: list[str] = [
        f"data_dir={intermediate!s}",
        f"codes_dir={eq_preprocessed_dataset!s}",
        "split=tuning",
        "input_shard=0",
        "prediction_times_per_subject=3",
        "min_context_per_subject=1",
        "codes=[HR,TEMP]",
        "durations=[1,7,30]",
        "seed=7",
        "subject_subsample_fraction=0.5",
    ]

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    for out in (out_a, out_b):
        run_and_check(
            ["EQ_generate_evaluation_tasks", f"out_dir={out!s}", *common_args],
            env=ENSURE_ENV_PLACEHOLDERS,
            timeout=120.0,
        )

    sort_keys = [
        TaskQuerySchema.subject_id_name,
        TaskQuerySchema.prediction_time_name,
        TaskQuerySchema.query_name,
        TaskQuerySchema.duration_days_name,
    ]
    df_a = pl.read_parquet(out_a / "eval" / "tuning" / "0.parquet").sort(sort_keys)
    df_b = pl.read_parquet(out_b / "eval" / "tuning" / "0.parquet").sort(sort_keys)
    assert df_a.equals(df_b), (
        "EQ_generate_evaluation_tasks should be deterministic with subject_subsample_fraction set"
    )


# ---------------------------------------------------------------------------
# Unit tests for subsample_subject_ids
# ---------------------------------------------------------------------------


def _events(subject_ids: list[int]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            DataSchema.subject_id_name: subject_ids,
            "time": list(range(len(subject_ids))),
        }
    )


def test_subsample_subject_ids_pinned_selection() -> None:
    """Pinned subject IDs for ``fraction=0.1, seed=42`` over subjects 0..199.

    Locks down the polars ``hash()`` behavior — if a polars upgrade silently
    changes default hash seeds or the algorithm, this test fails loudly rather
    than letting reproducibility regress in production.
    """
    df = _events(list(range(200)))
    out = subsample_subject_ids(df, 0.1, seed=42)
    kept = sorted(out[DataSchema.subject_id_name].unique().to_list())
    assert kept == [
        13,
        19,
        20,
        23,
        29,
        43,
        48,
        60,
        63,
        70,
        87,
        93,
        95,
        110,
        133,
        140,
        153,
        172,
        185,
        188,
        193,
        199,
    ]


def test_subsample_subject_ids_seed_changes_selection() -> None:
    df = _events(list(range(500)))
    a = set(subsample_subject_ids(df, 0.2, seed=1)[DataSchema.subject_id_name].to_list())
    b = set(subsample_subject_ids(df, 0.2, seed=2)[DataSchema.subject_id_name].to_list())
    assert a != b, "different seeds should produce different subject sets"


def test_subsample_subject_ids_short_circuits() -> None:
    df = _events(list(range(50)))
    assert subsample_subject_ids(df, None, seed=0).equals(df)
    assert subsample_subject_ids(df, 1.0, seed=0).equals(df)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5, 2.0, float("inf"), float("nan")])
def test_subsample_subject_ids_rejects_invalid_fraction(bad: float) -> None:
    df = _events([1, 2, 3])
    with pytest.raises(ValueError, match="subject_subsample_fraction"):
        subsample_subject_ids(df, bad, seed=0)


@pytest.mark.parametrize("bad", [True, False, "0.5", object()])
def test_subsample_subject_ids_rejects_non_numeric_fraction(bad: object) -> None:
    """Booleans and other non-numeric types must fail loudly, not coerce to 0/1.

    ``True`` would otherwise become ``1.0`` and silently disable subsampling.
    """
    df = _events([1, 2, 3])
    with pytest.raises(TypeError, match="subject_subsample_fraction"):
        subsample_subject_ids(df, bad, seed=0)  # type: ignore[arg-type]


def test_subsample_subject_ids_handles_empty_frame() -> None:
    df = _events([])
    out = subsample_subject_ids(df, 0.1, seed=0)
    assert out.height == 0
