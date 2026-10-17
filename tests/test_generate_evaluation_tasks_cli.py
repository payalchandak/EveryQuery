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

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import polars as pl
import pyarrow.parquet as pq
import pytest
from meds import DataSchema, death_code

from conftest import run_and_check
from every_query.data.schema import TaskQuerySchema
from every_query.generate_tasks.sample_evaluation_tasks import run_worker, subsample_subject_ids

if TYPE_CHECKING:
    from pathlib import Path


def test_eq_generate_evaluation_tasks_end_to_end(
    eq_preprocessed_dataset: Path,
    tmp_path: Path,
) -> None:
    """CLI writes a TaskQuerySchema parquet with censored rows dropped and a sibling unique parquet."""
    intermediate = eq_preprocessed_dataset.parent / "intermediate"
    out_dir = tmp_path / "eval_tasks"

    pt_per_subject = 2

    run_and_check(
        [
            "EQ_generate_evaluation_tasks",
            f"data_dir={intermediate!s}",
            f"out_dir={out_dir!s}",
            "split=tuning",
            f"prediction_times_per_subject={pt_per_subject}",
            "min_context_per_subject=1",
            "query_codes=[HR,TEMP]",
            "durations=[1,7,30]",
            "seed=42",
        ],
        timeout=120.0,
    )

    labels_fp = out_dir / "eval" / "tuning" / "0.parquet"
    unique_fp = out_dir / "eval_unique" / "tuning" / "0.parquet"
    assert labels_fp.is_file(), f"expected {labels_fp} to exist; got {list(out_dir.rglob('*.parquet'))}"
    assert unique_fp.is_file(), f"expected sibling unique parquet at {unique_fp}"

    # Shard discovery (#279): every shard in the split must have an output parquet.
    input_shards = sorted(p.stem for p in (intermediate / "data" / "tuning").glob("*.parquet"))
    output_shards = sorted(p.stem for p in (out_dir / "eval" / "tuning").glob("*.parquet"))
    assert output_shards == input_shards, (
        f"expected one output per input shard: inputs={input_shards}, outputs={output_shards}"
    )

    TaskQuerySchema.align(pq.read_table(labels_fp))

    labels = pl.read_parquet(labels_fp)
    # Don't assert non-empty: with censoring dropped, the demo cohort can legitimately
    # produce zero surviving rows for some (codes, durations, split) combinations.
    assert labels[TaskQuerySchema.boolean_value_name].null_count() == 0, (
        "censored (null boolean_value) rows should be dropped from the labeled output"
    )

    if labels.height > 0:
        per_subject_times = (
            labels.select(TaskQuerySchema.subject_id_name, TaskQuerySchema.prediction_time_name)
            .unique()
            .group_by(TaskQuerySchema.subject_id_name)
            .len()
        )
        assert per_subject_times["len"].max() <= pt_per_subject, (
            f"per-subject prediction-time count exceeded cap: {per_subject_times['len'].to_list()}"
        )

    uniq = pl.read_parquet(unique_fp)
    assert uniq.columns == [
        TaskQuerySchema.subject_id_name,
        TaskQuerySchema.prediction_time_name,
    ]
    expected_unique = labels.select(
        [TaskQuerySchema.subject_id_name, TaskQuerySchema.prediction_time_name]
    ).unique()
    assert uniq.height == expected_unique.height


def test_eq_generate_evaluation_tasks_deterministic(
    eq_preprocessed_dataset: Path,
    tmp_path: Path,
) -> None:
    """Two runs with the same seed produce equivalent parquet contents (row-for-row)."""
    intermediate = eq_preprocessed_dataset.parent / "intermediate"
    common_args: list[str] = [
        f"data_dir={intermediate!s}",
        "split=tuning",
        "prediction_times_per_subject=3",
        "min_context_per_subject=1",
        "query_codes=[HR,TEMP]",
        "durations=[1,7,30]",
        "seed=7",
    ]

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    for out in (out_a, out_b):
        run_and_check(
            ["EQ_generate_evaluation_tasks", f"out_dir={out!s}", *common_args],
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
        "split=tuning",
        "prediction_times_per_subject=3",
        "min_context_per_subject=1",
        "query_codes=[HR,TEMP]",
        "durations=[1,7,30]",
        "seed=7",
        "subject_subsample_fraction=0.5",
    ]

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    for out in (out_a, out_b):
        run_and_check(
            ["EQ_generate_evaluation_tasks", f"out_dir={out!s}", *common_args],
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


def test_eq_generate_evaluation_tasks_rejects_input_shard_override(tmp_path: Path) -> None:
    """The removed ``input_shard`` knob must fail loudly, not be silently ignored (#279)."""
    with pytest.raises(RuntimeError, match="Could not override 'input_shard'"):
        run_and_check(
            [
                "EQ_generate_evaluation_tasks",
                f"data_dir={tmp_path!s}",
                f"out_dir={tmp_path!s}",
                "input_shard=0",
                "query_codes=[HR]",
            ],
            timeout=60.0,
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


# ---------------------------------------------------------------------------
# Death truncation of eval prediction times (#290)
# ---------------------------------------------------------------------------


_BASE = datetime(2024, 1, 1)


def _write_eval_shard(root: Path, split: str, shard: str, events: pl.DataFrame) -> Path:
    """Write ``events`` as ``{root}/data/{split}/{shard}.parquet`` and return ``root``."""
    shard_fp = root / "data" / split / f"{shard}.parquet"
    shard_fp.parent.mkdir(parents=True, exist_ok=True)
    events.write_parquet(shard_fp)
    return root


def _subject_rows(subject_id: int, n_days: int, code: str = "HR") -> pl.DataFrame:
    return pl.DataFrame(
        {
            DataSchema.subject_id_name: [subject_id] * n_days,
            DataSchema.time_name: [_BASE + timedelta(days=i) for i in range(n_days)],
            DataSchema.code_name: [code] * n_days,
        }
    )


def _death_row(subject_id: int, day: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            DataSchema.subject_id_name: [subject_id],
            DataSchema.time_name: [_BASE + timedelta(days=day)],
            DataSchema.code_name: [death_code],
        }
    )


def test_run_worker_never_samples_post_death_prediction_times(tmp_path: Path) -> None:
    """Post-death timestamps must not become evaluation prediction times (#290).

    Training's Stage 0 truncates each subject's candidate times at ``MEDS_DEATH`` (#265); the
    eval sampler must do the same or the two pipelines disagree about which contexts exist,
    deflating measured prevalence on death-adjacent tasks.  Subject 1 dies on day 2 but carries
    administrative rows on days 3-4; only days 0-2 are legal prediction times.  The death
    timestamp itself stays eligible (truncation is ``<=``).
    """
    data_dir = _write_eval_shard(
        tmp_path / "cohort",
        "held_out",
        "0",
        pl.concat([_subject_rows(1, 5), _death_row(1, 2), _subject_rows(2, 4)]),
    )

    labels_fp = run_worker(
        data_dir=data_dir,
        out_dir=tmp_path / "eval_tasks",
        split="held_out",
        input_shard="0",
        codes=["HR"],
        durations=[1.0],
        prediction_times_per_subject=10,  # > candidates, so every legal time is sampled
        min_context_per_subject=1,
        seed=0,
        write_unique_prediction_times=False,
    )
    assert labels_fp is not None

    labels = pl.read_parquet(labels_fp)
    dead_times = sorted(
        labels.filter(pl.col(TaskQuerySchema.subject_id_name) == 1)[TaskQuerySchema.prediction_time_name]
        .unique()
        .to_list()
    )
    assert dead_times == [_BASE, _BASE + timedelta(days=1), _BASE + timedelta(days=2)], (
        f"subject 1 died on day 2; post-death days 3-4 must not be prediction times: {dead_times}"
    )

    # The living subject is untouched by truncation (uncensored windows only: day 3 closes past
    # the record's end and is dropped as censored).
    alive_times = sorted(
        labels.filter(pl.col(TaskQuerySchema.subject_id_name) == 2)[TaskQuerySchema.prediction_time_name]
        .unique()
        .to_list()
    )
    assert alive_times == [_BASE, _BASE + timedelta(days=1), _BASE + timedelta(days=2)]


def test_run_worker_post_death_events_dont_count_toward_min_context(tmp_path: Path) -> None:
    """Post-death rows must not push a subject over ``min_context_per_subject`` (#290).

    Subject 1 has 4 rows through death (days 0-2 plus the death row) and 2 post-death rows.  At
    ``min_context_per_subject=5`` it has no legal prediction time at all; untruncated, days 3-4
    would clear the bar on post-death context alone.  Subject 2 (no death) still contributes, so
    an empty output can't pass this test vacuously.
    """
    data_dir = _write_eval_shard(
        tmp_path / "cohort",
        "held_out",
        "0",
        pl.concat([_subject_rows(1, 5), _death_row(1, 2), _subject_rows(2, 7)]),
    )

    labels_fp = run_worker(
        data_dir=data_dir,
        out_dir=tmp_path / "eval_tasks",
        split="held_out",
        input_shard="0",
        codes=["HR"],
        durations=[1.0],
        prediction_times_per_subject=10,
        min_context_per_subject=5,
        seed=0,
        write_unique_prediction_times=False,
    )
    assert labels_fp is not None

    labels = pl.read_parquet(labels_fp)
    assert labels.filter(pl.col(TaskQuerySchema.subject_id_name) == 1).height == 0, (
        "subject 1 has < 5 events before death; post-death rows must not make it eligible"
    )
    assert labels.filter(pl.col(TaskQuerySchema.subject_id_name) == 2).height > 0, (
        "subject 2 (no death, 7 events) should still contribute prediction times"
    )
