"""Explicit window starts on ``QuerySeqSchema`` rows (issue #27): schema, labeling, and the dataset guard.

The window rule, stated once (it is the multitask sampler's rule, reused through ``interval_table``)::

    start  = prediction_time + start_duration   OR  first start_event strictly after prediction_time
    end    = start + duration                   OR  first bound_event strictly after the RESOLVED start
    answer = start < some occurrence of the query < end          (both endpoints open)

with a start event that never occurs leaving the window empty (``False``, even if the end is also
unresolved) and an end event that never occurs after a resolved start letting the window run to the
end of the record.  Every labeling test below is checked against a plain-Python oracle written from
that sentence and sharing no code with the labeler.

The dataset half pins the safety property: the ordinary sequence models cannot represent a window
that opens away from the prediction time, so ``QuerySeqPytorchDataset`` must refuse such a
grid unless the caller opts in (``allow_active_starts=True``, the multitask prediction adapter),
while absent and all-default start columns keep loading as before.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pyarrow as pa
import pytest
import torch
from meds import train_split
from meds_torchdata.config import MEDSTorchDataConfig

from conftest import _PRED_TIMES, _TRAIN_SUBJECTS
from every_query.data.query_seq_dataset import (
    EVENT_BOUND_DURATION_SENTINEL,
    NO_BOUND_INDEX,
    QuerySeqBatch,
    QuerySeqPytorchDataset,
)
from every_query.data.schema import QuerySeqSchema
from every_query.generate_tasks import query_sequence_labeling as sqs
from every_query.generate_tasks import sample_multitask_sequences as sms
from every_query.generate_tasks.query_sequence_labeling import (
    BOUND_COL,
    START_DURATION_COL,
    START_EVENT_COL,
    label_binary_occurrence,
    label_query_sequences,
    label_with_event_bounds,
    label_with_explicit_starts,
)

if TYPE_CHECKING:
    from pathlib import Path

PT = datetime(2024, 1, 1)
SENTINEL = EVENT_BOUND_DURATION_SENTINEL


def _day(n: int) -> datetime:
    return PT + timedelta(days=n)


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------


def _events(rows: list[tuple[int, datetime, str]]) -> pl.DataFrame:
    return pl.DataFrame(
        {"subject_id": [r[0] for r in rows], "time": [r[1] for r in rows], "code": [r[2] for r in rows]},
        schema={"subject_id": pl.Int64, "time": pl.Datetime("us"), "code": pl.Utf8},
    )


def _index(
    specs: list[tuple[str, float, str | None, float, str | None]],
    *,
    subject: int = 1,
    pt: datetime = PT,
    with_bounds: bool = True,
    with_starts: bool = True,
    ctx_id: int = 0,
) -> pl.DataFrame:
    """One sequence; each spec is ``(query, duration_days, bound_event, start_duration_days, start_event)``.

    Every position shares the row's ``subject`` and ``pt``.
    """
    n = len(specs)
    data = {
        "_ctx_id": [ctx_id] * n,
        "_position": list(range(n)),
        "subject_id": [subject] * n,
        "prediction_time": [pt] * n,
        "query": [s[0] for s in specs],
        "duration_days": [float(s[1]) for s in specs],
    }
    schema = {
        "_ctx_id": pl.UInt32,
        "_position": pl.Int64,
        "subject_id": pl.Int64,
        "prediction_time": pl.Datetime("us"),
        "query": pl.Utf8,
        "duration_days": pl.Float32,
    }
    if with_bounds:
        data[BOUND_COL] = [s[2] for s in specs]
        schema[BOUND_COL] = pl.Utf8
    if with_starts:
        data[START_DURATION_COL] = [float(s[3]) for s in specs]
        data[START_EVENT_COL] = [s[4] for s in specs]
        schema[START_DURATION_COL] = pl.Float32
        schema[START_EVENT_COL] = pl.Utf8
    return pl.DataFrame(data, schema=schema)


def _oracle(
    events: list[tuple[int, datetime, str]],
    subject: int,
    pt: datetime,
    spec: tuple[str, float, str | None, float, str | None],
) -> bool:
    """The written rule, in plain Python, sharing nothing with the labeler."""
    query, duration, bound, start_duration, start_event = spec
    subj = [(t, c) for s, t, c in events if s == subject]
    if start_event is None:
        start = pt + timedelta(days=start_duration)
    else:
        hits = [t for t, c in subj if c == start_event and t > pt]
        if not hits:
            return False  # empty window, whatever the end says
        start = min(hits)
    if bound is None:
        end = start + timedelta(days=duration)
    else:
        hits = [t for t, c in subj if c == bound and t > start]
        end = min(hits) if hits else None  # None: runs to the end of the record
    return any(c == query and start < t and (end is None or t < end) for t, c in subj)


def _answers(index_df: pl.DataFrame, events_df: pl.DataFrame) -> list[bool]:
    return (
        label_query_sequences(index_df, events_df)
        .sort("subject_id", "prediction_time")["answers"]
        .explode()
        .to_list()
    )


# The one hand-built record every semantic test reads.  Day numbers are relative to ``PT``.
RECORD: list[tuple[int, datetime, str]] = [
    (1, _day(1), "DISCHARGE"),  # a discharge BEFORE the admission: closes nothing that opens later
    (1, _day(2), "A"),
    (1, _day(5), "ADMIT"),
    (1, _day(8), "A"),
    (1, _day(11), "DISCHARGE"),
    (1, _day(11), "C"),  # charted at the discharge instant
    (1, _day(19), "B"),
    (1, _day(29), "TIMELINE//END"),
]


# ---------------------------------------------------------------------------
# 1. The semantics, case by case, against the oracle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "expected", "why"),
    [
        (("A", 30.0, None, 0.0, None), True, "prediction-time start: A on day 2 is inside (0, 30)"),
        (("A", 2.0, None, 7.0, None), True, "duration start day 7, end day 9: A on day 8 is inside"),
        (("A", 1.0, None, 7.0, None), False, "duration start day 7, end day 8: A on day 8 is AT the end"),
        (("B", 30.0, None, 19.0, None), False, "duration start day 19: B on day 19 is AT the start"),
        (("B", 30.0, None, 18.0, None), True, "duration start day 18: B on day 19 is one day inside"),
        (("A", 30.0, None, SENTINEL, "ADMIT"), True, "event start at the admission (day 5): A on day 8"),
        (
            ("A", 3.0, None, SENTINEL, "ADMIT"),
            False,
            "event start day 5, end day 8: A on day 8 is AT the end",
        ),
        (("A", SENTINEL, "DISCHARGE", SENTINEL, "ADMIT"), True, "admission -> discharge: A on day 8"),
        (("C", SENTINEL, "DISCHARGE", SENTINEL, "ADMIT"), False, "C shares the discharge instant: excluded"),
        (("A", 30.0, None, SENTINEL, "NEVER"), False, "start event never occurs: empty window"),
        (
            ("A", SENTINEL, "NEVER_END", SENTINEL, "NEVER"),
            False,
            "missing start AND missing end: still empty",
        ),
        (
            ("B", SENTINEL, "NEVER_END", SENTINEL, "ADMIT"),
            True,
            "missing end after a resolved start: to end of record",
        ),
        (
            ("A", SENTINEL, "NEVER_END", 0.0, None),
            True,
            "missing end after a prediction-time start: legacy rule",
        ),
        (
            ("A", SENTINEL, "DISCHARGE", 0.0, None),
            False,
            "prediction-time start closes at the day-1 discharge",
        ),
    ],
)
def test_each_start_form_against_the_hand_computed_oracle(spec, expected, why):
    assert _oracle(RECORD, 1, PT, spec) == expected, (
        f"oracle disagrees with the case's own expectation: {why}"
    )
    assert _answers(_index([spec]), _events(RECORD)) == [expected], why


def test_end_boundary_is_searched_after_the_resolved_start_not_the_prediction_time():
    """The discharge on day 1 lies between the prediction time and the admission on day 5.

    A labeler that resolved the end against the prediction time would close the window at day 1 and answer
    False; the rule resolves it against the resolved start (day 5), finds the day-11 discharge, and sees A on
    day 8.
    """
    spec = ("A", SENTINEL, "DISCHARGE", SENTINEL, "ADMIT")
    assert _answers(_index([spec]), _events(RECORD)) == [True]
    # The same pair with a prediction-time start really does close at day 1: the two answers differ
    # only because of where the end search begins.
    assert _answers(_index([("A", SENTINEL, "DISCHARGE", 0.0, None)]), _events(RECORD)) == [False]


def test_both_endpoints_are_open_for_every_start_form():
    """An occurrence exactly at the start instant or the end instant is outside the window."""
    tick = timedelta(microseconds=1)
    events = _events(
        [
            (1, _day(5), "ADMIT"),
            (1, _day(5), "AT_START"),
            (1, _day(5) + tick, "JUST_AFTER_START"),
            (1, _day(15) - tick, "JUST_BEFORE_END"),
            (1, _day(15), "AT_END"),
            (1, _day(15), "CLOSE"),
        ]
    )
    for start_duration, start_event in [(5.0, None), (SENTINEL, "ADMIT")]:
        for duration, bound in [(10.0, None), (SENTINEL, "CLOSE")]:
            specs = [
                (q, duration, bound, start_duration, start_event)
                for q in ("AT_START", "JUST_AFTER_START", "JUST_BEFORE_END", "AT_END")
            ]
            assert _answers(_index(specs), events) == [False, True, True, False], (
                start_duration,
                start_event,
                duration,
                bound,
            )


def test_unresolved_start_never_becomes_an_infinite_window():
    """With no start the window is empty even though the end is +inf too; nothing at all counts."""
    events = _events([(1, _day(3), "A"), (1, _day(9), "B"), (1, _day(40), "C")])
    specs = [(q, SENTINEL, "NEVER_END", SENTINEL, "NEVER") for q in ("A", "B", "C")]
    assert _answers(_index(specs), events) == [False, False, False]
    specs = [(q, 1000.0, None, SENTINEL, "NEVER") for q in ("A", "B", "C")]
    assert _answers(_index(specs), events) == [False, False, False]


def test_randomised_differential_against_the_oracle():
    """Random day-granular records x random specs over all three start forms and both end forms."""
    rng = np.random.default_rng(27)
    codes = ["A", "B", "C", "ADMIT", "DISCHARGE"]
    for trial in range(40):
        subjects = [1, 2, 3]
        events = [
            (s, _day(int(d)), codes[int(rng.integers(0, len(codes)))])
            for s in subjects
            for d in rng.integers(-5, 60, size=int(rng.integers(3, 25)))
        ]
        frames, expected = [], []
        for ctx, s in enumerate(subjects):
            specs = []
            for _ in range(int(rng.integers(1, 5))):
                form = rng.integers(0, 3)
                if form == 0:
                    start = (0.0, None)
                elif form == 1:
                    start = (float(rng.integers(1, 20)), None)
                else:
                    start = (SENTINEL, codes[int(rng.integers(0, len(codes)))])
                if rng.random() < 0.5:
                    end = (float(rng.integers(1, 30)), None)
                else:
                    end = (SENTINEL, codes[int(rng.integers(0, len(codes)))])
                specs.append((codes[int(rng.integers(0, 3))], end[0], end[1], start[0], start[1]))
            pt = _day(int(rng.integers(0, 10)))
            frames.append(_index(specs, subject=s, pt=pt, ctx_id=ctx))
            expected.append([_oracle(events, s, pt, sp) for sp in specs])
        labeled = label_query_sequences(pl.concat(frames), _events(events))
        got = [row["answers"] for row in labeled.sort("subject_id").iter_rows(named=True)]
        assert got == expected, f"trial {trial}"


# ---------------------------------------------------------------------------
# 2. Compatibility with the legacy labelers
# ---------------------------------------------------------------------------


def _legacy_pair(with_bounds: bool) -> tuple[pl.DataFrame, pl.DataFrame]:
    """A mixed multi-sequence frame with and without the explicit (all-default) start columns."""
    rng = np.random.default_rng(5)
    codes = ["A", "B", "C", "D"]
    events = [
        (s, _day(int(d)), codes[int(rng.integers(0, 4))]) for s in (1, 2) for d in rng.integers(-3, 50, 30)
    ]
    frames_legacy, frames_started = [], []
    for ctx in range(6):
        specs = []
        for _ in range(int(rng.integers(1, 4))):
            if with_bounds and rng.random() < 0.5:
                specs.append(
                    (codes[int(rng.integers(0, 4))], SENTINEL, codes[int(rng.integers(0, 4))], 0.0, None)
                )
            else:
                specs.append((codes[int(rng.integers(0, 4))], float(rng.integers(1, 40)), None, 0.0, None))
        subject = 1 + ctx % 2
        pt = _day(int(rng.integers(0, 10)))
        frames_legacy.append(
            _index(specs, subject=subject, pt=pt, with_bounds=with_bounds, with_starts=False, ctx_id=ctx)
        )
        frames_started.append(
            _index(specs, subject=subject, pt=pt, with_bounds=with_bounds, with_starts=True, ctx_id=ctx)
        )
    return pl.concat(frames_legacy), pl.concat(frames_started), _events(events)


@pytest.mark.parametrize("with_bounds", [False, True])
def test_explicit_default_starts_label_identically_to_the_legacy_path(with_bounds):
    """``start_duration=0.0`` / ``start_event=null`` spelled out gives the legacy answers exactly."""
    legacy_df, started_df, events = _legacy_pair(with_bounds)
    legacy = (label_with_event_bounds if with_bounds else label_binary_occurrence)(legacy_df, events)
    started = label_with_explicit_starts(started_df, events)
    key = ["subject_id", "prediction_time", "queries"]
    a = legacy.sort(key).select(key, "durations", "answers")
    b = started.sort(key).select(key, "durations", "answers")
    assert a.equals(b)
    if with_bounds:
        assert legacy.sort(key)["bound_events"].to_list() == started.sort(key)["bound_events"].to_list()
    # and the seam routes both the same way
    assert (
        label_query_sequences(legacy_df, events)
        .sort(key)
        .select(key, "answers")
        .equals(label_query_sequences(started_df, events).sort(key).select(key, "answers"))
    )


def test_dispatch_keys_on_the_frame_and_requires_both_start_columns():
    legacy_df, started_df, events = _legacy_pair(True)
    assert "start_events" not in label_query_sequences(legacy_df, events).columns
    out = label_query_sequences(started_df, events)
    assert {"start_durations", "start_events", "bound_events"} <= set(out.columns)
    with pytest.raises(ValueError, match="must be present together"):
        label_query_sequences(started_df.drop(START_EVENT_COL), events)
    with pytest.raises(ValueError, match="must be present together"):
        label_query_sequences(started_df.drop(START_DURATION_COL), events)


def test_labeler_rejects_contradictory_start_rows_at_the_seam():
    """The representation contract is enforced where the answers are produced, not only upstream:

    a negative / non-finite non-sentinel start or an event start without the sentinel would otherwise resolve
    to an arbitrary instant and label silently.
    """
    events = _events([(1, _day(5), "A"), (1, _day(30), "B")])
    for bad_duration, bad_event in [(-1.0, None), (-3.0, None), (float("nan"), None), (float("inf"), None)]:
        idx = _index([("A", 30.0, None, bad_duration, bad_event)])
        with pytest.raises(ValueError, match="disagree on which queries are event-defined"):
            label_query_sequences(idx, events)
    idx = _index([("A", 30.0, None, 7.0, "B")])
    with pytest.raises(ValueError, match="disagree on which queries are event-defined"):
        label_query_sequences(idx, events)
    # A null start duration is not a legal duration start either.
    idx = _index([("A", 30.0, None, 0.0, None)]).with_columns(
        pl.lit(None).cast(pl.Float32).alias(START_DURATION_COL)
    )
    with pytest.raises(ValueError, match="disagree on which queries are event-defined"):
        label_query_sequences(idx, events)


@pytest.mark.parametrize("unit", ["ms", "ns"])
def test_labeler_normalises_datetime_units_before_the_interval_table(unit):
    """``interval_table`` works in int64 microseconds; a ``ms``/``ns`` frame must label exactly as the ``us``
    frame does rather than being read as the wrong instants."""
    events = _events([(1, _day(3), "A"), (1, _day(6), "ADMIT"), (1, _day(9), "A"), (1, _day(20), "DIS")])
    idx = _index(
        [("A", 30.0, None, 0.0, None), ("A", -1.0, "DIS", SENTINEL, "ADMIT"), ("A", 2.0, None, 7.0, None)]
    )
    reference = label_query_sequences(idx, events)["answers"].to_list()
    assert reference == [[True, True, False]]  # A at day 9 sits ON the (pt+7d, pt+9d) end: excluded
    events_u = events.with_columns(pl.col("time").cast(pl.Datetime(unit)))
    idx_u = idx.with_columns(pl.col("prediction_time").cast(pl.Datetime(unit)))
    assert label_query_sequences(idx_u, events_u)["answers"].to_list() == reference
    assert label_query_sequences(idx, events_u)["answers"].to_list() == reference
    assert label_query_sequences(idx_u, events)["answers"].to_list() == reference


def test_start_columns_come_back_as_list_columns_aligned_with_queries():
    specs = [
        ("A", 30.0, None, 7.0, None),
        ("B", SENTINEL, "DISCHARGE", SENTINEL, "ADMIT"),
        ("C", 1.0, None, 0.0, None),
    ]
    row = label_with_explicit_starts(_index(specs), _events(RECORD)).row(0, named=True)
    assert row["queries"] == ["A", "B", "C"]
    assert row["start_durations"] == [7.0, SENTINEL, 0.0]
    assert row["start_events"] == [None, "ADMIT", None]
    assert row["bound_events"] == [None, "DISCHARGE", None]
    # A bound-free index keeps a bound-free output.
    row = label_with_explicit_starts(_index(specs[:1], with_bounds=False), _events(RECORD)).row(0, named=True)
    assert "bound_events" not in row and row["start_durations"] == [7.0]


def test_empty_index_labels_to_an_empty_frame():
    out = label_with_explicit_starts(_index([]).head(0), _events(RECORD))
    assert out.height == 0


def test_window_semantics_constants_agree_with_the_multitask_sampler():
    """The scalar labeler states the same contract the multitask sampler's manifest records."""
    for name in (
        "WINDOW_SEMANTICS",
        "START_REFERENCE",
        "DURATION_END_REFERENCE",
        "MISSING_EVENT_START",
        "MISSING_EVENT_BOUNDARY",
    ):
        assert getattr(sqs, name) == getattr(sms, name), name


def test_scalar_labeler_matches_the_multitask_oracle_bit():
    """Differential against ``interval_table.naive_window_labels``, the multitask sampler's own oracle."""
    from every_query.generate_tasks.interval_table import US_PER_DAY, naive_window_labels

    code_index = {"A": 0, "B": 1, "ADMIT": 2, "DISCHARGE": 3, "C": 4}
    ev = [
        (s, int((t - datetime(1970, 1, 1)).total_seconds() * 1e6), code_index[c])
        for s, t, c in RECORD
        if c in code_index
    ]
    pt_us = int((PT - datetime(1970, 1, 1)).total_seconds() * 1e6)
    windows = [
        ((0.0, None), (30.0, None)),
        ((7.0, None), (2.0, None)),
        ((None, code_index["ADMIT"]), (None, code_index["DISCHARGE"])),
        ((None, code_index["ADMIT"]), (None, 99)),  # end never occurs
        ((None, 98), (30.0, None)),  # start never occurs
    ]
    _, _, targets = naive_window_labels(ev, [(1, pt_us)] * len(windows), [[w] for w in windows], 100, 1)
    assert US_PER_DAY == 86_400_000_000
    specs = []
    for (sd, sc), (ed, ec) in windows:
        inv = {v: k for k, v in code_index.items()}
        start = (SENTINEL, inv.get(sc, "NEVER")) if sc is not None else (float(sd), None)
        end = (SENTINEL, inv.get(ec, "NEVER_END")) if ec is not None else (float(ed), None)
        specs.append(("A", end[0], end[1], start[0], start[1]))
    got = _answers(_index(specs), _events(RECORD))
    assert got == [bool(targets[i, 0, code_index["A"]]) for i in range(len(windows))]


# ---------------------------------------------------------------------------
# 3. Schema
# ---------------------------------------------------------------------------


def _table(**extra) -> pa.Table:
    base = {
        "subject_id": 1,
        "prediction_time": datetime(2023, 1, 1),
        "queries": ["A", "B"],
        "durations": [30.0, 7.0],
        "answers": [False, True],
    }
    base.update(extra)
    return pa.Table.from_pylist([base])


def test_schema_accepts_old_rows_without_start_columns_and_new_rows_with_them():
    assert [f.name for f in QuerySeqSchema.align(_table()).schema] == [
        "subject_id",
        "prediction_time",
        "queries",
        "durations",
        "answers",
    ]
    aligned = QuerySeqSchema.align(_table(start_durations=[7.0, -1.0], start_events=[None, "ADMIT"]))
    assert aligned.column("start_durations").type == pa.large_list(pa.float32())
    assert aligned.column("start_events").to_pylist() == [[None, "ADMIT"]]


# ---------------------------------------------------------------------------
# 4. The dataset guard
# ---------------------------------------------------------------------------

_CODES = ["HR", "TEMP"]  # real codes of the fixture cohort


def _write_labels(root: Path, rows: list[dict], *, drop: tuple[str, ...] = ()) -> Path:
    """Write one train-split shard of ``QuerySeqSchema`` rows; ``rows`` may carry start columns."""
    split_dir = root / train_split
    split_dir.mkdir(parents=True, exist_ok=True)
    cols = {
        "subject_id": pl.Int64,
        "prediction_time": pl.Datetime("us"),
        "queries": pl.List(pl.Utf8),
        "durations": pl.List(pl.Float32),
        "answers": pl.List(pl.Boolean),
    }
    if "bound_events" in rows[0]:
        cols["bound_events"] = pl.List(pl.Utf8)
    if "start_durations" in rows[0]:
        cols["start_durations"] = pl.List(pl.Float32)
        cols["start_events"] = pl.List(pl.Utf8)
    df = pl.DataFrame(rows, schema=cols).drop(*drop)
    df.write_parquet(split_dir / "0.parquet")
    return root


def _rows(starts: list[tuple[list[float], list[str | None]]] | None) -> list[dict]:
    rows = []
    for i, subj in enumerate(_TRAIN_SUBJECTS):
        n = 3 if i % 2 == 0 else 2
        row = {
            "subject_id": subj,
            "prediction_time": _PRED_TIMES[subj],
            "queries": [_CODES[j % 2] for j in range(n)],
            "durations": [30.0, 7.0, 60.0][:n],
            "answers": [True, False, True][:n],
        }
        if starts is not None:
            row["start_durations"], row["start_events"] = starts[i % len(starts)]
            row["start_durations"] = row["start_durations"][:n]
            row["start_events"] = row["start_events"][:n]
        rows.append(row)
    return rows


def _dataset(cohort: Path, labels: Path, **kw) -> QuerySeqPytorchDataset:
    cfg = MEDSTorchDataConfig(
        tensorized_cohort_dir=str(cohort),
        task_labels_dir=str(labels),
        max_seq_len=64,
        seq_sampling_strategy="to_end",
        static_inclusion_mode="omit",
        batch_mode="SM",
    )
    return QuerySeqPytorchDataset(cfg, split=train_split, **kw)


DEFAULT_STARTS = [([0.0, 0.0, 0.0], [None, None, None])]
ACTIVE_STARTS = [([7.0, 0.0, 0.0], [None, None, None]), ([SENTINEL, 0.0, 0.0], ["HR", None, None])]


def test_old_parquets_without_start_columns_load(tensorized_cohort_dir, tmp_path):
    ds = _dataset(tensorized_cohort_dir, _write_labels(tmp_path / "labels", _rows(None)))
    assert not ds.has_starts
    batch = ds.collate([ds[i] for i in range(len(ds))])
    assert batch.q_start_durations is None and batch.q_start_codes is None


def test_all_default_start_columns_load_and_stay_out_of_the_batch(tensorized_cohort_dir, tmp_path):
    ds = _dataset(tensorized_cohort_dir, _write_labels(tmp_path / "labels", _rows(DEFAULT_STARTS)))
    assert ds.has_starts and not ds.allow_active_starts
    batch = ds.collate([ds[i] for i in range(len(ds))])
    assert batch.q_start_durations is None  # ordinary batches are unchanged


@pytest.mark.parametrize("active", ACTIVE_STARTS, ids=["duration_start", "event_start"])
def test_ordinary_dataset_rejects_active_starts(tensorized_cohort_dir, tmp_path, active):
    labels = _write_labels(tmp_path / "labels", _rows([active]))
    with pytest.raises(ValueError, match="active start"):
        _dataset(tensorized_cohort_dir, labels)


def test_opt_in_tensorizes_active_starts(tensorized_cohort_dir, tmp_path):
    ds = _dataset(
        tensorized_cohort_dir,
        _write_labels(tmp_path / "labels", _rows(ACTIVE_STARTS)),
        allow_active_starts=True,
    )
    batch = ds.collate([ds[i] for i in range(len(ds))])
    assert isinstance(batch, QuerySeqBatch)
    assert batch.q_start_durations.shape == batch.q_codes.shape == batch.q_start_codes.shape
    hr = ds.code_to_index["HR"]
    for i in range(len(ds)):
        n = int(batch.q_mask[i].sum())
        want_d, want_e = ACTIVE_STARTS[i % 2]
        assert batch.q_start_durations[i, :n].tolist() == pytest.approx(want_d[:n])
        assert batch.q_start_codes[i, :n].tolist() == [
            NO_BOUND_INDEX if e is None else hr for e in want_e[:n]
        ]
        assert (batch.q_start_durations[i, n:] == 0).all() and (
            batch.q_start_codes[i, n:] == NO_BOUND_INDEX
        ).all()


def test_opt_in_without_start_columns_yields_default_tensors(tensorized_cohort_dir, tmp_path):
    ds = _dataset(
        tensorized_cohort_dir, _write_labels(tmp_path / "labels", _rows(None)), allow_active_starts=True
    )
    batch = ds.collate([ds[i] for i in range(len(ds))])
    assert torch.equal(batch.q_start_durations, torch.zeros_like(batch.q_durations))
    assert torch.equal(batch.q_start_codes, torch.full_like(batch.q_codes, NO_BOUND_INDEX))


@pytest.mark.parametrize("drop", ["start_durations", "start_events"])
def test_one_start_column_without_the_other_is_rejected(tensorized_cohort_dir, tmp_path, drop):
    labels = _write_labels(tmp_path / "labels", _rows(DEFAULT_STARTS), drop=(drop,))
    with pytest.raises(ValueError, match="present together"):
        _dataset(tensorized_cohort_dir, labels)


def test_ragged_start_lists_are_rejected(tensorized_cohort_dir, tmp_path):
    rows = _rows(DEFAULT_STARTS)
    rows[0]["start_durations"] = rows[0]["start_durations"][:-1]
    rows[0]["start_events"] = rows[0]["start_events"][:-1]
    with pytest.raises(ValueError, match="list lengths disagree"):
        _dataset(tensorized_cohort_dir, _write_labels(tmp_path / "labels", rows))


@pytest.mark.parametrize(
    ("durations", "events", "match"),
    [
        ([SENTINEL, 0.0, 0.0], [None, None, None], "disagree"),  # sentinel without an event
        ([7.0, 0.0, 0.0], ["HR", None, None], "disagree"),  # event without the sentinel
        ([0.0, 0.0, 0.0], ["HR", None, None], "disagree"),  # event with a zero duration
        ([-3.0, 0.0, 0.0], [None, None, None], "disagree"),  # negative non-sentinel
        ([float("inf"), 0.0, 0.0], [None, None, None], "disagree"),
        ([float("nan"), 0.0, 0.0], [None, None, None], "NaN"),
        ([SENTINEL, 0.0, 0.0], ["NOPE", None, None], "not in this run's vocabulary"),
    ],
    ids=["sentinel_no_event", "event_no_sentinel", "event_zero", "negative", "inf", "nan", "unknown_code"],
)
def test_invalid_start_representations_are_rejected_even_under_opt_in(
    tensorized_cohort_dir, tmp_path, durations, events, match
):
    labels = _write_labels(tmp_path / "labels", _rows([(durations, events)]))
    with pytest.raises(ValueError, match=match):
        _dataset(tensorized_cohort_dir, labels, allow_active_starts=True)


def test_batch_requires_both_start_tensors_or_neither():
    kw = {
        "code": torch.tensor([[3, 4]]),
        "numeric_value": torch.zeros(1, 2),
        "numeric_value_mask": torch.zeros(1, 2, dtype=torch.bool),
        "time_delta_days": torch.zeros(1, 2),
        "q_codes": torch.tensor([[7, 8]]),
        "q_durations": torch.tensor([[30.0, 7.0]]),
        "q_answers": torch.tensor([[1, 0]]),
        "q_mask": torch.tensor([[True, True]]),
    }
    assert QuerySeqBatch(**kw).q_start_durations is None
    with pytest.raises(ValueError, match="given together"):
        QuerySeqBatch(**kw, q_start_durations=torch.zeros(1, 2))
    with pytest.raises(ValueError, match="q_start_codes"):
        QuerySeqBatch(
            **kw, q_start_durations=torch.zeros(1, 2), q_start_codes=torch.zeros(1, 3, dtype=torch.long)
        )
    ok = QuerySeqBatch(
        **kw, q_start_durations=torch.zeros(1, 2), q_start_codes=torch.zeros(1, 2, dtype=torch.long)
    )
    assert ok.q_start_codes.shape == (1, 2)
