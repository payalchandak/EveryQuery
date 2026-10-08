"""Canonical, cross-labeller contract for the query window's edges.

**THIS FILE IS WHERE THE RULE LIVES.**  If prose anywhere else in the repo contradicts what is
asserted here, this file wins and that prose is a bug.

The rule, stated once::

    answer = True   iff   prediction_time  <  event_time  <  window_close

- The **lower** bound is **STRICT / open**.  An event landing on the prediction instant itself is
  *outside* the window: it is not "in the future" relative to the prediction.
- The **upper** bound is **STRICT / open** too.  An event landing exactly on the instant the window
  closes is *outside* it.
- ``window_close`` is ``prediction_time + duration_days`` for a time-bounded query, and the
  timestamp of the first occurrence of the boundary code strictly after ``prediction_time`` for an
  event-bounded one.  Both ends follow the same rule; there is one window definition, not two.

The window is therefore symmetric -- neither endpoint instant belongs to it -- and an event-bounded
query reads as the plain English it is meant to be: *"Sepsis, strictly after the prediction time
and strictly before discharge."*

Why this file exists
--------------------
The repo has several independent labellers.  They silently disagreed about the horizon instant for
a long time -- ``sample_tasks.evaluate_index_df`` compared with ``<=`` while both labellers in
``query_sequence_labeling`` compared with ``<`` -- and nothing caught it, because each labeller's
tests only ever checked that labeller against expectations written by the same author.  The
divergence was found by reading, not by testing.  Consistent-looking, silently-wrong labels are the
failure mode; nothing crashes.

So the defence here is deliberately shaped against that: **one** table of boundary cases
(:data:`BOUNDARY_CASES`), built **once**, driving **every** window decider through **identical
event frames** (:func:`_events_for`).  No labeller gets its own fixture, so no labeller can be
handed subtly different inputs and quietly answer a different question.

Note that the direction of the rule has changed once already (2026-08-22: the upper bound was
briefly pinned CLOSED across every labeller, then reopened).  This file is indifferent to which
direction is chosen -- its job is that *all* the deciders move together, and it is the thing that
must be edited first when the direction changes again.

What is covered (the inventory)
-------------------------------
:data:`WINDOW_DECIDERS` names every function in ``src/`` that decides whether an event falls inside
a query window:

1. ``query_sequence_labeling.label_binary_occurrence`` -- Stage 4' plain occurrence.
2. ``query_sequence_labeling.label_with_event_bounds`` with a null ``bound_event`` (time-bounded).
3. ``query_sequence_labeling.label_with_event_bounds`` with a boundary event placed exactly on the
   horizon (event-bounded) -- so the event-bound path is measured against the *same* edges.
4. ``sample_tasks.evaluate_index_df`` -- Stage 4, the single-query pipeline.  It alone has a
   censoring notion; see :func:`_run_evaluate_index_df` for how the window is made observed, and
   note that the adapter *refuses* a censored answer rather than papering over it.
5. ``query_sequence_labeling.label_query_sequences`` -- the public seam that both the training
   shards and the dense evaluation grid actually call.
6. ``query_sequence_labeling.label_with_explicit_starts`` -- the issue-#27 interval-table path, in
   three configurations: a spelled-out prediction-time start (time-bounded and event-bounded, so it
   is measured on the identical interval as 1-3), and a *resolved* start placed on the prediction
   instant itself via a start event, so the open lower bound is checked at a resolved start too.
   Its ``end`` is resolved relative to the resolved start, which here *is* ``PREDICTION_TIME``.

**A labeller missing from that list is exactly how this drifted the first time.**  Anything new
that decides window membership belongs in it.

No site outside ``src/`` decides window membership any more.  Two once did -- the
``scripts/eval_occurs_uncensored.py`` and ``scripts/eval_macro_position.py`` research drivers,
which recomputed it inline rather than calling a labeller, and so could drift from this list
without any test noticing.  Both were deleted with the rest of the conditional-seq analysis
drivers.  The hazard is worth remembering rather than the files: a scorer that decides membership
for itself corrupts *reported numbers* rather than training labels, which is the harder failure to
see, so a new one belongs behind a labeller in this list -- not in ``scripts/``.

Where the event-bound rule is pinned
------------------------------------
An event-bounded query whose query code shares the boundary event's exact timestamp answers
``False``.  That is the consequence the repo owner may most want to revisit, because MEDS clusters
many codes onto a single timestamp, so it decides real rows rather than a measure-zero edge.  It is
pinned in exactly two places in this file, and nowhere else in it:

- the ``exactly_at_the_horizon_instant`` case of the ``label_with_event_bounds_event_bounded``
  decider, in the sweep and in the agreement matrix, and
- :func:`test_a_query_at_the_exact_instant_of_the_boundary_event_does_not_count`, which spells the
  scenario out in clinical terms.

Making the event bound inclusive again therefore means changing one comparison in
``label_with_event_bounds`` and flipping those two expectations -- deliberately, in the open.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import polars as pl
import pytest
from meds import DataSchema

from every_query.data.query_seq_dataset import EVENT_BOUND_DURATION_SENTINEL
from every_query.data.schema import TaskQuerySchema
from every_query.generate_tasks.query_sequence_labeling import (
    BOUND_COL,
    CTX_ID_COL,
    POSITION_COL,
    START_DURATION_COL,
    START_EVENT_COL,
    label_binary_occurrence,
    label_query_sequences,
    label_with_event_bounds,
    label_with_explicit_starts,
)
from every_query.generate_tasks.sample_tasks import evaluate_index_df

if TYPE_CHECKING:
    from collections.abc import Callable

# ---------------------------------------------------------------------------
# The one shared geometry.  Every labeller sees these exact instants.
# ---------------------------------------------------------------------------

SUBJECT = 1
PREDICTION_TIME = datetime(2024, 1, 1, 0, 0, 0)
DURATION_DAYS = 10.0
#: The instant the window closes.  Inside the window (closed upper bound).
WINDOW_CLOSE = PREDICTION_TIME + timedelta(days=DURATION_DAYS)
#: Microsecond-precision datetimes: this is the smallest representable step, so "one tick past the
#: horizon" is genuinely adjacent to it and the two cases cannot both pass under one operator.
TICK = timedelta(microseconds=1)

QUERY_CODE = "QUERY//CODE"
#: Boundary code for the event-bounded decider, planted at exactly ``WINDOW_CLOSE`` so the
#: event-bounded window and the time-bounded window are the *same* interval.
BOUND_CODE = "BOUND//EVENT"
#: A far-future event that exists only to make the record span the window, so
#: ``evaluate_index_df`` sees an observed (uncensored) window.  Inert for every other decider:
#: it never matches the query code and lands far outside the window.
TAIL_CODE = "OBSERVED//TAIL"
TAIL_TIME = PREDICTION_TIME + timedelta(days=365)
#: Start event for the explicit-start decider, planted one tick BEFORE the prediction instant so the
#: window resolved from it -- "first occurrence strictly after the *anchor*" -- opens exactly at
#: ``PREDICTION_TIME`` when the anchor is that earlier instant.  See ``_run_label_with_explicit_starts``.
START_CODE = "START//EVENT"
START_ANCHOR = PREDICTION_TIME - timedelta(microseconds=1)

_SID = DataSchema.subject_id_name
_TIME = DataSchema.time_name
_CODE = DataSchema.code_name

RULE = (
    "    answer = True  iff  prediction_time < event_time < window_close\n"
    "        lower bound STRICT (an event AT prediction_time is outside)\n"
    "        upper bound STRICT (an event AT window_close is outside)"
)


# ---------------------------------------------------------------------------
# The one shared table of boundary cases.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundaryCase:
    """One position of a query event relative to the window, and the answer the rule requires."""

    name: str
    #: When the query code occurs, or ``None`` for "the code never occurs".
    query_event_time: datetime | None
    expected: bool
    #: Human phrasing used in the failure message, e.g. "an event exactly at the horizon instant".
    description: str
    #: Why the rule gives that answer -- printed on failure so the reader is told the rule.
    why: str


BOUNDARY_CASES: tuple[BoundaryCase, ...] = (
    BoundaryCase(
        name="strictly_inside",
        query_event_time=PREDICTION_TIME + timedelta(days=5),
        expected=True,
        description="an event strictly inside the window",
        why="it is after the prediction instant and before the window closes; both bounds agree",
    ),
    BoundaryCase(
        name="one_microsecond_before_the_horizon",
        query_event_time=WINDOW_CLOSE - TICK,
        expected=True,
        description="an event one microsecond before the horizon",
        why=(
            "the upper bound is strict but not wider than one tick -- anything before it is "
            "inside.  Without this case a labeller that had lost its upper bound in the other "
            "direction (answering False for everything near the top edge) would pass"
        ),
    ),
    BoundaryCase(
        name="exactly_at_the_horizon_instant",
        query_event_time=WINDOW_CLOSE,
        expected=False,
        description="an event exactly at the horizon instant",
        why=(
            "the upper bound is STRICT -- an event landing exactly on the instant the window "
            "closes is OUTSIDE it.  A `<=` here instead of `<` is the exact defect this file "
            "exists to stop"
        ),
    ),
    BoundaryCase(
        name="one_microsecond_past_the_horizon",
        query_event_time=WINDOW_CLOSE + TICK,
        expected=False,
        description="an event one microsecond past the horizon",
        why="past the horizon under any reading of the bound; not a boundary case at all",
    ),
    BoundaryCase(
        name="exactly_at_the_prediction_instant",
        query_event_time=PREDICTION_TIME,
        expected=False,
        description="an event exactly at the prediction instant",
        why=(
            "the lower bound is STRICT -- an event simultaneous with the prediction is not in "
            "its future.  This bound has never moved; the upper one now matches it"
        ),
    ),
    BoundaryCase(
        name="one_microsecond_after_the_prediction_instant",
        query_event_time=PREDICTION_TIME + TICK,
        expected=True,
        description="an event one microsecond after the prediction instant",
        why="the lower bound is strict but not wider than one tick; anything after it is inside",
    ),
    BoundaryCase(
        name="no_event_at_all",
        query_event_time=None,
        expected=False,
        description="a query code that never occurs",
        why="no occurrence anywhere in the record is False, never null and never True",
    ),
)


def _events_for(case: BoundaryCase) -> pl.DataFrame:
    """The one event frame every decider is handed for ``case`` -- built once, shared verbatim.

    Carries four codes: the query code at the case's instant (absent for ``no_event_at_all``), the
    boundary code planted on ``WINDOW_CLOSE``, a start code planted on ``PREDICTION_TIME``, and a
    far-future tail event.  The latter three are inert for the time-bounded deciders (different
    codes; the start code sits *on* the open lower bound and the others far outside the window) but
    let the event-bounded, explicit-start and censoring-aware deciders be driven from this same
    frame instead of from a look-alike of their own.
    """
    times: list[datetime] = [TAIL_TIME, WINDOW_CLOSE, PREDICTION_TIME]
    codes: list[str] = [TAIL_CODE, BOUND_CODE, START_CODE]
    if case.query_event_time is not None:
        times.append(case.query_event_time)
        codes.append(QUERY_CODE)
    return (
        pl.DataFrame({_SID: [SUBJECT] * len(times), _TIME: times, _CODE: codes})
        .with_columns(pl.col(_TIME).cast(pl.Datetime("us")))
        .sort(_SID, _TIME)
    )


def _sequence_index_df(bound_event: str | None, with_bound_col: bool) -> pl.DataFrame:
    """A one-query, one-sequence index frame for the ``query_sequence_labeling`` labellers.

    An event-bounded row carries ``EVENT_BOUND_DURATION_SENTINEL`` instead of a horizon, exactly as
    ``QuerySequenceDistribution.sample_sequences`` emits it; a time-bounded row carries the real horizon.
    """
    duration = EVENT_BOUND_DURATION_SENTINEL if bound_event is not None else DURATION_DAYS
    index_df = pl.DataFrame(
        {
            CTX_ID_COL: [0],
            POSITION_COL: [0],
            TaskQuerySchema.subject_id_name: [SUBJECT],
            TaskQuerySchema.prediction_time_name: [PREDICTION_TIME],
            TaskQuerySchema.query_name: [QUERY_CODE],
            TaskQuerySchema.duration_days_name: [duration],
        }
    ).with_columns(
        pl.col(CTX_ID_COL).cast(pl.UInt32),
        pl.col(TaskQuerySchema.prediction_time_name).cast(pl.Datetime("us")),
        pl.col(TaskQuerySchema.duration_days_name).cast(pl.Float32),
    )
    if with_bound_col:
        index_df = index_df.with_columns(pl.lit(bound_event, dtype=pl.Utf8).alias(BOUND_COL))
    return index_df


def _flat_index_df() -> pl.DataFrame:
    """A one-row ``TaskQuerySchema``-shaped index frame for ``sample_tasks.evaluate_index_df``."""
    return pl.DataFrame(
        {
            TaskQuerySchema.subject_id_name: [SUBJECT],
            TaskQuerySchema.prediction_time_name: [PREDICTION_TIME],
            TaskQuerySchema.query_name: [QUERY_CODE],
            TaskQuerySchema.duration_days_name: [DURATION_DAYS],
        }
    ).with_columns(
        pl.col(TaskQuerySchema.prediction_time_name).cast(pl.Datetime("us")),
        pl.col(TaskQuerySchema.duration_days_name).cast(pl.Float32),
    )


# ---------------------------------------------------------------------------
# Adapters: one per window decider.  Each returns the single boolean answer.
# ---------------------------------------------------------------------------


def _run_label_binary_occurrence(case: BoundaryCase) -> bool:
    row = label_binary_occurrence(
        _sequence_index_df(bound_event=None, with_bound_col=False), _events_for(case)
    ).row(0, named=True)
    return bool(row["answers"][0])


def _run_label_with_event_bounds_time_bounded(case: BoundaryCase) -> bool:
    row = label_with_event_bounds(
        _sequence_index_df(bound_event=None, with_bound_col=True), _events_for(case)
    ).row(0, named=True)
    assert row["bound_events"][0] is None, "adapter bug: this row was meant to be time-bounded"
    return bool(row["answers"][0])


def _run_label_with_event_bounds_event_bounded(case: BoundaryCase) -> bool:
    """Drive the event-bounded path over the *same* interval as the time-bounded ones.

    ``BOUND_CODE`` sits on ``WINDOW_CLOSE``, so ``(prediction_time, boundary)`` and
    ``(prediction_time, prediction_time + duration_days)`` are the identical interval and the
    shared table's expectations apply unchanged.  That equality is the point: the two window kinds
    are meant to be one rule, not two.
    """
    row = label_with_event_bounds(
        _sequence_index_df(bound_event=BOUND_CODE, with_bound_col=True), _events_for(case)
    ).row(0, named=True)
    assert row["bound_events"][0] == BOUND_CODE, "adapter bug: this row was meant to be event-bounded"
    return bool(row["answers"][0])


def _run_label_query_sequences(case: BoundaryCase) -> bool:
    """The public seam -- training shards *and* the dense eval grid both call this."""
    row = label_query_sequences(
        _sequence_index_df(bound_event=None, with_bound_col=False), _events_for(case)
    ).row(0, named=True)
    return bool(row["answers"][0])


def _explicit_start_index_df(
    bound_event: str | None, start_event: str | None, prediction_time: datetime
) -> pl.DataFrame:
    """A one-query index frame for ``label_with_explicit_starts``: bound column always present."""
    index_df = _sequence_index_df(bound_event=bound_event, with_bound_col=True).with_columns(
        pl.lit(prediction_time).cast(pl.Datetime("us")).alias(TaskQuerySchema.prediction_time_name),
        pl.lit(EVENT_BOUND_DURATION_SENTINEL if start_event is not None else 0.0)
        .cast(pl.Float32)
        .alias(START_DURATION_COL),
        pl.lit(start_event, dtype=pl.Utf8).alias(START_EVENT_COL),
    )
    return index_df


def _run_label_with_explicit_starts_time_bounded(case: BoundaryCase) -> bool:
    """A spelled-out prediction-time start (``0.0`` / null) on the time-bounded window."""
    row = label_with_explicit_starts(
        _explicit_start_index_df(None, None, PREDICTION_TIME), _events_for(case)
    ).row(0, named=True)
    assert row["start_events"][0] is None and row["start_durations"][0] == 0.0, "adapter bug"
    return bool(row["answers"][0])


def _run_label_with_explicit_starts_event_bounded(case: BoundaryCase) -> bool:
    """A spelled-out prediction-time start with ``BOUND_CODE`` on the horizon -- the same interval."""
    row = label_with_explicit_starts(
        _explicit_start_index_df(BOUND_CODE, None, PREDICTION_TIME), _events_for(case)
    ).row(0, named=True)
    assert row["bound_events"][0] == BOUND_CODE, "adapter bug: this row was meant to be event-bounded"
    return bool(row["answers"][0])


def _run_label_with_explicit_starts_resolved_start(case: BoundaryCase) -> bool:
    """A *resolved* start that lands exactly on ``PREDICTION_TIME``.

    The row's own prediction time is one tick earlier (``START_ANCHOR``) and its start event is
    ``START_CODE``, which ``_events_for`` plants on ``PREDICTION_TIME``: "first occurrence strictly
    after the anchor" therefore resolves the window's start to ``PREDICTION_TIME`` itself, and the
    ``DURATION_DAYS`` end is measured from that resolved start, so the interval is once more the
    shared ``(PREDICTION_TIME, WINDOW_CLOSE)``.  This is the configuration that checks the open
    lower bound at a resolved (event-defined) start rather than at the prediction time.
    """
    row = label_with_explicit_starts(
        _explicit_start_index_df(None, START_CODE, START_ANCHOR), _events_for(case)
    ).row(0, named=True)
    assert row["start_events"][0] == START_CODE, "adapter bug: this row was meant to have a start event"
    return bool(row["answers"][0])


def _run_evaluate_index_df(case: BoundaryCase) -> bool:
    """Stage 4's labeller, which alone has a censoring notion.

    ``evaluate_index_df`` returns a *nullable* ``boolean_value``: ``null`` means "the record ends
    before the window closes, so the answer is unknown".  That is a third state the sequence
    labellers do not have, and it would mask a boundary disagreement -- a censored row answers
    ``null`` no matter which way the comparison points.

    The shared frame already defuses it honestly rather than by special-casing: ``TAIL_CODE`` at
    ``prediction_time + 365d`` makes ``max_time`` far exceed ``window_end``, so the window is fully
    observed and censoring cannot fire.  If a ``null`` comes back anyway, the adapter refuses to
    guess -- a censored answer is reported as an adapter failure, not silently coerced to ``False``,
    because coercing it is precisely how a boundary defect would hide here.
    """
    labeled = evaluate_index_df(_flat_index_df(), _events_for(case))
    value = labeled.row(0, named=True)[TaskQuerySchema.boolean_value_name]
    if value is None:
        raise AssertionError(
            f"evaluate_index_df returned a CENSORED (null) label for case {case.name!r}.\n"
            "The shared event frame plants an event at prediction_time + 365d so the window is "
            "fully observed and censoring cannot fire, so this means the censoring rule itself "
            "changed.  Fix the adapter in tests/test_window_bounds_contract.py deliberately -- do "
            "NOT coerce null to False, which would hide a boundary defect behind the censoring "
            "notion."
        )
    return bool(value)


@dataclass(frozen=True)
class WindowDecider:
    """One function (in one configuration) that decides whether an event is inside a query window."""

    #: pytest parameter id -- kept a valid identifier so node ids stay greppable.
    id: str
    #: Fully-qualified name, printed in failure messages so the culprit is named outright.
    qualname: str
    #: How this configuration was set up, printed alongside the qualname.
    configuration: str
    run: Callable[[BoundaryCase], bool]

    @property
    def short_name(self) -> str:
        return self.qualname.rsplit(".", 1)[-1]


WINDOW_DECIDERS: tuple[WindowDecider, ...] = (
    WindowDecider(
        id="label_binary_occurrence",
        qualname="every_query.generate_tasks.query_sequence_labeling.label_binary_occurrence",
        configuration="plain time-bounded occurrence",
        run=_run_label_binary_occurrence,
    ),
    WindowDecider(
        id="label_with_event_bounds_time_bounded",
        qualname="every_query.generate_tasks.query_sequence_labeling.label_with_event_bounds",
        configuration="null bound_event, so the window ends at the horizon",
        run=_run_label_with_event_bounds_time_bounded,
    ),
    WindowDecider(
        id="label_with_event_bounds_event_bounded",
        qualname="every_query.generate_tasks.query_sequence_labeling.label_with_event_bounds",
        configuration=f"bound_event={BOUND_CODE!r} planted exactly on the horizon instant",
        run=_run_label_with_event_bounds_event_bounded,
    ),
    WindowDecider(
        id="evaluate_index_df",
        qualname="every_query.generate_tasks.sample_tasks.evaluate_index_df",
        configuration="Stage 4 single-query labeller, window arranged fully observed",
        run=_run_evaluate_index_df,
    ),
    WindowDecider(
        id="label_query_sequences",
        qualname="every_query.generate_tasks.query_sequence_labeling.label_query_sequences",
        configuration="public Stage 4' seam, time-bounded dispatch",
        run=_run_label_query_sequences,
    ),
    WindowDecider(
        id="label_with_explicit_starts_time_bounded",
        qualname="every_query.generate_tasks.query_sequence_labeling.label_with_explicit_starts",
        configuration="explicit prediction-time start (0.0 / null), window ends at the horizon",
        run=_run_label_with_explicit_starts_time_bounded,
    ),
    WindowDecider(
        id="label_with_explicit_starts_event_bounded",
        qualname="every_query.generate_tasks.query_sequence_labeling.label_with_explicit_starts",
        configuration=f"explicit prediction-time start, bound_event={BOUND_CODE!r} on the horizon instant",
        run=_run_label_with_explicit_starts_event_bounded,
    ),
    WindowDecider(
        id="label_with_explicit_starts_resolved_start",
        qualname="every_query.generate_tasks.query_sequence_labeling.label_with_explicit_starts",
        configuration=f"start_event={START_CODE!r} resolving to the prediction instant; horizon from it",
        run=_run_label_with_explicit_starts_resolved_start,
    ),
)


def _fail(decider: WindowDecider, case: BoundaryCase, got: bool) -> str:
    """Build the message the next person to flip an operator will read."""
    at = "never occurs" if case.query_event_time is None else str(case.query_event_time)
    return (
        f"{decider.short_name}: {case.description} answered {got!r}, expected {case.expected!r}.\n"
        f"\n"
        f"  labeller        : {decider.qualname}\n"
        f"  configuration   : {decider.configuration}\n"
        f"  case            : {case.name}\n"
        f"  prediction_time : {PREDICTION_TIME}\n"
        f"  window closes   : {WINDOW_CLOSE}\n"
        f"  query event at  : {at}\n"
        f"  why {case.expected!s:<5}       : {case.why}\n"
        f"\n"
        f"THE RULE (canonical -- it lives in tests/test_window_bounds_contract.py):\n"
        f"{RULE}\n"
        f"\n"
        f"Every window decider in this repo implements that one window, and they are driven here\n"
        f"from a single shared table so they cannot drift apart unnoticed again.  If you meant to\n"
        f"change the rule, change it in tests/test_window_bounds_contract.py FIRST and move every\n"
        f"decider in WINDOW_DECIDERS together -- flipping one labeller's comparison produces\n"
        f"silently wrong labels, never a crash, which is how the previous divergence survived."
    )


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", BOUNDARY_CASES, ids=lambda c: c.name)
@pytest.mark.parametrize("decider", WINDOW_DECIDERS, ids=lambda d: d.id)
def test_every_window_decider_agrees_on_every_boundary_case(
    decider: WindowDecider, case: BoundaryCase
) -> None:
    """Each window decider, on each shared boundary case, answers what the rule requires.

    The node id names both the labeller and the case, so a re-flipped operator is identified by the test name
    alone, before the message is even read.
    """
    got = decider.run(case)
    assert got == case.expected, _fail(decider, case, got)


def test_the_window_deciders_do_not_disagree_with_each_other() -> None:
    """No two window deciders may answer the shared table differently.

    The per-case test above already pins each decider to the rule, so this cannot fail on its own.
    It exists for its message: it prints the whole agreement matrix at once, which is the artefact
    that would have made the original ``<`` / ``<=`` split obvious in a single run, instead of
    needing someone to read three modules side by side.
    """
    answers = {d.id: [d.run(case) for case in BOUNDARY_CASES] for d in WINDOW_DECIDERS}
    expected = [case.expected for case in BOUNDARY_CASES]

    if all(v == expected for v in answers.values()):
        return

    name_w = max(len(c.name) for c in BOUNDARY_CASES)
    col_w = max(len(d.id) for d in WINDOW_DECIDERS) + 2
    lines = [
        "Window deciders disagree about the query window.  Full agreement matrix "
        "(`!` marks a disagreement with the rule):",
        "",
        f"  {'case':<{name_w}}  {'RULE':<6}  " + "  ".join(f"{d.id:<{col_w}}" for d in WINDOW_DECIDERS),
    ]
    for i, case in enumerate(BOUNDARY_CASES):
        cells = [
            f"{str(answers[d.id][i]) + ('!' if answers[d.id][i] != expected[i] else ''):<{col_w}}"
            for d in WINDOW_DECIDERS
        ]
        lines.append(f"  {case.name:<{name_w}}  {expected[i]!s:<6}  " + "  ".join(cells))
    lines += [
        "",
        "THE RULE (canonical -- it lives in tests/test_window_bounds_contract.py):",
        RULE,
    ]
    raise AssertionError("\n".join(lines))


def test_a_query_at_the_exact_instant_of_the_boundary_event_does_not_count() -> None:
    """An event-bounded query code sharing the boundary event's exact timestamp answers False.

    **This is the assertion the repo owner may most want to revisit**, and it is deliberately kept
    in its own test, apart from the time-bounded horizon cases, so that revisiting it is a small,
    visible change rather than an edit buried inside a sweep.

    Why it matters more than the horizon edge: MEDS clusters many codes onto a single timestamp.  A
    discharge and everything charted with it routinely share one instant, so "at the boundary" is a
    common, real shape here -- not the measure-zero coincidence it is for a horizon computed from a
    floating-point duration.  Opening this bound moves real label mass on event-bounded queries.

    It is open for consistency: the boundary is the window's upper end, and the upper end is open.
    One rule, not two.  It is also what makes the query mean what it says in English -- *"Sepsis
    before discharge, after prediction time"* -- rather than "before or simultaneous with".  The
    scenario below is that clinical case, spelled out: a lab charted one tick before the discharge,
    one charted at the discharge instant itself, and one charted one tick after.

    Both complements are load-bearing.  Without ``LAB//AFTER`` a labeller that dropped the upper
    bound entirely (``answer = does it occur at all``) would pass; without ``LAB//BEFORE`` one that
    always answered False would.
    """
    boundary_time = PREDICTION_TIME + timedelta(days=5)
    events = (
        pl.DataFrame(
            {
                _SID: [SUBJECT] * 5,
                _TIME: [
                    boundary_time - TICK,  # lab charted one tick before the discharge
                    boundary_time,  # lab charted at the discharge instant itself
                    boundary_time,  # the discharge that closes the window
                    boundary_time + TICK,  # lab charted one tick after the discharge
                    TAIL_TIME,
                ],
                _CODE: ["LAB//BEFORE", "LAB//AT", "DISCHARGE", "LAB//AFTER", TAIL_CODE],
            }
        )
        .with_columns(pl.col(_TIME).cast(pl.Datetime("us")))
        .sort(_SID, _TIME)
    )
    queries = ["LAB//BEFORE", "LAB//AT", "LAB//AFTER"]
    index_df = pl.DataFrame(
        {
            CTX_ID_COL: [0, 0, 0],
            POSITION_COL: [0, 1, 2],
            TaskQuerySchema.subject_id_name: [SUBJECT] * 3,
            TaskQuerySchema.prediction_time_name: [PREDICTION_TIME] * 3,
            TaskQuerySchema.query_name: queries,
            TaskQuerySchema.duration_days_name: [EVENT_BOUND_DURATION_SENTINEL] * 3,
            BOUND_COL: ["DISCHARGE"] * 3,
        }
    ).with_columns(
        pl.col(CTX_ID_COL).cast(pl.UInt32),
        pl.col(TaskQuerySchema.prediction_time_name).cast(pl.Datetime("us")),
        pl.col(TaskQuerySchema.duration_days_name).cast(pl.Float32),
    )

    row = label_with_event_bounds(index_df, events).row(0, named=True)
    assert row["queries"] == queries, "adapter bug: the answer list is not in the order asked"
    got = dict(zip(queries, row["answers"], strict=True))
    expected = {"LAB//BEFORE": True, "LAB//AT": False, "LAB//AFTER": False}
    if got == expected:
        return

    detail = []
    if got["LAB//AT"] is not False:
        detail.append(
            f"  LAB//AT     : a query code AT the DISCHARGE instant answered {got['LAB//AT']!r}, "
            "expected False.\n"
            "                The event bound is STRICT: an occurrence sharing the boundary\n"
            "                event's timestamp did NOT occur *before* it, so it is outside the\n"
            "                window.  Someone has made `label_with_event_bounds` compare\n"
            "                `_q_time <= window_end` again."
        )
    if got["LAB//AFTER"] is not False:
        detail.append(
            f"  LAB//AFTER  : a query code one microsecond AFTER the DISCHARGE answered "
            f"{got['LAB//AFTER']!r}, expected False.\n"
            "                It is past the boundary under any reading of the bound, so this is\n"
            "                not a boundary flip -- something larger is wrong."
        )
    if got["LAB//BEFORE"] is not True:
        detail.append(
            f"  LAB//BEFORE : a query code one microsecond BEFORE the DISCHARGE answered "
            f"{got['LAB//BEFORE']!r}, expected True.\n"
            "                That one is inside the window under either reading of the bound, so\n"
            "                this is not a boundary flip -- the upper bound has been lost\n"
            "                entirely, or the window is empty."
        )

    raise AssertionError(
        "label_with_event_bounds: the event-bounded window's upper bound is wrong.\n\n"
        + "\n".join(detail)
        + "\n\n"
        + f"  prediction_time : {PREDICTION_TIME}\n"
        + f"  DISCHARGE at    : {boundary_time}  (this instant closes the window, and is OUTSIDE it)"
        + "\n\n"
        + "THE RULE (canonical -- it lives in tests/test_window_bounds_contract.py):\n"
        + RULE
        + "\n\n"
        + "MEDS clusters codes onto shared timestamps, so this edge decides real rows.  If the\n"
        + "owner has decided to make the event bound inclusive, flip this test and the\n"
        + "`label_with_event_bounds_event_bounded` rows of\n"
        + "`test_every_window_decider_agrees_on_every_boundary_case` together -- those two places\n"
        + "are the only ones in this file that pin it."
    )


def test_the_boundary_search_itself_is_strict_at_the_prediction_instant() -> None:
    """A boundary-code occurrence *at* ``prediction_time`` does not close the window.

    The lower bound is strict for the boundary search too, not only for the query search -- so a boundary code
    charted at the prediction instant is skipped and the *next* occurrence closes the window.  Without this,
    closing the upper bound would make every event-bounded query whose boundary code sits on the prediction
    instant collapse to an empty window and answer False forever, which is the mirror-image silent-wrong-label
    defect.
    """
    events = (
        pl.DataFrame(
            {
                _SID: [SUBJECT] * 4,
                _TIME: [
                    PREDICTION_TIME,  # a DISCHARGE at the prediction instant: must NOT close it
                    PREDICTION_TIME + timedelta(days=2),  # the query occurrence
                    PREDICTION_TIME + timedelta(days=5),  # the DISCHARGE that does close it
                    TAIL_TIME,
                ],
                _CODE: ["DISCHARGE", QUERY_CODE, "DISCHARGE", TAIL_CODE],
            }
        )
        .with_columns(pl.col(_TIME).cast(pl.Datetime("us")))
        .sort(_SID, _TIME)
    )
    got = label_with_event_bounds(
        _sequence_index_df(bound_event="DISCHARGE", with_bound_col=True), events
    ).row(0, named=True)["answers"]
    assert got == [True], (
        f"label_with_event_bounds: answered {got!r}, expected [True].\n"
        "A DISCHARGE at the prediction instant closed the window, leaving an empty interval, so a\n"
        "query occurring 2 days later was missed.  The boundary search is STRICT at the lower end:\n"
        f"the boundary is the first occurrence STRICTLY AFTER prediction_time ({PREDICTION_TIME}).\n"
        "\nTHE RULE (canonical -- it lives in tests/test_window_bounds_contract.py):\n" + RULE
    )
