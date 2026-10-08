"""The golden truth table, checked three ways.

1. :func:`test_oracle_matches_truth_table` — the independent oracle reproduces every hand-written
   answer.  This is what earns the oracle the right to be used as a reference anywhere else.
2. :func:`test_production_matches_truth_table` — the production labeler reproduces them too.
3. :func:`test_enabling_ontology_does_not_move_leaf_answers` — turning the DAG on may only ADD
   the ability to ask about ancestors; it must never change what a leaf query means.

Tests 2 and 3 are the ones that catch the two closure defects.
"""

from __future__ import annotations

import pytest

from tests.ontology_suite import production as prod
from tests.ontology_suite.golden import (
    DECLARED_PARENTS,
    EVENTS,
    LEAVES,
    ONTOLOGY,
    T0,
    TRUTH_TABLE,
    Case,
)
from tests.ontology_suite.oracle import label_duration, label_event_bounded

DURATION_CASES = [c for c in TRUTH_TABLE if c.bound_event is None]
EVENT_CASES = [c for c in TRUTH_TABLE if c.bound_event is not None]


@pytest.fixture(scope="module")
def closure_df():
    """The production closure built from the golden vocabulary."""
    _nodes, _mix, closure = prod.build_artifacts(LEAVES, DECLARED_PARENTS)
    return closure


def _oracle_answer(case: Case) -> bool:
    if case.bound_event is None:
        return label_duration(EVENTS, case.subject_id, ONTOLOGY, T0, case.query, case.duration_days)
    return label_event_bounded(EVENTS, case.subject_id, ONTOLOGY, T0, case.query, case.bound_event)


# --------------------------------------------------------------------------------------------
# 1. The oracle reproduces the hand-written table
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("case", TRUTH_TABLE, ids=lambda c: c.case_id)
def test_oracle_matches_truth_table(case: Case):
    """The oracle is only trustworthy if it independently lands on every hand-computed answer."""
    assert _oracle_answer(case) is case.expected, (
        f"{case.case_id}: oracle said {_oracle_answer(case)}, hand-computed {case.expected}.\n  {case.why}"
    )


def test_truth_table_covers_the_required_shapes():
    """Guard against the fixture quietly losing a case class it is supposed to exercise."""
    assert DURATION_CASES, "no duration-bounded cases"
    assert EVENT_CASES, "no event-bounded cases"
    kinds = {(c.form, c.target_kind) for c in TRUTH_TABLE}
    for want in (("duration", "leaf"), ("duration", "ancestor"), ("event", "leaf"), ("event", "ancestor")):
        assert want in kinds, f"truth table has no {want[0]}-bounded / {want[1]}-target case"

    assert any(c.ontology_required for c in TRUTH_TABLE), "no ontology-required case"
    assert any(c.censored for c in TRUTH_TABLE), "no censored case"
    assert any(c.expected for c in TRUTH_TABLE) and any(not c.expected for c in TRUTH_TABLE)


def test_censor_flags_agree_with_the_oracle():
    """`censored` in the table is the `TIMELINE//END` answer, not a separate hand-waved concept."""
    from tests.ontology_suite.oracle import label_censor

    for case in DURATION_CASES:
        got = label_censor(EVENTS, case.subject_id, ONTOLOGY, T0, case.duration_days)
        assert got is case.censored, f"{case.case_id}: censor {got}, expected {case.censored}"


# --------------------------------------------------------------------------------------------
# 2. Production reproduces the hand-written table
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("case", DURATION_CASES, ids=lambda c: c.case_id)
def test_production_matches_truth_table_duration(case: Case, closure_df):
    rows = [(case.subject_id, T0, case.query, case.duration_days, None)]
    got = prod.label(rows, EVENTS, closure_df=closure_df, with_bounds=False)[0]
    assert got is case.expected, (
        f"{case.case_id}: production said {got}, hand-computed {case.expected}.\n  {case.why}"
    )


@pytest.mark.parametrize("case", EVENT_CASES, ids=lambda c: c.case_id)
def test_production_matches_truth_table_event_bounded(case: Case, closure_df):
    rows = [(case.subject_id, T0, case.query, prod.EVENT_BOUND_SENTINEL, case.bound_event)]
    got = prod.label(rows, EVENTS, closure_df=closure_df, with_bounds=True)[0]
    assert got is case.expected, (
        f"{case.case_id}: production said {got}, hand-computed {case.expected}.\n  {case.why}"
    )


# --------------------------------------------------------------------------------------------
# 3. Enabling the ontology must not move leaf answers
# --------------------------------------------------------------------------------------------


def test_enabling_ontology_does_not_move_leaf_answers(closure_df):
    """The whole safety property in one test.

    For every LEAF query in the table, the answer with the ontology switched on must equal the answer with it
    switched off.  Ancestor queries are excluded — they are only answerable with the ontology on, which is the
    feature.
    """
    leaf_cases = [c for c in DURATION_CASES if c.target_kind == "leaf"]
    assert leaf_cases, "no leaf duration cases to compare"

    rows = [(c.subject_id, T0, c.query, c.duration_days, None) for c in leaf_cases]
    off = prod.label(rows, EVENTS, closure_df=None, with_bounds=False)
    on = prod.label(rows, EVENTS, closure_df=closure_df, with_bounds=False)

    moved = [
        f"{c.case_id} ({c.query}, d={c.duration_days}): off={o} on={n} — {c.why}"
        for c, o, n in zip(leaf_cases, off, on, strict=True)
        if o != n
    ]
    assert not moved, "enabling the ontology changed ordinary leaf labels:\n  " + "\n  ".join(moved)


def test_ontology_required_cases_are_impossible_without_the_closure(closure_df):
    """An ontology-required positive must be False with the ontology off and True with it on.

    This is the other half of the safety property: it proves the ontology-required cases are
    genuinely testing the closure and not passing for some incidental reason.
    """
    required = [c for c in DURATION_CASES if c.ontology_required and c.expected]
    assert required, "no positive ontology-required duration cases"

    rows = [(c.subject_id, T0, c.query, c.duration_days, None) for c in required]
    off = prod.label(rows, EVENTS, closure_df=None, with_bounds=False)
    on = prod.label(rows, EVENTS, closure_df=closure_df, with_bounds=False)

    for c, o, n in zip(required, off, on, strict=True):
        assert o is False, f"{c.case_id}: expected unanswerable without the closure, got {o}"
        assert n is True, f"{c.case_id}: closure should make this True, got {n}.\n  {c.why}"
