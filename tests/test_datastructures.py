"""Unit tests for the redesigned task sampler's frozen dataclasses (issue #188)."""

import dataclasses
from datetime import datetime

import pytest

from every_query.generate_tasks.datastructures import (
    ContextDistribution,
    PatientContext,
    QueryDistribution,
    QuerySpec,
)


# ---------------------------------------------------------------------------
# Construction + field round-trip
# ---------------------------------------------------------------------------


def test_query_spec_construction():
    q = QuerySpec(code="ICD//I10", duration_days=30.0)
    assert q.code == "ICD//I10"
    assert q.duration_days == 30.0


def test_query_distribution_construction():
    d = QueryDistribution(
        codes=["A", "B"],
        max_duration_days=365.0,
        min_duration_days=2.0,
        duration_days_distribution="uniform",
    )
    assert d.codes == ["A", "B"]
    assert d.max_duration_days == 365.0
    assert d.min_duration_days == 2.0
    assert d.duration_days_distribution == "uniform"


def test_patient_context_construction():
    pt = datetime(2023, 1, 1)
    c = PatientContext(subject_id=7, prediction_time=pt)
    assert c.subject_id == 7
    assert c.prediction_time == pt


def test_context_distribution_construction():
    c = ContextDistribution(min_context_events=5, with_replacement=True)
    assert c.min_context_events == 5
    assert c.with_replacement is True


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_query_distribution_defaults():
    d = QueryDistribution(codes=["A"], max_duration_days=365.0)
    assert d.min_duration_days == 1.0
    assert d.duration_days_distribution == "log-uniform"


# ---------------------------------------------------------------------------
# Frozen / immutable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "obj, field, value",
    [
        (QuerySpec("A", 1.0), "code", "B"),
        (QueryDistribution(["A"], 365.0), "max_duration_days", 1.0),
        (PatientContext(1, datetime(2023, 1, 1)), "subject_id", 2),
        (ContextDistribution(1, True), "with_replacement", False),
    ],
)
def test_frozen(obj, field, value):
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(obj, field, value)


# ---------------------------------------------------------------------------
# Equality + hashability (duplicates are allowed and must compare/hash equal)
# ---------------------------------------------------------------------------


def test_query_spec_equality_and_hash():
    a = QuerySpec("A", 30.0)
    b = QuerySpec("A", 30.0)
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1


def test_patient_context_equality_and_hash():
    pt = datetime(2023, 1, 1)
    a = PatientContext(1, pt)
    b = PatientContext(1, pt)
    assert a == b
    assert hash(a) == hash(b)


# ---------------------------------------------------------------------------
# Validation (all raise ValueError)
# ---------------------------------------------------------------------------


def test_empty_codes_raises():
    with pytest.raises(ValueError, match="codes must be non-empty"):
        QueryDistribution(codes=[], max_duration_days=365.0)


@pytest.mark.parametrize("bad_min", [0.0, -1.0])
def test_nonpositive_min_duration_raises(bad_min):
    with pytest.raises(ValueError, match="min_duration_days must be > 0"):
        QueryDistribution(codes=["A"], max_duration_days=365.0, min_duration_days=bad_min)


def test_max_below_min_raises():
    with pytest.raises(ValueError, match="must be >= min_duration_days"):
        QueryDistribution(codes=["A"], max_duration_days=5.0, min_duration_days=10.0)


def test_bad_duration_distribution_raises():
    with pytest.raises(ValueError, match="duration_days_distribution must be one of"):
        QueryDistribution(
            codes=["A"],
            max_duration_days=365.0,
            duration_days_distribution="gaussian",  # type: ignore[arg-type]
        )


def test_negative_min_context_events_raises():
    with pytest.raises(ValueError, match="min_context_events must be >= 0"):
        ContextDistribution(min_context_events=-1, with_replacement=False)


def test_max_equal_min_is_valid():
    d = QueryDistribution(codes=["A"], max_duration_days=10.0, min_duration_days=10.0)
    assert d.max_duration_days == d.min_duration_days
