"""Frozen dataclasses the redesigned task sampler is built on.

These four immutable value types are the stable contract shared by the redesigned
sampler's pure stage functions (``sample_queries``, ``sample_patient_contexts``,
``match_queries_and_contexts``, ``compute_labels`` — see
``task-sampling-design.md``):

- :class:`QuerySpec` — one sampled query/task (replaces the old ``TaskSpec`` in
  ``sample_tasks.py``).
- :class:`QueryDistribution` — the distribution queries are drawn from.
- :class:`PatientContext` — one sampled ``(subject_id, prediction_time)`` context.
- :class:`ContextDistribution` — eligibility + replacement policy for contexts.

All are ``@dataclass(frozen=True)`` so they are hashable and safe to use as iid
samples (the sampler deliberately allows duplicate ``QuerySpec`` / ``PatientContext``
values — the repeats *are* the sample — so value equality and hashing must behave).
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

# How ``QuerySpec.duration_days`` is drawn over ``[min_duration_days, max_duration_days]``.
# Plain string Literals (rather than an Enum) so Hydra/YAML configs can pass the value
# through directly without enum coercion. Spelling matches the design doc.
DurationDistribution = Literal["uniform", "log-uniform"]

_LEGAL_DURATION_DISTRIBUTIONS = ("uniform", "log-uniform")


@dataclass(frozen=True)
class QuerySpec:
    """A single pre-training query/task: one query code and one prediction-window duration (in days).

    Replaces the old ``TaskSpec``. ``duration_days`` is a ``float`` (continuous horizon)
    to match ``TaskQuerySchema.duration_days`` (``float32``).
    """

    code: str
    duration_days: float


@dataclass(frozen=True)
class QueryDistribution:
    """Distribution that provides samples of queries for EQ training.

    A sample is a `QuerySpec` instance, i.e. a ``(code, duration_days)`` tuple. The
    ``code`` is drawn **uniformly** over ``codes`` (no per-code weighting); only the
    duration draw is configurable.
    """

    codes: list[str]  # QuerySpec.code is drawn uniformly at random from this list
    max_duration_days: float  # upper bound of the duration draw
    min_duration_days: float = 1.0  # lower bound (default 1 day; must be > 0 so log-uniform is well-defined)
    # how QuerySpec.duration_days is drawn over [min_duration_days, max_duration_days]
    duration_days_distribution: DurationDistribution = "log-uniform"

    def __post_init__(self) -> None:
        if not self.codes:
            raise ValueError("codes must be non-empty")
        if self.min_duration_days <= 0:
            raise ValueError(
                f"min_duration_days must be > 0 (got {self.min_duration_days}); "
                "log-uniform needs positive bounds"
            )
        if self.max_duration_days < self.min_duration_days:
            raise ValueError(
                f"max_duration_days ({self.max_duration_days}) must be >= "
                f"min_duration_days ({self.min_duration_days})"
            )
        if self.duration_days_distribution not in _LEGAL_DURATION_DISTRIBUTIONS:
            raise ValueError(
                f"duration_days_distribution must be one of {_LEGAL_DURATION_DISTRIBUTIONS} "
                f"(got {self.duration_days_distribution!r})"
            )


@dataclass(frozen=True)
class PatientContext:
    """A single patient context: one subject and the prediction time that cuts off observable history.

    Everything *at or before* ``prediction_time`` is model input; the label is evaluated
    over the window *strictly after* ``prediction_time``. The inclusive cutoff matches
    ``meds-torch-data``'s loader (backward ``join_asof``, ``time <= prediction_time``), so
    the event at the cutoff is fed to the model and the input/label boundary partitions the
    timeline with no overlap and no gap.
    """

    subject_id: int
    prediction_time: datetime


@dataclass(frozen=True)
class ContextDistribution:
    """Eligibility + replacement policy for sampling patient contexts.

    A sample from the distribution is a single :class:`PatientContext`, i.e. a
    ``(subject_id, prediction_time)`` tuple drawn from the eligible ``(subject, event-time)``
    pairs in the dataset.
    """

    min_context_events: int  # a (subject, time) pair is eligible only once the subject has >= this many prior events
    with_replacement: bool  # True for PT, so N*M may exceed the number of eligible pairs

    def __post_init__(self) -> None:
        if self.min_context_events < 0:
            raise ValueError(f"min_context_events must be >= 0 (got {self.min_context_events})")
