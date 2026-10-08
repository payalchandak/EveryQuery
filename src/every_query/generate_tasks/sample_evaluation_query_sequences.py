"""Dense-grid evaluation-task generator for conditional query *sequences*.

Sibling of :mod:`~every_query.generate_tasks.query_sequence_labeling` (scattered shape: every
context draws its own independent query sequence).  This module produces the **dense** shape:
a fixed set of ``N`` query sequences, each labeled at **every** context of one cohort.  For a
given sequence the only thing varying across its rows is the patient, which is what per-sequence
metrics need — pooling across heterogeneous sequences instead measures cross-query base-rate
separation rather than within-task skill.

Relationship to ``sample_evaluation_tasks``: **same knobs, same semantics, different task axis.**
That module cross-joins its cohort with a ``codes x durations`` grid and emits flat
``TaskQuerySchema`` rows for ``EQ_predict``; this one cross-joins the *same* cohort with ``N``
ordered :class:`SequenceSpec` s and emits :class:`QuerySeqSchema` rows (``queries`` /
``durations`` / ``answers`` list columns) for ``EQ_predict_multitask``.  Everything on the cohort
side is imported from ``sample_evaluation_tasks`` and driven by its knobs and seed axes, so for
the same ``(seed, split, prediction_times_per_subject, min_context_per_subject,
subject_subsample_fraction)`` the two generators score the **identical** ``(subject, time)`` set
— the flat grid and the sequence grid are directly comparable.

Pipeline, one worker per shard of ``{data_dir}/data/{split}/*.parquet`` (all shards, no knob):

    1. Resolve the ``N`` :class:`SequenceSpec` s **once**: read designed ones from
       ``sequences_path``, or draw ``num_evaluation_sequences`` from the training query distribution
       (:func:`sample_sequence_specs`, seeded on ``(seed, "eval_seq_specs", split)`` alone so the
       same specs come out for any cohort).  Validate every code against the model vocabulary.
    2. Per shard: read its events; take the cohort either from ``contexts_path`` (this shard's
       subjects only) or by sampling it exactly as ``sample_evaluation_tasks`` does —
       :func:`~every_query.generate_tasks.sample_evaluation_tasks.subsample_subject_ids` then
       :func:`~every_query.generate_tasks.sample_evaluation_tasks.sample_prediction_times_per_subject`
       on the same ``("subject_subsample" | "prediction_times", split, shard)`` seed axes.
    3. Cross-join cohort x specs into the flat per-query index frame
       (:func:`build_dense_sequence_index_df`), label with
       :func:`~every_query.generate_tasks.query_sequence_labeling.label_query_sequences`, align to
       ``QuerySeqSchema``, write.

Ontology support mirrors the training sampler on both of its halves, because a grid that mirrors
only one measures the wrong thing without ever failing: ``ontology_dir`` extends the query
universe (:func:`~every_query.generate_tasks.query_sequence_labeling.build_query_universe`) so
ancestor nodes are drawn - as queries and as boundaries - and accepted as designed-spec codes, and
it explodes the event stream through the closure
(:func:`~every_query.generate_tasks.query_sequence_labeling.maybe_expand_to_matching_query_nodes`) so an
ancestor query is labeled by ordinary occurrence.  Without the explosion an ancestor query is
labeled ``False`` at every context — the ancestor's *name* is in no event stream, only its
descendants' are — which is a well-formed parquet of wrong answers, not an error.

Output layout, the same as ``sample_evaluation_tasks``':

    ``{out_dir}/eval/{split}/{shard}.parquet``          the labeled grid, one file per shard
    ``{out_dir}/eval_unique/{split}/{shard}.parquet``   deduped ``(subject_id, prediction_time)``
                                                        (``write_unique_prediction_times``)

Every output parquet gets a provenance sidecar under ``{out_dir}_artifacts/_labeled/`` recording the
inputs that determine its rows — the ontology, the specs, the cohort source and the optional
minimum-positive threshold — and ``overwrite=false`` skips a shard only when these match
(:func:`_run_fingerprint`).  Re-running into the same ``out_dir`` with a different
``sequences_path``/``num_evaluation_sequences``/``seed``/cohort
therefore relabels rather than silently keeping the previous grid.

When ``min_task_positives`` is set, all shards are labeled before a split-wide pass counts true
final answers for each conditional task (including prior observed answers in its key).  Every row
of an eligible task survives, preserving its negative examples.  A split-level completion marker
keeps partially filtered outputs from being mistaken for a reusable finished grid; changing the
threshold or retrying an interrupted run relabels the full grid before filtering.

``{out_dir}/eval`` is directly consumable as ``EQ_predict_multitask tasks_dir=...`` (MEDS-TorchData
rglobs it, so point it at ``eval/`` — never at ``out_dir`` itself, or the ``eval_unique/`` frames
are read as labels).  Give this generator and ``EQ_generate_evaluation_tasks`` distinct ``out_dir``
roots: both write ``eval/{split}/{shard}.parquet``, in incompatible schemas.

Within one shard file spec identity is recoverable two ways: the ``queries``/``durations`` columns
(plus ``bound_events`` / ``start_durations`` / ``start_events`` when present) *are* the spec, and row
order is context-major — row ``i`` is ``(contexts[i // N], specs[i % N])``.

Window starts (issue #27).  A query's window may open later than the prediction time — after a
delay (``start_duration_days``) or at the next occurrence of a start event — with the end then
measured from the *resolved start*::

    start  = prediction_time + start_duration  |  first start_event strictly after prediction_time
    end    = start + duration                  |  first bound_event strictly after the resolved start
    answer = start < occurrence(query) < end   (both endpoints open; no start event => empty window;
                                                no end event after a resolved start => end of record)

These are the multitask sampler's semantics, labeled here through the same ``interval_table``
(:func:`~every_query.generate_tasks.query_sequence_labeling.label_with_explicit_starts`), so a
multitask model can be scored on this grid.  Designed specs set starts explicitly (the
``start_event`` / ``start_duration_days`` keys of each entry, or the parquet columns of the same
names); sampled specs draw them from
the ``eventstart_fraction`` / ``prediction_time_start_fraction`` / ``start_duration_*`` /
``start_event_codes`` knobs on three seed axes of their own.  With the default knobs every window
opens at the prediction time and the output carries no start columns, exactly as before.  Active
starts are opt-in on the reader side too: ``QuerySeqPytorchDataset`` refuses to tensorize them
unless asked, so only ``EQ_predict_multitask`` consumes a grid that carries them.

Forced answers.  A designed spec may fix the answer an earlier query must have when it conditions a
later one (``forced_answer``: "among contexts where Q1 = YES, what is P(Q2)?").  It selects the spec's
cohort: it is tiled onto the labeled grid as the optional ``forced_answers`` list column
(:func:`attach_forced_answers`), and :func:`drop_forced_mismatches` then drops every row whose labeled
truth disagrees, so a forced spec is **not** dense — it holds only the contexts where its conditioning
really happened, and the model is never told a counterfactual.  ``answers`` is never touched, so the
label a row is scored against is always the truth.  The final query of a sequence is the scored one
and conditions nothing, so its forced answer must be null (:class:`SequenceSpec` rejects anything
else).  Designed specs spell out every field of every entry — see :func:`read_sequence_specs`.

Answers follow the module-wide sequence contract: binary, never null; an unobservable occurrence
(record ends before the window does) is ``False``, and censoring is carried by an explicit
``TIMELINE//END`` query rather than a null answer.  Unlike ``sample_evaluation_tasks`` there is
therefore no censored-row filter here.  If you need censored windows *excluded* from metrics rather
than counted as negatives, that filtering belongs downstream of ``EQ_predict_multitask``, on the
prediction rows: pair each row with its ``TIMELINE//END`` query at the same window and drop the
rows that one answers ``True``.  Nothing in this tree does it for you.
"""

import dataclasses
import hashlib
import json
import logging
import math
import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from omegaconf import DictConfig

from every_query.data.query_seq_dataset import (
    ANSWERS_COL,
    EVENT_BOUND_DURATION_SENTINEL,
    FORCED_ANSWERS_COL,
    QUERIES_COL,
)
from every_query.data.query_seq_task import TASK_KEY, _with_prior_answers, normalize_task_columns
from every_query.data.schema import QuerySeqSchema, TaskQuerySchema
from every_query.generate_tasks.query_sequence_labeling import (
    BOUND_COL,
    CTX_ID_COL,
    POSITION_COL,
    START_DURATION_COL,
    START_EVENT_COL,
    QuerySequenceDistribution,
    build_query_universe,
    label_query_sequences,
    maybe_expand_to_matching_query_nodes,
)
from every_query.generate_tasks.sample_evaluation_tasks import (
    _labels_fp,
    _unique_fp,
    sample_prediction_times_per_subject,
    subsample_subject_ids,
)
from every_query.generate_tasks.sample_tasks import (
    LABELED_DIRNAME,
    _atomic_write_json,
    _atomic_write_parquet,
    _read_event_shard,
    _require_path_arg,
    _split_shards,
    default_artifacts_dir,
    read_query_codes,
)
from every_query.utils.digest import frame_digest
from every_query.utils.seeds import derive_seed

logger = logging.getLogger(__name__)

SPEC_ID_COL = "_spec_id"
#: Designed-spec input key / long-format parquet column; the labeled grid carries the per-row list
#: under ``FORCED_ANSWERS_COL``.
FORCED_ANSWER_COL = "forced_answer"

# Spec names become directory names under ``per_spec_dirs``, and YAML keys / parquet ``seq_id``
# values are free-form (MEDS codes contain ``/``).  Anything outside this set is replaced with
# ``_``; collisions after sanitising are rejected rather than silently overwritten.
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class SequenceSpec:
    """One query sequence to evaluate at every cohort context.

    Attributes:
        name: Identifier, used as the output directory name under ``per_spec_dirs``.
        queries: Ordered MEDS code strings.
        durations: Per-query horizons in days after the resolved start, aligned with ``queries``.
        bounds: Optional per-query boundary codes aligned with ``queries``. Null entries are
            horizon-bounded; non-null entries require the event-bound duration sentinel.
        start_durations: Optional per-query window starts in days after the prediction time
            (issue #27), aligned with ``queries``: ``0.0`` opens the window at the prediction time
            (the default when the tuple is empty), ``> 0`` later, and the event-bound sentinel
            (``-1.0``) marks an event-defined start named in ``start_events``.
        start_events: Optional per-query start-event codes aligned with ``start_durations`` (null
            for a duration-defined start).  Both start tuples are empty together (every window
            opens at the prediction time) or aligned together.
        forced_answers: Optional per-query conditioning answers aligned with ``queries``: ``True`` /
            ``False`` is fed to the decoder as that query's answer *instead of* the labeled one
            ("assume Q1 = YES"), ``None`` teacher-forces the ground truth.  Empty means no position
            is forced.  The labeled ``answers`` are never touched, so the scoring label stays the
            truth.  The **last** position must be ``None``: it is the scored query, its answer is
            never fed to the decoder, and a value there would be a silent no-op.

    Examples:
        >>> SequenceSpec("mortality_30d", ("TIMELINE//END", "MEDS_DEATH"), (1.0, 30.0))
        SequenceSpec(name='mortality_30d', queries=('TIMELINE//END', 'MEDS_DEATH'),
                     durations=(1.0, 30.0), bounds=(), start_durations=(), start_events=(),
                     forced_answers=())

        A position may instead be bounded by an *event*: its window runs to the next occurrence
        of that code, so it carries the duration sentinel rather than a horizon.

        >>> spec = SequenceSpec("sepsis_before_discharge", ("SEPSIS",), (-1.0,),
        ...                         ("HOSPITAL_DISCHARGE//HOME",))
        >>> spec.bound_at(0)
        'HOSPITAL_DISCHARGE//HOME'

        A position may also *open* later than the prediction time — after a delay, or at the next
        occurrence of a start event (the sentinel marks the event form).  The end is then measured
        from that resolved start.  A spec without start tuples opens every window at the prediction
        time, so ``start_at`` is always answerable:

        >>> post = SequenceSpec("post_admission", ("LAB//X", "ICD//I10"), (30.0, 30.0),
        ...                     start_durations=(-1.0, 7.0), start_events=("HOSPITAL_ADMISSION", None))
        >>> post.start_at(0), post.start_at(1), post.has_active_starts
        ((-1.0, 'HOSPITAL_ADMISSION'), (7.0, None), True)
        >>> spec.start_at(0), spec.has_active_starts
        ((0.0, None), False)

        An earlier query's conditioning answer may be dictated rather than teacher-forced from the
        truth; the final (scored) query's may not, and neither may a non-boolean:

        >>> SequenceSpec("cf", ("TIMELINE//END", "MEDS_DEATH"), (30.0, 30.0),
        ...              forced_answers=(False, None)).forced_answers
        (False, None)
        >>> SequenceSpec("bad", ("TIMELINE//END", "MEDS_DEATH"), (30.0, 30.0), forced_answers=(None, True))
        Traceback (most recent call last):
            ...
        ValueError: spec 'bad' forces the answer of its final query 'MEDS_DEATH' (position 1) to
        True; the final query is the scored one and its answer is never fed to the model, so its
        forced_answer must be null
        >>> SequenceSpec("bad", ("A",), (1.0,), forced_answers=(False,))
        Traceback (most recent call last):
            ...
        ValueError: spec 'bad' forces the answer of its final query 'A' (position 0) to False; the
        final query is the scored one and its answer is never fed to the model, so its
        forced_answer must be null
        >>> SequenceSpec("bad", ("A", "B"), (1.0, 1.0), forced_answers=(1, None))
        Traceback (most recent call last):
            ...
        TypeError: spec 'bad' forced answer at position 0 must be true, false or null, got int: 1

        A bounded position with a real horizon is a contradiction and is rejected, as is a start
        event without the sentinel or a sentinel without a start event:

        >>> SequenceSpec("bad", ("SEPSIS",), (30.0,), ("DISCHARGE",))
        Traceback (most recent call last):
            ...
        ValueError: spec 'bad' position 0 is bounded by 'DISCHARGE', so its duration must be
        the -1.0 sentinel, got 30.0
        >>> SequenceSpec("bad", ("A",), (1.0,), start_durations=(7.0,), start_events=("ADMIT",))
        Traceback (most recent call last):
            ...
        ValueError: spec 'bad' position 0 starts at 'ADMIT', so its start duration must be the
        -1.0 sentinel, got 7.0
        >>> SequenceSpec("bad", ("A",), (1.0,), start_durations=(-1.0,), start_events=(None,))
        Traceback (most recent call last):
            ...
        ValueError: spec 'bad' start duration -1.0 at position 0 must be a finite number >= 0
        (or the -1.0 sentinel with a start_event)

        Ragged or empty specs are rejected at construction:

        >>> SequenceSpec("bad", ("A", "B"), (1.0,))
        Traceback (most recent call last):
            ...
        ValueError: spec 'bad' has 2 queries but 1 duration(s)
        >>> SequenceSpec("bad", (), ())
        Traceback (most recent call last):
            ...
        ValueError: spec 'bad' is empty; a sequence needs at least one query
        >>> SequenceSpec("bad", ("A",), (0.0,))
        Traceback (most recent call last):
            ...
        ValueError: spec 'bad' duration 0.0 at position 0 must be a finite number > 0
        >>> SequenceSpec("bad", ("A", "B"), (1.0, 1.0), start_durations=(0.0,), start_events=(None,))
        Traceback (most recent call last):
            ...
        ValueError: spec 'bad' has 2 queries but 1 start duration(s) / 1 start event(s)
        >>> SequenceSpec("bad", ("A",), (1.0,), start_durations=(0.0,))
        Traceback (most recent call last):
            ...
        ValueError: spec 'bad' has start_durations but not start_events; give both or neither
    """

    name: str
    queries: tuple[str, ...]
    durations: tuple[float, ...]
    bounds: tuple[str | None, ...] = ()
    start_durations: tuple[float, ...] = ()
    start_events: tuple[str | None, ...] = ()
    forced_answers: tuple[bool | None, ...] = ()

    def __post_init__(self) -> None:
        if not self.queries:
            raise ValueError(f"spec {self.name!r} is empty; a sequence needs at least one query")
        if len(self.queries) != len(self.durations):
            raise ValueError(
                f"spec {self.name!r} has {len(self.queries)} queries but {len(self.durations)} duration(s)"
            )
        for i, q in enumerate(self.queries):
            if not isinstance(q, str) or not q:
                raise ValueError(f"spec {self.name!r} query at position {i} must be a non-empty string")
        if self.bounds and len(self.bounds) != len(self.queries):
            raise ValueError(
                f"spec {self.name!r} has {len(self.queries)} queries but {len(self.bounds)} bound(s)"
            )
        for i, bound in enumerate(self.bounds):
            if bound is not None and (not isinstance(bound, str) or not bound):
                raise ValueError(
                    f"spec {self.name!r} boundary at position {i} must be a non-empty string or null"
                )
        for i, d in enumerate(self.durations):
            # ``bool`` is an ``int`` subclass, so ``True`` would otherwise become ``1.0`` days.
            if isinstance(d, bool) or not isinstance(d, int | float):
                raise TypeError(
                    f"spec {self.name!r} duration at position {i} must be a number, "
                    f"got {type(d).__name__}: {d!r}"
                )
            # An event-bounded position has no horizon — its window ends at a boundary event —
            # so it carries the negative sentinel instead.  That is the *only* licence for a
            # non-positive duration; an unbounded position must still name a real horizon.
            if self.bound_at(i) is not None:
                if float(d) != EVENT_BOUND_DURATION_SENTINEL:
                    raise ValueError(
                        f"spec {self.name!r} position {i} is bounded by {self.bound_at(i)!r}, so its "
                        f"duration must be the {EVENT_BOUND_DURATION_SENTINEL} sentinel, got {float(d)}"
                    )
                continue
            if not math.isfinite(d) or d <= 0:
                raise ValueError(
                    f"spec {self.name!r} duration {float(d)} at position {i} must be a finite number > 0"
                )
        self._validate_starts()
        self._validate_forced_answers()

    def _validate_forced_answers(self) -> None:
        """Forced answers are bool-or-null, aligned to the queries, and null on the final query."""
        if not self.forced_answers:
            return
        if len(self.forced_answers) != len(self.queries):
            raise ValueError(
                f"spec {self.name!r} has {len(self.queries)} queries but {len(self.forced_answers)} "
                "forced answer(s)"
            )
        for i, fa in enumerate(self.forced_answers):
            # Strictly bool: ``1`` / ``"yes"`` are far likelier a typo than an intended answer.
            if fa is not None and not isinstance(fa, bool):
                raise TypeError(
                    f"spec {self.name!r} forced answer at position {i} must be true, false or null, "
                    f"got {type(fa).__name__}: {fa!r}"
                )
        last = len(self.queries) - 1
        if self.forced_answers[last] is not None:
            raise ValueError(
                f"spec {self.name!r} forces the answer of its final query {self.queries[last]!r} "
                f"(position {last}) to {self.forced_answers[last]}; the final query is the scored one and "
                "its answer is never fed to the model, so its forced_answer must be null"
            )

    def _validate_starts(self) -> None:
        """The start half of the contract: both tuples together or neither, one form per position."""
        if bool(self.start_durations) != bool(self.start_events):
            have, lack = (
                ("start_durations", "start_events")
                if self.start_durations
                else ("start_events", "start_durations")
            )
            raise ValueError(f"spec {self.name!r} has {have} but not {lack}; give both or neither")
        if not self.start_durations:
            return
        if len(self.start_durations) != len(self.queries) or len(self.start_events) != len(self.queries):
            raise ValueError(
                f"spec {self.name!r} has {len(self.queries)} queries but {len(self.start_durations)} start "
                f"duration(s) / {len(self.start_events)} start event(s)"
            )
        for i, (sd, se) in enumerate(zip(self.start_durations, self.start_events, strict=True)):
            if se is not None and (not isinstance(se, str) or not se):
                raise ValueError(
                    f"spec {self.name!r} start event at position {i} must be a non-empty string or null"
                )
            if isinstance(sd, bool) or not isinstance(sd, int | float):
                raise TypeError(
                    f"spec {self.name!r} start duration at position {i} must be a number, "
                    f"got {type(sd).__name__}: {sd!r}"
                )
            if se is not None:
                if float(sd) != EVENT_BOUND_DURATION_SENTINEL:
                    raise ValueError(
                        f"spec {self.name!r} position {i} starts at {se!r}, so its start duration must be "
                        f"the {EVENT_BOUND_DURATION_SENTINEL} sentinel, got {float(sd)}"
                    )
                continue
            if not math.isfinite(sd) or sd < 0:
                raise ValueError(
                    f"spec {self.name!r} start duration {float(sd)} at position {i} must be a finite "
                    f"number >= 0 (or the {EVENT_BOUND_DURATION_SENTINEL} sentinel with a start_event)"
                )

    def bound_at(self, i: int) -> str | None:
        """Boundary code at position ``i``, or ``None`` when that position is time-bounded."""
        return self.bounds[i] if self.bounds else None

    def start_at(self, i: int) -> tuple[float, str | None]:
        """``(start_duration_days, start_event)`` at position ``i``; ``(0.0, None)`` without starts."""
        if not self.start_durations:
            return (0.0, None)
        return (float(self.start_durations[i]), self.start_events[i])

    @property
    def has_active_starts(self) -> bool:
        """Whether any position opens somewhere other than the prediction time."""
        return any(
            se is not None or float(sd) > 0
            for sd, se in zip(self.start_durations, self.start_events, strict=True)
        )

    @property
    def has_forced_answers(self) -> bool:
        """Whether any position's conditioning answer is dictated rather than teacher-forced."""
        return any(fa is not None for fa in self.forced_answers)

    def forced_at(self, i: int) -> bool | None:
        """Forced answer at position ``i``, or ``None`` when the ground truth is teacher-forced."""
        return self.forced_answers[i] if self.forced_answers else None

    def __len__(self) -> int:
        return len(self.queries)


def _sanitise_names(specs: list[SequenceSpec]) -> list[SequenceSpec]:
    """Make every spec name filesystem-safe, rejecting post-sanitisation collisions."""
    seen: dict[str, str] = {}
    out: list[SequenceSpec] = []
    for spec in specs:
        safe = _SAFE_NAME_RE.sub("_", spec.name).strip("._") or "seq"
        if safe in seen:
            raise ValueError(
                f"spec names {seen[safe]!r} and {spec.name!r} both sanitise to {safe!r}; "
                f"rename one so per-spec output directories stay distinct."
            )
        seen[safe] = spec.name
        out.append(spec if safe == spec.name else dataclasses.replace(spec, name=safe))
    return out


def _make_spec(
    name: str,
    queries: Sequence[str],
    durations: Sequence[float],
    bounds: Sequence[str | None],
    start_durations: Sequence[float],
    start_events: Sequence[str | None],
    forced_answers: Sequence[bool | None],
) -> SequenceSpec:
    """Assemble a spec, keeping the optional tuples empty when every position is the default.

    An all-null ``bounds`` becomes ``()``, an all-``(0.0, None)`` start pair becomes ``((), ())`` and
    an all-null ``forced_answers`` becomes ``()``, so a designed file — which must spell every field
    out — that only ever writes the defaults is indistinguishable from a spec built without the
    feature.  That is what keeps the fingerprint and the output columns of a default grid unchanged.
    """
    active_start = any(
        se is not None or float(sd) != 0.0 for sd, se in zip(start_durations, start_events, strict=True)
    )
    return SequenceSpec(
        name=name,
        queries=tuple(queries),
        durations=tuple(durations),
        bounds=tuple(bounds) if any(b is not None for b in bounds) else (),
        start_durations=tuple(float(sd) for sd in start_durations) if active_start else (),
        start_events=tuple(start_events) if active_start else (),
        forced_answers=tuple(forced_answers) if any(fa is not None for fa in forced_answers) else (),
    )


#: Every designed entry must spell out all of these — ``null`` is a legal *value*, a missing key is
#: not.  A designed file is read by a collaborator's eyes as often as by this parser, and an entry
#: that names every field cannot be misread as "the default I did not know existed".
_MAPPING_ENTRY_KEYS = frozenset(
    {"query", "start_event", "start_duration_days", "bound_event", "duration_days", "forced_answer"}
)


def _entry_from_mapping(
    name: str, i: int, entry: dict
) -> tuple[str, float, str | None, float, str | None, bool | None]:
    """Read one fully spelled-out entry; see :data:`_MAPPING_ENTRY_KEYS`.

    Exactly one representation is active per window endpoint, and a ``null`` duration next to an
    event takes the sentinel that event implies, so a designed file never has to write ``-1``:

    ===========================================  ==================================================
    ``start_event`` / ``start_duration_days``    code + (null | -1), or null + days ``>= 0``
                                                 (``0`` / null = the prediction time)
    ``bound_event`` / ``duration_days``          code + (null | -1), or null + days ``> 0``
    ``forced_answer``                            true | false | null (= teacher-force the truth)
    ===========================================  ==================================================

    Examples:
        >>> full = {"query": "A", "start_event": None, "start_duration_days": 0, "bound_event": None,
        ...         "duration_days": 30, "forced_answer": True}
        >>> _entry_from_mapping("s", 0, full)
        ('A', 30, None, 0, None, True)
        >>> _entry_from_mapping("s", 0, full | {"bound_event": "DISCHARGE", "duration_days": None,
        ...                                     "start_event": "ADMIT", "start_duration_days": None})
        ('A', -1.0, 'DISCHARGE', -1.0, 'ADMIT', True)
        >>> _entry_from_mapping("s", 0, {"query": "A", "duration_days": 30})
        Traceback (most recent call last):
            ...
        ValueError: sequence 's' entry 0 is missing required key(s) ['bound_event', 'forced_answer',
        'start_duration_days', 'start_event']; every entry must spell out all of ['bound_event',
        'duration_days', 'forced_answer', 'query', 'start_duration_days', 'start_event'] (null is a
        legal value): {'query': 'A', 'duration_days': 30}
        >>> _entry_from_mapping("s", 0, full | {"duration_days": None})
        Traceback (most recent call last):
            ...
        ValueError: sequence 's' entry 0 has neither a duration_days nor a bound_event, so its window
        has no end
    """
    unknown = sorted(set(entry) - _MAPPING_ENTRY_KEYS)
    if unknown:
        raise ValueError(
            f"sequence {name!r} entry {i} has unknown key(s) {unknown}; "
            f"allowed: {sorted(_MAPPING_ENTRY_KEYS)}"
        )
    missing = sorted(_MAPPING_ENTRY_KEYS - set(entry))
    if missing:
        raise ValueError(
            f"sequence {name!r} entry {i} is missing required key(s) {missing}; every entry must spell "
            f"out all of {sorted(_MAPPING_ENTRY_KEYS)} (null is a legal value): {entry!r}"
        )
    bound = entry["bound_event"]
    start_event = entry["start_event"]
    # A null duration takes the form its event implies — the sentinel next to an event — the same
    # rule the long-format parquet reader applies.  A null start with no event is the prediction
    # time; a null end with no event is no window at all.
    start_duration = entry["start_duration_days"]
    if start_duration is None:
        start_duration = EVENT_BOUND_DURATION_SENTINEL if start_event is not None else 0.0
    duration = entry["duration_days"]
    if duration is None:
        if bound is None:
            raise ValueError(
                f"sequence {name!r} entry {i} has neither a duration_days nor a bound_event, so its "
                "window has no end"
            )
        duration = EVENT_BOUND_DURATION_SENTINEL
    return entry["query"], duration, bound, start_duration, start_event, entry["forced_answer"]


def _spec_from_entries(name: str, entries: object) -> SequenceSpec:
    """Build one spec from a list of fully spelled-out mapping entries::

        - query: TIMELINE//END
          start_event: null
          start_duration_days: 0             # opens at the prediction time ...
          bound_event: null
          duration_days: 30                  # ... and closes 30 days later
          forced_answer: false               # condition the next query on "record did NOT end"
        - query: PROCEDURE//X
          start_event: HOSPITAL_ADMISSION    # opens at the next admission ...
          start_duration_days: null
          bound_event: HOSPITAL_DISCHARGE    # ... closes at the next discharge after it
          duration_days: null
          forced_answer: null                # the final query is scored, never forced

    Every key is required in every entry (see :func:`_entry_from_mapping`); the ``[code, duration]``
    list shorthand this reader once took is rejected, since it cannot say what it leaves out.
    """
    if not isinstance(entries, Sequence) or isinstance(entries, str):
        raise ValueError(
            f"sequence {name!r} must be a list of {{{', '.join(sorted(_MAPPING_ENTRY_KEYS))}}} "
            f"mappings, got {entries!r}"
        )
    columns: tuple[list, ...] = ([], [], [], [], [], [])
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(
                f"sequence {name!r} entry {i} must be a mapping with all of "
                f"{sorted(_MAPPING_ENTRY_KEYS)} (null is a legal value), got {entry!r}"
            )
        for column, value in zip(columns, _entry_from_mapping(name, i, entry), strict=True):
            column.append(value)
    return _make_spec(name, *columns)


def read_sequence_specs(path: str | Path) -> list[SequenceSpec]:
    """Read designed query sequences from a YAML/JSON or parquet file.

    YAML/JSON is a mapping of ``name -> [entry, ...]``, every entry a mapping that spells out **all
    six keys** (``null`` is a legal value; a missing key is an error — see
    :func:`_entry_from_mapping` for the value rules)::

        death_given_not_censored:
          - {query: TIMELINE//END, start_event: null, start_duration_days: 0,
             bound_event: null, duration_days: 30, forced_answer: false}
          - {query: MEDS_DEATH, start_event: null, start_duration_days: 0,
             bound_event: null, duration_days: 30, forced_answer: null}

    or a bare list of such sequences, in which case names are generated (``seq_0000``, ...).

    ``forced_answer`` dictates the answer fed to the model for that query when it conditions a later
    one; ``null`` teacher-forces the labeled truth.  It must be ``null`` on a sequence's final query.

    A parquet is read as long-format ``(seq_id, position, query, start_event, start_duration_days,
    bound_event, duration_days, forced_answer)``, one row per query and **every column required**;
    ``seq_id`` becomes the spec name and ``position`` fixes the within-sequence order.  Nulls mean
    what they mean in the mapping form.  That form is the convenient one when the specs are
    themselves generated by a script.

    Raises:
        ValueError: If the file is empty, has an unsupported suffix, or any sequence is malformed.
    """
    p = Path(path)
    if p.suffix == ".parquet":
        df = pl.read_parquet(p)
        # Accept both ``position`` and the internal ``_position`` spelling, so a frame dumped
        # straight out of ``build_dense_sequence_index_df`` can be fed back in.
        pos_col = "position" if "position" in df.columns else POSITION_COL
        required = {
            "seq_id",
            pos_col,
            "query",
            "duration_days",
            BOUND_COL,
            START_EVENT_COL,
            START_DURATION_COL,
            FORCED_ANSWER_COL,
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{p} is missing required column(s) {sorted(missing)}")
        grouped = (
            df.select(
                pl.col("seq_id").cast(pl.Utf8),
                pl.col(pos_col).cast(pl.Int64),
                pl.col("query").cast(pl.Utf8),
                pl.col("duration_days").cast(pl.Float64),
                pl.col(BOUND_COL).cast(pl.Utf8),
                pl.col(START_EVENT_COL).cast(pl.Utf8),
                pl.col(START_DURATION_COL).cast(pl.Float64),
                pl.col(FORCED_ANSWER_COL).cast(pl.Boolean),
            )
            .with_columns(
                # A null duration takes the form its event implies (see the mapping reader): the
                # sentinel next to an event; a null start with no event is the prediction time.
                pl.col(START_DURATION_COL).fill_null(
                    pl.when(pl.col(START_EVENT_COL).is_not_null())
                    .then(pl.lit(EVENT_BOUND_DURATION_SENTINEL))
                    .otherwise(pl.lit(0.0))
                ),
                pl.col("duration_days").fill_null(
                    pl.when(pl.col(BOUND_COL).is_not_null()).then(pl.lit(EVENT_BOUND_DURATION_SENTINEL))
                ),
            )
            .sort("seq_id", pos_col)
            .group_by("seq_id", maintain_order=True)
            .agg(
                pl.col("query"),
                pl.col("duration_days"),
                pl.col(BOUND_COL),
                pl.col(START_DURATION_COL),
                pl.col(START_EVENT_COL),
                pl.col(FORCED_ANSWER_COL),
            )
        )
        specs = [
            _make_spec(
                row["seq_id"],
                row["query"],
                row["duration_days"],
                row[BOUND_COL],
                row[START_DURATION_COL],
                row[START_EVENT_COL],
                row[FORCED_ANSWER_COL],
            )
            for row in grouped.iter_rows(named=True)
        ]
    elif p.suffix in {".yaml", ".yml", ".json"}:
        if p.suffix == ".json":
            data = json.loads(p.read_text())
        else:
            import yaml

            data = yaml.safe_load(p.read_text())
        if isinstance(data, dict) and "sequences" in data:
            data = data["sequences"]
        if isinstance(data, dict):
            specs = [_spec_from_entries(str(name), entries) for name, entries in data.items()]
        elif isinstance(data, list):
            specs = [_spec_from_entries(f"seq_{i:04d}", entries) for i, entries in enumerate(data)]
        else:
            raise ValueError(
                f"{p} must contain a mapping of name -> [entry, ...] or a list of such sequences, "
                f"got {type(data).__name__}"
            )
    else:
        raise ValueError(f"{p} must be a .yaml/.yml/.json or .parquet file (got suffix {p.suffix!r})")

    if not specs:
        raise ValueError(f"{p} contains no sequences")
    return _sanitise_names(specs)


_VALID_START_DISTRIBUTIONS = ("uniform", "log-uniform")


def _validate_start_sampling(
    eventstart_fraction: float,
    prediction_time_start_fraction: float,
    start_duration_min: float,
    start_duration_max: float,
    start_duration_distribution: str,
    start_event_codes: Sequence[str],
    universe: Collection[str],
) -> None:
    """Mirror ``BoundaryDistribution.__post_init__``'s start checks for the eval grid's start draw."""
    if not 0.0 <= eventstart_fraction <= 1.0:
        raise ValueError(f"eventstart_fraction must be in [0, 1] (got {eventstart_fraction})")
    if not 0.0 <= prediction_time_start_fraction <= 1.0:
        raise ValueError(
            f"prediction_time_start_fraction must be in [0, 1] (got {prediction_time_start_fraction})"
        )
    if eventstart_fraction + prediction_time_start_fraction > 1.0:
        raise ValueError(
            "eventstart_fraction + prediction_time_start_fraction must be <= 1 (got "
            f"{eventstart_fraction} + {prediction_time_start_fraction})"
        )
    if eventstart_fraction > 0 and not start_event_codes:
        raise ValueError("eventstart_fraction > 0 requires a non-empty start_event_codes pool")
    if start_duration_min <= 0:
        raise ValueError(f"start_duration_min must be > 0 (got {start_duration_min})")
    if start_duration_max < start_duration_min:
        raise ValueError(
            f"start_duration_max ({start_duration_max}) must be >= start_duration_min ({start_duration_min})"
        )
    if start_duration_distribution not in _VALID_START_DISTRIBUTIONS:
        raise ValueError(
            f"start_duration_distribution must be one of {_VALID_START_DISTRIBUTIONS} "
            f"(got {start_duration_distribution!r})"
        )
    unknown = sorted(set(start_event_codes) - set(universe))
    if unknown:
        raise ValueError(
            f"{len(unknown)} start_event_codes are outside the query universe: {unknown[:10]}; "
            "start events must be codes the model can encode."
        )


def sample_sequence_specs(
    n_sequences: int,
    query_codes: list[str],
    min_queries: int,
    max_queries: int,
    duration_low: float,
    duration_high: float,
    seed: int,
    eos_first_fraction: float = 0.0,
    duration_mode: str = "random",
    duration_distribution: str = "log-uniform",
    eventbound_fraction: float = 0.0,
    eventstart_fraction: float = 0.0,
    prediction_time_start_fraction: float = 1.0,
    start_duration_min: float = 1.0,
    start_duration_max: float = 180.0,
    start_duration_distribution: str = "log-uniform",
    start_event_codes: Sequence[str] | None = None,
) -> list[SequenceSpec]:
    """Draw ``n_sequences`` specs from the *training* query distribution, once.

    Delegates to
    :class:`~every_query.generate_tasks.query_sequence_labeling.QuerySequenceDistribution` on the
    same three seed axes the scattered sampler uses, so the code/duration/bound draw is
    byte-for-byte the one training saw — an evaluation grid drawn from a subtly different
    distribution than training is the kind of drift that shows up as unexplained metric shifts.
    No contexts are involved: these specs are then applied to *every* real context.

    **Window starts (issue #27) and the parity decision.**  The query / duration / end-bound draw
    above stays parity-anchored to ``query_sequence_labeling.py``.  Only the *start* component
    mirrors the multitask sampler (``BoundaryDistribution.sample``): per query one uniform ``u``
    picks the form — ``u < eventstart_fraction`` an event-defined start (code iid uniform over
    ``start_event_codes``, ``None`` = the whole query universe), ``u < eventstart_fraction +
    prediction_time_start_fraction`` the prediction time itself, otherwise a positive
    ``start_duration_min..max`` delay from ``start_duration_distribution`` — and every start stream
    is drawn in full before masking.  The three start axes (``derive_seed(seed, "start_forms")``,
    ``"start_durations"``, ``"start_codes"``) are independent of the three legacy axes, so changing
    any start setting perturbs neither the lengths, the codes, the horizons nor the end bounds, and
    the defaults (``eventstart_fraction=0``, ``prediction_time_start_fraction=1``) reproduce the
    pre-#27 specs exactly.  This does **not** claim the complete sampled sequence equals a
    ``sample_multitask_sequences.py`` draw; controlled multitask evaluations use ``sequences_path``.

    Examples:
        >>> specs = sample_sequence_specs(3, ["A", "B", "TIMELINE//END"], 2, 2, 1, 365, seed=0)
        >>> len(specs)
        3
        >>> [len(s) for s in specs]
        [2, 2, 2]
        >>> all(q in {"A", "B", "TIMELINE//END"} for s in specs for q in s.queries)
        True

        Determinism, and independence from the cohort it will be applied to:

        >>> sample_sequence_specs(3, ["A", "B"], 1, 3, 1, 365, 7) == sample_sequence_specs(
        ...     3, ["A", "B"], 1, 3, 1, 365, 7)
        True

        ``eos_first_fraction=1.0`` forces position 0 to the end-of-timeline query:

        >>> specs = sample_sequence_specs(
        ...     2, ["A", "B", "TIMELINE//END"], 2, 2, 1, 365, 0, eos_first_fraction=1.0)
        >>> {s.queries[0] for s in specs}
        {'TIMELINE//END'}

        ``eventbound_fraction`` converts that share of queries into event-bounded ones, carrying a
        boundary drawn from the same universe and the duration sentinel instead of a horizon:

        >>> (spec,) = sample_sequence_specs(1, ["A", "B"], 4, 4, 1, 365, 0, eventbound_fraction=1.0)
        >>> set(spec.bounds) <= {"A", "B"}, set(spec.durations)
        (True, {-1.0})

        Starts are a separate component on their own seed axes: turning them on leaves the draw
        above untouched and only adds the start tuples.

        >>> (started,) = sample_sequence_specs(1, ["A", "B"], 4, 4, 1, 365, 0, eventbound_fraction=1.0,
        ...     eventstart_fraction=0.5, prediction_time_start_fraction=0.0, start_event_codes=["B"])
        >>> (started.queries, started.durations, started.bounds) == (
        ...     spec.queries, spec.durations, spec.bounds)
        True
        >>> started.has_active_starts, set(started.start_events) <= {"B", None}
        (True, True)
    """
    if n_sequences < 1:
        raise ValueError(f"n_sequences must be >= 1 (got {n_sequences})")
    start_pool = list(query_codes) if start_event_codes is None else list(start_event_codes)
    _validate_start_sampling(
        eventstart_fraction,
        prediction_time_start_fraction,
        float(start_duration_min),
        float(start_duration_max),
        start_duration_distribution,
        start_pool,
        query_codes,
    )

    dist = QuerySequenceDistribution(
        query_codes=list(query_codes),
        min_duration=float(duration_low),
        max_duration=float(duration_high),
        duration_distribution=duration_distribution,
        min_queries=min_queries,
        max_queries=max_queries,
        eos_first_fraction=eos_first_fraction,
        duration_mode=duration_mode,
        eventbound_fraction=eventbound_fraction,
    )
    sequences = dist.sample_sequences(
        n_sequences,
        np.random.default_rng(derive_seed(seed, "queries")),
        np.random.default_rng(derive_seed(seed, "sequences")),
        np.random.default_rng(derive_seed(seed, "bounds")),
    )
    start_durations, start_events = _sample_starts(
        total=sum(len(s) for s in sequences),
        pool=start_pool,
        eventstart_fraction=eventstart_fraction,
        prediction_time_start_fraction=prediction_time_start_fraction,
        start_duration_min=float(start_duration_min),
        start_duration_max=float(start_duration_max),
        start_duration_distribution=start_duration_distribution,
        form_rng=np.random.default_rng(derive_seed(seed, "start_forms")),
        duration_rng=np.random.default_rng(derive_seed(seed, "start_durations")),
        code_rng=np.random.default_rng(derive_seed(seed, "start_codes")),
    )
    specs = []
    offset = 0
    for i, seq in enumerate(sequences):
        n = len(seq)
        specs.append(
            _make_spec(
                f"seq_{i:04d}",
                [q.code for q in seq],
                [float(q.duration_days) for q in seq],
                [q.bound_event for q in seq],
                start_durations[offset : offset + n].tolist(),
                start_events[offset : offset + n].tolist(),
                [None] * n,  # forced answers are a designed-spec feature; sampled specs never force
            )
        )
        offset += n
    return specs


def _sample_starts(
    total: int,
    pool: Sequence[str],
    eventstart_fraction: float,
    prediction_time_start_fraction: float,
    start_duration_min: float,
    start_duration_max: float,
    start_duration_distribution: str,
    form_rng: np.random.Generator,
    duration_rng: np.random.Generator,
    code_rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """The multitask sampler's start draw for ``total`` flat query slots (see ``BoundaryDistribution``).

    Returns ``(start_durations float32, start_events object)``; every stream is drawn in full over
    all ``total`` slots and masked afterwards, so the realised form of one slot never shifts the
    duration or code drawn for another.

    Examples:
        >>> rngs = lambda: [np.random.default_rng(i) for i in range(3)]
        >>> d, e = _sample_starts(6, ["A", "B"], 0.5, 0.0, 1.0, 10.0, "uniform", *rngs())
        >>> d.shape, d.dtype, e.shape
        ((6,), dtype('float32'), (6,))
        >>> bool(((d == -1.0) == (e != None)).all()), bool((d[e == None] >= 1.0).all())  # noqa: E711
        (True, True)
        >>> d0, e0 = _sample_starts(6, ["A", "B"], 0.0, 1.0, 1.0, 10.0, "uniform", *rngs())
        >>> bool((d0 == 0.0).all()) and all(x is None for x in e0)
        True
    """
    u = form_rng.random(total)
    is_event = u < eventstart_fraction
    is_pt = ~is_event & (u < eventstart_fraction + prediction_time_start_fraction)
    if start_duration_distribution == "log-uniform":
        durations = np.exp(
            duration_rng.uniform(np.log(start_duration_min), np.log(start_duration_max), total)
        ).astype(np.float32)
    else:
        durations = duration_rng.uniform(start_duration_min, start_duration_max, total).astype(np.float32)
    events = np.full(total, None, dtype=object)
    if pool:
        picks = code_rng.integers(0, len(pool), size=total)
        events[is_event] = np.array(pool, dtype=object)[picks[is_event]]
    durations[is_pt] = 0.0
    durations[is_event] = EVENT_BOUND_DURATION_SENTINEL
    return durations, events


def build_dense_sequence_index_df(
    contexts: pl.DataFrame,
    specs: list[SequenceSpec],
) -> pl.DataFrame:
    """Cross-join every context with every spec into the flat per-query index frame.

    The output is shaped exactly like ``build_sequence_index``'s — ``(_ctx_id, _position,
    subject_id, prediction_time, query, duration_days)`` — so
    :func:`~every_query.generate_tasks.query_sequence_labeling.label_binary_occurrence` consumes it
    unchanged.  ``_ctx_id = context_row * len(specs) + spec_index``, so sorting by ``_ctx_id`` puts
    the frame in context-major order: labeled row ``i`` is ``(contexts[i // N], specs[i % N])``.

    Args:
        contexts: ``(subject_id, prediction_time)`` rows; extra columns are ignored.
        specs: The ``N`` sequences to evaluate at every context.

    Returns:
        DataFrame with ``contexts.height * sum(len(s) for s in specs)`` rows.

    Examples:
        >>> from datetime import datetime
        >>> ctx = pl.DataFrame({
        ...     "subject_id": [1, 2],
        ...     "prediction_time": [datetime(2024, 1, 1), datetime(2024, 2, 1)],
        ... }, schema={"subject_id": pl.Int64, "prediction_time": pl.Datetime("us")})
        >>> specs = [
        ...     SequenceSpec("a", ("TIMELINE//END", "X"), (1.0, 30.0)),
        ...     SequenceSpec("b", ("Y",), (7.0,)),
        ... ]
        >>> idx = build_dense_sequence_index_df(ctx, specs)
        >>> idx.height  # 2 contexts x (2 + 1) queries
        6
        >>> idx.columns
        ['_ctx_id', '_position', 'subject_id', 'prediction_time', 'query', 'duration_days']

        Every context gets every spec, and ``_ctx_id`` is context-major:

        >>> idx.select("_ctx_id", "subject_id", "query").rows()
        [(0, 1, 'TIMELINE//END'), (0, 1, 'X'), (1, 1, 'Y'), (2, 2, 'TIMELINE//END'), (2, 2, 'X'), (3, 2, 'Y')]

        Empty cohort yields an empty frame with the right schema:

        >>> build_dense_sequence_index_df(ctx.head(0), specs).height
        0
    """
    if not specs:
        raise ValueError("specs must be non-empty")

    sid = TaskQuerySchema.subject_id_name
    pt = TaskQuerySchema.prediction_time_name
    q = TaskQuerySchema.query_name
    d = TaskQuerySchema.duration_days_name

    # The bound column appears only when some spec is event-bounded, so a bound-free grid stays
    # byte-identical to one built before the feature existed; the two start columns likewise appear
    # only when some spec opens a window away from the prediction time (issue #27).
    any_bounds = any(s.bounds for s in specs)
    any_starts = any(s.has_active_starts for s in specs)

    out_schema = {
        CTX_ID_COL: pl.UInt32,
        POSITION_COL: pl.Int64,
        sid: contexts.schema.get(sid, pl.Int64),
        pt: contexts.schema.get(pt, pl.Datetime("us")),
        q: pl.Utf8,
        d: pl.Float32,
    }
    if any_bounds:
        out_schema[BOUND_COL] = pl.Utf8
    if any_starts:
        out_schema[START_DURATION_COL] = pl.Float32
        out_schema[START_EVENT_COL] = pl.Utf8
    if contexts.height == 0:
        return pl.DataFrame(schema=out_schema)

    n_specs = len(specs)
    n_sequences = contexts.height * n_specs
    # ``_ctx_id`` is UInt32 to match ``build_sequence_index`` (whose ids come from
    # ``with_row_index``); a silently-wrapped id would merge two contexts' sequences into one
    # labeled row, so check rather than trusting the cast.
    if n_sequences > (1 << 32) - 1:
        raise ValueError(
            f"{contexts.height} contexts x {n_specs} specs = {n_sequences} sequences exceeds the "
            f"UInt32 _ctx_id range; shard the cohort across multiple runs."
        )

    spec_cols = {
        SPEC_ID_COL: [i for i, s in enumerate(specs) for _ in s.queries],
        POSITION_COL: [p for s in specs for p in range(len(s))],
        q: [c for s in specs for c in s.queries],
        d: [float(dd) for s in specs for dd in s.durations],
    }
    spec_schema = {SPEC_ID_COL: pl.Int64, POSITION_COL: pl.Int64, q: pl.Utf8, d: pl.Float32}
    if any_bounds:
        spec_cols[BOUND_COL] = [s.bound_at(p) for s in specs for p in range(len(s))]
        spec_schema[BOUND_COL] = pl.Utf8
    if any_starts:
        spec_cols[START_DURATION_COL] = [s.start_at(p)[0] for s in specs for p in range(len(s))]
        spec_cols[START_EVENT_COL] = [s.start_at(p)[1] for s in specs for p in range(len(s))]
        spec_schema[START_DURATION_COL] = pl.Float32
        spec_schema[START_EVENT_COL] = pl.Utf8

    spec_frame = pl.DataFrame(spec_cols, schema=spec_schema)

    return (
        contexts.select(sid, pt)
        # Cast to Int64 before the multiply: ``with_row_index`` yields UInt32, and
        # ``row * n_specs`` on UInt32 would wrap before the range check above could help.
        .with_row_index("_row")
        .with_columns(pl.col("_row").cast(pl.Int64))
        .join(spec_frame, how="cross")
        .with_columns((pl.col("_row") * n_specs + pl.col(SPEC_ID_COL)).cast(pl.UInt32).alias(CTX_ID_COL))
        .select(list(out_schema))
        .sort(CTX_ID_COL, POSITION_COL)
    )


def resolve_specs(
    sequences_path: str | Path | None,
    query_codes: list[str],
    n_sequences: int,
    min_queries: int,
    max_queries: int,
    duration_min: float,
    duration_max: float,
    seed: int,
    eos_first_fraction: float = 0.0,
    duration_mode: str = "random",
    duration_distribution: str = "log-uniform",
    eventbound_fraction: float = 0.5,
    eventstart_fraction: float = 0.0,
    prediction_time_start_fraction: float = 1.0,
    start_duration_min: float = 1.0,
    start_duration_max: float = 180.0,
    start_duration_distribution: str = "log-uniform",
    start_event_codes: Sequence[str] | None = None,
) -> list[SequenceSpec]:
    """Read designed specs from ``sequences_path``, or sample them when it is ``None``."""
    if sequences_path is not None:
        specs = read_sequence_specs(sequences_path)
        logger.info("Read %d designed query sequence(s) from %s", len(specs), sequences_path)
        return specs
    specs = sample_sequence_specs(
        n_sequences=n_sequences,
        query_codes=query_codes,
        min_queries=min_queries,
        max_queries=max_queries,
        duration_low=duration_min,
        duration_high=duration_max,
        seed=seed,
        eos_first_fraction=eos_first_fraction,
        duration_mode=duration_mode,
        duration_distribution=duration_distribution,
        eventbound_fraction=eventbound_fraction,
        eventstart_fraction=eventstart_fraction,
        prediction_time_start_fraction=prediction_time_start_fraction,
        start_duration_min=start_duration_min,
        start_duration_max=start_duration_max,
        start_duration_distribution=start_duration_distribution,
        start_event_codes=start_event_codes,
    )
    n_started = sum(s.has_active_starts for s in specs)
    logger.info(
        "Sampled %d query sequence(s) from the training query distribution (%d with a window start "
        "away from the prediction time)",
        len(specs),
        n_started,
    )
    return specs


def model_query_vocab(query_codes: Sequence[str], ontology_dir: str | Path | None) -> set[str]:
    """Every name ``QuerySeqPytorchDataset.encode_query`` will accept for this run.

    Built by the same :func:`~every_query.data.ontology.extend_code_map` the dataset uses, so a
    spec that validates here is a spec the dataset can encode — and vice versa.  Note this is
    *wider* than the sampling universe from ``build_query_universe`` (which drops ``TIMELINE``
    ancestors): a designed spec does not sample, it only needs the name to resolve.
    """
    if not ontology_dir:
        return set(query_codes)

    from every_query.data.ontology import extend_code_map

    return set(extend_code_map(dict.fromkeys(query_codes, 0), ontology_dir))


def validate_spec_codes(specs: list[SequenceSpec], vocab: Collection[str]) -> None:
    """Fail fast on spec codes outside the model's query vocabulary.

    ``QuerySeqPytorchDataset.encode_query`` raises ``KeyError`` on an unknown code, which
    surfaces deep inside ``collate`` partway through an inference run — long after the minutes of
    labeling and model loading this check precedes.  Hand-written designed specs are exactly where
    a typo'd or wrong-vocabulary MEDS code enters.
    """
    # Boundary and start events are checked too — an event the encoder never saw cannot define a
    # window, and an unchecked one would surface as a KeyError deep inside collate, after the
    # labeling and model load.
    mentioned = [q for s in specs for q in s.queries]
    mentioned += [b for s in specs for b in s.bounds if b is not None]
    mentioned += [e for s in specs for e in s.start_events if e is not None]
    unknown = sorted(set(mentioned) - set(vocab))
    if unknown:
        shown = ", ".join(repr(c) for c in unknown[:10])
        more = f" (and {len(unknown) - 10} more)" if len(unknown) > 10 else ""
        raise ValueError(
            f"{len(unknown)} spec query/start/bound code(s) are absent from the query vocabulary: "
            f"{shown}{more}. Check that `query_codes` points at the same codes.parquet the model was "
            "trained on."
        )


# ---------------------------------------------------------------------------
# The cohort
# ---------------------------------------------------------------------------


def read_supplied_contexts(contexts_path: str | Path) -> pl.DataFrame:
    """Read a supplied ``(subject_id, prediction_time)`` cohort parquet, deduplicated.

    Extra columns are dropped; ``prediction_time`` is cast to ``Datetime("us")`` for the same reason
    ``_read_event_shard`` casts event times (the ``+1us`` strict-``>`` shift in
    :func:`label_binary_occurrence` rounds to zero at millisecond precision).  Duplicate rows are
    dropped with a warning: a dense grid asks every sequence about every context once, and a
    duplicated cohort row would silently double-weight that patient in the metrics.

    Raises:
        ValueError: If a required column is missing, or the file has no rows.
    """
    sid = TaskQuerySchema.subject_id_name
    pt = TaskQuerySchema.prediction_time_name

    df = pl.read_parquet(contexts_path)
    missing = {sid, pt} - set(df.columns)
    if missing:
        raise ValueError(f"{contexts_path} is missing required column(s) {sorted(missing)}")
    contexts = df.select(pl.col(sid).cast(pl.Int64), pl.col(pt).cast(pl.Datetime("us")))
    unique = contexts.unique(maintain_order=True)
    if unique.height < contexts.height:
        logger.warning(
            "Dropped %d duplicate (subject_id, prediction_time) row(s) from %s; %d unique contexts remain.",
            contexts.height - unique.height,
            contexts_path,
            unique.height,
        )
    if unique.height == 0:
        raise ValueError(f"{contexts_path} has no rows; nothing to evaluate.")
    return unique


def assert_subjects_in_split(data_dir: Path, split: str, shards: list[str], subjects: pl.Series) -> None:
    """Fail fast if any supplied-cohort subject has no shard in ``split``.

    Silent here would mean all-``False`` answers now and a silent row-drop in
    ``EQ_predict_multitask`` later (the dataset's schema_df semi-join drops subjects absent from the
    split without erroring).  Only the ``subject_id`` column is scanned, so this is cheap even on a real
    split, and it runs before any shard is labeled rather than after the last one.
    """
    sid = TaskQuerySchema.subject_id_name
    present = pl.concat(
        [
            pl.scan_parquet(data_dir / "data" / split / f"{shard}.parquet").select(pl.col(sid).cast(pl.Int64))
            for shard in shards
        ]
    ).collect()[sid]
    wanted = subjects.unique()
    missing = wanted.filter(~wanted.is_in(present.implode()))
    if missing.len():
        raise ValueError(
            f"{missing.len()} of {subjects.n_unique()} supplied subjects have no events under "
            f"{data_dir / 'data' / split} (e.g. {missing.head(5).to_list()}); "
            f"check that `split` matches the cohort."
        )


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def _frame_digest(df: pl.DataFrame) -> str:
    """Serialization-independent digest of a frame's logical rows: ``"{height}:{hash16}"``.

    :func:`every_query.utils.digest.frame_digest` — the same construction
    :func:`~every_query.generate_tasks.sample_tasks._index_fingerprint` uses for Stage 4's index
    partitions, and the one a checkpoint's closure check digests with.
    """
    return frame_digest(df)


def _ontology_fingerprint(ontology_dir: str | Path | None, query_codes: Sequence[str]) -> str | None:
    """Digest of the two ontology-derived inputs to a grid's labels, or ``None`` with no ontology.

    The output path encodes the shard, never the ontology, so a leaf-only grid and an ancestor grid
    land on the same file and an existence-only skip would silently keep the first one's labels.
    The two things that would make those labels wrong are covered here:

    - the **closure**, which decides whether an ancestor query labels ``True`` at all
      (:func:`~every_query.data.ontology.closure_fingerprint`, the same digest
      ``EQ_predict_multitask`` compares a checkpoint's ontology against), and
    - the **query universe**, which is where the ontology's ancestor nodes (and the leaf
      vocabulary) land after
      :func:`~every_query.generate_tasks.query_sequence_labeling.build_query_universe`; the
      universe is digested with its slot index, since order steers the draw.

    The two halves are joined by ``"|"``; :func:`split_ontology_fingerprint` takes them apart.
    ``None`` when no ontology is configured; it is one component of :func:`_run_fingerprint`.
    """
    if not ontology_dir:
        return None
    from every_query.data.ontology import closure_fingerprint

    universe = pl.DataFrame(
        {"slot": list(range(len(query_codes))), "code": list(query_codes)},
        schema={"slot": pl.Int64, "code": pl.String},
    )
    return f"{closure_fingerprint(ontology_dir)}|{_frame_digest(universe)}"


def split_ontology_fingerprint(fingerprint: str | None) -> tuple[str | None, str | None]:
    """``(closure, universe)`` halves of a recorded ``ontology_fingerprint``; ``(None, None)`` for ``None``.

    Examples:
        >>> split_ontology_fingerprint("3:00ab|5:00cd")
        ('3:00ab', '5:00cd')
        >>> split_ontology_fingerprint(None)
        (None, None)
    """
    if fingerprint is None:
        return None, None
    closure, _, universe = str(fingerprint).partition("|")
    return closure, universe or None


def _specs_fingerprint(specs: list[SequenceSpec]) -> str:
    """Digest of the specs' *content* — ``(queries, durations, bounds)`` in order — never their names.

    Names do not reach the output (``QuerySeqSchema`` has no spec-name column), so two runs whose
    specs differ only by name write identical grids and must fingerprint identically.  Window starts
    (issue #27) join the payload only when some spec has an active one, so a start-free run keeps
    the fingerprint it had before starts existed and its previously written shards stay current.

    Examples:
        >>> a = [SequenceSpec("x", ("A", "B"), (1.0, 30.0))]
        >>> _specs_fingerprint(a) == _specs_fingerprint([SequenceSpec("renamed", ("A", "B"), (1.0, 30.0))])
        True
        >>> _specs_fingerprint(a) == _specs_fingerprint([SequenceSpec("x", ("A", "B"), (1.0, 31.0))])
        False

        Spelling the default start out changes nothing; an active start does:

        >>> default = [SequenceSpec("x", ("A", "B"), (1.0, 30.0), start_durations=(0.0, 0.0),
        ...                         start_events=(None, None))]
        >>> _specs_fingerprint(a) == _specs_fingerprint(default)
        True
        >>> delayed = [SequenceSpec("x", ("A", "B"), (1.0, 30.0), start_durations=(0.0, 7.0),
        ...                         start_events=(None, None))]
        >>> _specs_fingerprint(a) == _specs_fingerprint(delayed)
        False

        Forced answers join the same way — only when some spec has one — so flipping one relabels
        instead of serving the shard written under the other value:

        >>> unforced = [SequenceSpec("x", ("A", "B"), (1.0, 30.0), forced_answers=(None, None))]
        >>> _specs_fingerprint(a) == _specs_fingerprint(unforced)
        True
        >>> yes, no = ([SequenceSpec("x", ("A", "B"), (1.0, 30.0), forced_answers=(fa, None))]
        ...            for fa in (True, False))
        >>> len({_specs_fingerprint(a), _specs_fingerprint(yes), _specs_fingerprint(no)})
        3
    """
    payload = [[list(s.queries), list(s.durations), list(s.bounds)] for s in specs]
    if any(s.has_active_starts for s in specs):
        for entry, s in zip(payload, specs, strict=True):
            entry.append([[s.start_at(i)[0], s.start_at(i)[1]] for i in range(len(s))])
    if any(s.has_forced_answers for s in specs):
        for entry, s in zip(payload, specs, strict=True):
            # "forced_cohort", not "forced_answers": renamed when a forced answer became a cohort
            # filter, so a dense forced grid written before that reads as stale and is relabeled.
            entry.append({"forced_cohort": [s.forced_at(i) for i in range(len(s))]})
    return f"{len(specs)}:{hashlib.sha256(json.dumps(payload).encode()).hexdigest()[:16]}"


def _cohort_fingerprint(
    contexts: pl.DataFrame | None,
    seed: int,
    prediction_times_per_subject: int,
    min_context_per_subject: int,
    subject_subsample_fraction: float | None,
) -> str:
    """Digest of whatever decides the cohort: the supplied frame, or the sampling knobs.

    A supplied cohort is digested whole (:func:`_frame_digest`, order-independent), so editing the
    cohort parquet relabels every shard.  A sampled cohort is a pure function of the shard's events
    and ``(seed, K, min_context, fraction)`` — the same axes ``run_worker`` derives its seeds from —
    so those are what is recorded.

    Examples:
        >>> _cohort_fingerprint(None, 1, 2, 50, None)
        'sampled|seed=1|k=2|min_context=50|fraction=None'
        >>> ctx = pl.DataFrame({"subject_id": [1], "prediction_time": [0]})
        >>> _cohort_fingerprint(ctx, 1, 2, 50, None).startswith("supplied|")
        True
    """
    if contexts is not None:
        return f"supplied|{_frame_digest(contexts)}"
    return (
        f"sampled|seed={seed}|k={prediction_times_per_subject}|min_context={min_context_per_subject}"
        f"|fraction={subject_subsample_fraction}"
    )


def _run_fingerprint(
    ontology: str | None, specs: str, cohort: str, min_task_positives: int | None = None
) -> dict[str, str | int | None]:
    """The sidecar payload: everything that decides an output's rows, so "exists" can mean "current"."""
    return {
        "ontology_fingerprint": ontology,
        "specs_fingerprint": specs,
        "cohort_fingerprint": cohort,
        "min_task_positives": min_task_positives,
    }


def _provenance_path(out_dir: Path, fp: Path) -> Path:
    """Sidecar recording the ontology a written output was labeled under.

    Mirrors :func:`~every_query.generate_tasks.sample_tasks.labeled_fingerprint_path`: provenance
    lives in the ``{name}_artifacts`` sibling, never in the final-output root, so the output tree
    keeps holding nothing but the parquets ``EQ_predict_multitask`` rglobs (invariant 7).  The
    output's path *below* ``out_dir`` is kept as the sidecar's own, so every shard of every split
    gets its own and none can collide.

    Examples:
        >>> _provenance_path(Path("/x/grid"), Path("/x/grid/eval/held_out/0.parquet"))
        PosixPath('/x/grid_artifacts/_labeled/eval/held_out/0.json')
    """
    rel = fp.relative_to(out_dir).with_suffix(".json")
    return default_artifacts_dir(out_dir) / LABELED_DIRNAME / rel


def _recorded_fingerprint(out_dir: Path, fp: Path) -> dict | None:
    """Read back :func:`_run_fingerprint` for an existing output; ``None`` if absent/unreadable."""
    try:
        recorded = json.loads(_provenance_path(out_dir, fp).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return recorded if isinstance(recorded, dict) else None


def _output_is_current(out_dir: Path, fp: Path, fingerprint: dict[str, str | int | None]) -> bool:
    """Whether an existing output was labeled under the ontology, specs and cohort of this run.

    A missing or unreadable sidecar is stale, as in :func:`label_one_sequence_shard`: the sidecar
    is the only record of what produced the parquet, and an output nothing vouches for is not one
    to reuse.  A pre-fingerprint sidecar carrying only ``ontology_fingerprint`` compares unequal for
    the same reason.
    """
    return fp.exists() and _recorded_fingerprint(out_dir, fp) == fingerprint


def _support_filter_marker(out_dir: Path, split: str) -> Path:
    return default_artifacts_dir(out_dir) / "_support_filter" / f"{split}.json"


def filter_tasks_by_min_positives(
    files: list[Path], min_positives: int, unique_files: list[Path] | None = None
) -> dict[str, int]:
    """Count final-answer positives globally, then retain complete conditional tasks per shard.

    Only the compact per-shard task counts are concatenated; the full grid is read one shard at a
    time.  This bounds memory for large dense evaluation grids.

    The optional columns are filled only to build the task key: each shard is written back with
    the columns it had, so a filtered grid reads exactly as an unfiltered one does.  When
    ``unique_files`` is given (aligned with ``files``), each ``eval_unique`` frame is rewritten
    from its shard's retained rows, so it keeps mirroring ``eval/``.
    """
    counts = []
    rows_before = 0
    for fp in files:
        shard = _with_prior_answers(normalize_task_columns(pl.read_parquet(fp)))
        rows_before += shard.height
        counts.append(
            shard.group_by(TASK_KEY).agg(
                pl.len().alias("n_rows"),
                pl.col(ANSWERS_COL).list.last().cast(pl.Int64).sum().alias("n_positive"),
            )
        )

    global_counts = (
        pl.concat(counts, how="vertical_relaxed")
        .group_by(TASK_KEY)
        .agg(pl.col("n_rows").sum(), pl.col("n_positive").sum())
    )
    eligible = global_counts.filter(pl.col("n_positive") >= min_positives).select(TASK_KEY)
    sid = TaskQuerySchema.subject_id_name
    pt = TaskQuerySchema.prediction_time_name
    rows_after = 0
    for fp, unique_fp in zip(files, unique_files or [None] * len(files), strict=True):
        raw = pl.read_parquet(fp)
        shard = _with_prior_answers(normalize_task_columns(raw))
        retained = shard.join(eligible, on=TASK_KEY, how="semi", nulls_equal=True).select(raw.columns)
        rows_after += retained.height
        _atomic_write_parquet(retained, fp)
        if unique_fp is not None:
            _atomic_write_parquet(retained.select(sid, pt).unique().sort(sid, pt), unique_fp)

    summary = {
        "tasks_before": global_counts.height,
        "tasks_after": eligible.height,
        "rows_before": rows_before,
        "rows_after": rows_after,
    }
    logger.info("Global task support filter (min_positives=%d): %s", min_positives, summary)
    return summary


def attach_forced_answers(labeled: pl.DataFrame, specs: list[SequenceSpec]) -> pl.DataFrame:
    """Add the per-row ``forced_answers`` list column, or nothing when no spec forces an answer.

    A forced answer is a property of the spec, not of the events, so the labelers never see it; it is
    tiled onto their output instead, relying on the dense grid's context-major row order (row ``i``
    is ``specs[i % N]``).  That order is checked against ``queries`` rather than trusted: attaching a
    forced answer to the wrong sequence would be a well-formed parquet of wrong conditioning.

    Examples:
        >>> specs = [SequenceSpec("a", ("A", "B"), (1.0, 1.0), forced_answers=(True, None)),
        ...          SequenceSpec("b", ("C",), (1.0,))]
        >>> labeled = pl.DataFrame({"queries": [["A", "B"], ["C"], ["A", "B"], ["C"]]})
        >>> attach_forced_answers(labeled, specs)["forced_answers"].to_list()
        [[True, None], [None], [True, None], [None]]
        >>> attach_forced_answers(labeled, specs[1:]).columns
        ['queries']
        >>> attach_forced_answers(labeled[::-1], specs)
        Traceback (most recent call last):
            ...
        RuntimeError: labeled grid row 0 holds queries ['C'] where spec 'a' (['A', 'B']) was
        expected; the grid is not in context-major spec order, so forced answers cannot be attached
    """
    if not any(s.has_forced_answers for s in specs):
        return labeled
    n = len(specs)
    if labeled.height % n:
        raise RuntimeError(f"labeled grid has {labeled.height} rows, not a multiple of the {n} spec(s)")
    for i, got in enumerate(labeled[QUERIES_COL].to_list()):
        if got != list(specs[i % n].queries):
            raise RuntimeError(
                f"labeled grid row {i} holds queries {got} where spec {specs[i % n].name!r} "
                f"({list(specs[i % n].queries)}) was expected; the grid is not in context-major spec "
                "order, so forced answers cannot be attached"
            )
    per_spec = [[s.forced_at(p) for p in range(len(s))] for s in specs]
    tiled = pl.Series(FORCED_ANSWERS_COL, per_spec * (labeled.height // n), dtype=pl.List(pl.Boolean))
    return labeled.with_columns(tiled)


def drop_forced_mismatches(labeled: pl.DataFrame) -> pl.DataFrame:
    """Keep only the rows whose labeled truth agrees with every designed ``forced_answers`` entry.

    A forced answer selects a spec's cohort: "P(death | record did not end)" is scored on the contexts
    where the record really did not end, never on a counterfactual.  So what the model is told always
    equals the truth, and a forced spec's rows are exactly the matching ``prior_answers`` cell of the
    same spec left unforced.  Specs that force nothing — every sampled spec — keep every context.

    Examples:
        >>> labeled = pl.DataFrame({
        ...     "queries": [["A", "B"]] * 3 + [["C"]],
        ...     "answers": [[True, False], [False, True], [True, True], [False]],
        ...     "forced_answers": [[True, None]] * 3 + [[None]],
        ... })
        >>> drop_forced_mismatches(labeled)["answers"].to_list()
        [[True, False], [True, True], [False]]
        >>> drop_forced_mismatches(labeled.drop("forced_answers")).height
        4
    """
    if FORCED_ANSWERS_COL not in labeled.columns:
        return labeled
    forced, truth = pl.col(FORCED_ANSWERS_COL), pl.col(ANSWERS_COL)
    rows = labeled.with_row_index("_row")
    contradicted = (
        rows.select("_row", FORCED_ANSWERS_COL, ANSWERS_COL)
        .explode(FORCED_ANSWERS_COL, ANSWERS_COL)
        .filter(forced.is_not_null() & (forced != truth))["_row"]
    )
    return rows.filter(~pl.col("_row").is_in(contradicted.implode())).drop("_row")


def _write(labeled: pl.DataFrame, fp: Path, out_dir: Path, fingerprint: dict[str, str | int | None]) -> None:
    """Align to ``QuerySeqSchema``, atomically write one output parquet, and record its provenance.

    The sidecar is written *after* the parquet is committed, so a present sidecar always describes
    a present output — the ordering :func:`label_one_sequence_shard` uses for the same reason.
    """
    aligned = QuerySeqSchema.align(labeled.to_arrow())
    fp.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_parquet(pl.from_arrow(aligned), fp)
    _atomic_write_json(fingerprint, _provenance_path(out_dir, fp))
    logger.info("Wrote %d labeled query sequences to %s", labeled.height, fp)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def run_worker(
    data_dir: Path,
    out_dir: Path,
    split: str,
    input_shard: str,
    specs: list[SequenceSpec],
    prediction_times_per_subject: int,
    min_context_per_subject: int,
    seed: int,
    overwrite: bool = False,
    subject_subsample_fraction: float | None = None,
    contexts: pl.DataFrame | None = None,
    ontology_dir: str | None = None,
    fingerprint: dict[str, str | int | None] | None = None,
    write_unique_prediction_times: bool = True,
    unique_out_dir: Path | None = None,
) -> Path | None:
    """Label ``specs`` at every context of one shard, writing one ``QuerySeqSchema`` parquet.

    The sequence analogue of :func:`~every_query.generate_tasks.sample_evaluation_tasks.run_worker`,
    step for step: same output paths (``out_dir`` is already the ``eval/`` root), same skip rule,
    same event reader, and — when ``contexts`` is ``None`` — the same cohort sampler on the same
    seed axes, so the two workers draw the identical cohort for one ``(seed, split, shard)``.

    Args:
        data_dir: MEDS root; this shard's events are at ``{data_dir}/data/{split}/{input_shard}.parquet``.
        out_dir: The ``eval/`` output root.  Provenance sidecars are keyed on the path below its
            *parent*, so ``out_dir.parent`` is what :func:`_provenance_path` receives.
        split: Split whose shard to label.
        input_shard: Shard stem.
        specs: The ``N`` sequences to evaluate at every context — resolved once by the caller and
            shared across shards.
        prediction_times_per_subject: ``K`` prediction times to sample per subject.  Ignored when
            ``contexts`` is supplied.
        min_context_per_subject: Prior events a subject needs before a time is a candidate.
            Ignored when ``contexts`` is supplied.
        seed: Top-level seed; the cohort axes derive on ``(split, input_shard)`` from it.
        overwrite: Regenerate outputs that already exist.
        subject_subsample_fraction: Optional per-subject hash-threshold subsample.  Ignored when
            ``contexts`` is supplied.
        contexts: A supplied ``(subject_id, prediction_time)`` cohort; only this shard's subjects
            are taken from it.  ``None`` samples the cohort from the shard instead.
        ontology_dir: Directory of ``EQ_build_ontology`` artifacts.  Explodes the event stream
            through the closure before labeling, exactly as the training sampler's Stage 4' does.
            ``None`` is a no-op.
        fingerprint: :func:`_run_fingerprint` for this run, recorded beside the output so
            "already exists" means "already labeled under this ontology, these specs and this
            cohort".  ``None`` (direct callers) derives it from the other arguments.
        write_unique_prediction_times: Also write the deduped cohort under ``unique_out_dir``.
        unique_out_dir: The ``eval_unique/`` root.

    Returns:
        The written parquet path, or ``None`` if the output existed and ``overwrite=False``.
    """
    root = out_dir.parent
    if fingerprint is None:
        fingerprint = _run_fingerprint(
            _ontology_fingerprint(ontology_dir, [c for s in specs for c in s.queries]),
            _specs_fingerprint(specs),
            _cohort_fingerprint(
                contexts,
                seed,
                prediction_times_per_subject,
                min_context_per_subject,
                subject_subsample_fraction,
            ),
        )
    labels_fp = _labels_fp(out_dir, split, input_shard)
    unique_fp = (
        _unique_fp(unique_out_dir, split, input_shard)
        if write_unique_prediction_times and unique_out_dir is not None
        else None
    )
    if (
        not overwrite
        and _output_is_current(root, labels_fp, fingerprint)
        and (unique_fp is None or unique_fp.exists())
    ):
        logger.info("Labels already exist at %s, skipping.", labels_fp)
        return None

    sid = TaskQuerySchema.subject_id_name
    pt = TaskQuerySchema.prediction_time_name

    shard_path = data_dir / "data" / split / f"{input_shard}.parquet"
    events_df = _read_event_shard(shard_path)
    logger.info("Loaded %d events from %s", events_df.height, shard_path)

    if contexts is not None:
        # Subjects are disjoint across shards, so every supplied context lands in exactly one file.
        shard_contexts = contexts.filter(pl.col(sid).is_in(events_df[sid].unique().implode())).sort(sid, pt)
    else:
        if subject_subsample_fraction is not None:
            subj_seed = derive_seed(seed, "subject_subsample", split, input_shard)
            events_df = subsample_subject_ids(events_df, subject_subsample_fraction, subj_seed)
            logger.info(
                "Subsampled to %d events / %d subjects (fraction=%.4f)",
                events_df.height,
                events_df[sid].n_unique(),
                subject_subsample_fraction,
            )
        pt_seed = derive_seed(seed, "prediction_times", split, input_shard)
        shard_contexts = sample_prediction_times_per_subject(
            events_df=events_df,
            k=prediction_times_per_subject,
            min_context_per_subject=min_context_per_subject,
            seed=pt_seed,
        )
    logger.info(
        "Cohort: %d contexts across %d subjects x %d sequences = %d rows to label",
        shard_contexts.height,
        shard_contexts[sid].n_unique() if shard_contexts.height else 0,
        len(specs),
        shard_contexts.height * len(specs),
    )

    # After the cohort draw (the flat sampler samples from raw events too) and once per shard,
    # mirroring the training sampler's Stage 4' worker.
    events_df = maybe_expand_to_matching_query_nodes(events_df, ontology_dir)

    # An empty cohort still writes an empty, well-formed parquet, so downstream sees a complete
    # split even when a sparse shard yields no eligible prediction time.
    index_df = build_dense_sequence_index_df(shard_contexts, specs)
    labeled = attach_forced_answers(label_query_sequences(index_df, events_df), specs)
    labeled = drop_forced_mismatches(labeled)
    _write(labeled, labels_fp, root, fingerprint)

    if unique_fp is not None:
        unique_df = labeled.select(sid, pt).unique().sort(sid, pt)
        _atomic_write_parquet(unique_df, unique_fp)
        logger.info("Wrote %d unique (subject_id, prediction_time) rows to %s", unique_df.height, unique_fp)

    return labels_fp


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------


CONFIGS = str(files("every_query") / "generate_tasks" / "configs")


@hydra.main(version_base=None, config_path=CONFIGS, config_name="sample_evaluation_query_sequences_config")
def main(cfg: DictConfig) -> None:
    """Produce one evaluation-sequences parquet per shard of the chosen split.

    Shards are discovered from ``{data_dir}/data/{split}/*.parquet`` and all of them are processed
    in this one invocation, exactly as ``EQ_generate_evaluation_tasks`` does.

    Usage (``N`` sequences drawn from the training distribution at ``K`` sampled times per
    subject — the zero-argument default beyond the three path roots):

        EQ_generate_evaluation_query_sequences \\
            data_dir=... out_dir=... query_codes=... split=held_out \\
            prediction_times_per_subject=2 num_evaluation_sequences=64

    Usage (designed sequences on a supplied cohort):

        EQ_generate_evaluation_query_sequences \\
            data_dir=... out_dir=... query_codes=... split=held_out \\
            contexts_path=cohort.parquet sequences_path=tasks.yaml
    """
    data_dir = _require_path_arg(cfg.get("data_dir"), "data_dir")
    out_dir = _require_path_arg(cfg.get("out_dir"), "out_dir")
    split = str(cfg.split)
    seed = int(cfg.seed)
    ontology_dir = cfg.get("ontology_dir")
    threshold = cfg.get("min_task_positives")
    if threshold is not None and (
        isinstance(threshold, bool) or not isinstance(threshold, int) or threshold < 1
    ):
        raise ValueError(f"min_task_positives must be a positive integer or null, got {threshold!r}")

    # The same two arguments, read the same way, as ``QuerySequenceDistribution.from_config``: an
    # eval grid drawn from a *narrower* universe than training's is the drift this whole module
    # exists to avoid, and the ancestor half of the universe is no exception.
    query_codes = build_query_universe(read_query_codes(cfg.get("query_codes")), ontology_dir=ontology_dir)

    # The sequence axis, once: the specs are shared by every shard, and are seeded independently of
    # the cohort so the same ``(seed, split)`` yields the same ``N`` sequences for any cohort.
    sequences_path = cfg.get("sequences_path")
    eventbound_fraction = cfg.get("eventbound_fraction")
    # Window starts (issue #27): keys absent from an older config fall back to prediction-time
    # starts, exactly as the multitask sampler reads them.  ``start_event_codes`` is an explicit
    # list / file (``read_query_codes``) or null for the whole effective universe.
    pts = cfg.get("prediction_time_start_fraction")
    start_event_codes = cfg.get("start_event_codes")
    if start_event_codes is not None:
        start_event_codes = read_query_codes(start_event_codes)
    specs = resolve_specs(
        sequences_path=sequences_path,
        query_codes=query_codes,
        n_sequences=int(cfg.num_evaluation_sequences),
        min_queries=int(cfg.min_queries),
        max_queries=int(cfg.max_queries),
        duration_min=float(cfg.duration_min),
        duration_max=float(cfg.duration_max),
        seed=derive_seed(seed, "eval_seq_specs", split),
        eos_first_fraction=float(cfg.get("eos_first_fraction", 0.0)),
        duration_mode=str(cfg.get("duration_mode", "random")),
        duration_distribution=str(cfg.get("duration_distribution", "log-uniform")),
        eventbound_fraction=eventbound_fraction,
        eventstart_fraction=float(cfg.get("eventstart_fraction", 0.0) or 0.0),
        prediction_time_start_fraction=1.0 if pts is None else float(pts),
        start_duration_min=float(cfg.get("start_duration_min", 1.0)),
        start_duration_max=float(cfg.get("start_duration_max", 180.0)),
        start_duration_distribution=str(cfg.get("start_duration_distribution", "log-uniform")),
        start_event_codes=start_event_codes,
    )
    validate_spec_codes(specs, model_query_vocab(query_codes, ontology_dir))
    # Only *designed* specs get the whole-day check: sampled ones draw continuous float durations
    # by design.  A designed spec's fractional duration labels correctly but is usually a slip.
    if sequences_path is not None:
        non_integer = sorted(
            {float(d) for s in specs for d in (*s.durations, *s.start_durations) if not float(d).is_integer()}
        )
        if non_integer:
            logger.warning(
                "%d designed spec duration(s) / start duration(s) are not whole days (e.g. %s); labeling "
                "honours the fraction — check this is intended.",
                len(non_integer),
                non_integer[:5],
            )
    ontology_fingerprint = _ontology_fingerprint(ontology_dir, query_codes)

    # Reject booleans up front: Hydra/OmegaConf parses ``subject_subsample_fraction=true`` as a
    # Python ``True``, which would otherwise become ``1.0`` and silently disable subsampling.
    ssf_raw = cfg.get("subject_subsample_fraction")
    if isinstance(ssf_raw, bool) or (ssf_raw is not None and not isinstance(ssf_raw, int | float)):
        raise TypeError(
            f"cfg.subject_subsample_fraction must be a number in (0, 1] or null, "
            f"got {type(ssf_raw).__name__}: {ssf_raw!r}"
        )
    subject_subsample_fraction = None if ssf_raw is None else float(ssf_raw)

    shards = _split_shards(data_dir, split)

    contexts_path = cfg.get("contexts_path")
    contexts = None
    if contexts_path is not None:
        contexts = read_supplied_contexts(str(contexts_path))
        assert_subjects_in_split(data_dir, split, shards, contexts[TaskQuerySchema.subject_id_name])
        logger.info("Read %d supplied context(s) from %s", contexts.height, contexts_path)

    # Everything that decides an output's rows, recorded beside it: a rerun into the same
    # ``out_dir`` with different specs or a different cohort must relabel, not skip.
    fingerprint = _run_fingerprint(
        ontology_fingerprint,
        _specs_fingerprint(specs),
        _cohort_fingerprint(
            contexts,
            seed,
            int(cfg.prediction_times_per_subject),
            int(cfg.min_context_per_subject),
            subject_subsample_fraction,
        ),
        threshold,
    )

    write_unique_prediction_times = bool(cfg.get("write_unique_prediction_times", True))
    # A per-shard sidecar alone cannot certify a global filtering pass.  If the completion marker
    # is absent or stale, regenerate *every* shard: some may already have had tasks removed when
    # an earlier attempt stopped, and those dropped rows cannot be recovered by recounting.
    marker = _support_filter_marker(Path(out_dir), split) if threshold is not None else None
    files = [_labels_fp(Path(out_dir) / "eval", split, shard) for shard in shards]
    if marker is not None:
        try:
            completed = json.loads(marker.read_text()).get("fingerprint") == fingerprint
        except (OSError, json.JSONDecodeError, AttributeError):
            completed = False
        if (
            completed
            and not bool(cfg.get("overwrite", False))
            and all(_output_is_current(Path(out_dir), fp, fingerprint) for fp in files)
        ):
            logger.info("Filtered evaluation grid is current for %s, skipping.", split)
            return
        marker.unlink(missing_ok=True)

    for input_shard in shards:
        run_worker(
            data_dir=data_dir,
            out_dir=Path(out_dir) / "eval",
            split=split,
            input_shard=input_shard,
            specs=specs,
            prediction_times_per_subject=int(cfg.prediction_times_per_subject),
            min_context_per_subject=int(cfg.min_context_per_subject),
            seed=seed,
            overwrite=bool(cfg.get("overwrite", False)) or marker is not None,
            subject_subsample_fraction=subject_subsample_fraction,
            contexts=contexts,
            ontology_dir=ontology_dir,
            fingerprint=fingerprint,
            write_unique_prediction_times=write_unique_prediction_times,
            unique_out_dir=Path(out_dir) / "eval_unique" if write_unique_prediction_times else None,
        )

    if marker is not None:
        unique_files = (
            [_unique_fp(Path(out_dir) / "eval_unique", split, shard) for shard in shards]
            if write_unique_prediction_times
            else None
        )
        summary = filter_tasks_by_min_positives(files, threshold, unique_files)
        marker.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json({"fingerprint": fingerprint, "summary": summary}, marker)


if __name__ == "__main__":
    main()
