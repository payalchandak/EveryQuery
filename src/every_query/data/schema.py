"""Cross-stage schema for EveryQuery task-query rows.

``TaskQuerySchema`` is the contract shared between ``generate_tasks/`` (producer) and the
future ``predict/`` + ``evaluate/`` stages (consumers).  Each row specifies: *"for
subject_id at prediction_time, will ``query`` occur within ``duration_days``?"*

Extends the MEDS ``LabelSchema`` — so it inherits ``subject_id``, ``prediction_time``, and
the optional label columns (``boolean_value``, ``integer_value``, etc.) — and adds the two
required query fields ``query`` and ``duration_days``.  Mirrors the pattern
``meds-evaluation``'s `PredictionSchema
<https://github.com/kamilest/meds-evaluation/blob/main/src/meds_evaluation/schema.py>`_ uses.

The ``query`` column holds the MEDS code the query asks about.  The field is named
``query`` (not ``code``) to match the existing column name used by ``sample_tasks``
output, ``EveryQueryPytorchDataset``, and the ``EveryQueryBatch.query`` tensor —
renaming to ``code`` would (a) collide with the inherited ``EveryQueryBatch.code``
sequence-token tensor, which is a different thing semantically, and (b) churn every
downstream consumer for no functional win.

Initial scope (per #80) is intentionally narrow: a flat single code + a continuous (float)
duration.  Extensions — compound ANY/ALL queries, structured task payloads — are out of
scope for the initial schema and will be added as the inference / evaluation pipelines
evolve.
"""

import polars as pl
import pyarrow as pa
from flexible_schema import Optional, Required
from meds import LabelSchema


class TaskQuerySchema(LabelSchema):
    """An EveryQuery task-query row: a MEDS prediction-time label plus the query that defines it.

    Each row is a single ``(subject_id, prediction_time, query, duration_days)`` tuple with
    optional label columns inherited from ``LabelSchema``.  When the ground-truth label is
    present it lives on the inherited ``boolean_value`` column — *"did ``query`` occur for
    ``subject_id`` within ``duration_days`` of ``prediction_time``?"* — so the schema serves
    both inference input (no label) and evaluation input (label filled in) without a branch.

    Attributes:
        query: The MEDS code the query asks about.  Stored as ``pa.large_string``
            (polars' ``Utf8`` serializes to ``large_string`` when a DataFrame is
            converted to arrow, so this matches producer output natively and
            ``TaskQuerySchema.align()`` works without type coercion; also matches
            MEDS's own ``DataSchema.code`` convention which uses ``large_string``
            for the same 2 GB-offset reason).  Named ``query`` rather than ``code``
            to match the column name already used throughout the sampler / dataset /
            batch layer (``EveryQueryBatch.code`` is the distinct event-sequence-token
            tensor).
        duration_days: The horizon, in days (continuous — ``float32``) within which the
            ``query`` code must occur for the query to be positive.  Allowing fractional
            days keeps the contract flexible for future finer-grained horizons.

    Examples:
        A row with just the query (inference input) validates:

        >>> from datetime import datetime
        >>> import pyarrow as pa
        >>> data = pa.Table.from_pylist([
        ...     {"subject_id": 1, "prediction_time": datetime(2023, 1, 1),
        ...      "query": "ICD//I10", "duration_days": 30.0},
        ... ])
        >>> aligned = TaskQuerySchema.align(data)
        >>> [f.name for f in aligned.schema]
        ['subject_id', 'prediction_time', 'query', 'duration_days']

        A row with the inherited ``boolean_value`` label filled in (evaluation input) also
        validates:

        >>> data = pa.Table.from_pylist([
        ...     {"subject_id": 1, "prediction_time": datetime(2023, 1, 1),
        ...      "query": "ICD//I10", "duration_days": 30.0, "boolean_value": True},
        ...     {"subject_id": 2, "prediction_time": datetime(2023, 1, 1),
        ...      "query": "ICD//I10", "duration_days": 30.0, "boolean_value": False},
        ... ])
        >>> aligned = TaskQuerySchema.align(data)
        >>> [f.name for f in aligned.schema]
        ['subject_id', 'prediction_time', 'boolean_value', 'query', 'duration_days']

        Fractional durations are supported:

        >>> data = pa.Table.from_pylist([
        ...     {"subject_id": 1, "prediction_time": datetime(2023, 1, 1),
        ...      "query": "ICD//I10", "duration_days": 0.5},
        ... ])
        >>> _ = TaskQuerySchema.align(data)
    """

    query: Required(pa.large_string(), nullable=False)
    duration_days: Required(pa.float32(), nullable=False)
    # Override ``boolean_value`` from ``LabelSchema`` (which declares it
    # ``Optional(bool, nullable=NONE)``) to allow nulls — the EveryQuery task-label
    # convention is to use null as the "censored" sentinel (closes #122).  The column
    # stays optional at the schema level so inference-only inputs (no ground truth)
    # continue to validate.
    boolean_value: Optional(pa.bool_(), nullable=True)


class QuerySeqSchema(LabelSchema):
    """A conditional query-sequence row: one patient context plus an ordered list of queries.

    Each row is a ``(subject_id, prediction_time)`` context with three aligned list columns plus
    up to three optional ones (``bound_events``, ``start_durations``, ``start_events``) that
    refine each query's window.  For query position ``j`` the window is::

        start[j] = prediction_time + start_durations[j]
                   OR the first occurrence of start_events[j] strictly after prediction_time
        end[j]   = start[j] + durations[j]
                   OR the first occurrence of bound_events[j] strictly after start[j]
        answers[j] = some occurrence of queries[j] satisfies  start[j] < occurrence < end[j]

    The start is resolved first and the end **relative to the resolved start** — a duration end
    is measured from it, and an event end is searched strictly after it, never after
    ``prediction_time``.  Both endpoints are open.  A start event that never occurs after the
    prediction time leaves the window empty (answer ``False``, even if the end is also
    unresolved); an end event that never occurs after a resolved start lets the window run to
    the end of the record.  With the default prediction-time start this is the legacy
    ``(prediction_time, prediction_time + duration)`` / ``(prediction_time, boundary)`` window —
    bit-for-bit for whole-day horizons; a fractional horizon can differ by one microsecond at the
    window end between the two labelers (see ``label_with_explicit_starts``).

    Attributes:
        queries: Ordered list of MEDS code strings (any vocabulary code, random order — including
            the end-of-timeline code ``TIMELINE//END``, which is an ordinary code).
        durations: Per-query horizons in days (``float32``) after the **resolved start**, aligned
            with ``queries``.
        answers: Per-query booleans aligned with ``queries`` — *"was ``queries[j]`` observed in
            ``(start[j], end[j])``?"*; with the default prediction-time start that window is
            ``(prediction_time, prediction_time + durations[j])``.  The window is **open at both
            ends**: an occurrence exactly at the start instant is outside it, and so is one landing
            exactly on the end instant.  Binary, never null; an unobservable event (record ends
            first) is ``False``.  Censoring is carried by a ``TIMELINE//END`` query rather than a
            null answer.
        bound_events: Optional per-query boundary-event codes.  ``bound_events[j]`` is null for
            an ordinary time-bounded query (``durations[j]`` is the horizon), and a vocabulary
            code for an **event-bounded** one — the window then ends at the first occurrence of
            that code strictly after the resolved start, open at both ends exactly as the
            time-bounded window is, so a query code sharing the boundary event's instant does NOT
            count.  ``durations[j]`` holds the ``EVENT_BOUND_DURATION_SENTINEL`` (-1.0) rather than
            a horizon.  The whole column is absent from bound-free parquets, which stay valid.
        start_durations: Optional per-query window starts in days after ``prediction_time``
            (``float32``).  ``0.0`` opens the window at the prediction time (the legacy and default
            form), ``> 0`` at ``prediction_time + start_durations[j]``; an event-defined start
            holds ``EVENT_BOUND_DURATION_SENTINEL`` (-1.0).  Must be finite; negative values other
            than the sentinel are invalid.
        start_events: Optional per-query start-event codes aligned with ``start_durations``: null
            for a duration-defined start, a vocabulary code for an **event-defined** one (the
            window opens at its first occurrence strictly after ``prediction_time``; none ⇒ the
            window is empty).  Exactly one representation is active per position: a non-null
            ``start_events[j]`` requires ``start_durations[j] == -1.0`` and a null one requires
            ``start_durations[j] >= 0``.  The two start columns are present together or absent
            together; absent means ``[0.0] * K`` / ``[None] * K``.  Parquets written before starts
            existed carry neither and remain valid.  ``align`` itself only types the columns; the
            pairing, alignment and representation rules are enforced by the readers
            (``QuerySeqPytorchDataset`` and the ``label_query_sequences`` dispatch).  Active
            starts are consumed only by the multitask predictor; ``QuerySeqPytorchDataset``
            rejects them unless the caller opts in.
        forced_answers: Optional per-query designed conditioning answers aligned with ``queries``:
            ``true`` / ``false`` is fed to the model in place of ``answers[j]`` when query ``j``
            conditions a later one, null teacher-forces the labeled truth.  ``answers`` is never
            altered, so the scoring label stays the truth.  Null on every row's final query (the
            scored one).  ``EQ_generate_evaluation_query_sequences`` writes a forced row only where
            the truth agrees with it, so in its grids the fed value is always the truth.  Absent
            from grids that force nothing; like active starts, consumed only by the multitask
            predictor and rejected by ``QuerySeqPytorchDataset`` otherwise.

    Examples:
        >>> from datetime import datetime
        >>> import pyarrow as pa
        >>> data = pa.Table.from_pylist([
        ...     {"subject_id": 1, "prediction_time": datetime(2023, 1, 1),
        ...      "queries": ["TIMELINE//END", "ICD//I10"], "durations": [30.0, 7.0],
        ...      "answers": [False, True]},
        ... ])
        >>> aligned = QuerySeqSchema.align(data)
        >>> [f.name for f in aligned.schema]
        ['subject_id', 'prediction_time', 'queries', 'durations', 'answers']

        ``bound_events`` is optional: it appears only when the data carry it, so parquets
        generated before the feature existed remain valid and readable.

        >>> bounded = pa.Table.from_pylist([
        ...     {"subject_id": 1, "prediction_time": datetime(2023, 1, 1),
        ...      "queries": ["ICD//I10", "ICD//E11"], "durations": [30.0, -1.0],
        ...      "answers": [False, True], "bound_events": [None, "HOSPITAL_DISCHARGE//HOME"]},
        ... ])
        >>> aligned = QuerySeqSchema.align(bounded)
        >>> [f.name for f in aligned.schema]
        ['subject_id', 'prediction_time', 'queries', 'durations', 'answers', 'bound_events']
        >>> aligned.column("bound_events").to_pylist()
        [[None, 'HOSPITAL_DISCHARGE//HOME']]

        Explicit window starts are optional in the same way.  Here the first query's window opens
        seven days after the prediction time and the second's at the next admission (its start
        duration is the ``-1.0`` sentinel):

        >>> started = pa.Table.from_pylist([
        ...     {"subject_id": 1, "prediction_time": datetime(2023, 1, 1),
        ...      "queries": ["ICD//I10", "LAB//X"], "durations": [30.0, 30.0],
        ...      "answers": [False, True], "bound_events": [None, None],
        ...      "start_durations": [7.0, -1.0], "start_events": [None, "HOSPITAL_ADMISSION"]},
        ... ])
        >>> aligned = QuerySeqSchema.align(started)
        >>> [f.name for f in aligned.schema]  # doctest: +NORMALIZE_WHITESPACE
        ['subject_id', 'prediction_time', 'queries', 'durations', 'answers', 'bound_events',
         'start_durations', 'start_events']
        >>> aligned.column("start_events").to_pylist()
        [[None, 'HOSPITAL_ADMISSION']]
    """

    queries: Required(pa.large_list(pa.large_string()), nullable=False)
    durations: Required(pa.large_list(pa.float32()), nullable=False)
    answers: Required(pa.large_list(pa.bool_()), nullable=False)
    bound_events: Optional(pa.large_list(pa.large_string()), nullable=True)
    start_durations: Optional(pa.large_list(pa.float32()), nullable=True)
    start_events: Optional(pa.large_list(pa.large_string()), nullable=True)
    forced_answers: Optional(pa.large_list(pa.bool_()), nullable=True)


class MultitaskBoundarySchema(LabelSchema):
    """One multitask-boundary context: a ``(subject_id, prediction_time)`` plus exactly ``K`` windows.

    The all-vocabulary targets are **not** in the parquet; they live row-aligned in the bit-packed
    ``<shard>.labels.npy`` sidecar next to it (see
    :mod:`~every_query.generate_tasks.sample_multitask_sequences`).  This schema only carries the
    window definitions - a start and an end specification per slot (issue #24):

    Attributes:
        start_durations: ``K`` window starts in days after ``prediction_time`` (``float32``).  ``0``
            opens the window at the prediction time (the pre-#24 behaviour), ``> 0`` at
            ``prediction_time + start_duration``; an event-defined start holds
            ``EVENT_BOUND_DURATION_SENTINEL`` (``-1.0``).
        start_events: ``K`` start codes aligned with ``start_durations``: null for a duration-defined
            start, an event code for an event-defined one (the window opens at its first occurrence
            strictly after ``prediction_time``; if there is none the window is empty).  Normally a
            base code; an ontology node name in a boundaries ontology mode, where "its occurrence"
            means an occurrence of any descendant leaf.  Exactly one representation is active per
            slot.  Parquets written before #24 lack both start columns and are read as ``[0.0] * K``
            / ``[null] * K``.
        durations: ``K`` horizons in days after the **resolved start** (``float32``).  A
            duration-bounded slot holds ``>= 0``; an event-bounded slot holds
            ``EVENT_BOUND_DURATION_SENTINEL`` (``-1.0``).
        bound_events: ``K`` boundary codes aligned with ``durations``: null for a duration-bounded
            slot, an event code for an event-bounded one (the first occurrence strictly after the
            resolved start).  Exactly one representation is active per slot.  Normally a base
            code; an ontology node name when the sampler ran in a boundaries ontology mode, in
            which case "occurrence" means an occurrence of any descendant leaf.
        condition_codes: ``K-1`` conditioning codes (non-PAD, never null); code ``j`` is the query
            whose answer at boundary ``j`` is teacher-forced into later boundaries.  Base codes,
            or ontology node names in a conditioning ontology mode.
        condition_answers: ``K-1`` booleans; ``answers[j]`` is the all-vocabulary target bit of
            ``condition_codes[j]`` at boundary ``j`` (open-window semantics) - for an ontology node,
            the OR of that bit over the node's closure descendants.

    Examples:
        >>> from datetime import datetime
        >>> import pyarrow as pa
        >>> data = pa.Table.from_pylist([
        ...     {"subject_id": 1, "prediction_time": datetime(2023, 1, 1),
        ...      "start_durations": [7.0, -1.0], "start_events": [None, "ADMISSION"],
        ...      "durations": [30.0, -1.0], "bound_events": [None, "ICD//I10"],
        ...      "condition_codes": ["LAB//X"], "condition_answers": [True]},
        ... ])
        >>> aligned = MultitaskBoundarySchema.align(data)
        >>> [f.name for f in aligned.schema]  # doctest: +NORMALIZE_WHITESPACE
        ['subject_id', 'prediction_time', 'start_durations', 'start_events', 'durations', 'bound_events',
         'condition_codes', 'condition_answers']
        >>> aligned.column("start_events").to_pylist(), aligned.column("bound_events").to_pylist()
        ([[None, 'ADMISSION']], [[None, 'ICD//I10']])
    """

    start_durations: Required(pa.large_list(pa.float32()), nullable=False)
    start_events: Required(pa.large_list(pa.large_string()), nullable=False)
    durations: Required(pa.large_list(pa.float32()), nullable=False)
    bound_events: Required(pa.large_list(pa.large_string()), nullable=False)
    condition_codes: Required(pa.large_list(pa.large_string()), nullable=False)
    condition_answers: Required(pa.large_list(pa.bool_()), nullable=False)


def empty_task_query_df() -> pl.DataFrame:
    """Build an empty polars DataFrame shaped like ``TaskQuerySchema``'s required columns plus the inherited
    ``boolean_value`` (the collapsed label column).

    Only the required columns + ``boolean_value`` are included — not every
    ``LabelSchema`` optional column — because (a) that's what the sampler's empty-input
    fast path needs, and (b) a polars-arrow round-trip coerces ``pa.string`` →
    ``pa.large_string`` on the inherited ``categorical_value`` column, so a schema-
    complete empty frame would fail ``TaskQuerySchema.validate`` after the round-trip
    unless we bypassed polars entirely.  Keeping the shape focused on what downstream
    writers actually emit avoids that type-drift landmine.

    Polars dtypes are derived from ``TaskQuerySchema``'s arrow types at call time
    (``pl.from_arrow`` on an empty arrow table) rather than hardcoded — so any future
    change to the PyArrow type declarations flows through automatically instead of
    drifting silently.  ``pa.large_string`` coerces to ``pl.Utf8`` in polars' type
    system, which is the correct mapping here.

    Callers use this at the empty-input fast path (e.g., ``evaluate_index_df`` when no
    tasks were sampled) so the produced parquet still aligns to the schema via
    ``TaskQuerySchema.align`` at the write boundary.

    Examples:
        >>> df = empty_task_query_df()
        >>> df.height
        0
        >>> for name, dtype in df.schema.items():
        ...     print(f"{name}: {dtype}")
        subject_id: Int64
        prediction_time: Datetime(time_unit='us', time_zone=None)
        query: String
        duration_days: Float32
        boolean_value: Boolean
    """
    field_names = [
        TaskQuerySchema.subject_id_name,
        TaskQuerySchema.prediction_time_name,
        TaskQuerySchema.query_name,
        TaskQuerySchema.duration_days_name,
        TaskQuerySchema.boolean_value_name,
    ]
    pa_fields = [pa.field(name, getattr(TaskQuerySchema, f"{name}_dtype")) for name in field_names]
    return pl.from_arrow(pa.schema(pa_fields).empty_table())
