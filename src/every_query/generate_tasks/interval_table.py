"""Interval table: an ``O(E)`` "time to next occurrence" representation for all-vocabulary labeling.

The multitask sampler (:mod:`~every_query.generate_tasks.sample_multitask_sequences`) needs, for every
sampled context and every vocabulary code, *"does the code occur strictly inside the open window
``(resolved_start, resolved_end)``?"*.  Answering that with one lookup per ``(context, window, code)``
is ``O(N x K x V)``; this module answers it in cost proportional to the subject's interval rows plus
the emitted ``(interval, window)`` matches.

Build from the event stream once; the prediction times are never an input::

    table = build_interval_table(subject_id, time_us, code_index)   # one row per event

Every row ``(subject, code, start, end)`` claims: for any lookup time ``t`` with ``start <= t < end``,
the next occurrence of ``code`` for ``subject`` **strictly after** ``t`` is at ``end``.  Rows chain per
``(subject, code)`` (one row's ``end`` is the next row's ``start``), so they tile the timeline; a ``t``
inside no row of a code has no future occurrence of it.  A code therefore fires for a window opening
at ``t`` iff ``end < window_end`` - the strict comparison is the open upper endpoint, and the strict
"after ``t``" is the open lower one.

The production entry points are :func:`resolve_start_times` / :func:`resolve_end_times` (``N x K``
point searches each, one vectorised ``searchsorted`` on a composite key built once at construction;
issue #24 resolves every window's start first and its end relative to that start) and
:func:`label_sorted_contexts_packed` (subject-sorted interval-range labeling that packs and writes
each bounded chunk as it goes; the lookup time of a row is its resolved window start).
:func:`resolve_bound_times` is the #20 form (every start at the prediction time), kept as a thin
wrapper.  :func:`label_dense_reference` is the small, unbounded helper the unit tests compare against;
it is not used in production.

Times are ``int64`` microseconds (cast MEDS ``datetime[us]`` columns with ``.astype("int64")``); codes
are integer vocabulary indices aligned to the cohort's ``code/vocab_index``.

The interval representation and the strict next-occurrence semantics follow the ``interval_table.py``
prototype in ``gkondas/test-fast-labeler``; everything about orchestration, chunking and output here is
adapted to the multi-bound, subject-sorted, bounded-memory contract of issue #20.
"""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator

#: Start of the first interval of every ``(subject, code)`` chain: "no earlier occurrence".
NEG_INF = -(2**62)

#: Boundary time of an event bound whose code never recurs: any later occurrence is eligible.
INF = 2**62

#: Microseconds per day; durations in (fractional) days are converted with ``round(d * US_PER_DAY)``.
US_PER_DAY = 86_400_000_000

#: Width of the time-rank field packed into the low bits of the composite lookup key.
_RANK_BITS = 28


class IntervalTable(NamedTuple):
    """One shard's intervals plus the indexing state needed for cheap per-subject and per-group slices.

    Rows are sorted by ``(subject_index, code_index, end)``.  All arrays except ``subjects`` and the
    two offset arrays have one entry per interval row (= per non-null-time event).

    Attributes:
        subjects: Sorted unique subject ids (``int64``).
        code_index: Vocabulary index of each interval row (``int64``).
        start: Previous same-``(subject, code)`` occurrence time, or :data:`NEG_INF` (``int64`` µs).
        end: This occurrence's time (``int64`` µs).
        subject_offsets: ``len(subjects) + 1`` offsets; subject ``i`` owns rows
            ``[subject_offsets[i], subject_offsets[i + 1])``.
        group_keys: Sorted unique ``(subject_index << code_bits) | code_index`` keys.
        group_offsets: ``len(group_keys) + 1`` offsets into the rows, so group ``g`` owns rows
            ``[group_offsets[g], group_offsets[g + 1])``, whose ``end`` values are that
            ``(subject, code)``'s ascending occurrence times.
        code_bits: Width of the code field inside a group key.
        unique_times: Sorted unique occurrence times, the rank space of ``lookup_key``.
        lookup_key: Per-row ``(group_key << _RANK_BITS) | rank(end)`` - sorted, because rows are sorted
            by ``(group, end)`` - so "first occurrence of ``(subject, code)`` strictly after ``pt``" is
            one ``searchsorted`` on it.
    """

    subjects: np.ndarray
    code_index: np.ndarray
    start: np.ndarray
    end: np.ndarray
    subject_offsets: np.ndarray
    group_keys: np.ndarray
    group_offsets: np.ndarray
    code_bits: int
    unique_times: np.ndarray
    lookup_key: np.ndarray

    @property
    def n_rows(self) -> int:
        return int(self.end.shape[0])

    @property
    def n_subjects(self) -> int:
        return int(self.subjects.shape[0])

    def subject_index(self, subject_ids: np.ndarray) -> np.ndarray:
        """Row index into ``subjects`` for each id, or ``-1`` where the subject has no events."""
        subject_ids = np.asarray(subject_ids, dtype=np.int64)
        pos = np.searchsorted(self.subjects, subject_ids)
        pos = np.minimum(pos, max(self.n_subjects - 1, 0))
        if self.n_subjects == 0:
            return np.full(subject_ids.shape, -1, dtype=np.int64)
        known = self.subjects[pos] == subject_ids
        return np.where(known, pos, -1)


def build_interval_table(
    subject_id: np.ndarray, time_us: np.ndarray, code_index: np.ndarray, vocab_size: int | None = None
) -> IntervalTable:
    """Build the interval table from one shard's (non-null-time) events in ``O(E log E)``.

    Args:
        subject_id: ``(E,)`` subject id per event.
        time_us: ``(E,)`` occurrence time per event, ``int64`` microseconds (no nulls - drop static rows
            before calling; their vocabulary indices are not reassigned, they simply never fire).
        code_index: ``(E,)`` vocabulary index per event (``>= 0``).
        vocab_size: Size of the *global* vocabulary.  Sizes the code field of the group key so any
            valid code - including ones absent from this shard - fits without aliasing another
            ``(subject, code)`` group.  Defaults to the shard's own max code; lookups for wider codes
            then resolve to :data:`INF` instead of aliasing (see :func:`next_occurrence_after`).

    Duplicate events (same subject, code and timestamp) produce a zero-width interval ``(t, t)`` that
    no prediction time can satisfy (``t <= pt < t`` is empty), so they never change a label.

    Examples:
        >>> t = build_interval_table(np.array([1, 1, 1]), np.array([3, 6, 9]), np.array([0, 1, 0]))
        >>> t.code_index.tolist(), t.start.tolist(), t.end.tolist()
        ([0, 0, 1], [-4611686018427387904, 3, -4611686018427387904], [3, 9, 6])
        >>> t.subject_offsets.tolist(), t.group_offsets.tolist()
        ([0, 3], [0, 2, 3])
    """
    subject_id = np.asarray(subject_id, dtype=np.int64)
    time_us = np.asarray(time_us, dtype=np.int64)
    code_index = np.asarray(code_index, dtype=np.int64)
    if not (subject_id.shape == time_us.shape == code_index.shape) or subject_id.ndim != 1:
        raise ValueError("subject_id, time_us and code_index must be 1-D arrays of the same length")
    if code_index.size and code_index.min() < 0:
        raise ValueError("code_index must be non-negative")

    max_code = int(code_index.max()) if code_index.size else 0
    if vocab_size is not None:
        if max_code >= vocab_size:
            raise ValueError(f"code_index {max_code} is outside the vocabulary of size {vocab_size}")
        max_code = vocab_size - 1
    code_bits = max(max_code.bit_length(), 1)
    subjects, s_idx = np.unique(subject_id, return_inverse=True)
    s_idx = s_idx.astype(np.int64)
    group = (s_idx << code_bits) | code_index
    order = np.lexsort((time_us, group))
    group, time_us, code_index = group[order], time_us[order], code_index[order]

    start = np.empty_like(time_us)
    if time_us.size:
        start[0] = NEG_INF
        same = group[1:] == group[:-1]
        start[1:] = np.where(same, time_us[:-1], NEG_INF)

    subj_of_row = group >> code_bits
    subject_offsets = np.searchsorted(subj_of_row, np.arange(len(subjects) + 1), side="left")

    group_keys, first = np.unique(group, return_index=True)
    group_offsets = np.append(first, len(group)).astype(np.int64)

    unique_times = np.unique(time_us)
    if len(unique_times) >= 2**_RANK_BITS:
        raise ValueError(f"too many distinct timestamps for the {_RANK_BITS}-bit rank field")
    max_group = int(group_keys[-1]) if group_keys.size else 0
    if max_group >= 2 ** (63 - _RANK_BITS):
        raise ValueError("group key space too large for the composite lookup key")
    rank = np.searchsorted(unique_times, time_us, side="left")
    lookup_key = (group << _RANK_BITS) | rank

    return IntervalTable(
        subjects=subjects,
        code_index=code_index,
        start=start,
        end=time_us,
        subject_offsets=subject_offsets.astype(np.int64),
        group_keys=group_keys,
        group_offsets=group_offsets,
        code_bits=code_bits,
        unique_times=unique_times,
        lookup_key=lookup_key,
    )


def next_occurrence_after(
    table: IntervalTable, subject_ids: np.ndarray, prediction_times: np.ndarray, code_index: np.ndarray
) -> np.ndarray:
    """Time of the first occurrence of ``code_index`` for ``subject_ids`` **strictly after** each prediction
    time, or :data:`INF` when there is none.

    One vectorised ``searchsorted`` over the composite ``lookup_key``: the query key ranks the
    prediction time with ``side="right"`` over ``unique_times`` (so an occurrence exactly *at* the
    prediction time does not qualify) and asks for the first row of the ``(subject, code)`` group at or
    above that rank.  Unknown subjects and codes the subject never has resolve to :data:`INF`.

    Examples:
        >>> t = build_interval_table(np.array([1, 1, 1]), np.array([3, 6, 9]), np.array([0, 1, 0]))
        >>> next_occurrence_after(t, np.array([1, 1, 1, 1, 2]), np.array([4, 4, 10, 3, 4]),
        ...                       np.array([0, 1, 0, 0, 0])).tolist() == [9, 6, INF, 9, INF]
        True

        A lookup at :data:`INF` (an unresolved event-defined start) has no occurrence after it:

        >>> next_occurrence_after(t, np.array([1]), np.array([INF]), np.array([0])).tolist() == [INF]
        True
    """
    subject_ids = np.asarray(subject_ids, dtype=np.int64)
    prediction_times = np.asarray(prediction_times, dtype=np.int64)
    code_index = np.broadcast_to(np.asarray(code_index, dtype=np.int64), subject_ids.shape)
    out = np.full(subject_ids.shape, INF, dtype=np.int64)
    if table.n_rows == 0 or subject_ids.size == 0:
        return out

    s_idx = table.subject_index(subject_ids)
    # A code too wide for the group key's code field cannot be in the table; without this guard its
    # high bits would spill into the subject field and alias another subject's group.
    known = (s_idx >= 0) & (code_index < (1 << table.code_bits))
    if not known.any():
        return out
    qgroup = (s_idx[known] << table.code_bits) | code_index[known]
    rq = np.searchsorted(table.unique_times, prediction_times[known], side="right")
    pos = np.searchsorted(table.lookup_key, (qgroup << _RANK_BITS) | rq, side="left")
    hit = pos < table.n_rows
    pos_c = np.minimum(pos, table.n_rows - 1)
    hit &= (table.lookup_key[pos_c] >> _RANK_BITS) == qgroup
    resolved = np.where(hit, table.end[pos_c], INF)
    out[known] = resolved
    return out


def _duration_offsets_us(durations_days: np.ndarray, by_event: np.ndarray) -> np.ndarray:
    """``round(days * US_PER_DAY)`` as ``int64``, zero at event slots (their days value is ignored)."""
    return np.rint(np.where(by_event, 0.0, durations_days) * US_PER_DAY).astype(np.int64)


def _check_nk(name: str, subject_ids: np.ndarray, reference: np.ndarray, days: np.ndarray, codes: np.ndarray):
    n, k = codes.shape
    if days.shape != (n, k) or subject_ids.shape != (n,) or reference.shape not in ((n,), (n, k)):
        raise ValueError(f"shape mismatch between contexts and (N, K) {name} arrays")
    return n, k


def resolve_start_times(
    table: IntervalTable,
    subject_ids: np.ndarray,
    prediction_times: np.ndarray,
    start_durations_days: np.ndarray,
    start_code_index: np.ndarray,
) -> np.ndarray:
    """Resolve the ``(N, K)`` ``int64`` window-start matrix (issue #24).

    Args:
        table: Interval table of the shard the contexts live in.
        subject_ids: ``(N,)``.
        prediction_times: ``(N,)`` ``int64`` µs.
        start_durations_days: ``(N, K)`` float days after the prediction time (``0`` = the prediction
            time itself); ignored where ``start_code_index >= 0``.
        start_code_index: ``(N, K)`` ``int64``; ``-1`` marks a duration-defined start, otherwise the
            vocabulary index of the start event.

    Duration start: ``prediction_time + round(days * US_PER_DAY)``.  Event start: the first occurrence
    of the start code strictly after the prediction time (an occurrence *at* it does not open the
    window), or :data:`INF` when there is none - the window then never opens.

    Examples:
        >>> t = build_interval_table(np.array([1, 1, 1]), np.array([3, 6, 9]), np.array([0, 1, 0]))
        >>> d = np.array([[0.0, 1e-11, 0.0, 0.0]]); c = np.array([[-1, -1, 1, 0]])
        >>> resolve_start_times(t, np.array([1]), np.array([4]), d, c).tolist() == [[4, 5, 6, 9]]
        True
        >>> resolve_start_times(t, np.array([1]), np.array([9]), d[:, :1], c[:, 3:]).tolist() == [[INF]]
        True
    """
    subject_ids = np.asarray(subject_ids, dtype=np.int64)
    prediction_times = np.asarray(prediction_times, dtype=np.int64)
    start_durations_days = np.asarray(start_durations_days, dtype=np.float64)
    start_code_index = np.asarray(start_code_index, dtype=np.int64)
    n, _ = _check_nk("start", subject_ids, prediction_times, start_durations_days, start_code_index)
    if prediction_times.shape != (n,):
        # Broadcast below assumes one prediction time per context; an (N, K) array would silently
        # produce a wrongly-shaped result instead of raising.
        raise ValueError("prediction_times must be (N,), one per context")

    by_event = start_code_index >= 0
    start_times = prediction_times[:, None] + _duration_offsets_us(start_durations_days, by_event)
    if by_event.any():
        rows = np.nonzero(by_event)[0]
        start_times[by_event] = next_occurrence_after(
            table, subject_ids[rows], prediction_times[rows], start_code_index[by_event]
        )
    return start_times


def resolve_end_times(
    table: IntervalTable,
    subject_ids: np.ndarray,
    start_times: np.ndarray,
    durations_days: np.ndarray,
    bound_code_index: np.ndarray,
) -> np.ndarray:
    """Resolve the ``(N, K)`` ``int64`` window-end matrix relative to the resolved starts.

    Args:
        table: Interval table of the shard the contexts live in.
        subject_ids: ``(N,)``.
        start_times: ``(N, K)`` ``int64`` µs from :func:`resolve_start_times` (:data:`INF` = unresolved).
        durations_days: ``(N, K)`` float days after the *resolved start*; ignored where
            ``bound_code_index >= 0``.
        bound_code_index: ``(N, K)`` ``int64``; ``-1`` marks a duration-bounded slot, otherwise the
            vocabulary index of the boundary code.

    Duration slot: ``start + round(days * US_PER_DAY)``, saturating so an unresolved start stays
    exactly :data:`INF` (never ``INF + offset``).  Event slot: the first occurrence of the boundary code
    strictly after the resolved start (occurrences before or at the start are ignored; an unresolved
    start yields :data:`INF`), or :data:`INF` when it never recurs.  At most ``N x K`` point searches.

    Examples:
        >>> t = build_interval_table(np.array([1, 1, 1]), np.array([3, 6, 9]), np.array([0, 1, 0]))
        >>> st = np.array([[4, 4, INF, INF]]); d = np.array([[1e-11, 0.0, 5.0, 0.0]])
        >>> b = np.array([[-1, 0, -1, 0]])
        >>> resolve_end_times(t, np.array([1]), st, d, b).tolist() == [[5, 9, INF, INF]]
        True
    """
    subject_ids = np.asarray(subject_ids, dtype=np.int64)
    start_times = np.asarray(start_times, dtype=np.int64)
    durations_days = np.asarray(durations_days, dtype=np.float64)
    bound_code_index = np.asarray(bound_code_index, dtype=np.int64)
    n, k = _check_nk("bound", subject_ids, start_times, durations_days, bound_code_index)
    if start_times.shape != (n, k):
        raise ValueError("shape mismatch between contexts and (N, K) bound arrays")

    by_event = bound_code_index >= 0
    offsets = _duration_offsets_us(durations_days, by_event)
    end_times = np.where(start_times == INF, INF, start_times + offsets)
    if by_event.any():
        rows = np.nonzero(by_event)[0]
        end_times[by_event] = next_occurrence_after(
            table, subject_ids[rows], start_times[by_event], bound_code_index[by_event]
        )
    return end_times


def resolve_bound_times(
    table: IntervalTable,
    subject_ids: np.ndarray,
    prediction_times: np.ndarray,
    durations_days: np.ndarray,
    bound_code_index: np.ndarray,
) -> np.ndarray:
    """Resolve the ``(N, K)`` ``int64`` boundary-time matrix for windows opening at the prediction time.

    The issue #20 form: every window starts at ``prediction_times`` (broadcast over ``K``) and the
    ends are resolved by :func:`resolve_end_times` relative to it.

    Examples:
        >>> t = build_interval_table(np.array([1, 1, 1]), np.array([3, 6, 9]), np.array([0, 1, 0]))
        >>> d = np.array([[1e-11, 0.0]]); b = np.array([[-1, 1]])  # 1e-11 days rounds to 1 us
        >>> resolve_bound_times(t, np.array([1]), np.array([4]), d, b).tolist()
        [[5, 6]]
    """
    subject_ids = np.asarray(subject_ids, dtype=np.int64)
    prediction_times = np.asarray(prediction_times, dtype=np.int64)
    durations_days = np.asarray(durations_days, dtype=np.float64)
    bound_code_index = np.asarray(bound_code_index, dtype=np.int64)
    n, k = _check_nk("bound", subject_ids, prediction_times, durations_days, bound_code_index)
    if prediction_times.shape != (n,):
        raise ValueError("shape mismatch between contexts and (N, K) bound arrays")
    start_times = np.broadcast_to(prediction_times[:, None], (n, k)).copy()
    return resolve_end_times(table, subject_ids, start_times, durations_days, bound_code_index)


# ---------------------------------------------------------------------------
# Subject-sorted interval-range labeling
# ---------------------------------------------------------------------------


def _iter_subject_runs(sorted_subject_ids: np.ndarray) -> Iterator[tuple[int, int, int]]:
    """Yield ``(subject_id, lo, hi)`` runs of a subject-sorted id array."""
    if sorted_subject_ids.size == 0:
        return
    change = np.flatnonzero(sorted_subject_ids[1:] != sorted_subject_ids[:-1]) + 1
    bounds = np.concatenate(([0], change, [sorted_subject_ids.size]))
    for lo, hi in pairwise(bounds):
        yield int(sorted_subject_ids[lo]), int(lo), int(hi)


def _fill_dense_for_subject(
    dense: np.ndarray,
    table: IntervalTable,
    s_idx: int,
    row_lo: int,
    row_hi: int,
    prediction_times: np.ndarray,
    bound_times: np.ndarray,
    max_matches: int,
) -> int:
    """Set ``dense[row_lo:row_hi, :, code]`` for one subject's rows by scanning its intervals once.

    ``prediction_times[row_lo:row_hi]`` are the rows' lookup times (window starts) and must be
    ascending (the chunk is subject/time sorted).  For every interval ``(code, start, end)`` of the
    subject the matching rows are exactly ``searchsorted(p, start, "left") : searchsorted(p, end,
    "left")`` (``start <= t < end``), and each of those rows fires ``code`` for every bound with
    ``end < bound_time``.  A lookup time of :data:`INF` matches no interval, so its bits stay false.

    Matches are emitted in blocks of at most ``max_matches`` ``(interval, context)`` pairs so the
    scratch for the expansion stays bounded independently of how many distinct codes the subject has.
    Returns the number of matches emitted.
    """
    a, b = int(table.subject_offsets[s_idx]), int(table.subject_offsets[s_idx + 1])
    if a == b:
        return 0
    p = prediction_times[row_lo:row_hi]
    los = np.searchsorted(p, table.start[a:b], side="left")
    his = np.searchsorted(p, table.end[a:b], side="left")
    counts = his - los
    nz = np.flatnonzero(counts > 0)
    if nz.size == 0:
        return 0
    counts = counts[nz]
    cum = np.cumsum(counts)
    total = int(cum[-1])

    emitted = 0
    block_start = 0
    while block_start < nz.size:
        # Largest block whose cumulative match count stays within the budget (always >= 1 interval).
        base = int(cum[block_start - 1]) if block_start else 0
        block_end = int(np.searchsorted(cum, base + max_matches, side="right"))
        block_end = max(block_end, block_start + 1)
        sel = nz[block_start:block_end]
        n_sel = counts[block_start:block_end]
        n_total = int(cum[block_end - 1]) - base

        iv = np.repeat(sel, n_sel)
        within = np.arange(n_total) - np.repeat(np.cumsum(n_sel) - n_sel, n_sel)
        rows = row_lo + np.repeat(los[sel], n_sel) + within
        codes = table.code_index[a + iv]
        ends = table.end[a + iv]
        dense[rows, :, codes] = ends[:, None] < bound_times[rows, :]

        emitted += n_total
        block_start = block_end
    assert emitted == total
    return emitted


def iter_packed_label_chunks(
    table: IntervalTable,
    subject_ids: np.ndarray,
    prediction_times: np.ndarray,
    bound_times: np.ndarray,
    vocab_size: int,
    chunk_rows: int,
    *,
    max_matches: int | None = None,
) -> Iterator[tuple[int, int, np.ndarray]]:
    """Yield ``(row_lo, row_hi, packed)`` for consecutive chunks of at most ``chunk_rows`` rows.

    A row is a lookup time (the window start: the prediction time in the #20 form, the resolved start
    of a flattened window in the #24 form) with ``K`` bound times.  ``subject_ids`` /
    ``prediction_times`` / ``bound_times`` must already be sorted by ``(subject_id, lookup time)``.
    One ``(chunk_rows, K, V)`` boolean scratch is allocated per call and reused; each subject's
    interval slice is scanned once per chunk that holds any of its rows (a subject with more than
    ``chunk_rows`` rows is split across chunks and its slice rescanned per chunk - never expanded to
    its unbounded row count).  ``packed`` is ``np.packbits(dense, axis=-1, bitorder="little")`` with
    shape ``(rows, K, ceil(V / 8))``.
    """
    subject_ids = np.asarray(subject_ids, dtype=np.int64)
    prediction_times = np.asarray(prediction_times, dtype=np.int64)
    bound_times = np.asarray(bound_times, dtype=np.int64)
    n, k = bound_times.shape
    if subject_ids.shape != (n,) or prediction_times.shape != (n,):
        raise ValueError("subject_ids / prediction_times must be (N,) matching bound_times (N, K)")
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be >= 1")
    if n > 1:
        order_ok = (subject_ids[1:] > subject_ids[:-1]) | (
            (subject_ids[1:] == subject_ids[:-1]) & (prediction_times[1:] >= prediction_times[:-1])
        )
        if not order_ok.all():
            raise ValueError("rows must be sorted by (subject_id, lookup time)")
    if max_matches is None:
        max_matches = max(chunk_rows * k * 8, 1 << 16)

    dense = np.zeros((min(chunk_rows, max(n, 0)), k, vocab_size), dtype=bool)
    for lo in range(0, n, chunk_rows):
        hi = min(lo + chunk_rows, n)
        view = dense[: hi - lo]
        view[...] = False
        for sid, r_lo, r_hi in _iter_subject_runs(subject_ids[lo:hi]):
            s_idx = table.subject_index(np.array([sid]))[0]
            if s_idx < 0:
                continue
            _fill_dense_for_subject(
                view, table, int(s_idx), r_lo, r_hi, prediction_times[lo:hi], bound_times[lo:hi], max_matches
            )
        yield lo, hi, np.packbits(view, axis=-1, bitorder="little")


def label_sorted_contexts_packed(
    table: IntervalTable,
    subject_ids: np.ndarray,
    prediction_times: np.ndarray,
    bound_times: np.ndarray,
    vocab_size: int,
    out: np.ndarray,
    chunk_rows: int,
    *,
    max_matches: int | None = None,
) -> int:
    """Label subject-sorted contexts and write packed rows into ``out`` (``(N, K, ceil(V/8))`` uint8).

    ``out`` is typically a writable ``np.lib.format.open_memmap``; each chunk is packed and stored the
    moment it is labeled, so no unpacked shard-wide tensor ever exists.  Returns the number of rows
    written.

    Examples:
        >>> t = build_interval_table(np.array([1, 1, 1]), np.array([3, 6, 9]), np.array([0, 1, 0]))
        >>> out = np.zeros((2, 1, 1), np.uint8)
        >>> bt = np.array([[7], [9]])  # windows (4, 7) and (4, 9): code 1 at 6 fires, code 0 at 9 does not
        >>> label_sorted_contexts_packed(t, np.array([1, 1]), np.array([4, 4]), bt, 2, out, chunk_rows=1)
        2
        >>> np.unpackbits(out, axis=-1, count=2, bitorder="little").tolist()
        [[[0, 1]], [[0, 1]]]
    """
    n = int(bound_times.shape[0])
    if tuple(out.shape) != (n, bound_times.shape[1], (vocab_size + 7) // 8):
        raise ValueError(
            f"out has shape {out.shape}, expected {(n, bound_times.shape[1], (vocab_size + 7) // 8)}"
        )
    written = 0
    for lo, hi, packed in iter_packed_label_chunks(
        table, subject_ids, prediction_times, bound_times, vocab_size, chunk_rows, max_matches=max_matches
    ):
        out[lo:hi] = packed
        written += hi - lo
    return written


# ---------------------------------------------------------------------------
# Small reference helpers - tests only
# ---------------------------------------------------------------------------


def label_dense_reference(
    table: IntervalTable,
    subject_ids: np.ndarray,
    prediction_times: np.ndarray,
    bound_times: np.ndarray,
    vocab_size: int,
) -> np.ndarray:
    """Dense ``(N, K, V)`` boolean labels via ``V`` point searches per context (tests only).

    Independent of the interval-range scan: it uses :func:`next_occurrence_after` for every code, so
    it cross-checks the range algorithm against the point-search one.  Unbounded in memory by design.
    """
    subject_ids = np.asarray(subject_ids, dtype=np.int64)
    prediction_times = np.asarray(prediction_times, dtype=np.int64)
    bound_times = np.asarray(bound_times, dtype=np.int64)
    n, k = bound_times.shape
    dense = np.zeros((n, k, vocab_size), dtype=bool)
    for code in range(vocab_size):
        nxt = next_occurrence_after(table, subject_ids, prediction_times, np.full(n, code))
        dense[:, :, code] = nxt[:, None] < bound_times
    return dense


def naive_multibound_labels(
    events: list[tuple[int, int, int]],
    contexts: list[tuple[int, int]],
    bounds: list[list[tuple[float | None, int | None]]],
    vocab_size: int,
    num_bounds: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Plain-Python oracle: ``(bound_times (N, K), targets (N, K, V))`` from the written semantics.

    ``events`` are ``(subject_id, time_us, code_index)`` tuples; ``contexts`` are
    ``(subject_id, prediction_time_us)``; ``bounds[i][k]`` is ``(duration_days, None)`` for a duration
    slot or ``(None, code_index)`` for an event slot.  Shares no code with the table implementation.
    """
    n, k = len(contexts), num_bounds
    bound_times = np.full((n, k), INF, dtype=np.int64)
    targets = np.zeros((n, k, vocab_size), dtype=bool)
    for i, (sid, pt) in enumerate(contexts):
        future = [(t, c) for s, t, c in events if s == sid and t > pt]
        for j, (dur, bcode) in enumerate(bounds[i]):
            if bcode is None:
                bt = pt + round(dur * US_PER_DAY)
            else:
                hits = [t for t, c in future if c == bcode]
                bt = min(hits) if hits else INF
            bound_times[i, j] = bt
            for t, c in future:
                if t < bt:
                    targets[i, j, c] = True
    return bound_times, targets


def naive_window_labels(
    events: list[tuple[int, int, int]],
    contexts: list[tuple[int, int]],
    windows: list[list[tuple[tuple[float | None, int | None], tuple[float | None, int | None]]]],
    vocab_size: int,
    num_bounds: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plain-Python oracle for issue #24 windows: ``(start_times, end_times, targets)``.

    ``events`` are ``(subject_id, time_us, code_index)``; ``contexts`` are ``(subject_id,
    prediction_time_us)``; ``windows[i][k]`` is ``(start_spec, end_spec)`` where each spec is
    ``(duration_days, None)`` or ``(None, code_index)``.  The start resolves against the prediction
    time (duration: ``pt + days``; event: first occurrence strictly after ``pt``, else :data:`INF`);
    the end against the resolved start (duration: ``start + days``, :data:`INF` if the start is; event:
    first occurrence strictly after the start, else :data:`INF`).  ``targets[i, k, v]`` is
    ``start < t_v < end`` for some occurrence ``t_v`` of ``v``.  Shares no code with the table.
    """
    n, k = len(contexts), num_bounds
    start_times = np.full((n, k), INF, dtype=np.int64)
    end_times = np.full((n, k), INF, dtype=np.int64)
    targets = np.zeros((n, k, vocab_size), dtype=bool)
    for i, (sid, pt) in enumerate(contexts):
        subj = [(t, c) for s, t, c in events if s == sid]
        for j, ((s_dur, s_code), (e_dur, e_code)) in enumerate(windows[i]):
            if s_code is None:
                st = pt + round(s_dur * US_PER_DAY)
            else:
                hits = [t for t, c in subj if c == s_code and t > pt]
                st = min(hits) if hits else INF
            if st == INF:
                et = INF
            elif e_code is None:
                et = st + round(e_dur * US_PER_DAY)
            else:
                hits = [t for t, c in subj if c == e_code and t > st]
                et = min(hits) if hits else INF
            start_times[i, j], end_times[i, j] = st, et
            for t, c in subj:
                if st < t < et:
                    targets[i, j, c] = True
    return start_times, end_times, targets
