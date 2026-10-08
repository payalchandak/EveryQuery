"""Pure interval-table kernel tests: construction, next-occurrence semantics, and the naive oracle."""

import numpy as np
import pytest

from every_query.generate_tasks import interval_table as it
from every_query.generate_tasks.interval_table import (
    INF,
    NEG_INF,
    US_PER_DAY,
    build_interval_table,
    iter_packed_label_chunks,
    label_dense_reference,
    label_sorted_contexts_packed,
    naive_multibound_labels,
    next_occurrence_after,
    resolve_bound_times,
)

DAY = US_PER_DAY


def test_build_chains_intervals_per_subject_code() -> None:
    t = build_interval_table(np.array([1, 1, 1, 2]), np.array([3, 9, 6, 5]), np.array([0, 0, 1, 0]))
    assert t.subjects.tolist() == [1, 2]
    assert t.code_index.tolist() == [0, 0, 1, 0]
    assert t.start.tolist() == [NEG_INF, 3, NEG_INF, NEG_INF]
    assert t.end.tolist() == [3, 9, 6, 5]
    assert t.subject_offsets.tolist() == [0, 3, 4]
    assert t.group_offsets.tolist() == [0, 2, 3, 4]
    assert t.n_rows == 4


def test_build_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="same length"):
        build_interval_table(np.array([1]), np.array([1, 2]), np.array([0]))
    with pytest.raises(ValueError, match="non-negative"):
        build_interval_table(np.array([1]), np.array([1]), np.array([-1]))


def test_empty_table() -> None:
    t = build_interval_table(
        np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    )
    assert t.n_rows == 0 and t.n_subjects == 0
    assert next_occurrence_after(t, np.array([1]), np.array([5]), np.array([0])).tolist() == [INF]
    out = np.zeros((1, 2, 1), np.uint8)
    assert label_sorted_contexts_packed(t, np.array([1]), np.array([5]), np.array([[10, 20]]), 3, out, 4) == 1
    assert out.sum() == 0


def test_next_occurrence_is_strictly_after() -> None:
    t = build_interval_table(np.array([1, 1, 1]), np.array([3, 6, 9]), np.array([0, 1, 0]))
    qs, qc = np.array([1] * 5), np.array([0, 0, 0, 1, 1])
    got = next_occurrence_after(t, qs, np.array([2, 3, 9, 5, 6]), qc)
    # pt=3: the occurrence *at* 3 does not qualify; pt=9: nothing follows; pt=6: the code-1 event at 6 is out.
    assert got.tolist() == [3, 9, INF, 6, INF]


def test_unknown_subject_and_unknown_code_resolve_to_inf() -> None:
    t = build_interval_table(np.array([1, 1]), np.array([3, 6]), np.array([0, 1]))
    assert next_occurrence_after(t, np.array([7, 1]), np.array([0, 0]), np.array([0, 5])).tolist() == [
        INF,
        INF,
    ]


@pytest.mark.parametrize("vocab_size", [None, 4])
def test_absent_high_code_never_aliases_another_subjects_group(vocab_size: int | None) -> None:
    # Two subjects, only code 1 observed in the shard -> a shard-local code field is 1 bit wide.  A
    # globally valid code 3 (= 0b11) for subject 1 must NOT resolve to subject 2's code-1 group: with a
    # 1-bit field ``(0 << 1) | 3 == (1 << 1) | 1``.  Regression for the PR #21 review, blocker 1.
    t = build_interval_table(np.array([1, 2]), np.array([5, 5]), np.array([1, 1]), vocab_size=vocab_size)
    got = next_occurrence_after(t, np.array([1, 2, 1]), np.array([0, 0, 0]), np.array([3, 3, 1]))
    assert got.tolist() == [INF, INF, 5]
    bt = resolve_bound_times(t, np.array([1]), np.array([0]), np.array([[0.0]]), np.array([[3]]))
    assert bt.tolist() == [[INF]]
    if vocab_size is not None:
        with pytest.raises(ValueError, match="outside the vocabulary"):
            build_interval_table(np.array([1]), np.array([5]), np.array([4]), vocab_size=4)


def test_duplicate_events_do_not_change_labels() -> None:
    plain = build_interval_table(np.array([1, 1]), np.array([3, 6]), np.array([0, 0]))
    dup = build_interval_table(np.array([1, 1, 1, 1]), np.array([3, 3, 6, 6]), np.array([0, 0, 0, 0]))
    qs, qt = np.array([1, 1, 1]), np.array([2, 3, 5])
    for tab in (plain, dup):
        assert next_occurrence_after(tab, qs, qt, np.zeros(3, int)).tolist() == [3, 6, 6]
    bt = np.array([[7], [7], [7]])
    a, b = np.zeros((3, 1, 1), np.uint8), np.zeros((3, 1, 1), np.uint8)
    label_sorted_contexts_packed(plain, qs, qt, bt, 1, a, 2)
    label_sorted_contexts_packed(dup, qs, qt, bt, 1, b, 2)
    assert np.array_equal(a, b)
    # A zero-width duplicate interval never matches a prediction time sitting exactly on it.
    zero_width = dup.start == dup.end
    assert zero_width.sum() == 2


def test_resolve_bound_times_duration_rounding_and_event_inf() -> None:
    t = build_interval_table(np.array([1, 1]), np.array([DAY, 3 * DAY]), np.array([2, 4]))
    d = np.array([[0.5, 1.0 / 3.0, 0.0, 0.0]])
    b = np.array([[-1, -1, 4, 5]])
    bt = resolve_bound_times(t, np.array([1]), np.array([0]), d, b)
    assert bt[0, 0] == DAY // 2
    assert bt[0, 1] == round(DAY / 3.0)
    assert bt[0, 2] == 3 * DAY
    assert bt[0, 3] == INF


def test_resolve_bound_times_shape_mismatch() -> None:
    t = build_interval_table(np.array([1]), np.array([1]), np.array([0]))
    with pytest.raises(ValueError, match="shape mismatch"):
        resolve_bound_times(t, np.array([1]), np.array([0]), np.zeros((1, 2)), np.zeros((1, 3), int))


def test_labeling_requires_sorted_contexts() -> None:
    t = build_interval_table(np.array([1, 2]), np.array([1, 1]), np.array([0, 0]))
    with pytest.raises(ValueError, match="sorted"):
        list(iter_packed_label_chunks(t, np.array([2, 1]), np.array([0, 0]), np.zeros((2, 1), int), 1, 4))


def _random_case(rng: np.random.Generator, k: int = 5):
    n_ev, n_ctx, v = int(rng.integers(0, 200)), int(rng.integers(0, 60)), int(rng.integers(1, 12))
    es = rng.integers(0, 8, n_ev)
    et = rng.integers(0, 50, n_ev) * DAY // 4
    ec = rng.integers(0, v, n_ev)
    qs = rng.integers(0, 10, n_ctx)
    qt = rng.integers(0, 50, n_ctx) * DAY // 4
    order = np.lexsort((qt, qs))
    qs, qt = qs[order], qt[order]
    dur = rng.uniform(0, 30, (n_ctx, k))
    bc = np.where(rng.random((n_ctx, k)) < 0.5, rng.integers(0, v, (n_ctx, k)), -1)
    return es, et, ec, qs, qt, dur, bc, v


@pytest.mark.parametrize("seed", range(25))
def test_fuzz_against_naive_oracle(seed: int) -> None:
    rng = np.random.default_rng(seed)
    es, et, ec, qs, qt, dur, bc, v = _random_case(rng)
    k = bc.shape[1]
    t = build_interval_table(es, et, ec)
    bt = resolve_bound_times(t, qs, qt, dur, bc)

    events = list(zip(es.tolist(), et.tolist(), ec.tolist(), strict=True))
    bounds = [
        [(None, int(bc[i, j])) if bc[i, j] >= 0 else (float(dur[i, j]), None) for j in range(k)]
        for i in range(len(qs))
    ]
    naive_bt, naive_targets = naive_multibound_labels(
        events, list(zip(qs.tolist(), qt.tolist(), strict=True)), bounds, v, k
    )
    assert np.array_equal(naive_bt, bt)
    assert np.array_equal(label_dense_reference(t, qs, qt, bt, v), naive_targets)

    for chunk_rows in (1, 3, 1000):
        out = np.zeros((len(qs), k, (v + 7) // 8), np.uint8)
        label_sorted_contexts_packed(t, qs, qt, bt, v, out, chunk_rows, max_matches=7)
        got = np.unpackbits(out, axis=-1, count=v, bitorder="little").astype(bool)
        assert np.array_equal(got, naive_targets), chunk_rows


def test_chunked_and_unchunked_are_identical_and_chunks_are_bounded() -> None:
    rng = np.random.default_rng(99)
    es, et, ec, qs, qt, dur, bc, v = _random_case(rng)
    t = build_interval_table(es, et, ec)
    bt = resolve_bound_times(t, qs, qt, dur, bc)
    whole = (
        np.concatenate([p for _, _, p in iter_packed_label_chunks(t, qs, qt, bt, v, max(len(qs), 1))])
        if len(qs)
        else None
    )
    sizes = []
    parts = []
    for lo, hi, packed in iter_packed_label_chunks(t, qs, qt, bt, v, 4):
        sizes.append(hi - lo)
        parts.append(packed)
    assert all(s <= 4 for s in sizes)
    if whole is not None:
        assert np.array_equal(np.concatenate(parts), whole)


def test_one_subject_with_more_contexts_than_chunk_rows_stays_bounded(monkeypatch) -> None:
    """A single subject spanning many chunks is rescanned per chunk; scratch never exceeds chunk_rows."""
    rng = np.random.default_rng(5)
    n_ctx, v, k = 50, 9, 5
    es = np.zeros(80, int)
    et = rng.integers(0, 100, 80) * DAY
    ec = rng.integers(0, v, 80)
    qs = np.zeros(n_ctx, int)
    qt = np.sort(rng.integers(0, 100, n_ctx) * DAY)
    dur = rng.uniform(0, 50, (n_ctx, k))
    bc = np.where(rng.random((n_ctx, k)) < 0.5, rng.integers(0, v, (n_ctx, k)), -1)
    t = build_interval_table(es, et, ec)
    bt = resolve_bound_times(t, qs, qt, dur, bc)

    seen_shapes = []
    real = it._fill_dense_for_subject

    def spy(dense, *args, **kwargs):
        seen_shapes.append(dense.shape)
        return real(dense, *args, **kwargs)

    monkeypatch.setattr(it, "_fill_dense_for_subject", spy)
    out = np.zeros((n_ctx, k, (v + 7) // 8), np.uint8)
    label_sorted_contexts_packed(t, qs, qt, bt, v, out, chunk_rows=7)
    assert seen_shapes and all(s[0] <= 7 and s[1:] == (k, v) for s in seen_shapes)
    assert len(seen_shapes) == -(-n_ctx // 7)  # one scan of the subject's slice per chunk
    expect = label_dense_reference(t, qs, qt, bt, v)
    assert np.array_equal(np.unpackbits(out, axis=-1, count=v, bitorder="little").astype(bool), expect)


def test_packbits_roundtrip_with_non_multiple_of_eight_vocab() -> None:
    v = 13
    dense = np.random.default_rng(0).random((4, 5, v)) < 0.3
    packed = np.packbits(dense, axis=-1, bitorder="little")
    assert packed.shape == (4, 5, 2)
    assert np.array_equal(np.unpackbits(packed, axis=-1, count=v, bitorder="little").astype(bool), dense)


# --- issue #24: explicit window starts -------------------------------------------------------------


def test_resolve_start_times_duration_and_event_forms() -> None:
    # Subject 1: code 0 at days 3 and 9, code 1 at day 6.  pt = day 3 (an occurrence of code 0 sits
    # exactly at pt and must not open the window).
    t = build_interval_table(np.array([1, 1, 1]), np.array([3 * DAY, 6 * DAY, 9 * DAY]), np.array([0, 1, 0]))
    d = np.array([[0.0, 7.0, 0.0, 0.0, 0.0]])
    c = np.array([[-1, -1, 0, 1, 5]])
    st = it.resolve_start_times(t, np.array([1]), np.array([3 * DAY]), d, c)
    assert st.tolist() == [[3 * DAY, 10 * DAY, 9 * DAY, 6 * DAY, INF]]
    # Unknown subject: every event start is unresolved, duration starts still resolve.
    st = it.resolve_start_times(t, np.array([7]), np.array([0]), d[:, :3], c[:, :3])
    assert st.tolist() == [[0, 7 * DAY, INF]]
    with pytest.raises(ValueError, match="shape mismatch"):
        it.resolve_start_times(t, np.array([1]), np.array([0]), np.zeros((1, 2)), np.zeros((1, 3), int))
    # One prediction time per context: an (N, K) reference would broadcast to a wrongly-shaped result
    # instead of raising (resolve_end_times legitimately takes the (N, K) resolved starts).
    with pytest.raises(ValueError, match=r"prediction_times must be \(N,\)"):
        it.resolve_start_times(t, np.array([1]), np.zeros((1, 3), int), np.zeros((1, 3)), np.full((1, 3), -1))


def test_resolve_end_times_is_relative_to_start_and_strictly_after_it() -> None:
    # code 0 at days 8 and 20, code 1 at day 15.
    t = build_interval_table(
        np.array([1, 1, 1]), np.array([8 * DAY, 20 * DAY, 15 * DAY]), np.array([0, 0, 1])
    )
    st = np.array([[10 * DAY, 10 * DAY, 10 * DAY, 8 * DAY, 20 * DAY]])
    d = np.array([[30.0, 0.0, 0.0, 0.0, 0.0]])
    b = np.array([[-1, 0, 1, 0, 0]])
    et = it.resolve_end_times(t, np.array([1]), st, d, b)
    # day 10 + 30 = day 40 (never day 30); the day-8 occurrence of code 0 is ignored -> day 20;
    # a boundary occurrence exactly at the start does not close the window -> day 20; none after
    # day 20 -> INF.
    assert et.tolist() == [[40 * DAY, 20 * DAY, 15 * DAY, 20 * DAY, INF]]
    with pytest.raises(ValueError, match="shape mismatch"):
        it.resolve_end_times(t, np.array([1]), np.zeros((1, 2), int), np.zeros((1, 3)), np.zeros((1, 3), int))


def test_inf_start_plus_duration_is_exactly_inf() -> None:
    t = build_interval_table(np.array([1]), np.array([DAY]), np.array([0]))
    st = np.array([[INF, INF, INF]])
    d = np.array([[1826.0, 1e6, 0.0]])
    b = np.array([[-1, -1, 0]])
    et = it.resolve_end_times(t, np.array([1]), st, d, b)
    assert et.tolist() == [[INF, INF, INF]]
    assert et.dtype == np.int64
    # And the point search at INF itself is INF (no wraparound into a negative rank).
    assert next_occurrence_after(t, np.array([1]), np.array([INF]), np.array([0])).tolist() == [INF]


def test_resolve_bound_times_is_the_prediction_time_start_wrapper() -> None:
    rng = np.random.default_rng(3)
    es, et, ec, qs, qt, dur, bc, v = _random_case(rng)
    t = build_interval_table(es, et, ec)
    k = bc.shape[1]
    st = np.broadcast_to(qt[:, None], (len(qs), k))
    assert np.array_equal(resolve_bound_times(t, qs, qt, dur, bc), it.resolve_end_times(t, qs, st, dur, bc))
    assert np.array_equal(
        it.resolve_start_times(t, qs, qt, np.zeros((len(qs), k)), np.full((len(qs), k), -1)), st
    )


def test_labeling_sort_error_names_the_lookup_time() -> None:
    t = build_interval_table(np.array([1, 2]), np.array([1, 1]), np.array([0, 0]))
    with pytest.raises(ValueError, match="sorted by \\(subject_id, lookup time\\)"):
        list(iter_packed_label_chunks(t, np.array([1, 1]), np.array([5, 0]), np.zeros((2, 1), int), 1, 4))


def _random_window_case(rng: np.random.Generator, k: int = 4):
    """Events + contexts + per-position (start, end) specs covering all six combinations."""
    n_ev, n_ctx, v = int(rng.integers(0, 150)), int(rng.integers(0, 40)), int(rng.integers(1, 10))
    es = rng.integers(0, 6, n_ev)
    et = rng.integers(0, 40, n_ev) * DAY // 2
    ec = rng.integers(0, v, n_ev)
    qs = rng.integers(0, 8, n_ctx)
    qt = rng.integers(0, 40, n_ctx) * DAY // 2
    form = rng.integers(0, 3, (n_ctx, k))  # 0 = prediction time, 1 = positive duration, 2 = event
    s_dur = np.where(form == 1, rng.uniform(0.5, 20, (n_ctx, k)), 0.0)
    s_code = np.where(form == 2, rng.integers(0, v, (n_ctx, k)), -1)
    e_dur = rng.uniform(0, 25, (n_ctx, k))
    e_code = np.where(rng.random((n_ctx, k)) < 0.5, rng.integers(0, v, (n_ctx, k)), -1)
    return es, et, ec, qs, qt, s_dur, s_code, e_dur, e_code, v


def _flatten_sort_label_scatter(t, qs, st, et, v, chunk_rows: int) -> np.ndarray:
    """The #24 labeling path at kernel level: flatten (N, K) windows, sort by (subject, start), run the K'=1
    packed kernel, scatter back.

    Mirrors ``label_multitask_index``; returns dense (N, K, V).
    """
    n, k = st.shape
    subj_flat = np.repeat(qs, k)
    order = np.lexsort((st.ravel(), subj_flat))
    packed = np.zeros((n, k, (v + 7) // 8), np.uint8)
    for lo, hi, chunk in iter_packed_label_chunks(
        t, subj_flat[order], st.ravel()[order], et.ravel()[order][:, None], v, chunk_rows, max_matches=5
    ):
        sel = order[lo:hi]
        packed[sel // k, sel % k] = chunk[:, 0]
    return np.unpackbits(packed, axis=-1, count=v, bitorder="little").astype(bool)


@pytest.mark.parametrize("seed", range(25))
def test_fuzz_window_starts_against_naive_oracle(seed: int) -> None:
    rng = np.random.default_rng(seed)
    es, et, ec, qs, qt, s_dur, s_code, e_dur, e_code, v = _random_window_case(rng)
    n, k = s_code.shape
    t = build_interval_table(es, et, ec)
    st = it.resolve_start_times(t, qs, qt, s_dur, s_code)
    en = it.resolve_end_times(t, qs, st, e_dur, e_code)

    events = list(zip(es.tolist(), et.tolist(), ec.tolist(), strict=True))
    windows = [
        [
            (
                (None, int(s_code[i, j])) if s_code[i, j] >= 0 else (float(s_dur[i, j]), None),
                (None, int(e_code[i, j])) if e_code[i, j] >= 0 else (float(e_dur[i, j]), None),
            )
            for j in range(k)
        ]
        for i in range(n)
    ]
    naive_st, naive_en, naive_targets = it.naive_window_labels(
        events, list(zip(qs.tolist(), qt.tolist(), strict=True)), windows, v, k
    )
    assert np.array_equal(naive_st, st)
    assert np.array_equal(naive_en, en)
    # Point-search reference on the flattened windows (start as the lookup time).
    ref = label_dense_reference(t, np.repeat(qs, k), st.ravel(), en.ravel()[:, None], v).reshape(n, k, v)
    assert np.array_equal(ref, naive_targets)
    # Range kernel: flatten / sort / scatter, chunk sizes forcing one context's windows across chunks.
    for chunk_rows in (1, 2, 3, 1000):
        assert np.array_equal(_flatten_sort_label_scatter(t, qs, st, en, v, chunk_rows), naive_targets)


def test_windows_of_one_context_straddle_chunks_and_unresolved_starts_are_all_false() -> None:
    # One subject, one context, K=3 windows whose resolved starts interleave with a second context's.
    es = np.zeros(6, int)
    et = np.array([1, 2, 3, 4, 5, 6]) * DAY
    ec = np.array([0, 1, 2, 0, 2, 2])  # code 1 occurs only at day 2
    t = build_interval_table(es, et, ec)
    qs = np.array([0, 0])
    qt = np.array([0, 0])
    s_dur = np.array([[0.0, 3.5, 0.0], [1.5, 0.0, 0.0]])
    s_code = np.array([[-1, -1, 7], [-1, 1, -1]])  # code 7 never occurs -> INF start
    e_dur = np.array([[10.0, 10.0, 10.0], [10.0, 10.0, 10.0]])
    e_code = np.full((2, 3), -1)
    st = it.resolve_start_times(t, qs, qt, s_dur, s_code)
    assert st.tolist() == [[0, 3 * DAY + DAY // 2, INF], [DAY + DAY // 2, 2 * DAY, 0]]
    en = it.resolve_end_times(t, qs, st, e_dur, e_code)
    assert en[0, 2] == INF
    expect = np.zeros((2, 3, 3), bool)
    expect[0, 0] = [True, True, True]  # (0, 10): everything
    expect[0, 1] = [True, False, True]  # (3.5, 13.5): days 4, 5, 6 -> codes 0, 2
    expect[0, 2] = [False, False, False]  # never opens
    expect[1, 0] = [True, True, True]  # (1.5, 11.5): days 2..6
    expect[1, 1] = [True, False, True]  # (2, 12): the code-1 start event at day 2 itself is excluded
    expect[1, 2] = [True, True, True]
    for chunk_rows in (1, 2, 3):
        assert np.array_equal(_flatten_sort_label_scatter(t, qs, st, en, 3, chunk_rows), expect), chunk_rows
