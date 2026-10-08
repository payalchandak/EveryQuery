"""Tests for event-bounded duration queries.

A query's window may end at the next occurrence of a **boundary event** rather than after a
fixed horizon.  Covered here, in pipeline order:

1. Labeling — that an occurrence at or after the boundary is outside the window, that time- and
   event-bounded queries can coexist in one sequence, and that the degenerate "boundary never
   fires" case behaves as documented
   (window runs to the end of the record) and is *reported* rather than hidden.
2. Sampling — that bounds are drawn on their own seed axis, so turning the feature on does not
   perturb the code/duration draw that the sampler's parity contract depends on, and that the
   boundary pool is the query universe itself.
3. Dataset/batch — that ``bound_events`` is optional on disk, and that an unknown boundary code
   raises instead of silently decaying into an unbounded query.
4. Model — that the boundary reaches the window's end slot, that it is causally local, and that
   a bound-free batch is answered *identically* to a model without the feature.

Section 4 was originally written against the encoder-decoder conditional query-sequence model
(since deleted with the rest of that pipeline), because it was the only model when event bounds
landed.  The surviving consumer is
:class:`~every_query.model.conditional_multitask_ar_model.ConditionalMultitaskARModel`,
whose windows carry a *pair* of specs — the boundary owns the window's **end**, and the issue-#27
start event owns its beginning — so "the duration slot" below is the end half of the window
token, and "block-local" is the causal ``[..., W_i, C_i, A_i, W_{i+1}, ...]`` stream rather than
a block-causal decoder mask.
"""

from datetime import datetime

import numpy as np
import polars as pl
import pytest
import torch

from every_query.data.query_seq_dataset import (
    EVENT_BOUND_DURATION_SENTINEL,
    NO_BOUND_INDEX,
    QuerySeqBatch,
)
from every_query.generate_tasks.query_sequence_labeling import (
    BOUND_COL,
    QuerySequenceDistribution,
    label_binary_occurrence,
    label_query_sequences,
    label_with_event_bounds,
    log_degenerate_bounds,
)
from every_query.generate_tasks.sample_evaluation_query_sequences import sample_sequence_specs
from every_query.model.answers import ANSWER_NO, ANSWER_YES
from every_query.model.conditional_multitask_ar_model import TYPE_WINDOW

# The multitask model's own construction idiom, reused rather than re-invented.
from tests.test_conditional_multitask_ar_model import make_batch, tiny_model

# The margin that separates "the feature is live" from float32 rounding, shared with
# ``test_feature_liveness`` so it cannot drift between the files that measure it.
from tests.test_feature_liveness import LIVE

PT = datetime(2024, 1, 1)


def _events(rows: list[tuple[int, datetime, str]]) -> pl.DataFrame:
    return pl.DataFrame(
        {"subject_id": [r[0] for r in rows], "time": [r[1] for r in rows], "code": [r[2] for r in rows]},
        schema={"subject_id": pl.Int64, "time": pl.Datetime("us"), "code": pl.Utf8},
    )


def _index(queries, durations, bounds=None, subject: int = 1) -> pl.DataFrame:
    n = len(queries)
    data = {
        "_ctx_id": [0] * n,
        "_position": list(range(n)),
        "subject_id": [subject] * n,
        "prediction_time": [PT] * n,
        "query": list(queries),
        "duration_days": [float(d) for d in durations],
    }
    schema = {
        "_ctx_id": pl.UInt32,
        "_position": pl.Int64,
        "subject_id": pl.Int64,
        "prediction_time": pl.Datetime("us"),
        "query": pl.Utf8,
        "duration_days": pl.Float32,
    }
    if bounds is not None:
        data[BOUND_COL] = list(bounds)
        schema[BOUND_COL] = pl.Utf8
    return pl.DataFrame(data, schema=schema)


# ── 1. labeling semantics ───────────────────────────────────────────────


def test_boundary_closes_the_window():
    """An occurrence after the boundary does not count, even well inside any horizon."""
    events = _events(
        [
            (1, datetime(2024, 1, 5), "DISCHARGE"),
            (1, datetime(2024, 1, 9), "SEPSIS"),  # after the discharge
        ]
    )
    idx = _index(["SEPSIS"], [EVENT_BOUND_DURATION_SENTINEL], ["DISCHARGE"])
    assert label_with_event_bounds(idx, events).row(0, named=True)["answers"] == [False]


def test_occurrence_before_boundary_counts():
    events = _events(
        [
            (1, datetime(2024, 1, 3), "SEPSIS"),
            (1, datetime(2024, 1, 5), "DISCHARGE"),
        ]
    )
    idx = _index(["SEPSIS"], [EVENT_BOUND_DURATION_SENTINEL], ["DISCHARGE"])
    assert label_with_event_bounds(idx, events).row(0, named=True)["answers"] == [True]


def test_boundary_is_strict():
    """An occurrence exactly *at* the boundary instant is outside the window.

    The window is open at the top everywhere (see the RESOLUTION note in
    ``tests/test_event_bounds_oracle.py``), and for a bounded query the top IS the boundary
    event's timestamp -- so a SEPSIS charted in the same instant as the DISCHARGE does NOT count
    as having happened before it.  That is what makes the query mean "SEPSIS before DISCHARGE".
    MEDS clusters codes onto one timestamp, so this is a common shape rather than an edge case.
    """
    same = datetime(2024, 1, 5)
    events = _events([(1, same, "SEPSIS"), (1, same, "DISCHARGE")])
    idx = _index(["SEPSIS"], [EVENT_BOUND_DURATION_SENTINEL], ["DISCHARGE"])
    assert label_with_event_bounds(idx, events).row(0, named=True)["answers"] == [False]


def test_mixed_sequence_labels_each_query_by_its_own_rule():
    """One sequence may mix both kinds; each query must use its own window."""
    events = _events(
        [
            (1, datetime(2024, 1, 3), "SEPSIS"),
            (1, datetime(2024, 1, 4), "DISCHARGE"),
            (1, datetime(2024, 1, 10), "LATE"),
        ]
    )
    idx = _index(
        ["SEPSIS", "LATE", "LATE"],
        [30.0, 30.0, EVENT_BOUND_DURATION_SENTINEL],
        [None, None, "DISCHARGE"],
    )
    answers = label_with_event_bounds(idx, events).row(0, named=True)["answers"]
    # SEPSIS within 30d: yes.  LATE within 30d: yes.  LATE before discharge (day 4): no.
    assert answers == [True, True, False]


def test_unbounded_rows_match_the_plain_labeler_exactly():
    """With every bound null, the bound-aware labeler must agree with the original."""
    events = _events(
        [
            (1, datetime(2024, 1, 3), "A"),
            (1, datetime(2024, 1, 20), "B"),
        ]
    )
    queries, durations = ["A", "B", "A"], [30.0, 5.0, 1.0]
    plain = label_binary_occurrence(_index(queries, durations), events).row(0, named=True)
    bounded = label_with_event_bounds(_index(queries, durations, [None, None, None]), events).row(
        0, named=True
    )
    assert plain["answers"] == bounded["answers"]


def test_missing_boundary_runs_to_end_of_record():
    """Documented degenerate case: no boundary occurrence -> 'does it ever occur again'."""
    events = _events([(1, datetime(2024, 6, 1), "SEPSIS")])  # no DISCHARGE at all
    idx = _index(["SEPSIS"], [EVENT_BOUND_DURATION_SENTINEL], ["DISCHARGE"])
    assert label_with_event_bounds(idx, events).row(0, named=True)["answers"] == [True]


def test_degenerate_rate_is_reported():
    """The degenerate case is legitimate but misleading, so it must be measured, not hidden."""
    events = _events([(1, datetime(2024, 1, 4), "DISCHARGE")])
    idx = _index(
        ["A", "A"],
        [EVENT_BOUND_DURATION_SENTINEL] * 2,
        ["DISCHARGE", "NEVER_HAPPENS"],
    )
    rates = log_degenerate_bounds(idx, events)
    assert rates["DISCHARGE"] == 0.0
    assert rates["NEVER_HAPPENS"] == 1.0


def test_dispatch_keys_on_the_frame_not_a_flag():
    """An index carrying bounds is always labelled bound-aware, flag or no flag."""
    events = _events([(1, datetime(2024, 1, 9), "SEPSIS"), (1, datetime(2024, 1, 5), "DISCHARGE")])
    bounded = label_query_sequences(
        _index(["SEPSIS"], [EVENT_BOUND_DURATION_SENTINEL], ["DISCHARGE"]), events
    )
    assert "bound_events" in bounded.columns
    assert bounded.row(0, named=True)["answers"] == [False]

    plain = label_query_sequences(_index(["SEPSIS"], [30.0]), events)
    assert "bound_events" not in plain.columns


# ── 2. sampling ─────────────────────────────────────────────────────────


def test_bounds_do_not_perturb_the_code_duration_draw():
    """The whole point of the separate seed axis: parity with an unbounded run is preserved."""
    plain = sample_sequence_specs(2, ["A", "B", "C"], 3, 3, 1, 365, seed=11)
    bounded = sample_sequence_specs(2, ["A", "B", "C"], 3, 3, 1, 365, seed=11, eventbound_fraction=0.5)
    assert [s.queries for s in plain] == [s.queries for s in bounded], (
        "turning bounds on must not change which codes were drawn"
    )
    # Durations change only where a bound replaced them with the sentinel.
    for p, b in zip(plain, bounded, strict=True):
        for i, bound in enumerate(b.bounds):
            if bound is None:
                assert b.durations[i] == p.durations[i]
            else:
                assert b.durations[i] == EVENT_BOUND_DURATION_SENTINEL


def _dist(**overrides) -> QuerySequenceDistribution:
    kwargs = {
        "query_codes": ["A", "B", "C"],
        "min_duration": 1.0,
        "max_duration": 365.0,
        "duration_distribution": "log-uniform",
        "min_queries": 1,
        "max_queries": 1,
    }
    kwargs.update(overrides)
    return QuerySequenceDistribution(**kwargs)


def _draw(dist: QuerySequenceDistribution, n: int, bound_seed: int | None = 7):
    return dist.sample_sequences(
        n,
        np.random.default_rng(0),
        np.random.default_rng(1),
        None if bound_seed is None else np.random.default_rng(bound_seed),
    )


def test_distribution_rejects_a_bad_bound_fraction():
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        _dist(eventbound_fraction=1.5)


def test_bound_rng_is_required_when_bounds_are_on():
    """Silently drawing bounds from another axis would perturb the query or structure stream."""
    with pytest.raises(ValueError, match="bound_rng"):
        _draw(_dist(eventbound_fraction=0.5), 4, bound_seed=None)
    assert len(_draw(_dist(), 4, bound_seed=None)) == 4, "no bounds, no generator needed"


def test_bound_draw_is_deterministic():
    dist = _dist(eventbound_fraction=0.5)
    a = [q.bound_event for s in _draw(dist, 8) for q in s]
    b = [q.bound_event for s in _draw(dist, 8) for q in s]
    assert a == b


@pytest.mark.parametrize("fraction", [0.2, 0.5, 0.8, 1.0])
def test_bound_fraction_is_realised_per_query(fraction):
    """`eventbound_fraction` is the per-query rate, and bounded rows are marked consistently.

    The rate itself is pinned more tightly in ``tests/test_event_bounds_oracle.py``; this checks
    the pairing that test does not: every bounded query carries the sentinel duration, and every
    unbounded one carries a real horizon.
    """
    n = 2000
    qs = [q for s in _draw(_dist(eventbound_fraction=fraction), n) for q in s]
    assert len(qs) == n
    bounded = [q for q in qs if q.bound_event is not None]
    assert len(bounded) / n == pytest.approx(fraction, abs=0.04)
    assert all(q.duration_days == EVENT_BOUND_DURATION_SENTINEL for q in bounded)
    assert all(q.duration_days > 0 for q in qs if q.bound_event is None)


def test_boundaries_are_drawn_from_the_query_universe():
    """No separate pool: every node a query can be, a boundary can be too."""
    bounds = {q.bound_event for s in _draw(_dist(eventbound_fraction=1.0), 300) for q in s}
    assert bounds == {"A", "B", "C"}


# ── 3. dataset / batch ──────────────────────────────────────────────────


def test_batch_without_bounds_is_still_valid():
    """The column is optional end to end; a pre-feature dataset must keep working."""
    batch = QuerySeqBatch(
        code=torch.tensor([[3, 4]]),
        numeric_value=torch.zeros(1, 2),
        numeric_value_mask=torch.zeros(1, 2, dtype=torch.bool),
        time_delta_days=torch.zeros(1, 2),
        q_codes=torch.tensor([[7, 8]]),
        q_durations=torch.tensor([[30.0, 7.0]]),
        q_answers=torch.tensor([[ANSWER_YES, ANSWER_NO]]),
        q_mask=torch.tensor([[True, True]]),
    )
    assert batch.q_bound_codes is None


def test_batch_validates_bound_shape():
    with pytest.raises(ValueError, match="q_bound_codes"):
        QuerySeqBatch(
            code=torch.tensor([[3, 4]]),
            numeric_value=torch.zeros(1, 2),
            numeric_value_mask=torch.zeros(1, 2, dtype=torch.bool),
            time_delta_days=torch.zeros(1, 2),
            q_codes=torch.tensor([[7, 8]]),
            q_durations=torch.tensor([[30.0, 7.0]]),
            q_answers=torch.tensor([[ANSWER_YES, ANSWER_NO]]),
            q_mask=torch.tensor([[True, True]]),
            q_bound_codes=torch.tensor([[1]]),  # wrong width
        )


# ── 4. model ────────────────────────────────────────────────────────────

# ``make_batch``'s default windows: window 0 is duration-bounded (7d), window 1 ends at boundary
# code 10, window 2 opens at start event 9 and is duration-bounded.  Indices are derived from the
# multitask stream layout, not transliterated from the encoder-decoder version's query blocks.
BOUNDED_WINDOW = 1


def _unbounded_batch(**over):
    """``make_batch`` with every window duration-bounded — the shape of the evaluation grid."""
    batch = make_batch(**over)
    batch.q_durations = torch.tensor([[7.0, 30.0, 4.0, 2.0, 1.0][: batch.q_durations.shape[1]]] * 2)
    batch.q_bound_codes = torch.zeros_like(batch.q_bound_codes)
    return batch


def _bound(batch, code: int, window: int = BOUNDED_WINDOW):
    batch.q_bound_codes[:, window] = code
    batch.q_durations[:, window] = EVENT_BOUND_DURATION_SENTINEL
    return batch


def test_no_bounds_is_identical_to_the_feature_being_absent():
    """The safety property: an all-zero bound column changes nothing at all.

    Stated as "changing the feature's only parameter must not move the answer", which is a
    stronger claim than the encoder-decoder original could make: there the bound column could be
    left off the batch entirely, whereas ``q_bound_codes`` is a required field of
    ``MultitaskBoundaryBatch``, so ``NO_BOUND_INDEX`` *is* the "feature absent" form.
    """
    model = tiny_model()
    batch = _unbounded_batch()
    assert (batch.q_bound_codes == NO_BOUND_INDEX).all()
    with torch.no_grad():
        _, before = model(batch)
        model.bound_marker.add_(10.0)
        _, after = model(batch)
    assert torch.equal(before.logits, after.logits), (
        "the boundary machinery reaches a window that carries no boundary"
    )


def test_boundary_code_changes_the_prediction():
    """And it is the boundary *code* that moves the answer, not merely the sentinel duration.

    Bounding a window does two things at once — it writes ``EVENT_BOUND_DURATION_SENTINEL`` into
    ``q_durations`` and a code into ``q_bound_codes`` — so a comparison against the unbounded
    batch alone is satisfied by a model that reads the sentinel through the duration MLP and
    ignores the code entirely.  The sentinel-only control separates the two.
    """
    model = tiny_model()
    sentinel_only = _unbounded_batch()
    sentinel_only.q_durations[:, BOUNDED_WINDOW] = EVENT_BOUND_DURATION_SENTINEL
    with torch.no_grad():
        _, unbounded = model(_unbounded_batch())
        _, sentinel = model(sentinel_only)
        _, bounded = model(_bound(_unbounded_batch(), 9))
    assert (bounded.logits - unbounded.logits).abs().max().item() > LIVE
    assert (bounded.logits - sentinel.logits).abs().max().item() > LIVE


def test_different_boundaries_give_different_predictions():
    """'before discharge' and 'before death' must not be the same question."""
    model = tiny_model()
    with torch.no_grad():
        _, a = model(_bound(_unbounded_batch(), 9))
        _, b = model(_bound(_unbounded_batch(), 10))
    assert (b.logits - a.logits).abs().max().item() > LIVE


def test_bound_does_not_leak_backwards_across_windows():
    """Causal structure holds: bounding a later window must not move an earlier one's answer.

    ``W_1`` physically follows ``W_0`` in the combined stream, so no property of window 1 — its
    boundary included — can reach window 0's hidden state.  The encoder-decoder version made the
    same claim about its block-causal decoder mask; here it is the ordinary causal mask over
    ``[patient..., W0, C0, A0, W1, C1, A1, W2]``.
    """
    model = tiny_model()
    with torch.no_grad():
        _, base = model(_unbounded_batch())
        _, later = model(_bound(_unbounded_batch(), 11, window=2))
    assert torch.equal(base.logits[:, :2], later.logits[:, :2]), (
        "a bound on a later window must not change an earlier window's answer"
    )
    assert (later.logits[:, 2] - base.logits[:, 2]).abs().max().item() > LIVE


def test_boundary_owns_the_window_end_and_leaves_the_start_alone():
    """A boundary replaces *what closes the window*, never *when it opens*.

    ``_window_embeds`` sums a start spec and an end spec; an event bound must swap the end
    spec's scalar-duration path for ``embedding(code) + bound_marker`` and touch nothing else.
    """
    model = tiny_model()
    batch = _bound(_unbounded_batch(), 9)
    with torch.no_grad():
        got = model._window_embeds(batch)
        starts, start_codes = model._start_fields(batch)
        embedding = model.HF_model.get_input_embeddings()
        start_spec = torch.where(
            (start_codes > 0).unsqueeze(-1),
            embedding(start_codes) + model.start_marker,
            model.start_duration_embed((starts / 365.0).unsqueeze(-1)),
        )
        end_duration = model.end_duration_embed((batch.q_durations / 365.0).unsqueeze(-1))
        end_event = embedding(batch.q_bound_codes) + model.bound_marker
        rest = model.token_type_embed.weight[TYPE_WINDOW] + model.block_pos_embed(
            torch.arange(batch.q_durations.shape[1])
        ).unsqueeze(0)

    # The bounded window's end spec is the boundary embedding, not the scalar MLP.
    torch.testing.assert_close(got[:, BOUNDED_WINDOW], (start_spec + end_event + rest)[:, BOUNDED_WINDOW])
    assert (
        got[:, BOUNDED_WINDOW] - (start_spec + end_duration + rest)[:, BOUNDED_WINDOW]
    ).abs().max().item() > LIVE, "the bounded window's end slot is replaced"
    # Every other window keeps the scalar path exactly.
    other = [k for k in range(batch.q_durations.shape[1]) if k != BOUNDED_WINDOW]
    torch.testing.assert_close(got[:, other], (start_spec + end_duration + rest)[:, other])


def test_bound_marker_receives_gradient():
    """The marker is what separates 'bounded by X' from 'asking about X'; it must train."""
    model = tiny_model()
    model.train()
    loss, _ = model(_bound(_unbounded_batch(), 9))
    loss.backward()
    assert model.bound_marker.grad is not None
    assert torch.isfinite(model.bound_marker.grad).all()
    assert model.bound_marker.grad.abs().sum() > 0
