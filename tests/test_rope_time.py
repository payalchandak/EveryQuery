"""Tests for the RoPE time representation and the delta-token strip.

Covers, in pipeline order:

1. :mod:`every_query.data.rope_time` — strip semantics, elapsed-time preservation, row
   isolation, and agreement between the keep mask and the strip.
2. :class:`~every_query.data.query_seq_dataset.QuerySeqPytorchDataset` — that
   ``strip_delta_tokens=True`` removes the delta tokens from the collated encoder input,
   emits aligned ``time_pos_ids``, and leaves the query tensors untouched.
3. :class:`~every_query.model.conditional_multitask_ar_model.ConditionalMultitaskARModel` — that
   ``use_rope_time`` actually reaches the backbone's rotary machinery *through the real
   ``forward`` path* (the window hidden states move when only the times move), that the query
   tokens are pinned to the last patient event's clinical hour rather than advancing time, and
   that **both** half-configurations are refused rather than silently falling back to
   token-index positions: a RoPE model handed a batch with no times, and a non-RoPE model handed
   a batch that carries them.  The second direction was written down here as a *desirable*
   property once ("a non-RoPE model answers a timed batch identically") — that configuration
   leaves the backbone with zero elapsed-time information, so the correct behaviour is a
   refusal.  ``use_rope_time=False`` is inert only on a batch that carries no times, which is
   the half of that claim kept below.

Section 3 was originally written against the encoder-decoder conditional query-sequence model
(since deleted with the rest of that pipeline), because it was the only model that consumed
``time_pos_ids`` when the feature landed.  ``ConditionalMultitaskARModel`` is the surviving
consumer; the claims are
the same, the index arithmetic is its own (one causal stream of ``S`` patient tokens followed by
``3K-2`` query tokens, not an encoder feeding a cross-attending decoder).

Measurement level, throughout section 3: assertions about whether times *reached* the backbone
read ``window_hidden_states`` and use the ``LIVE`` margin.  That is the tensor the tied readout
reads, so it is where the effect lives and it does not depend on the readout's initialisation —
the same reasoning that made the encoder-decoder version measure ``last_hidden_state`` rather
than ``answer_logits``, where a randomly-initialised head squashed the difference to ~1e-07,
i.e. into float32 rounding noise, and a bare ``torch.equal`` inequality was satisfied by one ULP
and passed just as happily when RoPE was dead.
"""

import pytest
import torch
from meds import train_split
from meds_torchdata import MEDSTorchDataConfig

from every_query.data.query_seq_dataset import QuerySeqBatch, QuerySeqPytorchDataset
from every_query.data.rope_time import (
    DELTA_TOKEN_PREFIX,
    build_keep_mask,
    compact_by_keep,
    delta_vocab_ids,
    strip_delta_tokens,
)
from every_query.model.answers import validate_rope_time_pair
from every_query.model.conditional_multitask_ar_model import TOKENS_PER_WINDOW

# The multitask model's own construction idiom, reused rather than re-invented.
from tests.test_conditional_multitask_ar_model import make_batch, tiny_model

# Imported rather than redeclared so the margin that separates "RoPE is live" from float32
# rounding cannot drift between the three files that measure it.
from tests.test_feature_liveness import LIVE

DELTA_ID = 90

# ``make_batch``'s patient stream: row 0 has four real tokens, row 1 has two and two PADs.
NEAR_TIMES = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
FAR_TIMES = torch.tensor([[0, 240, 1000, 5000], [0, 5, 9, 30]])


# ── 1. strip semantics ──────────────────────────────────────────────────


def test_delta_vocab_ids_selects_by_prefix():
    vocab = {"A": 1, f"{DELTA_TOKEN_PREFIX}//1h": 5, f"{DELTA_TOKEN_PREFIX}//1d": 2, "B": 3}
    assert delta_vocab_ids(vocab).tolist() == [2, 5]


def test_strip_removes_deltas_and_preserves_elapsed_time():
    """The delta tokens vanish, but the time they encoded survives in ``time_pos_ids``."""
    code = torch.tensor([[5, DELTA_ID, 6, DELTA_ID, 7, 0]])
    tdd = torch.tensor([[0.0, 1.0, 0.0, 0.5, 0.0, 0.0]])
    out_code, _, _, _, pos = strip_delta_tokens(
        code,
        torch.zeros(1, 6),
        torch.zeros(1, 6, dtype=torch.bool),
        tdd,
        torch.tensor([DELTA_ID]),
    )
    assert out_code.tolist() == [[5, 6, 7]], "delta tokens and padding must be gone"
    # 0d, then +1d = 24h, then +0.5d = 36h.
    assert pos.tolist() == [[0, 24, 36]]


def test_strip_is_row_isolated():
    """One row's delta pattern must not leak into another row's positions."""
    code = torch.tensor([[5, DELTA_ID, 6, 0], [5, 6, 0, 0]])
    tdd = torch.tensor([[0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    _, _, _, _, pos = strip_delta_tokens(
        code,
        torch.zeros(2, 4),
        torch.zeros(2, 4, dtype=torch.bool),
        tdd,
        torch.tensor([DELTA_ID]),
    )
    assert pos[0].tolist() == [0, 48]
    assert pos[1].tolist() == [0, 0], "row without deltas stays at time zero"


def test_strip_rebases_each_row_to_zero():
    """Absolute offset is arbitrary under RoPE; equal relative spacing must give equal ids."""
    code = torch.tensor([[DELTA_ID, 5, DELTA_ID, 6], [DELTA_ID, 5, DELTA_ID, 6]])
    tdd = torch.tensor([[1.0, 0.0, 1.0, 0.0], [40.0, 0.0, 1.0, 0.0]])
    _, _, _, _, pos = strip_delta_tokens(
        code,
        torch.zeros(2, 4),
        torch.zeros(2, 4, dtype=torch.bool),
        tdd,
        torch.tensor([DELTA_ID]),
    )
    assert pos[0].tolist() == pos[1].tolist() == [0, 24]


def test_strip_recomputes_time_delta_days_from_survivors():
    """After the strip, ``time_delta_days`` must still mean "days since the previous token".

    The gaps live on the delta tokens being dropped, so a naive compaction would leave this
    field all zeros — a silent trap sitting next to ``time_pos_ids``, which holds the truth.
    """
    code = torch.tensor([[5, DELTA_ID, 6, DELTA_ID, 7, 0]])
    tdd = torch.tensor([[0.0, 1.0, 0.0, 0.5, 0.0, 0.0]])
    _, _, _, out_tdd, pos = strip_delta_tokens(
        code,
        torch.zeros(1, 6),
        torch.zeros(1, 6, dtype=torch.bool),
        tdd,
        torch.tensor([DELTA_ID]),
    )
    assert out_tdd[0].tolist() == [0.0, 1.0, 0.5]
    assert out_tdd.abs().sum() > 0, "gaps must survive the strip somewhere other than time_pos_ids"
    # The two representations must agree: cumulative days == positions in hours.
    assert torch.allclose(out_tdd.cumsum(1) * 24.0, pos.float())


def test_strip_zeroes_time_delta_at_padding():
    """Padded tails carry no gap — in particular no negative gap at the pad boundary."""
    code = torch.tensor([[5, DELTA_ID, 6, 0], [5, 0, 0, 0]])
    tdd = torch.tensor([[0.0, 3.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    _, _, _, out_tdd, _ = strip_delta_tokens(
        code,
        torch.zeros(2, 4),
        torch.zeros(2, 4, dtype=torch.bool),
        tdd,
        torch.tensor([DELTA_ID]),
    )
    assert out_tdd[0].tolist() == [0.0, 3.0]
    assert out_tdd[1].tolist() == [0.0, 0.0], "shorter row's padded tail must stay zero"
    assert (out_tdd >= 0).all(), "no negative gap may appear at the padding boundary"


def test_keep_mask_agrees_with_strip():
    code = torch.tensor([[5, DELTA_ID, 6, 0]])
    keep = build_keep_mask(code, torch.tensor([DELTA_ID]))
    assert keep.tolist() == [[True, False, True, False]]
    out_code, _, _, _, _ = strip_delta_tokens(
        code,
        torch.zeros(1, 4),
        torch.zeros(1, 4, dtype=torch.bool),
        torch.zeros(1, 4),
        torch.tensor([DELTA_ID]),
    )
    assert compact_by_keep(code, keep, out_code.shape[1]).tolist() == out_code.tolist()


def test_strip_handles_nan_time_deltas():
    """meds_torchdata leaves NaN at unknown deltas; those must not poison the cumsum."""
    code = torch.tensor([[5, DELTA_ID, 6]])
    tdd = torch.tensor([[float("nan"), 1.0, 0.0]])
    _, _, _, _, pos = strip_delta_tokens(
        code,
        torch.zeros(1, 3),
        torch.zeros(1, 3, dtype=torch.bool),
        tdd,
        torch.tensor([DELTA_ID]),
    )
    assert pos.isfinite().all() and pos.tolist() == [[0, 24]]


# ── 2. dataset integration ──────────────────────────────────────────────


def _seq_cfg(tensorized_cohort_dir, seq_task_labels_dir) -> MEDSTorchDataConfig:
    return MEDSTorchDataConfig(
        tensorized_cohort_dir=str(tensorized_cohort_dir),
        task_labels_dir=str(seq_task_labels_dir),
        max_seq_len=64,
        seq_sampling_strategy="to_end",
        static_inclusion_mode="omit",
        batch_mode="SM",
    )


def test_dataset_without_strip_emits_no_time_pos_ids(seq_sample_batch):
    """Default behaviour is unchanged: no rotary times, delta tokens left in the stream."""
    assert seq_sample_batch.time_pos_ids is None


def test_dataset_strip_emits_aligned_time_pos_ids(tensorized_cohort_dir, seq_task_labels_dir):
    """With stripping on, ``time_pos_ids`` aligns to ``code`` and no delta token survives."""
    ds = QuerySeqPytorchDataset(
        _seq_cfg(tensorized_cohort_dir, seq_task_labels_dir),
        split=train_split,
        strip_delta_tokens=True,
    )
    batch = ds.collate([ds[i] for i in range(len(ds))])

    assert batch.time_pos_ids is not None
    assert batch.time_pos_ids.shape == batch.code.shape
    assert not torch.isin(batch.code, ds.delta_ids).any(), "no delta token may survive the strip"
    # Elapsed time never runs backwards within a row.
    real = batch.code != QuerySeqBatch.PAD_INDEX
    for i in range(batch.code.shape[0]):
        row = batch.time_pos_ids[i][real[i]]
        assert (row.diff() >= 0).all() if row.numel() > 1 else True


def test_dataset_strip_compacts_the_real_collated_stream(tensorized_cohort_dir, seq_task_labels_dir):
    """Exercise the collate-time strip on real batches.

    The fixture cohort carries no ``TIMELINE//DELTA*`` codes, which would make a "nothing
    survives the strip" assertion vacuous.  So we nominate a code that *is* present as the
    delta id and check the stream is genuinely compacted, every other per-token field stays
    aligned, and the surviving tokens keep their order.
    """
    cfg = _seq_cfg(tensorized_cohort_dir, seq_task_labels_dir)
    plain = QuerySeqPytorchDataset(cfg, split=train_split)
    before = plain.collate([plain[i] for i in range(len(plain))])

    real = before.code[before.code != QuerySeqBatch.PAD_INDEX]
    victim = int(real.mode().values.item())  # the most common real token

    ds = QuerySeqPytorchDataset(cfg, split=train_split, strip_delta_tokens=True)
    ds.delta_ids = torch.tensor([victim])
    after = ds.collate([ds[i] for i in range(len(ds))])

    assert (after.code == victim).sum() == 0, "the nominated delta token must be gone"
    assert after.code.shape[1] < before.code.shape[1], "the stream must actually get shorter"
    assert after.time_pos_ids.shape == after.code.shape
    assert after.numeric_value.shape == after.code.shape
    assert after.numeric_value_mask.shape == after.code.shape

    for i in range(before.code.shape[0]):
        kept = [int(c) for c in before.code[i] if int(c) not in (victim, QuerySeqBatch.PAD_INDEX)]
        got = [int(c) for c in after.code[i] if int(c) != QuerySeqBatch.PAD_INDEX]
        assert got == kept, "surviving tokens must keep their original order"


def test_strip_emits_times_even_when_the_cohort_has_no_delta_tokens(
    tensorized_cohort_dir, seq_task_labels_dir
):
    """``time_pos_ids`` means "the strip was requested", not "delta tokens were deleted".

    ``QuerySeqPytorchDataset`` handles the empty-``delta_ids`` cohort explicitly — it
    warns and carries on, emitting ``time_pos_ids`` while deleting nothing — so the presence of
    the field is not by itself proof that ``batch.code`` was rewritten.  Pinned here because
    ``_encoder_position_kwargs``'s docstring reasons about what that presence implies, and
    because the obvious "tidy-up" (suppressing ``time_pos_ids`` when nothing was stripped)
    would silently turn a caught misconfiguration back into a silent one: a user who asked for
    the strip against a cohort whose delta tokens are missing or differently named still needs
    the mismatch reported, not smoothed over.
    """
    cfg = _seq_cfg(tensorized_cohort_dir, seq_task_labels_dir)
    plain = QuerySeqPytorchDataset(cfg, split=train_split)
    ds = QuerySeqPytorchDataset(cfg, split=train_split, strip_delta_tokens=True)
    assert ds.delta_ids.numel() == 0, "this fixture cohort must have no TIMELINE//DELTA* codes"

    before = plain.collate([plain[i] for i in range(len(plain))])
    after = ds.collate([ds[i] for i in range(len(ds))])

    assert after.time_pos_ids is not None, "the positions are emitted even with nothing to strip"
    assert after.time_pos_ids.shape == after.code.shape
    pad = QuerySeqBatch.PAD_INDEX
    for i in range(before.code.shape[0]):
        kept = [int(c) for c in before.code[i] if int(c) != pad]
        got = [int(c) for c in after.code[i] if int(c) != pad]
        assert got == kept, "no delta ids means no token may be removed from the stream"

    # The guard is still right in this case — the user asked for the strip, and a
    # use_rope_time=False model would drop the hours it produced on the floor.  Asserted on the
    # shared validator both conditional architectures call, so it holds whichever model consumes
    # this batch.
    with pytest.raises(ValueError, match="strip_delta_tokens"):
        validate_rope_time_pair(False, after.time_pos_ids)


def test_dataset_strip_never_touches_the_static_table(tensorized_cohort_dir, seq_task_labels_dir):
    """``static_code`` & friends are a separate table, not per-token fields of the dynamic stream.

    The old width heuristic compacted *any* ``(B, n_old)`` tensor with the dynamic keep mask, so
    with ``static_inclusion_mode=include`` the static table was corrupted whenever it happened to
    be exactly as wide as the padded dynamic stream.  Pick ``max_seq_len`` so that it is.
    """

    def cfg(max_seq_len: int) -> MEDSTorchDataConfig:
        return MEDSTorchDataConfig(
            tensorized_cohort_dir=str(tensorized_cohort_dir),
            task_labels_dir=str(seq_task_labels_dir),
            max_seq_len=max_seq_len,
            seq_sampling_strategy="to_end",
            static_inclusion_mode="include",
            batch_mode="SM",
        )

    plain = before = None
    for max_seq_len in range(1, 9):
        plain = QuerySeqPytorchDataset(cfg(max_seq_len), split=train_split)
        before = plain.collate([plain[i] for i in range(len(plain))])
        if before.code.shape[1] == before.static_code.shape[1]:
            break
    assert before.code.shape[1] == before.static_code.shape[1], "fixture never lines the widths up"

    real = before.code[before.code != QuerySeqBatch.PAD_INDEX]
    victim = int(real.mode().values.item())
    ds = QuerySeqPytorchDataset(cfg(plain.config.max_seq_len), split=train_split, strip_delta_tokens=True)
    ds.delta_ids = torch.tensor([victim])
    after = ds.collate([ds[i] for i in range(len(ds))])

    assert after.code.shape[1] < before.code.shape[1], "the dynamic stream must actually get shorter"
    assert torch.equal(after.static_code, before.static_code)
    assert torch.equal(after.static_numeric_value, before.static_numeric_value)
    assert torch.equal(after.static_numeric_value_mask, before.static_numeric_value_mask)


def test_dataset_strip_leaves_query_tensors_untouched(tensorized_cohort_dir, seq_task_labels_dir):
    """Stripping touches the encoder stream only; the decoder's query blocks are unaffected."""
    cfg = _seq_cfg(tensorized_cohort_dir, seq_task_labels_dir)
    plain = QuerySeqPytorchDataset(cfg, split=train_split)
    stripped = QuerySeqPytorchDataset(cfg, split=train_split, strip_delta_tokens=True)

    a = plain.collate([plain[i] for i in range(len(plain))])
    b = stripped.collate([stripped[i] for i in range(len(stripped))])

    assert torch.equal(a.q_codes, b.q_codes)
    assert torch.equal(a.q_durations, b.q_durations)
    assert torch.equal(a.q_answers, b.q_answers)
    assert torch.equal(a.q_mask, b.q_mask)


# ── 3. model wiring ─────────────────────────────────────────────────────


def _hidden(model, batch) -> torch.Tensor:
    """The window hidden states the tied readout projects — where RoPE actually acts."""
    with torch.no_grad():
        return model.window_hidden_states(batch)


def _record_backbone_calls(model) -> list[tuple[dict, torch.Tensor]]:
    """Record ``(kwargs, last_hidden_state)`` for every backbone call ``forward`` itself makes.

    Every other measurement in this file reaches the backbone through ``window_hidden_states``,
    which proves the *seam* works and says nothing about whether the training ``forward`` uses
    it.  A ``forward`` that computed the positions and then dropped them — the guards firing,
    the seam correct, RoPE stone dead in the only path training and evaluation take — would be
    invisible to all of them.  Hooking the backbone is what closes that.
    """
    calls: list[tuple[dict, torch.Tensor]] = []

    def hook(module, args, kwargs, output):
        calls.append((dict(kwargs), output.last_hidden_state.detach().clone()))

    model.HF_model.register_forward_hook(hook, with_kwargs=True)
    return calls


def test_rope_time_reaches_rotary():
    """Same tokens, different elapsed times must give a different representation.

    Asserted on the window hidden states rather than on the logits: that is the tensor the tied
    readout reads, so the claim does not depend on the readout's initialisation.  The margin is
    ``LIVE`` rather than a bare inequality because a bitwise difference is satisfied by one ULP
    of float32 rounding.
    """
    model = tiny_model(use_rope_time=True)
    near = _hidden(model, make_batch(time_pos_ids=NEAR_TIMES))
    far = _hidden(model, make_batch(time_pos_ids=FAR_TIMES))
    assert (near - far).abs().max().item() > LIVE, (
        "time_pos_ids must change the backbone geometry when use_rope_time=True"
    )


def test_rope_time_is_the_only_thing_that_moved():
    """Holding times fixed reproduces the hidden states exactly — the change is time, not noise."""
    model = tiny_model(use_rope_time=True)
    once = _hidden(model, make_batch(time_pos_ids=NEAR_TIMES))
    twice = _hidden(model, make_batch(time_pos_ids=NEAR_TIMES))
    assert torch.equal(once, twice)


def test_forward_hands_the_times_to_the_backbone():
    """The real ``forward`` path — not a hand-assembled call — must use the positions.

    Replaces an assertion that read ``assert not torch.equal(near.answer_logits,
    far.answer_logits)`` after two ``model(batch)`` calls.  That was the repo's only defence
    against ``forward`` ignoring the position kwargs, and it was a one-ULP defence: an untrained
    head compresses this difference to ~1e-07, so *any* two non-identical float paths satisfy
    it.  Here the batches go through ``model(batch)`` exactly as training does, and the claim is
    measured where the effect lives.

    The positions themselves are checked semantically rather than against
    ``model._position_ids``, which would only compare the method with itself: the patient prefix
    carries the batch's own elapsed hours, and every query token sits at the last *real* patient
    event's hour, because a query about the future must not advance clinical time.
    """
    model = tiny_model(use_rope_time=True)
    calls = _record_backbone_calls(model)
    near_batch = make_batch(time_pos_ids=NEAR_TIMES)
    far_batch = make_batch(time_pos_ids=FAR_TIMES)
    n_query_tokens = TOKENS_PER_WINDOW * near_batch.q_durations.shape[1] - 2

    with torch.no_grad():
        model(near_batch)
        model(far_batch)

    assert len(calls) == 2, "forward must call the backbone exactly once per batch"
    for (kwargs, _), batch in zip(calls, [near_batch, far_batch], strict=True):
        position_ids = kwargs.get("position_ids")
        assert position_ids is not None, (
            "forward computed the rotary positions and did not pass them to the backbone"
        )
        n_patient = (batch.code != batch.PAD_INDEX).sum(dim=1)
        for row, n in enumerate(n_patient.tolist()):
            assert position_ids[row, :n].tolist() == batch.time_pos_ids[row, :n].tolist(), (
                "the patient prefix must carry the batch's own elapsed hours"
            )
            last_hour = int(batch.time_pos_ids[row, n - 1])
            assert position_ids[row, n : n + n_query_tokens].tolist() == [last_hour] * n_query_tokens, (
                "a query token must sit at the last patient event's hour, not advance past it"
            )

    (_, near), (_, far) = calls
    assert (near - far).abs().max().item() > LIVE, (
        "the backbone forward actually ran must move when only the elapsed times move"
    )


def test_non_rope_model_refuses_a_batch_carrying_times():
    """The mirror of ``test_rope_model_refuses_a_batch_without_times``, and just as necessary.

    This test used to assert the opposite — that such a batch is "answered identically" by a
    non-RoPE model — which wrote the defect down as intended behaviour.  ``time_pos_ids`` is
    emitted only by the strip path, so answering that batch normally means answering with a
    backbone that has *no* elapsed-time signal at all: the delta tokens gone from ``code`` and
    the hours that replaced them discarded, while training, validating and checkpointing with
    entirely normal-looking numbers.  See ``test_rope_strip_guard.py`` for the measurement
    proving that blindness.
    """
    model = tiny_model(use_rope_time=False)
    with pytest.raises(ValueError, match="time_pos_ids"):
        model(make_batch(time_pos_ids=FAR_TIMES))


def test_non_rope_model_without_times_is_unperturbed():
    """The half of the old claim that survives: no times, no RoPE, no change and no refusal.

    The guard must be narrow.  A refusal that fired on every ``use_rope_time=False`` batch, or a
    fallback that started handing the backbone positions of its own, would both be caught here:
    the backbone must receive ``position_ids=None``, exactly as it did before the feature
    existed, and the model must still answer.
    """
    model = tiny_model(use_rope_time=False)
    batch = make_batch()
    assert batch.time_pos_ids is None

    calls = _record_backbone_calls(model)
    with torch.no_grad():
        loss, out = model(batch)

    assert len(calls) == 1
    assert calls[0][0].get("position_ids", "absent") is None, (
        "a non-RoPE model must reach the backbone exactly as it did before the feature existed"
    )
    assert loss.isfinite() and out.logits.isfinite().all()


def test_rope_model_refuses_a_batch_without_times():
    """A RoPE model handed an ordinary batch must fail loudly, not silently use token indices.

    Falling back is indistinguishable from working: the upstream experiment scored an entire
    eval grid against a model that never received its time positions before noticing.
    """
    model = tiny_model(use_rope_time=True)
    with pytest.raises(ValueError, match="time_pos_ids"):
        model(make_batch())


def test_rope_positions_beyond_max_position_embeddings_are_finite():
    """Hour-scale positions exceed max_position_embeddings; rotary computes them on the fly.

    The *count* of positions is still budgeted (``max_seq_len + 3 * max_windows``); it is only
    their magnitude that is unbounded, which is the distinction this pins.
    """
    model = tiny_model(use_rope_time=True)
    batch = make_batch(time_pos_ids=torch.tensor([[0, 20_000, 60_000, 90_000], [0, 40_000, 0, 0]]))
    assert int(batch.time_pos_ids.max()) > model.max_seq_len
    with torch.no_grad():
        loss, out = model(batch)
    assert loss.isfinite() and out.logits.isfinite().all()


def test_use_rope_time_is_recorded_in_hparams():
    """Checkpoints must round-trip the flag, or a reloaded model silently changes semantics."""
    assert tiny_model(use_rope_time=True).hparams["use_rope_time"] is True
    assert tiny_model().hparams["use_rope_time"] is False
