"""The other half of the RoPE/strip guard: ``use_rope_time=False`` on a *stripped* batch.

``use_rope_time`` (model) and ``strip_delta_tokens`` (dataset) are two halves of one setting.
The model has always refused ``use_rope_time=True`` with no ``time_pos_ids``.  The reverse
mismatch used to be accepted in silence, and it is the worse of the two:

- ``strip_delta_tokens=True`` deletes the quantized ``TIMELINE//DELTA*`` tokens from
  ``batch.code`` and moves the time they encoded into ``time_pos_ids``.
- The backbone reads *only* the token embeddings, the attention mask, and ``position_ids``.
  ``time_delta_days`` survives on the batch but is never passed to it.
- So ``use_rope_time=False`` on such a batch throws the positions away and leaves a backbone with
  **zero** elapsed-time information.  It trains, validates and checkpoints with normal-looking
  numbers; nothing in a loss curve distinguishes it from a healthy run.

None of that is a property of any one architecture.  ``validate_rope_time_pair`` is shared, and
``MultitaskBoundaryPytorchDataset`` has its own ``strip_delta_tokens``, so the identical silent
failure is reachable on ``ConditionalMultitaskARModel`` — which is what this file locks.  The
shipped ``conditional_multitask_ar_config.yaml`` ties the two halves together
(``strip_delta_tokens: ${lightning_module.model.use_rope_time}``), so they cannot drift on their
own; the guard exists for the override that breaks that link — a bare
``datamodule.dataset_kwargs.strip_delta_tokens=true`` on the command line, or a hand-built
datamodule — which is exactly how the mismatch is reached in practice.

The tests below lock six things:

1. the new guard fires, and its message names the config key to change;
2. the original opposite guard still fires (regression lock);
3. neither correctly-configured pair is disturbed — and for the ``use_rope_time=False`` +
   no-times pair, the hidden states are *bit-identical* to the pre-guard implementation;
4. the semantics: the caught configuration is provably blind to time, measured at
   ``window_hidden_states``;
5. each row of a batch is positioned against *its own* timeline — the whole suite used to hand
   both rows identical ``time_pos_ids``, under which broadcasting patient 0's timeline over the
   entire batch is indistinguishable from correct;
6. both public entry points — ``forward`` (training and validation) and ``score_final_query``
   (test and prediction) — actually pass the positions to the backbone.  Every other measurement
   here goes through ``window_hidden_states``, so a caller that reached the backbone another way
   and discarded the positions would leave the seam provably correct and RoPE provably dead.

On (4)'s measurement level, following ``test_feature_liveness``: an untrained readout compresses
an ~8e-05 difference in the hidden states down to ~1e-07 downstream, i.e. into float32 rounding
noise.  Every assertion here therefore reads ``window_hidden_states`` — the tensor the tied
readout projects — and uses a margin (``LIVE``), never a bare ``torch.equal`` inequality, except
where *equality* is the claim, which is exact and needs no margin.

Where the pre-guard behaviour is needed, it is obtained by restoring the *old*
``validate_rope_time_pair`` under the real code path (``_pre_guard_validator``) rather than by
re-deriving the position arithmetic in the test.  That keeps "what it used to do" a statement
about the shipped implementation instead of about a copy of it.
"""

from contextlib import contextmanager

import pytest
import torch

from every_query.data.multitask_dataset import MultitaskBoundaryBatch
from every_query.model import conditional_multitask_ar_model as multitask

# Reused deliberately rather than redeclared: the tiny-model config and the liveness margin must
# not drift between the file that proves RoPE is live and the file that proves the
# misconfiguration is dead.
from tests.test_conditional_multitask_ar_model import make_batch, tiny_model
from tests.test_feature_liveness import LIVE
from tests.test_rope_time import _record_backbone_calls

N = 6  # patient-stream width of every batch below

# Two batches whose patient streams are token-for-token identical and whose *only* difference is
# how much time elapsed between those events.  This is what a stripped batch looks like: no
# TIMELINE//DELTA* tokens left in `code`, the gaps living in `time_pos_ids` (elapsed hours).
_HOURS_DENSE = [0, 1, 2, 3, 4, 5]  # six events within six hours
_HOURS_SPARSE = [0, 24, 300, 1200, 4000, 9000]  # the same six events spread over a year

# Two patients with identical code streams and no padding, so that when their timelines differ
# the *only* thing that can separate their representations is the elapsed time.
_SAME_CODES = [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]


@contextmanager
def _pre_guard_validator():
    """Run the real model with ``validate_rope_time_pair`` as it read before the symmetric guard.

    The multitask model calls the shared validator by name from its own namespace, so rebinding
    it here puts the *shipped* ``_position_ids`` / ``window_hidden_states`` back on their previous
    behaviour without the test re-deriving any of the position arithmetic.  The lenient version
    below is the whole of the old contract: refuse ``use_rope_time=True`` with no times, and
    silently discard times when ``use_rope_time`` is off.
    """
    original = multitask.validate_rope_time_pair

    def lenient(use_rope_time, time_pos):
        if not use_rope_time:
            return None
        if time_pos is None:
            raise ValueError("use_rope_time=True but the batch carries no time_pos_ids.")
        return time_pos

    multitask.validate_rope_time_pair = lenient
    try:
        yield
    finally:
        multitask.validate_rope_time_pair = original


def _stripped_batch(hours: list[int] | list[list[int]], code=None) -> MultitaskBoundaryBatch:
    """A ``strip_delta_tokens=True``-shaped batch with the given per-token elapsed hours.

    ``hours`` is either one timeline shared by both rows, or one timeline per row.  Per-row
    timelines matter: with both rows identical, ``time_pos[:1].expand_as(batch.code)`` — a
    plausible broadcast slip applying patient 0's timeline to every patient in the batch — is
    numerically indistinguishable from the correct implementation.

    ``time_delta_days`` is filled in consistently with ``hours`` rather than left at zero, so two
    batches built from different timelines differ in *every* field that carries time.  That makes
    the blindness result below stronger: not even the surviving ``time_delta_days`` rescues the
    backbone, because the backbone never reads it.
    """
    rows = hours if isinstance(hours[0], list) else [hours] * 2
    per_row = torch.tensor(rows, dtype=torch.long)
    assert per_row.shape[1] == N
    batch = make_batch(code=code or _SAME_CODES, time_pos_ids=per_row)
    batch.time_delta_days = torch.diff(per_row.float(), dim=1, prepend=torch.zeros(len(rows), 1)) / 24.0
    return batch


def _untimed_batch() -> MultitaskBoundaryBatch:
    """The same patient stream with no ``time_pos_ids`` — an unstripped dataset's batch."""
    return make_batch(code=_SAME_CODES)


def _one_row(batch: MultitaskBoundaryBatch, i: int) -> MultitaskBoundaryBatch:
    """Row ``i`` of ``batch`` on its own — the same patient, positioned without the other one."""
    row = slice(i, i + 1)
    return MultitaskBoundaryBatch(
        code=batch.code[row],
        numeric_value=batch.numeric_value[row],
        numeric_value_mask=batch.numeric_value_mask[row],
        time_delta_days=batch.time_delta_days[row],
        q_start_durations=batch.q_start_durations[row],
        q_start_codes=batch.q_start_codes[row],
        q_durations=batch.q_durations[row],
        q_bound_codes=batch.q_bound_codes[row],
        q_mask=batch.q_mask[row],
        targets=batch.targets[row],
        condition_codes=batch.condition_codes[row],
        condition_answers=batch.condition_answers[row],
        time_pos_ids=batch.time_pos_ids[row],
    )


def _hidden(model, batch) -> torch.Tensor:
    """The window hidden states the tied readout projects — where RoPE actually acts."""
    with torch.no_grad():
        return model.window_hidden_states(batch)


# ── 1. the new guard fires ─────────────────────────────────────────────────────────────


def test_non_rope_model_refuses_a_stripped_batch():
    """``use_rope_time=False`` + ``time_pos_ids`` is a misconfiguration, not a no-op."""
    model = tiny_model(use_rope_time=False)
    with pytest.raises(ValueError) as excinfo:
        model(_stripped_batch(_HOURS_SPARSE))

    message = str(excinfo.value)
    # Both halves of the pair are named, because either one fixes it and only the user knows
    # which they meant.  A message that just said "mismatch" would leave them guessing.
    assert "datamodule.dataset_kwargs.strip_delta_tokens" in message, message
    assert "use_rope_time" in message, message
    assert "time_pos_ids" in message, message
    # It must say *why*, not merely that the two flags disagree.
    assert "elapsed" in message.lower(), message


def test_the_guard_fires_before_the_backbone_runs():
    """The refusal comes from the position seam itself, and no tensor reaches the backbone.

    ``window_hidden_states`` is where the two halves of the setting are both in scope, and it is
    the single seam every entry point goes through, so the guard cannot be bypassed by a second
    one being added later.  Asserted by recording backbone calls rather than by inspecting the
    traceback: a check that fired *after* the backbone ran would still raise, and would still let
    a misconfigured run burn an epoch before dying.
    """
    for entry in (
        lambda m, b: m(b),
        lambda m, b: m.score_final_query(b, torch.tensor([2, 3])),
    ):
        model = tiny_model(use_rope_time=False)
        calls = _record_backbone_calls(model)
        with pytest.raises(ValueError, match="strip_delta_tokens"):
            entry(model, _stripped_batch(_HOURS_DENSE))
        assert calls == [], "the backbone ran before the guard fired"


# ── 2. regression lock on the original guard ───────────────────────────────────────────


def test_rope_model_still_refuses_a_batch_without_times():
    """The pre-existing opposite guard must survive the symmetric one being added."""
    model = tiny_model(use_rope_time=True)
    with pytest.raises(ValueError) as excinfo:
        model(_untimed_batch())

    message = str(excinfo.value)
    assert "time_pos_ids" in message, message
    assert "datamodule.dataset_kwargs.strip_delta_tokens" in message, message


# ── 3. neither correct configuration is disturbed ──────────────────────────────────────


def test_rope_model_with_times_still_runs_and_still_uses_them():
    """(a) ``use_rope_time=True`` + ``time_pos_ids``: runs, and the positions are handed over."""
    model = tiny_model(use_rope_time=True)
    batch = _stripped_batch(_HOURS_SPARSE)
    calls = _record_backbone_calls(model)

    with torch.no_grad():
        loss, out = model(batch)

    assert loss.isfinite() and out.logits.isfinite().all()
    (kwargs, _) = calls[0]
    position_ids = kwargs["position_ids"]
    assert position_ids is not None
    # The patient prefix is the batch's own elapsed hours; the query tokens are pinned to the
    # last real patient event's hour, because a query about the future does not advance the clock.
    assert torch.equal(position_ids[:, :N], batch.time_pos_ids)
    assert (position_ids[:, N:] == batch.time_pos_ids[:, -1:]).all()


def test_non_rope_model_without_times_is_bit_identical_to_the_pre_guard_model():
    """(b) ``use_rope_time=False`` + no ``time_pos_ids``: not one bit moves for correct users.

    Compared against the shipped implementation running under the *previous*
    ``validate_rope_time_pair`` — rather than against a hardcoded number or a re-derivation — so
    this stays a statement about the guard rather than about this machine's float arithmetic.
    """
    model = tiny_model(use_rope_time=False)
    batch = _untimed_batch()
    assert batch.time_pos_ids is None

    guarded = _hidden(model, batch)
    with _pre_guard_validator():
        reference = _hidden(model, batch)
    assert torch.equal(reference, guarded), "the guard perturbed a correctly-configured backbone"

    calls = _record_backbone_calls(model)
    with torch.no_grad():
        loss, out = model(batch)
    assert calls[0][0]["position_ids"] is None, "the guard started inventing positions of its own"
    assert loss.isfinite() and out.logits.isfinite().all()


# ── 4. why the caught configuration is bad ─────────────────────────────────────────────


def test_the_caught_configuration_is_provably_blind_to_elapsed_time():
    """Two runs differing only in elapsed time are represented *identically* without positions.

    This is the whole justification for making the mismatch an error rather than a warning.  The
    two batches hold the same six codes in the same order; one spans six hours, the other a year.
    Their delta tokens have been stripped, so the token stream cannot express that.

    The blind path is not simulated by hand-assembling a backbone call: ``_pre_guard_validator``
    puts the shipped ``window_hidden_states`` back on its pre-guard behaviour, so what is measured
    is literally what a ``use_rope_time=False`` model used to do with these batches.  Read at
    ``window_hidden_states``: an untrained readout would squash this difference toward float32
    noise, and that measurement error has already produced one false negative on this branch.
    """
    off = tiny_model(use_rope_time=False)
    on = tiny_model(use_rope_time=True)
    # ``tiny_model`` seeds, so the two instances hold the same weights and the only independent
    # variable below is whether the positions are passed.  Checked, not assumed.
    for (name, a), (_, b) in zip(off.state_dict().items(), on.state_dict().items(), strict=True):
        assert torch.equal(a, b), f"the two models differ at {name}; the comparison is confounded"

    dense, sparse = _stripped_batch(_HOURS_DENSE), _stripped_batch(_HOURS_SPARSE)
    assert torch.equal(dense.code, sparse.code), "the codes must be the only thing held fixed"
    assert (dense.time_pos_ids != sparse.time_pos_ids).any()
    assert not torch.equal(dense.time_delta_days, sparse.time_delta_days)

    with _pre_guard_validator():
        blind_dense, blind_sparse = _hidden(off, dense), _hidden(off, sparse)
    assert torch.equal(blind_dense, blind_sparse), (
        "the point of the guard: without time positions the backbone cannot distinguish six "
        "hours from a year, and every checkpoint trained that way looks healthy"
    )

    # The correctly-configured pair does see it, at a margin far above rounding noise.
    assert (_hidden(on, dense) - _hidden(on, sparse)).abs().max().item() > LIVE

    # And the blind path is genuinely the one a `use_rope_time=False` model takes: same weights,
    # same absent positions, same numbers — only reachable now via a batch that carries no times,
    # which is exactly what the guard forces.
    untimed = _untimed_batch()
    assert torch.equal(untimed.code, dense.code)
    assert torch.equal(_hidden(off, untimed), blind_dense)


# ── 5. every row is positioned against its own timeline ────────────────────────────────


def test_each_row_is_positioned_against_its_own_timeline():
    """Patient 1's representation must follow patient 1's clock, not patient 0's.

    Until this test, every RoPE fixture in the repo gave both rows of the batch *identical*
    ``time_pos_ids``, so ``time_pos[:1].expand(...)`` — patient 0's timeline broadcast over the
    whole batch — passed the entire RoPE suite.  The multitask model is if anything easier to
    break this way: ``_position_ids`` gathers a per-row ``last_hour`` and scatters it across that
    row's query tokens, so a slip has two places to collapse rather than one.  In training it is a
    silently wrong model: every patient but the first is scored against someone else's elapsed
    time, and nothing in a loss curve shows it.

    Two checks, because either alone is weak.  The rows must differ from *each other* (a broadcast
    collapses them to bit-identical), and each row must match what that patient's timeline produces
    when positioned alone (which pins *whose* clock each row got, not merely that they differ).
    """
    model = tiny_model(use_rope_time=True)
    batch = _stripped_batch([_HOURS_DENSE, _HOURS_SPARSE])
    assert torch.equal(batch.code[0], batch.code[1]), "codes held fixed; only the clocks differ"
    assert (batch.time_pos_ids[0] != batch.time_pos_ids[1]).any()

    both = _hidden(model, batch)
    assert (both[0] - both[1]).abs().max().item() > LIVE, (
        "same codes, different timelines: the two rows must not come out alike — a broadcast of "
        "row 0's positions over the batch makes them identical"
    )

    for i in range(2):
        solo = _hidden(model, _one_row(batch, i))
        torch.testing.assert_close(
            both[i],
            solo[0],
            atol=1e-5,
            rtol=1e-5,
            msg=f"row {i} must be positioned against its own hours, whoever it shares a batch with",
        )


# ── 6. the call sites: every entry point hands the positions over ──────────────────────


@pytest.mark.parametrize(
    ("name", "entry"),
    [
        ("forward", lambda m, b: m(b)),
        ("score_final_query", lambda m, b: m.score_final_query(b, torch.tensor([2, 3]))),
    ],
)
def test_every_entry_point_passes_the_positions_to_the_backbone(name, entry):
    """The guarded seam is useless if a caller throws its result away.

    Everything above reaches the backbone through ``window_hidden_states``, so all of it stays
    green if a public entry point computes the positions — guards firing and all — and then hands
    the backbone none.  That is the headline defect wearing a different hat: a RoPE run that
    trains, or scores an entire evaluation grid, on token-index positions.

    Both entry points are locked because ``score_final_query`` does **not** go through
    ``forward``: it is the path ``test_step`` and ``predict_step`` take, and it is the same bypass
    that makes ``window_hidden_states`` clear the ontology mixed-table cache itself rather than
    relying on ``forward``'s pre-hook.  A guard verified only on the training path would leave
    every reported score unprotected.
    """
    model = tiny_model(use_rope_time=True)
    calls = _record_backbone_calls(model)
    dense, sparse = _stripped_batch(_HOURS_DENSE), _stripped_batch(_HOURS_SPARSE)

    with torch.no_grad():
        entry(model, dense)
        entry(model, sparse)

    assert len(calls) == 2, f"{name} must call the backbone exactly once per batch"
    for (kwargs, _), batch in zip(calls, [dense, sparse], strict=True):
        assert kwargs.get("position_ids") is not None, f"{name} dropped the rotary positions"
        assert torch.equal(kwargs["position_ids"][:, :N], batch.time_pos_ids)

    (_, hidden_dense), (_, hidden_sparse) = calls
    assert (hidden_dense - hidden_sparse).abs().max().item() > LIVE, (
        "the representation the readout projects must move when only the elapsed times move"
    )
