"""Are the new features LIVE in the model, or merely wired to it?

A feature can be plumbed through collate, reach the forward pass, and then be multiplied by
zero -- passing every shape, dtype and "runs without error" assertion in the suite while
contributing nothing to the output.  Nothing in ``test_event_bounded.py`` or
``test_feature_composition.py`` can detect that, because they assert the model *runs* with the
new tensors, not that it *responds* to them.  The same gap exists in
``test_conditional_multitask_ar_model.py``: it pins the window token's *structure*
(``test_window_specs_use_matching_rows_and_distinct_roles`` rebuilds the expected embedding out
of the model's own parameters, which an all-zero marker satisfies exactly) and the RoPE
*guards*, but nothing there measures magnitude.  That is what this file is for.

Three probes here that a dead feature cannot pass:

1. **Gradient** -- each new parameter receives a non-zero gradient from a batch exercising it.
2. **Sensitivity** -- perturbing one new input field alone moves the output.
3. **Atom invariance** -- an unbounded, prediction-time-started batch is bit-identical with and
   without the new tensors attached.  The evaluation grid is entirely atomic single-code,
   time-bounded queries, so this is the property every reported AUROC number rests on: if
   attaching the machinery perturbed those queries, the scores would describe a different model
   than the one that was trained.

A note on measurement level, learned the hard way: on the encoder-decoder model these probes
were originally written against, a randomly-initialised decoder and output head compressed an
8e-05 encoder-output difference down to ~1e-07 at the logits, which is the same magnitude as
float32 rounding noise.  ``ConditionalMultitaskARModel`` reads out through a tied linear
projection rather than a decoder tower, so the attenuation is milder -- but the discipline is
kept: RoPE is asserted on ``window_hidden_states`` (the tensor the readout actually reads), and
no assertion anywhere here uses bare ``torch.equal`` inequality to mean "responds to".  A
bitwise inequality is satisfied by one ULP of rounding, which is how a test can be green from
the moment it is written while the feature it names does nothing.
"""

import pytest
import torch

# The multitask model's own construction idiom, reused rather than re-invented so a change to
# the batch contract shows up here too.
from tests.test_conditional_multitask_ar_model import make_batch, tiny_model

# Comfortably above float32 rounding on these tensors (~1e-7) and below any real effect (~1e-5).
LIVE = 1e-6


def _atomic(**over):
    """``make_batch`` reduced to atomic windows: every window timed, opened at prediction time.

    This is the shape of the whole evaluation grid, and the batch the "attaching the machinery changes
    nothing" claim below is about.
    """
    batch = make_batch(**over)
    rows, windows = batch.q_durations.shape
    batch.q_durations = torch.tensor([[7.0, 30.0, 4.0, 2.0, 1.0][:windows]] * rows)
    batch.q_bound_codes = torch.zeros_like(batch.q_bound_codes)
    if batch.q_start_codes is not None:
        batch.q_start_durations = torch.zeros_like(batch.q_start_durations)
        batch.q_start_codes = torch.zeros_like(batch.q_start_codes)
    return batch


# ── 1. gradient ────────────────────────────────────────────────────────────────────────


def test_role_markers_receive_gradient():
    """The parameters these features add to the model must not be inert.

    ``make_batch``'s default windows exercise both: window 1 ends at boundary code 10, window 2
    opens at start event 9.  ``torch.where`` computes both branches of the start/end spec, so a
    marker that never reached the *selected* branch would still leave the forward finite.
    """
    model = tiny_model()
    model.train()
    _, out = model(make_batch())
    out.logits.sum().backward()

    for name in ("bound_marker", "start_marker"):
        grad = getattr(model, name).grad
        total = 0.0 if grad is None else grad.abs().sum().item()
        assert total > 0, f"{name} received no gradient -- it is wired in but inert"


# ── 2. sensitivity ─────────────────────────────────────────────────────────────────────


def test_bound_code_identity_changes_output():
    model = tiny_model()
    ref = make_batch()
    moved = make_batch()
    moved.q_bound_codes[:, 1] = 11  # was 10
    with torch.no_grad():
        _, a = model(ref)
        _, b = model(moved)
    assert (b.logits - a.logits).abs().max().item() > LIVE


def test_start_code_identity_changes_output():
    """The issue-#27 twin of the above: *when the window opens* is part of the question asked."""
    model = tiny_model()
    ref = make_batch()
    moved = make_batch()
    moved.q_start_codes[:, 2] = 8  # was 9
    with torch.no_grad():
        _, a = model(ref)
        _, b = model(moved)
    assert (b.logits - a.logits).abs().max().item() > LIVE


def test_markers_separate_a_code_from_the_role_it_plays():
    """The same code id must not embed identically as a boundary and as a query subject.

    One role richer than the encoder-decoder original, because a multitask window has *two*
    code-valued slots: a single shared (or zero) marker would collapse "ends at X" into "opens
    at X" into "X itself".  ``test_window_specs_use_matching_rows_and_distinct_roles`` pins that
    the markers are *added* on the right paths and are distinct objects; this pins that they are
    distinct *values*, which an all-zero init would satisfy the first way and not the second.
    """
    model = tiny_model()
    ids = torch.tensor([[10, 10]])
    with torch.no_grad():
        plain = model.HF_model.get_input_embeddings()(ids)
        as_bound = plain + model.bound_marker
        as_start = plain + model.start_marker
    assert (as_bound - plain).abs().max().item() > LIVE, "bound_marker is inert: bounded-by-X == X"
    assert (as_start - plain).abs().max().item() > LIVE, "start_marker is inert: opens-at-X == X"
    assert (as_bound - as_start).abs().max().item() > LIVE, "the two roles share one marker"


# ── 3. RoPE ────────────────────────────────────────────────────────────────────────────


def test_time_positions_reach_the_backbone():
    """Same tokens, different elapsed times must give different window hidden states.

    Measured at ``window_hidden_states`` -- the tensor the tied readout reads -- rather than at
    the logits, for the same reason the encoder-decoder version measured the encoder output: it
    is where the effect lives, and it does not depend on the readout's initialisation.
    """
    model = tiny_model(use_rope_time=True)
    even = make_batch(time_pos_ids=torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]))
    uneven = make_batch(time_pos_ids=torch.tensor([[0, 240, 1000, 5000], [0, 5, 9, 30]]))
    with torch.no_grad():
        h_even = model.window_hidden_states(even)
        h_uneven = model.window_hidden_states(uneven)
    assert (h_even - h_uneven).abs().max().item() > LIVE


def test_rope_without_time_positions_refuses_rather_than_falling_back():
    model = tiny_model(use_rope_time=True)
    with pytest.raises(ValueError, match="time_pos_ids"):
        model(make_batch())


# ── 4. atom invariance ─────────────────────────────────────────────────────────────────


def test_atomic_batch_is_bit_identical_with_and_without_the_feature_machinery():
    """The property every reported AUROC rests on -- the eval grid is entirely atomic queries.

    ``q_bound_codes`` is a required field of ``MultitaskBoundaryBatch``, so "without the feature
    tensors" means the two things that are genuinely optional or inert on an atomic window: the
    issue-#27 start pair left off the batch entirely (the pre-#24 on-disk form), and the two role
    markers, which an all-zero bound/start column must never reach.
    """
    model = tiny_model()
    with torch.no_grad():
        _, explicit = model(_atomic())
        _, legacy = model(_atomic(starts=False))
        assert torch.equal(explicit.logits, legacy.logits), (
            "attaching all-zero start tensors perturbs a purely atomic batch"
        )

        model.bound_marker.add_(10.0)
        model.start_marker.add_(10.0)
        _, marked = model(_atomic())
    assert torch.equal(explicit.logits, marked.logits), (
        "the role markers reach a window that is neither event-bounded nor event-started; every "
        "reported score would then describe a different model than the one that was trained"
    )
