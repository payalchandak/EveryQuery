"""All three ported features enabled at once.

Each feature has its own suite; this one exists because the reason they were folded into one
model behind flags — rather than shipped as the upstream forks' separate subclass towers — was so
that any combination could run together.  That claim needs a test, not just an argument.

The interactions that could plausibly break are checked individually below:

- RoPE strips the delta tokens and drives rotary from elapsed time; event bounds put a code in
  the window's end slot.  They touch the position ids and the token embeddings respectively and
  must not interact.
- The ontology wrapper substitutes the shared token table, so a window's boundary code and start
  code go through it too — which is the intended generalisation, and worth pinning.
- ``TIMELINE//END`` must survive the delta strip, because it is this model's entire censoring
  mechanism.

Originally written against the encoder-decoder conditional query-sequence model (since deleted
with the rest of that pipeline), the only model carrying all three flags when they landed.  It
now drives
:class:`~every_query.model.conditional_multitask_ar_model.ConditionalMultitaskARModel`.  The
composition claim is if anything sharper there: that model has *four* code slots reading the
shared table (patient stream, window start event, window boundary event, conditioning code), and
its window token carries a start spec and an end spec rather than a single duration slot.
"""

import torch

from every_query.data.query_seq_dataset import EOS_CODE, NO_BOUND_INDEX
from every_query.data.rope_time import DELTA_TOKEN_PREFIX, build_keep_mask, delta_vocab_ids
from every_query.model.conditional_multitask_ar_model import TOKENS_PER_WINDOW
from every_query.model.ontology_embedding import OntologyEmbedding

# The multitask model's own construction idiom, reused rather than re-invented.
from tests.test_conditional_multitask_ar_model import VOCAB, make_batch, ontology_model

# ``make_batch``'s patient stream is four tokens wide (row 1 right-padded to two).
TIMES = torch.tensor([[0, 24, 36, 84], [0, 24, 0, 0]])


def _all_features_model(tmp_path):
    """RoPE time + event bounds + ontology, all on one ``ConditionalMultitaskARModel``."""
    return ontology_model(tmp_path, use_rope_time=True)


def _all_features_batch():
    """Three windows: one event-bounded, one event-started, all timed.  Plus RoPE positions.

    ``make_batch``'s defaults already span the three features — window 1 ends at boundary code
    10, window 2 opens at start event 9, and the rest are duration specs — so the batch is taken
    from the shared idiom rather than hand-rolled here.
    """
    batch = make_batch(time_pos_ids=TIMES)
    assert batch.q_bound_codes[0].tolist() == [NO_BOUND_INDEX, 10, NO_BOUND_INDEX]
    assert batch.q_start_codes[0].tolist() == [NO_BOUND_INDEX, NO_BOUND_INDEX, 9]
    return batch


def test_all_three_features_run_together(tmp_path):
    """The composability claim: RoPE + event bounds + ontology in one forward."""
    model, _, v_ext = _all_features_model(tmp_path)
    loss, out = model(_all_features_batch())
    assert loss.isfinite()
    assert out.logits.shape == (2, 3, v_ext)
    assert out.logits.isfinite().all()
    # Leaf-only targets were widened against the ontology, not merely accepted at face value.
    assert v_ext > VOCAB and model.base_vocab_size == VOCAB


def test_all_three_features_train_together(tmp_path):
    model, _, _ = _all_features_model(tmp_path)
    model.train()
    loss, _ = model(_all_features_batch())
    loss.backward()

    # Every feature's own parameters must receive gradient in the combined configuration.
    raw = model.HF_model.get_input_embeddings().tok
    for name, param in (
        ("ontology raw table", raw.weight),
        ("bound_marker", model.bound_marker),
        ("start_marker", model.start_marker),
    ):
        assert param.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} gradient is not finite"
        assert param.grad.abs().sum() > 0, f"{name} gradient is identically zero"


def test_ontology_wrapper_covers_the_query_slots_too(tmp_path):
    """A window's boundary and start codes must go through the mixed table, not around it."""
    model, _, _ = _all_features_model(tmp_path)
    assert isinstance(model.HF_model.get_input_embeddings(), OntologyEmbedding)

    batch = _all_features_batch()
    wrapper = model.HF_model.get_input_embeddings()

    # Record the id tensors themselves, not their shapes: q_bound_codes and q_start_codes are
    # both (B, K), so a shape alone cannot tell which of them reached the wrapper.
    calls = []
    original = wrapper.forward
    wrapper.forward = lambda ids: (calls.append(ids.detach().clone()), original(ids))[1]
    try:
        with torch.no_grad():
            model(batch)
    finally:
        wrapper.forward = original

    def was_passed(t):
        return any(c.shape == t.shape and torch.equal(c, t) for c in calls)

    assert was_passed(batch.code), "patient stream must go through the wrapper"
    assert was_passed(batch.q_bound_codes), "boundary codes must go through the wrapper"
    assert was_passed(batch.q_start_codes), "start codes must go through the wrapper"
    assert was_passed(batch.condition_codes), "conditioning codes must go through the wrapper"


def test_rope_time_still_moves_the_backbone_with_everything_on(tmp_path):
    """RoPE must not be neutralised by the other features sharing the backbone."""
    model, _, _ = _all_features_model(tmp_path)

    def hidden(time_pos):
        batch = make_batch(time_pos_ids=time_pos)
        with torch.no_grad():
            return model.window_hidden_states(batch)

    near = hidden(torch.tensor([[0, 1, 2, 3], [0, 1, 0, 0]]))
    far = hidden(torch.tensor([[0, 240, 1000, 5000], [0, 900, 0, 0]]))
    assert not torch.allclose(near, far)


def test_eos_code_survives_the_delta_strip():
    """TIMELINE//END is the model's whole censoring mechanism; the strip must not eat it.

    The strip keys on the ``TIMELINE//DELTA`` prefix, and ``TIMELINE//END`` does not start with
    it — but the two names are close enough that a prefix widened to ``TIMELINE`` would silently
    delete censoring from every sequence.
    """
    vocab = {EOS_CODE: 5, f"{DELTA_TOKEN_PREFIX}//1d": 9, f"{DELTA_TOKEN_PREFIX}//1h": 10}
    delta_ids = delta_vocab_ids(vocab)
    assert 5 not in delta_ids.tolist()

    code = torch.tensor([[5, 9, 5, 10]])
    keep = build_keep_mask(code, delta_ids)
    assert keep.tolist() == [[True, False, True, False]]


def test_event_bounds_own_the_window_end_and_leave_the_conditioning_alone(tmp_path):
    """A boundary replaces *what closes the window*, never *what is being asked about*.

    On the encoder-decoder model that meant "the duration slot, not the code slot" of a query
    block.  Here a window token has no code of its own — the model scores the whole vocabulary —
    so the thing that must stay untouched is the conditioning stream ``C_i`` / ``A_i``, and the
    thing that must move is the bounded window's own token and no other.
    """
    model, _, _ = _all_features_model(tmp_path)
    bounded = _all_features_batch()
    unbounded = _all_features_batch()
    # Window 1 alone changes: its boundary is dropped and its sentinel duration becomes a real
    # horizon.  Every other window's spec is left byte-identical, so a difference anywhere else
    # is the boundary leaking, not the fixture moving.
    unbounded.q_bound_codes[:, 1] = NO_BOUND_INDEX
    unbounded.q_durations[:, 1] = 30.0

    with torch.no_grad():
        a = model._query_tokens(bounded)
        b = model._query_tokens(unbounded)

    torch.testing.assert_close(a[:, 1::TOKENS_PER_WINDOW], b[:, 1::TOKENS_PER_WINDOW])
    torch.testing.assert_close(a[:, 2::TOKENS_PER_WINDOW], b[:, 2::TOKENS_PER_WINDOW])

    windows_a, windows_b = a[:, 0::TOKENS_PER_WINDOW], b[:, 0::TOKENS_PER_WINDOW]
    moved = [k for k in range(3) if not torch.equal(windows_a[:, k], windows_b[:, k])]
    assert moved == [1], f"a boundary on window 1 changed windows {moved}"


def test_batch_validates_every_optional_tensor_together(tmp_path):
    """All the optional per-window tensors are shape-checked in the combined configuration."""
    batch = _all_features_batch()
    expected = (batch.batch_size, batch.num_bounds)
    assert batch.q_bound_codes.shape == expected
    assert batch.q_start_codes.shape == batch.q_start_durations.shape == expected
    assert batch.condition_codes.shape == batch.condition_answers.shape == (expected[0], expected[1] - 1)
    assert batch.time_pos_ids.shape == batch.code.shape
