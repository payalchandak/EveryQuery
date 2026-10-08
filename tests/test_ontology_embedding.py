"""``OntologyEmbedding``: the mixed table, its gradients, its cache, and its installation.

Every claim here is asserted at the level where the effect lives.  The handoff's two traps apply
throughout: a bare ``assert not torch.equal(a, b)`` is satisfied by one ULP of float32 rounding,
and a randomly-initialised decoder compresses an 8e-05 encoder difference to ~1e-07 at the logits.
So liveness is asserted with a margin, and against the embedding output rather than the logits.

The model-level halves — the pre-hook in section 3, all of section 4, and the checkpoint /
determinism tests in section 5 — were originally driven through the encoder-decoder conditional
query-sequence model (since deleted with the rest of that pipeline), because it was the only
model that installed the wrapper.  They now run against
:class:`~every_query.model.conditional_multitask_ar_model.ConditionalMultitaskARModel`.
``OntologyEmbedding`` and ``wrap_tok_embeddings`` themselves are shared and unchanged, so every
section-1/2 claim is model-independent and untouched.
"""

from __future__ import annotations

import copy

import polars as pl
import pytest
import torch

from every_query.data.ontology import (
    EMBEDDING_MIX_FILE,
    EVENT_TO_QUERY_NODES_FILE,
    ONTOLOGY_VOCAB_FILE,
    build_event_to_query_nodes,
    build_ontology,
    load_mix_matrix,
)
from every_query.model.ontology_embedding import OntologyEmbedding, wrap_tok_embeddings

# The multitask model's own construction idiom, reused rather than re-invented.
from tests.test_conditional_multitask_ar_model import make_batch, ontology_model, tiny_model

#: Liveness margin.  Anything smaller is float noise, not an effect.
LIVE = 1e-4


@pytest.fixture(autouse=True)
def _no_demo_model():
    """Override the repo-root autouse fixture; nothing here downloads a HuggingFace model."""
    yield


# ── helpers ─────────────────────────────────────────────────────────────


def _sparse(entries: list[tuple[int, int, float]], n: int) -> torch.Tensor:
    """A sparse ``(n, n)`` matrix from ``(row, col, value)`` triples."""
    idx = torch.tensor([[r for r, _, _ in entries], [c for _, c, _ in entries]], dtype=torch.long)
    val = torch.tensor([v for _, _, v in entries], dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, size=(n, n)).coalesce()


def _raw(n: int, h: int, seed: int = 0) -> torch.nn.Embedding:
    torch.manual_seed(seed)
    emb = torch.nn.Embedding(n, h)
    with torch.no_grad():
        # Distinct, easily-identified rows: row i is the constant vector i.
        emb.weight.copy_(torch.arange(n, dtype=torch.float32).unsqueeze(1).expand(n, h).contiguous())
    return emb


def _write_ontology(out, codes: list[str]):
    out.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame({"code": codes, "code/vocab_index": list(range(1, len(codes) + 1))})
    nodes, mix = build_ontology(frame)
    nodes.write_parquet(out / ONTOLOGY_VOCAB_FILE)
    mix.write_parquet(out / EMBEDDING_MIX_FILE)
    build_event_to_query_nodes(nodes, mix).write_parquet(out / EVENT_TO_QUERY_NODES_FILE)
    return out


def _unwrapped_model(vocab_size: int):
    """A multitask model sized to ``vocab_size`` with no ontology installed.

    Used by the ``wrap_tok_embeddings`` size checks, which must construct the mismatch that a
    correctly-sized ``ontology_model`` never can.
    """
    return tiny_model(config_overrides={"vocab_size": vocab_size})


# ── 1. the mixed table itself ───────────────────────────────────────────


def test_lookup_equals_the_explicit_matrix_product():
    """``emb(ids)`` must equal ``(A @ W)[ids]`` computed the long way."""
    n, h = 6, 4
    raw = _raw(n, h)
    mix = _sparse([(0, 0, 1.0), (1, 1, 0.5), (1, 0, 0.5), (2, 2, 1.0)], n)
    emb = OntologyEmbedding(raw, mix)

    expected = torch.sparse.mm(mix, raw.weight)
    ids = torch.tensor([0, 1, 2, 1])
    torch.testing.assert_close(emb(ids), expected[ids])


def test_orientation_is_row_is_node_not_column_is_node():
    """Row ``i`` of ``A`` must be node ``i``'s recipe, not its contribution to others.

    The two orientations are only distinguishable on an asymmetric matrix, so node 1 mixes node 0 while node 0
    mixes nothing.  Under the transposed reading node 0 would be the mixed one, which this pins.
    """
    n, h = 3, 2
    raw = _raw(n, h)
    mix = _sparse([(0, 0, 1.0), (1, 0, 0.5), (1, 1, 0.5), (2, 2, 1.0)], n)
    emb = OntologyEmbedding(raw, mix)

    # Node 0 has only a self-loop, so it is untouched.
    torch.testing.assert_close(emb(torch.tensor([0])), raw.weight[0:1])
    # Node 1 is the average of rows 0 (all 0s) and 1 (all 1s).
    torch.testing.assert_close(emb(torch.tensor([1])), torch.full((1, h), 0.5))
    # And the transposed reading is genuinely different here, so the test can fail.
    transposed = torch.sparse.mm(mix.t().coalesce(), raw.weight)
    assert (transposed[0] - raw.weight[0]).abs().max() > LIVE


def test_rows_are_normalised_including_the_self_row(tmp_path):
    """``load_mix_matrix`` must make every node's row sum to 1, self-loop included."""
    _write_ontology(tmp_path, ["A//B//C", "A//B//D", "Z"])
    a = load_mix_matrix(tmp_path, normalize=True).to_dense()

    sums = a.sum(dim=1)
    nonzero = sums[sums > 0]
    torch.testing.assert_close(nonzero, torch.ones_like(nonzero))

    unnormalised = load_mix_matrix(tmp_path, normalize=False).to_dense()
    assert unnormalised.sum(dim=1).max() > 1.0 + LIVE, "nothing to normalise; test is vacuous"


def test_identity_mixing_reproduces_an_ordinary_embedding():
    """``A = I`` must be indistinguishable from the raw table."""
    n, h = 8, 5
    raw = _raw(n, h, seed=3)
    emb = OntologyEmbedding(raw, _sparse([(i, i, 1.0) for i in range(n)], n))
    ids = torch.tensor([[0, 3], [7, 1]])
    torch.testing.assert_close(emb(ids), raw(ids))


def test_padding_row_embeds_to_zero_and_does_not_blow_up():
    """Row 0 (PAD) is never a node, so it has no mix entries; the clamp must keep it finite."""
    n, h = 4, 3
    raw = _raw(n, h)
    emb = OntologyEmbedding(raw, _sparse([(1, 1, 1.0), (2, 2, 1.0), (3, 3, 1.0)], n))
    out = emb(torch.tensor([0]))
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, torch.zeros(1, h))


def test_multi_parent_weights_are_what_the_matrix_says():
    """A node with two parents mixes both at their declared weights."""
    n, h = 4, 2
    raw = _raw(n, h)
    # node 3 = 0.5*self + 0.25*node1 + 0.25*node2
    mix = _sparse([(3, 3, 0.5), (3, 1, 0.25), (3, 2, 0.25), (1, 1, 1.0), (2, 2, 1.0)], n)
    emb = OntologyEmbedding(raw, mix)
    expected = 0.5 * 3.0 + 0.25 * 1.0 + 0.25 * 2.0
    torch.testing.assert_close(emb(torch.tensor([3])), torch.full((1, h), expected))


# ── 2. gradients ────────────────────────────────────────────────────────


def test_descendant_lookup_sends_gradient_to_its_ancestor_rows():
    """The whole point: an ancestor row is trained by every descendant that occurs."""
    n, h = 5, 3
    raw = _raw(n, h)
    mix = _sparse([(2, 2, 0.5), (2, 1, 0.5), (1, 1, 1.0)], n)
    emb = OntologyEmbedding(raw, mix)

    emb(torch.tensor([2])).sum().backward()
    grad = raw.weight.grad

    assert grad[2].abs().sum() > LIVE, "the node's own row got no gradient"
    assert grad[1].abs().sum() > LIVE, "the ancestor row got no gradient"
    torch.testing.assert_close(grad[1], torch.full((h,), 0.5))


def test_unrelated_rows_receive_no_gradient():
    """A node outside the closure must stay untouched, or 'mixing' is just leakage."""
    n, h = 5, 3
    raw = _raw(n, h)
    mix = _sparse([(2, 2, 0.5), (2, 1, 0.5), (1, 1, 1.0), (4, 4, 1.0)], n)
    emb = OntologyEmbedding(raw, mix)

    emb(torch.tensor([2])).sum().backward()
    grad = raw.weight.grad
    assert grad[4].abs().max() == 0.0, "an unrelated row received gradient"
    assert grad[3].abs().max() == 0.0, "a node with no mix row received gradient"


# ── 3. the per-forward cache ────────────────────────────────────────────


def test_cache_is_reused_within_a_forward_and_dropped_by_clear():
    n, h = 4, 2
    emb = OntologyEmbedding(_raw(n, h), _sparse([(i, i, 1.0) for i in range(n)], n))

    first = emb.mixed_weight()
    assert emb.mixed_weight() is first, "the mixed table was recomputed within one forward"
    emb.clear_cache()
    assert emb._mixed is None
    assert emb.mixed_weight() is not first


def test_a_cached_table_cannot_go_stale_across_an_optimizer_step():
    """The failure this guards: a weight update that the next forward does not see."""
    n, h = 4, 2
    raw = _raw(n, h)
    emb = OntologyEmbedding(raw, _sparse([(i, i, 1.0) for i in range(n)], n))

    before = emb(torch.tensor([2])).clone()
    with torch.no_grad():
        raw.weight[2] += 1.0
    emb.clear_cache()
    after = emb(torch.tensor([2]))

    assert (after - before).abs().max() > LIVE, "the update was invisible: the cache went stale"


def test_the_model_pre_hook_clears_the_cache_once_per_forward(tmp_path):
    """A real ``model(batch)`` must start from a cleared cache, not inherit a stale product.

    Asserted through the real hook that ``wrap_tok_embeddings`` registers, not a hand-fired
    stand-in: the registration itself is half of what could break.
    """
    model, _, _ = ontology_model(tmp_path)
    table = model.HF_model.get_input_embeddings()

    # Warm the cache outside a forward, then mutate the raw weights underneath it.
    stale = table.mixed_weight().clone()
    with torch.no_grad():
        table.weight[3] += 5.0

    with torch.no_grad():
        model(make_batch())

    fresh = table.mixed_weight()
    assert (fresh - stale).abs().max() > LIVE, (
        "the forward reused a cached table computed before the weight change: the pre-hook "
        "either was not registered or did not fire"
    )


def test_the_evaluation_path_clears_the_cache_without_going_through_forward(tmp_path):
    """The pre-hook is registered on ``forward``, and the evaluation path does not call it.

    ``ConditionalMultitaskARModel.score_final_query`` reaches ``window_hidden_states`` directly,
    so the registration above covers only half of this model.  ``window_hidden_states`` clears
    the cache itself for exactly that reason; without it, every evaluation pass after the first
    would score against the mixed table as it stood at the last training forward.  The
    encoder-decoder model had a single entry point and so could not express this failure.
    """
    model, _, _ = ontology_model(tmp_path)
    table = model.HF_model.get_input_embeddings()
    stale = table.mixed_weight().clone()
    with torch.no_grad():
        table.weight[3] += 5.0
        model.score_final_query(make_batch(), torch.tensor([2, 3]))
    assert (table.mixed_weight() - stale).abs().max() > LIVE, (
        "score_final_query reused a table cached before the weight change"
    )


def test_two_backward_passes_do_not_reuse_one_graph():
    """A cached product holds an autograd graph; reusing it across backwards would raise."""
    n, h = 4, 2
    raw = _raw(n, h)
    emb = OntologyEmbedding(raw, _sparse([(i, i, 1.0) for i in range(n)], n))

    emb(torch.tensor([1])).sum().backward()
    emb.clear_cache()
    emb(torch.tensor([1])).sum().backward()  # must not raise "backward through the graph a second time"


# ── 4. installation through the HuggingFace contract ────────────────────


def test_wrap_installs_through_get_and_set_input_embeddings(tmp_path):
    _write_ontology(tmp_path, [f"G//{i}" for i in range(1, 15)])
    v_ext = int(pl.read_parquet(tmp_path / ONTOLOGY_VOCAB_FILE)["token_id"].max()) + 1
    model = _unwrapped_model(v_ext)

    wrapper = wrap_tok_embeddings(model, load_mix_matrix(tmp_path))
    assert model.HF_model.get_input_embeddings() is wrapper, (
        "the wrapper is not what the backbone hands out; the shared-table claim is void"
    )
    assert isinstance(wrapper, OntologyEmbedding)


def test_wrap_refuses_a_table_that_is_not_v_ext(tmp_path):
    """Too few rows means every ancestor index is out of range — fail loudly, not later."""
    _write_ontology(tmp_path, [f"G//{i}" for i in range(1, 15)])
    model = _unwrapped_model(8)
    with pytest.raises(ValueError, match="V_ext"):
        wrap_tok_embeddings(model, load_mix_matrix(tmp_path))


def test_every_code_slot_shares_one_mixed_table(tmp_path):
    """The composition claim: every call site reaches the same module.

    On the encoder-decoder model the three slots were patient stream / query code / boundary
    code.  ``ConditionalMultitaskARModel`` has four — the patient stream, a window's start event,
    a window's boundary event, and the conditioning code — and substituting the module rather
    than the call sites is what makes all four inherit the ontology for free.  Asserted by
    recording the id tensors themselves, not their shapes: ``condition_codes`` and
    ``q_bound_codes`` are both ``(B, K)``-ish, so a shape alone cannot tell which reached it.
    """
    model, _, _ = ontology_model(tmp_path)
    table = model.HF_model.get_input_embeddings()
    assert isinstance(table, OntologyEmbedding)

    batch = make_batch()
    calls: list[torch.Tensor] = []
    original = table.forward
    table.forward = lambda ids: (calls.append(ids.detach().clone()), original(ids))[1]
    try:
        with torch.no_grad():
            model(batch)
    finally:
        table.forward = original

    def was_passed(t):
        return any(c.shape == t.shape and torch.equal(c, t) for c in calls)

    assert was_passed(batch.code), "the patient stream must go through the wrapper"
    assert was_passed(batch.q_start_codes), "window start events must go through the wrapper"
    assert was_passed(batch.q_bound_codes), "window boundary events must go through the wrapper"
    assert was_passed(batch.condition_codes), "conditioning codes must go through the wrapper"

    # And the same id read twice gives the same vector — the module is one shared table, not a
    # per-slot copy that could drift.
    ids = torch.tensor([3, 4])
    once = table(ids).clone()
    table.clear_cache()
    torch.testing.assert_close(once, table(ids))
    assert model.HF_model.get_input_embeddings() is table


def test_the_readout_uses_the_mixed_table_not_the_raw_one(tmp_path):
    """The assumption the encoder-decoder version could only state, now checkable.

    That model had no tied output head, so this test asserted "there is no tied output head here;
    if one is added it must use the MIXED table".  ``ConditionalMultitaskARModel`` adds exactly
    that head, so the conditional becomes a fact: the readout is the effective table ``A @ W``
    that every input lookup reads, and no parameter outside the backbone aliases the raw one.
    (The numerical equality of readout and mixed table is pinned in
    ``test_conditional_multitask_ar_model.py::test_tied_readout_uses_the_mixed_table_under_an_ontology``.)
    """
    model, _, _ = ontology_model(tmp_path)
    table = model.HF_model.get_input_embeddings()
    raw_w = table.weight  # the underlying learned parameter
    tied = [name for name, p in model.named_parameters() if p is raw_w and not name.startswith("HF_model.")]
    assert not tied, f"a non-backbone parameter is tied to the raw embedding table: {tied}"
    assert "lm_head" not in dict(model.named_modules())

    readout = model._readout_weight()
    assert readout is not raw_w
    assert (readout - raw_w.detach()).abs().max() > LIVE, (
        "the readout projects onto the raw table, scoring ancestors through rows the input side never sees"
    )


# ── 5. checkpoint, dtype and device ─────────────────────────────────────


def test_checkpoint_round_trip_preserves_predictions(tmp_path):
    """The wrapper changes state-dict keys, so a silent key mismatch is the failure mode."""
    model, _, _ = ontology_model(tmp_path)
    # ``tiny_model`` seeds, so two constructions are bit-identical: perturb one, or the reload
    # below would be indistinguishable from doing nothing.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    batch = make_batch()
    with torch.no_grad():
        before = model(batch)[1].logits.clone()

    state = copy.deepcopy(model.state_dict())
    reloaded, _, _ = ontology_model(tmp_path)
    with torch.no_grad():
        assert not torch.allclose(reloaded(batch)[1].logits, before), "the reload cannot be vacuous"

    missing, unexpected = reloaded.load_state_dict(state, strict=True), None
    assert unexpected is None
    with torch.no_grad():
        after = reloaded(batch)[1].logits

    torch.testing.assert_close(before, after)
    assert missing is not None  # load_state_dict returns a NamedTuple; keys matched under strict


def test_the_sparse_mix_is_not_persisted_but_is_rebuilt(tmp_path):
    """`mix` is a non-persistent buffer: it must be absent from the state dict AND rebuilt on load.

    If it were persisted, a checkpoint would silently pin an old ontology; if it were neither persisted nor
    rebuilt, the model would load with no mixing at all and look fine.
    """
    model, _, _ = ontology_model(tmp_path)

    keys = [k for k in model.state_dict() if k.endswith(".mix")]
    assert not keys, f"the sparse mix was persisted into the state dict: {keys}"

    table = model.HF_model.get_input_embeddings()
    assert table.mix._nnz() > 0, "the mix was not rebuilt from ontology_dir on construction"


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mixing_survives_dtype_changes(dtype, tmp_path):
    """Under autocast the sparse fp32 matrix must not meet a bf16 dense operand."""
    n, h = 6, 4
    raw = _raw(n, h).to(dtype)
    mix = _sparse([(2, 2, 0.5), (2, 1, 0.5), (1, 1, 1.0)], n)
    emb = OntologyEmbedding(raw, mix)

    out = emb(torch.tensor([2]))
    assert out.dtype == dtype
    expected = 0.5 * 2.0 + 0.5 * 1.0
    torch.testing.assert_close(out.float(), torch.full((1, h), expected), rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_cpu_and_gpu_agree_within_tolerance():
    n, h = 6, 4
    raw = _raw(n, h)
    mix = _sparse([(2, 2, 0.5), (2, 1, 0.5), (1, 1, 1.0)], n)

    cpu = OntologyEmbedding(raw, mix)
    ids = torch.tensor([1, 2])
    on_cpu = cpu(ids)

    gpu = OntologyEmbedding(copy.deepcopy(raw), mix.clone()).cuda()
    on_gpu = gpu(ids.cuda()).cpu()

    torch.testing.assert_close(on_cpu, on_gpu, rtol=1e-5, atol=1e-6)


def test_lookup_is_deterministic_across_repeated_calls(tmp_path):
    """Single-device determinism is the precondition for any DDP-equality claim."""
    model, _, _ = ontology_model(tmp_path)
    batch = make_batch()

    with torch.no_grad():
        runs = [model(batch)[1].logits.clone() for _ in range(3)]
    for later in runs[1:]:
        torch.testing.assert_close(runs[0], later)
