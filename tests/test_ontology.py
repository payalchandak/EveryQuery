"""Tests for ontology / hierarchical embeddings.

Covered in pipeline order:

1. DAG construction — prefix semantics, index layout (leaves untouched, ancestors appended),
   the "every indexed node has a mix row" invariant, decay, and the two hazards this port fixes
   (unknown codes silently dropped by the closure join).
2. Embedding wrapper — the mix arithmetic, gradient reaching ancestor rows, and the per-forward
   cache actually being a cache.
3. Query addressing — ancestors becoming queryable, and the universe mixing that keeps the
   sampler's RNG contract intact.
4. Model integration — that the wrapper composes with the window and condition slots rather than
   only the patient stream.

Section 4 (and the two ``wrap_tok_embeddings`` size checks in section 2) were originally written
against the encoder-decoder conditional query-sequence model (since deleted with the rest of that
pipeline), because it was the only model when the ontology landed.  They now run against
:class:`~every_query.model.conditional_multitask_ar_model.ConditionalMultitaskARModel`, the
surviving consumer.  Sections 1-3 are about :mod:`every_query.data.ontology` and the query
universe and are model-independent.
"""

import polars as pl
import pytest
import torch

from every_query.data.ontology import (
    EMBEDDING_MIX_FILE,
    EVENT_TO_QUERY_NODES_FILE,
    ONTOLOGY_VOCAB_FILE,
    build_event_to_query_nodes,
    build_ontology,
    expand_events_to_query_nodes,
    extend_code_map,
    extended_vocab_size,
    load_mix_matrix,
    string_ancestors,
)
from every_query.generate_tasks.query_sequence_labeling import build_query_universe
from every_query.model.ontology_embedding import OntologyEmbedding, wrap_tok_embeddings

# The multitask model's own construction idiom, reused rather than re-invented.
from tests.test_conditional_multitask_ar_model import make_batch, ontology_model, tiny_model


def _codes(codes, parents=None) -> pl.DataFrame:
    data = {"code": codes, "code/vocab_index": list(range(1, len(codes) + 1))}
    if parents is not None:
        data["parent_codes"] = parents
    return pl.DataFrame(data)


def _write_ontology(tmp_path, codes_df, decay: float = 0.5):
    nodes, mix = build_ontology(codes_df, decay=decay)
    closure = build_event_to_query_nodes(nodes, mix)
    nodes.write_parquet(tmp_path / ONTOLOGY_VOCAB_FILE)
    mix.write_parquet(tmp_path / EMBEDDING_MIX_FILE)
    closure.write_parquet(tmp_path / EVENT_TO_QUERY_NODES_FILE)
    return nodes, mix, closure


# ── 1. DAG construction ─────────────────────────────────────────────────


def test_single_slash_is_not_a_separator():
    """MEDS uses `//` between levels; a lone `/` inside a code is part of the name."""
    assert string_ancestors("ICD10CM/A04.72") == []
    assert string_ancestors("LAB//220645//mEq/L") == ["LAB//220645", "LAB"]


def test_leaf_indices_are_preserved():
    """An ontology must be droppable onto an existing cohort without renumbering it."""
    nodes, _ = build_ontology(_codes(["A//B//C", "A//B//D", "E"]))
    leaves = nodes.filter(pl.col("is_observed_code")).sort("token_id")
    assert leaves["node_name"].to_list() == ["A//B//C", "A//B//D", "E"]
    assert leaves["token_id"].to_list() == [1, 2, 3]


def test_ancestors_are_appended_above_the_highest_leaf():
    nodes, _ = build_ontology(_codes(["A//B//C", "E"]))
    max_leaf = nodes.filter(pl.col("is_observed_code"))["token_id"].max()
    assert (nodes.filter(~pl.col("is_observed_code"))["token_id"] > max_leaf).all()


def test_every_indexed_node_has_a_mix_row():
    """Otherwise the node embeds to the zero vector, permanently and silently."""
    nodes, mix = build_ontology(_codes(["A//B//C", "X//Y"], parents=[["G//H//I"], None]))
    assert set(nodes["token_id"].to_list()) == set(mix["target_token_id"].unique().to_list())


def test_parent_codes_prefixes_are_closed_to_a_fixed_point():
    """A grouper's own prefixes, and their prefixes, must all become nodes."""
    nodes, _ = build_ontology(_codes(["LEAF"], parents=[["G//H//I"]]))
    ancestors = set(nodes.filter(~pl.col("is_observed_code"))["node_name"].to_list())
    assert {"G//H//I", "G//H", "G"} <= ancestors


def test_decay_controls_ancestor_weight():
    """Decay=0 is the structure-without-mixing control: each node is only itself."""
    _, mix_half = build_ontology(_codes(["A//B//C"]), decay=0.5)
    _, mix_zero = build_ontology(_codes(["A//B//C"]), decay=0.0)

    def leaf_components(mix):
        rows = mix.filter((pl.col("target_token_id") == 1) & (pl.col("unnormalized_weight") > 0))
        return set(rows["component_token_id"].to_list())

    assert leaf_components(mix_zero) == {1}, "no ancestor may carry weight at decay=0"
    # At decay=0.5 the leaf mixes itself plus both prefixes (A//B, A).
    assert len(leaf_components(mix_half)) == 3


def test_separator_like_characters_do_not_drop_ancestors():
    """Queries are bare codes, so ``|``, ``>`` and ``&`` in a name are just characters."""
    nodes, _ = build_ontology(_codes(["A&B//C", "X|Y//Z", "P>Q//R", "PLAIN//X"]))
    ancestors = set(nodes.filter(~pl.col("is_observed_code"))["node_name"].to_list())
    assert {"A&B", "X|Y", "P>Q", "PLAIN"} <= ancestors


def test_parenthesised_ancestor_names_are_not_dropped():
    """Parentheses are not separators, and most of a real MEDS vocabulary contains them.

    Reserving them cost the ancestor names of 7,804 of 13,908 MIMIC-IV codes: value-bin
    segments like ``value_[4.0,6.0)`` and unit names like ``(MICU)`` are ordinary parts of a
    code, not grammar, and an ancestor built from them is perfectly addressable.
    """
    codes = ["ICU//STAY (MICU)//LOS", "LAB//GLUCOSE//value_[4.0,6.0)"]
    nodes, _ = build_ontology(_codes(codes))
    ancestors = set(nodes.filter(~pl.col("is_observed_code"))["node_name"].to_list())
    assert "ICU//STAY (MICU)" in ancestors
    assert "LAB//GLUCOSE//value_[4.0,6.0)" not in ancestors, "that one is a leaf, not an ancestor"
    assert "LAB//GLUCOSE" in ancestors


def test_mix_rows_are_normalised(tmp_path):
    _write_ontology(tmp_path, _codes(["A//B//C", "A//B//D"]))
    mix = load_mix_matrix(tmp_path)
    row_sums = torch.sparse.sum(mix, dim=1).to_dense()
    nonzero = row_sums[row_sums > 0]
    assert torch.allclose(nonzero, torch.ones_like(nonzero), atol=1e-6)


def test_closure_pairs_each_leaf_with_itself_and_its_ancestors(tmp_path):
    _, _, closure = _write_ontology(tmp_path, _codes(["A//B//C"]))
    pairs = set(zip(closure["event_code"].to_list(), closure["query_node"].to_list(), strict=True))
    assert ("A//B//C", "A//B//C") in pairs
    assert ("A//B//C", "A//B") in pairs
    assert ("A//B//C", "A") in pairs


def test_explode_keeps_events_the_ontology_does_not_know():
    """An inner join here would silently DELETE data when the ontology is out of date."""
    from datetime import datetime

    events = pl.DataFrame(
        {"subject_id": [1, 1], "time": [datetime(2024, 1, 1)] * 2, "code": ["A//B", "ORPHAN"]}
    )
    closure = pl.DataFrame({"event_code": ["A//B", "A//B"], "query_node": ["A//B", "A"]})
    out = expand_events_to_query_nodes(events, closure)
    assert "ORPHAN" in out["code"].to_list(), "an unknown code must survive, not vanish"
    assert set(out["code"].to_list()) == {"A//B", "A", "ORPHAN"}


# ── 2. embedding wrapper ────────────────────────────────────────────────


def _two_node_wrapper():
    raw = torch.nn.Embedding(2, 3)
    raw.weight.data.copy_(torch.tensor([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]]))
    mix = torch.sparse_coo_tensor(
        torch.tensor([[0, 1, 1], [0, 0, 1]]), torch.tensor([1.0, 0.5, 0.5]), (2, 2)
    ).coalesce()
    return OntologyEmbedding(raw, mix), raw


def test_wrapper_returns_the_ancestor_mixed_average():
    emb, _ = _two_node_wrapper()
    assert torch.allclose(emb(torch.tensor([1])), torch.full((1, 3), 2.0))


def test_wrapper_handles_multidimensional_ids():
    """Aggregate component blocks are (B, L, K); the wrapper must index them in one call."""
    emb, _ = _two_node_wrapper()
    assert emb(torch.zeros(2, 4, 3, dtype=torch.long)).shape == (2, 4, 3, 3)


def test_gradient_reaches_ancestor_rows():
    """An ancestor never appears in a patient stream; it can only learn through the mix."""
    emb, raw = _two_node_wrapper()
    emb(torch.tensor([1])).sum().backward()
    assert raw.weight.grad[0].abs().sum() > 0, "the ancestor row must receive gradient"


def test_cache_is_reused_within_a_forward_and_cleared_between():
    emb, _ = _two_node_wrapper()
    first = emb.mixed_weight()
    assert emb.mixed_weight() is first, "the mixed table must be computed once per forward"
    emb.clear_cache()
    assert emb.mixed_weight() is not first, "clear_cache must drop the cached product"


def _identity_mix(v_ext: int) -> torch.Tensor:
    return torch.sparse_coo_tensor(torch.tensor([[0], [0]]), torch.tensor([1.0]), (v_ext, v_ext)).coalesce()


def test_wrap_rejects_an_undersized_table():
    """Sizing the backbone to V rather than V_ext puts every ancestor index out of range."""
    model = tiny_model(config_overrides={"vocab_size": 4})
    with pytest.raises(ValueError, match="V_ext"):
        wrap_tok_embeddings(model, _identity_mix(16))


def test_wrap_rejects_an_oversized_table():
    """``(V_ext, V_ext) @ (V_model, H)`` needs equality, not just enough rows.

    A backbone left at a stock 50k vocabulary satisfies "big enough" while being just as wrong
    as an undersized one — and would fail deep inside ``torch.sparse.mm`` instead.
    """
    model = tiny_model(config_overrides={"vocab_size": 99})
    with pytest.raises(ValueError, match="V_ext"):
        wrap_tok_embeddings(model, _identity_mix(16))


# ── 3. query addressing ─────────────────────────────────────────────────


def test_ancestors_become_queryable(tmp_path):
    _write_ontology(tmp_path, _codes(["A//B//C"]))
    extended = extend_code_map({"A//B//C": 1}, tmp_path)
    assert "A//B" in extended and "A" in extended


def test_a_name_that_is_both_code_and_prefix_keeps_its_leaf_index(tmp_path):
    _write_ontology(tmp_path, _codes(["A", "A//B"]))
    extended = extend_code_map({"A": 1, "A//B": 2}, tmp_path)
    assert extended["A"] == 1, "a real code's canonical index must win over any node index"


def test_extended_vocab_size_covers_every_node(tmp_path):
    nodes, _, _ = _write_ontology(tmp_path, _codes(["A//B//C", "D//E"]))
    assert extended_vocab_size(tmp_path) == int(nodes["token_id"].max()) + 1


def test_query_universe_is_untouched_without_an_ontology():
    assert build_query_universe(["A", "B"]) == ["A", "B"]
    assert build_query_universe(["A", "B"], ontology_dir=None) == ["A", "B"]


def test_query_universe_is_every_node_exactly_once_leaves_first(tmp_path):
    """Every leaf and every usable ancestor, each once, so uniform sampling is uniform over nodes.

    Leaves keep their slot indices (they come first, in input order) so a no-ontology run and an ontology run
    draw the same leaf at the same slot; ancestors follow in sorted order.
    """
    _write_ontology(tmp_path, _codes(["A//B//C", "D//E//F"]))
    universe = build_query_universe(["A//B//C", "D//E//F"], ontology_dir=tmp_path)
    assert universe == ["A//B//C", "D//E//F", "A", "A//B", "D", "D//E"]


def test_query_universe_never_drops_a_leaf_code(tmp_path):
    """A leaf missing from the universe is a code the model is never asked about.

    An earlier, sampled construction lost a long tail of the vocabulary — and on the real
    cohort it lost ``TIMELINE//END``, which is this model's entire censoring mechanism.
    """
    leaves = [f"LAB//{i}" for i in range(200)] + ["TIMELINE//END"]
    _write_ontology(tmp_path, _codes(leaves))
    universe = build_query_universe(leaves, ontology_dir=tmp_path)
    assert set(leaves) <= set(universe), "leaves dropped"
    assert len(universe) == len(set(universe)), "a node appears more than once"


def test_query_universe_excludes_tautological_timeline_ancestors(tmp_path):
    """'did any TIMELINE event occur' is free positives and teaches nothing."""
    leaves = ["TIMELINE//END", "TIMELINE//DELTA//1d", "LAB//X//Y"]
    _write_ontology(tmp_path, _codes(leaves))
    universe = build_query_universe(leaves, ontology_dir=tmp_path)
    assert "TIMELINE" not in set(universe)
    assert "TIMELINE//DELTA" not in set(universe)


# ── 4. model integration ────────────────────────────────────────────────


def test_ontology_dir_is_recorded_in_hparams(tmp_path):
    """Checkpoints must round-trip it: the wrapper changes state-dict keys."""
    model, onto, _ = ontology_model(tmp_path)
    assert model.hparams["ontology_dir"] == str(onto)
    assert tiny_model().hparams["ontology_dir"] is None


def test_wrapper_is_installed_on_the_shared_table(tmp_path):
    """It must be the backbone's input embedding, which is what every code slot also reads."""
    model, _, _ = ontology_model(tmp_path)
    assert isinstance(model.HF_model.get_input_embeddings(), OntologyEmbedding)


def test_ontology_model_runs_and_trains(tmp_path):
    model, _, v_ext = ontology_model(tmp_path)
    model.train()
    loss, out = model(make_batch())
    assert loss.isfinite() and out.logits.shape == (2, 3, v_ext)
    loss.backward()
    raw = model.HF_model.get_input_embeddings().tok
    assert raw.weight.grad is not None and torch.isfinite(raw.weight.grad).all()


def test_cache_is_cleared_between_forwards(tmp_path):
    """Without the pre-hook, a cached product would be reused across backward passes."""
    model, _, _ = ontology_model(tmp_path)
    model.train()
    for _ in range(2):
        loss, _ = model(make_batch())
        loss.backward()  # would raise "backward through the graph a second time" if stale
