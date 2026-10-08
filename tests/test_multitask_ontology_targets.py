"""Derived-at-load ancestor targets for the multitask model: the closure index and its derivation.

The multitask sampler's ``.labels.npy`` sidecars are leaf-only.  Under the window rule an ancestor's
target bit is exactly the OR of its descendant leaves' bits, so the model derives the ancestor block
per batch from the leaf block and the ``event_to_query_nodes.parquet`` closure.  What is pinned here:

1. :func:`derive_ancestor_targets` is the brute-force OR over the closure, on random bits, with a
   dual-role node, a multi-parent leaf, a leaf without ancestors and the PAD column;
2. deriving from the **multitask labeler's** leaf bits reproduces the independent
   ``tests/ontology_suite`` oracle for every ancestor node on the hand-checked golden cohort - the
   proof that derivation equals the scalar path's event-explosion semantics;
3. :func:`load_closure_index` rejects an ontology built from a different cohort - by width, and, given
   the cohort's ``code -> index`` rows or their fingerprint, a *same-width* foreign or permuted one
   that the width checks alone would accept (:func:`check_ontology_cohort`);
4. :func:`closure_fingerprint` is the closure half of the evaluation grid's provenance digest, and
   :func:`ontology_vocab_fingerprint` is the multitask manifest's ``vocab_fingerprint``;
5. (PR B) a QuerySeq evaluation grid labeled *with* an ontology and multitask training labels
   generated *without* one agree on the same ancestor query at the same contexts once derived.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
import torch
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from every_query.data.ontology import (
    EMBEDDING_MIX_FILE,
    EVENT_TO_QUERY_NODES_FILE,
    ONTOLOGY_VOCAB_FILE,
    ClosureIndex,
    build_event_to_query_nodes,
    build_ontology,
    check_ontology_cohort,
    closure_fingerprint,
    derive_ancestor_targets,
    extended_vocab_size,
    load_closure_index,
    load_event_to_query_nodes,
    load_nodes,
    ontology_vocab_fingerprint,
)
from every_query.generate_tasks import sample_evaluation_query_sequences as eval_seq
from every_query.generate_tasks import sample_multitask_sequences as sms
from every_query.generate_tasks.sample_multitask_sequences import (
    LABELS_SUFFIX,
    TargetVocabulary,
    build_target_vocabulary,
    label_multitask_index,
)
from every_query.utils.digest import vocab_fingerprint
from tests.designed_specs import entry
from tests.multitask.conftest import CODES, base_cfg, make_events, make_index, write_cohort
from tests.ontology_suite.golden import DECLARED_PARENTS, EVENTS, LEAVES, ONTOLOGY, T0, TRUTH_TABLE
from tests.ontology_suite.oracle import label_duration, label_event_bounded
from tests.ontology_suite.production import codes_frame, events_to_frame

if TYPE_CHECKING:
    from datetime import datetime


def _write_ontology(root: Path, codes: list[str], parents: dict[str, list[str]] | None = None) -> Path:
    """The three ``EQ_build_ontology`` artifacts for ``codes`` at leaf ids ``1..len(codes)``."""
    nodes, mix = build_ontology(codes_frame(codes, parents or {}))
    root.mkdir(parents=True, exist_ok=True)
    nodes.write_parquet(root / ONTOLOGY_VOCAB_FILE)
    mix.write_parquet(root / EMBEDDING_MIX_FILE)
    build_event_to_query_nodes(nodes, mix).write_parquet(root / EVENT_TO_QUERY_NODES_FILE)
    return root


def _ids(onto: Path) -> dict[str, int]:
    nodes = load_nodes(onto)
    return dict(zip(nodes["node_name"].to_list(), nodes["token_id"].to_list(), strict=True))


def _ancestor_names(onto: Path) -> list[str]:
    return load_nodes(onto).filter(~pl.col("is_observed_code"))["node_name"].to_list()


# --- 1: brute force ------------------------------------------------------------------------------

# A dual-role name (``X`` is a leaf and the prefix of ``X//1`` / ``X//2``), a multi-parent leaf (``M//L``
# under ``M`` by prefix and ``G//P`` by ``parent_codes``), a leaf with no ancestor at all (``LONE``) and
# an unrelated family.
_BF_LEAVES = ["X", "X//1", "X//2", "M//L", "LONE", "U//A", "U//B", "U//C//D"]
_BF_PARENTS = {"M//L": ["G//P"]}


@pytest.mark.parametrize("seed", range(4))
def test_derive_ancestor_targets_matches_brute_force_or(tmp_path: Path, seed: int):
    onto = _write_ontology(tmp_path / "onto", _BF_LEAVES, _BF_PARENTS)
    v = len(_BF_LEAVES) + 1
    closure = load_closure_index(onto, v)
    ids = _ids(onto)
    assert {"X//ANY", "M", "G", "G//P", "U", "U//C"} <= set(_ancestor_names(onto))
    assert closure.v_ext == extended_vocab_size(onto) == v + len(_ancestor_names(onto))

    # The closure as the python dict the derivation must reproduce: ancestor -> its leaf ids.
    pairs = load_event_to_query_nodes(onto).filter(pl.col("event_code") != pl.col("query_node"))
    under: dict[int, set[int]] = {}
    for leaf, ancestor in zip(pairs["event_code"].to_list(), pairs["query_node"].to_list(), strict=True):
        under.setdefault(ids[ancestor], set()).add(ids[leaf])
    assert under[ids["X//ANY"]] == {ids["X"], ids["X//1"], ids["X//2"]}, "dual-role subtree includes the leaf"
    assert under[ids["G//P"]] == {ids["M//L"]} and under[ids["M"]] == {ids["M//L"]}, "multi-parent leaf"
    assert not any(ids["LONE"] in leaves for leaves in under.values()), "LONE has no ancestor"
    assert set(under) == set(range(v, closure.v_ext)), "every ancestor column has at least one leaf"

    rng = np.random.default_rng(seed)
    leaf = torch.from_numpy(rng.random((3, 4, v)) < 0.3)
    leaf[..., 0] = torch.from_numpy(rng.random((3, 4)) < 0.5)  # PAD bits: passed through, never read
    got = derive_ancestor_targets(leaf, closure)
    assert got.shape == (3, 4, closure.v_ext) and got.dtype == torch.bool
    assert torch.equal(got[..., :v], leaf), "the leaf block is untouched, PAD column included"
    expected = torch.zeros(3, 4, closure.v_ext - v, dtype=torch.bool)
    for ancestor, leaves in under.items():
        for l_id in leaves:
            expected[..., ancestor - v] |= leaf[..., l_id]
    assert torch.equal(got[..., v:], expected)
    # The PAD column never feeds an ancestor.
    flipped = leaf.clone()
    flipped[..., 0] = ~flipped[..., 0]
    assert torch.equal(derive_ancestor_targets(flipped, closure)[..., v:], expected)
    # Any leading shape, and a uint8 leaf block, work the same.
    assert torch.equal(derive_ancestor_targets(leaf.to(torch.uint8), closure), got)
    assert torch.equal(derive_ancestor_targets(leaf[0], closure), got[0])


def test_derive_without_ancestors_or_pairs_is_the_identity():
    leaf = torch.tensor([[[True, False, True]]])
    no_ancestors = ClosureIndex(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), 3, 3)
    assert torch.equal(derive_ancestor_targets(leaf, no_ancestors), leaf)
    # An ancestor column with no leaf under it (possible for a node above other ancestors only) stays false.
    orphan = ClosureIndex(torch.tensor([1]), torch.tensor([3]), 3, 5)
    assert derive_ancestor_targets(leaf, orphan).int().tolist() == [[[1, 0, 1, 0, 0]]]
    leaf_under = ClosureIndex(torch.tensor([2]), torch.tensor([4]), 3, 5)
    assert derive_ancestor_targets(leaf, leaf_under).int().tolist() == [[[1, 0, 1, 0, 1]]]


# --- 2: the oracle ---------------------------------------------------------------------------------

# Windows over the golden cohort at ``T0``: the truth table's horizons and both of its boundary codes.
_WINDOWS = [(7.0, None), (8.0, None), (30.0, None), (40.0, None), (-1.0, "DISCHARGE"), (-1.0, "ADMISSION")]
_SUBJECTS = [1, 2, 3, 4, 5]


def _oracle(subject: int, node: str, window: tuple[float, str | None]) -> bool:
    duration, bound = window
    if bound is None:
        return label_duration(EVENTS, subject, ONTOLOGY, T0, node, duration)
    return label_event_bounded(EVENTS, subject, ONTOLOGY, T0, node, bound)


def test_derive_ancestor_targets_agrees_with_the_oracle(tmp_path: Path):
    """Multitask leaf bits (labeled with **no** ontology) + derivation == the independent oracle's answer
    for every node - leaf and ancestor - at every window of every golden subject, and the truth table's
    ``ontology_required`` cases come out as hand-derived."""
    onto = _write_ontology(tmp_path / "onto", LEAVES, DECLARED_PARENTS)
    vocab = TargetVocabulary.from_pairs(LEAVES, list(range(1, len(LEAVES) + 1)))
    closure = load_closure_index(onto, vocab.size)
    ids = _ids(onto)
    ancestors = _ancestor_names(onto)
    assert set(ancestors) == ONTOLOGY.ancestors, "the production DAG and the oracle DAG name the same nodes"

    k = len(_WINDOWS)
    index = make_index([(s, T0) for s in _SUBJECTS], [_WINDOWS] * len(_SUBJECTS), fill_condition="LAB//GLU")
    meta, packed, _ = label_multitask_index(index, events_to_frame(EVENTS), vocab, k, ontology_dir=None)
    dense = np.unpackbits(packed, axis=-1, count=vocab.size, bitorder="little").astype(bool)
    derived = derive_ancestor_targets(torch.from_numpy(dense), closure).numpy()
    assert derived.shape == (len(_SUBJECTS), k, closure.v_ext)

    checked_ancestors = 0
    for row, subject in enumerate(meta["subject_id"].to_list()):
        for j, window in enumerate(_WINDOWS):
            for node in [*LEAVES, *ancestors]:
                assert bool(derived[row, j, ids[node]]) == _oracle(subject, node, window), (
                    f"subject {subject}, window {window}, node {node!r}"
                )
                checked_ancestors += node in ancestors
    assert checked_ancestors >= len(_SUBJECTS) * k * 10

    # The hand-derived truth table, for every case whose window is one of ours.
    row_of = {s: i for i, s in enumerate(meta["subject_id"].to_list())}
    pinned = 0
    for case in TRUTH_TABLE:
        window = (case.duration_days if case.bound_event is None else -1.0, case.bound_event)
        if window not in _WINDOWS:
            continue
        got = bool(derived[row_of[case.subject_id], _WINDOWS.index(window), ids[case.query]])
        assert got == case.expected, f"{case.case_id}: {case.why}"
        pinned += case.ontology_required
    assert pinned >= 15, f"only {pinned} ontology-required cases were reachable"

    # Leaf bits are byte-identical whether or not the ancestor block is derived from them.
    assert np.array_equal(derived[..., : vocab.size], dense)


# --- 3 / 4: the index rejects a foreign ontology; the fingerprint is the closure half --------------


def test_closure_index_rejects_a_foreign_ontology(tmp_path: Path):
    onto = _write_ontology(tmp_path / "onto", _BF_LEAVES, _BF_PARENTS)
    v = len(_BF_LEAVES) + 1
    assert load_closure_index(onto).base_vocab_size == v
    # A cohort with more codes than the ontology knows, or fewer.
    too_wide = rf"leaves span \[\.\., {v}\) but the cohort's vocab_size is {v + 3}"
    with pytest.raises(ValueError, match=too_wide):
        load_closure_index(onto, v + 3)
    with pytest.raises(ValueError, match=r"different codes\.parquet"):
        load_closure_index(onto, v - 1)
    # Nodes renumbered so an ancestor sits below a leaf (the leaf ids stay 1..8, ancestors moved to 0-based).
    nodes = load_nodes(onto)
    broken = tmp_path / "broken"
    broken.mkdir()
    lowest_ancestor = int(nodes.filter(~pl.col("is_observed_code"))["token_id"].min())
    nodes.with_columns(
        pl.when(pl.col("token_id") == lowest_ancestor).then(1).otherwise(pl.col("token_id")).alias("token_id")
    ).write_parquet(broken / ONTOLOGY_VOCAB_FILE)
    load_event_to_query_nodes(onto).write_parquet(broken / EVENT_TO_QUERY_NODES_FILE)
    with pytest.raises(ValueError, match="below the cohort's vocab_size"):
        load_closure_index(broken, v)
    # A closure naming a node the vocabulary does not carry: the artifacts were not written together.
    torn = tmp_path / "torn"
    torn.mkdir()
    nodes.write_parquet(torn / ONTOLOGY_VOCAB_FILE)
    pl.concat(
        [load_event_to_query_nodes(onto), pl.DataFrame({"event_code": ["X"], "query_node": ["NOT//A//NODE"]})]
    ).write_parquet(torn / EVENT_TO_QUERY_NODES_FILE)
    with pytest.raises(ValueError, match="absent from"):
        load_closure_index(torn, v)


def test_closure_index_rejects_a_same_width_foreign_or_permuted_ontology(tmp_path: Path):
    """Regression for the PR #32 review: the width checks only compare the highest observed index with ``V``.

    Two unrelated ``codes.parquet`` files of the same size, or this cohort's codes at permuted indices, pass
    them with the same ``V`` (and, for the permutation, the same ``V_ext``) and would silently pair leaf
    target columns with the wrong closure rows.  Given the cohort's ``code -> index`` rows or their
    fingerprint the loader refuses both, naming what differs; the genuine ontology passes every form and
    reads identically.
    """
    onto = _write_ontology(tmp_path / "onto", _BF_LEAVES, _BF_PARENTS)
    cohort = {c: i + 1 for i, c in enumerate(_BF_LEAVES)}
    v = len(_BF_LEAVES) + 1
    fingerprint = vocab_fingerprint(cohort)
    assert ontology_vocab_fingerprint(onto) == fingerprint

    bare = load_closure_index(onto, v)
    for identity in (
        {"code_to_index": cohort},
        {"vocab_fingerprint": fingerprint},
        {"code_to_index": cohort, "vocab_fingerprint": fingerprint},
    ):
        got = load_closure_index(onto, v, **identity)
        assert torch.equal(got.leaf_ids, bare.leaf_ids) and torch.equal(got.ancestor_ids, bare.ancestor_ids)
        assert (got.base_vocab_size, got.v_ext) == (bare.base_vocab_size, bare.v_ext)

    # Same codes, two leaf indices swapped: same V, same V_ext, internally consistent artifacts.
    swapped = list(_BF_LEAVES)
    i, j = swapped.index("X//1"), swapped.index("M//L")
    swapped[i], swapped[j] = swapped[j], swapped[i]
    permuted = _write_ontology(tmp_path / "permuted", swapped, _BF_PARENTS)
    width_only = load_closure_index(permuted, v)
    assert (width_only.base_vocab_size, width_only.v_ext) == (v, bare.v_ext), (
        "widths cannot see a permutation"
    )
    renumbered = (
        r"different codes\.parquet.*2 code\(s\) sit at a different index "
        r"\('M//L': ontology \d+ vs cohort \d+, 'X//1': ontology \d+ vs cohort \d+\)"
    )
    with pytest.raises(ValueError, match=renumbered):
        load_closure_index(permuted, v, code_to_index=cohort)
    with pytest.raises(ValueError, match=r"different codes\.parquet.*observed nodes digest to"):
        load_closure_index(permuted, v, vocab_fingerprint=fingerprint)
    with pytest.raises(ValueError, match=renumbered):
        check_ontology_cohort(permuted, code_to_index=cohort, vocab_fingerprint=fingerprint)
    # The permutation is not harmless: the cohort's ``M//L`` column is the permuted closure's ``X//1``, so
    # ``M//L``'s ancestors go dark and ``X``'s subtree node lights up instead.
    ids = _ids(onto)
    ancestor_ids = {n: i for n, i in ids.items() if i >= v}
    assert ancestor_ids == {n: i for n, i in _ids(permuted).items() if i >= v}, "ancestors are minted by name"
    leaf = torch.zeros(1, 1, v, dtype=torch.bool)
    leaf[..., cohort["M//L"]] = True
    right = derive_ancestor_targets(leaf, bare)[0, 0]
    wrong = derive_ancestor_targets(leaf, width_only)[0, 0]
    assert right[ids["M"]] and right[ids["G//P"]] and not right[ids["X//ANY"]]
    assert not wrong[ids["M"]] and not wrong[ids["G//P"]] and wrong[ids["X//ANY"]]

    # Unrelated codes, the same number of them: same V again; both directions of the row diff are reported.
    n = len(_BF_LEAVES)
    foreign = _write_ontology(tmp_path / "foreign", [f"F//{k}" for k in range(n)])
    assert load_closure_index(foreign, v).base_vocab_size == v
    both_ways = (
        rf"{n} cohort code\(s\) are not ontology leaves \(e\.g\. \['LONE'.*"
        rf"{n} ontology leaf name\(s\) are not cohort codes \(e\.g\. \['F//0'"
    )
    with pytest.raises(ValueError, match=both_ways):
        load_closure_index(foreign, v, code_to_index=cohort)
    with pytest.raises(ValueError, match=r"different codes\.parquet"):
        check_ontology_cohort(foreign, vocab_fingerprint=fingerprint)
    # One code renamed at the same index: one missing, one extra, nothing renumbered.
    renamed = _write_ontology(tmp_path / "renamed", [*_BF_LEAVES[:-1], "U//C//E"], _BF_PARENTS)
    one_each = (
        r"\(1 cohort code\(s\) are not ontology leaves \(e\.g\. \['U//C//D'\]\); "
        r"1 ontology leaf name\(s\) are not cohort codes \(e\.g\. \['U//C//E'\]\)\)\."
    )
    with pytest.raises(ValueError, match=one_each):
        check_ontology_cohort(renamed, code_to_index=cohort)
    # The check needs to be told which cohort it is comparing against.
    with pytest.raises(ValueError, match="needs the cohort's code_to_index or its vocab_fingerprint"):
        check_ontology_cohort(onto)


def test_ontology_vocab_fingerprint_is_the_multitask_manifest_fingerprint(tmp_path: Path):
    """The ontology's observed nodes digest exactly as the multitask sampler digests the cohort's
    ``codes.parquet`` (``TargetVocabulary.fingerprint``, the manifest's ``vocab_fingerprint``), so a
    checkpoint's ``cohort_vocab_fingerprint``, a manifest and an ontology all compare like with like."""
    onto = _write_ontology(tmp_path / "onto", _BF_LEAVES, _BF_PARENTS)
    vocab = TargetVocabulary.from_pairs(list(reversed(_BF_LEAVES)), list(range(len(_BF_LEAVES), 0, -1)))
    assert ontology_vocab_fingerprint(onto) == vocab.fingerprint == vocab_fingerprint(vocab.code_to_index())
    # Ancestor nodes are not part of it (they are derived), and a same-width permutation moves it.
    assert vocab_fingerprint(_ids(onto)) != vocab.fingerprint
    swapped = dict(vocab.code_to_index())
    swapped["X//1"], swapped["M//L"] = swapped["M//L"], swapped["X//1"]
    assert vocab_fingerprint(swapped) != vocab.fingerprint


def test_closure_fingerprint_is_the_closure_half_of_the_grid_provenance(tmp_path: Path):
    onto = _write_ontology(tmp_path / "onto", _BF_LEAVES, _BF_PARENTS)
    stale = _write_ontology(tmp_path / "stale", _BF_LEAVES[:-1], _BF_PARENTS)
    recorded = eval_seq._ontology_fingerprint(onto, list(_BF_LEAVES))
    closure_half, universe_half = eval_seq.split_ontology_fingerprint(recorded)
    assert closure_half == closure_fingerprint(onto) and universe_half
    no_ontology = eval_seq._ontology_fingerprint(None, _BF_LEAVES)
    assert eval_seq.split_ontology_fingerprint(no_ontology) == (None, None)
    assert closure_fingerprint(stale) != closure_fingerprint(onto)
    # The universe does not enter the closure digest, and re-writing the same closure does not move it.
    assert eval_seq.split_ontology_fingerprint(eval_seq._ontology_fingerprint(onto, ["X"]))[0] == closure_half
    rewritten = tmp_path / "rewritten"
    rewritten.mkdir()
    load_nodes(onto).write_parquet(rewritten / ONTOLOGY_VOCAB_FILE)
    load_event_to_query_nodes(onto).reverse().write_parquet(rewritten / EVENT_TO_QUERY_NODES_FILE)
    assert closure_fingerprint(rewritten) == closure_fingerprint(onto)


# --- 5 (PR B): the multitask training path and the QuerySeq grid agree on an ancestor query --------

_HORIZON = 30.0
_ANCESTOR = "C"  # every synthetic leaf ``C//i`` sits under it; ``TIMELINE//END`` does not


def _run_eval(**overrides) -> None:
    # The production default (40 positives) would drop every task in these tiny fixtures.
    overrides = {"min_task_positives": "null", **overrides}
    with initialize_config_dir(config_dir=eval_seq.CONFIGS, version_base=None):
        cfg = compose(
            config_name="sample_evaluation_query_sequences_config",
            overrides=[f"{k}={v}" for k, v in overrides.items()],
        )
    eval_seq.main.__wrapped__(cfg)


def test_multitask_eval_and_scalar_grid_agree_on_the_same_ancestor_query(tmp_path: Path):
    """The multitask analog of ``test_eval_and_training_paths_agree_on_the_same_ancestor_query``.

    Multitask training labels are generated with **no** ontology (every window ``(t, t + 30d)``); a
    QuerySeq grid is generated **with** the ontology at exactly the sampled contexts, asking the pure
    ancestor ``C``.  The derived training bit for ``C`` must equal the grid's label at every context.
    """
    rng = np.random.default_rng(11)
    shards = {shard: make_events(rng, [int(shard) * 100 + s for s in range(1, 6)]) for shard in ("0", "1")}
    cohort = write_cohort(tmp_path / "cohort", shards)
    onto = _write_ontology(tmp_path / "onto", CODES)
    assert _ANCESTOR in _ancestor_names(onto)
    vocab = build_target_vocabulary(cohort)
    closure = load_closure_index(onto, vocab.size)
    ids = _ids(onto)

    train_out = tmp_path / "multitask"
    sms.run(
        OmegaConf.create(
            base_cfg(
                cohort,
                train_out,
                num_training_examples=40,
                num_bounds=2,
                duration_min=_HORIZON,
                duration_max=_HORIZON,
                eventbound_fraction=0.0,
                eventstart_fraction=0.0,
                prediction_time_start_fraction=1.0,
                min_prediction_times_per_subject=3,
                max_workers=1,
            )
        )
    )
    derived: dict[tuple[int, datetime], bool] = {}
    for fp in sorted((train_out / "train").glob("*.parquet")):
        meta = pl.read_parquet(fp)
        packed = np.load(train_out / "train" / f"{fp.stem}{LABELS_SUFFIX}")
        assert (meta["durations"].explode() == _HORIZON).all()
        assert meta["bound_events"].explode().is_null().all()
        assert (meta["start_durations"].explode() == 0.0).all()
        dense = np.unpackbits(packed, axis=-1, count=vocab.size, bitorder="little").astype(bool)
        wide = derive_ancestor_targets(torch.from_numpy(dense), closure).numpy()
        contexts = zip(meta["subject_id"].to_list(), meta["prediction_time"].to_list(), strict=True)
        for row, (sid, pt) in enumerate(contexts):
            # Both windows are the same window, so both derived bits are the same bit.
            assert wide[row, 0, ids[_ANCESTOR]] == wide[row, 1, ids[_ANCESTOR]]
            derived[(int(sid), pt)] = bool(wide[row, 0, ids[_ANCESTOR]])
    assert derived and any(derived.values()) and not all(derived.values()), "need both labels"

    contexts = tmp_path / "contexts.parquet"
    pl.DataFrame(
        {"subject_id": [k[0] for k in derived], "prediction_time": [k[1] for k in derived]}
    ).with_columns(pl.col("prediction_time").cast(pl.Datetime("us"))).write_parquet(contexts)
    leaf_codes = tmp_path / "codes.yaml"
    leaf_codes.write_text(yaml.safe_dump(list(CODES)))
    specs = tmp_path / "spec.yaml"
    specs.write_text(yaml.safe_dump({"any_c": [entry(_ANCESTOR, _HORIZON)]}))
    eval_out = tmp_path / "grid"
    _run_eval(
        data_dir=cohort,
        out_dir=eval_out,
        query_codes=leaf_codes,
        contexts_path=contexts,
        sequences_path=specs,
        split="train",
        ontology_dir=onto,
    )
    grid = pl.concat([pl.read_parquet(fp) for fp in sorted((eval_out / "eval" / "train").glob("*.parquet"))])
    labeled = {
        (int(r["subject_id"]), r["prediction_time"]): bool(r["answers"][0])
        for r in grid.iter_rows(named=True)
        if r["queries"] == [_ANCESTOR]
    }
    assert set(labeled) == set(derived), "the grid must cover exactly the sampled contexts"
    assert labeled == derived, "the two paths disagree about the same ancestor query at the same contexts"
