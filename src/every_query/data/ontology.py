"""Code ontology: a DAG over the cohort vocabulary, and the embedding mix it induces.

MEDS codes are already hierarchical in their names — ``LAB//220645//mEq/L//value_[135,136)``
sits under ``LAB//220645``, which sits under ``LAB`` — and a cohort's ``codes.parquet`` may also
carry an explicit ``parent_codes`` column.  This module turns that structure into two things:

1. **An extended vocabulary.**  Every ``//``-prefix that is not itself a code becomes an
   *ancestor node* with its own index, appended above the highest leaf index so leaf indices are
   preserved exactly.  That makes an ancestor directly *addressable as a query*: you can ask
   about a whole drug class rather than one specific product.
2. **A mix matrix** ``A`` (row-normalised, sparse, ``V_ext x V_ext``).  A node's embedding
   becomes the weighted average of its own raw row and its ancestors', with weight
   ``decay ** distance``.  Installed as the encoder's input-embedding module, this ties a rare
   leaf's representation to its better-estimated parents — the actual hypothesis under test.

Three artifacts are written by ``EQ_build_ontology`` into one directory:

- ``ontology_vocab.parquet``  ``(node_name, token_id, is_observed_code)`` — the extended vocabulary;
  ``V_ext``.  ``is_observed_code`` means "can appear directly in the event stream".
- ``embedding_mix.parquet``  ``(target_token_id, component_token_id, unnormalized_weight)`` — COO
  entries of ``A`` before row normalisation.
- ``event_to_query_nodes.parquet``  ``(event_code, query_node)`` — every observed code paired with
  itself and each query node it satisfies, used to explode an event stream so ancestor queries can
  be labelled by ordinary occurrence.

The upstream experiment's own verdict is worth stating plainly, because it bears on whether to
turn this on: the **embedding** effect on leaf tasks did not replicate (a seed-2 run reversed it,
and the unbundling suite scored it null), while the **DAG structure** was worth about +0.039
AUROC on ancestor queries.  The value found so far is in being able to *ask* about an ancestor,
not in the mixing improving ordinary leaf queries.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import torch

from every_query.utils.digest import frame_digest, vocab_fingerprint

logger = logging.getLogger(__name__)

SEP = "//"

ONTOLOGY_VOCAB_FILE = "ontology_vocab.parquet"
EMBEDDING_MIX_FILE = "embedding_mix.parquet"
EVENT_TO_QUERY_NODES_FILE = "event_to_query_nodes.parquet"

#: The PAD index.  Never a node: no leaf sits at it and no ancestor can be appended below ``V``.
PAD_INDEX = 0


def string_ancestors(code: str) -> list[str]:
    """All proper ``//``-prefix ancestors of a code, nearest first.

    A single ``/`` is not a separator, so ``ICD10CM/A04.72`` has no ancestors — the separator is
    the two-character ``//`` that MEDS uses between hierarchy levels.

    Examples:
        >>> string_ancestors("LAB//220645//mEq/L//value_[135.0,136.0)")
        ['LAB//220645//mEq/L', 'LAB//220645', 'LAB']
        >>> string_ancestors("BMI")
        []
        >>> string_ancestors("ICD10CM/A04.72")
        []
    """
    parts = code.split(SEP)
    return [SEP.join(parts[:i]) for i in range(len(parts) - 1, 0, -1)]


def build_ontology(
    codes_df: pl.DataFrame, decay: float = 0.5, subtree_suffix: str | None = "ANY"
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build ``(nodes_df, mix_df)`` from a ``codes.parquet``-shaped frame.

    Args:
        codes_df: Columns ``code``, ``code/vocab_index``, optionally ``parent_codes``.
        decay: Per-level weight decay.  An ancestor ``d`` levels up contributes ``decay ** d``
            before row normalisation, so ``0.0`` disables mixing entirely and ``1.0`` weights
            every ancestor as much as the node itself.
        subtree_suffix: Name suffix for the *subtree node* minted beside every name that is both
            a real code and someone's ancestor (see the dual-role section in the body).  ``None``
            skips minting them, leaving such names purely exact and their subtree meaning
            unaskable.  Changing this changes ``V_ext``, so an ontology and the encoder sized
            from it must be built with the same value.

    Returns:
        ``nodes_df`` ``(node_name, token_id, is_observed_code)`` — leaves keep their original indices,
        ancestors get fresh ones from ``max_leaf_index + 1`` upward, sorted by name — and
        ``mix_df`` ``(target_token_id, component_token_id, unnormalized_weight)``, unnormalised, with a
        weight-1 self-loop for every node.

    Examples:
        >>> df = pl.DataFrame({
        ...     "code": ["A//B//C", "A//B//D", "E"],
        ...     "code/vocab_index": [1, 2, 3],
        ...     "parent_codes": [["G/x"], None, None],
        ... })
        >>> nodes, mix = build_ontology(df)
        >>> sorted(nodes.filter(~pl.col("is_observed_code"))["node_name"].to_list())
        ['A', 'A//B', 'G/x']

        Leaf indices are untouched, so an ontology can be dropped onto an existing cohort:

        >>> nodes.filter(pl.col("is_observed_code")).sort("token_id")["token_id"].to_list()
        [1, 2, 3]

        ``A//B//C`` mixes itself (1.0), ``A//B`` (0.5), ``A`` (0.25) and its declared parent
        ``G/x`` (0.5, a grouper edge counts as distance 1):

        >>> m = {r["component_token_id"]: r["unnormalized_weight"]
        ...      for r in mix.filter(pl.col("target_token_id") == 1).iter_rows(named=True)}
        >>> idx = dict(zip(nodes["node_name"], nodes["token_id"]))
        >>> m[idx["A//B//C"]], m[idx["A//B"]], m[idx["A"]], m[idx["G/x"]]
        (1.0, 0.5, 0.25, 0.5)

        Every node reachable in the index has a mix row, so no node can end up with an
        all-zero embedding:

        >>> set(nodes["token_id"]) == set(mix["target_token_id"].unique())
        True

        A **dual-role** name -- both a real code and the prefix of another code -- keeps its
        exact meaning and gains a separate subtree node, so both questions stay askable:

        >>> df = pl.DataFrame({
        ...     "code": ["INF//220949", "INF//220949//value_lo", "INF//220949//value_hi"],
        ...     "code/vocab_index": [1, 2, 3],
        ... })
        >>> nodes, mix = build_ontology(df)
        >>> sorted(nodes.filter(~pl.col("is_observed_code"))["node_name"].to_list())
        ['INF', 'INF//220949//ANY']

        ``INF//220949`` stays a leaf (it names 365k real events in the real cohort), while
        ``INF//220949//ANY`` is the node that means "the drug, valued or not":

        >>> closure = build_event_to_query_nodes(nodes, mix)
        >>> sorted(closure.filter(pl.col("query_node") == "INF//220949//ANY")["event_code"].to_list())
        ['INF//220949', 'INF//220949//value_hi', 'INF//220949//value_lo']

        Nothing rolls up to the leaf itself except the leaf, which is what keeps its ordinary
        query exact:

        >>> sorted(closure.filter(pl.col("query_node") == "INF//220949")["event_code"].to_list())
        ['INF//220949']

        With ``subtree_suffix=None`` the extra node is not minted at all:

        >>> nodes2, _ = build_ontology(df, subtree_suffix=None)
        >>> sorted(nodes2.filter(~pl.col("is_observed_code"))["node_name"].to_list())
        ['INF']
    """
    if not 0.0 <= decay <= 1.0:
        raise ValueError(f"decay must be in [0, 1], got {decay}")

    has_parents = "parent_codes" in codes_df.columns
    leaves = codes_df.select(pl.col("code"), pl.col("code/vocab_index").cast(pl.Int64).alias("token_id"))
    leaf_names = set(leaves["code"].to_list())

    # Declared `parent_codes` edges, keyed by the code that declares them.
    declared: dict[str, list[str]] = {}
    if has_parents:
        for row in codes_df.iter_rows(named=True):
            code = row["code"]
            pcs = row.get("parent_codes")
            if pcs:
                kept = [pc for pc in pcs if pc and pc != code]
                if kept:
                    declared[code] = kept

    def direct_parents(name: str) -> list[str]:
        """One hop up: the immediate ``//``-prefix parent, plus any declared groupers.

        Both edge kinds cost exactly one hop, which is what reproduces the historical distance
        scale — ``A//B//C`` reaches ``A//B`` at 1 and ``A`` at 2, and a declared grouper sits at
        1 with its own prefixes at 2, 3, ...  Since ``decay ** dist`` sets the mix weights,
        changing this scale would silently reweight every embedding.
        """
        out: list[str] = []
        if SEP in name:
            out.append(name.rsplit(SEP, 1)[0])
        out.extend(declared.get(name, ()))
        return out

    def ancestors_with_distance(start: str) -> dict[str, int]:
        """Every strict ancestor of ``start`` and its minimum hop count.

        A breadth-first walk over the *union* of prefix and declared edges.  Walking both kinds
        with one traversal is the whole point: the previous implementation followed a declared
        edge exactly one hop and then only string prefixes, so a chain ``X -> P -> GRP//G``
        stopped at ``P`` and ``GRP//G`` never became an ancestor of ``X`` — precisely the
        multi-level DAG that ``parent_codes`` exists to express.

        Note the walk passes *through* a leaf when a leaf is someone's declared parent.  That is
        deliberate and is not in tension with :func:`build_event_to_query_nodes` refusing to make that leaf
        addressable as a subtree query: traversal and addressability are different questions.

        ``parent_codes`` is caller-supplied data, so cycles are possible.  ``start`` is never
        admitted to its own ancestor set, and a name already seen at an equal-or-shorter
        distance is not re-expanded, which bounds the walk.
        """
        dist: dict[str, int] = {}
        frontier = [(p, 1) for p in direct_parents(start)]
        while frontier:
            name, d = frontier.pop()
            if name == start:
                continue
            if name in dist and dist[name] <= d:
                continue
            dist[name] = d
            frontier.extend((p, d + 1) for p in direct_parents(name))
        return dist

    node_ancestors: dict[str, dict[str, int]] = {}
    ancestor_names: set[str] = set()
    for code in leaves["code"].to_list():
        amap = ancestors_with_distance(code)
        node_ancestors[code] = amap
        ancestor_names.update(a for a in amap if a not in leaf_names)

    # Every *indexed* node also needs its own mix row, or it embeds to the zero vector.  The walk
    # above is already transitive, so one pass over the ancestor set normally suffices; the loop
    # is kept so "every indexed node has a mix row" holds by construction rather than by luck.
    pending = sorted(ancestor_names)
    while pending:
        nxt: list[str] = []
        for anc in pending:
            if anc in node_ancestors:
                continue
            amap = ancestors_with_distance(anc)
            node_ancestors[anc] = amap
            nxt.extend(a for a in amap if a not in leaf_names and a not in ancestor_names)
        ancestor_names.update(nxt)
        pending = sorted(set(nxt))

    # ---- Dual-role names get a distinct subtree node ----------------------------------------
    #
    # A name can be both a real code and something else's ancestor: `INFUSION_START//220949` is
    # 365,723 unvalued infusion events *and* the prefix of ten `//value_[lo,hi)` variants.  One
    # string cannot mean both "exactly this code" and "this code or any descendant" without one
    # of the two meanings becoming unaskable, and the labeler has to pick one.
    #
    # So mint a second name.  The leaf keeps its exact meaning; a fresh ancestor node
    # `<name>//<suffix>` means the subtree, and every reference to the leaf *as an ancestor* is
    # rewritten to it.  Both rungs of the ladder stay addressable:
    #
    #     INFUSION_START                        all 635 infusion-start codes
    #     INFUSION_START//220949//ANY           the drug, valued or not
    #     INFUSION_START//220949                the 365,723 unvalued events
    #     INFUSION_START//220949//value_[...]   one rate bin
    #
    # `subtree_suffix=None` disables this and leaves dual-role names purely exact -- the subtree
    # meaning is then simply not expressible, which is the cheaper option when a cohort has no
    # dual-role names to begin with.
    if subtree_suffix:
        # Both shapes of dual role are caught by asking which *leaves* turned up as an ancestor:
        # `//`-prefix collisions and a declared `parent_codes` edge pointing at a real code.
        dual_role = {a for amap in node_ancestors.values() for a in amap if a in leaf_names}
        subtree_name = {leaf: f"{leaf}{SEP}{subtree_suffix}" for leaf in dual_role}

        clashes = sorted(n for n in subtree_name.values() if n in leaf_names)
        if clashes:
            raise ValueError(
                f"{len(clashes)} subtree node name(s) collide with real codes (e.g. {clashes[:3]}). "
                f"Pass a different `subtree_suffix` than {subtree_suffix!r}."
            )

        # Rewrite every ancestor reference to a dual-role leaf into its subtree node.  Distances
        # are preserved: the subtree node sits exactly where the leaf used to sit, so the
        # `decay ** dist` mix weights are unchanged for every descendant.
        def rewrite(amap: dict[str, int]) -> dict[str, int]:
            out: dict[str, int] = {}
            for name, dist in amap.items():
                key = subtree_name.get(name, name)
                out[key] = min(out.get(key, 10**9), dist)
            return out

        node_ancestors = {node: rewrite(amap) for node, amap in node_ancestors.items()}

        for leaf, sub in subtree_name.items():
            # The subtree node generalises the leaf, so it inherits the leaf's own ancestors...
            node_ancestors[sub] = dict(node_ancestors[leaf])
            # ...and sits one hop above the leaf itself, which is a member of its own subtree.
            node_ancestors[leaf] = dict(node_ancestors[leaf]) | {sub: 1}

    ancestor_names = (
        set(node_ancestors) | {a for amap in node_ancestors.values() for a in amap}
    ) - leaf_names

    max_leaf = int(leaves["token_id"].max())
    anc_sorted = sorted(ancestor_names)
    anc_index = {a: max_leaf + 1 + i for i, a in enumerate(anc_sorted)}
    name_to_index = dict(zip(leaves["code"].to_list(), leaves["token_id"].to_list(), strict=True)) | anc_index

    assert set(ancestor_names) <= node_ancestors.keys(), (
        "every indexed ancestor must have a mix row; otherwise it embeds to the zero vector"
    )

    nodes_df = pl.concat(
        [
            leaves.with_columns(pl.lit(True).alias("is_observed_code")).rename({"code": "node_name"}),
            pl.DataFrame(
                {
                    "node_name": anc_sorted,
                    "token_id": [anc_index[a] for a in anc_sorted],
                    "is_observed_code": [False] * len(anc_sorted),
                }
            ),
        ],
        how="vertical_relaxed",
    )

    mix_rows: dict[str, list] = {"target_token_id": [], "component_token_id": [], "unnormalized_weight": []}
    for node, amap in node_ancestors.items():
        ni = name_to_index[node]
        mix_rows["target_token_id"].append(ni)
        mix_rows["component_token_id"].append(ni)
        mix_rows["unnormalized_weight"].append(1.0)
        for anc, dist in amap.items():
            mix_rows["target_token_id"].append(ni)
            mix_rows["component_token_id"].append(name_to_index[anc])
            mix_rows["unnormalized_weight"].append(decay**dist)

    return nodes_df, pl.DataFrame(mix_rows)


def build_event_to_query_nodes(nodes_df: pl.DataFrame, mix_df: pl.DataFrame) -> pl.DataFrame:
    """``(event_code, query_node)`` rows: every leaf paired with itself and each ancestor *node* above it.

    This is what lets an ancestor query be answered by the ordinary occurrence labeler — explode
    the event stream through it and "did any descendant of X occur" becomes "did X occur".

    **Only non-leaf components survive, plus the self-pair.**  A name that is itself a real code
    is never addressable as a subtree query — ``build_query_universe`` adds only non-leaf nodes
    to the universe — so emitting ``(A//B//C -> A//B)`` for a leaf ``A//B`` could not widen
    anything the sampler would ask; it could only corrupt the ordinary leaf query ``A//B``,
    silently changing its meaning from "this exact code occurred" to "this code **or any
    descendant** occurred".  That flipped labels False -> True with no crash and no warning.

    The self-pair must survive the filter: ``(event_code=A//B, query_node=A//B)`` is what makes a leaf query
    answerable at all.  Dropping rows by ``component_token_id == target_token_id`` instead would break
    every leaf query, so the two conditions are deliberately spelled out separately below.

    ``mix_df`` is left alone.  Embedding *sharing* between ``A//B//C`` and ``A//B`` is desirable
    and is not what was broken — only the labeling closure is narrowed here.

    Examples:
        >>> nodes = pl.DataFrame({
        ...     "node_name": ["A//B", "A"], "token_id": [1, 2], "is_observed_code": [True, False]})
        >>> mix = pl.DataFrame({
        ...     "target_token_id": [1, 1, 2], "component_token_id": [1, 2, 2],
        ...     "unnormalized_weight": [1.0, 0.5, 1.0]})
        >>> build_event_to_query_nodes(nodes, mix).sort("query_node")["query_node"].to_list()
        ['A', 'A//B']

        A leaf that prefixes another leaf keeps its exact meaning — ``A//B`` is a real code here,
        so it is not emitted as a node above ``A//B//C``:

        >>> nodes = pl.DataFrame({
        ...     "node_name": ["A//B", "A//B//C", "A"], "token_id": [1, 2, 3],
        ...     "is_observed_code": [True, True, False]})
        >>> mix = pl.DataFrame({
        ...     "target_token_id": [1, 1, 2, 2, 2, 3],
        ...     "component_token_id": [1, 3, 2, 1, 3, 3],
        ...     "unnormalized_weight": [1.0, 0.5, 1.0, 0.5, 0.25, 1.0]})
        >>> cl = build_event_to_query_nodes(nodes, mix).sort("event_code", "query_node")
        >>> list(zip(cl["event_code"], cl["query_node"]))
        [('A//B', 'A'), ('A//B', 'A//B'), ('A//B//C', 'A'), ('A//B//C', 'A//B//C')]
    """
    index_to_name = dict(zip(nodes_df["token_id"], nodes_df["node_name"], strict=True))
    leaf_indices = set(nodes_df.filter(pl.col("is_observed_code"))["token_id"].to_list())

    leaf_list = list(leaf_indices)
    rows = mix_df.filter(
        pl.col("target_token_id").is_in(leaf_list)
        & (
            (pl.col("component_token_id") == pl.col("target_token_id"))
            | ~pl.col("component_token_id").is_in(leaf_list)
        )
    )
    return pl.DataFrame(
        {
            "event_code": [index_to_name[i] for i in rows["target_token_id"].to_list()],
            "query_node": [index_to_name[i] for i in rows["component_token_id"].to_list()],
        }
    )


def load_nodes(ontology_dir: str | Path) -> pl.DataFrame:
    """Read ``ontology_vocab.parquet`` — the extended vocabulary."""
    return pl.read_parquet(Path(ontology_dir) / ONTOLOGY_VOCAB_FILE)


def extended_vocab_size(ontology_dir: str | Path) -> int:
    """``V_ext``: one past the highest node index, i.e. the embedding table's required height."""
    return int(load_nodes(ontology_dir)["token_id"].max()) + 1


def extend_code_map(code_to_index: dict[str, int], ontology_dir: str | Path) -> dict[str, int]:
    """Add ancestor names to a cohort's ``code -> index`` map, making them queryable.

    ``setdefault`` semantics: when a name is both a real code and some other code's prefix, its
    canonical leaf index wins.

    Examples:
        >>> import tempfile, polars as pl
        >>> with tempfile.TemporaryDirectory() as d:
        ...     nodes = pl.DataFrame({"node_name": ["A//B", "A"], "token_id": [1, 7],
        ...                           "is_observed_code": [True, False]})
        ...     _ = nodes.write_parquet(Path(d) / ONTOLOGY_VOCAB_FILE)
        ...     extend_code_map({"A//B": 1}, d)
        {'A//B': 1, 'A': 7}
    """
    extended = dict(code_to_index)
    nodes = load_nodes(ontology_dir)
    for node, idx in zip(nodes["node_name"], nodes["token_id"], strict=True):
        extended.setdefault(node, int(idx))
    return extended


def load_mix_matrix(ontology_dir: str | Path, normalize: bool = True) -> torch.Tensor:
    """Load the sparse ``(V_ext, V_ext)`` embedding-mix matrix; rows sum to 1 when normalised."""
    ontology_dir = Path(ontology_dir)
    mix = pl.read_parquet(ontology_dir / EMBEDDING_MIX_FILE)
    v_ext = extended_vocab_size(ontology_dir)

    idx = torch.tensor(
        [mix["target_token_id"].to_list(), mix["component_token_id"].to_list()], dtype=torch.long
    )
    w = torch.tensor(mix["unnormalized_weight"].to_list(), dtype=torch.float32)
    A = torch.sparse_coo_tensor(idx, w, size=(v_ext, v_ext)).coalesce()  # noqa: N806 — `A` is the mix matrix throughout the docs

    if normalize:
        # clamp guards row 0 (PAD), which is never a node and therefore has no entries at all.
        row_sums = torch.sparse.sum(A, dim=1).to_dense().clamp(min=1e-9)
        vals = A.values() / row_sums[A.indices()[0]]
        A = torch.sparse_coo_tensor(A.indices(), vals, size=(v_ext, v_ext)).coalesce()  # noqa: N806
    return A


def load_event_to_query_nodes(ontology_dir: str | Path) -> pl.DataFrame:
    """Read ``event_to_query_nodes.parquet`` — ``(event_code, query_node)`` rows for event explosion."""
    return pl.read_parquet(Path(ontology_dir) / EVENT_TO_QUERY_NODES_FILE)


def expand_events_to_query_nodes(
    events_df: pl.DataFrame, event_to_query_nodes_df: pl.DataFrame
) -> pl.DataFrame:
    """Repeat each event under every ancestor node name, so ancestor queries label normally.

    Codes absent from ``event_to_query_nodes_df`` would be **dropped** by the inner join, silently deleting
    events — which happens whenever the ontology was built from a different ``codes.parquet``
    than the cohort.  Those codes are passed through unchanged instead, and the mismatch is
    reported.

    Examples:
        >>> from datetime import datetime
        >>> ev = pl.DataFrame({"subject_id": [1], "time": [datetime(2024, 1, 1)], "code": ["A//B"]})
        >>> cl = pl.DataFrame({"event_code": ["A//B", "A//B"], "query_node": ["A//B", "A"]})
        >>> sorted(expand_events_to_query_nodes(ev, cl)["code"].to_list())
        ['A', 'A//B']

        An event whose code the ontology does not know survives as itself:

        >>> ev2 = pl.DataFrame({"subject_id": [1, 1], "time": [datetime(2024, 1, 1)] * 2,
        ...                     "code": ["A//B", "UNKNOWN"]})
        >>> sorted(expand_events_to_query_nodes(ev2, cl)["code"].to_list())
        ['A', 'A//B', 'UNKNOWN']
    """
    known = set(event_to_query_nodes_df["event_code"].to_list())
    # ``.unique()`` before ``.to_list()``: the set is over distinct codes either way, but the naive
    # form materialises one Python string per *event*, which on a full shard is gigabytes of
    # transient objects in every Stage 4M worker before a single interval table exists.
    present = set(events_df["code"].unique().to_list())
    missing = present - known
    if missing:
        logger.warning(
            "%d event code(s) are absent from the ontology closure and are kept unexploded "
            "(the ontology may have been built from a different codes.parquet): %s",
            len(missing),
            sorted(missing)[:5],
        )

    exploded = (
        events_df.join(event_to_query_nodes_df, left_on="code", right_on="event_code", how="inner")
        .drop("code")
        .rename({"query_node": "code"})
        .select(events_df.columns)
    )
    if not missing:
        return exploded
    return pl.concat([exploded, events_df.filter(pl.col("code").is_in(list(missing)))], how="vertical")


# ---------------------------------------------------------------------------
# The closure as index pairs: deriving ancestor targets from leaf targets
# ---------------------------------------------------------------------------
#
# Under the multitask window rule ("some occurrence of ``v`` falls strictly inside the window"), an
# ancestor node's target bit is exactly the OR of its descendant leaves' bits::
#
#     ancestor[b, k, a] = OR_{v in descendants(a)} leaf[b, k, v]
#
# ``descendants(a)`` is the ``event_to_query_nodes.parquet`` closure - the same table the scalar
# QuerySeq sampler explodes events through - so ancestor bits carry no information beyond the leaf
# row plus the closure.  The multitask sampler therefore keeps its ``.labels.npy`` sidecars
# leaf-only, and the model derives the ancestor block per batch, on whatever device the batch is
# on, from the pairs below.


def closure_fingerprint(ontology_dir: str | Path) -> str:
    """Digest of ``event_to_query_nodes.parquet`` - the closure that decides every ancestor label.

    Two ontologies with the same closure label identically, whatever else differs between them
    (decay, the mix); two closures that differ by one pair may flip a label.  The evaluation grid
    sampler records this as the first half of its provenance ``ontology_fingerprint`` and
    ``EQ_predict_multitask`` compares it against the checkpoint's ontology before scoring.  Both go
    through :func:`every_query.utils.digest.frame_digest`, so they compare like with like.
    """
    return frame_digest(load_event_to_query_nodes(ontology_dir))


def cohort_code_map(code_metadata_fp: str | Path) -> dict[str, int]:
    """The cohort's ``code -> code/vocab_index`` mapping from its ``codes.parquet``.

    The same two columns every dataset reads to encode codes, and the same rows ``EQ_build_ontology``
    turns into the ontology's observed nodes - so :func:`check_ontology_cohort` compares the two
    sides like with like.  Rows without an index are not part of the vocabulary and are dropped.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as d:
        ...     fp = Path(d) / "codes.parquet"
        ...     _ = pl.DataFrame({"code": ["A", "B", "C"], "code/vocab_index": [2, 1, None],
        ...                       "description": ["a", "b", "c"]}).write_parquet(fp)
        ...     cohort_code_map(fp)
        {'A': 2, 'B': 1}
    """
    codes = pl.read_parquet(Path(code_metadata_fp), columns=["code", "code/vocab_index"]).filter(
        pl.col("code").is_not_null() & pl.col("code/vocab_index").is_not_null()
    )
    return {
        c: int(i) for c, i in zip(codes["code"].to_list(), codes["code/vocab_index"].to_list(), strict=True)
    }


def observed_code_map(ontology_dir: str | Path) -> dict[str, int]:
    """The ontology's observed nodes as ``code -> token_id``: the ``codes.parquet`` rows it was built from.

    ``build_ontology`` keeps every leaf at its own ``code/vocab_index``, so for an ontology built
    from a cohort this equals that cohort's :func:`cohort_code_map` exactly.  Unindexed rows are
    dropped on both sides, so the two maps stay comparable on a cohort that carries them.
    """
    leaves = load_nodes(ontology_dir).filter(pl.col("is_observed_code") & pl.col("token_id").is_not_null())
    return {
        c: int(i) for c, i in zip(leaves["node_name"].to_list(), leaves["token_id"].to_list(), strict=True)
    }


def ontology_vocab_fingerprint(ontology_dir: str | Path) -> str:
    """:func:`~every_query.utils.digest.vocab_fingerprint` of the ontology's observed nodes.

    The digest the multitask sampler records in every manifest as ``vocab_fingerprint`` and the
    multitask dataset recomputes from the cohort's ``codes.parquet`` - so an ontology built from that
    cohort digests to the same string, and a checkpoint that persists the cohort's fingerprint can
    re-verify the ontology it is pointed at on every load.
    """
    return vocab_fingerprint(observed_code_map(ontology_dir))


def check_ontology_cohort(
    ontology_dir: str | Path,
    *,
    code_to_index: Mapping[str, int] | None = None,
    vocab_fingerprint: str | None = None,
) -> None:
    """Require the ontology's observed nodes to *be* the cohort's ``(code, code/vocab_index)`` rows.

    Widths alone cannot tell two cohorts apart: two unrelated ``codes.parquet`` files of the same
    size, or the same codes at permuted indices, give the same ``V`` while every leaf target column
    would be paired with the wrong closure and mix rows.  This is the check that establishes the
    ontology was built from *this* cohort, in whichever form the caller has the cohort:

    Args:
        ontology_dir: The ``EQ_build_ontology`` output directory.
        code_to_index: The cohort's mapping (:func:`cohort_code_map`, or a dataset's
            ``code_to_index`` before any ontology extension).  Compared row for row; the error names
            codes the ontology lacks, codes it has that the cohort does not, and codes at a different
            index.
        vocab_fingerprint: The cohort's :func:`~every_query.utils.digest.vocab_fingerprint` (a
            multitask manifest's ``vocab_fingerprint``, or the one a checkpoint recorded).  Compared
            against :func:`ontology_vocab_fingerprint`.

    At least one of the two must be given; both are checked when both are.

    Raises:
        ValueError: On any difference, or when neither form of the cohort was given.

    Examples:
        The same three codes at permuted indices have the same width and are still refused:

        >>> import tempfile
        >>> codes = pl.DataFrame({"code": ["A//B", "A//C", "D"], "code/vocab_index": [1, 2, 3]})
        >>> nodes, _ = build_ontology(codes)
        >>> with tempfile.TemporaryDirectory() as d:
        ...     _ = nodes.write_parquet(Path(d) / ONTOLOGY_VOCAB_FILE)
        ...     check_ontology_cohort(d, code_to_index={"A//B": 1, "A//C": 2, "D": 3})
        ...     check_ontology_cohort(d, vocab_fingerprint=vocab_fingerprint({"A//B": 1, "A//C": 2, "D": 3}))
        ...     check_ontology_cohort(d, code_to_index={"A//B": 1, "A//C": 3, "D": 2})
        Traceback (most recent call last):
            ...
        ValueError: The ontology at ... was built from a different codes.parquet than this cohort ...
        >>> check_ontology_cohort("/nowhere")
        Traceback (most recent call last):
            ...
        ValueError: check_ontology_cohort needs the cohort's code_to_index or its vocab_fingerprint
    """
    if code_to_index is None and vocab_fingerprint is None:
        raise ValueError("check_ontology_cohort needs the cohort's code_to_index or its vocab_fingerprint")
    observed = observed_code_map(ontology_dir)
    if code_to_index is not None:
        cohort = {str(c): int(i) for c, i in code_to_index.items()}
        missing = sorted(set(cohort) - set(observed))
        extra = sorted(set(observed) - set(cohort))
        renumbered = sorted(
            (c, observed[c], cohort[c]) for c in set(cohort) & set(observed) if observed[c] != cohort[c]
        )
        if missing or extra or renumbered:
            details = []
            if missing:
                details.append(f"{len(missing)} cohort code(s) are not ontology leaves (e.g. {missing[:5]})")
            if extra:
                details.append(f"{len(extra)} ontology leaf name(s) are not cohort codes (e.g. {extra[:5]})")
            if renumbered:
                shown = ", ".join(f"{c!r}: ontology {o} vs cohort {k}" for c, o, k in renumbered[:5])
                details.append(f"{len(renumbered)} code(s) sit at a different index ({shown})")
            raise ValueError(
                f"The ontology at {ontology_dir} was built from a different codes.parquet than this cohort "
                f"({'; '.join(details)}).  Its leaf indices would pair the cohort's target columns with the "
                "wrong closure and mix rows; rebuild it with EQ_build_ontology from this cohort."
            )
    if vocab_fingerprint is not None:
        actual = ontology_vocab_fingerprint(ontology_dir)
        if actual != vocab_fingerprint:
            raise ValueError(
                f"The ontology at {ontology_dir} was built from a different codes.parquet than this cohort: "
                f"its observed nodes digest to {actual[:12]}... but the cohort's vocabulary fingerprint is "
                f"{vocab_fingerprint[:12]}....  Rebuild it with EQ_build_ontology from this cohort."
            )


@dataclass(frozen=True)
class ClosureIndex:
    """The strict closure as index pairs: leaf ``leaf_ids[i]`` lies under ancestor ``ancestor_ids[i]``.

    Self pairs (a leaf under itself) are dropped: they are the identity, already present in the
    leaf block.  A dual-role ``X`` / ``X//ANY`` pair survives - ``X//ANY`` is a genuine ancestor
    node and ``X`` one of its descendants.

    Attributes:
        leaf_ids: ``(P,)`` int64, each in ``[0, base_vocab_size)``.
        ancestor_ids: ``(P,)`` int64, each in ``[base_vocab_size, v_ext)``.
        base_vocab_size: ``V``, the cohort's own width (one past the highest leaf index).
        v_ext: ``V_ext``, one past the highest ancestor index; the extended table's width.
    """

    leaf_ids: torch.Tensor
    ancestor_ids: torch.Tensor
    base_vocab_size: int
    v_ext: int

    @property
    def n_ancestors(self) -> int:
        return self.v_ext - self.base_vocab_size

    def to(self, device: torch.device | str) -> "ClosureIndex":
        return ClosureIndex(
            self.leaf_ids.to(device), self.ancestor_ids.to(device), self.base_vocab_size, self.v_ext
        )


def load_closure_index(
    ontology_dir: str | Path,
    base_vocab_size: int | None = None,
    *,
    code_to_index: Mapping[str, int] | None = None,
    vocab_fingerprint: str | None = None,
) -> ClosureIndex:
    """Read the closure as :class:`ClosureIndex`, checking it against the cohort it is used with.

    Two kinds of check.  The **width** checks (always run) require the leaves to span exactly
    ``[.., base_vocab_size)`` and every ancestor to sit above them - the shape the derivation needs.
    They cannot tell two same-width cohorts apart, so the **identity** check
    (:func:`check_ontology_cohort`) runs whenever the caller can say which cohort this is: pass the
    cohort's ``code -> index`` mapping, its vocabulary fingerprint, or both.  Every production load
    passes one; a bare call is width-only and suits tests that built the ontology themselves.

    Args:
        ontology_dir: The ``EQ_build_ontology`` output directory.
        base_vocab_size: The cohort's ``vocab_size`` (``V``).  When given it must equal one past
            the ontology's highest leaf index; when ``None`` that number is taken as ``V``.
        code_to_index: The cohort's ``code -> code/vocab_index`` mapping (:func:`cohort_code_map`),
            compared row for row against the ontology's observed nodes.
        vocab_fingerprint: The cohort's :func:`~every_query.utils.digest.vocab_fingerprint`,
            compared against :func:`ontology_vocab_fingerprint`.

    Raises:
        ValueError: If the ontology's leaves do not span exactly ``[.., base_vocab_size)``, if any
            closure leaf lies at or past ``base_vocab_size``, if any closure ancestor lies below it,
            or if the observed nodes are not the cohort's rows (see :func:`check_ontology_cohort`).

    Examples:
        >>> import tempfile
        >>> codes = pl.DataFrame({"code": ["A//B", "A//C", "D"], "code/vocab_index": [1, 2, 3]})
        >>> nodes, mix = build_ontology(codes)
        >>> with tempfile.TemporaryDirectory() as d:
        ...     _ = nodes.write_parquet(Path(d) / ONTOLOGY_VOCAB_FILE)
        ...     _ = build_event_to_query_nodes(nodes, mix).write_parquet(Path(d) / EVENT_TO_QUERY_NODES_FILE)
        ...     closure = load_closure_index(d, base_vocab_size=4)
        ...     load_closure_index(d, base_vocab_size=5)
        Traceback (most recent call last):
            ...
        ValueError: The ontology's leaves span [.., 4) but the cohort's vocab_size is 5; ...
        >>> closure.base_vocab_size, closure.v_ext, closure.n_ancestors
        (4, 5, 1)
        >>> sorted(zip(closure.leaf_ids.tolist(), closure.ancestor_ids.tolist()))
        [(1, 4), (2, 4)]

        A cohort with the same three codes at permuted indices has the same width, so it passes the
        width checks alone and is refused only once the loader is told which cohort it is:

        >>> permuted = {"A//B": 2, "A//C": 1, "D": 3}
        >>> with tempfile.TemporaryDirectory() as d:
        ...     _ = nodes.write_parquet(Path(d) / ONTOLOGY_VOCAB_FILE)
        ...     _ = build_event_to_query_nodes(nodes, mix).write_parquet(Path(d) / EVENT_TO_QUERY_NODES_FILE)
        ...     width_only = load_closure_index(d, base_vocab_size=4)  # passes: same V, same V_ext
        ...     load_closure_index(d, base_vocab_size=4, code_to_index=permuted)
        Traceback (most recent call last):
            ...
        ValueError: The ontology at ... was built from a different codes.parquet than this cohort ...
        >>> width_only.n_ancestors
        1
    """
    nodes = load_nodes(ontology_dir)
    leaves = nodes.filter(pl.col("is_observed_code"))
    inferred = int(leaves["token_id"].max()) + 1
    if base_vocab_size is None:
        base_vocab_size = inferred
    elif inferred != base_vocab_size:
        raise ValueError(
            f"The ontology's leaves span [.., {inferred}) but the cohort's vocab_size is {base_vocab_size}; "
            f"the ontology at {ontology_dir} was built from a different codes.parquet than this cohort."
        )
    v_ext = int(nodes["token_id"].max()) + 1
    ancestors = nodes.filter(~pl.col("is_observed_code"))
    if ancestors.height and int(ancestors["token_id"].min()) < base_vocab_size:
        raise ValueError(
            f"The ontology at {ontology_dir} has an ancestor node at index "
            f"{int(ancestors['token_id'].min())}, below the cohort's vocab_size {base_vocab_size}; ancestor "
            "nodes must be appended above every leaf."
        )

    name_to_id = dict(zip(nodes["node_name"].to_list(), nodes["token_id"].to_list(), strict=True))
    closure = load_event_to_query_nodes(ontology_dir).filter(pl.col("event_code") != pl.col("query_node"))
    unknown = sorted(
        set(closure["event_code"].to_list() + closure["query_node"].to_list()) - name_to_id.keys()
    )
    if unknown:
        raise ValueError(
            f"{len(unknown)} closure name(s) are absent from {ONTOLOGY_VOCAB_FILE} (e.g. {unknown[:3]}); "
            f"the artifacts under {ontology_dir} were not written together."
        )
    leaf_ids = torch.tensor([name_to_id[c] for c in closure["event_code"].to_list()], dtype=torch.long)
    ancestor_ids = torch.tensor([name_to_id[c] for c in closure["query_node"].to_list()], dtype=torch.long)
    if leaf_ids.numel():
        if int(leaf_ids.max()) >= base_vocab_size or int(leaf_ids.min()) < 0:
            raise ValueError(
                f"closure leaf index {int(leaf_ids.max())} is outside the cohort vocabulary "
                f"[0, {base_vocab_size}); the ontology at {ontology_dir} is not this cohort's."
            )
        if int(ancestor_ids.min()) < base_vocab_size or int(ancestor_ids.max()) >= v_ext:
            raise ValueError(
                f"closure ancestor indices must lie in [{base_vocab_size}, {v_ext}); got "
                f"[{int(ancestor_ids.min())}, {int(ancestor_ids.max())}] from the ontology at {ontology_dir}."
            )
    if code_to_index is not None or vocab_fingerprint is not None:
        check_ontology_cohort(ontology_dir, code_to_index=code_to_index, vocab_fingerprint=vocab_fingerprint)
    return ClosureIndex(leaf_ids, ancestor_ids, base_vocab_size, v_ext)


def derive_ancestor_targets(leaf: torch.Tensor, closure: ClosureIndex) -> torch.Tensor:
    """``(B, K, V)`` leaf bits -> ``(B, K, V_ext)`` bits with every ancestor column OR-ed from its leaves.

    Pure tensor arithmetic, no autograd: gather the closure's leaf columns, ``index_add_`` them into
    a zero ``(B, K, n_ancestors)`` buffer keyed by ``ancestor_id - V``, threshold ``> 0`` and append
    after the untouched leaf block.  Works on CPU and GPU alike; about 50k pairs on a real cohort
    cost well under a millisecond per batch.  The PAD column is part of the leaf block and is
    passed through as-is (it is never a closure leaf and never an ancestor).

    Examples:
        Leaves 1 and 2 sit under ancestor 3; leaf 1 alone is also under ancestor 4:

        >>> closure = ClosureIndex(torch.tensor([1, 2, 1]), torch.tensor([3, 3, 4]), 3, 5)
        >>> leaf = torch.tensor([[[False, False, True], [False, False, False]]])
        >>> derive_ancestor_targets(leaf, closure).int().tolist()
        [[[0, 0, 1, 1, 0], [0, 0, 0, 0, 0]]]
        >>> derive_ancestor_targets(torch.tensor([[[False, True, False]]]), closure).int().tolist()
        [[[0, 1, 0, 1, 1]]]

        Any leading shape works; only the last axis must be ``V``:

        >>> derive_ancestor_targets(torch.zeros(2, 3, 3, dtype=torch.bool), closure).shape
        torch.Size([2, 3, 5])
        >>> derive_ancestor_targets(torch.zeros(1, 4, dtype=torch.bool), closure)
        Traceback (most recent call last):
            ...
        ValueError: leaf targets are 4 wide but the closure expects the cohort width V=3
    """
    if leaf.shape[-1] != closure.base_vocab_size:
        raise ValueError(
            f"leaf targets are {leaf.shape[-1]} wide but the closure expects the cohort width "
            f"V={closure.base_vocab_size}"
        )
    leaf = leaf.bool()
    n_anc = closure.n_ancestors
    if n_anc == 0:
        return leaf
    leaf_ids = closure.leaf_ids.to(leaf.device)
    ancestor_slots = closure.ancestor_ids.to(leaf.device) - closure.base_vocab_size
    counts = torch.zeros(*leaf.shape[:-1], n_anc, dtype=torch.int32, device=leaf.device)
    if leaf_ids.numel():
        counts.index_add_(leaf.dim() - 1, ancestor_slots, leaf[..., leaf_ids].to(torch.int32))
    return torch.cat([leaf, counts > 0], dim=-1)
