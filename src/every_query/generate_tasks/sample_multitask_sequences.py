"""All-vocabulary, multi-bound *multitask* label generator (issue #20).

For every sampled patient context the sampler draws a fixed sequence of ``K = num_bounds`` windows
and produces one boolean target per **base-vocabulary code** per window::

    target[i, k, v] = resolved_start[i, k] < t_v < resolved_end[i, k]

for some occurrence ``t_v`` of code ``v``.  Issue #24 gives every window an explicit start and end::

    start:  prediction_time + start_duration            (0 = the prediction time itself, issue #20)
            OR first occurrence of start_event strictly after prediction_time (+inf if none)
    end:    resolved_start + duration
            OR first occurrence of bound_event strictly after resolved_start (+inf if it never recurs)

The start is resolved first and the end relative to it; an unresolved event start never opens the
window (every target false, the end is ``+inf`` too, never ``pt``), and an unresolved event end runs to
the end of the record.  The window is open at both ends, exactly as the scalar
:func:`~every_query.generate_tasks.query_sequence_labeling.label_with_event_bounds` path defines the
``(prediction_time, boundary)`` window; that function is the correctness oracle the tests compare
against (fed the resolved start as its prediction time).

The staged sampler architecture is reused::

    Stage 0    build + cache the canonical prediction-time map (reused as-is)
    Stage 1M   sample one sequence of K windows (start + end spec) per future context
               (BoundaryDistribution)
    Stage 2    sample patient contexts (reused as-is)
    Stage 3M   zip windows with contexts, resolve prediction times, partition by event shard,
               sort each partition by (subject_id, prediction_time, _ctx_id)
    Stage 4M   per shard: build the interval table, resolve the (N, K) start then end matrices,
               flatten to N x K logical windows sorted by (subject_id, resolved_start), label every
               vocabulary code with the subject-sorted interval-range kernel, scatter the packed rows
               back to (N, K), and write

Issue #22 adds ``K-1`` **conditioning** code/answer pairs per context for teacher forcing the planned
decoder-only model: Stage 1M draws ``condition_codes`` iid uniform over all non-PAD base codes from a
dedicated RNG stream (never perturbing the boundary streams), Stage 3M carries them through the index,
and Stage 4M materializes ``condition_answers[i, j] = target[i, j, vocab_index(condition_codes[i, j])]``.
In a conditioning ontology mode the pool is the ontology's non-PAD *nodes* instead (the manifest's
``condition_policy`` says which), and an ancestor's answer is the OR of that bit over its closure
descendants - resolved from the expanded interval table, since no ancestor has a stored column.

Output, per event shard of the split::

    out_dir/{split}/{shard}.parquet             MultitaskBoundarySchema metadata, one row per context
    out_dir/{split}/{shard}.labels.npy          uint8 (rows, K, ceil(V/8)), little bit order,
                                                row-aligned with the parquet
    out_dir/{split}/_multitask_manifest.json    split-level manifest, written by the driver alone

Targets are bit-packed and written incrementally through a temporary ``open_memmap``; no unpacked
shard-wide target tensor is ever allocated, and no worker holds shard-wide ``(context, code,
next_time)`` triples.

**Leaf-only by design**: the target vocabulary is the cohort's ``codes.parquet`` - observable codes,
bits aligned to the unchanged ``code/vocab_index``.  Ancestor *targets* need no sampler support:
under the window rule an ancestor's bit is the OR of its descendant leaves' bits, so
:class:`~every_query.model.conditional_multitask_ar_model.ConditionalMultitaskARModel` derives them
per batch from these leaf sidecars and the ontology's closure
(:func:`~every_query.data.ontology.derive_ancestor_targets`); storing them would only add ~50% to
every ``.labels.npy`` for no information.

**Ancestors as events**: what an ontology *does* change here is the other half of a window - an
ancestor-valued ``start_event`` / ``bound_event`` ("until the next occurrence of any ``LAB//X//*``")
and an ancestor-valued conditioning code.  Set ``ontology_dir`` and, optionally, ``ontology_mode``
(``boundaries`` | ``conditions`` | ``boundaries+conditions``, the default) to enable them.  The
three seams do it: :func:`build_target_vocabulary` widens the *event* names to the ontology's nodes
at their ``[V, V_ext)`` token ids, :func:`prepare_events_for_labeling` explodes the stream through
the closure so an ancestor has ordinary intervals, and :func:`resolve_event_boundaries` then needs
no ancestor-specific code at all.

The bits never move: the labeling interval table is rebuilt from the ``code_index < V`` rows of the
expanded stream - which the self-pairs in the closure make identical to the unexpanded stream - so
``vocab_size``, ``packed_width_bytes``, ``vocab_fingerprint`` and every label bit are the same in
every mode.  The manifest records ``ontology_mode``, ``ontology_fingerprint`` (the closure digest)
and ``boundary_vocab_size`` (``V_ext``) so a reader can tell which event vocabulary the windows were
drawn against.
"""

from __future__ import annotations

import os

# Pin polars to a single thread BEFORE importing polars, mirroring the sibling samplers: Stage 4M
# workers inherit this env, and process-level fan-out already saturates cores.
os.environ.setdefault("POLARS_MAX_THREADS", "1")

import hashlib
import json
import logging
import multiprocessing
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import numpy as np
import polars as pl
from meds import DataSchema
from omegaconf import DictConfig, ListConfig

from every_query.data.query_seq_dataset import EVENT_BOUND_DURATION_SENTINEL
from every_query.data.schema import MultitaskBoundarySchema, TaskQuerySchema
from every_query.generate_tasks.interval_table import (
    INF,
    IntervalTable,
    build_interval_table,
    iter_packed_label_chunks,
    next_occurrence_after,
    resolve_end_times,
    resolve_start_times,
)
from every_query.generate_tasks.query_sequence_labeling import resolve_prediction_times
from every_query.generate_tasks.sample_tasks import (
    INDEX_DIRNAME,
    LABELED_DIRNAME,
    _atomic_write_json,
    _atomic_write_parquet,
    _index_fingerprint,
    _read_event_shard,
    _require_path_arg,
    _unique_tmp_path,
    build_prediction_times,
    default_artifacts_dir,
    index_path,
    prediction_time_counts_path,
    resolve_workers,
    sample_patient_contexts,
)
from every_query.utils.digest import VOCAB_FINGERPRINT_VERSION as VOCAB_FINGERPRINT_VERSION  # re-export
from every_query.utils.digest import vocab_fingerprint
from every_query.utils.seeds import derive_seed

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

logger = logging.getLogger(__name__)

FORMAT_VERSION = 3
# The vocabulary fingerprint salt (``VOCAB_FINGERPRINT_VERSION``, owned by ``every_query.utils.digest``
# and re-exported above) is pinned independently of FORMAT_VERSION so a format bump does not change
# ``vocab_fingerprint`` and legacy (v2) outputs still pass the cohort check.
MANIFEST_NAME = "_multitask_manifest.json"
LABELS_SUFFIX = ".labels.npy"
ONTOLOGY_MODE_NONE = "none"
# An ancestor node may act as an *event* (a start / bound code, "until the next occurrence of any
# ``LAB//X//*``") and/or as a conditioning code.  Targets are never affected: the bits stay leaf-only
# in every mode, so ``vocab_size`` / ``packed_width_bytes`` / ``vocab_fingerprint`` are mode-invariant.
ONTOLOGY_MODE_BOUNDARIES = "boundaries"
ONTOLOGY_MODE_CONDITIONS = "conditions"
ONTOLOGY_MODE_BOTH = "boundaries+conditions"
ONTOLOGY_MODES = (
    ONTOLOGY_MODE_NONE,
    ONTOLOGY_MODE_BOUNDARIES,
    ONTOLOGY_MODE_CONDITIONS,
    ONTOLOGY_MODE_BOTH,
)
BITORDER = "little"
WINDOW_SEMANTICS = "open_open"
# Issue #24 window semantics, recorded in the manifest and folded into the config fingerprint.
START_REFERENCE = "prediction_time"
DURATION_END_REFERENCE = "resolved_start"
MISSING_EVENT_START = "empty_window"
MISSING_EVENT_BOUNDARY = "infinity"
DATETIME_UNIT = "us"
# Issue #22: K-1 conditioning codes per context, iid uniform with replacement over all non-PAD base
# codes, from a dedicated RNG stream; answer j is the target bit of that code at boundary j.
CONDITION_POLICY = "uniform_base_vocab_no_pad"
# ...and, when the mode lets an ancestor be a conditioning code, the pool is the ontology's whole
# non-PAD node set instead.  The manifest records which, because it is the record of what a stored
# ``condition_answers`` bit means: for an ancestor the answer is the OR over its closure descendants.
CONDITION_POLICY_QUERY_NODES = "uniform_query_node_no_pad"


def condition_policy(mode: str) -> str:
    """The manifest's ``condition_policy`` for an ontology mode.

    Examples:
        >>> condition_policy("none"), condition_policy("boundaries")
        ('uniform_base_vocab_no_pad', 'uniform_base_vocab_no_pad')
        >>> condition_policy("conditions"), condition_policy("boundaries+conditions")
        ('uniform_query_node_no_pad', 'uniform_query_node_no_pad')
    """
    return CONDITION_POLICY_QUERY_NODES if ontology_conditions_enabled(mode) else CONDITION_POLICY


CTX_ID_COL = "_ctx_id"
START_DURATIONS_COL = "start_durations"
START_EVENTS_COL = "start_events"
DURATIONS_COL = "durations"
BOUND_EVENTS_COL = "bound_events"
CONDITION_CODES_COL = "condition_codes"
CONDITION_ANSWERS_COL = "condition_answers"
SID = TaskQuerySchema.subject_id_name
PT = TaskQuerySchema.prediction_time_name

START_COLUMNS = [START_DURATIONS_COL, START_EVENTS_COL]
INDEX_COLUMNS = [CTX_ID_COL, SID, PT, *START_COLUMNS, DURATIONS_COL, BOUND_EVENTS_COL, CONDITION_CODES_COL]
METADATA_COLUMNS = [SID, PT, *START_COLUMNS, DURATIONS_COL, BOUND_EVENTS_COL, CONDITION_CODES_COL]


def resolve_ontology_mode(ontology_dir: object, mode: object = None) -> str:
    """The manifest's ``ontology_mode`` for a config's ``(ontology_dir, ontology_mode)`` pair.

    Without an ontology the only legal mode is ``"none"``.  With one, ``None`` means "use the whole
    feature" (:data:`ONTOLOGY_MODE_BOTH`); an explicit mode narrows it, and ``"none"`` is how a run
    keeps an ontology on disk while sampling exactly the leaf-only draws it sampled before.

    Examples:
        >>> resolve_ontology_mode(None)
        'none'
        >>> resolve_ontology_mode("/onto")
        'boundaries+conditions'
        >>> resolve_ontology_mode("/onto", "boundaries")
        'boundaries'
        >>> resolve_ontology_mode(None, "boundaries")
        Traceback (most recent call last):
            ...
        ValueError: ontology_mode='boundaries' needs an ontology_dir; got None
        >>> resolve_ontology_mode("/onto", "targets")
        Traceback (most recent call last):
            ...
        ValueError: ontology_mode must be one of ('none', 'boundaries', 'conditions', 'boundaries+conditions'), got 'targets'
    """  # noqa: E501 — the doctest's error message is the value being pinned
    mode = None if mode is None else str(mode)
    if mode is not None and mode not in ONTOLOGY_MODES:
        raise ValueError(f"ontology_mode must be one of {ONTOLOGY_MODES}, got {mode!r}")
    if ontology_dir is None or not str(ontology_dir).strip():
        if mode not in (None, ONTOLOGY_MODE_NONE):
            raise ValueError(f"ontology_mode={mode!r} needs an ontology_dir; got None")
        return ONTOLOGY_MODE_NONE
    return ONTOLOGY_MODE_BOTH if mode is None else mode


def ontology_boundaries_enabled(mode: str) -> bool:
    """Whether ``mode`` lets an ancestor node act as a start / bound event."""
    return mode in (ONTOLOGY_MODE_BOUNDARIES, ONTOLOGY_MODE_BOTH)


def ontology_conditions_enabled(mode: str) -> bool:
    """Whether ``mode`` lets an ancestor node be drawn as a conditioning code."""
    return mode in (ONTOLOGY_MODE_CONDITIONS, ONTOLOGY_MODE_BOTH)


# ---------------------------------------------------------------------------
# Vocabulary (extension seam 1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetVocabulary:
    """The cohort's base vocabulary, bit-aligned to its unchanged ``code/vocab_index``.

    The ontology fields are the *event* half of the vocabulary and never touch the bits: an ancestor
    node can be drawn as a start / bound event or as a conditioning code, but it is never a target
    column, so ``size``, ``packed_width`` and ``fingerprint`` are identical with and without one.

    Attributes:
        codes: Codes sorted by vocabulary index.
        indices: ``int64`` vocabulary index per code (same order).  Unique, ``>= 0``.
        size: ``V = max(index) + 1`` - the bit width of every packed boundary row.
        fingerprint: Deterministic digest of the ordered ``(index, code)`` mapping.
        ontology_mode: One of :data:`ONTOLOGY_MODES`; ``"none"`` when no ontology is attached.
        ancestor_names: Non-observed ontology node names, ordered by token id (empty without one).
        ancestor_indices: Their ``[V, V_ext)`` token ids, same order.
        ontology_fingerprint: :func:`~every_query.data.ontology.closure_fingerprint` of the closure
            that resolved the ancestor events, or ``None``.

    Examples:
        >>> v = TargetVocabulary.from_pairs(["B", "A", "PAD"], [2, 1, 0])
        >>> v.codes, v.indices.tolist(), v.size
        (('PAD', 'A', 'B'), [0, 1, 2], 3)
        >>> v.boundary_candidates()
        ['A', 'B']
        >>> v.fingerprint == TargetVocabulary.from_pairs(["A", "B", "PAD"], [1, 2, 0]).fingerprint
        True

        Without an ontology the extended width collapses onto the leaf width and both candidate
        pools are the same leaf pool:

        >>> v.boundary_size, v.condition_candidates()
        (3, ['A', 'B'])

        Attaching a two-node ontology widens the *event* vocabulary only:

        >>> w = v.with_ontology("boundaries", ("A//ANY", "ROOT"), (3, 4), "deadbeef")
        >>> w.size, w.packed_width, w.fingerprint == v.fingerprint
        (3, 1, True)
        >>> w.boundary_size, w.boundary_candidates()
        (5, ['A', 'B', 'A//ANY', 'ROOT'])
        >>> w.condition_candidates()          # "boundaries" mode leaves conditioning codes leaf-only
        ['A', 'B']
        >>> sorted(w.boundary_code_to_index().items())
        [('A', 1), ('A//ANY', 3), ('B', 2), ('PAD', 0), ('ROOT', 4)]
    """

    codes: tuple[str, ...]
    indices: np.ndarray
    size: int
    fingerprint: str
    ontology_mode: str = ONTOLOGY_MODE_NONE
    ancestor_names: tuple[str, ...] = ()
    ancestor_indices: tuple[int, ...] = ()
    ontology_fingerprint: str | None = None
    packed_width: int = field(init=False)
    boundary_size: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "packed_width", (self.size + 7) // 8)
        if self.ontology_mode not in ONTOLOGY_MODES:
            raise ValueError(f"ontology_mode must be one of {ONTOLOGY_MODES}, got {self.ontology_mode!r}")
        if len(self.ancestor_names) != len(self.ancestor_indices):
            raise ValueError("ancestor_names and ancestor_indices must have the same length")
        if self.ancestor_indices and min(self.ancestor_indices) < self.size:
            raise ValueError(
                f"ancestor token ids must all be >= the cohort width V={self.size}; got "
                f"{min(self.ancestor_indices)}"
            )
        v_ext = max(self.ancestor_indices) + 1 if self.ancestor_indices else self.size
        object.__setattr__(self, "boundary_size", v_ext)

    def with_ontology(
        self,
        mode: str,
        ancestor_names: Sequence[str],
        ancestor_indices: Sequence[int],
        ontology_fingerprint: str | None,
    ) -> TargetVocabulary:
        """A copy carrying the ontology's ancestor nodes as extra *event* names.

        Bits are untouched.
        """
        return TargetVocabulary(
            codes=self.codes,
            indices=self.indices,
            size=self.size,
            fingerprint=self.fingerprint,
            ontology_mode=mode,
            ancestor_names=tuple(str(n) for n in ancestor_names),
            ancestor_indices=tuple(int(i) for i in ancestor_indices),
            ontology_fingerprint=ontology_fingerprint,
        )

    @classmethod
    def from_pairs(cls, codes: Sequence[str], indices: Sequence[int]) -> TargetVocabulary:
        if len(codes) != len(indices):
            raise ValueError("codes and indices must have the same length")
        idx = np.asarray(indices, dtype=np.int64)
        if idx.size == 0:
            raise ValueError("the target vocabulary is empty")
        if idx.min() < 0:
            raise ValueError("code/vocab_index must be non-negative")
        if len(np.unique(idx)) != idx.size:
            raise ValueError("code/vocab_index values must be unique")
        if len(set(codes)) != len(codes):
            raise ValueError("codes must be unique")
        order = np.argsort(idx, kind="stable")
        ordered_codes = tuple(str(codes[i]) for i in order)
        ordered_idx = idx[order]
        size = int(ordered_idx[-1]) + 1
        # The one digest every "is this the cohort I labeled under" check compares: the multitask
        # dataset recomputes it from ``codes.parquet`` and the ontology loader over the observed nodes.
        fingerprint = vocab_fingerprint(dict(zip(ordered_codes, ordered_idx.tolist(), strict=True)))
        return cls(codes=ordered_codes, indices=ordered_idx, size=size, fingerprint=fingerprint)

    def code_to_index(self) -> dict[str, int]:
        return dict(zip(self.codes, self.indices.tolist(), strict=True))

    def boundary_code_to_index(self) -> dict[str, int]:
        """``code_to_index`` plus every ancestor node name, whatever the mode.

        The *resolution* map, not the draw pool: a supplied index may name an ancestor even in a mode
        that never draws one, and it must still encode.  Leaf names win over same-named nodes, the
        ``setdefault`` semantics of :func:`~every_query.data.ontology.extend_code_map`.
        """
        extended = self.code_to_index()
        for name, idx in zip(self.ancestor_names, self.ancestor_indices, strict=True):
            extended.setdefault(name, int(idx))
        return extended

    def _leaf_candidates(self) -> list[str]:
        return [c for c, i in zip(self.codes, self.indices.tolist(), strict=True) if i != 0]

    def boundary_candidates(self) -> list[str]:
        """Names drawable as event boundaries / starts: every non-PAD base code, ancestors if enabled."""
        base = self._leaf_candidates()
        if ontology_boundaries_enabled(self.ontology_mode):
            return base + list(self.ancestor_names)
        return base

    def condition_candidates(self) -> list[str]:
        """Names drawable as conditioning codes: every non-PAD base code, ancestors if enabled."""
        base = self._leaf_candidates()
        if ontology_conditions_enabled(self.ontology_mode):
            return base + list(self.ancestor_names)
        return base


def _codes_parquet_path(source: object) -> Path:
    p = Path(str(source))
    if p.is_dir():
        p = p / "metadata" / "codes.parquet"
    if p.suffix != ".parquet":
        raise ValueError(
            f"query_codes must be a metadata root dir or a codes.parquet path (got {source!r}); the "
            "multitask sampler aligns bits to code/vocab_index, which an explicit code list lacks."
        )
    return p


def build_target_vocabulary(
    source: object, ontology_dir: object = None, ontology_mode: object = None
) -> TargetVocabulary:
    """Extension seam 1: the vocabulary whose codes are targets (and boundary candidates).

    The **target** half is always the base cohort vocabulary from ``codes.parquet`` (``code`` +
    ``code/vocab_index``): every bit is an observable leaf, and codes absent from a given split are
    neither removed nor renumbered - their bits simply stay false for that split.  Ancestor targets
    are derived from these leaf bits inside the model, never stored.

    With an ``ontology_dir`` the **event** half widens: the ontology's non-observed nodes are added
    at their ``[V, V_ext)`` token ids so an ancestor can bound (or start, or condition) a window.
    The ontology is checked against this cohort by identity - not merely by width - so a same-width
    ontology of another cohort, or of these codes at permuted indices, is refused rather than paired
    with the wrong closure.
    """
    mode = resolve_ontology_mode(ontology_dir, ontology_mode)
    if isinstance(source, list | tuple | ListConfig):
        raise ValueError(
            "query_codes must be a metadata root dir or a codes.parquet path; the multitask sampler "
            "aligns bits to code/vocab_index, which an explicit code list lacks."
        )
    if source is None or not str(source).strip():
        raise ValueError("query_codes is unset; pass a metadata root dir or a codes.parquet path.")
    fp = _codes_parquet_path(source)
    df = pl.read_parquet(fp, columns=["code", "code/vocab_index"]).filter(
        pl.col("code/vocab_index").is_not_null() & pl.col("code").is_not_null()
    )
    if df.height == 0:
        raise ValueError(f"{fp} holds no indexed codes")
    vocab = TargetVocabulary.from_pairs(df["code"].to_list(), df["code/vocab_index"].to_list())
    if mode == ONTOLOGY_MODE_NONE:
        return vocab
    return attach_ontology(vocab, str(ontology_dir), mode)


def attach_ontology(vocab: TargetVocabulary, ontology_dir: str | Path, mode: str) -> TargetVocabulary:
    """Widen ``vocab``'s *event* names with an ontology's ancestor nodes, checking cohort identity.

    Kept out of :func:`build_target_vocabulary` so a caller that already holds a vocabulary (the
    Stage 4M workers, which reconstruct it from the same ``codes.parquet``) attaches the same
    ancestors without re-reading the cohort.
    """
    from every_query.data.ontology import check_ontology_cohort, closure_fingerprint, load_nodes

    check_ontology_cohort(ontology_dir, code_to_index=vocab.code_to_index())
    nodes = load_nodes(ontology_dir).filter(~pl.col("is_observed_code")).sort("token_id")
    return vocab.with_ontology(
        mode,
        nodes["node_name"].to_list(),
        nodes["token_id"].to_list(),
        closure_fingerprint(ontology_dir),
    )


def read_boundary_codes(
    spec: object,
    vocab: TargetVocabulary,
    exclude_prefixes: Sequence[str] = (),
    ontology_dir: str | Path | None = None,
) -> list[str]:
    """Resolve the boundary-code pool: ``None`` => all base codes (index >= 1); else a list / YAML path.

    Every listed code must be in the vocabulary and must not be PAD; unknown codes are hard errors.
    Order-preserving dedup.  ``exclude_prefixes`` drops codes by name prefix *after* resolution, so an
    explicit pool and the all-vocabulary default are filtered the same way - and with an
    ``ontology_dir`` it also drops any node sitting above an excluded leaf
    (:func:`nodes_over_excluded_leaves`), which a name-prefix test alone cannot reach.
    """
    return _read_code_pool(spec, vocab, "boundary", exclude_prefixes, ontology_dir)


def read_start_event_codes(
    spec: object,
    vocab: TargetVocabulary,
    exclude_prefixes: Sequence[str] = (),
    ontology_dir: str | Path | None = None,
) -> list[str]:
    """Resolve the start-event pool (issue #24) with exactly the :func:`read_boundary_codes` rules."""
    return _read_code_pool(spec, vocab, "start_event", exclude_prefixes, ontology_dir)


def read_exclude_prefixes(spec: object) -> tuple[str, ...]:
    """Normalize the ``exclude_boundary_prefixes`` config value to a tuple of strings.

    Examples:
        >>> read_exclude_prefixes(None)
        ()
        >>> read_exclude_prefixes("TIMELINE//DELTA")
        ('TIMELINE//DELTA',)
        >>> read_exclude_prefixes(["A//", "B//"])
        ('A//', 'B//')
    """
    if spec is None:
        return ()
    if isinstance(spec, str):
        return (spec,)
    return tuple(str(p) for p in spec)


def _read_code_pool(
    spec: object,
    vocab: TargetVocabulary,
    what: str,
    exclude_prefixes: Sequence[str] = (),
    ontology_dir: str | Path | None = None,
) -> list[str]:
    over_excluded: Collection[str] = ()
    if ontology_dir is not None and exclude_prefixes:
        over_excluded = nodes_over_excluded_leaves(ontology_dir, exclude_prefixes)
    if spec is None:
        return _apply_prefix_exclusions(vocab.boundary_candidates(), exclude_prefixes, what, over_excluded)
    if isinstance(spec, list | tuple | ListConfig):
        raw = list(spec)
    else:
        from every_query.generate_tasks.sample_tasks import read_query_codes

        raw = read_query_codes(str(spec))
    seen: set[str] = set()
    codes = [c for c in raw if not (c in seen or seen.add(c))]
    if not codes:
        raise ValueError(f"{what}_codes resolved to an empty list")
    # The extended map, so an explicit pool can name ancestor nodes ("bound on exactly these three
    # subtrees") - the most obvious use of the feature.  Under no ontology it is the base map.
    c2i = vocab.boundary_code_to_index()
    unknown = [c for c in codes if c not in c2i]
    if unknown:
        raise ValueError(f"{len(unknown)} {what} code(s) are not in the base vocabulary: {unknown[:10]}")
    pad = [c for c in codes if c2i[c] == 0]
    if pad:
        raise ValueError(f"{what} code(s) at vocab index 0 (PAD) are not allowed: {pad}")
    return _apply_prefix_exclusions(codes, exclude_prefixes, what, over_excluded)


def nodes_over_excluded_leaves(ontology_dir: str | Path, exclude_prefixes: Sequence[str]) -> set[str]:
    """Ontology nodes that cover at least one leaf whose name matches ``exclude_prefixes``.

    A name-prefix filter cannot reach these on its own: an ancestor's name is a *shorter* string than
    the leaves under it, so ``TIMELINE`` never starts with ``TIMELINE//DELTA``.  Yet drawing
    ``TIMELINE`` as a boundary means "the next occurrence of any ``TIMELINE//*`` event", delta tokens
    included - exactly what the exclusion exists to prevent.  The closure is the only thing that
    knows, so it decides.

    Conservative on purpose: one excluded descendant removes the node.  A node that covers both
    excluded and wanted leaves cannot express "any of these except those", so keeping it would
    silently reintroduce the excluded events.
    """
    from every_query.data.ontology import load_event_to_query_nodes

    prefixes = tuple(exclude_prefixes)
    if not prefixes:
        return set()
    closure = load_event_to_query_nodes(ontology_dir)
    matched = pl.col("event_code").str.starts_with(prefixes[0])
    for p in prefixes[1:]:
        matched = matched | pl.col("event_code").str.starts_with(p)
    return set(closure.filter(matched)["query_node"].to_list())


def _apply_prefix_exclusions(
    codes: list[str],
    exclude_prefixes: Sequence[str],
    what: str,
    also_exclude: Collection[str] = (),
) -> list[str]:
    """Drop every code whose name starts with one of ``exclude_prefixes``, plus ``also_exclude``.

    ``also_exclude`` carries the ontology nodes :func:`nodes_over_excluded_leaves` found sitting above
    an excluded leaf; without it a prefix exclusion would drop a subtree's leaves and leave a node
    that draws them all back in.

    Examples:
        >>> _apply_prefix_exclusions(["A", "T//1", "T//2"], ("T//",), "boundary")
        ['A']
        >>> _apply_prefix_exclusions(["A", "B"], (), "boundary")
        ['A', 'B']
        >>> _apply_prefix_exclusions(["A", "T"], ("T//",), "boundary", also_exclude={"T"})
        ['A']
        >>> _apply_prefix_exclusions(["T//1"], ("T//",), "boundary")
        Traceback (most recent call last):
            ...
        ValueError: excluding prefixes ('T//',) empties the boundary pool
    """
    if not exclude_prefixes and not also_exclude:
        return codes
    prefixes = tuple(exclude_prefixes)
    drop = set(also_exclude)
    kept = [c for c in codes if c not in drop and not (prefixes and c.startswith(prefixes))]
    if not kept:
        raise ValueError(f"excluding prefixes {prefixes} empties the {what} pool")
    return kept


def _ancestor_code_weights(
    stat: dict[str, object], ontology_dir: str | Path, column: str
) -> dict[str, float]:
    """``node -> aggregated descendant statistic`` for every ontology node ``codes.parquet`` lacks.

    How the descendants combine depends on what the statistic counts, and getting this wrong is not a
    rounding error:

    - a **count of occurrences** adds up.  An ancestor occurs whenever any descendant does, so its
      occurrence count is the sum over the closure - the same table that decides its labels - up to
      several descendants sharing a timestamp.
    - a **count of subjects** does not.  A subject carrying two descendant codes would be counted
      twice, so a wide subtree's "n_subjects" can exceed the cohort's subject count by the mean
      number of distinct descendant codes per subject.  The maximum is used instead: a true lower
      bound on "how many subjects have any descendant", and never absurd.  It under-weights a node
      whose descendants reach disjoint subjects, which is the safe direction.

    Only names absent from ``stat`` are added, so every leaf keeps the exact statistic
    ``codes.parquet`` gives it and a weighted leaf-only pool draws identically with and without an
    ontology.
    """
    from every_query.data.ontology import load_event_to_query_nodes

    closure = load_event_to_query_nodes(ontology_dir)
    leaf_stat = pl.DataFrame(
        {
            "event_code": list(stat.keys()),
            "_stat": [0.0 if v is None else float(v) for v in stat.values()],
        },
        schema={"event_code": pl.Utf8, "_stat": pl.Float64},
    )
    combine = pl.col("_stat").max() if "subject" in column else pl.col("_stat").sum()
    agg = closure.join(leaf_stat, on="event_code", how="inner").group_by("query_node").agg(combine)
    return {
        n: float(s)
        for n, s in zip(agg["query_node"].to_list(), agg["_stat"].to_list(), strict=True)
        if n not in stat
    }


def build_code_weights(
    source: object,
    codes: Sequence[str],
    column: str,
    power: float,
    ontology_dir: str | Path | None = None,
) -> tuple[float, ...]:
    """Sampling weights for ``codes``, proportional to ``codes.parquet[column] ** power``.

    The column is a per-code prevalence statistic of the *cohort* (``code/n_occurrences`` or
    ``code/n_subjects``); ``power`` tempers it - ``1.0`` is proportional to prevalence, ``0.5``
    square-root damped, ``0.0`` uniform.  Codes with a null or zero statistic get the smallest
    positive weight in the pool rather than zero, so no pool member becomes undrawable.  Returns
    weights normalized to sum to 1, aligned to ``codes`` positionally.

    An ancestor node has no row in ``codes.parquet``; given an ``ontology_dir`` it inherits an
    aggregate of its descendants' statistic - the **sum** for an occurrence count, the **max** for a
    subject count, where the max is a lower-bound proxy rather than the true union
    (:func:`_ancestor_code_weights` explains why the two columns cannot combine the same way).  That
    is what makes a weighted draw over an ancestor-bearing pool prefer the ancestors that recur.
    """
    if power < 0:
        raise ValueError(f"code_weight_power must be >= 0 (got {power})")
    fp = _codes_parquet_path(source)
    df = pl.read_parquet(fp, columns=["code", column])
    stat: dict[str, object] = dict(zip(df["code"].to_list(), df[column].to_list(), strict=True))
    if ontology_dir is not None:
        stat.update(_ancestor_code_weights(stat, ontology_dir, column))
    missing = [c for c in codes if c not in stat]
    if missing:
        raise ValueError(f"{len(missing)} weighted code(s) are absent from {fp}: {missing[:10]}")
    raw = np.array([0.0 if stat[c] is None else float(stat[c]) for c in codes], dtype=np.float64)
    positive = raw[raw > 0]
    if positive.size == 0:
        raise ValueError(f"{fp} column {column!r} is zero or null for every code in the pool")
    raw[raw <= 0] = positive.min()
    w = np.power(raw, float(power))
    return tuple((w / w.sum()).tolist())


def resolve_boundary_pools(
    cfg: DictConfig, vocab: TargetVocabulary
) -> tuple[list[str], tuple[float, ...], list[str], tuple[float, ...]]:
    """The event-boundary and event-start pools and their sampling weights, from a Hydra config.

    Reads ``boundary_codes`` / ``start_event_codes`` (pools), ``exclude_boundary_prefixes`` (applied
    to both pools) and ``code_weighting`` / ``code_weight_column`` / ``code_weight_power`` (the
    shared weighting policy).  ``code_weighting: null`` keeps the historical iid-uniform draw.

    Returns ``(boundary_codes, boundary_weights, start_event_codes, start_event_weights)``; the
    weight tuples are empty when weighting is off.
    """
    # Keyed on "is an ontology attached", NOT on "does this mode draw ancestors": an *explicit* pool
    # may name ancestor nodes in any attached mode (that is what ``boundary_code_to_index`` is for),
    # and such a pool needs both the closure-aware exclusion and the ancestor statistics.
    onto = None if vocab.ontology_mode == ONTOLOGY_MODE_NONE else cfg.get("ontology_dir")
    exclude = read_exclude_prefixes(cfg.get("exclude_boundary_prefixes"))
    boundary_codes = read_boundary_codes(cfg.get("boundary_codes"), vocab, exclude, onto)
    start_event_codes = read_start_event_codes(cfg.get("start_event_codes"), vocab, exclude, onto)

    weighting = cfg.get("code_weighting")
    if weighting is None or str(weighting).lower() in ("", "null", "none", "uniform"):
        return boundary_codes, (), start_event_codes, ()
    if str(weighting).lower() != "prevalence":
        raise ValueError(f"code_weighting must be null or 'prevalence' (got {weighting!r})")
    column = str(cfg.get("code_weight_column", "code/n_occurrences"))
    power = float(cfg.get("code_weight_power", 1.0))
    source = cfg.get("query_codes")
    return (
        boundary_codes,
        build_code_weights(source, boundary_codes, column, power, onto),
        start_event_codes,
        build_code_weights(source, start_event_codes, column, power, onto),
    )


# ---------------------------------------------------------------------------
# Stage 1M - the boundary-sequence distribution
# ---------------------------------------------------------------------------


def _validate_weights(weights: tuple[float, ...], pool: tuple[str, ...], what: str) -> None:
    """Weights are optional, but when given must be one non-negative number per pool member.

    Examples:
        >>> _validate_weights((), ("A", "B"), "boundary")
        >>> _validate_weights((0.5, 0.5), ("A", "B"), "boundary")
        >>> _validate_weights((0.5,), ("A", "B"), "boundary")
        Traceback (most recent call last):
            ...
        ValueError: boundary_weights has 1 entries but the boundary pool has 2
        >>> _validate_weights((1.0, -1.0), ("A", "B"), "boundary")
        Traceback (most recent call last):
            ...
        ValueError: boundary_weights must be non-negative and sum to a positive value
    """
    if not weights:
        return
    if len(weights) != len(pool):
        raise ValueError(f"{what}_weights has {len(weights)} entries but the {what} pool has {len(pool)}")
    arr = np.asarray(weights, dtype=np.float64)
    if (arr < 0).any() or not arr.sum() > 0:
        raise ValueError(f"{what}_weights must be non-negative and sum to a positive value")
    if abs(float(arr.sum()) - 1.0) > 1e-6:
        raise ValueError(f"{what}_weights must be normalized to sum to 1 (got {float(arr.sum())})")


@dataclass(frozen=True)
class BoundarySample:
    """``num_contexts x K`` window draws plus ``num_contexts x (K-1)`` conditioning codes.

    ``durations`` (float32) and ``bound_events`` (object, None) describe the window ends;
    ``start_durations`` (float32; ``0`` = the prediction time, ``> 0`` = days after it,
    ``EVENT_BOUND_DURATION_SENTINEL`` = event-defined) and ``start_events`` (object, None) describe the
    starts (issue #24); ``condition_codes`` (object) holds the code whose answer at window ``j`` is
    revealed to windows ``j+1..K-1`` (issue #22).
    """

    durations: np.ndarray
    bound_events: np.ndarray
    condition_codes: np.ndarray
    start_durations: np.ndarray
    start_events: np.ndarray

    @property
    def n(self) -> int:
        return int(self.durations.shape[0])

    @property
    def k(self) -> int:
        return int(self.durations.shape[1])


@dataclass(frozen=True)
class BoundaryDistribution:
    """Stage 1M: one fixed-length sequence of ``K`` windows (+ ``K-1`` conditioning codes) per context.

    Every slot is drawn independently.  End: a Bernoulli(``eventbound_fraction``) form draw, a duration
    from the configured distribution, and a boundary code iid uniform with replacement over
    ``boundary_codes``.  Start (issue #24): one uniform ``u`` per slot picks the form - ``u <
    eventstart_fraction`` => event-defined start (code iid uniform over ``start_event_codes``),
    ``eventstart_fraction <= u < eventstart_fraction + prediction_time_start_fraction`` => the
    prediction time itself (``start_duration == 0``), else a positive duration from the start
    distribution.  Conditioning codes are iid uniform with replacement over ``condition_codes`` (every
    non-PAD base code).  The seven axes use seven caller-owned generators and every stream is drawn in
    full before masking, so changing one axis perturbs none of the others.

    The start parameters default to the issue #20 behaviour (every start at the prediction time).

    Examples:
        >>> dist = BoundaryDistribution(num_bounds=3, min_duration=1.0, max_duration=10.0,
        ...     duration_distribution="uniform", eventbound_fraction=0.5, boundary_codes=("A", "B"),
        ...     condition_codes=("A", "B", "C"), eventstart_fraction=0.3,
        ...     prediction_time_start_fraction=0.3, start_event_codes=("C",))
        >>> rngs = lambda: [np.random.default_rng(i) for i in range(7)]
        >>> s = dist.sample(4, *rngs())
        >>> s.durations.shape, s.bound_events.shape, s.condition_codes.shape, s.durations.dtype
        ((4, 3), (4, 3), (4, 2), dtype('float32'))
        >>> bool(((s.durations == -1.0) == (s.bound_events != None)).all())  # noqa: E711
        True
        >>> s.start_durations.shape, s.start_durations.dtype, s.start_events.shape
        ((4, 3), dtype('float32'), (4, 3))
        >>> bool(((s.start_durations == -1.0) == (s.start_events != None)).all())  # noqa: E711
        True
        >>> bool((s.start_durations[s.start_events == None] >= 0).all())  # noqa: E711
        True

        Durations of duration-bounded slots are unaffected by the event fraction:

        >>> off = BoundaryDistribution(3, 1.0, 10.0, "uniform", 0.0, ("A", "B"), ("A", "B", "C"))
        >>> off = off.sample(4, *rngs())
        >>> keep = s.bound_events == None  # noqa: E711
        >>> bool((s.durations[keep] == off.durations[keep]).all())
        True

        With the legacy defaults every start is the prediction time:

        >>> bool((off.start_durations == 0).all()) and bool((off.start_events == None).all())  # noqa: E711
        True
    """

    num_bounds: int
    min_duration: float
    max_duration: float
    duration_distribution: str
    eventbound_fraction: float
    boundary_codes: tuple[str, ...]
    condition_codes: tuple[str, ...]
    eventstart_fraction: float = 0.0
    prediction_time_start_fraction: float = 1.0
    start_min_duration: float = 1.0
    start_max_duration: float = 180.0
    start_duration_distribution: str = "log-uniform"
    start_event_codes: tuple[str, ...] = ()
    # Per-pool sampling weights, positionally aligned to the pool.  Empty => iid uniform (the
    # issue #20 behaviour); non-empty => iid weighted, both with replacement.
    boundary_weights: tuple[float, ...] = ()
    start_event_weights: tuple[float, ...] = ()

    _VALID_DISTRIBUTIONS = ("uniform", "log-uniform")

    def __post_init__(self) -> None:
        if self.num_bounds < 1:
            raise ValueError(f"num_bounds must be >= 1 (got {self.num_bounds})")
        if self.num_bounds > 1 and not self.condition_codes:
            raise ValueError("num_bounds > 1 requires a non-empty condition_codes pool")
        if self.min_duration <= 0:
            raise ValueError(f"min_duration must be > 0 (got {self.min_duration})")
        if self.max_duration < self.min_duration:
            raise ValueError(
                f"max_duration ({self.max_duration}) must be >= min_duration ({self.min_duration})"
            )
        if self.duration_distribution not in self._VALID_DISTRIBUTIONS:
            raise ValueError(
                f"duration_distribution must be one of {self._VALID_DISTRIBUTIONS} "
                f"(got {self.duration_distribution!r})"
            )
        if not 0.0 <= self.eventbound_fraction <= 1.0:
            raise ValueError(f"eventbound_fraction must be in [0, 1] (got {self.eventbound_fraction})")
        if self.eventbound_fraction > 0 and not self.boundary_codes:
            raise ValueError("eventbound_fraction > 0 requires a non-empty boundary_codes pool")
        if self.eventstart_fraction < 0:
            raise ValueError(f"eventstart_fraction must be >= 0 (got {self.eventstart_fraction})")
        if self.prediction_time_start_fraction < 0:
            raise ValueError(
                f"prediction_time_start_fraction must be >= 0 (got {self.prediction_time_start_fraction})"
            )
        if self.eventstart_fraction + self.prediction_time_start_fraction > 1.0:
            raise ValueError(
                "eventstart_fraction + prediction_time_start_fraction must be <= 1 (got "
                f"{self.eventstart_fraction} + {self.prediction_time_start_fraction})"
            )
        if self.eventstart_fraction > 0 and not self.start_event_codes:
            raise ValueError("eventstart_fraction > 0 requires a non-empty start_event_codes pool")
        if self.start_min_duration <= 0:
            raise ValueError(f"start_min_duration must be > 0 (got {self.start_min_duration})")
        if self.start_max_duration < self.start_min_duration:
            raise ValueError(
                f"start_max_duration ({self.start_max_duration}) must be >= start_min_duration "
                f"({self.start_min_duration})"
            )
        if self.start_duration_distribution not in self._VALID_DISTRIBUTIONS:
            raise ValueError(
                f"start_duration_distribution must be one of {self._VALID_DISTRIBUTIONS} "
                f"(got {self.start_duration_distribution!r})"
            )
        _validate_weights(self.boundary_weights, self.boundary_codes, "boundary")
        _validate_weights(self.start_event_weights, self.start_event_codes, "start_event")

    @classmethod
    def from_config(
        cls,
        cfg: DictConfig,
        boundary_codes: Sequence[str],
        condition_codes: Sequence[str],
        start_event_codes: Sequence[str] = (),
        boundary_weights: Sequence[float] = (),
        start_event_weights: Sequence[float] = (),
    ) -> BoundaryDistribution:
        """Build from a Hydra config.

        Start keys absent from ``cfg`` fall back to the issue #20
        behaviour (``eventstart_fraction=0``, ``prediction_time_start_fraction=1``: every window opens at
        the prediction time); ``start_duration_min/max`` default to 1/180 days, log-uniform.
        """
        pts = cfg.get("prediction_time_start_fraction")
        return cls(
            num_bounds=int(cfg.get("num_bounds", 5)),
            min_duration=float(cfg.duration_min),
            max_duration=float(cfg.duration_max),
            duration_distribution=str(cfg.get("duration_distribution", "log-uniform")),
            eventbound_fraction=float(cfg.get("eventbound_fraction", 0.0) or 0.0),
            boundary_codes=tuple(boundary_codes),
            condition_codes=tuple(condition_codes),
            eventstart_fraction=float(cfg.get("eventstart_fraction", 0.0) or 0.0),
            prediction_time_start_fraction=1.0 if pts is None else float(pts),
            start_min_duration=float(cfg.get("start_duration_min", 1.0)),
            start_max_duration=float(cfg.get("start_duration_max", 180.0)),
            start_duration_distribution=str(cfg.get("start_duration_distribution", "log-uniform")),
            start_event_codes=tuple(start_event_codes),
            boundary_weights=tuple(boundary_weights),
            start_event_weights=tuple(start_event_weights),
        )

    @staticmethod
    def _draw_codes(
        rng: np.random.Generator,
        pool: tuple[str, ...],
        weights: tuple[float, ...],
        shape: tuple[int, int],
    ) -> np.ndarray:
        """``shape`` iid picks (with replacement) into ``pool``: uniform when ``weights`` is empty."""
        if not weights:
            return rng.integers(0, len(pool), size=shape)
        return rng.choice(len(pool), size=shape, p=np.asarray(weights, dtype=np.float64))

    @staticmethod
    def _draw_durations(
        rng: np.random.Generator, lo: float, hi: float, distribution: str, shape: tuple[int, int]
    ) -> np.ndarray:
        if distribution == "log-uniform":
            return np.exp(rng.uniform(np.log(lo), np.log(hi), shape)).astype(np.float32)
        return rng.uniform(lo, hi, shape).astype(np.float32)

    def sample(
        self,
        num_contexts: int,
        form_rng: np.random.Generator,
        duration_rng: np.random.Generator,
        code_rng: np.random.Generator,
        condition_rng: np.random.Generator,
        start_form_rng: np.random.Generator,
        start_duration_rng: np.random.Generator,
        start_code_rng: np.random.Generator,
    ) -> BoundarySample:
        if num_contexts < 0:
            raise ValueError(f"num_contexts must be >= 0 (got {num_contexts})")
        shape = (num_contexts, self.num_bounds)
        cond_shape = (num_contexts, self.num_bounds - 1)
        condition_codes = np.full(cond_shape, None, dtype=object)
        if self.condition_codes:
            pool = np.array(self.condition_codes, dtype=object)
            condition_codes = pool[condition_rng.integers(0, len(pool), size=cond_shape)]

        # End specification (issue #20): unchanged draws from the four original streams.
        is_event = form_rng.random(shape) < self.eventbound_fraction
        durations = self._draw_durations(
            duration_rng, self.min_duration, self.max_duration, self.duration_distribution, shape
        )
        bound_events = np.full(shape, None, dtype=object)
        if self.boundary_codes:
            picks = self._draw_codes(code_rng, self.boundary_codes, self.boundary_weights, shape)
            pool = np.array(self.boundary_codes, dtype=object)
            bound_events[is_event] = pool[picks[is_event]]
        durations[is_event] = EVENT_BOUND_DURATION_SENTINEL

        # Start specification (issue #24): one uniform per slot picks event / prediction time /
        # positive duration; the duration and code streams are drawn in full before masking.
        u = start_form_rng.random(shape)
        start_is_event = u < self.eventstart_fraction
        start_is_pt = ~start_is_event & (u < self.eventstart_fraction + self.prediction_time_start_fraction)
        start_durations = self._draw_durations(
            start_duration_rng,
            self.start_min_duration,
            self.start_max_duration,
            self.start_duration_distribution,
            shape,
        )
        start_events = np.full(shape, None, dtype=object)
        if self.start_event_codes:
            picks = self._draw_codes(start_code_rng, self.start_event_codes, self.start_event_weights, shape)
            pool = np.array(self.start_event_codes, dtype=object)
            start_events[start_is_event] = pool[picks[start_is_event]]
        start_durations[start_is_pt] = 0.0
        start_durations[start_is_event] = EVENT_BOUND_DURATION_SENTINEL
        return BoundarySample(
            durations=durations,
            bound_events=bound_events,
            condition_codes=condition_codes,
            start_durations=start_durations,
            start_events=start_events,
        )


# ---------------------------------------------------------------------------
# Stage 3M - zip with contexts, resolve prediction times, partition and sort
# ---------------------------------------------------------------------------


def _list_column(name: str, values: np.ndarray, dtype: pl.DataType) -> pl.Series:
    """``(N, M)`` array -> ``List`` column via a flat series + reshape: no per-slot Python objects."""
    n, m = values.shape
    if m == 0:  # K == 1: zero conditioning slots; polars cannot reshape to a zero-width array
        return pl.Series(name, [[] for _ in range(n)], dtype=pl.List(dtype))
    # Object arrays go through a flat list of references to the (shared) pool strings; an object
    # ndarray would be inferred as polars Object when every slot is None.
    flat = values.ravel().tolist() if values.dtype == object else values.ravel()
    return pl.Series(name, flat, dtype=dtype).reshape((n, m)).arr.to_list()


def _bounds_to_columns(sample: BoundarySample) -> dict[str, pl.Series]:
    return {
        START_DURATIONS_COL: _list_column(START_DURATIONS_COL, sample.start_durations, pl.Float32),
        START_EVENTS_COL: _list_column(START_EVENTS_COL, sample.start_events, pl.Utf8),
        DURATIONS_COL: _list_column(DURATIONS_COL, sample.durations, pl.Float32),
        BOUND_EVENTS_COL: _list_column(BOUND_EVENTS_COL, sample.bound_events, pl.Utf8),
        CONDITION_CODES_COL: _list_column(CONDITION_CODES_COL, sample.condition_codes, pl.Utf8),
    }


def normalize_index(index_df: pl.DataFrame, num_bounds: int) -> pl.DataFrame:
    """Fill the issue #24 start columns with prediction-time starts when an index lacks them.

    A supplied (or legacy) index without ``start_durations`` / ``start_events`` labels exactly as it
    did under issue #20: ``start_durations = [0.0] * K``, ``start_events = [null] * K``.  Having only
    one of the two columns is an error.

    Examples:
        >>> idx = normalize_index(pl.DataFrame({"subject_id": [1, 2]}), 2)
        >>> idx["start_durations"].to_list(), idx["start_events"].to_list()
        ([[0.0, 0.0], [0.0, 0.0]], [[None, None], [None, None]])
        >>> idx.schema["start_durations"], idx.schema["start_events"]
        (List(Float32), List(String))
    """
    present = [c for c in START_COLUMNS if c in index_df.columns]
    if len(present) == len(START_COLUMNS):
        return index_df
    if present:
        raise ValueError(f"index has {present} but not all of {START_COLUMNS}")
    n = index_df.height
    return index_df.with_columns(
        _list_column(START_DURATIONS_COL, np.zeros((n, num_bounds), dtype=np.float32), pl.Float32),
        _list_column(START_EVENTS_COL, np.full((n, num_bounds), None, dtype=object), pl.Utf8),
    )


def sort_index_for_labeling(index_df: pl.DataFrame) -> pl.DataFrame:
    """Give every context a stable id and sort by ``(subject_id, prediction_time, _ctx_id)``.

    A missing ``_ctx_id`` is filled with the row position, so a supplied index labels the same way as
    a sampled one; the sort is what lets Stage 4M scan each subject's interval slice once per chunk.
    """
    if CTX_ID_COL not in index_df.columns:
        index_df = index_df.with_row_index(CTX_ID_COL).with_columns(pl.col(CTX_ID_COL).cast(pl.Int64))
    return index_df.with_columns(pl.col(CTX_ID_COL).cast(pl.Int64)).sort(SID, PT, CTX_ID_COL)


def build_multitask_index(
    sample: BoundarySample,
    contexts: pl.DataFrame,
    artifacts_dir: Path,
    split: str,
) -> int:
    """Stage 3M: zip boundary sequences with contexts, resolve prediction times, write per-shard index.

    ``sample`` row ``i`` belongs to ``contexts`` row ``i``.  Each partition is written sorted by
    ``(subject_id, prediction_time, _ctx_id)``; ``_ctx_id`` is the global sampling position, so the
    original order is recoverable but need not be preserved downstream.
    """
    if sample.n != contexts.height:
        raise ValueError(f"sample has {sample.n} rows but contexts has {contexts.height}")
    if sample.n == 0:
        raise ValueError("no contexts to index")

    combined = contexts.with_columns(
        pl.Series(CTX_ID_COL, np.arange(sample.n, dtype=np.int64)), **_bounds_to_columns(sample)
    )
    index_dir = artifacts_dir / split / INDEX_DIRNAME
    if index_dir.exists():
        shutil.rmtree(index_dir)

    n_shards = 0
    combined = combined.sort("shard")
    for shard_key, group in combined.group_by("shard", maintain_order=True):
        (shard_name,) = shard_key
        joined = resolve_prediction_times(group, artifacts_dir, split, str(shard_name))
        joined = joined.with_columns(pl.col(PT).cast(pl.Datetime("us")))
        _atomic_write_parquet(
            sort_index_for_labeling(joined.select(INDEX_COLUMNS)),
            index_path(artifacts_dir, split, str(shard_name)),
        )
        n_shards += 1
    return n_shards


# ---------------------------------------------------------------------------
# Manifest + fingerprints
# ---------------------------------------------------------------------------


def effective_support(weights: Sequence[float], pool_size: int) -> float:
    """``exp(H)`` of a weight vector - how many codes the pool behaves like.  Uniform => ``pool_size``.

    Examples:
        >>> effective_support((), 100)
        100.0
        >>> round(effective_support((0.5, 0.5), 2), 6)
        2.0
        >>> round(effective_support((0.99, 0.01), 2), 3)
        1.058
    """
    if not weights:
        return float(pool_size)
    p = np.asarray(weights, dtype=np.float64)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def _sha256_json(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def config_fingerprint(dist: BoundaryDistribution, vocab: TargetVocabulary) -> str:
    """Digest of everything that changes the *meaning* of a stored label bit.

    A changed duration distribution, event-bound fraction, number of windows, boundary pool, any start
    parameter or pool (issue #24), base vocabulary or window semantics changes this digest and so invalidates
    existing labels.
    """
    return _sha256_json(
        {
            "format_version": FORMAT_VERSION,
            "num_bounds": dist.num_bounds,
            "duration_min": dist.min_duration,
            "duration_max": dist.max_duration,
            "duration_distribution": dist.duration_distribution,
            "eventbound_fraction": dist.eventbound_fraction,
            "boundary_codes": _sha256_json(list(dist.boundary_codes)),
            "boundary_weights": _sha256_json([round(w, 12) for w in dist.boundary_weights]),
            "eventstart_fraction": dist.eventstart_fraction,
            "prediction_time_start_fraction": dist.prediction_time_start_fraction,
            "start_duration_min": dist.start_min_duration,
            "start_duration_max": dist.start_max_duration,
            "start_duration_distribution": dist.start_duration_distribution,
            "start_event_codes": _sha256_json(list(dist.start_event_codes)),
            "start_event_weights": _sha256_json([round(w, 12) for w in dist.start_event_weights]),
            "condition_codes": _sha256_json(list(dist.condition_codes)),
            "condition_policy": condition_policy(vocab.ontology_mode),
            "vocab_fingerprint": vocab.fingerprint,
            "vocab_size": vocab.size,
            "window": WINDOW_SEMANTICS,
            "window_semantics": WINDOW_SEMANTICS,
            "start_reference": START_REFERENCE,
            "duration_end_reference": DURATION_END_REFERENCE,
            "missing_event_start": MISSING_EVENT_START,
            "missing_event_boundary": MISSING_EVENT_BOUNDARY,
            "missing_event_end": MISSING_EVENT_BOUNDARY,
            "datetime_unit": DATETIME_UNIT,
            "ontology_mode": vocab.ontology_mode,
            # Ontology keys enter the digest only when there IS one, so a leaf-only run fingerprints
            # exactly as it did before this feature and its existing labels stay reusable.
            **(
                {}
                if vocab.ontology_mode == ONTOLOGY_MODE_NONE
                else {
                    "ontology_fingerprint": vocab.ontology_fingerprint,
                    "boundary_vocab_size": vocab.boundary_size,
                }
            ),
        }
    )


def build_manifest(dist: BoundaryDistribution, vocab: TargetVocabulary) -> dict:
    return {
        "format_version": FORMAT_VERSION,
        "num_bounds": dist.num_bounds,
        "vocab_size": vocab.size,
        "packed_width_bytes": vocab.packed_width,
        "bitorder": BITORDER,
        "window": WINDOW_SEMANTICS,
        "window_semantics": WINDOW_SEMANTICS,
        "start_reference": START_REFERENCE,
        "duration_end_reference": DURATION_END_REFERENCE,
        "missing_event_start": MISSING_EVENT_START,
        "missing_event_boundary": MISSING_EVENT_BOUNDARY,
        "missing_event_end": MISSING_EVENT_BOUNDARY,
        "datetime_unit": DATETIME_UNIT,
        "event_bound_duration_sentinel": EVENT_BOUND_DURATION_SENTINEL,
        "vocab_fingerprint": vocab.fingerprint,
        # Leaf-only bits in every mode: vocab_size / packed_width_bytes / vocab_fingerprint above are
        # ontology-invariant.  These three describe the *event* vocabulary the windows were drawn and
        # resolved against, which is what a reader needs to accept an ancestor start / bound code.
        "ontology_mode": vocab.ontology_mode,
        "ontology_fingerprint": vocab.ontology_fingerprint,
        "boundary_vocab_size": vocab.boundary_size,
        "condition_policy": condition_policy(vocab.ontology_mode),
        "num_condition_codes": dist.num_bounds - 1,
        "n_boundary_codes": len(dist.boundary_codes),
        "n_start_event_codes": len(dist.start_event_codes),
        "boundary_code_policy": "weighted" if dist.boundary_weights else "uniform",
        "start_event_code_policy": "weighted" if dist.start_event_weights else "uniform",
        "config_fingerprint": config_fingerprint(dist, vocab),
        "labels_suffix": LABELS_SUFFIX,
    }


def manifest_path(split_dir: Path) -> Path:
    return split_dir / MANIFEST_NAME


def validate_manifest(manifest: dict) -> dict:
    """Check the fields every reader relies on; return the manifest for chaining."""
    required = {
        "format_version",
        "num_bounds",
        "vocab_size",
        "packed_width_bytes",
        "bitorder",
        "window",
        "window_semantics",
        "start_reference",
        "duration_end_reference",
        "missing_event_start",
        "missing_event_boundary",
        "missing_event_end",
        "datetime_unit",
        "vocab_fingerprint",
        "ontology_mode",
        "config_fingerprint",
    }
    missing = sorted(required - manifest.keys())
    if missing:
        raise ValueError(f"multitask manifest is missing {missing}")
    if manifest["format_version"] != FORMAT_VERSION:
        raise ValueError(f"unsupported manifest format_version {manifest['format_version']!r}")
    if manifest["bitorder"] != BITORDER:
        raise ValueError(f"manifest bitorder must be {BITORDER!r}, got {manifest['bitorder']!r}")
    mode = manifest["ontology_mode"]
    if mode not in ONTOLOGY_MODES:
        raise ValueError(f"manifest ontology_mode must be one of {ONTOLOGY_MODES}, got {mode!r}")
    if manifest["packed_width_bytes"] != (int(manifest["vocab_size"]) + 7) // 8:
        raise ValueError("manifest packed_width_bytes disagrees with vocab_size")
    # Pre-ontology manifests carry neither key; a mode that uses ancestors must carry both, and the
    # extended width can only ever be at or above the leaf width (the bits never move).
    boundary_size = int(manifest.get("boundary_vocab_size") or manifest["vocab_size"])
    if boundary_size < int(manifest["vocab_size"]):
        raise ValueError(
            f"manifest boundary_vocab_size {boundary_size} is narrower than vocab_size "
            f"{manifest['vocab_size']}"
        )
    if mode == ONTOLOGY_MODE_NONE:
        if boundary_size != int(manifest["vocab_size"]):
            raise ValueError(
                f"manifest ontology_mode is {ONTOLOGY_MODE_NONE!r} but boundary_vocab_size "
                f"{boundary_size} exceeds vocab_size {manifest['vocab_size']}"
            )
        if manifest.get("ontology_fingerprint") is not None:
            raise ValueError(
                f"manifest ontology_mode is {ONTOLOGY_MODE_NONE!r} but it records an ontology_fingerprint"
            )
    else:
        # Both keys, so the writer and ``every_query.data.multitask_dataset.read_manifest`` agree on
        # what an ancestor-bearing manifest must carry: the width bounds the ids, the digest says
        # which closure gives them meaning.
        absent = [k for k in ("ontology_fingerprint", "boundary_vocab_size") if not manifest.get(k)]
        if absent:
            raise ValueError(f"manifest ontology_mode is {mode!r} but it records no {' or '.join(absent)}")
    if int(manifest["num_bounds"]) < 1:
        raise ValueError("manifest num_bounds must be >= 1")
    return manifest


def write_manifest(split_dir: Path, manifest: dict) -> dict:
    """Driver-only: atomically write the split manifest, then read it back and validate it."""
    validate_manifest(manifest)
    split_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(manifest, manifest_path(split_dir))
    return read_manifest(split_dir)


def read_manifest(split_dir: Path) -> dict:
    fp = manifest_path(split_dir)
    if not fp.exists():
        raise FileNotFoundError(f"no multitask manifest at {fp}")
    return validate_manifest(json.loads(fp.read_text()))


# ---------------------------------------------------------------------------
# Stage 4M - labeling kernel
# ---------------------------------------------------------------------------


def prepare_events_for_labeling(events_df: pl.DataFrame, ontology_dir: object = None) -> pl.DataFrame:
    """Extension seam 2: the event stream the interval table is built from.

    Without an ontology, the original stream, unexpanded.  With one,
    :func:`~every_query.data.ontology.expand_events_to_query_nodes` repeats each event under every
    node of its closure - the scalar QuerySeq sampler's own explosion, so an ancestor node becomes an
    ordinary code with ordinary intervals and "the next occurrence of any ``LAB//X//*``" is just a
    boundary lookup.

    The closure keeps each leaf paired with *itself*, so the ``code_index < V`` rows of the expanded
    stream are exactly the unexpanded stream: the leaf interval table built from them - the only one
    that ever labels a bit - is unchanged.  Given the same resolved windows, labeling therefore
    writes byte-identical bits.  That is a statement about *this* seam only: a boundaries mode still
    changes which windows get drawn, and so what the bits contain.  What the expansion cannot touch
    is the target vocabulary, the packed width, or what any one bit means.
    """
    if ontology_dir is None:
        return events_df
    from every_query.data.ontology import expand_events_to_query_nodes, load_event_to_query_nodes

    return expand_events_to_query_nodes(events_df, load_event_to_query_nodes(ontology_dir))


def resolve_event_boundaries(
    table: IntervalTable,
    subject_ids: np.ndarray,
    prediction_times: np.ndarray,
    start_durations: np.ndarray,
    start_code_index: np.ndarray,
    durations: np.ndarray,
    bound_code_index: np.ndarray,
    ontology_dir: object = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extension seam 3: the ``(N, K)`` ``(start_times, end_times)`` matrices, start resolved first.

    Leaf-code event starts / bounds via
    :func:`~every_query.generate_tasks.interval_table.resolve_start_times` and
    :func:`~every_query.generate_tasks.interval_table.resolve_end_times`.  Ancestor-valued starts and
    bounds need no code here at all: ``table`` is built over the closure-expanded stream (seam 2), so
    an ancestor id in ``[V, V_ext)`` is an ordinary code with ordinary intervals and "the first
    occurrence after ``t``" resolves through the same searchsorted.  ``ontology_dir`` is accepted for
    signature compatibility and is unused.
    """
    del ontology_dir  # the expansion happened in seam 2; resolution is code-agnostic
    start_times = resolve_start_times(table, subject_ids, prediction_times, start_durations, start_code_index)
    end_times = resolve_end_times(table, subject_ids, start_times, durations, bound_code_index)
    return start_times, end_times


def _check_slot_pair(index_df: pl.DataFrame, dur_col: str, ev_col: str, what: str) -> None:
    flat = index_df.select(pl.col(dur_col).explode().alias("d"), pl.col(ev_col).explode().alias("b"))
    by_event = flat["b"].is_not_null()
    ok = (by_event & (flat["d"] == EVENT_BOUND_DURATION_SENTINEL)) | (~by_event & (flat["d"] >= 0))
    if not ok.all():
        raise ValueError(
            f"each {what} slot must be either {dur_col} >= 0 with a null {ev_col}, or {dur_col} == "
            f"{EVENT_BOUND_DURATION_SENTINEL} with a non-null {ev_col}"
        )


def validate_index(index_df: pl.DataFrame, num_bounds: int) -> None:
    """Check the Stage 3M / supplied index contract: K slots per row, one active start and one active end
    representation each, and exactly K-1 non-null conditioning codes.

    The start columns are optional (a legacy index labels with prediction-time starts, see
    :func:`normalize_index`); when present they must satisfy the same pair rule as the end columns.
    """
    for col in (SID, PT, DURATIONS_COL, BOUND_EVENTS_COL, CONDITION_CODES_COL):
        if col not in index_df.columns:
            raise ValueError(f"index is missing required column {col!r}")
    index_df = normalize_index(index_df, num_bounds)
    if index_df.height == 0:
        return
    lens = index_df.select(
        pl.col(DURATIONS_COL).list.len().alias("d"),
        pl.col(BOUND_EVENTS_COL).list.len().alias("b"),
        pl.col(START_DURATIONS_COL).list.len().alias("sd"),
        pl.col(START_EVENTS_COL).list.len().alias("se"),
        pl.col(CONDITION_CODES_COL).list.len().alias("c"),
    )
    if not ((lens["d"] == num_bounds) & (lens["b"] == num_bounds)).all():
        raise ValueError(f"every index row must carry exactly {num_bounds} durations and bound_events")
    if not ((lens["sd"] == num_bounds) & (lens["se"] == num_bounds)).all():
        raise ValueError(f"every index row must carry exactly {num_bounds} start_durations and start_events")
    if not (lens["c"] == num_bounds - 1).all():
        raise ValueError(f"every index row must carry exactly {num_bounds - 1} condition_codes")
    if index_df[CONDITION_CODES_COL].explode().null_count() and num_bounds > 1:
        raise ValueError("condition_codes must not contain nulls")
    _check_slot_pair(index_df, DURATIONS_COL, BOUND_EVENTS_COL, "boundary")
    _check_slot_pair(index_df, START_DURATIONS_COL, START_EVENTS_COL, "start")


def _all_codes_unknown_msg(n_unknown: int) -> str:
    """A shard where *nothing* matched is not a sparse shard, it is the wrong input.

    String codes that are not in ``codes.parquet`` - most likely the string-coded
    ``tokenized_events`` instead of the vocab-indexed intermediate ``data_dir``.  Every label would
    silently be false.
    """
    return (
        f"all {n_unknown} timed event(s) of this shard carry codes outside the target vocabulary; "
        "no label could ever be true. Check that data_dir is the preprocessing *intermediate* "
        "dir (vocab-indexed codes, matching query_codes/codes.parquet) rather than the "
        "string-coded tokenized_events."
    )


def restrict_to_labelable_events(
    events_df: pl.DataFrame, vocab: TargetVocabulary
) -> tuple[pl.DataFrame, int]:
    """Drop what the cohort vocabulary cannot label, *before* the closure expansion.

    On the leaf-only path this filtering happens after encoding and nothing else can go wrong.  Under
    an ontology the expansion runs first, and two kinds of event would otherwise mint ancestor rows
    that no leaf bit backs:

    - **a code absent from ``codes.parquet``.**
      :func:`~every_query.data.ontology.expand_events_to_query_nodes` passes an unknown code through
      unexploded, and if that string happens to *be* an ontology node's name the extended map then
      resolves it to that node - a phantom occurrence of an ancestor none of whose descendants
      occurred.  Its window would close, and its conditioning answer would be true while every
      descendant leaf bit is false: exactly the disagreement ``derive_ancestor_targets`` reports,
      which is what the dataset's ``collate`` check raises on.
    - **a code at vocabulary index 0.**  Its own bit is false by construction, so the ancestors above
      it must not be true either.

    Returns the filtered frame and the number of *timed* rows dropped as unknown, which the caller
    folds into the same statistic :func:`_encode_events` reports on the leaf-only path.
    """
    c2i = vocab.code_to_index()
    code = pl.col(DataSchema.code_name).cast(pl.Utf8)
    timed = pl.col(DataSchema.time_name).is_not_null()
    n_unknown = int(events_df.filter(timed & ~code.is_in(list(c2i.keys()))).height)
    kept = events_df.filter(code.is_in([c for c, i in c2i.items() if i > 0]))
    if kept.filter(timed).height == 0 and n_unknown > 0:
        raise ValueError(_all_codes_unknown_msg(n_unknown))
    return kept, n_unknown


def _encode_events(
    events_df: pl.DataFrame, vocab: TargetVocabulary
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """``(subject_id, time_us, code_index)`` of the non-null-time, in-vocabulary events + drop count.

    Index 0 (PAD) is excluded before the table is built, so target bit 0 is false by construction.
    Under an ontology the stream is closure-expanded (seam 2) and the ancestor node names resolve
    through :meth:`TargetVocabulary.boundary_code_to_index`, so the returned indices span
    ``[1, V_ext)``; the caller splits them at ``V`` into the leaf table that labels and the extended
    table that only resolves boundaries.
    """
    ext = vocab.boundary_code_to_index()
    codes_df = pl.DataFrame(
        {"code": list(ext.keys()), "_code_index": np.fromiter(ext.values(), dtype=np.int64, count=len(ext))}
    )
    ev = (
        events_df.select(SID, DataSchema.time_name, DataSchema.code_name)
        .filter(pl.col(DataSchema.time_name).is_not_null())
        .with_columns(pl.col(DataSchema.code_name).cast(pl.Utf8))
        .join(codes_df, on="code", how="left")
    )
    n_unknown = int(ev["_code_index"].null_count())
    ev = ev.filter(pl.col("_code_index") > 0)  # drops unknown (null) and PAD (0)
    if ev.height == 0 and n_unknown > 0:
        raise ValueError(_all_codes_unknown_msg(n_unknown))
    sid = ev[SID].to_numpy().astype(np.int64)
    t = ev[DataSchema.time_name].cast(pl.Datetime("us")).cast(pl.Int64).to_numpy().astype(np.int64)
    ci = ev["_code_index"].to_numpy().astype(np.int64)
    return sid, t, ci, n_unknown


def _map_codes(codes: pl.Series, vocab: TargetVocabulary, what: str) -> np.ndarray:
    """Flat code series -> ``int64`` vocab indices (``-1`` for nulls); unknown / PAD codes are hard errors.

    Resolved through :meth:`TargetVocabulary.boundary_code_to_index`, which is the base map when no
    ontology is attached and the base map plus the ancestor nodes when one is: a *supplied* index may
    name an ancestor in any mode, and it must encode even in a mode that would never draw one.
    """
    c2i = vocab.boundary_code_to_index()
    distinct = codes.drop_nulls().unique().to_list()
    unknown = sorted(c for c in distinct if c not in c2i)
    if unknown:
        raise ValueError(f"{len(unknown)} {what} code(s) are not in the base vocabulary: {unknown[:10]}")
    pad = [c for c in distinct if c2i[c] == 0]
    if pad:
        raise ValueError(f"{what} code(s) at vocab index 0 (PAD) are not allowed: {pad}")
    mapping = pl.DataFrame(
        {"code": distinct, "_i": [c2i[c] for c in distinct]}, schema={"code": pl.Utf8, "_i": pl.Int64}
    )
    return (
        pl.DataFrame({"code": codes.cast(pl.Utf8)})
        .join(mapping, on="code", how="left", maintain_order="left")["_i"]
        .fill_null(-1)
        .to_numpy()
        .astype(np.int64)
    )


def _encode_bounds(
    index_df: pl.DataFrame, vocab: TargetVocabulary, num_bounds: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(start_durations (N, K) float32, start_code_index (N, K) int64 with -1 for duration starts, durations
    (N, K) float32, bound_code_index (N, K) int64 with -1 for duration slots, condition_code_index (N, K-1)
    int64)``.

    ``index_df`` must carry the start columns.
    """
    n = index_df.height

    def days(col: str) -> np.ndarray:
        return np.asarray(index_df[col].explode().to_numpy(), dtype=np.float32).reshape(n, num_bounds)

    start_durations = days(START_DURATIONS_COL)
    starts = _map_codes(index_df[START_EVENTS_COL].explode(), vocab, "start_event").reshape(n, num_bounds)
    durations = days(DURATIONS_COL)
    bounds = _map_codes(index_df[BOUND_EVENTS_COL].explode(), vocab, "boundary").reshape(n, num_bounds)
    if num_bounds > 1:
        conds = _map_codes(index_df[CONDITION_CODES_COL].explode(), vocab, "condition")
        conds = conds.reshape(n, num_bounds - 1)
    else:
        conds = np.zeros((n, 0), dtype=np.int64)
    return start_durations, starts, durations, bounds, conds


@dataclass
class LabelStats:
    n_events: int = 0
    n_unknown_code_events: int = 0
    n_intervals: int = 0
    # Ontology-only (zero without one): the closure-expanded event count and the extended interval
    # table's size, the two numbers that say what the ancestor boundaries cost in RAM this shard.
    n_query_node_events: int = 0
    n_event_table_intervals: int = 0
    n_ancestor_condition_slots: int = 0
    n_contexts: int = 0
    vocab_size: int = 0
    packed_width: int = 0
    num_bounds: int = 0
    build_seconds: float = 0.0
    bound_seconds: float = 0.0
    label_seconds: float = 0.0
    contexts_per_second: float = 0.0
    n_event_bounds: int = 0
    frac_event_bounds_inf: float = 0.0
    n_event_starts: int = 0
    frac_event_starts_unresolved: float = 0.0
    frac_empty_windows: float = 0.0
    mean_positives_per_window: float = 0.0
    mean_positives_per_context_boundary: float = 0.0
    output_bytes: int = 0
    peak_rss_bytes: int | None = None

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _peak_rss_bytes() -> int | None:
    try:
        import resource

        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(kb) * 1024
    except Exception:  # pragma: no cover - platform without getrusage
        return None


def label_multitask_index(
    index_df: pl.DataFrame,
    events_df: pl.DataFrame,
    vocab: TargetVocabulary,
    num_bounds: int,
    *,
    chunk_rows: int = 2000,
    out: np.ndarray | None = None,
    ontology_dir: object = None,
    window_times_out: dict[str, np.ndarray] | None = None,
) -> tuple[pl.DataFrame, np.ndarray, LabelStats]:
    """Stage 4M kernel: label a (possibly supplied) index against an event stream.

    Returns ``(metadata_df, packed, stats)``: ``metadata_df`` is the index sorted by
    ``(subject_id, prediction_time, _ctx_id)`` and reduced to :data:`METADATA_COLUMNS`; ``packed`` is
    the row-aligned ``(N, K, ceil(V/8))`` uint8 target array (``out`` when supplied - typically a
    writable memmap - else a freshly allocated packed array).  No dense ``(N, K, V)`` tensor is ever
    built; scratch is ``chunk_rows x K x V`` booleans.

    Issue #24: the ``(N, K)`` starts are resolved first, the ends relative to them, and the ``N x K``
    windows are then flattened, sorted by ``(subject_id, resolved_start)`` (stable over the row-major
    ``(context_row, k)`` order) and labeled as one shared stream of ``K' = 1`` rows through
    :func:`~every_query.generate_tasks.interval_table.iter_packed_label_chunks` with ``chunk_rows * K``
    rows per chunk (the same ``C x K x V`` scratch), each packed chunk being scattered back into
    ``out[context_row, k]``.  An index without start columns labels with prediction-time starts.

    Sampling policy never enters here: everything about a window is fixed by the index rows.

    Diagnostics seam: pass a dict as ``window_times_out`` to receive the resolved ``(N, K)``
    ``start_times`` / ``end_times`` matrices (int64 microseconds, ``INF`` where a boundary event never
    occurred), row-aligned with the returned metadata frame.  They are what separates "the window was
    a real bounded interval" from "the boundary never occurred, so the window ran to the end of the
    timeline (or stayed empty)", which no stored label bit records.  Default ``None`` costs nothing.

    Examples:
        >>> from datetime import datetime
        >>> vocab = TargetVocabulary.from_pairs(["A", "B", "END"], [1, 2, 3])
        >>> events = pl.DataFrame({"subject_id": [1, 1, 1],
        ...     "time": [datetime(2024, 1, 3), datetime(2024, 1, 6), datetime(2024, 1, 9)],
        ...     "code": ["A", "B", "END"]}).with_columns(pl.col("time").cast(pl.Datetime("us")))
        >>> idx = pl.DataFrame({"subject_id": [1], "prediction_time": [datetime(2024, 1, 1)],
        ...     "durations": [[30.0, -1.0]], "bound_events": [[None, "B"]],
        ...     "condition_codes": [["END"]]}).with_columns(
        ...     pl.col("prediction_time").cast(pl.Datetime("us")),
        ...     pl.col("durations").cast(pl.List(pl.Float32)),
        ...     pl.col("bound_events").cast(pl.List(pl.Utf8)))
        >>> meta, packed, stats = label_multitask_index(idx, events, vocab, 2)
        >>> np.unpackbits(packed, axis=-1, count=vocab.size, bitorder="little").tolist()
        [[[0, 1, 1, 1], [0, 1, 0, 0]]]

        The conditioning answer is the target bit of the conditioning code at the matching boundary:

        >>> meta["condition_answers"].to_list()
        [[True]]

        With an event-defined start the window opens at that event and the end is relative to it: from
        the first ``A`` (Jan 3) for 4 days, and from the first ``A`` to the next ``END``:

        >>> idx = idx.with_columns(pl.Series("start_durations", [[-1.0, -1.0]], dtype=pl.List(pl.Float32)),
        ...     pl.Series("start_events", [["A", "A"]], dtype=pl.List(pl.Utf8)),
        ...     pl.Series("durations", [[4.0, -1.0]], dtype=pl.List(pl.Float32)),
        ...     pl.Series("bound_events", [[None, "END"]], dtype=pl.List(pl.Utf8)))
        >>> _, packed, _ = label_multitask_index(idx, events, vocab, 2)
        >>> np.unpackbits(packed, axis=-1, count=vocab.size, bitorder="little").tolist()
        [[[0, 0, 1, 0], [0, 0, 1, 0]]]
    """
    if (ontology_dir is None) != (vocab.ontology_mode == ONTOLOGY_MODE_NONE):
        raise ValueError(
            f"ontology_dir={ontology_dir!r} disagrees with the vocabulary's ontology_mode="
            f"{vocab.ontology_mode!r}; build the vocabulary with build_target_vocabulary(..., "
            "ontology_dir) and pass the same directory here."
        )
    validate_index(index_df, num_bounds)
    stats = LabelStats(vocab_size=vocab.size, packed_width=vocab.packed_width, num_bounds=num_bounds)

    index_df = sort_index_for_labeling(normalize_index(index_df, num_bounds))
    n = index_df.height
    stats.n_contexts = n

    t0 = time.perf_counter()
    n_unlabelable = 0
    if ontology_dir is not None:
        events_df, n_unlabelable = restrict_to_labelable_events(events_df, vocab)
    events_df = prepare_events_for_labeling(events_df, ontology_dir)
    ev_sid, ev_t, ev_ci, n_unknown = _encode_events(events_df, vocab)
    stats.n_unknown_code_events = n_unknown + n_unlabelable
    # Two tables under an ontology, one without.  ``table`` is leaf-only and is the ONLY one that ever
    # labels a bit, so the packed output cannot depend on the expansion; ``event_table`` spans
    # [0, V_ext) and exists purely so an ancestor id resolves as a start / bound / conditioning event.
    #
    # Two tables rather than one masked table: the labeling kernel has no notion of a code ceiling -
    # ``dense[rows, :, codes] = ...`` would IndexError on the first ancestor interval that contains a
    # lookup time - so a single V_ext table would need a change inside ``interval_table.py``, whose
    # other caller builds its own code universe.  The cost is the leaf table's extra rows (~32 B
    # each) beside the expanded ones; the labeling scratch and the packed output are leaf-wide either
    # way.  What the split buys is that byte-identity is *structural*: the closure pairs every leaf
    # with itself, so the ``code_index < V`` rows are exactly the unexpanded stream.
    if vocab.boundary_size == vocab.size:
        table = event_table = build_interval_table(ev_sid, ev_t, ev_ci, vocab_size=vocab.size)
        stats.n_events = int(ev_sid.size)
    else:
        leaf = ev_ci < vocab.size
        table = build_interval_table(ev_sid[leaf], ev_t[leaf], ev_ci[leaf], vocab_size=vocab.size)
        event_table = build_interval_table(ev_sid, ev_t, ev_ci, vocab_size=vocab.boundary_size)
        stats.n_events = int(leaf.sum())
        stats.n_query_node_events = int(ev_sid.size)
        stats.n_event_table_intervals = event_table.n_rows
        del leaf
    del ev_sid, ev_t, ev_ci
    stats.n_intervals = table.n_rows
    stats.build_seconds = time.perf_counter() - t0

    subject_ids = index_df[SID].to_numpy().astype(np.int64)
    prediction_times = index_df[PT].cast(pl.Datetime("us")).cast(pl.Int64).to_numpy().astype(np.int64)
    start_durations, start_code_index, durations, bound_code_index, condition_index = _encode_bounds(
        index_df, vocab, num_bounds
    )

    t1 = time.perf_counter()
    start_times, end_times = resolve_event_boundaries(
        event_table,
        subject_ids,
        prediction_times,
        start_durations,
        start_code_index,
        durations,
        bound_code_index,
        ontology_dir,
    )
    stats.bound_seconds = time.perf_counter() - t1
    is_event_start = start_code_index >= 0
    resolved = start_times != INF
    stats.n_event_starts = int(is_event_start.sum())
    stats.frac_event_starts_unresolved = (
        float((~resolved[is_event_start]).mean()) if is_event_start.any() else 0.0
    )
    is_event = (bound_code_index >= 0) & resolved  # only windows whose start resolved
    stats.n_event_bounds = int((bound_code_index >= 0).sum())
    stats.frac_event_bounds_inf = float((end_times[is_event] == INF).mean()) if is_event.any() else 0.0
    if window_times_out is not None:
        window_times_out["start_times"] = start_times.copy()
        window_times_out["end_times"] = end_times.copy()

    # Conditioning answers for ANCESTOR codes (PR D).  A leaf answer is read straight off the packed
    # row in the chunk loop below, but an ancestor has no packed column, so its answer is resolved
    # here from the expanded table while the window matrices are still alive: the bit is "some
    # occurrence of the node falls strictly inside (start, end)", i.e. the first occurrence strictly
    # after the start is strictly before the end.  On the expanded stream that is the same OR over
    # descendant leaves the model's ``derive_ancestor_targets`` computes - one searchsorted per slot
    # instead of a V_ext-wide dense chunk.
    kc = num_bounds - 1
    ancestor_answers: np.ndarray | None = None
    is_ancestor_condition = condition_index >= vocab.size
    if kc and is_ancestor_condition.any():
        stats.n_ancestor_condition_slots = int(is_ancestor_condition.sum())
        rows = np.nonzero(is_ancestor_condition)[0]
        first = next_occurrence_after(
            event_table,
            subject_ids[rows],
            start_times[:, :kc][is_ancestor_condition],
            condition_index[is_ancestor_condition],
        )
        ancestor_answers = np.zeros((n, kc), dtype=bool)
        ancestor_answers[is_ancestor_condition] = first < end_times[:, :kc][is_ancestor_condition]
        del rows, first
    del start_durations, start_code_index, durations, bound_code_index, resolved, is_event, is_event_start

    shape = (n, num_bounds, vocab.packed_width)
    if out is None:
        out = np.zeros(shape, dtype=np.uint8)
    elif tuple(out.shape) != shape or out.dtype != np.uint8:
        raise ValueError(f"out must be uint8 with shape {shape}, got {out.dtype} {out.shape}")

    # Flatten the (N, K) windows to N*K logical rows (row-major: flat = ctx_row * K + k) and sort them
    # by (subject_id, resolved_start); the stable lexsort keeps (ctx_row, k) order among ties.  Only
    # the sorted copies survive: int32 context row + int32 window position per flattened row (a uint8
    # position would silently wrap at num_bounds >= 256 and scatter a window's bits onto another slot).
    subj_flat = np.repeat(subject_ids, num_bounds)
    order = np.lexsort((start_times.ravel(), subj_flat))
    ctx_sorted = (order // num_bounds).astype(np.int32)
    k_sorted = (order % num_bounds).astype(np.int32)
    subj_sorted = subj_flat[order]
    start_sorted = start_times.ravel()[order]
    end_sorted = end_times.ravel()[order][:, None]
    del subj_flat, order, start_times, end_times

    t2 = time.perf_counter()
    positives = 0
    empty_windows = 0
    answers = np.zeros((n, kc), dtype=bool)
    for lo, hi, packed in iter_packed_label_chunks(
        table, subj_sorted, start_sorted, end_sorted, vocab.size, chunk_rows * num_bounds
    ):
        rows, ks = ctx_sorted[lo:hi], k_sorted[lo:hi]
        packed = packed[:, 0, :]  # K' == 1: one window per flattened row
        out[rows, ks] = packed  # scatter back to (ctx_row, k); every pair occurs exactly once
        counts = np.bitwise_count(packed).sum(axis=-1)  # popcount on packed bytes: no unpacked scratch
        positives += int(counts.sum())
        empty_windows += int((counts == 0).sum())
        # answers[i, j] = targets[i, j, condition_index[i, j]] for j < K-1, read straight off the packed
        # bytes PER flattened row: after the start-sort a context's K windows may straddle chunks.
        # Ancestor slots have no packed column (their byte offset would run past packed_width); they
        # were resolved from the expanded table above and are written in after the loop.
        sel = np.flatnonzero(ks < kc)
        if sel.size:
            r, j = rows[sel], ks[sel]
            if ancestor_answers is not None:
                keep = ~is_ancestor_condition[r, j]
                sel, r, j = sel[keep], r[keep], j[keep]
        if sel.size:
            ci = condition_index[r, j]
            answers[r, j] = (packed[sel, ci >> 3] >> (ci & 7)) & 1
    if ancestor_answers is not None:
        answers[is_ancestor_condition] = ancestor_answers[is_ancestor_condition]
    stats.label_seconds = time.perf_counter() - t2
    stats.contexts_per_second = n / stats.label_seconds if stats.label_seconds > 0 else float("inf")
    n_windows = n * num_bounds
    stats.mean_positives_per_window = positives / n_windows if n else 0.0
    stats.mean_positives_per_context_boundary = stats.mean_positives_per_window
    stats.frac_empty_windows = empty_windows / n_windows if n else 0.0
    stats.output_bytes = int(np.prod(shape))
    stats.peak_rss_bytes = _peak_rss_bytes()

    metadata = index_df.select(METADATA_COLUMNS).with_columns(
        _list_column(CONDITION_ANSWERS_COL, answers, pl.Boolean)
    )
    return metadata, out, stats


# ---------------------------------------------------------------------------
# Stage 4M - per-shard worker with atomic sidecars and fingerprint-keyed reuse
# ---------------------------------------------------------------------------


def labels_path(out_dir: Path, shard: str) -> Path:
    return out_dir / f"{shard}{LABELS_SUFFIX}"


def _clean_stale_multitask_temps(out_dir: Path, shard: str) -> int:
    removed = 0
    for pattern in (f".{shard}.parquet.tmp.*", f".{shard}{LABELS_SUFFIX}.tmp.*"):
        for tmp in out_dir.glob(pattern):
            tmp.unlink(missing_ok=True)
            removed += 1
    return removed


def _packed_shape(fp: Path) -> tuple[int, ...] | None:
    try:
        return tuple(np.load(fp, mmap_mode="r").shape)
    except Exception:
        return None


def output_is_reusable(
    final_parquet: Path,
    final_labels: Path,
    sidecar_fp: Path,
    index_fingerprint: str,
    manifest: dict,
) -> bool:
    """The full reuse gate: both files, agreeing row counts, right packed shape, all fingerprints."""
    if not (final_parquet.exists() and final_labels.exists()):
        return False
    try:
        recorded = json.loads(sidecar_fp.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if recorded.get("index_fingerprint") != index_fingerprint:
        return False
    if recorded.get("vocab_fingerprint") != manifest["vocab_fingerprint"]:
        return False
    if recorded.get("config_fingerprint") != manifest["config_fingerprint"]:
        return False
    # The mode and the closure both decide what the windows were, so both gate reuse.  ``config_
    # fingerprint`` already covers them for a run whose manifest was written by this code, but a
    # sidecar left by an earlier run carries neither key and must not be mistaken for a match.
    if recorded.get("ontology_mode") != manifest["ontology_mode"]:
        return False
    if recorded.get("ontology_fingerprint") != manifest.get("ontology_fingerprint"):
        return False
    shape = _packed_shape(final_labels)
    if shape is None:
        return False
    try:
        n_rows = pl.scan_parquet(final_parquet).select(pl.len()).collect().item()
    except Exception:
        return False
    return shape == (n_rows, int(manifest["num_bounds"]), int(manifest["packed_width_bytes"]))


def write_labeled_shard(
    metadata: pl.DataFrame | None,
    out_dir: Path,
    shard: str,
    *,
    labels_tmp: Path | None = None,
) -> None:
    """Commit a shard: rename the flushed temp labels file, then atomically write the parquet.

    Order matters for restart safety - labels first, parquet second, fingerprint sidecar (written by
    the caller) last - so a present sidecar always describes a present, complete pair.
    """
    if labels_tmp is not None:
        os.replace(labels_tmp, labels_path(out_dir, shard))
    aligned = MultitaskBoundarySchema.align(metadata.to_arrow())
    _atomic_write_parquet(pl.from_arrow(aligned), out_dir / f"{shard}.parquet")


def label_one_multitask_shard(
    shard: str,
    index_dir: Path,
    data_dir: Path,
    out_dir: Path,
    labeled_dir: Path,
    codes_source: str,
    manifest: dict,
    overwrite: bool = False,
    chunk_rows: int = 2000,
    ontology_dir: str | None = None,
) -> tuple[str, str, dict]:
    """Stage 4M worker: label one index partition; write ``{shard}.labels.npy`` + ``{shard}.parquet``.

    Module-level so it pickles under ``spawn``.  Reads the manifest the driver already wrote (it never
    writes one).  Returns ``(shard, status, stats)`` with status ``"skipped"`` or ``"labeled"``.

    ``ontology_dir`` must agree with the manifest's ``ontology_mode``: the worker rebuilds the
    vocabulary from ``codes_source`` and re-attaches the ontology itself (nothing ontology-shaped
    crosses the process boundary), then checks the closure it loaded against the fingerprint the
    driver recorded, so a worker pointed at a different ontology fails instead of mislabeling.
    """
    validate_manifest(manifest)
    num_bounds = int(manifest["num_bounds"])
    final_parquet = out_dir / f"{shard}.parquet"
    final_labels = labels_path(out_dir, shard)
    sidecar_fp = labeled_dir / f"{shard}.json"

    index_df = pl.read_parquet(index_dir / f"{shard}.parquet")
    current_fingerprint = _index_fingerprint(index_df)

    if not overwrite and output_is_reusable(
        final_parquet, final_labels, sidecar_fp, current_fingerprint, manifest
    ):
        return shard, "skipped", {}

    _clean_stale_multitask_temps(out_dir, shard)
    mode = str(manifest["ontology_mode"])
    if (ontology_dir is None) != (mode == ONTOLOGY_MODE_NONE):
        raise ValueError(
            f"shard {shard}: ontology_dir={ontology_dir!r} disagrees with the manifest's "
            f"ontology_mode={mode!r}"
        )
    vocab = build_target_vocabulary(codes_source, ontology_dir, None if ontology_dir is None else mode)
    if vocab.fingerprint != manifest["vocab_fingerprint"] or vocab.size != int(manifest["vocab_size"]):
        raise ValueError(
            f"shard {shard}: the vocabulary at {codes_source} (size {vocab.size}, {vocab.fingerprint[:12]}) "
            f"does not match the manifest (size {manifest['vocab_size']}, "
            f"{manifest['vocab_fingerprint'][:12]})"
        )
    if vocab.ontology_fingerprint != manifest.get("ontology_fingerprint"):
        raise ValueError(
            f"shard {shard}: the ontology at {ontology_dir} has closure "
            f"{(vocab.ontology_fingerprint or '-')[:12]} but the manifest was written against "
            f"{(manifest.get('ontology_fingerprint') or '-')[:12]}"
        )
    if vocab.boundary_size != int(manifest.get("boundary_vocab_size") or manifest["vocab_size"]):
        raise ValueError(
            f"shard {shard}: the ontology at {ontology_dir} is {vocab.boundary_size} wide but the "
            f"manifest records boundary_vocab_size {manifest.get('boundary_vocab_size')}"
        )

    events_df = _read_event_shard(data_dir / f"{shard}.parquet")

    shape = (index_df.height, num_bounds, vocab.packed_width)
    labels_tmp = _unique_tmp_path(final_labels)
    try:
        if index_df.height == 0:
            metadata = (
                sort_index_for_labeling(normalize_index(index_df, num_bounds))
                .select(METADATA_COLUMNS)
                .with_columns(pl.Series(CONDITION_ANSWERS_COL, [], dtype=pl.List(pl.Boolean)))
            )
            # np.save would append '.npy' to a suffix-less temp name; write through the handle.
            with open(labels_tmp, "wb") as f:
                np.save(f, np.zeros(shape, dtype=np.uint8))
            stats = LabelStats(vocab_size=vocab.size, packed_width=vocab.packed_width, num_bounds=num_bounds)
        else:
            mm = np.lib.format.open_memmap(labels_tmp, mode="w+", dtype=np.uint8, shape=shape)
            try:
                metadata, _, stats = label_multitask_index(
                    index_df,
                    events_df,
                    vocab,
                    num_bounds,
                    chunk_rows=chunk_rows,
                    out=mm,
                    ontology_dir=ontology_dir,
                )
                mm.flush()
            finally:
                del mm
        write_labeled_shard(metadata, out_dir, shard, labels_tmp=labels_tmp)
    except Exception:
        labels_tmp.unlink(missing_ok=True)
        raise

    _atomic_write_json(
        {
            "index_fingerprint": current_fingerprint,
            "vocab_fingerprint": vocab.fingerprint,
            "config_fingerprint": manifest["config_fingerprint"],
            "ontology_mode": vocab.ontology_mode,
            "ontology_fingerprint": vocab.ontology_fingerprint,
            "n_rows": index_df.height,
            "stats": stats.as_dict(),
        },
        sidecar_fp,
    )
    return shard, "labeled", stats.as_dict()


def _log_shard_stats(shard: str, stats: dict) -> None:
    if not stats:
        return
    rss = stats.get("peak_rss_bytes")
    logger.info(
        "Stage 4M shard %s: events=%s (unknown-code dropped=%s) intervals=%s contexts=%s V=%s "
        "packed_width=%s "
        "build=%.2fs bounds=%.2fs label+pack=%.2fs (%.0f ctx/s) event_starts=%s unresolved_frac=%.3f "
        "event_bounds=%s inf_frac=%.3f empty_windows_frac=%.3f mean_pos/window=%.2f output=%s bytes "
        "peak_rss=%s",
        shard,
        f"{stats['n_events']:,}",
        f"{stats['n_unknown_code_events']:,}",
        f"{stats['n_intervals']:,}",
        f"{stats['n_contexts']:,}",
        stats["vocab_size"],
        stats["packed_width"],
        stats["build_seconds"],
        stats["bound_seconds"],
        stats["label_seconds"],
        stats["contexts_per_second"],
        f"{stats['n_event_starts']:,}",
        stats["frac_event_starts_unresolved"],
        f"{stats['n_event_bounds']:,}",
        stats["frac_event_bounds_inf"],
        stats["frac_empty_windows"],
        stats["mean_positives_per_window"],
        f"{stats['output_bytes']:,}",
        f"{rss / 2**20:.0f} MiB" if rss else "n/a",
    )


def _prune_stale_multitask_outputs(out_dir: Path, labeled_dir: Path, current: set[str]) -> None:
    for fp in out_dir.glob("*.parquet"):
        if fp.stem not in current:
            fp.unlink()
    for fp in out_dir.glob(f"*{LABELS_SUFFIX}"):
        shard = fp.name[: -len(LABELS_SUFFIX)]
        if shard not in current:
            fp.unlink()
    if labeled_dir.exists():
        for fp in labeled_dir.glob("*.json"):
            if fp.stem not in current:
                fp.unlink()


def _label_multitask_shards(
    shards: list[str],
    index_dir: Path,
    data_dir: Path,
    out_dir: Path,
    labeled_dir: Path,
    codes_source: str,
    manifest: dict,
    overwrite: bool,
    n_workers: int,
    chunk_rows: int,
    ontology_dir: str | None = None,
) -> dict[str, str]:
    """Fan one worker per shard through a ``spawn`` pool (fork would inherit polars' locked threads)."""
    mp_context = multiprocessing.get_context("spawn")
    statuses: dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_context) as ex:
        futs = {
            ex.submit(
                label_one_multitask_shard,
                s,
                index_dir,
                data_dir,
                out_dir,
                labeled_dir,
                codes_source,
                manifest,
                overwrite,
                chunk_rows,
                ontology_dir,
            ): s
            for s in shards
        }
        for fut in as_completed(futs):
            shard, status, stats = fut.result()
            statuses[shard] = status
            _log_shard_stats(shard, stats)
    return statuses


def _validate_context_count(out_dir: Path, expected: int, manifest: dict) -> int:
    written = 0
    for fp in sorted(out_dir.glob("*.parquet")):
        n = pl.scan_parquet(fp).select(pl.len()).collect().item()
        shape = _packed_shape(labels_path(out_dir, fp.stem))
        want = (n, int(manifest["num_bounds"]), int(manifest["packed_width_bytes"]))
        if shape != want:
            raise ValueError(f"{fp.stem}: packed labels have shape {shape}, expected {want}")
        written += n
    if written != expected:
        raise ValueError(
            f"Expected {expected:,} labeled contexts but found {written:,} across {out_dir}. "
            "The output directory may hold a partial run; rerun with overwrite=true."
        )
    return written


def label_multitask_shards(
    cfg: DictConfig,
    path_to_data: Path,
    out_root: Path,
    artifacts_dir: Path,
    manifest: dict,
    total_contexts: int,
) -> int:
    """Stage 4M driver half: prune stale outputs, fan out workers, validate the final row count."""
    index_dir = artifacts_dir / cfg.split / INDEX_DIRNAME
    labeled_dir = artifacts_dir / cfg.split / LABELED_DIRNAME
    data_dir = path_to_data / "data" / cfg.split
    out_dir = out_root / cfg.split
    out_dir.mkdir(parents=True, exist_ok=True)
    labeled_dir.mkdir(parents=True, exist_ok=True)

    shards = sorted(p.stem for p in index_dir.glob("*.parquet"))
    _prune_stale_multitask_outputs(out_dir, labeled_dir, set(shards))

    n_workers = resolve_workers(cfg.get("max_workers"))
    chunk_rows = int(cfg.get("label_chunk_rows", 2000))
    # Taken from the manifest, not the config: the manifest is what the workers validate against, and
    # a resumed run must label with the ontology its existing sidecars were written under.
    ontology_dir = None if manifest["ontology_mode"] == ONTOLOGY_MODE_NONE else str(cfg.ontology_dir)
    logger.info(
        "Stage 4M: labeling %s shard(s) across %s worker(s), chunk_rows=%s (scratch ~%.0f MiB/worker).",
        f"{len(shards):,}",
        f"{n_workers:,}",
        f"{chunk_rows:,}",
        chunk_rows * int(manifest["num_bounds"]) * int(manifest["vocab_size"]) / 2**20,
    )
    statuses = _label_multitask_shards(
        shards,
        index_dir,
        data_dir,
        out_dir,
        labeled_dir,
        str(cfg.query_codes),
        manifest,
        bool(cfg.overwrite),
        n_workers,
        chunk_rows,
        ontology_dir,
    )
    n_skipped = sum(s == "skipped" for s in statuses.values())
    written = _validate_context_count(out_dir, total_contexts, manifest)
    logger.info(
        "Pipeline complete: %s contexts x %s boundaries x %s codes in %s (%s shard(s) reused).",
        f"{written:,}",
        manifest["num_bounds"],
        f"{manifest['vocab_size']:,}",
        out_dir,
        n_skipped,
    )
    return written


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(cfg: DictConfig) -> None:
    """Execute Stages 0-4M for a fully-resolved config (no Hydra side effects)."""
    # Fail on a malformed ontology_mode / a mode without an ontology before any Stage 0 work.
    ontology_mode = resolve_ontology_mode(cfg.get("ontology_dir"), cfg.get("ontology_mode"))
    ontology_dir = None if ontology_mode == ONTOLOGY_MODE_NONE else str(cfg.get("ontology_dir"))

    path_to_data = _require_path_arg(cfg.get("data_dir"), "data_dir")
    out_root = _require_path_arg(cfg.get("out_dir"), "out_dir")
    artifacts_dir = default_artifacts_dir(out_root)

    vocab = build_target_vocabulary(cfg.get("query_codes"), ontology_dir, ontology_mode)
    boundary_codes, boundary_weights, start_event_codes, start_event_weights = resolve_boundary_pools(
        cfg, vocab
    )
    dist = BoundaryDistribution.from_config(
        cfg,
        boundary_codes,
        vocab.condition_candidates(),
        start_event_codes,
        boundary_weights,
        start_event_weights,
    )
    logger.info(
        "Vocabulary: %s codes, V=%s (packed width %s bytes), fingerprint %s; %s boundary candidate(s), "
        "%s start-event candidate(s); code_weighting=%s (%s^%s), excluded prefixes %s, "
        "effective boundary support %s.",
        f"{len(vocab.codes):,}",
        f"{vocab.size:,}",
        vocab.packed_width,
        vocab.fingerprint[:12],
        f"{len(boundary_codes):,}",
        f"{len(start_event_codes):,}",
        cfg.get("code_weighting"),
        cfg.get("code_weight_column", "code/n_occurrences"),
        cfg.get("code_weight_power", 1.0),
        list(read_exclude_prefixes(cfg.get("exclude_boundary_prefixes"))),
        f"{effective_support(boundary_weights, len(boundary_codes)):,.0f}",
    )

    n_subjects = build_prediction_times(
        path_to_data=path_to_data,
        training_task_artifacts_dir=artifacts_dir,
        split=cfg.split,
        min_prediction_times_per_subject=cfg.min_prediction_times_per_subject,
        overwrite=cfg.overwrite,
    )
    logger.info("Stage 0: %s eligible subject(s) for split=%s.", f"{n_subjects:,}", cfg.split)

    num_contexts = int(cfg.num_training_examples)
    context_rng = np.random.default_rng(derive_seed(cfg.seed, "contexts"))
    form_rng = np.random.default_rng(derive_seed(cfg.seed, "bound_forms"))
    duration_rng = np.random.default_rng(derive_seed(cfg.seed, "bound_durations"))
    code_rng = np.random.default_rng(derive_seed(cfg.seed, "bound_codes"))
    condition_rng = np.random.default_rng(derive_seed(cfg.seed, "condition_codes"))
    start_form_rng = np.random.default_rng(derive_seed(cfg.seed, "start_forms"))
    start_duration_rng = np.random.default_rng(derive_seed(cfg.seed, "start_durations"))
    start_code_rng = np.random.default_rng(derive_seed(cfg.seed, "start_codes"))

    sample = dist.sample(
        num_contexts,
        form_rng,
        duration_rng,
        code_rng,
        condition_rng,
        start_form_rng,
        start_duration_rng,
        start_code_rng,
    )
    n_slots = max(num_contexts * dist.num_bounds, 1)
    n_event = int((sample.bound_events != None).sum())  # noqa: E711
    n_event_start = int((sample.start_events != None).sum())  # noqa: E711
    n_pt_start = int((sample.start_durations == 0).sum())
    logger.info(
        "Stage 1M: sampled %s window sequence(s) of K=%d (ends: %s event-bounded slots, %.1f%%; %s "
        "durations over [%g, %g] days; starts: %s event-defined, %.1f%%, %s at the prediction time, "
        "%.1f%%, the rest %s durations over [%g, %g] days) + %d conditioning code(s) per context from %s "
        "candidate(s).",
        f"{num_contexts:,}",
        dist.num_bounds,
        f"{n_event:,}",
        100.0 * n_event / n_slots,
        dist.duration_distribution,
        dist.min_duration,
        dist.max_duration,
        f"{n_event_start:,}",
        100.0 * n_event_start / n_slots,
        f"{n_pt_start:,}",
        100.0 * n_pt_start / n_slots,
        dist.start_duration_distribution,
        dist.start_min_duration,
        dist.start_max_duration,
        dist.num_bounds - 1,
        f"{len(dist.condition_codes):,}",
    )

    counts = pl.read_parquet(prediction_time_counts_path(artifacts_dir, cfg.split)).sort("subject_id")
    contexts = sample_patient_contexts(
        prediction_time_counts=counts,
        n=num_contexts,
        min_prediction_times_per_subject=cfg.min_prediction_times_per_subject,
        rng=context_rng,
    )
    logger.info("Stage 2: sampled %s patient context(s).", f"{contexts.height:,}")

    n_shards = build_multitask_index(sample, contexts, artifacts_dir, cfg.split)
    logger.info(
        "Stage 3M: wrote partitioned index for split=%s across %s shard(s).", cfg.split, f"{n_shards:,}"
    )

    # The driver - never a worker - owns the split manifest, written and validated before the pool.
    manifest = write_manifest(out_root / cfg.split, build_manifest(dist, vocab))
    label_multitask_shards(cfg, path_to_data, out_root, artifacts_dir, manifest, num_contexts)


CONFIGS = str(files("every_query") / "generate_tasks" / "configs")


@hydra.main(version_base=None, config_path=CONFIGS, config_name="sample_multitask_sequences_config")
def main(cfg: DictConfig) -> None:
    """Hydra entry point (``EQ_generate_multitask_sequences``)."""
    run(cfg)


if __name__ == "__main__":
    main()
