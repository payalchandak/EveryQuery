"""Query-*sequence* window labelling, and the sampling pipeline built on top of it.

**This module is a library, not a CLI.**  It has no ``[project.scripts]`` entry point; the name
says what the rest of the repo imports it for.  Its labellers are the one place the
``(start, end]`` window semantics are implemented, and three surviving callers depend on them:

- :mod:`~every_query.generate_tasks.sample_evaluation_query_sequences` (the evaluation-grid
  generator) routes every grid through :func:`label_query_sequences` and draws its query universe
  and ontology expansion from here;
- :mod:`~every_query.generate_tasks.sample_multitask_sequences` takes
  :func:`resolve_prediction_times` from here, and its ``(N, K, V)`` interval-table labeller is
  differentially tested against :func:`label_with_event_bounds` as the scalar oracle;
- :mod:`every_query.data.ontology`'s production probe and the window-bounds contract test pin the
  labellers directly.

The Hydra ``run``/``main`` tail below is kept for one further reason: it is the *training-pipeline
arm* of the sampler differential in ``tests/sampler/test_sequence_orchestration.py``, which runs
one designed spec table through both the dense evaluation entry point and
:func:`label_one_sequence_shard` and asserts they label DAG boundaries and ancestor targets
identically.  The multitask sampler cannot stand in for that arm — it accepts no supplied window
table and its targets are leaf-only by design — so deleting the tail would delete one side of a
differential rather than move it.  Its config, ``configs/sample_query_sequences_config.yaml``, is
kept with it, and stays key-identical to the evaluation config (a test asserts it).

Sibling of :mod:`~every_query.generate_tasks.sample_tasks`, running the *same* 5-stage pipeline but
emitting, per sampled patient context, an ordered *sequence* of queries rather than one scattered
``(code, duration)``::

    Stage 0   build + cache the canonical prediction-time map and subject counts (reused as-is)
    Stage 1'  sample ``num_training_sequence_examples`` variable-length query sequences
              (QuerySequenceDistribution), each query either horizon-bounded or event-bounded
    Stage 2   sample ``num_training_sequence_examples`` patient contexts (reused as-is)
    Stage 3'  resolve ``prediction_time_index -> prediction_time``, zip, write per-shard index
    Stage 4'  binary-label each index shard independently and write the final dataset parquet

Stages 0-3' run sequentially in the driver (:func:`run`); Stage 4' fans out one
:func:`label_one_sequence_shard` worker per shard via ``ProcessPoolExecutor``.  Only the primed
stages differ from ``sample_tasks``: Stage 1' draws ``L ~ Uniform{min_queries..max_queries}``
queries per sequence instead of one, and Stage 3' tags each row with ``_ctx_id`` / ``_position`` so
Stage 4' can reassemble the list columns.  Stage 3' does no sampling of its own: every property of
a query - code, horizon or boundary - is fixed by Stage 1'.

Every query is an ordinary node of the query universe drawn uniformly - including the
end-of-timeline code ``TIMELINE//END`` and, with an ontology, ancestor nodes.  There is **no**
privileged censor position and **no** null answer: censoring is represented explicitly as a
``TIMELINE//END`` query whose answer is ``True`` exactly when the record ends inside the window.
The model trained on these sequences learns ``P(A_j | patient, Q_1..Q_j, A_1..A_{j-1})``.

Output rows follow :class:`~every_query.data.schema.QuerySeqSchema`.

Two optional, default-off sweep knobs reweight sequence *structure* without touching the base
code/duration draw: ``eos_first_fraction`` (force position 0 to ``TIMELINE//END``) and
``duration_mode`` (``random`` | ``same`` | ``nondecreasing``).  A third, ``eventbound_fraction``,
turns that share of queries into *event-bounded* ones whose window ends at the next occurrence of
a boundary node drawn from the same universe the query codes come from.

Labeling a **supplied** ``(subject_id, prediction_time)`` cohort is the eval sampler's job
(:mod:`~every_query.generate_tasks.sample_evaluation_query_sequences`, ``contexts_path=``); this
module only samples.
"""

import os

# Pin polars to a single thread BEFORE importing polars (or anything that transitively imports it -
# meds, every_query.data.schema), mirroring ``sample_tasks``.  Stage 4' workers inherit this env;
# with process-level fan-out already saturating cores, intra-op polars threads would oversubscribe.
# A transitive ``import polars`` above this line would silently defeat the setting.
os.environ.setdefault("POLARS_MAX_THREADS", "1")

import json
import logging
import multiprocessing
import shutil
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from meds import DataSchema
from omegaconf import DictConfig

from every_query.data.query_seq_dataset import EOS_CODE, EVENT_BOUND_DURATION_SENTINEL
from every_query.data.schema import QuerySeqSchema, TaskQuerySchema
from every_query.generate_tasks.sample_tasks import (
    INDEX_DIRNAME,
    LABELED_DIRNAME,
    QueryDistribution,
    QuerySpec,
    _atomic_write_json,
    _atomic_write_parquet,
    _clean_stale_temps,
    _index_fingerprint,
    _prune_stale_outputs,
    _read_event_shard,
    _require_path_arg,
    build_prediction_times,
    default_artifacts_dir,
    index_path,
    prediction_time_counts_path,
    prediction_times_path,
    read_query_codes,
    resolve_workers,
    sample_patient_contexts,
)
from every_query.utils.seeds import derive_seed

logger = logging.getLogger(__name__)

CTX_ID_COL = "_ctx_id"
POSITION_COL = "_position"

INDEX_COLUMNS = [
    CTX_ID_COL,
    POSITION_COL,
    TaskQuerySchema.subject_id_name,
    TaskQuerySchema.prediction_time_name,
    TaskQuerySchema.query_name,
    TaskQuerySchema.duration_days_name,
]

# Per-query boundary code in the flat index frame; becomes the ``bound_events`` list column.
BOUND_COL = "bound_event"

# Per-query explicit window start in the flat index frame (issue #27): days after the prediction
# time (``0.0`` = the prediction time itself, ``EVENT_BOUND_DURATION_SENTINEL`` = event-defined) and
# the start-event code (null for a duration start).  They become the ``start_durations`` /
# ``start_events`` list columns and are present together or absent together.  Only the dense
# evaluation grid writes them; the training sampler never samples a start.
START_DURATION_COL = "start_duration_days"
START_EVENT_COL = "start_event"
START_INDEX_COLS = (START_DURATION_COL, START_EVENT_COL)

# The window semantics the explicit-start labeler implements, spelled with the multitask sampler's
# vocabulary (``sample_multitask_sequences.WINDOW_SEMANTICS`` and friends; ``tests`` assert the two
# agree).  Stated here so the contract is visible where the scalar labeler lives.
WINDOW_SEMANTICS = "open_open"
START_REFERENCE = "prediction_time"
DURATION_END_REFERENCE = "resolved_start"
MISSING_EVENT_START = "empty_window"
MISSING_EVENT_BOUNDARY = "infinity"


# ---------------------------------------------------------------------------
# Stage 1' - the sequence query distribution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuerySequenceDistribution(QueryDistribution):
    """Stage 1' - draw variable-length *sequences* of :class:`QuerySpec` s.

    Extends :class:`~every_query.generate_tasks.sample_tasks.QueryDistribution` rather than
    reimplementing it: the base ``sample`` call is what actually draws every code and duration, so
    a sequence run and a training-task run with the same ``rng`` and the same total query count
    produce **identical** ``(code, duration_days)`` draws.  That equality is the distribution-parity
    contract between the two samplers (durations are floats here too - no day-rounding).

    Sequence *structure* - how many queries each sequence gets, and the two sweep reweightings -
    is drawn from a **separate** generator (``structure_rng``) so it never perturbs the base draw;
    event bounds come from a third (``bound_rng``) for the same reason.

    Args:
        min_queries: Minimum queries per sequence (``>= 1``).
        max_queries: Maximum queries per sequence (``>= min_queries``).
        eos_first_fraction: Probability a sequence's position 0 is forced to the end-of-timeline
            code ``TIMELINE//END`` (upweights the censoring-control pattern).  ``0.0`` = pure random.
        duration_mode: Within-sequence duration coupling.  ``"random"`` (independent, default);
            ``"same"`` (every query reuses the sequence's first duration); ``"nondecreasing"``
            (durations sorted ascending) - the latter two reflect that many censoring-style asks
            reuse one horizon.
        eventbound_fraction: Probability, per query, that it is *event-bounded*: its window ends at
            the next occurrence of a boundary node instead of after a horizon.  The boundary is
            drawn uniformly from ``query_codes`` - the same universe the query codes come from, so
            any queryable node (a leaf, or with an ontology an ancestor such as ``X//ANY``) can
            bound - and the query's ``duration_days`` becomes ``EVENT_BOUND_DURATION_SENTINEL``.
            ``0.0`` = off.

    Examples:
        >>> import numpy as np
        >>> dist = QuerySequenceDistribution(
        ...     ["A", "B", EOS_CODE], min_duration=1.0, max_duration=365.0,
        ...     duration_distribution="log-uniform", min_queries=1, max_queries=4)
        >>> seqs = dist.sample_sequences(5, np.random.default_rng(0), np.random.default_rng(1))
        >>> len(seqs)
        5
        >>> all(1 <= len(s) <= 4 for s in seqs)
        True
        >>> all(q.code in {"A", "B", EOS_CODE} for s in seqs for q in s)
        True

        Parity with the base distribution - the flattened draw is identical to what
        ``QueryDistribution.sample`` yields for the same generator and total:

        >>> total = sum(len(s) for s in seqs)
        >>> base = QueryDistribution(["A", "B", EOS_CODE], 1.0, 365.0, "log-uniform")
        >>> [q for s in seqs for q in s] == base.sample(total, np.random.default_rng(0))
        True

        ``eos_first_fraction=1.0`` forces every position 0:

        >>> forced = QuerySequenceDistribution(
        ...     ["A", "B", EOS_CODE], 1.0, 365.0, "log-uniform", 2, 2, eos_first_fraction=1.0)
        >>> seqs = forced.sample_sequences(4, np.random.default_rng(0), np.random.default_rng(1))
        >>> {s[0].code for s in seqs}
        {'TIMELINE//END'}

        ``duration_mode="same"`` collapses each sequence to one horizon:

        >>> same = QuerySequenceDistribution(
        ...     ["A", "B"], 1.0, 365.0, "log-uniform", 3, 3, duration_mode="same")
        >>> seqs = same.sample_sequences(3, np.random.default_rng(0), np.random.default_rng(1))
        >>> all(len({q.duration_days for q in s}) == 1 for s in seqs)
        True

        ``eventbound_fraction=1.0`` bounds every query by a node of the same universe and replaces
        its horizon with the sentinel; the codes drawn are the ones the unbounded run draws:

        >>> bounded = QuerySequenceDistribution(
        ...     ["A", "B"], 1.0, 365.0, "log-uniform", 3, 3, eventbound_fraction=1.0)
        >>> rngs = lambda: (np.random.default_rng(0), np.random.default_rng(1))
        >>> seqs = bounded.sample_sequences(4, *rngs(), np.random.default_rng(2))
        >>> {q.bound_event for s in seqs for q in s} <= {"A", "B"}
        True
        >>> {q.duration_days for s in seqs for q in s}
        {-1.0}
        >>> plain = QuerySequenceDistribution(["A", "B"], 1.0, 365.0, "log-uniform", 3, 3)
        >>> codes = lambda seqs: [q.code for s in seqs for q in s]
        >>> codes(seqs) == codes(plain.sample_sequences(4, *rngs()))
        True

        ``num_sequences=0`` is valid and returns an empty list:

        >>> dist.sample_sequences(0, np.random.default_rng(0), np.random.default_rng(1))
        []
    """

    min_queries: int = 1
    max_queries: int = 5
    eos_first_fraction: float = 0.0
    duration_mode: str = "random"
    eventbound_fraction: float = 0.0

    _VALID_DURATION_MODES = ("random", "same", "nondecreasing")

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.min_queries < 1:
            raise ValueError(f"min_queries must be >= 1 (got {self.min_queries})")
        if self.max_queries < self.min_queries:
            raise ValueError(f"max_queries ({self.max_queries}) must be >= min_queries ({self.min_queries})")
        if not 0.0 <= self.eos_first_fraction <= 1.0:
            raise ValueError(f"eos_first_fraction must be in [0, 1] (got {self.eos_first_fraction})")
        if self.duration_mode not in self._VALID_DURATION_MODES:
            raise ValueError(
                f"duration_mode must be one of {self._VALID_DURATION_MODES} (got {self.duration_mode!r})"
            )
        if not 0.0 <= self.eventbound_fraction <= 1.0:
            raise ValueError(f"eventbound_fraction must be in [0, 1], got {self.eventbound_fraction}")
        # Forcing a code that is not in the sampling universe would emit queries the model's
        # vocabulary cannot encode - a config error worth failing on rather than discovering as an
        # out-of-vocab crash three stages later.
        if self.eos_first_fraction > 0 and EOS_CODE not in self.query_codes:
            raise ValueError(
                f"eos_first_fraction={self.eos_first_fraction} forces position 0 to {EOS_CODE!r}, "
                "but that code is not in query_codes"
            )

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "QuerySequenceDistribution":
        """Build from the sequence sampler's Hydra config plus :func:`read_query_codes`.

        Reads the duration bounds from ``duration_min`` / ``duration_max`` - the conditional
        configs' historical key names, deliberately kept (rather than renamed to the base class's
        ``min_duration`` / ``max_duration``) so this config and
        ``sample_evaluation_query_sequences_config.yaml`` stay key-identical; a test asserts that
        equality, since a training/eval mismatch in the duration draw silently invalidates eval.
        """
        return cls(
            # When an ontology is configured, ancestor nodes join the *universe* the codes (and
            # boundaries) are drawn uniformly from, so the draw itself — and its parity with
            # sample_tasks — is untouched.  See build_query_universe.
            query_codes=build_query_universe(
                read_query_codes(cfg.query_codes), ontology_dir=cfg.get("ontology_dir")
            ),
            min_duration=float(cfg.duration_min),
            max_duration=float(cfg.duration_max),
            duration_distribution=str(cfg.get("duration_distribution", "log-uniform")),
            min_queries=int(cfg.min_queries),
            max_queries=int(cfg.max_queries),
            eos_first_fraction=float(cfg.get("eos_first_fraction", 0.0)),
            duration_mode=str(cfg.get("duration_mode", "random")),
            eventbound_fraction=float(cfg.get("eventbound_fraction", 0.0) or 0.0),
        )

    def sample_sequences(
        self,
        num_sequences: int,
        query_rng: np.random.Generator,
        structure_rng: np.random.Generator,
        bound_rng: np.random.Generator | None = None,
    ) -> list[list[QuerySpec]]:
        """Draw ``num_sequences`` sequences of ``L ~ Uniform{min_queries..max_queries}`` queries.

        Three caller-owned generators keep the axes independent (mirroring the redesign's
        invariant 5): ``query_rng`` feeds *only* the inherited :meth:`QueryDistribution.sample`
        (seeded via ``derive_seed(seed, "queries")``, matching the training sampler);
        ``structure_rng`` (``derive_seed(seed, "sequences")``) draws lengths and applies the two
        sweep reweightings; ``bound_rng`` (``derive_seed(seed, "bounds")``) decides which queries
        are event-bounded and by what, and is required only when ``eventbound_fraction > 0``.
        Drawing bounds from either other generator would shift every later code/duration draw
        (breaking parity with ``sample_tasks``) or perturb the structure knobs.  Draws happen in a
        fixed order, so output is deterministic for fixed generators.

        Bounds are drawn i.i.d. **per query**, not per sequence, so one sequence mixes horizon and
        event-bounded asks and the model has to read each slot to know which kind it is given.
        Because the boundary pool *is* the query pool, a query can draw itself (or, with an
        ontology, an ancestor of itself) as its boundary; such a query is unconditionally
        ``False`` under the strict bound (see :func:`label_with_event_bounds`).

        Raises:
            ValueError: If ``num_sequences < 0``, or ``eventbound_fraction > 0`` without a
                ``bound_rng``.
        """
        if num_sequences < 0:
            raise ValueError(f"num_sequences must be >= 0 (got {num_sequences})")
        if self.eventbound_fraction > 0 and bound_rng is None:
            raise ValueError(f"eventbound_fraction={self.eventbound_fraction} > 0 requires a bound_rng")
        if num_sequences == 0:
            return []

        lengths = structure_rng.integers(self.min_queries, self.max_queries + 1, size=num_sequences)
        offsets = np.concatenate([[0], np.cumsum(lengths)])

        # The single base draw - this is the parity anchor with the training sampler.
        specs = self.sample(int(offsets[-1]), query_rng)
        sequences = [list(specs[offsets[i] : offsets[i + 1]]) for i in range(num_sequences)]

        if self.eos_first_fraction > 0:
            forced = structure_rng.random(num_sequences) < self.eos_first_fraction
            for i in np.flatnonzero(forced):
                head = sequences[i][0]
                sequences[i][0] = QuerySpec(code=EOS_CODE, duration_days=head.duration_days)

        if self.duration_mode == "same":
            sequences = [[QuerySpec(q.code, s[0].duration_days) for q in s] for s in sequences]
        elif self.duration_mode == "nondecreasing":
            sequences = [
                [QuerySpec(q.code, d) for q, d in zip(s, sorted(x.duration_days for x in s), strict=True)]
                for s in sequences
            ]

        # Last, so duration_mode couples the real horizons and only then some are replaced.
        if self.eventbound_fraction > 0:
            n = int(offsets[-1])
            chosen = bound_rng.random(n) < self.eventbound_fraction
            picks = bound_rng.integers(0, self.query_universe_size, size=n)
            flat = iter(zip(chosen, picks, strict=True))
            sequences = [
                [
                    QuerySpec(q.code, EVENT_BOUND_DURATION_SENTINEL, self.query_codes[p]) if c else q
                    for q, (c, p) in zip(s, flat, strict=False)
                ]
                for s in sequences
            ]

        return sequences


def _expand_sequences(sequences: list[list[QuerySpec]]) -> pl.DataFrame:
    """Flatten sequences into one row per query: ``(_ctx_id, _position, query, duration_days)``.

    ``_ctx_id`` is the sequence's position in ``sequences`` (so it is unique across the whole run,
    not just within a shard) and ``_position`` is the query's zero-based rank inside its sequence.
    Rows come out in ``(_ctx_id, _position)`` order.

    The ``bound_event`` column is present iff some query carries a bound - keyed on the data, not
    on a flag, which is the same rule ``build_dense_sequence_index_df`` and the labeling dispatch
    in :func:`label_query_sequences` use.  A bound-free run is therefore byte-identical to one
    from before the feature existed (the doctest below relies on that).

    Examples:
        >>> seqs = [[QuerySpec("A", 1.0), QuerySpec("B", 2.0)], [QuerySpec("C", 3.0)]]
        >>> _expand_sequences(seqs).to_dicts() == [
        ...     {"_ctx_id": 0, "_position": 0, "query": "A", "duration_days": 1.0},
        ...     {"_ctx_id": 0, "_position": 1, "query": "B", "duration_days": 2.0},
        ...     {"_ctx_id": 1, "_position": 0, "query": "C", "duration_days": 3.0}]
        True
        >>> seqs = [[QuerySpec("A", 1.0), QuerySpec("B", -1.0, "X")]]
        >>> _expand_sequences(seqs)["bound_event"].to_list()
        [None, 'X']
    """
    lengths = np.array([len(s) for s in sequences], dtype=np.int64)
    if lengths.size and lengths.min() < 1:
        raise ValueError("every sequence must contain at least one query")

    ctx_ids = np.repeat(np.arange(lengths.size, dtype=np.int64), lengths)
    positions = (
        np.concatenate([np.arange(n, dtype=np.int64) for n in lengths])
        if lengths.size
        else np.empty(0, dtype=np.int64)
    )
    flat = [q for s in sequences for q in s]
    columns = {
        CTX_ID_COL: pl.Series(ctx_ids, dtype=pl.UInt32),
        POSITION_COL: pl.Series(positions, dtype=pl.Int64),
        TaskQuerySchema.query_name: pl.Series([q.code for q in flat], dtype=pl.Utf8),
        TaskQuerySchema.duration_days_name: pl.Series([q.duration_days for q in flat], dtype=pl.Float32),
    }
    if any(q.bound_event is not None for q in flat):
        columns[BOUND_COL] = pl.Series([q.bound_event for q in flat], dtype=pl.Utf8)
    return pl.DataFrame(columns)


def _attach_queries_to_contexts(contexts: pl.DataFrame, per_query: pl.DataFrame) -> pl.DataFrame:
    """Gather one context row per query row and hstack the query columns.

    A row-gather by ``_ctx_id`` (which *is* the context row index) rather than a join: it is
    positional, order-preserving, and needs no join key on a frame the size of the fan-out.  Polars
    joins carry no order guarantee, so a join here would silently scramble ``_position`` order.

    ``bound_event`` rides along when present, so an event-bounded draw survives the gather.
    """
    attached = [
        per_query[CTX_ID_COL],
        per_query[POSITION_COL],
        per_query[TaskQuerySchema.query_name],
        per_query[TaskQuerySchema.duration_days_name],
    ]
    if BOUND_COL in per_query.columns:
        attached.append(per_query[BOUND_COL])
    return contexts[per_query[CTX_ID_COL].to_numpy()].with_columns(*attached)


# Ancestors of the TIMELINE//* namespace are tautological as queries ("did any timeline event
# occur"), so they teach nothing and would inflate a macro ancestor AUROC with free positives.
_TAUTOLOGICAL_ANCESTOR_PREFIXES = ("TIMELINE",)


def build_query_universe(query_codes: Sequence[str], ontology_dir: str | Path | None = None) -> list[str]:
    """The list of distinct nodes the sampler draws query codes *and* boundaries from, uniformly.

    Without an ontology this is ``query_codes`` unchanged.  With one, every usable ancestor node
    (the ``TIMELINE`` namespace excluded, see :data:`_TAUTOLOGICAL_ANCESTOR_PREFIXES`) is appended
    **once**, after the leaves and in sorted order, so a leaf's slot index is the same as in a
    no-ontology run and every node - leaf or ancestor - is equally likely to be drawn.

    Two properties matter here and neither is negotiable:

    - *No leaf is ever dropped.*  The sampler draws codes uniformly from this list, so a code
      missing from it is a code the model never sees asked about.  An earlier version built the
      universe by sampling a fixed number of slots, which silently lost a long tail of the
      vocabulary — and, on a real cohort, lost ``TIMELINE//END`` itself, the code this model's
      entire censoring mechanism runs through.
    - *The draw is not perturbed.*  Reshaping the universe rather than biasing the draw leaves
      the code/duration RNG stream untouched, which is what keeps parity with ``sample_tasks``.

    Examples:
        Off without an ontology:

        >>> build_query_universe(["A", "B"]) == ["A", "B"]
        True
    """
    if not ontology_dir:
        return list(query_codes)

    from every_query.data.ontology import load_nodes

    leaves = list(query_codes)
    seen = set(leaves)
    nodes = load_nodes(ontology_dir)
    ancestors = sorted(nodes.filter(~pl.col("is_observed_code"))["node_name"].to_list())
    ancestors = [a for a in ancestors if not a.startswith(_TAUTOLOGICAL_ANCESTOR_PREFIXES) and a not in seen]
    if not ancestors:
        logger.warning(
            "The ontology at %s contributes no usable ancestor nodes; the query universe is unchanged.",
            ontology_dir,
        )
        return leaves

    universe = leaves + ancestors
    logger.info(
        "Ontology: query universe is %d distinct node(s) — %d leaf code(s) plus %d ancestor node(s) "
        "(%.1f%% ancestors), each drawn with equal probability.",
        len(universe),
        len(leaves),
        len(ancestors),
        100.0 * len(ancestors) / len(universe),
    )
    return universe


def resolve_prediction_times(
    contexts: pl.DataFrame,
    training_task_artifacts_dir: Path,
    split: str,
    shard: str,
) -> pl.DataFrame:
    """Resolve one shard's ``prediction_time_index`` ranks into real ``prediction_time`` stamps.

    Stage 2 returns ``(subject_id, shard, prediction_time_index)`` where the index is an ``Int64``
    *rank*, not a datetime; everything downstream needs the timestamp.  Upstream buries this join
    inside :func:`~every_query.generate_tasks.sample_tasks.build_index`, which is not reusable here
    (it zips one query per context), so it lives on its own — both :func:`build_sequence_index` and
    the evaluation-grid sampler's sampled-context branch join exactly this way.

    ``contexts`` must be a **single shard's** rows: the join reads only that shard's Stage 0
    ``_prediction_times/{shard}.parquet`` map, which keeps the driver holding one payload-free map
    at a time so memory stays flat as the shard count grows.

    Preserves the two upstream guards verbatim:

    - a **join-key dtype check** — a mismatch (e.g. ``Int64`` vs ``UInt32`` ``subject_id``) makes
      polars produce a silent all-null join, so it fails loudly instead of being cast over;
    - a **left-join-then-raise on null** ``prediction_time`` — the join is total by design (the
      contexts carry the same eligibility bound Stage 0 wrote the map with), so a null is a hard
      error, reported with a small sample of offending keys.  An inner join would silently drop the
      rows instead.

    Returns:
        ``contexts`` with the map's ``time`` column joined on and renamed to ``prediction_time``
        (``prediction_time_index`` is retained; select it away if unwanted).

    Raises:
        ValueError: On a join-key dtype mismatch, or if any row resolves to a null timestamp.
    """
    join_keys = ["subject_id", "prediction_time_index"]
    pt_map = pl.read_parquet(prediction_times_path(training_task_artifacts_dir, split, str(shard)))

    for key in join_keys:
        ctx_dtype, map_dtype = contexts.schema[key], pt_map.schema[key]
        if ctx_dtype != map_dtype:
            raise ValueError(
                f"Join key {key!r} dtype mismatch: contexts has {ctx_dtype}, the "
                f"_prediction_times map has {map_dtype}. A mismatch silently produces null "
                "prediction_times; fix the dtype upstream."
            )

    joined = contexts.join(pt_map, on=join_keys, how="left").rename(
        {"time": TaskQuerySchema.prediction_time_name}
    )

    null_rows = joined.filter(pl.col(TaskQuerySchema.prediction_time_name).is_null())
    if null_rows.height > 0:
        sample = null_rows.select(join_keys).head(5).to_dicts()
        raise ValueError(
            f"Shard {shard}: {null_rows.height} contexts have null prediction_time after join. "
            "The _prediction_times map may be stale or contexts reference invalid "
            f"(subject_id, prediction_time_index) pairs. Sample of offending rows: {sample}"
        )

    return joined


def build_sequence_index(
    sequences: list[list[QuerySpec]],
    contexts: pl.DataFrame,
    training_task_artifacts_dir: Path,
    split: str,
) -> int:
    """Stage 3': zip sequences with contexts, resolve prediction times, write partitioned index.

    The sequence analogue of :func:`~every_query.generate_tasks.sample_tasks.build_index`.  Where
    that function ``np.repeat`` s one query across ``num_contexts_per_query`` contexts, this one
    expands each context into the ``L`` queries of its sequence, tagged with ``_ctx_id`` /
    ``_position`` so Stage 4' can reassemble the list columns.  ``sequences[i]`` belongs to
    ``contexts`` row ``i`` - the two are drawn with the same length and zipped positionally.

    Nothing is sampled here.  Every property of a query - its code, its horizon, or the boundary
    node that replaces the horizon - is already fixed on the :class:`QuerySpec` by Stage 1'; this
    stage only carries it into the index (the ``bound_event`` column appears iff some query is
    event-bounded, see :func:`_expand_sequences`).

    Everything else matches ``build_index`` deliberately: the rank -> timestamp resolution runs one
    shard at a time through :func:`resolve_prediction_times` (one ``read_parquet`` per shard, so
    driver memory stays flat, with its dtype guard and raise-on-null intact), and ``sort("shard")``
    + ``group_by(maintain_order=True)`` give deterministic shard order.

    Every query of a sequence shares one context, hence one shard, so no sequence is ever split
    across index partitions.

    Args:
        sequences: Stage 1' output - one list of :class:`QuerySpec` per sequence.
        contexts: Stage 2 output - ``(subject_id, shard, prediction_time_index)``, same height.
        training_task_artifacts_dir: Intermediate-artifacts root.
        split: Dataset split name (e.g. ``"train"``).

    Returns:
        ``n_shards`` - the number of index partitions written.

    Raises:
        ValueError: If ``len(sequences) != contexts.height``, or a context fails to resolve a
            prediction time (null after join).
        AssertionError: If ``sequences`` or ``contexts`` is empty - there is no supported
            empty-budget path through this pipeline.
    """
    if len(sequences) != contexts.height:
        raise ValueError(
            f"len(sequences) ({len(sequences)}) must equal contexts.height ({contexts.height}); "
            "Stage 1' and Stage 2 are drawn with the same length and zipped positionally"
        )

    assert sequences, "sequences must be non-empty"
    assert contexts.height > 0, "contexts must be non-empty"

    per_query = _expand_sequences(sequences)
    columns = [*INDEX_COLUMNS, BOUND_COL] if BOUND_COL in per_query.columns else INDEX_COLUMNS
    combined = _attach_queries_to_contexts(contexts, per_query)

    index_dir = training_task_artifacts_dir / split / INDEX_DIRNAME
    if index_dir.exists():
        shutil.rmtree(index_dir)

    n_shards = 0
    # Sort + ``maintain_order=True`` make the order shards are processed (and logged) deterministic;
    # ``group_by`` alone is not order-preserving.  Output content is unaffected (shards are
    # independent), but the deterministic order keeps reruns and logs stable.
    combined = combined.sort("shard")
    for shard_key, shard_group in combined.group_by("shard", maintain_order=True):
        (shard_name,) = shard_key
        joined = resolve_prediction_times(shard_group, training_task_artifacts_dir, split, str(shard_name))

        _atomic_write_parquet(
            joined.select(columns).sort(CTX_ID_COL, POSITION_COL),
            index_path(training_task_artifacts_dir, split, str(shard_name)),
        )
        n_shards += 1

    return n_shards


# ---------------------------------------------------------------------------
# Stage 4' - per-shard binary labeling
# ---------------------------------------------------------------------------


def label_binary_occurrence(index_df: pl.DataFrame, events_df: pl.DataFrame) -> pl.DataFrame:
    """Label every query with a binary *observed-occurrence* answer and reassemble into list rows.

    ``answer = True`` iff an event whose code equals the query occurs strictly within
    ``(prediction_time, prediction_time + duration_days)`` **and is present in the record**.  The
    interval is **open at both ends**: an occurrence exactly at ``prediction_time`` is outside it,
    and so is one landing exactly on the horizon ``prediction_time + duration_days`` (the
    comparison below is ``<``, not ``<=``).  There
    is no censoring/null: an event we cannot observe (because the record ends first) is ``False``.
    Censoring is captured separately by the ``TIMELINE//END`` query - which, being the last event
    of each record, answers ``True`` exactly when the record ends within the window.

    The strict ``>`` lower bound is enforced by shifting the asof key ``+1us`` (microsecond-precision
    datetimes); the strict ``<`` upper bound is compared directly.  Both ends are open, so the rule
    is symmetric -- neither endpoint instant belongs to the window.  Every other window decider in
    the repo labels this same interval the same way; ``tests/test_window_bounds_contract.py`` is
    where that agreement is pinned.

    Returns:
        DataFrame ``(subject_id, prediction_time, queries, durations, answers)``, one row per
        ``_ctx_id``, lists ordered by ``_position``.

    Examples:
        >>> from datetime import datetime
        >>> events = pl.DataFrame({
        ...     "subject_id": [1, 1, 1],
        ...     "time": [datetime(2024, 1, 1), datetime(2024, 1, 5), datetime(2024, 1, 20)],
        ...     "code": ["A", "B", "TIMELINE//END"],
        ... }).with_columns(pl.col("time").cast(pl.Datetime("us")))
        >>> idx = pl.DataFrame({
        ...     "_ctx_id": [0, 0, 0],
        ...     "_position": [0, 1, 2],
        ...     "subject_id": [1, 1, 1],
        ...     "prediction_time": [datetime(2024, 1, 2)] * 3,
        ...     "query": ["B", "A", "TIMELINE//END"],
        ...     "duration_days": [10.0, 10.0, 10.0],
        ... }).with_columns(
        ...     pl.col("prediction_time").cast(pl.Datetime("us")),
        ...     pl.col("duration_days").cast(pl.Float32),
        ...     pl.col("_ctx_id").cast(pl.UInt32))
        >>> row = label_binary_occurrence(idx, events).row(0, named=True)
        >>> row["answers"]  # B in (2,12)=yes; A at day1 not >day2 =no; END at day20 not in window =no
        [True, False, False]
    """
    sid = TaskQuerySchema.subject_id_name
    pt = TaskQuerySchema.prediction_time_name
    q = TaskQuerySchema.query_name
    d = TaskQuerySchema.duration_days_name

    left = index_df.with_columns((pl.col(pt) + pl.duration(microseconds=1)).alias("_pts")).sort(
        sid, q, "_pts"
    )
    right = (
        events_df.rename({DataSchema.code_name: q})
        .select(sid, q, DataSchema.time_name)
        .sort(sid, q, DataSchema.time_name)
    )
    joined = left.join_asof(
        right, by=[sid, q], left_on="_pts", right_on=DataSchema.time_name, strategy="forward"
    )
    window_end = pl.col(pt) + pl.duration(days=pl.col(d))
    answer = pl.col(DataSchema.time_name).is_not_null() & (pl.col(DataSchema.time_name) < window_end)

    flat = joined.with_columns(answer.alias("answer")).sort(CTX_ID_COL, POSITION_COL)
    return (
        flat.group_by(CTX_ID_COL, maintain_order=True)
        .agg(
            pl.col(sid).first(),
            pl.col(pt).first(),
            pl.col(q).alias("queries"),
            pl.col(d).alias("durations"),
            pl.col("answer").alias("answers"),
        )
        .drop(CTX_ID_COL)
    )


def _first_occurrence_after(
    left: pl.DataFrame, events_df: pl.DataFrame, code_col: str, out_col: str
) -> pl.DataFrame:
    """Forward-asof each row of ``left`` to the first event whose code matches ``code_col``.

    Attaches ``out_col`` — the timestamp of the first occurrence strictly after the row's
    ``_pts`` key — or null when the code never occurs again.  Rows whose ``code_col`` is null
    (a time-bounded query has no boundary) simply find no match.
    """
    sid = TaskQuerySchema.subject_id_name
    time_col = DataSchema.time_name

    right = (
        events_df.rename({DataSchema.code_name: code_col})
        .select(sid, code_col, time_col)
        .sort(sid, code_col, time_col)
    )
    return (
        left.sort(sid, code_col, "_pts")
        .join_asof(right, by=[sid, code_col], left_on="_pts", right_on=time_col, strategy="forward")
        .rename({time_col: out_col})
    )


def label_with_event_bounds(index_df: pl.DataFrame, events_df: pl.DataFrame) -> pl.DataFrame:
    """Label a mixed frame of time-bounded and event-bounded queries.

    A query with a null ``bound_event`` behaves exactly as
    :func:`label_binary_occurrence` — occurrence strictly inside the open interval
    ``(prediction_time, prediction_time + duration_days)``, the horizon instant itself excluded,
    matching :attr:`~every_query.data.schema.QuerySeqSchema.answers`.

    A query with a boundary code is answered over ``(prediction_time, boundary)`` instead,
    where ``boundary`` is the **first occurrence of the boundary code strictly after the
    prediction time**.  The search for that boundary is itself strict at the bottom: an
    occurrence of the boundary code *at* ``prediction_time`` does not close the window.

    **A query code occurring at the exact same instant as the boundary event does NOT count.**
    The upper bound is open here for the same reason it is open on the horizon — one rule, not
    two — and it is what makes an event-bounded query read as *"the query code, strictly after
    the prediction time and strictly before the boundary"*.  The consequence is much larger here
    than on the horizon, though, because MEDS clusters many codes onto a single timestamp: a
    discharge and the codes charted with it routinely share an instant, so this decides real rows
    rather than a measure-zero edge.  It is deliberately not configurable;
    ``test_a_query_at_the_exact_boundary_instant_does_not_count`` in
    ``tests/test_event_bounds_oracle.py`` is the one test that pins it, so making the boundary
    inclusive again is a one-line change with one test to flip.

    One corollary worth stating outright: when ``bound_event`` equals the query code the answer is
    unconditionally ``False``, because the first occurrence after the prediction time is
    simultaneously the boundary and the earliest candidate, and nothing can be strictly before
    itself.

    **When the boundary never occurs again, the window runs to the end of the record**, which
    degenerates the query into "does this code ever occur again".  That is the upstream
    experiment's semantics, kept deliberately so results stay comparable — but it is a real
    trap for boundary codes that do not recur (``MEDS_DEATH`` most obviously), so
    :func:`label_query_sequences` logs how often it fires, per boundary code.

    Examples:
        >>> from datetime import datetime
        >>> events = pl.DataFrame({
        ...     "subject_id": [1, 1, 1, 1],
        ...     "time": [datetime(2024, 1, 3), datetime(2024, 1, 6), datetime(2024, 1, 9),
        ...              datetime(2024, 1, 20)],
        ...     "code": ["A", "DISCHARGE", "A", "TIMELINE//END"],
        ... }).with_columns(pl.col("time").cast(pl.Datetime("us")))
        >>> idx = pl.DataFrame({
        ...     "_ctx_id": [0, 0],
        ...     "_position": [0, 1],
        ...     "subject_id": [1, 1],
        ...     "prediction_time": [datetime(2024, 1, 1)] * 2,
        ...     "query": ["A", "A"],
        ...     "duration_days": [30.0, -1.0],
        ...     "bound_event": [None, "DISCHARGE"],
        ... }).with_columns(
        ...     pl.col("prediction_time").cast(pl.Datetime("us")),
        ...     pl.col("duration_days").cast(pl.Float32),
        ...     pl.col("_ctx_id").cast(pl.UInt32))
        >>> row = label_with_event_bounds(idx, events).row(0, named=True)

        Time-bounded: A occurs on the 3rd, inside 30 days.  Event-bounded: A on the 3rd is
        before the DISCHARGE on the 6th, so also True.

        >>> row["answers"]
        [True, True]
        >>> row["bound_events"]
        [None, 'DISCHARGE']

        The bound is strict, and only occurrences *before* it count.  Asking about a code that
        occurs only after the boundary answers False even though it is well inside 30 days:

        >>> idx2 = idx.head(1).with_columns(
        ...     pl.lit("LATE").alias("query"),
        ...     pl.lit(-1.0).cast(pl.Float32).alias("duration_days"),
        ...     pl.lit("DISCHARGE").alias("bound_event"))
        >>> late_events = pl.concat([events, pl.DataFrame({
        ...     "subject_id": [1], "time": [datetime(2024, 1, 8)], "code": ["LATE"],
        ... }).with_columns(pl.col("time").cast(pl.Datetime("us")))])
        >>> label_with_event_bounds(idx2, late_events).row(0, named=True)["answers"]
        [False]
    """
    sid = TaskQuerySchema.subject_id_name
    pt = TaskQuerySchema.prediction_time_name
    q = TaskQuerySchema.query_name
    d = TaskQuerySchema.duration_days_name

    left = index_df.with_columns((pl.col(pt) + pl.duration(microseconds=1)).alias("_pts"))
    joined = _first_occurrence_after(left, events_df, q, "_q_time")
    joined = _first_occurrence_after(joined, events_df, BOUND_COL, "_b_time")

    is_bounded = pl.col(BOUND_COL).is_not_null()
    # A boundary that never recurs leaves the window open to the end of the record, so any
    # occurrence at all counts.  Expressed as "no upper bound" rather than as a far-future
    # sentinel date, which would need a tz-aware literal to compare against these naive columns.
    window_open = is_bounded & pl.col("_b_time").is_null()
    window_end = (
        pl.when(is_bounded).then(pl.col("_b_time")).otherwise(pl.col(pt) + pl.duration(days=pl.col(d)))
    )
    answer = pl.col("_q_time").is_not_null() & (window_open | (pl.col("_q_time") < window_end))

    flat = joined.with_columns(answer.alias("answer")).sort(CTX_ID_COL, POSITION_COL)
    return (
        flat.group_by(CTX_ID_COL, maintain_order=True)
        .agg(
            pl.col(sid).first(),
            pl.col(pt).first(),
            pl.col(q).alias("queries"),
            pl.col(d).alias("durations"),
            pl.col("answer").alias("answers"),
            pl.col(BOUND_COL).alias("bound_events"),
        )
        .drop(CTX_ID_COL)
    )


def log_degenerate_bounds(index_df: pl.DataFrame, events_df: pl.DataFrame) -> dict[str, float]:
    """Report how often a boundary never fires, and return the per-boundary never-fires rate.

    A boundary that never occurs after the prediction time silently relabels the row "does this
    code ever occur again" (see :func:`label_with_event_bounds`).  A boundary that degenerates most
    of the time is not really being learned as a boundary, and the resulting AUROC would say
    otherwise — so it is logged at generation time, when it is still cheap to notice: one summary
    line plus the ten boundary nodes that never fire most often (boundaries are drawn from the
    whole query universe, so a per-node line would be one per vocabulary entry).
    """
    if BOUND_COL not in index_df.columns:
        return {}
    bounded = index_df.filter(pl.col(BOUND_COL).is_not_null())
    if bounded.height == 0:
        return {}

    left = bounded.with_columns(
        (pl.col(TaskQuerySchema.prediction_time_name) + pl.duration(microseconds=1)).alias("_pts")
    )
    joined = _first_occurrence_after(left, events_df, BOUND_COL, "_b_time")
    rates = joined.group_by(BOUND_COL).agg(
        pl.col("_b_time").is_null().mean().alias("degenerate_rate"),
        pl.col("_b_time").is_null().sum().alias("n_degenerate"),
        pl.len().alias("n"),
    )
    logger.info(
        "Event bounds: %d bounded quer%s over %d distinct boundary node(s); %.1f%% have no boundary "
        "occurrence after the prediction time (window runs to end of record — the query degenerates "
        "to 'does this code ever occur again').",
        bounded.height,
        "y" if bounded.height == 1 else "ies",
        rates.height,
        100.0 * rates["n_degenerate"].sum() / bounded.height,
    )
    worst = rates.filter(pl.col("n_degenerate") > 0).sort("n_degenerate", BOUND_COL, descending=[True, False])
    for row in worst.head(10).iter_rows(named=True):
        logger.info(
            "  never fires: %r — %d of %d (%.1f%%)",
            row[BOUND_COL],
            row["n_degenerate"],
            row["n"],
            100.0 * row["degenerate_rate"],
        )
    return {r[BOUND_COL]: float(r["degenerate_rate"]) for r in rates.sort(BOUND_COL).iter_rows(named=True)}


def label_with_explicit_starts(index_df: pl.DataFrame, events_df: pl.DataFrame) -> pl.DataFrame:
    """Label a frame whose queries carry explicit window starts (issue #27).

    Each flat query row is one window resolved by the multitask sampler's rules, reusing
    :mod:`~every_query.generate_tasks.interval_table` rather than re-deriving them::

        start = prediction_time + start_duration_days           (start_event null)
              | first occurrence of start_event STRICTLY AFTER prediction_time   (else INF)
        end   = start + duration_days                            (bound_event null)
              | first occurrence of bound_event STRICTLY AFTER the RESOLVED START (else INF)
        answer = some occurrence of query satisfies  start < occurrence < end

    That is ``WINDOW_SEMANTICS = "open_open"``, ``START_REFERENCE = "prediction_time"``,
    ``DURATION_END_REFERENCE = "resolved_start"``, ``MISSING_EVENT_START = "empty_window"`` and
    ``MISSING_EVENT_BOUNDARY = "infinity"``.  Concretely:

    - Both endpoints are open: an occurrence at the start instant or at the end instant is outside.
    - The end is resolved **relative to the resolved start**, never to ``prediction_time``: a
      duration end is ``start + duration``, and an event end is the boundary's first occurrence
      after the start — an occurrence of the boundary code between the prediction time and the
      start does not close the window.
    - A start event with no occurrence after the prediction time leaves the window empty: the
      answer is ``False`` even when the end is also unresolved (``INF < INF`` is false), so an
      unresolved start never becomes an infinite positive window.
    - An end event with no occurrence after a resolved start lets the window run to the end of the
      record, as :func:`label_with_event_bounds` does.
    - A prediction-time start (``0.0`` / null) reproduces :func:`label_binary_occurrence` /
      :func:`label_with_event_bounds` exactly for whole-day horizons.  One caveat for fractional
      horizons: the legacy Polars path computes ``prediction_time + pl.duration(days=d)``, which
      *truncates* the microsecond offset, while ``interval_table`` rounds it with ``np.rint``, so the
      two can disagree on an occurrence landing within a microsecond of a non-integer-day horizon
      (e.g. ``2.7`` days: 233280004119 vs 233280004120 µs).  Neither rounding is changed here; the
      multitask training labels use the ``interval_table`` rule, which is what an explicit-start
      grid is scored against.

    Only the scalar query named by each row is labeled — no ``N x K x V`` target is built.  Codes
    are indexed locally over the union of the event stream and every query/start/bound code, so a
    code absent from the events simply never occurs (``INF``).

    Returns:
        One row per ``_ctx_id`` with ``queries`` / ``durations`` / ``answers`` list columns, plus
        ``bound_events`` when the index carried ``bound_event`` and the ``start_durations`` /
        ``start_events`` list columns (the index must carry both start columns).

    Examples:
        >>> from datetime import datetime
        >>> events = pl.DataFrame({
        ...     "subject_id": [1, 1, 1, 1],
        ...     "time": [datetime(2024, 1, 3), datetime(2024, 1, 6), datetime(2024, 1, 9),
        ...              datetime(2024, 1, 20)],
        ...     "code": ["A", "ADMIT", "A", "DISCHARGE"],
        ... }).with_columns(pl.col("time").cast(pl.Datetime("us")))
        >>> idx = pl.DataFrame({
        ...     "_ctx_id": [0, 0, 0],
        ...     "_position": [0, 1, 2],
        ...     "subject_id": [1, 1, 1],
        ...     "prediction_time": [datetime(2024, 1, 1)] * 3,
        ...     "query": ["A", "A", "A"],
        ...     "duration_days": [30.0, -1.0, 30.0],
        ...     "bound_event": [None, "DISCHARGE", None],
        ...     "start_duration_days": [0.0, -1.0, -1.0],
        ...     "start_event": [None, "ADMIT", "NEVER"],
        ... }).with_columns(
        ...     pl.col("prediction_time").cast(pl.Datetime("us")),
        ...     pl.col("duration_days").cast(pl.Float32),
        ...     pl.col("start_duration_days").cast(pl.Float32),
        ...     pl.col("_ctx_id").cast(pl.UInt32))
        >>> row = label_with_explicit_starts(idx, events).row(0, named=True)

        Position 0 opens at the prediction time (A on the 3rd is inside 30 days); position 1 opens
        at the admission on the 6th and closes at the discharge (A on the 9th is inside; A on the
        3rd is before the start and does not count); position 2's start event never occurs, so the
        window is empty:

        >>> row["answers"]
        [True, True, False]
        >>> row["start_events"], row["bound_events"]
        ([None, 'ADMIT', 'NEVER'], [None, 'DISCHARGE', None])
    """
    from every_query.generate_tasks.interval_table import (
        INF,
        build_interval_table,
        next_occurrence_after,
        resolve_end_times,
        resolve_start_times,
    )

    sid = TaskQuerySchema.subject_id_name
    pt = TaskQuerySchema.prediction_time_name
    q = TaskQuerySchema.query_name
    d = TaskQuerySchema.duration_days_name
    missing = [c for c in START_INDEX_COLS if c not in index_df.columns]
    if missing:
        raise ValueError(
            f"index frame carries {[c for c in START_INDEX_COLS if c in index_df.columns]} but not {missing}"
        )
    has_bounds = BOUND_COL in index_df.columns

    ordered = index_df.sort(CTX_ID_COL, POSITION_COL)
    n = ordered.height

    # A local code index over everything that can occur or be named; codes named by a query but
    # absent from the stream get an index too, and simply never resolve.
    events = events_df.filter(pl.col(DataSchema.time_name).is_not_null())
    named = [ordered[q], ordered[START_EVENT_COL]]
    if has_bounds:
        named.append(ordered[BOUND_COL])
    universe = (
        pl.concat([events[DataSchema.code_name].cast(pl.Utf8), *[s.cast(pl.Utf8) for s in named]])
        .drop_nulls()
        .unique()
        .sort()
    )
    code_to_index = {c: i for i, c in enumerate(universe.to_list())}
    vocab_size = max(len(code_to_index), 1)

    def encode(series: pl.Series) -> np.ndarray:
        """Vocabulary index per row, ``-1`` where the code is null (a duration-defined slot)."""
        if n == 0:
            return np.empty((0, 1), dtype=np.int64)
        mapped = series.cast(pl.Utf8).replace_strict(code_to_index, default=-1, return_dtype=pl.Int64)
        return mapped.fill_null(-1).to_numpy().astype(np.int64).reshape(n, 1)

    # ``interval_table`` works in int64 microseconds (``US_PER_DAY``), so both time columns are
    # normalised to ``Datetime("us")`` before the cast — a ``ms``/``ns`` frame would otherwise be
    # silently mislabeled, exactly as the multitask sampler guards against.
    table = build_interval_table(
        events[sid].to_numpy().astype(np.int64),
        events[DataSchema.time_name].cast(pl.Datetime("us")).cast(pl.Int64).to_numpy(),
        events[DataSchema.code_name]
        .cast(pl.Utf8)
        .replace_strict(code_to_index, return_dtype=pl.Int64)
        .to_numpy(),
        vocab_size=vocab_size,
    )

    subject_ids = ordered[sid].to_numpy().astype(np.int64)
    prediction_times = ordered[pt].cast(pl.Datetime("us")).cast(pl.Int64).to_numpy()
    start_durations = ordered[START_DURATION_COL].cast(pl.Float64).fill_null(np.nan).to_numpy().reshape(n, 1)
    start_codes = encode(ordered[START_EVENT_COL])
    durations = ordered[d].cast(pl.Float64).to_numpy().reshape(n, 1)
    bound_codes = encode(ordered[BOUND_COL]) if has_bounds else np.full((n, 1), -1, dtype=np.int64)
    query_codes = encode(ordered[q])[:, 0]

    # The representation contract is enforced at the seam that produces the answers, not only in
    # the generator and the dataset: a negative or non-finite non-sentinel start would otherwise
    # resolve to an arbitrary instant *before* the prediction time and label silently.
    by_event = start_codes >= 0
    bad_duration = ~by_event & ~(np.isfinite(start_durations) & (start_durations >= 0))
    bad_event = by_event & (start_durations != EVENT_BOUND_DURATION_SENTINEL)
    if bad_duration.any() or bad_event.any():
        first = int(np.flatnonzero((bad_duration | bad_event)[:, 0])[0])
        raise ValueError(
            "start_duration_days/start_event disagree on which queries are event-defined: each row must "
            "be either start_duration_days >= 0 (finite) with a null start_event, or start_duration_days "
            f"== {EVENT_BOUND_DURATION_SENTINEL} with a non-null start_event; first bad row {first} has "
            f"start_duration_days={start_durations[first, 0]!r}, "
            f"start_event={ordered[START_EVENT_COL][first]!r}."
        )

    starts = resolve_start_times(table, subject_ids, prediction_times, start_durations, start_codes)
    ends = resolve_end_times(table, subject_ids, starts, durations, bound_codes)
    # First occurrence of the query strictly after the resolved start; an unresolved start (INF)
    # has nothing after it, so the window stays empty even when the end is INF too.
    next_query = next_occurrence_after(table, subject_ids, starts[:, 0], query_codes)
    answers = next_query < ends[:, 0]

    if n:
        n_event_start = int((start_codes >= 0).sum())
        n_empty = int((starts == INF).sum())
        if n_event_start:
            logger.info(
                "Explicit starts: %d event-defined start(s), %d (%.1f%%) never resolve (empty window).",
                n_event_start,
                n_empty,
                100.0 * n_empty / n_event_start,
            )
        # The explicit-start counterpart of ``log_degenerate_bounds``: an event end that never
        # recurs after the *resolved* start runs to the end of the record (the query degenerates to
        # "does this code ever occur again").  Anchored at the resolved start, not the prediction
        # time, so only windows that actually opened are counted.
        opened_event_end = (bound_codes >= 0) & (starts != INF)
        n_opened = int(opened_event_end.sum())
        if n_opened:
            n_degenerate = int((opened_event_end & (ends == INF)).sum())
            logger.info(
                "Event bounds after a resolved start: %d window(s), %d (%.1f%%) have no boundary "
                "occurrence after the start (window runs to end of record).",
                n_opened,
                n_degenerate,
                100.0 * n_degenerate / n_opened,
            )

    flat = ordered.with_columns(pl.Series("answer", answers, dtype=pl.Boolean))
    aggregations = [
        pl.col(sid).first(),
        pl.col(pt).first(),
        pl.col(q).alias("queries"),
        pl.col(d).alias("durations"),
        pl.col("answer").alias("answers"),
    ]
    if has_bounds:
        aggregations.append(pl.col(BOUND_COL).alias("bound_events"))
    aggregations += [
        pl.col(START_DURATION_COL).alias("start_durations"),
        pl.col(START_EVENT_COL).alias("start_events"),
    ]
    return flat.group_by(CTX_ID_COL, maintain_order=True).agg(aggregations).drop(CTX_ID_COL)


def maybe_expand_to_matching_query_nodes(events_df: pl.DataFrame, ontology_dir: str | Path | None):
    """Repeat each event under its ancestor node names, so ancestor queries label normally.

    A no-op without an ontology.  With one, "did any descendant of X occur" becomes an ordinary
    occurrence question about X, so :func:`label_binary_occurrence` needs no ancestor awareness
    at all — it re-sorts its own right frame, so the join-scrambled row order is irrelevant.

    Note this multiplies the event frame by the mean closure size, which raises peak memory in
    each Stage 4' worker.
    """
    if not ontology_dir:
        return events_df
    from every_query.data.ontology import expand_events_to_query_nodes, load_event_to_query_nodes

    return expand_events_to_query_nodes(events_df, load_event_to_query_nodes(ontology_dir))


def label_query_sequences(index_df: pl.DataFrame, events_df: pl.DataFrame) -> pl.DataFrame:
    """Stage 4' labeling entry point: answer every query in ``index_df`` against ``events_df``.

    The single seam through which *all* sequence labeling flows — both the sharded training
    path (:func:`label_one_sequence_shard`) and the dense evaluation grid — so a query form
    added here is answered identically in training data and eval grids.  A form that labels one
    way in training and another in evaluation produces a grid that looks fine and silently
    measures the wrong thing.

    Dispatch is on the *frame*, not on a config flag: an index carrying the explicit-start
    columns (``start_duration_days`` / ``start_event``, issue #27) is labelled with
    :func:`label_with_explicit_starts` on the interval table; one carrying only a ``bound_event``
    column with :func:`label_with_event_bounds`; anything else with
    :func:`label_binary_occurrence`, which stays the regression anchor for plain occurrence
    semantics.  Keying on the data means an eval grid built with bounds or starts is always scored
    with the matching labels, even if some caller forgets to pass a flag — and an index with only
    one of the two start columns is an error, not a legacy frame.

    This must remain a **module-level function**: Stage 4' fans shards out through a
    ``ProcessPoolExecutor`` with ``mp_context="spawn"``, so a closure or a locally-defined
    callable would fail to pickle.

    Args:
        index_df: Flat per-query index frame (one row per query of each sequence).
        events_df: The event stream to answer against, already closure-exploded by the caller
            when an ontology is in use, so ancestor queries and boundaries label correctly.

    Returns:
        One row per sequence, with the ``queries``/``durations``/``answers`` list columns
        (plus ``bound_events`` when the index carried bounds, and ``start_durations`` /
        ``start_events`` when it carried explicit starts).
    """
    present_starts = [c for c in START_INDEX_COLS if c in index_df.columns]
    if present_starts:
        if len(present_starts) != len(START_INDEX_COLS):
            raise ValueError(
                f"index frame carries {present_starts} but not all of {list(START_INDEX_COLS)}; the two "
                "explicit-start columns must be present together."
            )
        return label_with_explicit_starts(index_df, events_df)
    if BOUND_COL in index_df.columns:
        log_degenerate_bounds(index_df, events_df)
        return label_with_event_bounds(index_df, events_df)
    return label_binary_occurrence(index_df, events_df)


def label_one_sequence_shard(
    shard: str,
    index_dir: Path,
    data_dir: Path,
    out_dir: Path,
    overwrite: bool = False,
    ontology_dir: str | None = None,
) -> tuple[str, str]:
    """Stage 4' worker: label one index partition and write the final ``QuerySeqSchema`` parquet.

    The sequence analogue of :func:`~every_query.generate_tasks.sample_tasks.label_one_shard`,
    including its fingerprint-keyed skip: an existing output is reused only when the recorded
    :func:`~every_query.generate_tasks.sample_tasks._index_fingerprint` matches the current Stage 3'
    partition, so a changed sampling config relabels even under ``overwrite=False``.  A
    missing/unreadable fingerprint is treated as stale.

    Returns:
        ``(shard, status)`` where status is ``"skipped"`` or ``"labeled"``.
    """
    final = out_dir / f"{shard}.parquet"
    fingerprint_fp = index_dir.parent / LABELED_DIRNAME / f"{shard}.json"

    index_df = pl.read_parquet(index_dir / f"{shard}.parquet")
    current_fingerprint = _index_fingerprint(index_df)

    if not overwrite and final.exists():
        try:
            recorded = json.loads(fingerprint_fp.read_text()).get("index_fingerprint")
        except (OSError, json.JSONDecodeError, AttributeError):
            recorded = None
        if recorded == current_fingerprint:
            return shard, "skipped"

    _clean_stale_temps(out_dir, shard)

    events_df = _read_event_shard(data_dir / f"{shard}.parquet")
    events_df = maybe_expand_to_matching_query_nodes(events_df, ontology_dir)

    labeled = label_query_sequences(index_df, events_df)
    aligned = QuerySeqSchema.align(labeled.to_arrow())

    _atomic_write_parquet(pl.from_arrow(aligned), final)
    # Record the index fingerprint *after* the output is committed so a present sidecar always
    # describes a present, complete output (the parquet is the value, the sidecar its provenance).
    _atomic_write_json({"index_fingerprint": current_fingerprint}, fingerprint_fp)
    return shard, "labeled"


def _label_sequence_shards(
    shards: list[str],
    index_dir: Path,
    data_dir: Path,
    out_dir: Path,
    overwrite: bool,
    n_workers: int,
    ontology_dir: str | None = None,
) -> None:
    """Fan one :func:`label_one_sequence_shard` worker out per shard via a spawn-based pool.

    ``"spawn"``, not the Linux default ``"fork"``: by Stage 4' the driver has already run polars (which starts
    a rayon threadpool), and forking while those threads hold locks leaves the child with inherited-but-locked
    mutexes, deadlocking the worker the moment it touches polars (#210).
    """
    mp_context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_context) as ex:
        futs = {
            ex.submit(label_one_sequence_shard, s, index_dir, data_dir, out_dir, overwrite, ontology_dir): s
            for s in shards
        }
        for fut in as_completed(futs):
            fut.result()  # re-raise so a failed shard aborts the run loudly


def _validate_sequence_count(out_files: list[Path], expected: int) -> int:
    """Sum the final shards' row counts and check they account for every sampled sequence.

    One output row is one sequence, so the union of the split's shards must have exactly
    ``num_sequences`` rows; a mismatch means a dropped context or a partially-written shard.  Uses
    the parquet metadata row count, so no payload is read into memory.
    """
    written = sum(pl.scan_parquet(fp).select(pl.len()).collect().item() for fp in out_files)
    if written != expected:
        raise ValueError(
            f"Expected {expected:,} labeled sequences across {len(out_files):,} shard(s) but found "
            f"{written:,}. The output directory may hold a partial run; rerun with overwrite=true."
        )
    return written


def label_sequence_shards(
    cfg: DictConfig,
    path_to_data: Path,
    seq_tasks_dir: Path,
    seq_task_artifacts_dir: Path,
    total_sequences: int,
) -> int:
    """Stage 4': fan one labeling worker out per Stage 3' index shard; return sequences written.

    Shards are exactly the Stage 3' index partitions; workers receive ids/paths (never DataFrames)
    and write their own atomic output.  The driver creates ``out_dir`` once, before the pool.
    """
    index_dir = seq_task_artifacts_dir / cfg.split / INDEX_DIRNAME
    labeled_dir = seq_task_artifacts_dir / cfg.split / LABELED_DIRNAME
    data_dir = path_to_data / "data" / cfg.split
    out_dir = seq_tasks_dir / cfg.split
    out_dir.mkdir(parents=True, exist_ok=True)

    shards = sorted(p.stem for p in index_dir.glob("*.parquet"))
    _prune_stale_outputs(out_dir, labeled_dir, set(shards))

    n_workers = resolve_workers(cfg.get("max_workers"))
    logger.info("Stage 4': labeling %s shard(s) across %s worker(s).", f"{len(shards):,}", f"{n_workers:,}")
    _label_sequence_shards(
        shards,
        index_dir,
        data_dir,
        out_dir,
        bool(cfg.overwrite),
        n_workers,
        ontology_dir=cfg.get("ontology_dir"),
    )

    out_files = sorted(out_dir.glob("*.parquet"))
    written = _validate_sequence_count(out_files, total_sequences)
    logger.info("Pipeline complete: wrote %s query sequences to %s.", f"{written:,}", out_dir)
    return written


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(cfg: DictConfig) -> None:
    """Execute the 5-stage sequence pipeline for a fully-resolved config (no Hydra side effects).

    Mirrors :func:`~every_query.generate_tasks.sample_tasks.run`: Stages 0-3' run sequentially in
    this driver process and produce the partitioned Stage 3' index; Stage 4' then fans out one
    worker per shard via a ``ProcessPoolExecutor`` sized by
    :func:`~every_query.generate_tasks.sample_tasks.resolve_workers`.  The query, context, and
    sequence-structure axes are seeded independently via
    :func:`~every_query.utils.seeds.derive_seed` so they reproduce separately for a fixed
    ``cfg.seed``.

    Path roots come from ``cfg.data_dir`` / ``cfg.out_dir``; the intermediate-artifacts root has no
    key of its own and is always the ``{name}_artifacts`` sibling of ``out_dir`` (invariant 7).
    """
    path_to_data = _require_path_arg(cfg.get("data_dir"), "data_dir")
    seq_tasks_dir = _require_path_arg(cfg.get("out_dir"), "out_dir")
    seq_task_artifacts_dir = default_artifacts_dir(seq_tasks_dir)

    # Resolve the query distribution (and so the universe) before any Stage 0 work, so a config
    # error fails in seconds rather than after the prediction-time map has been rebuilt.
    query_dist = QuerySequenceDistribution.from_config(cfg)

    # Stage 0: precompute & cache subject prediction_time_indexes and per-subject counts.
    n_subjects = build_prediction_times(
        path_to_data=path_to_data,
        training_task_artifacts_dir=seq_task_artifacts_dir,
        split=cfg.split,
        min_prediction_times_per_subject=cfg.min_prediction_times_per_subject,
        overwrite=cfg.overwrite,
    )
    logger.info("Stage 0: %s eligible subject(s) for split=%s.", f"{n_subjects:,}", cfg.split)

    num_sequences = int(cfg.num_training_sequence_examples)

    # Independent RNG streams per axis (invariant 5): "queries" matches the training sampler's draw
    # exactly (parity), "contexts" matches its context draw, "sequences" carries the structure-only
    # draws that have no counterpart there, and "bounds" decides which queries are event-bounded.
    query_rng = np.random.default_rng(derive_seed(cfg.seed, "queries"))
    context_rng = np.random.default_rng(derive_seed(cfg.seed, "contexts"))
    structure_rng = np.random.default_rng(derive_seed(cfg.seed, "sequences"))
    bound_rng = np.random.default_rng(derive_seed(cfg.seed, "bounds"))

    # Stage 1': sample one variable-length query sequence per context-to-be.
    sequences = query_dist.sample_sequences(num_sequences, query_rng, structure_rng, bound_rng)
    n_queries = sum(len(s) for s in sequences)
    logger.info(
        "Stage 1': sampled %s sequence(s) totaling %s quer%s from a %s-node universe "
        "(%s durations over [%g, %g] days, lengths ~ U{%d..%d}, eos_first_fraction=%g, "
        "duration_mode=%s, eventbound_fraction=%g).",
        f"{num_sequences:,}",
        f"{n_queries:,}",
        "y" if n_queries == 1 else "ies",
        f"{query_dist.query_universe_size:,}",
        query_dist.duration_distribution,
        query_dist.min_duration,
        query_dist.max_duration,
        query_dist.min_queries,
        query_dist.max_queries,
        query_dist.eos_first_fraction,
        query_dist.duration_mode,
        query_dist.eventbound_fraction,
    )

    # Stage 2: one patient context per sequence.  Re-sort by subject_id so the subject_idx ->
    # subject_id mapping is independent of parquet round-trip order (see sample_tasks.run).
    prediction_time_counts_df = pl.read_parquet(
        prediction_time_counts_path(seq_task_artifacts_dir, cfg.split)
    ).sort("subject_id")
    contexts = sample_patient_contexts(
        prediction_time_counts=prediction_time_counts_df,
        n=num_sequences,
        min_prediction_times_per_subject=cfg.min_prediction_times_per_subject,
        rng=context_rng,
    )
    logger.info("Stage 2: sampled %s patient context(s).", f"{contexts.height:,}")

    # Stage 3': zip sequences with contexts, resolve prediction times, write the per-shard index.
    n_index_shards = build_sequence_index(
        sequences=sequences,
        contexts=contexts,
        training_task_artifacts_dir=seq_task_artifacts_dir,
        split=cfg.split,
    )
    logger.info(
        "Stage 3': wrote partitioned index for split=%s (%s query rows across %s shard(s)).",
        cfg.split,
        f"{n_queries:,}",
        f"{n_index_shards:,}",
    )

    # Stage 4': fan one labeling worker out per shard (one Stage 3' index partition each).
    label_sequence_shards(cfg, path_to_data, seq_tasks_dir, seq_task_artifacts_dir, num_sequences)


# ---------------------------------------------------------------------------
# Supplied-cohort helpers (used by the eval-grid sampler)
# ---------------------------------------------------------------------------


CONFIGS = str(files("every_query") / "generate_tasks" / "configs")


@hydra.main(version_base=None, config_path=CONFIGS, config_name="sample_query_sequences_config")
def main(cfg: DictConfig) -> None:
    """Hydra entry point for the sampled 5-stage pipeline via :func:`run`.

    There is no console script; reach it as ``python -m
    every_query.generate_tasks.query_sequence_labeling``.  It survives because it is one arm of the
    two-pipeline sampler differential (see this module's docstring).

    Path roots are required args with no ``.env``/env-var fallback (removed upstream in #235):
    ``data_dir``, ``out_dir``, and ``query_codes`` are mandatory, resolved through the same
    :func:`~every_query.generate_tasks.sample_tasks._require_path_arg` guard the training sampler
    uses so an unexported ``$VAR`` fails with one clear message instead of a literal ``None`` path.
    """
    run(cfg)


if __name__ == "__main__":
    main()
