"""Dataset + batch type for the all-vocabulary multi-bound multitask labels (issue #20).

On disk, ``EQ_generate_multitask_sequences`` writes per event shard of a split::

    {task_labels_dir}/{split}/{shard}.parquet             MultitaskBoundarySchema metadata rows
    {task_labels_dir}/{split}/{shard}.labels.npy          uint8 (rows, K, ceil(V/8)), bitorder little
    {task_labels_dir}/{split}/_multitask_manifest.json    vocabulary + semantics the bits were built under

The parquet and the ``.npy`` are row-aligned; that alignment **is** the contract.  This dataset

1. loads only the metadata parquets at init (never the packed labels);
2. opens each ``.labels.npy`` with ``mmap_mode="r"`` (read-only, lazily, per process);
3. tags every metadata row with its *source shard key* and *physical source row* while reading the
   parquet and carries both through the upstream task-sequence join, so the sidecar row is looked up
   explicitly rather than inferred from a concatenation order;
4. gathers packed rows in :meth:`MultitaskBoundaryPytorchDataset.collate` and unpacks **once per batch**
   with ``bitorder="little"``; ``__getitem__`` never unpacks;
5. keeps targets boolean until the loss casts them;
6. loads the ``K-1`` sampler-materialized conditioning codes/answers (issue #22) and, in ``collate``,
   checks every stored answer against the unpacked target bit.  It never samples either;
7. loads the sampler-materialized window starts (issue #24: ``start_durations`` / ``start_events``),
   reading parquets written before #24 - which lack both columns - as prediction-time starts
   (``[0.0] * K`` / ``[null] * K``), and maps start codes through the same vocabulary as the boundary
   and conditioning codes.  No start is ever sampled here.

Targets are always the cohort's leaf codes, ``V`` wide.  *Event* names - starts, bounds, conditioning
codes - may reach ``[V, V_ext)`` when the sampler ran with an ontology (the manifest's
``ontology_mode`` is not ``"none"``); pass that ``ontology_dir`` and they resolve through
``extend_code_map``, bounded by the manifest's ``boundary_vocab_size``.  An ancestor conditioning
code's stored answer is the OR over its closure descendants - the value ``derive_ancestor_targets``
would put in that column - and ``collate`` re-derives exactly the ancestor slots a batch names to
check it, rather than widening the whole block to ``V_ext`` to read back a handful of bits.

It fails loudly at init when the cohort's ``codes.parquet`` (or an explicit ``expected_vocab_size`` /
``expected_vocab_fingerprint``) disagrees with the manifest, and when the ontology it is given is not
the one the labels were sampled against.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import polars as pl
import torch
from meds import DataSchema, LabelSchema
from meds_torchdata import MEDSPytorchDataset
from meds_torchdata.config import MEDSTorchDataConfig
from meds_torchdata.types import MEDSTorchBatch

from every_query.data.query_seq_dataset import EVENT_BOUND_DURATION_SENTINEL, NO_BOUND_INDEX

logger = logging.getLogger(__name__)

START_DURATIONS_COL = "start_durations"
START_EVENTS_COL = "start_events"
DURATIONS_COL = "durations"
BOUND_EVENTS_COL = "bound_events"
CONDITION_CODES_COL = "condition_codes"
CONDITION_ANSWERS_COL = "condition_answers"
# The two start columns are optional per parquet (absent before issue #24); the rest are required.
START_COLS = (START_DURATIONS_COL, START_EVENTS_COL)
REQUIRED_LABEL_COLS = (DURATIONS_COL, BOUND_EVENTS_COL, CONDITION_CODES_COL, CONDITION_ANSWERS_COL)
MULTITASK_LABEL_COLS = (*START_COLS, *REQUIRED_LABEL_COLS)
SUPPORTED_FORMAT_VERSIONS = (2, 3)

SOURCE_SHARD_COL = "_source_shard"
SOURCE_ROW_COL = "_source_row"
_INTERNAL_COLS = (SOURCE_SHARD_COL, SOURCE_ROW_COL)

MANIFEST_NAME = "_multitask_manifest.json"
LABELS_SUFFIX = ".labels.npy"
BITORDER = "little"
# The sampler's ``ontology_mode`` value that means "no ancestor ever acted as an event".  Declared
# here rather than imported so reading a labels directory never pulls in the generator package.
ONTOLOGY_MODE_NONE = "none"


@dataclass
class MultitaskBoundaryBatch(MEDSTorchBatch):
    """MEDS batch extended with ``K`` windows per sample and all-vocabulary boolean targets.

    Attributes:
        q_start_durations: ``(B, K)`` float32 - days after ``prediction_time`` the window opens;
            ``0.0`` opens it at the prediction time (legacy), ``EVENT_BOUND_DURATION_SENTINEL``
            (``-1.0``) marks an event-defined start.  Always filled by ``collate`` (zeros for files
            written before issue #24); ``None`` only on a hand-built batch, which the model reads as
            zero-duration starts.
        q_start_codes: ``(B, K)`` int64 - vocabulary index of the start event; ``NO_BOUND_INDEX`` (0)
            for duration / prediction-time starts, never PAD when active; an ontology node in
            ``[V, V_ext)`` under a boundaries ontology mode.  Filled together with
            ``q_start_durations``: exactly one of the two being ``None`` is an error.
        q_durations: ``(B, K)`` float - horizon in days after the **resolved start**;
            ``EVENT_BOUND_DURATION_SENTINEL`` at event-bounded slots.
        q_bound_codes: ``(B, K)`` long - boundary vocabulary index; ``NO_BOUND_INDEX`` (0) for a
            duration-bounded slot.  May be an ontology node in ``[V, V_ext)`` when the labels were
            sampled in a boundaries ontology mode, where it means "the next occurrence of any
            descendant leaf".
        q_mask: ``(B, K)`` bool - True at real slots (all True: ``K`` is fixed; kept for the model
            contract).
        targets: ``(B, K, V)`` bool - ``targets[b, k, v]`` is "code ``v`` occurs strictly inside the
            open window ``(resolved_start, resolved_end)`` of window ``k``".  Bit ``0`` (PAD) is
            always False and must be masked from the loss.
        condition_codes: ``(B, K-1)`` long - vocabulary index of the sampler-drawn conditioning code
            for boundaries ``0..K-2`` (never PAD).  May be an ontology node in ``[V, V_ext)`` when
            the labels were sampled in a conditioning ontology mode.
        condition_answers: ``(B, K-1)`` bool - ``targets[b, j, condition_codes[b, j]]``, materialized
            by the sampler and verified in ``collate``.  For an ancestor code there is no such
            column and the answer is the OR over its closure descendants instead - the value
            ``derive_ancestor_targets`` puts in that column.

    Examples:
        >>> batch = MultitaskBoundaryBatch(
        ...     code=torch.tensor([[1, 2, 3], [4, 5, 0]]),
        ...     numeric_value=torch.zeros(2, 3),
        ...     numeric_value_mask=torch.zeros(2, 3, dtype=torch.bool),
        ...     time_delta_days=torch.zeros(2, 3),
        ...     q_start_durations=torch.tensor([[0.0, -1.0], [7.0, 0.0]]),
        ...     q_start_codes=torch.tensor([[0, 2], [0, 0]]),
        ...     q_durations=torch.tensor([[30.0, -1.0], [7.0, 2.0]]),
        ...     q_bound_codes=torch.tensor([[0, 4], [0, 0]]),
        ...     q_mask=torch.ones(2, 2, dtype=torch.bool),
        ...     targets=torch.zeros(2, 2, 6, dtype=torch.bool),
        ...     condition_codes=torch.tensor([[3], [5]]),
        ...     condition_answers=torch.zeros(2, 1, dtype=torch.bool),
        ... )
        >>> batch.num_bounds, batch.vocab_size
        (2, 6)

        The start fields may both be omitted (legacy: every window opens at the prediction time) but
        never just one of them:

        >>> kw = dict(code=torch.tensor([[1, 2], [3, 4]]), numeric_value=torch.zeros(2, 2),
        ...     numeric_value_mask=torch.zeros(2, 2, dtype=torch.bool),
        ...     time_delta_days=torch.zeros(2, 2), q_durations=torch.tensor([[30.0, -1.0], [7.0, 2.0]]),
        ...     q_bound_codes=torch.tensor([[0, 4], [0, 0]]), q_mask=torch.ones(2, 2, dtype=torch.bool),
        ...     targets=torch.zeros(2, 2, 6, dtype=torch.bool), condition_codes=torch.tensor([[3], [5]]),
        ...     condition_answers=torch.zeros(2, 1, dtype=torch.bool))
        >>> MultitaskBoundaryBatch(**kw).q_start_durations is None
        True
        >>> MultitaskBoundaryBatch(**kw, q_start_codes=torch.zeros(2, 2, dtype=torch.long))
        Traceback (most recent call last):
            ...
        ValueError: q_start_durations and q_start_codes must be given together (got q_start_durations=None)

        Mismatched shapes raise:

        >>> MultitaskBoundaryBatch(
        ...     code=torch.tensor([[1, 2], [3, 4]]),
        ...     numeric_value=torch.zeros(2, 2),
        ...     numeric_value_mask=torch.zeros(2, 2, dtype=torch.bool),
        ...     time_delta_days=torch.zeros(2, 2),
        ...     q_durations=torch.tensor([[30.0, -1.0], [7.0, 2.0]]),
        ...     q_bound_codes=torch.tensor([[0, 4], [0, 0]]),
        ...     q_mask=torch.ones(2, 2, dtype=torch.bool),
        ...     targets=torch.zeros(2, 3, 6, dtype=torch.bool),
        ...     condition_codes=torch.tensor([[3], [5]]),
        ...     condition_answers=torch.zeros(2, 1, dtype=torch.bool),
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Expected targets to have shape (2, 2, V), but got torch.Size([2, 3, 6])!
    """

    q_durations: torch.FloatTensor | None = None
    q_bound_codes: torch.LongTensor | None = None
    q_mask: torch.BoolTensor | None = None
    targets: torch.BoolTensor | None = None
    condition_codes: torch.LongTensor | None = None
    condition_answers: torch.BoolTensor | None = None
    # Issue #24 window starts; ``collate`` always fills both.
    q_start_durations: torch.FloatTensor | None = None
    q_start_codes: torch.LongTensor | None = None
    # Per-patient-token elapsed hours for rotary time encoding; ``None`` unless the dataset strips
    # delta tokens (mirrors ``QuerySeqBatch.time_pos_ids``).
    time_pos_ids: torch.LongTensor | None = None

    LABEL_TENSOR_NAMES: ClassVar[tuple[str]] = (
        "boolean_value",
        "q_start_durations",
        "q_start_codes",
        "q_durations",
        "q_bound_codes",
        "targets",
        "condition_codes",
        "condition_answers",
    )

    @property
    def num_bounds(self) -> int | None:
        return None if self.q_durations is None else self.q_durations.shape[1]

    @property
    def vocab_size(self) -> int | None:
        return None if self.targets is None else self.targets.shape[-1]

    def __post_init__(self):
        super().__post_init__()
        if (self.q_start_durations is None) != (self.q_start_codes is None):
            missing = "q_start_durations" if self.q_start_durations is None else "q_start_codes"
            raise ValueError(
                f"q_start_durations and q_start_codes must be given together (got {missing}=None)"
            )
        if self.q_durations is None:
            return
        expected = (self.batch_size, self.q_durations.shape[1])
        for name in ("q_durations", "q_bound_codes", "q_mask"):
            tensor = getattr(self, name)
            if tensor is None or tuple(tensor.shape) != expected:
                got = None if tensor is None else tensor.shape
                raise ValueError(f"Expected shape {expected} for {name}, but got {got}!")
        for name in ("q_start_durations", "q_start_codes"):
            tensor = getattr(self, name)
            if tensor is not None and tuple(tensor.shape) != expected:
                raise ValueError(f"Expected shape {expected} for {name}, but got {tensor.shape}!")
        if self.targets is None or self.targets.dim() != 3 or tuple(self.targets.shape[:2]) != expected:
            got = None if self.targets is None else self.targets.shape
            raise ValueError(
                f"Expected targets to have shape {(*expected, 'V')}, but got {got}!".replace("'", "")
            )
        if self.targets.dtype != torch.bool:
            raise TypeError(f"targets must be boolean, got {self.targets.dtype}")
        cond_shape = (self.batch_size, self.q_durations.shape[1] - 1)
        for name in ("condition_codes", "condition_answers"):
            tensor = getattr(self, name)
            if tensor is None or tuple(tensor.shape) != cond_shape:
                got = None if tensor is None else tensor.shape
                raise ValueError(f"Expected shape {cond_shape} for {name}, but got {got}!")
        if self.condition_answers.dtype != torch.bool:
            raise TypeError(f"condition_answers must be boolean, got {self.condition_answers.dtype}")


def read_manifest(split_dir: Path) -> dict:
    fp = split_dir / MANIFEST_NAME
    if not fp.exists():
        raise FileNotFoundError(
            f"No multitask manifest at {fp}. Generate labels with EQ_generate_multitask_sequences; the "
            "driver writes the manifest before any shard is labeled."
        )
    manifest = json.loads(fp.read_text())
    for key in (
        "num_bounds",
        "vocab_size",
        "packed_width_bytes",
        "bitorder",
        "vocab_fingerprint",
        "ontology_mode",
    ):
        if key not in manifest:
            raise ValueError(f"multitask manifest {fp} is missing {key!r}")
    if manifest["bitorder"] != BITORDER:
        raise ValueError(f"multitask manifest bitorder must be {BITORDER!r}, got {manifest['bitorder']!r}")
    # ``boundary_vocab_size`` is not in the required set: manifests written before ancestor-valued
    # boundaries existed carry neither it nor ``ontology_fingerprint``, and must keep loading.  A
    # manifest that *claims* an ontology mode without it is corrupt, though - the width is the only
    # thing that says how far an ancestor start / bound / condition code may reach.
    if manifest["ontology_mode"] != ONTOLOGY_MODE_NONE:
        missing = [k for k in ("boundary_vocab_size", "ontology_fingerprint") if not manifest.get(k)]
        if missing:
            raise ValueError(
                f"multitask manifest {fp} has ontology_mode {manifest['ontology_mode']!r} but no "
                f"{' or '.join(missing)}; regenerate the labels with the current "
                "EQ_generate_multitask_sequences."
            )
    # Format 2 (issue #20/#22: every window opens at the prediction time, no start columns) and
    # format 3 (issue #24: explicit starts) are both readable; a missing key means legacy format 2.
    version = manifest.get("format_version", 2)
    if version not in SUPPORTED_FORMAT_VERSIONS:
        raise ValueError(
            f"multitask manifest {fp} has format_version {version!r}; this dataset reads "
            f"{list(SUPPORTED_FORMAT_VERSIONS)}. Regenerate the labels with the current "
            "EQ_generate_multitask_sequences."
        )
    return manifest


def _vocab_fingerprint_from_codes_parquet(fp: Path) -> tuple[int, str]:
    """``(V, fingerprint)`` of a cohort ``codes.parquet``, computed exactly as the sampler does."""
    from every_query.generate_tasks.sample_multitask_sequences import build_target_vocabulary

    vocab = build_target_vocabulary(fp)
    return vocab.size, vocab.fingerprint


def _check_rows_aligned(base: pl.DataFrame, surviving: pl.DataFrame) -> None:
    """Require ``base`` and ``surviving`` to agree row-for-row on subject_id and prediction_time.

    Examples:
        >>> from datetime import datetime
        >>> t = [datetime(2020, 1, 1), datetime(2020, 1, 2)]
        >>> a = pl.DataFrame({"subject_id": [1, 2], "prediction_time": t})
        >>> _check_rows_aligned(a, a.with_columns(pl.lit(0).alias("x")))
        >>> _check_rows_aligned(a, a.reverse())
        Traceback (most recent call last):
            ...
        RuntimeError: label rows are misaligned with the sequence bounds: 2 of 2 row(s) differ ...
        >>> _check_rows_aligned(a, a.head(1))
        Traceback (most recent call last):
            ...
        RuntimeError: label rows are misaligned with the sequence bounds: 2 vs 1 row(s)...
    """
    sid, pt = DataSchema.subject_id_name, LabelSchema.prediction_time_name
    if base.height != surviving.height:
        raise RuntimeError(
            f"label rows are misaligned with the sequence bounds: {base.height} vs {surviving.height} "
            "row(s); the upstream join dropped or duplicated rows."
        )
    mismatch = (base[sid] != surviving[sid]) | (base[pt] != surviving[pt])
    n_bad = int(mismatch.sum())
    if n_bad:
        raise RuntimeError(
            f"label rows are misaligned with the sequence bounds: {n_bad} of {base.height} row(s) differ "
            f"in subject_id/prediction_time (first at row {int(mismatch.arg_max())}); the upstream join "
            "no longer preserves label order."
        )


class MultitaskBoundaryPytorchDataset(MEDSPytorchDataset):
    """MEDS dataset over multitask-boundary metadata parquets + packed ``.labels.npy`` sidecars."""

    @classmethod
    def get_task_seq_bounds_and_labels(cls, label_df: pl.DataFrame, schema_df: pl.DataFrame) -> pl.DataFrame:
        """Upstream seq-bounds computation + hstack of the boundary columns and the source keys.

        The upstream implementation preserves ``label_df`` input order for surviving rows (subjects
        present in ``schema_df``), so the extras are hstacked from a semi-filtered copy.  The two
        internal source columns ride along here: they are what lets ``collate`` find the packed row.
        """
        base = super().get_task_seq_bounds_and_labels(label_df, schema_df)
        extras = [
            c for c in (*MULTITASK_LABEL_COLS, *_INTERNAL_COLS) if c in label_df.collect_schema().names()
        ]
        if not extras:
            return base
        sid = DataSchema.subject_id_name
        # ``maintain_order="left"`` pins the semi join to ``label_df`` order (polars' default is
        # unspecified); the check below turns a future ordering drift into an error rather than
        # silently pairing one context's targets with another's patient window.
        surviving = label_df.join(
            schema_df.lazy().select(sid).unique().collect(), on=sid, how="semi", maintain_order="left"
        )
        _check_rows_aligned(base, surviving)
        return base.hstack(surviving.select(extras))

    def _check_code_width(self, codes: np.ndarray, what: str) -> None:
        """Refuse an index past the event vocabulary the labels were built with.

        Membership in ``code_to_index`` is not enough: it says a name is known, not that its id fits
        the table a model will embed it from.  Mirrors
        :meth:`~every_query.data.multitask_eval_dataset.QuerySeqMultitaskEvalDataset._check_codes`.

        Defence in depth rather than the front line.  When a real ontology is attached the width
        equality above it (``extended_vocab_size == boundary_vocab_size``) has already run, and
        ``extend_code_map`` cannot mint an id at or past ``extended_vocab_size``, so this can only
        fire on a manifest whose ``boundary_vocab_size`` disagrees with the labels it describes - a
        hand-edited one, or one written by a future sampler - and on the ``"none"``-mode path, where
        the ontology branch is skipped entirely and this is the only width check there is.
        """
        if codes.size and int(codes.max()) >= self.boundary_vocab_size:
            raise ValueError(
                f"{what} code index {int(codes.max())} in the labels at {self._split_dir} is outside "
                f"the boundary vocabulary of size {self.boundary_vocab_size} recorded in the "
                "manifest. Regenerate the labels against this cohort and ontology."
            )

    def __init__(
        self,
        cfg: MEDSTorchDataConfig,
        split: str,
        *,
        expected_vocab_size: int | None = None,
        expected_vocab_fingerprint: str | None = None,
        check_cohort_vocabulary: bool = True,
        strip_delta_tokens: bool = False,
        ontology_dir: str | Path | None = None,
    ):
        """Build the dataset.

        Args:
            cfg: Upstream MEDS torchdata config; ``task_labels_dir`` must point at the sampler's
                ``out_dir`` (the split subdirectory holds the manifest and sidecars).
            split: MEDS split name.
            expected_vocab_size: The model's output width; must equal the manifest's ``vocab_size``.
            expected_vocab_fingerprint: The vocabulary fingerprint the model was built against; must
                equal the manifest's.
            check_cohort_vocabulary: Also recompute the fingerprint from the cohort's
                ``codes.parquet`` (``cfg.code_metadata_fp``) and require it to match the manifest.
            strip_delta_tokens: As in :class:`~every_query.data.query_seq_dataset.QuerySeqPytorchDataset`.
            ontology_dir: The ``EQ_build_ontology`` directory whose node names the labels' ancestor
                start / bound / conditioning codes resolve through.  **Required** when the manifest's
                ``ontology_mode`` is not ``"none"``; harmless (and unused for the labels) otherwise,
                which is the ordinary derived-ancestor-targets setup - an ontology on the model with
                a leaf-only sampler run.
        """
        self._split_dir = Path(cfg.task_labels_dir) / split if cfg.task_labels_dir is not None else None
        if self._split_dir is None:
            raise ValueError("MultitaskBoundaryPytorchDataset requires task_labels_dir")
        self.manifest = read_manifest(self._split_dir)
        self.num_bounds = int(self.manifest["num_bounds"])
        self.vocab_size = int(self.manifest["vocab_size"])
        self.packed_width = int(self.manifest["packed_width_bytes"])
        self.vocab_fingerprint = str(self.manifest["vocab_fingerprint"])
        # The *event* vocabulary the windows were drawn against.  Equal to ``vocab_size`` unless an
        # ancestor node acted as a start / bound / conditioning code; the target bits are leaf-only
        # in every mode, so ``vocab_size`` / ``packed_width`` below are untouched by it.
        self.ontology_mode = str(self.manifest["ontology_mode"])
        self.boundary_vocab_size = int(self.manifest.get("boundary_vocab_size") or self.vocab_size)
        # One-sided on purpose: labels that name ancestors cannot be read without the ontology that
        # numbered them, but an ontology *without* ancestor-bearing labels is the normal
        # derived-targets configuration (leaf sampler, ancestor targets derived inside the model).
        if self.ontology_mode != ONTOLOGY_MODE_NONE and ontology_dir is None:
            raise ValueError(
                f"the multitask manifest at {self._split_dir} has ontology_mode "
                f"{self.ontology_mode!r}, so its start / bound / conditioning codes may name ontology "
                "nodes, but no ontology_dir was given; pass the directory the labels were sampled "
                "with."
            )
        if expected_vocab_size is not None and expected_vocab_size != self.vocab_size:
            raise ValueError(
                f"vocab_size mismatch: the model expects {expected_vocab_size} but the multitask manifest at "
                f"{self._split_dir} was built with V={self.vocab_size}. Regenerate the labels or fix the "
                "model."
            )
        if expected_vocab_fingerprint is not None and expected_vocab_fingerprint != self.vocab_fingerprint:
            raise ValueError(
                f"vocabulary fingerprint mismatch: expected {expected_vocab_fingerprint[:12]}... but the "
                f"manifest records {self.vocab_fingerprint[:12]}.... The labels were built against a "
                "different "
                "codes.parquet."
            )
        if check_cohort_vocabulary:
            v, fp = _vocab_fingerprint_from_codes_parquet(cfg.code_metadata_fp)
            if v != self.vocab_size or fp != self.vocab_fingerprint:
                raise ValueError(
                    f"The cohort vocabulary at {cfg.code_metadata_fp} (V={v}, {fp[:12]}...) does not match "
                    f"the multitask manifest (V={self.vocab_size}, {self.vocab_fingerprint[:12]}...). The "
                    "labels were "
                    "generated against a different codes.parquet; regenerate them for this cohort."
                )

        self._label_files: dict[str, Path] = {}
        self._memmaps: dict[str, np.ndarray] = {}

        super().__init__(cfg, split)

        schema_cols = self.schema_df.collect_schema().names()
        missing = [c for c in (*MULTITASK_LABEL_COLS, *_INTERNAL_COLS) if c not in schema_cols]
        if missing:
            raise ValueError(
                f"MultitaskBoundaryPytorchDataset requires columns {[*MULTITASK_LABEL_COLS]} (plus internal "
                f"source keys); missing {missing}. Generate labels with EQ_generate_multitask_sequences."
            )

        code_meta = pl.read_parquet(
            cfg.code_metadata_fp, columns=["code", "code/vocab_index"], use_pyarrow=True
        )
        self.code_to_index: dict[str, int] = {
            c: int(i)
            for c, i in zip(code_meta["code"].to_list(), code_meta["code/vocab_index"].to_list(), strict=True)
        }
        self._closure = None
        self._descendant_leaves: torch.Tensor | None = None
        self._descendant_offsets: torch.Tensor | None = None
        if ontology_dir is not None and self.ontology_mode != ONTOLOGY_MODE_NONE:
            from every_query.data.ontology import (
                check_ontology_cohort,
                closure_fingerprint,
                extend_code_map,
                extended_vocab_size,
                load_closure_index,
            )

            # Identity, not width: a same-width ontology of another cohort - or of these codes at
            # permuted indices - would pair every ancestor name with the wrong descendants.  Checked
            # against the *base* map, before the extension adds the nodes being vouched for.
            check_ontology_cohort(ontology_dir, code_to_index=self.code_to_index)
            # ``read_manifest`` has already refused an ancestor mode without one, so this is never
            # skipped for a manifest that claims to use ancestors.
            recorded = self.manifest.get("ontology_fingerprint")
            actual = closure_fingerprint(ontology_dir)
            if recorded and actual != recorded:
                raise ValueError(
                    f"the ontology at {ontology_dir} has closure {actual[:12]}... but these labels "
                    f"were sampled against {recorded[:12]}...; the same ancestor name would stand "
                    "for a different set of descendants."
                )
            # The model's embedding table is sized from this same ontology, so its width is the one
            # number that says an ancestor id in the labels is embeddable.  Comparing it here turns
            # an out-of-range lookup - a device-side assert with no attribution on CUDA - into a
            # sentence, and catches an ontology grown or pruned since the labels were sampled.
            v_ext = extended_vocab_size(ontology_dir)
            if v_ext != self.boundary_vocab_size:
                raise ValueError(
                    f"the ontology at {ontology_dir} is {v_ext} wide but the labels at "
                    f"{self._split_dir} were sampled against a boundary vocabulary of "
                    f"{self.boundary_vocab_size}; a model sized from this ontology could not embed "
                    "their ancestor codes."
                )
            self.code_to_index = extend_code_map(self.code_to_index, ontology_dir)
            # Loaded for every ontology mode, not just the conditioning ones: ``collate`` needs it to
            # re-derive an ancestor conditioning answer as the OR over its descendant leaves, and a
            # *supplied* index may carry an ancestor conditioning code even in a mode whose draw pool
            # would never produce one.  Without it that gather would run off the leaf block.
            self._closure = load_closure_index(
                ontology_dir,
                base_vocab_size=self.vocab_size,
                vocab_fingerprint=self.vocab_fingerprint,
            )
            # The closure as a CSR-style ancestor -> descendant-leaf index, so ``collate`` can OR the
            # handful of ancestor columns a batch actually names instead of materialising the whole
            # (B, K-1, V_ext) block to read back B*(K-1) bits.  Two int64 vectors; pickles to the
            # dataloader workers with the rest of the dataset.
            anc = self._closure.ancestor_ids - self.vocab_size
            order = torch.argsort(anc, stable=True)
            self._descendant_leaves = self._closure.leaf_ids[order]
            self._descendant_offsets = torch.searchsorted(
                anc[order], torch.arange(self._closure.n_ancestors + 1)
            )

        n = self.schema_df.height
        durations = self.schema_df[DURATIONS_COL]
        bound_events = self.schema_df[BOUND_EVENTS_COL]
        if (
            n
            and not (
                (durations.list.len() == self.num_bounds) & (bound_events.list.len() == self.num_bounds)
            ).all()
        ):
            raise ValueError(f"every label row must carry exactly {self.num_bounds} durations/bound_events")
        self._q_durations = (
            np.asarray(durations.explode().to_numpy(), dtype=np.float32).reshape(n, self.num_bounds)
            if n
            else np.zeros((0, self.num_bounds), dtype=np.float32)
        )
        # Map codes -> indices in polars (no per-slot Python objects): null slots -> NO_BOUND_INDEX,
        # non-null codes outside the vocabulary -> -1, which is an error.
        flat_bounds = bound_events.explode()
        mapped = pl.select(
            pl.when(flat_bounds.is_null())
            .then(NO_BOUND_INDEX)
            .otherwise(flat_bounds.replace_strict(self.code_to_index, default=-1, return_dtype=pl.Int64))
        ).to_series()
        unknown = flat_bounds.filter(mapped == -1).unique().sort().to_list()
        if unknown:
            raise ValueError(
                f"{len(unknown)} boundary code(s) in the labels are not in this cohort's vocabulary: "
                f"{unknown[:10]}. Regenerate the labels against this cohort's codes.parquet."
            )
        self._q_bound_codes = mapped.to_numpy().astype(np.int64).reshape(n, self.num_bounds)
        self._check_code_width(self._q_bound_codes, "boundary")
        sentinel_ok = (self._q_bound_codes != NO_BOUND_INDEX) == (
            self._q_durations == EVENT_BOUND_DURATION_SENTINEL
        )
        if not sentinel_ok.all():
            raise ValueError("durations/bound_events disagree on which slots are event-bounded")

        # Issue #24: sampler-materialized window starts (``labels_df`` filled prediction-time starts
        # for legacy parquets).  Same vocabulary as the targets; PAD / unknown start codes are errors.
        start_durations = self.schema_df[START_DURATIONS_COL]
        start_events = self.schema_df[START_EVENTS_COL]
        if (
            n
            and not (
                (start_durations.list.len() == self.num_bounds) & (start_events.list.len() == self.num_bounds)
            ).all()
        ):
            raise ValueError(
                f"every label row must carry exactly {self.num_bounds} start_durations/start_events"
            )
        self._q_start_durations = (
            np.asarray(start_durations.explode().to_numpy(), dtype=np.float32).reshape(n, self.num_bounds)
            if n
            else np.zeros((0, self.num_bounds), dtype=np.float32)
        )
        flat_starts = start_events.explode()
        start_mapped = flat_starts.replace_strict(self.code_to_index, default=-1, return_dtype=pl.Int64)
        bad = flat_starts.filter(flat_starts.is_not_null() & (start_mapped <= 0)).unique().sort().to_list()
        if bad:
            raise ValueError(
                f"{len(bad)} start-event code(s) in the labels are PAD or not in this cohort's vocabulary: "
                f"{bad[:10]}. Regenerate the labels against this cohort's codes.parquet."
            )
        self._q_start_codes = (
            pl.select(pl.when(flat_starts.is_null()).then(NO_BOUND_INDEX).otherwise(start_mapped))
            .to_series()
            .to_numpy()
            .astype(np.int64)
            .reshape(n, self.num_bounds)
        )
        self._check_code_width(self._q_start_codes, "start-event")
        by_event = self._q_start_codes != NO_BOUND_INDEX
        start_ok = np.where(
            by_event,
            self._q_start_durations == EVENT_BOUND_DURATION_SENTINEL,
            self._q_start_durations >= 0,
        )
        if not start_ok.all():
            raise ValueError(
                "start_durations/start_events disagree on which slots are event-defined: each slot must be "
                f"either start_duration >= 0 with a null start_event, or start_duration == "
                f"{EVENT_BOUND_DURATION_SENTINEL} with a non-null start_event"
            )

        # Issue #22: K-1 sampler-materialized conditioning codes/answers. Loaded and validated only;
        # never sampled here. Answers are cross-checked against the unpacked bits in ``collate``.
        kc = self.num_bounds - 1
        cond_codes = self.schema_df[CONDITION_CODES_COL]
        cond_answers = self.schema_df[CONDITION_ANSWERS_COL]
        if n and not ((cond_codes.list.len() == kc) & (cond_answers.list.len() == kc)).all():
            raise ValueError(f"every label row must carry exactly {kc} condition_codes/condition_answers")
        if n and kc:
            flat_cond = cond_codes.explode()
            cond_mapped = flat_cond.replace_strict(self.code_to_index, default=-1, return_dtype=pl.Int64)
            bad = flat_cond.filter(cond_mapped.is_null() | (cond_mapped <= 0)).unique().sort().to_list()
            if bad:
                raise ValueError(
                    f"{len(bad)} condition code(s) in the labels are null, PAD, or not in this cohort's "
                    f"vocabulary: {bad[:10]}. Regenerate the labels against this cohort's codes.parquet."
                )
            self._condition_codes = cond_mapped.to_numpy().astype(np.int64).reshape(n, kc)
            self._check_code_width(self._condition_codes, "condition")
            flat_ans = cond_answers.explode()
            if flat_ans.null_count():
                raise ValueError("condition_answers must not contain nulls")
            self._condition_answers = flat_ans.to_numpy().astype(bool).reshape(n, kc)
        else:
            self._condition_codes = np.zeros((n, kc), dtype=np.int64)
            self._condition_answers = np.zeros((n, kc), dtype=bool)

        # Shard keys are stored once; each row carries a compact integer id, not a Python string.
        self._shard_keys: list[str] = sorted(self._label_files)
        self._source_shard = (
            self.schema_df[SOURCE_SHARD_COL]
            .replace_strict({k: i for i, k in enumerate(self._shard_keys)}, return_dtype=pl.Int32)
            .to_numpy()
        )
        self._source_row = self.schema_df[SOURCE_ROW_COL].to_numpy().astype(np.int64)

        # Every referenced sidecar must exist and have the manifest's packed shape; check the header
        # only (np.load with mmap_mode reads no payload).
        for shard_id in np.unique(self._source_shard).tolist():
            key = self._shard_keys[shard_id]
            fp = self._label_files[key]
            if not fp.exists():
                raise FileNotFoundError(f"packed labels sidecar {fp} for metadata shard {key!r} is missing")
            shape = tuple(np.load(fp, mmap_mode="r").shape)
            if shape[1:] != (self.num_bounds, self.packed_width):
                raise ValueError(
                    f"{fp} has packed shape {shape}, expected (rows, {self.num_bounds}, {self.packed_width})"
                )
            rows = self._source_row[self._source_shard == shard_id]
            if rows.size and int(rows.max()) >= shape[0]:
                raise ValueError(
                    f"{fp} has {shape[0]} rows but the metadata references row {int(rows.max())}"
                )

        self.strip_delta_tokens = strip_delta_tokens
        from every_query.data import rope_time

        self.delta_ids = rope_time.delta_vocab_ids(self.code_to_index)

    # -- label loading ---------------------------------------------------------------------------

    @property
    def labels_df(self) -> pl.DataFrame:
        """Metadata rows only, each tagged with its source shard key and physical row offset."""
        if not self.has_task_index:
            return None
        required = [LabelSchema.subject_id_name, LabelSchema.prediction_time_name]

        frames = []
        # Only this split's files: the manifest and sidecars live in the split subdirectory, and the
        # upstream rglob would otherwise also read every other split's metadata just to drop it.
        fps = [fp for fp in self.config.task_labels_fps if self._split_dir in fp.parents]
        if not fps:
            raise FileNotFoundError(f"no metadata parquets under {self._split_dir}")
        for fp in fps:
            key = self._source_key(fp)
            self._label_files[key] = fp.with_name(fp.name[: -len(fp.suffix)] + LABELS_SUFFIX)
            schema = pl.read_parquet_schema(fp)
            missing = [c for c in (*required, *REQUIRED_LABEL_COLS) if c not in schema]
            if missing:
                raise ValueError(
                    f"{fp} is missing required column(s) {missing}; regenerate the labels with the "
                    "current EQ_generate_multitask_sequences."
                )
            # Issue #24 start columns: optional per parquet.  A pre-#24 file lacks both and is read as
            # prediction-time starts, typed here so a split mixing v2 and v3 shards concatenates.
            present = [c for c in START_COLS if c in schema]
            if present and len(present) != len(START_COLS):
                raise ValueError(
                    f"{fp} has {present} but not all of {list(START_COLS)}; regenerate the labels."
                )
            df = pl.read_parquet(fp, columns=[*required, *REQUIRED_LABEL_COLS, *present], use_pyarrow=True)
            if not present:
                df = df.with_columns(
                    pl.repeat(0.0, self.num_bounds, dtype=pl.Float32).implode().alias(START_DURATIONS_COL),
                    pl.repeat(None, self.num_bounds, dtype=pl.Utf8).implode().alias(START_EVENTS_COL),
                )
            frames.append(
                df.select(*required, *MULTITASK_LABEL_COLS)
                .with_row_index(SOURCE_ROW_COL)
                .with_columns(
                    pl.col(SOURCE_ROW_COL).cast(pl.Int64),
                    pl.lit(key).alias(SOURCE_SHARD_COL),
                )
            )
        logger.info(f"Reading multitask boundary metadata from {fps}")
        return pl.concat(frames, how="vertical")

    def _source_key(self, fp: Path) -> str:
        """The metadata parquet's identity: its path relative to ``task_labels_dir``, minus the suffix.

        Relative to the labels root (not the split dir) so ``train/0`` and ``tuning/0`` never collide.
        """
        return str(fp.relative_to(Path(self.config.task_labels_dir)).with_suffix(""))

    def _memmap(self, key: str) -> np.ndarray:
        mm = self._memmaps.get(key)
        if mm is None:
            mm = np.load(self._label_files[key], mmap_mode="r")
            self._memmaps[key] = mm
        return mm

    def __getstate__(self) -> dict:
        state = super().__getstate__()
        state["_memmaps"] = {}  # DataLoader workers reopen their own read-only maps
        return state

    # -- items + collate -------------------------------------------------------------------------

    def _seeded_getitem(self, idx: int, seed: int | None = None) -> dict[str, torch.Tensor]:
        out = super()._seeded_getitem(idx, seed)
        idx = range(len(self._source_row))[idx]
        out["q_start_durations"] = self._q_start_durations[idx]
        out["q_start_codes"] = self._q_start_codes[idx]
        out["q_durations"] = self._q_durations[idx]
        out["q_bound_codes"] = self._q_bound_codes[idx]
        out["condition_codes"] = self._condition_codes[idx]
        out["condition_answers"] = self._condition_answers[idx]
        out[SOURCE_SHARD_COL] = self._shard_keys[self._source_shard[idx]]
        out[SOURCE_ROW_COL] = int(self._source_row[idx])
        return out

    def gather_packed(self, batch: list[dict]) -> np.ndarray:
        """``(B, K, ceil(V/8))`` uint8 rows gathered from the read-only memmaps, one slice per shard."""
        B = len(batch)
        packed = np.empty((B, self.num_bounds, self.packed_width), dtype=np.uint8)
        keys = np.asarray([item[SOURCE_SHARD_COL] for item in batch], dtype=object)
        rows = np.asarray([item[SOURCE_ROW_COL] for item in batch], dtype=np.int64)
        for key in dict.fromkeys(keys.tolist()):
            sel = np.flatnonzero(keys == key)
            packed[sel] = self._memmap(key)[rows[sel]]
        return packed

    def unpack_targets(self, packed: np.ndarray) -> torch.BoolTensor:
        dense = np.unpackbits(packed, axis=-1, count=self.vocab_size, bitorder=BITORDER)
        return torch.from_numpy(dense.astype(bool, copy=False))

    def _ancestor_answers(
        self,
        leaf_targets: torch.Tensor,
        condition_codes: torch.LongTensor,
        ancestor: torch.Tensor,
        expect: torch.BoolTensor,
    ) -> torch.BoolTensor:
        """Fill ``expect`` at the ancestor slots with the OR over each code's descendant leaves.

        Deliberately not ``derive_ancestor_targets``: that is the right shape for the model, which
        needs every ``V_ext`` column for its loss, but here at most ``B * (K-1)`` bits are wanted and
        widening the whole block to read them costs a transient ``(B, K-1, V_ext)`` allocation per
        batch, in a dataloader worker, on CPU.  One pass per *distinct* ancestor code in the batch -
        typically a handful - reads the same bits from the leaf block directly.
        """
        if self._closure is None:
            raise ValueError(
                "the labels name ontology nodes as conditioning codes but this dataset has no "
                "closure; construct it with the ontology_dir the labels were sampled with."
            )
        expect = expect.clone()
        for code in torch.unique(condition_codes[ancestor]).tolist():
            slot = int(code) - self.vocab_size
            lo, hi = int(self._descendant_offsets[slot]), int(self._descendant_offsets[slot + 1])
            here = ancestor & (condition_codes == code)
            if lo == hi:  # a node with no descendant leaf can never be true
                expect[here] = False
                continue
            b, j = here.nonzero(as_tuple=True)
            expect[here] = leaf_targets[b, j][:, self._descendant_leaves[lo:hi]].any(dim=-1)
        return expect

    def collate(self, batch: list[dict]) -> MultitaskBoundaryBatch:
        out = dict(super().collate(batch).items())
        out.pop("boolean_value", None)

        time_pos_ids = None
        if self.strip_delta_tokens:
            from every_query.data.query_seq_dataset import QuerySeqPytorchDataset

            time_pos_ids = QuerySeqPytorchDataset._apply_rope_time(self, out)

        B = len(batch)
        # Start tensors are always emitted (zeros / NO_BOUND_INDEX for legacy files), exactly float32 / int64.
        q_start_durations = torch.from_numpy(
            np.stack([item["q_start_durations"] for item in batch]).astype(np.float32)
        )
        q_start_codes = torch.from_numpy(np.stack([item["q_start_codes"] for item in batch]).astype(np.int64))
        q_durations = torch.from_numpy(np.stack([item["q_durations"] for item in batch]).astype(np.float32))
        q_bound_codes = torch.from_numpy(np.stack([item["q_bound_codes"] for item in batch]).astype(np.int64))
        q_mask = torch.ones(B, self.num_bounds, dtype=torch.bool)
        targets = self.unpack_targets(self.gather_packed(batch))
        condition_codes = torch.from_numpy(
            np.stack([item["condition_codes"] for item in batch]).astype(np.int64)
        )
        condition_answers = torch.from_numpy(np.stack([item["condition_answers"] for item in batch]))

        # The stored answer must be the target bit of its code at its boundary (the sampler's
        # contract).  For an ancestor code there is no such column - ``targets`` is leaf-only - and
        # the contract is instead the OR over the code's closure descendants, the same value the
        # model's ``derive_ancestor_targets`` puts in that column.
        kc = self.num_bounds - 1
        if kc:
            leaf = torch.clamp(condition_codes, max=self.vocab_size - 1)
            expect = targets[:, :kc].gather(2, leaf.unsqueeze(-1)).squeeze(-1)
            ancestor = condition_codes >= self.vocab_size
            if bool(ancestor.any()):
                expect = self._ancestor_answers(targets[:, :kc], condition_codes, ancestor, expect)
            if not torch.equal(expect, condition_answers):
                bad = (expect != condition_answers).nonzero()[:5].tolist()
                raise ValueError(
                    f"condition_answers disagree with the packed targets at (batch, slot) {bad}; the "
                    "metadata and .labels.npy sidecar are inconsistent. Regenerate the labels."
                )

        return MultitaskBoundaryBatch(
            **out,
            q_start_durations=q_start_durations,
            q_start_codes=q_start_codes,
            q_durations=q_durations,
            q_bound_codes=q_bound_codes,
            q_mask=q_mask,
            targets=targets,
            condition_codes=condition_codes,
            condition_answers=condition_answers,
            time_pos_ids=time_pos_ids,
        )
