"""Dataset + batch types for conditional query sequences.

The on-disk label rows follow :class:`~every_query.data.schema.QuerySeqSchema`: one row per
``(subject_id, prediction_time)`` context carrying three aligned list columns —

- ``queries``: list of MEDS code strings (any vocabulary code, in random order — including the
  end-of-timeline code :data:`EOS_CODE`, which is an ordinary code, not a special sentinel);
- ``durations``: list of float horizon lengths (days);
- ``answers``: list of booleans.  ``answers[j]`` is simply *"was ``queries[j]`` observed strictly
  inside its window?"* — with the default prediction-time start and no bound event that window is
  ``(prediction_time, prediction_time + durations[j])``; a bound event closes it at that code's next
  occurrence after the (resolved) start instead — binary, never null.  An event we
  could not observe (because the record ends first) is ``False``; censoring is expressed by a
  separate ``TIMELINE//END`` query rather than a null answer.

The dataset tensorizes each sequence into fixed-position tensors padded to the batch's longest
query list:

- ``q_codes``     (B, L) long — vocab indices; padding is 0;
- ``q_durations`` (B, L) float — horizon in days, 0 at padding;
- ``q_answers``   (B, L) long — teacher-forcing classes ``ANSWER_NO`` / ``ANSWER_YES`` (padding
  holds ``ANSWER_NO``; it is ignored under ``q_mask``);
- ``q_mask``      (B, L) bool — True at real (non-padding) query positions.

Two optional label columns refine a query's window and ride along when present: ``bound_events``
(the window ends at the next occurrence of that code; ``q_bound_codes`` in the batch) and the
issue-#27 pair ``start_durations`` / ``start_events`` (the window *opens* later than the prediction
time).  The ordinary sequence models do not encode starts, so the dataset rejects any *active*
start unless built with ``allow_active_starts=True``, in which case the batch also carries
``q_start_durations`` / ``q_start_codes`` ``(B, L)`` for the multitask prediction adapter.

Unlike :class:`~every_query.data.dataset.EveryQueryPytorchDataset`, **no query token is prepended
to the patient event sequence** — the patient sequence feeds the bidirectional encoder unchanged,
and queries live exclusively in the decoder stream.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch
from meds import DataSchema, LabelSchema
from meds_torchdata import MEDSPytorchDataset
from meds_torchdata.config import MEDSTorchDataConfig
from meds_torchdata.types import MEDSTorchBatch

from every_query.data import rope_time
from every_query.model.answers import ANSWER_NO, ANSWER_YES

logger = logging.getLogger(__name__)

# End-of-timeline code: a real MEDS vocabulary code emitted once per subject at the record's last
# timestamp.  A query ``(EOS_CODE, d)`` therefore answers "does the record end within d?" — the
# mechanism by which these models handle censoring: there is no separate "censored" answer class,
# so censoring is expressed as an ordinary query (see :mod:`every_query.model.answers`).
EOS_CODE = "TIMELINE//END"

QUERIES_COL = "queries"
DURATIONS_COL = "durations"
ANSWERS_COL = "answers"
SEQ_LABEL_COLS = (QUERIES_COL, DURATIONS_COL, ANSWERS_COL)

# Event-bounded queries: a per-query boundary code replaces the scalar horizon.  Deliberately
# NOT in SEQ_LABEL_COLS — that tuple is the *required* set, and a dataset generated before this
# feature existed must keep loading.
BOUND_EVENTS_COL = "bound_events"
# Explicit window starts (issue #27): ``start_durations[j]`` days after the prediction time (0.0 =
# the prediction time itself, the legacy form), or ``EVENT_BOUND_DURATION_SENTINEL`` with a
# non-null ``start_events[j]`` for a window that opens at that event's next occurrence.  Present
# together or absent together; absent reads as ``[0.0] * K`` / ``[None] * K``.
START_DURATIONS_COL = "start_durations"
START_EVENTS_COL = "start_events"
START_COLS = (START_DURATIONS_COL, START_EVENTS_COL)
# Designed conditioning answers: ``forced_answers[j]`` (true/false) replaces ``answers[j]`` as the
# answer fed to the model when query ``j`` conditions a later one; null teacher-forces the truth.
# ``answers`` itself stays the labeled truth, so the scoring label is never affected.  Null on every
# row's final query, which is scored and never conditions anything.
FORCED_ANSWERS_COL = "forced_answers"
OPTIONAL_SEQ_LABEL_COLS = (BOUND_EVENTS_COL, *START_COLS, FORCED_ANSWERS_COL)
ALL_SEQ_LABEL_COLS = (*SEQ_LABEL_COLS, *OPTIONAL_SEQ_LABEL_COLS)

# Duration written for an event-bounded query.  The window is defined by the boundary event, so
# there is no horizon; a negative sentinel makes an accidental use as a horizon obvious rather
# than plausible.  Downstream bucketing must special-case it (see evaluate_multitask, which keys
# on the whole window spec rather than on the duration alone).
EVENT_BOUND_DURATION_SENTINEL = -1.0

# Vocabulary index meaning "this query has no boundary event".  Shares PAD_INDEX = 0, which is
# never a real query code.
NO_BOUND_INDEX = 0


@dataclass
class QuerySeqBatch(MEDSTorchBatch):
    """MEDS batch extended with per-sample query-sequence tensors.

    Examples:
        >>> batch = QuerySeqBatch(
        ...     code=torch.tensor([[1, 2, 3], [4, 5, 0]]),
        ...     numeric_value=torch.zeros(2, 3),
        ...     numeric_value_mask=torch.zeros(2, 3, dtype=torch.bool),
        ...     time_delta_days=torch.zeros(2, 3),
        ...     q_codes=torch.tensor([[9, 7], [9, 0]]),
        ...     q_durations=torch.tensor([[30.0, 7.0], [365.0, 0.0]]),
        ...     q_answers=torch.tensor([[1, 0], [0, 0]]),
        ...     q_mask=torch.tensor([[True, True], [True, False]]),
        ... )
        >>> batch.n_queries
        2
        >>> batch.q_mask.sum().item()
        3

        Mismatched query-tensor shapes raise:

        >>> QuerySeqBatch(
        ...     code=torch.tensor([[1, 2], [3, 4]]),
        ...     numeric_value=torch.zeros(2, 2),
        ...     numeric_value_mask=torch.zeros(2, 2, dtype=torch.bool),
        ...     time_delta_days=torch.zeros(2, 2),
        ...     q_codes=torch.tensor([[9, 7]]),
        ...     q_durations=torch.tensor([[30.0, 7.0]]),
        ...     q_answers=torch.tensor([[1, 0]]),
        ...     q_mask=torch.tensor([[True, True]]),
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Expected shape (2, 2) for q_codes, but got torch.Size([1, 2])!
    """

    q_codes: torch.LongTensor | None = None
    q_durations: torch.FloatTensor | None = None
    q_answers: torch.LongTensor | None = None
    q_mask: torch.BoolTensor | None = None
    # Per-query boundary-event vocab indices; ``NO_BOUND_INDEX`` (0) means "time-bounded".
    # ``None`` for a dataset whose labels carry no ``bound_events`` column at all.
    q_bound_codes: torch.LongTensor | None = None
    # Per-query window starts (issue #27): days after the prediction time (``0.0`` = the
    # prediction time, ``EVENT_BOUND_DURATION_SENTINEL`` = event-defined) and the start event's
    # vocab index (``NO_BOUND_INDEX`` for a duration start).  Filled only by a dataset built with
    # ``allow_active_starts=True`` (the multitask prediction adapter); the ordinary sequence models
    # do not encode starts, so their batches carry ``None`` here.  Given together or not at all.
    q_start_durations: torch.FloatTensor | None = None
    q_start_codes: torch.LongTensor | None = None
    # Per-patient-token elapsed time in integer hours, for rotary position encoding.  Present
    # only when the dataset was built with ``strip_delta_tokens=True``; ``None`` leaves the
    # encoder on ordinary token-index positions.  Shape matches ``code``, not the query tensors.
    time_pos_ids: torch.LongTensor | None = None

    LABEL_TENSOR_NAMES: ClassVar[tuple[str]] = (
        "boolean_value",
        "q_codes",
        "q_durations",
        "q_answers",
        "q_bound_codes",
        "q_start_durations",
        "q_start_codes",
    )

    @property
    def n_queries(self) -> int | None:
        return None if self.q_codes is None else self.q_codes.shape[1]

    def __post_init__(self):
        super().__post_init__()
        if (self.q_start_durations is None) != (self.q_start_codes is None):
            missing = "q_start_durations" if self.q_start_durations is None else "q_start_codes"
            raise ValueError(
                f"q_start_durations and q_start_codes must be given together (got {missing}=None)"
            )
        if self.q_codes is None:
            return
        expected = (self.batch_size, self.q_codes.shape[1])
        names = ["q_codes", "q_durations", "q_answers", "q_mask"]
        if self.q_bound_codes is not None:
            names.append("q_bound_codes")
        if self.q_start_durations is not None:
            names += ["q_start_durations", "q_start_codes"]
        for name in names:
            tensor = getattr(self, name)
            if tensor is None or tuple(tensor.shape) != expected:
                got = None if tensor is None else tensor.shape
                raise ValueError(f"Expected shape {expected} for {name}, but got {got}!")


def _ragged(series: pl.Series) -> tuple[np.ndarray, np.ndarray]:
    """``(offsets, values)`` of a list column: row ``i`` is ``values[offsets[i]:offsets[i+1]]``.

    Examples:
        >>> offsets, values = _ragged(pl.Series([["a", "b"], [], ["c"]]))
        >>> offsets.tolist(), values.tolist()
        ([0, 2, 2, 3], ['a', 'b', 'c'])
    """
    lengths = series.list.len().fill_null(0).to_numpy()
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    # Drop empty/null lists before exploding: they would otherwise yield a null row each.
    values = series.filter(series.list.len() > 0).explode().to_numpy()
    if len(values) != offsets[-1]:
        raise ValueError(f"list column exploded to {len(values)} values but lengths sum to {offsets[-1]}")
    return offsets, values


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


# Per-token fields of the dynamic stream that a delta-token strip must compact together with
# ``code``.  ``static_mask`` is the prepend-mode marker over that same stream; ``static_code`` /
# ``static_numeric_value`` / ``static_numeric_value_mask`` are the *separate* static table and
# must never be touched, however wide they happen to be.
_PER_TOKEN_FIELDS = (
    "code",
    "numeric_value",
    "numeric_value_mask",
    "time_delta_days",
    "event_mask",
    "static_mask",
)


class QuerySeqPytorchDataset(MEDSPytorchDataset):
    """MEDS dataset over query-sequence label parquets (see module docstring for tensor shapes)."""

    @classmethod
    def get_task_seq_bounds_and_labels(cls, label_df: pl.DataFrame, schema_df: pl.DataFrame) -> pl.DataFrame:
        """Upstream seq-bounds computation + hstack of the query-sequence list columns.

        Mirrors ``EveryQueryPytorchDataset.get_task_seq_bounds_and_labels``: the upstream
        implementation preserves ``label_df`` input order for surviving rows (subjects present in
        ``schema_df``), so the extras can be hstacked from a semi-filtered copy of ``label_df``.
        """
        base = super().get_task_seq_bounds_and_labels(label_df, schema_df)
        extras = [c for c in ALL_SEQ_LABEL_COLS if c in label_df.collect_schema().names()]
        if not extras:
            return base
        sid = DataSchema.subject_id_name
        # ``maintain_order="left"`` pins the semi join to ``label_df`` order (polars' default is
        # unspecified); the check below turns a future ordering drift into an error rather than
        # silently pairing one context's queries with another's patient window.
        surviving = label_df.join(
            schema_df.lazy().select(sid).unique().collect(), on=sid, how="semi", maintain_order="left"
        )
        _check_rows_aligned(base, surviving)
        return base.hstack(surviving.select(extras))

    def __init__(
        self,
        cfg: MEDSTorchDataConfig,
        split: str,
        *,
        strip_delta_tokens: bool = False,
        ontology_dir: str | None = None,
        allow_active_starts: bool = False,
        allow_forced_answers: bool = False,
    ):
        """Build the dataset.

        Args:
            cfg: Upstream MEDS torchdata config.
            split: MEDS split name.
            strip_delta_tokens: When True, drop ``TIMELINE//DELTA*`` tokens from the encoder
                input at collate time and emit :attr:`QuerySeqBatch.time_pos_ids`
                (elapsed integer hours per surviving token) for rotary position encoding.
                Pair with a model built ``use_rope_time=True``; see
                :mod:`every_query.data.rope_time`.
            ontology_dir: When set, ancestor node names from the ontology's ``ontology_vocab.parquet``
                are added to the query vocabulary, so a query may name a whole class rather
                than one leaf code.  Must be the same directory the model was built with —
                the indices have to agree, or a query would address the wrong embedding row.
            allow_active_starts: Opt in to tensorizing the optional ``start_durations`` /
                ``start_events`` label columns (issue #27) into
                :attr:`QuerySeqBatch.q_start_durations` / ``q_start_codes``.  A model that does
                not encode window starts must never be handed them, so with the default
                ``False`` a labels directory carrying any *active* start (a positive duration or
                an event start) is rejected at init rather than silently scored as if every window
                opened at the prediction time.  Absent or all-default (``0.0`` / null) starts are
                always accepted.  Only the multitask prediction adapter passes ``True``.
            allow_forced_answers: Opt in to the optional ``forced_answers`` label column, emitted
                per item as ``condition_answers`` (the forced value where one is given, else the
                labeled answer).  A model whose batch uses one answers tensor as both teacher-forcing
                input and target cannot honor a forced answer, so with the default ``False`` a
                grid carrying any non-null one is rejected at init.  An absent or all-null column
                is always accepted.  Only the multitask prediction adapter passes ``True``.
        """
        super().__init__(cfg, split)

        schema_cols = self.schema_df.collect_schema().names()
        missing = [c for c in SEQ_LABEL_COLS if c not in schema_cols]
        if missing:
            raise ValueError(
                f"QuerySeqPytorchDataset requires query-sequence label columns "
                f"{list(SEQ_LABEL_COLS)}; missing {missing}.  Generate labels with "
                f"EQ_generate_evaluation_query_sequences."
            )

        self.queries = self.schema_df[QUERIES_COL]
        self.durations = self.schema_df[DURATIONS_COL]
        self.answers = self.schema_df[ANSWERS_COL]

        # Event bounds are opt-in *in the data*: a labels directory generated before the feature
        # existed simply has no such column, and the batch then carries q_bound_codes=None.
        self.has_bound_events = BOUND_EVENTS_COL in schema_cols
        self.bound_events = self.schema_df[BOUND_EVENTS_COL] if self.has_bound_events else None

        # Window starts (issue #27) are likewise keyed on the data, but the two columns must travel
        # together: one without the other is a half-written grid, not a legacy one.
        present_starts = [c for c in START_COLS if c in schema_cols]
        if present_starts and len(present_starts) != len(START_COLS):
            raise ValueError(
                f"the labels carry {present_starts} but not all of {list(START_COLS)}; the two start "
                "columns must be present together or absent together."
            )
        self.has_starts = bool(present_starts)
        self.allow_active_starts = allow_active_starts
        self.has_forced_answers = FORCED_ANSWERS_COL in schema_cols
        self.allow_forced_answers = allow_forced_answers
        self.start_durations = self.schema_df[START_DURATIONS_COL] if self.has_starts else None
        self.start_events = self.schema_df[START_EVENTS_COL] if self.has_starts else None

        code_meta = pl.read_parquet(
            self.config.code_metadata_fp, columns=["code", "code/vocab_index"], use_pyarrow=True
        )
        self.code_to_index: dict[str, int] = {
            c: int(i)
            for c, i in zip(code_meta["code"].to_list(), code_meta["code/vocab_index"].to_list(), strict=True)
        }
        # The conditional model expresses censoring by querying the end-of-timeline code; a
        # production cohort should have it.  We warn rather than raise so the dataset still works
        # on cohorts (e.g. tiny test fixtures) that lack it — censoring queries simply aren't
        # available there.  ``eos_query_index`` is ``None`` when absent.
        if EOS_CODE not in self.code_to_index:
            logger.warning(
                "End-of-timeline code %r is not in the cohort vocabulary; censoring-as-a-query "
                "is unavailable for this cohort.",
                EOS_CODE,
            )
        self.ontology_dir = ontology_dir
        if ontology_dir is not None:
            from every_query.data.ontology import check_ontology_cohort, extend_code_map

            # The ontology's ancestor indices and the model's mix rows are only meaningful relative
            # to the leaf indices the ontology was built with, so its observed nodes must be exactly
            # this cohort's ``codes.parquet`` rows - not merely the same number of them.
            check_ontology_cohort(ontology_dir, code_to_index=self.code_to_index)
            n_before = len(self.code_to_index)
            self.code_to_index = extend_code_map(self.code_to_index, ontology_dir)
            logger.info(
                "Ontology: query vocabulary extended from %d codes to %d nodes (%d ancestors).",
                n_before,
                len(self.code_to_index),
                len(self.code_to_index) - n_before,
            )

        # Fail before the loader starts rather than as a KeyError deep inside collate: labels
        # generated against a different codes.parquet / ontology than this run's are the one
        # mistake gen-time validation cannot see.
        mentioned = set(self.queries.explode().drop_nulls().to_list())
        if self.has_bound_events:
            mentioned |= set(self.bound_events.explode().drop_nulls().to_list())
        if self.has_starts:
            mentioned |= set(self.start_events.explode().drop_nulls().to_list())
        unknown = sorted(mentioned - self.code_to_index.keys())
        if unknown:
            raise ValueError(
                f"{len(unknown)} query/bound/start code(s) in the labels are not in this run's vocabulary "
                f"(codes.parquet + ontology): {unknown[:10]}. Regenerate the labels with the same "
                f"query_codes/ontology_dir as this run, or pass the matching ones here."
            )

        self.eos_query_index: int | None = self.code_to_index.get(EOS_CODE)

        # Pre-encode the list columns to flat numpy here, in the parent, so DataLoader workers
        # only ever slice plain arrays.  A 10M-row run died in a forked worker on a query code
        # that read back as '' from a polars list-of-strings Series that scans clean in-process
        # (parquet, exploded column, and a full single-process getitem walk all agree); the
        # vocabulary check above already guarantees every code encodes, so this also makes an
        # out-of-vocabulary KeyError at collate time impossible.
        self._q_offsets, q = _ragged(self.queries)
        self._q_codes = np.fromiter((self.code_to_index[c] for c in q), dtype=np.int64, count=len(q))
        d_offsets, self._q_durations = _ragged(self.durations)
        a_offsets, self._q_answers = _ragged(self.answers)
        if not (np.array_equal(self._q_offsets, d_offsets) and np.array_equal(self._q_offsets, a_offsets)):
            raise ValueError("queries/durations/answers list lengths disagree in the labels.")
        self._q_durations = self._q_durations.astype(np.float32)  # copy: polars arrays are read-only
        self._q_answers = self._q_answers.astype(bool)
        if self.has_bound_events:
            b_offsets, b = _ragged(self.bound_events)
            if not np.array_equal(self._q_offsets, b_offsets):
                raise ValueError("queries/bound_events list lengths disagree in the labels.")
            self._q_bound_codes = np.fromiter(
                (NO_BOUND_INDEX if c is None else self.code_to_index[c] for c in b),
                dtype=np.int64,
                count=len(b),
            )
        if self.has_starts:
            self._q_start_durations, self._q_start_codes = self._encode_starts()
        if self.has_forced_answers:
            self._q_condition_answers = self._encode_condition_answers()

        self.strip_delta_tokens = strip_delta_tokens
        self.delta_ids = rope_time.delta_vocab_ids(self.code_to_index)
        if strip_delta_tokens:
            if self.delta_ids.numel() == 0:
                logger.warning(
                    "strip_delta_tokens=True but no %s* codes are in the cohort vocabulary; "
                    "time_pos_ids will still be emitted, but nothing is being stripped.",
                    rope_time.DELTA_TOKEN_PREFIX,
                )
            else:
                logger.info(
                    "RoPE time: stripping %d delta-token vocab ids from the encoder input.",
                    self.delta_ids.numel(),
                )

    def _encode_condition_answers(self) -> np.ndarray:
        """Validate ``forced_answers`` and fold it into the flat conditioning-answer array.

        Returns the bool array aligned with ``_q_offsets`` holding, per query, the answer to feed the
        model when that query conditions a later one: the forced value where one is given, else the
        labeled answer.  Raised here, at init: ragged lists; a forced answer on a row's **final**
        query (it is the scored one and conditions nothing, so the value would be a silent no-op
        that reads as a forced label); and — unless ``allow_forced_answers`` — any forced answer at
        all.  The generator enforces the final-query rule on specs; this is the same guard for a
        grid that did not come from it.
        """
        forced = self.schema_df[FORCED_ANSWERS_COL]
        lengths = forced.list.len().fill_null(0).to_numpy()
        if not np.array_equal(np.diff(self._q_offsets), lengths):
            raise ValueError("queries/forced_answers list lengths disagree in the labels.")
        flat = forced.filter(forced.list.len() > 0).explode()
        is_forced = flat.is_not_null().to_numpy()
        if not is_forced.any():
            return self._q_answers

        final = self._q_offsets[1:][lengths > 0] - 1
        if is_forced[final].any():
            row = int(np.flatnonzero(lengths > 0)[np.flatnonzero(is_forced[final])[0]])
            raise ValueError(
                f"labels row {row} forces the answer of its final query; the final query is the scored "
                "one and its answer is never fed to the model, so its forced_answers entry must be null."
            )
        if not self.allow_forced_answers:
            raise ValueError(
                f"{int(is_forced.sum())} query answer(s) in the labels are forced.  A model that uses "
                "one answers tensor as both its teacher-forcing input and its target cannot condition "
                "on a forced answer without also training/scoring against it, so this dataset refuses "
                "the grid instead.  Score it with EQ_predict_multitask (its adapter passes "
                "allow_forced_answers=True)."
            )
        return np.where(is_forced, flat.fill_null(False).to_numpy().astype(bool), self._q_answers)

    def _encode_starts(self) -> tuple[np.ndarray, np.ndarray]:
        """Validate the start columns against the module contract and pre-encode them.

        Returns ``(start_durations float32, start_codes int64)`` flat arrays aligned with
        ``_q_offsets``.  Every representation error is raised here, at init: ragged lists, a
        non-finite or negative non-sentinel duration, an event start without the ``-1.0`` sentinel,
        a sentinel without an event, and — unless ``allow_active_starts`` — any start that is not
        the prediction time itself.  The last rule is what keeps an ordinary sequence model from
        silently scoring a window it cannot represent.
        """
        s_offsets, s_durations = _ragged(self.start_durations)
        e_offsets, s_events = _ragged(self.start_events)
        if not (np.array_equal(self._q_offsets, s_offsets) and np.array_equal(self._q_offsets, e_offsets)):
            raise ValueError("queries/start_durations/start_events list lengths disagree in the labels.")

        durations = np.asarray(s_durations, dtype=np.float64)
        if durations.size and np.isnan(durations).any():
            raise ValueError("start_durations must not contain nulls or NaN.")
        by_event = np.array([c is not None for c in s_events], dtype=bool)
        if durations.size:
            bad_event = by_event & (durations != EVENT_BOUND_DURATION_SENTINEL)
            bad_duration = ~by_event & ~(np.isfinite(durations) & (durations >= 0))
            if bad_event.any() or bad_duration.any():
                first = int(np.flatnonzero(bad_event | bad_duration)[0])
                raise ValueError(
                    "start_durations/start_events disagree on which queries are event-defined: each "
                    f"position must be either start_duration >= 0 (finite) with a null start_event, or "
                    f"start_duration == {EVENT_BOUND_DURATION_SENTINEL} with a non-null start_event; "
                    f"first bad flat position {first} has start_duration={durations[first]!r}, "
                    f"start_event={s_events[first]!r}."
                )

        n_active = int(by_event.sum() + ((~by_event) & (durations > 0)).sum()) if durations.size else 0
        if n_active and not self.allow_active_starts:
            raise ValueError(
                f"{n_active} query window(s) in the labels have an active start (a positive "
                "start_duration or a start_event).  A model that does not encode window starts "
                "would silently treat every window as opening at the prediction time, so this "
                "dataset refuses the grid instead.  Score it with EQ_predict_multitask (its "
                "adapter passes allow_active_starts=True), or regenerate the grid with "
                "prediction-time starts."
            )
        start_codes = np.fromiter(
            (NO_BOUND_INDEX if c is None else self.code_to_index[c] for c in s_events),
            dtype=np.int64,
            count=len(s_events),
        )
        return durations.astype(np.float32), start_codes

    @property
    def labels_df(self) -> pl.DataFrame:
        """Task label rows incl.

        the query-sequence list columns, in MEDS Label schema order.
        """
        if not self.has_task_index:
            return None

        required = [LabelSchema.subject_id_name, LabelSchema.prediction_time_name]

        def read_df(fp: Path) -> pl.DataFrame:
            available = pq.read_schema(fp).names
            extras = [c for c in ALL_SEQ_LABEL_COLS if c in available]
            return pl.read_parquet(fp, columns=[*required, *extras], use_pyarrow=True)

        logger.info(f"Reading query-sequence tasks from {self.config.task_labels_fps}")
        return pl.concat([read_df(fp) for fp in self.config.task_labels_fps], how="vertical")

    def encode_query(self, code_name: str) -> int:
        """Map a query code string to its vocab index.

        Unknown codes raise rather than silently PAD-encode — a query against a code the encoder
        never saw would produce an arbitrary answer.  ``TIMELINE//END`` is an ordinary code here.
        """
        if code_name not in self.code_to_index:
            raise KeyError(f"Query code {code_name!r} is not in the training vocabulary.")
        return self.code_to_index[code_name]

    def _seeded_getitem(self, idx: int, seed: int | None = None) -> dict[str, torch.Tensor]:
        out = super()._seeded_getitem(idx, seed)
        idx = range(len(self._q_offsets) - 1)[idx]  # normalize negative indices
        s, e = self._q_offsets[idx], self._q_offsets[idx + 1]
        out["queries"] = self._q_codes[s:e]
        out["durations"] = self._q_durations[s:e]
        out["answers"] = self._q_answers[s:e]
        if self.has_bound_events:
            out["bound_events"] = self._q_bound_codes[s:e]
        if self.has_forced_answers:
            out["condition_answers"] = self._q_condition_answers[s:e]
        if self.allow_active_starts:
            # The adapter always gets start tensors: absent columns are the legacy all-default form.
            if self.has_starts:
                out["start_durations"] = self._q_start_durations[s:e]
                out["start_codes"] = self._q_start_codes[s:e]
            else:
                out["start_durations"] = np.zeros(e - s, dtype=np.float32)
                out["start_codes"] = np.full(e - s, NO_BOUND_INDEX, dtype=np.int64)
        return out

    def _apply_rope_time(self, out: dict) -> torch.LongTensor | None:
        """Strip delta tokens from ``out`` in place and return the matching ``time_pos_ids``.

        Every per-token field of the dynamic stream (:data:`_PER_TOKEN_FIELDS`) is compacted with
        the same keep mask so it stays aligned to ``code``.  The whitelist is explicit: a
        width-based rule used to compact ``static_code`` and friends too whenever the static table
        happened to be as wide as the padded dynamic stream.  Returns ``None`` when stripping is
        disabled.
        """
        if not self.strip_delta_tokens:
            return None

        code = out["code"]
        _, n_old = code.shape
        pad = QuerySeqBatch.PAD_INDEX

        zeros = torch.zeros_like(code, dtype=torch.float)
        new_code, new_nv, new_nvm, new_tdd, time_pos = rope_time.strip_delta_tokens(
            code,
            out.get("numeric_value") if out.get("numeric_value") is not None else zeros,
            out.get("numeric_value_mask")
            if out.get("numeric_value_mask") is not None
            else torch.zeros_like(code, dtype=torch.bool),
            out.get("time_delta_days") if out.get("time_delta_days") is not None else zeros,
            self.delta_ids,
            pad_index=pad,
        )

        keep = rope_time.build_keep_mask(code, self.delta_ids, pad_index=pad)
        new_n = new_code.shape[1]

        replacements = {
            "code": new_code,
            "numeric_value": new_nv,
            "numeric_value_mask": new_nvm,
            "time_delta_days": new_tdd,
        }
        for name in _PER_TOKEN_FIELDS:
            value = out.get(name)
            if value is None:
                continue
            if name in replacements:
                out[name] = replacements[name]
            elif isinstance(value, torch.Tensor) and value.dim() == 2 and value.shape[1] == n_old:
                out[name] = rope_time.compact_by_keep(value, keep, new_n, 0)

        return time_pos

    def collate(self, batch: list[dict]) -> QuerySeqBatch:
        out = dict(super().collate(batch).items())
        # The base batch type doesn't know about the seq label columns; drop anything
        # super() didn't consume and rebuild the query tensors below.
        out.pop("boolean_value", None)

        time_pos_ids = self._apply_rope_time(out)

        max_q = max(len(item["queries"]) for item in batch)
        B = len(batch)

        q_codes = torch.zeros(B, max_q, dtype=torch.long)
        q_durations = torch.zeros(B, max_q, dtype=torch.float)
        q_answers = torch.full((B, max_q), ANSWER_NO, dtype=torch.long)
        q_mask = torch.zeros(B, max_q, dtype=torch.bool)
        q_bound_codes = (
            torch.full((B, max_q), NO_BOUND_INDEX, dtype=torch.long) if self.has_bound_events else None
        )
        # Start tensors exist only under the explicit opt-in; an ordinary batch is unchanged.
        q_start_durations = torch.zeros(B, max_q, dtype=torch.float) if self.allow_active_starts else None
        q_start_codes = (
            torch.full((B, max_q), NO_BOUND_INDEX, dtype=torch.long) if self.allow_active_starts else None
        )

        for i, item in enumerate(batch):
            n = len(item["queries"])
            q_codes[i, :n] = torch.as_tensor(item["queries"], dtype=torch.long)
            q_durations[i, :n] = torch.as_tensor(item["durations"], dtype=torch.float)
            q_answers[i, :n] = torch.as_tensor(
                [ANSWER_YES if a else ANSWER_NO for a in item["answers"]],
                dtype=torch.long,
            )
            q_mask[i, :n] = True
            if q_bound_codes is not None:
                # Pre-encoded at init: NO_BOUND_INDEX for a time-bounded query, else the strict
                # vocab index (an unknown boundary code fails there, never silently PAD-encodes).
                q_bound_codes[i, :n] = torch.as_tensor(item["bound_events"], dtype=torch.long)
            if q_start_durations is not None:
                q_start_durations[i, :n] = torch.as_tensor(item["start_durations"], dtype=torch.float)
                q_start_codes[i, :n] = torch.as_tensor(item["start_codes"], dtype=torch.long)

        return QuerySeqBatch(
            **out,
            q_codes=q_codes,
            q_durations=q_durations,
            q_answers=q_answers,
            q_mask=q_mask,
            q_bound_codes=q_bound_codes,
            q_start_durations=q_start_durations,
            q_start_codes=q_start_codes,
            time_pos_ids=time_pos_ids,
        )
