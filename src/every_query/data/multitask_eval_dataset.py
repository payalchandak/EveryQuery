"""QuerySeq-to-multitask evaluation adapter (issue #28).

``EQ_predict_multitask`` scores a :class:`~every_query.data.schema.QuerySeqSchema` evaluation grid
(written by ``EQ_generate_evaluation_query_sequences``) with the all-vocabulary
:class:`~every_query.model.conditional_multitask_ar_model.ConditionalMultitaskARModel`.  The model's
input stream is ``[patient, W0, C0, A0, ..., W(K-2), C(K-2), A(K-2), W(K-1)]``; a QuerySeq row is an
ordered list of scalar queries with their answers.  This module maps one onto the other, **per real
row**, never per padded batch width:

    q_durations        <- durations
    q_bound_codes      <- bound_events        (NO_BOUND_INDEX where null / column absent)
    q_start_durations  <- start_durations     (0.0 where the column is absent)
    q_start_codes      <- start_events        (NO_BOUND_INDEX where null / column absent)
    condition_codes    <- queries[:-1]
    condition_answers  <- answers[:-1]
    scored_code        <- queries[-1]
    label              <- answers[-1]

so a row with ``n`` queries becomes ``n`` windows whose first ``n-1`` queries are teacher-forced
conditioning pairs and whose last query is the one scored.  The final query is *not* teacher-forced
into its own prediction: the model reads the prediction at ``W(n-1)``, which precedes ``C(n-1)`` in
the stream and can never attend to it.

Padding
-------
Rows of different lengths are right-padded to the batch's longest ``K``.  ``q_mask[i, :n_i]`` is
``True`` and the rest ``False``; ``[:-1]`` / ``[-1]`` above always refer to the row's own ``n_i``, so
the scored window of a 1-query row in a batch padded to ``K = 5`` is window ``0``, not window ``4``.
``condition_codes[i, n_i - 1:]`` and ``condition_answers[i, n_i - 1:]`` are padding (``PAD`` /
``False``).  The model attends ``q_mask`` per window (all three tokens of a real window, none of a
padded one), so the padded ``C(n_i - 1)`` / ``A(n_i - 1)`` tokens of the last real window *are*
attended — but they sit **after** ``W(n_i - 1)`` in the causal stream, and every token after the
scored window (padded conditions, padded windows) is invisible to it.  The hidden state the model
scores at ``W(n_i - 1)`` therefore depends only on the patient prefix, ``W0..W(n_i - 1)`` and the
real conditioning pairs ``(C0, A0)..(C(n_i - 2), A(n_i - 2))``, exactly as a training row with
``K = n_i`` windows would.

What the batch does *not* carry
-------------------------------
No ``(B, K, V)`` targets: the grid's scalar answers are the only labels, the earlier ones are model
inputs, and the last one is the scalar label.  Nothing here reads or writes a ``.labels.npy``, a
multitask manifest, ``eval_meta`` or ``eval_tasks.parquet``.

The dataset is the explicit opt-in path for active window starts: it builds the underlying
:class:`~every_query.data.query_seq_dataset.QuerySeqPytorchDataset` with
``allow_active_starts=True``, which the ordinary sequence models' path never does.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch
from meds import LabelSchema
from meds_torchdata import MEDSPytorchDataset
from meds_torchdata.config import MEDSTorchDataConfig
from meds_torchdata.types import MEDSTorchBatch

from every_query.data.query_seq_dataset import (
    ALL_SEQ_LABEL_COLS,
    NO_BOUND_INDEX,
    QuerySeqPytorchDataset,
)

logger = logging.getLogger(__name__)


@dataclass
class MultitaskEvalBatch(MEDSTorchBatch):
    """MEDS batch plus per-row multitask windows, conditioning pairs, and one scored query.

    Attributes:
        q_start_durations: ``(B, K)`` float32 - days after ``prediction_time`` the window opens
            (``0.0`` = the prediction time, ``EVENT_BOUND_DURATION_SENTINEL`` = event-defined).
        q_start_codes: ``(B, K)`` int64 - start-event vocabulary index, ``NO_BOUND_INDEX`` for a
            duration start.
        q_durations: ``(B, K)`` float32 - horizon in days after the resolved start;
            ``EVENT_BOUND_DURATION_SENTINEL`` at event-bounded slots.
        q_bound_codes: ``(B, K)`` int64 - boundary index, ``NO_BOUND_INDEX`` for a duration end.
        q_mask: ``(B, K)`` bool - right-padded prefix mask, ``True`` at the row's real windows.
        condition_codes: ``(B, K-1)`` int64 - ``queries[:-1]`` of each row, PAD-padded.
        condition_answers: ``(B, K-1)`` bool - ``answers[:-1]`` of each row, ``False``-padded.
        scored_codes: ``(B,)`` int64 - ``queries[-1]`` of each row, never PAD.
        labels: ``(B,)`` bool - ``answers[-1]`` of each row.
        n_queries: ``(B,)`` int64 - the row's real query count (``q_mask.sum(1)``).
        time_pos_ids: Optional per-patient-token elapsed hours (RoPE time), as in the other batches.

    Examples:
        >>> batch = MultitaskEvalBatch(
        ...     code=torch.tensor([[1, 2, 3], [4, 5, 0]]),
        ...     numeric_value=torch.zeros(2, 3),
        ...     numeric_value_mask=torch.zeros(2, 3, dtype=torch.bool),
        ...     time_delta_days=torch.zeros(2, 3),
        ...     q_start_durations=torch.tensor([[0.0, -1.0], [7.0, 0.0]]),
        ...     q_start_codes=torch.tensor([[0, 2], [0, 0]]),
        ...     q_durations=torch.tensor([[30.0, -1.0], [7.0, 0.0]]),
        ...     q_bound_codes=torch.tensor([[0, 4], [0, 0]]),
        ...     q_mask=torch.tensor([[True, True], [True, False]]),
        ...     condition_codes=torch.tensor([[3], [0]]),
        ...     condition_answers=torch.tensor([[True], [False]]),
        ...     scored_codes=torch.tensor([5, 3]),
        ...     labels=torch.tensor([False, True]),
        ...     n_queries=torch.tensor([2, 1]),
        ... )
        >>> batch.num_bounds
        2

        A padded batch must keep the prefix rule and one real window per row:

        >>> kw = dict(code=torch.tensor([[1, 2]]), numeric_value=torch.zeros(1, 2),
        ...     numeric_value_mask=torch.zeros(1, 2, dtype=torch.bool), time_delta_days=torch.zeros(1, 2),
        ...     q_start_durations=torch.zeros(1, 2), q_start_codes=torch.zeros(1, 2, dtype=torch.long),
        ...     q_durations=torch.ones(1, 2), q_bound_codes=torch.zeros(1, 2, dtype=torch.long),
        ...     condition_codes=torch.zeros(1, 1, dtype=torch.long),
        ...     condition_answers=torch.zeros(1, 1, dtype=torch.bool), scored_codes=torch.tensor([3]),
        ...     labels=torch.tensor([True]), n_queries=torch.tensor([1]))
        >>> MultitaskEvalBatch(**kw, q_mask=torch.tensor([[False, True]]))
        Traceback (most recent call last):
            ...
        ValueError: q_mask must be a right-padded prefix mask with at least one real window per row
        >>> MultitaskEvalBatch(**{**kw, "n_queries": torch.tensor([2])}, q_mask=torch.tensor([[True, False]]))
        Traceback (most recent call last):
            ...
        ValueError: n_queries must equal q_mask.sum(1)
    """

    q_start_durations: torch.FloatTensor | None = None
    q_start_codes: torch.LongTensor | None = None
    q_durations: torch.FloatTensor | None = None
    q_bound_codes: torch.LongTensor | None = None
    q_mask: torch.BoolTensor | None = None
    condition_codes: torch.LongTensor | None = None
    condition_answers: torch.BoolTensor | None = None
    scored_codes: torch.LongTensor | None = None
    labels: torch.BoolTensor | None = None
    n_queries: torch.LongTensor | None = None
    time_pos_ids: torch.LongTensor | None = None

    LABEL_TENSOR_NAMES: ClassVar[tuple[str]] = (
        "boolean_value",
        "q_start_durations",
        "q_start_codes",
        "q_durations",
        "q_bound_codes",
        "condition_codes",
        "condition_answers",
        "scored_codes",
        "labels",
        "n_queries",
    )

    @property
    def num_bounds(self) -> int | None:
        return None if self.q_durations is None else self.q_durations.shape[1]

    def __post_init__(self):
        super().__post_init__()
        if self.q_durations is None:
            return
        B, k = self.batch_size, self.q_durations.shape[1]
        for name in ("q_start_durations", "q_start_codes", "q_durations", "q_bound_codes", "q_mask"):
            tensor = getattr(self, name)
            if tensor is None or tuple(tensor.shape) != (B, k):
                got = None if tensor is None else tensor.shape
                raise ValueError(f"Expected shape {(B, k)} for {name}, but got {got}!")
        for name in ("condition_codes", "condition_answers"):
            tensor = getattr(self, name)
            if tensor is None or tuple(tensor.shape) != (B, k - 1):
                got = None if tensor is None else tensor.shape
                raise ValueError(f"Expected shape {(B, k - 1)} for {name}, but got {got}!")
        for name in ("scored_codes", "labels", "n_queries"):
            tensor = getattr(self, name)
            if tensor is None or tuple(tensor.shape) != (B,):
                got = None if tensor is None else tensor.shape
                raise ValueError(f"Expected shape {(B,)} for {name}, but got {got}!")
        for name in ("q_mask", "condition_answers", "labels"):
            if getattr(self, name).dtype != torch.bool:
                raise TypeError(f"{name} must be boolean, got {getattr(self, name).dtype}")
        for name in ("q_start_codes", "q_bound_codes", "condition_codes", "scored_codes", "n_queries"):
            if getattr(self, name).dtype != torch.long:
                raise TypeError(f"{name} must be int64, got {getattr(self, name).dtype}")
        prefix_ok = not (self.q_mask[:, 1:] & ~self.q_mask[:, :-1]).any()
        if not prefix_ok or not self.q_mask[:, 0].all():
            raise ValueError(
                "q_mask must be a right-padded prefix mask with at least one real window per row"
            )
        if not torch.equal(self.q_mask.sum(dim=1).long(), self.n_queries):
            raise ValueError("n_queries must equal q_mask.sum(1)")


class QuerySeqMultitaskEvalDataset(QuerySeqPytorchDataset):
    """``QuerySeqSchema`` evaluation grid tensorized for ``ConditionalMultitaskARModel`` scoring.

    A :class:`~every_query.data.query_seq_dataset.QuerySeqPytorchDataset` built with
    ``allow_active_starts=True`` (the only caller that does), whose :meth:`collate` emits a
    :class:`MultitaskEvalBatch` per the mapping in the module docstring.  On top of the parent's
    checks it requires:

    - the grid to be non-empty and every row to carry at least one query;
    - every grid row of this split to survive the cohort join - a grid subject absent from the
      tensorized cohort is an error, never a silently shrunken grid;
    - every query / start / bound code to map to a non-PAD index and, when ``expected_vocab_size``
      is given, below it (the checkpoint's tied embedding width: ``V_ext`` under an ontology);
    - when ``max_windows`` is given, no row longer than the checkpoint's window budget.

    ``ontology_dir`` goes through to the parent, whose ``extend_code_map`` adds the ontology's
    ancestor node names to ``code_to_index`` (at their ``[V, V_ext)`` indices), so a grid that asks
    about, starts at or is bounded by an ancestor node resolves; without it those names are unknown
    codes and fail at init like any other.

    Only ``{task_labels_dir}/{split}/*.parquet`` is read, so a grid root holding several splits'
    ``eval/{split}/`` trees is fine.  No manifest, packed labels, ``eval_meta`` or
    ``eval_tasks.parquet`` is looked for.
    """

    def __init__(
        self,
        cfg: MEDSTorchDataConfig,
        split: str,
        *,
        strip_delta_tokens: bool = False,
        expected_vocab_size: int | None = None,
        max_windows: int | None = None,
        ontology_dir: str | Path | None = None,
    ):
        if cfg.task_labels_dir is None:
            raise ValueError("QuerySeqMultitaskEvalDataset requires task_labels_dir (the grid's eval/ root)")
        self._split_dir = Path(cfg.task_labels_dir) / split
        self.n_grid_rows: int | None = None
        super().__init__(
            cfg,
            split,
            strip_delta_tokens=strip_delta_tokens,
            ontology_dir=None if ontology_dir is None else str(ontology_dir),
            allow_active_starts=True,
            allow_forced_answers=True,
        )

        if self.n_grid_rows is None:
            raise RuntimeError("labels_df was not read; the grid row count is unknown")
        if self.n_grid_rows == 0:
            raise ValueError(f"the evaluation grid under {self._split_dir} has no rows; nothing to score")
        if len(self) != self.n_grid_rows:
            raise RuntimeError(
                f"the evaluation grid under {self._split_dir} has {self.n_grid_rows} row(s) but the dataset "
                f"loaded {len(self)}; {self.n_grid_rows - len(self)} grid row(s) were dropped, most likely "
                "because their subject is absent from the tensorized cohort. Regenerate the grid against "
                "this cohort."
            )

        lengths = np.diff(self._q_offsets)
        if (lengths < 1).any():
            row = int(np.flatnonzero(lengths < 1)[0])
            raise ValueError(f"every grid row needs at least one query; row {row} has none")
        # The model's learned block positions cover ``max_windows`` windows; a longer row would fail
        # only when its batch reached ``window_hidden_states``, after model load and part of the run.
        self.max_windows = max_windows
        if max_windows is not None and lengths.size and int(lengths.max()) > max_windows:
            row = int(np.flatnonzero(lengths > max_windows)[0])
            raise ValueError(
                f"grid row {row} has {int(lengths[row])} queries but the checkpoint supports at most "
                f"max_windows={max_windows} windows per sequence; regenerate the grid with shorter sequences."
            )

        self.expected_vocab_size = expected_vocab_size
        # Mirror ``MultitaskBoundaryPytorchDataset``: a code that maps to PAD has no embedding row
        # (and, for a start / bound, would be read as "no event"), and an index at or past the
        # checkpoint's tied-embedding width has no row either.  Checked on the *strings*, before
        # the parent's NO_BOUND_INDEX substitution could hide a PAD-mapped code.
        self._check_codes("query", self.queries, self._q_codes)
        if self.has_bound_events:
            self._check_codes("bound", self.bound_events, self._q_bound_codes)
        if self.has_starts:
            self._check_codes("start", self.start_events, self._q_start_codes)

    def _check_codes(self, what: str, column: pl.Series, idx: np.ndarray) -> None:
        """PAD-mapped and out-of-width codes are errors: the model has no row to look them up in."""
        named = column.explode().drop_nulls().unique().sort().to_list()
        pad_mapped = [c for c in named if self.code_to_index[c] <= MEDSTorchBatch.PAD_INDEX]
        if pad_mapped:
            raise ValueError(
                f"{len(pad_mapped)} {what} code(s) in the grid map to the PAD index "
                f"{MEDSTorchBatch.PAD_INDEX}: {pad_mapped[:10]}; regenerate the grid against this cohort."
            )
        if self.expected_vocab_size is not None and idx.size and int(idx.max()) >= self.expected_vocab_size:
            raise ValueError(
                f"{what} code index {int(idx.max())} is outside the checkpoint's vocabulary of size "
                f"{self.expected_vocab_size}; the grid was generated against a different codes.parquet."
            )

    @property
    def labels_df(self) -> pl.DataFrame:
        """This split's grid rows only (``{task_labels_dir}/{split}/*.parquet``), counted as read."""
        if not self.has_task_index:
            return None
        required = [LabelSchema.subject_id_name, LabelSchema.prediction_time_name]
        fps = [fp for fp in self.config.task_labels_fps if self._split_dir in fp.parents]
        if not fps:
            raise FileNotFoundError(
                f"no QuerySeqSchema parquets under {self._split_dir}; point task_labels_dir at the "
                "`eval/` root written by EQ_generate_evaluation_query_sequences."
            )

        def read_df(fp: Path) -> pl.DataFrame:
            available = pq.read_schema(fp).names
            extras = [c for c in ALL_SEQ_LABEL_COLS if c in available]
            return pl.read_parquet(fp, columns=[*required, *extras], use_pyarrow=True)

        logger.info(f"Reading QuerySeq evaluation grid from {fps}")
        df = pl.concat([read_df(fp) for fp in fps], how="vertical")
        self.n_grid_rows = df.height
        return df

    def collate(self, batch: list[dict]) -> MultitaskEvalBatch:
        out = dict(MEDSPytorchDataset.collate(self, batch).items())
        out.pop("boolean_value", None)
        time_pos_ids = self._apply_rope_time(out)

        B = len(batch)
        lengths = [len(item["queries"]) for item in batch]
        if min(lengths) < 1:
            raise ValueError("every grid row needs at least one query")
        k = max(lengths)

        q_start_durations = torch.zeros(B, k, dtype=torch.float32)
        q_start_codes = torch.full((B, k), NO_BOUND_INDEX, dtype=torch.long)
        q_durations = torch.zeros(B, k, dtype=torch.float32)
        q_bound_codes = torch.full((B, k), NO_BOUND_INDEX, dtype=torch.long)
        q_mask = torch.zeros(B, k, dtype=torch.bool)
        condition_codes = torch.full((B, max(k - 1, 0)), MEDSTorchBatch.PAD_INDEX, dtype=torch.long)
        condition_answers = torch.zeros(B, max(k - 1, 0), dtype=torch.bool)
        scored_codes = torch.empty(B, dtype=torch.long)
        labels = torch.empty(B, dtype=torch.bool)
        n_queries = torch.tensor(lengths, dtype=torch.long)

        for i, (item, n) in enumerate(zip(batch, lengths, strict=True)):
            queries = torch.as_tensor(item["queries"], dtype=torch.long)
            answers = torch.as_tensor(np.asarray(item["answers"], dtype=bool))
            q_durations[i, :n] = torch.as_tensor(item["durations"], dtype=torch.float32)
            q_mask[i, :n] = True
            # The parent emits the start arrays under ``allow_active_starts`` (zeros / NO_BOUND_INDEX
            # when the grid has no start columns), so the batch never has to special-case them.
            q_start_durations[i, :n] = torch.as_tensor(item["start_durations"], dtype=torch.float32)
            q_start_codes[i, :n] = torch.as_tensor(item["start_codes"], dtype=torch.long)
            if "bound_events" in item:
                q_bound_codes[i, :n] = torch.as_tensor(item["bound_events"], dtype=torch.long)
            # Per real row: the first n-1 queries condition, the last one is scored.
            condition_codes[i, : n - 1] = queries[:-1]
            # A designed grid may dictate a conditioning answer (``forced_answers``); the parent has
            # already folded that over the truth.  ``labels`` below is always the labeled truth.
            conditioning = item.get("condition_answers", item["answers"])
            condition_answers[i, : n - 1] = torch.as_tensor(np.asarray(conditioning, dtype=bool))[:-1]
            scored_codes[i] = queries[-1]
            labels[i] = answers[-1]

        return MultitaskEvalBatch(
            **out,
            q_start_durations=q_start_durations,
            q_start_codes=q_start_codes,
            q_durations=q_durations,
            q_bound_codes=q_bound_codes,
            q_mask=q_mask,
            condition_codes=condition_codes,
            condition_answers=condition_answers,
            scored_codes=scored_codes,
            labels=labels,
            n_queries=n_queries,
            time_pos_ids=time_pos_ids,
        )
