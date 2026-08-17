import logging
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, NamedTuple

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch
from meds import DataSchema, LabelSchema
from meds_torchdata import MEDSPytorchDataset
from meds_torchdata.config import MEDSTorchDataConfig
from meds_torchdata.types import BatchMode, MEDSTorchBatch
from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict

from every_query.data.schema import TaskQuerySchema

logger = logging.getLogger(__name__)


@dataclass
class EveryQueryBatch(MEDSTorchBatch):
    """MEDS batch with EveryQuery-specific annotations.

    Applies only the outlined changes on top of MEDSTorchBatch:
      - Adds optional per-sample annotations `occurs` and `query`.
      - Extends `LABEL_TENSOR_NAMES` to include them for printing.
      - Validates their shapes in `__post_init__` if provided.

    All other behavior is inherited unchanged.
    """

    # Extra task annotations (subject-level)
    censor: torch.BoolTensor | None = None
    occurs: torch.LongTensor | None = None
    query: torch.LongTensor | None = None
    duration_days: torch.FloatTensor | None = None

    # Include new annotations in label tensor names for display
    LABEL_TENSOR_NAMES: ClassVar[tuple[str]] = ("boolean_value", "censor", "occurs", "query", "duration_days")

    def __post_init__(self):
        """Run base validations and check EveryQuery annotation shapes.

        Examples:
            Valid SM-mode batch with all EveryQuery annotations:

            >>> batch = EveryQueryBatch(
            ...     code=torch.tensor([[1, 2, 3], [4, 5, 0]]),
            ...     numeric_value=torch.tensor([[1.0, 0.0, -3.0], [0.0, 0.0, 0.0]]),
            ...     numeric_value_mask=torch.tensor([[True, False, True], [False, True, False]]),
            ...     time_delta_days=torch.tensor([[1.0, 0.0, 2.0], [4.0, 0.0, 0.0]]),
            ...     censor=torch.tensor([True, False]),
            ...     occurs=torch.tensor([1, 0]),
            ...     query=torch.tensor([10, 20]),
            ...     duration_days=torch.tensor([30.0, 60.0]),
            ... )
            >>> batch.batch_size
            2
            >>> batch.occurs.tolist()
            [1, 0]

            Annotations default to ``None`` when omitted:

            >>> batch = EveryQueryBatch(
            ...     code=torch.tensor([[1, 2], [3, 4]]),
            ...     numeric_value=torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
            ...     numeric_value_mask=torch.tensor([[True, False], [False, True]]),
            ...     time_delta_days=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            ... )
            >>> batch.occurs is None
            True

            Mismatched ``occurs`` shape raises ``ValueError``:

            >>> EveryQueryBatch(
            ...     code=torch.tensor([[1, 2], [3, 4]]),
            ...     numeric_value=torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
            ...     numeric_value_mask=torch.tensor([[True, False], [False, True]]),
            ...     time_delta_days=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            ...     occurs=torch.tensor([1, 0, 1]),
            ... )
            Traceback (most recent call last):
                ...
            ValueError: Expected shape (2,) for occurs, but got torch.Size([3])!

            Wrong dimensionality for ``duration_days`` also fails:

            >>> EveryQueryBatch(
            ...     code=torch.tensor([[1, 2], [3, 4]]),
            ...     numeric_value=torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
            ...     numeric_value_mask=torch.tensor([[True, False], [False, True]]),
            ...     time_delta_days=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            ...     duration_days=torch.tensor([[30.0], [60.0]]),
            ... )
            Traceback (most recent call last):
                ...
            ValueError: Expected shape (2,) for duration_days, but got torch.Size([2, 1])!

            SEM mode works with annotations and inherited ``boolean_value``:

            >>> batch = EveryQueryBatch(
            ...     code=torch.tensor([[[1, 2], [3, 0]], [[5, 6], [0, 0]]]),
            ...     numeric_value=torch.tensor(
            ...         [[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]),
            ...     numeric_value_mask=torch.tensor(
            ...         [[[True, False], [False, False]], [[False, True], [False, False]]]),
            ...     time_delta_days=torch.tensor([[1.0, 2.0], [4.0, 0.0]]),
            ...     event_mask=torch.tensor([[True, True], [True, False]]),
            ...     boolean_value=torch.tensor([True, False]),
            ...     censor=torch.tensor([True, False]),
            ...     occurs=torch.tensor([1, 0]),
            ... )
            >>> print(batch.mode)
            SEM
            >>> batch.has_labels
            True
            >>> batch.censor.tolist()
            [True, False]

            Mismatched ``censor`` shape raises ``ValueError``:

            >>> EveryQueryBatch(
            ...     code=torch.tensor([[1, 2], [3, 4]]),
            ...     numeric_value=torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
            ...     numeric_value_mask=torch.tensor([[True, False], [False, True]]),
            ...     time_delta_days=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            ...     censor=torch.tensor([True]),
            ... )
            Traceback (most recent call last):
                ...
            ValueError: Expected shape (2,) for censor, but got torch.Size([1])!
        """
        # Run base validations
        super().__post_init__()

        # Validate optional per-sample annotation shapes, if provided
        if self.censor is not None:
            self._MEDSTorchBatch__check_shape("censor", (self.batch_size,))
        if self.occurs is not None:
            self._MEDSTorchBatch__check_shape("occurs", (self.batch_size,))
        if self.query is not None:
            self._MEDSTorchBatch__check_shape("query", (self.batch_size,))
        if self.duration_days is not None:
            self._MEDSTorchBatch__check_shape("duration_days", (self.batch_size,))


class QueryData(NamedTuple):
    """Simple data structure to hold query data, capturing codes.

    As a `NamedTuple`, can be accessed both by index (e.g. `data[0]`) and by attribute (e.g. `data.code`).

    Attributes:
        code: List of integer codes.
    """

    code: list[int]

    def to_JNRT(self, batch_mode: BatchMode, schema: dict | None = None) -> JointNestedRaggedTensorDict:
        """Converts the query data into a JointNestedRaggedTensorDict representation.

        Raises:
            ValueError: If the batch mode is not SEM or SM.

        Examples:
            >>> from nested_ragged_tensors.ragged_numpy import pprint_dense
            >>> query = QueryData(code=[5])

        In SM mode, codes and NaN fillers are stored flat (one entry per code):

            >>> pprint_dense(query.to_JNRT(BatchMode.SM).to_dense())
            code
            [5]
            .
            numeric_value
            [nan]
            .
            time_delta_days
            [nan]

        Multiple codes in SM mode produce one entry per code:

            >>> pprint_dense(QueryData(code=[5, 7]).to_JNRT(BatchMode.SM).to_dense())
            code
            [5 7]
            .
            numeric_value
            [nan nan]
            .
            time_delta_days
            [nan nan]

        In SEM mode, codes are nested inside a single event:

            >>> pprint_dense(query.to_JNRT(BatchMode.SEM).to_dense())
            numeric_value
            [nan]
            .
            time_delta_days
            [nan]
            .
            ---
            .
            dim1/mask
            [[ True]]
            .
            code
            [[5]]

        SEM mode with multiple codes nests them inside one event:

            >>> pprint_dense(QueryData(code=[5, 7]).to_JNRT(BatchMode.SEM).to_dense())
            numeric_value
            [nan]
            .
            time_delta_days
            [nan]
            .
            ---
            .
            dim1/mask
            [[ True  True]]
            .
            code
            [[5 7]]

        A ``schema`` can override the dtype of stored arrays:

            >>> import numpy as np
            >>> pprint_dense(
            ...     QueryData(code=[5])
            ...     .to_JNRT(BatchMode.SM, schema={"code": np.float32})
            ...     .to_dense()
            ... )
            code
            [5.]
            .
            numeric_value
            [nan]
            .
            time_delta_days
            [nan]

        An invalid batch mode raises ``ValueError``:

            >>> query.to_JNRT("invalid")
            Traceback (most recent call last):
                ...
            ValueError: Invalid batch mode invalid!
        """

        match batch_mode:
            case BatchMode.SEM:
                query_dict = {
                    "time_delta_days": [np.nan],
                    "code": [self.code],
                    "numeric_value": [np.nan],
                }
            case BatchMode.SM:
                query_dict = {
                    "time_delta_days": [np.nan for _ in range(len(self.code))],
                    "code": self.code,
                    "numeric_value": [np.nan for _ in range(len(self.code))],
                }
            case _:
                raise ValueError(f"Invalid batch mode {batch_mode}!")

        return JointNestedRaggedTensorDict(query_dict, schema=schema)


class EveryQueryPytorchDataset(MEDSPytorchDataset):
    EXTRA_LABEL_COLS: ClassVar[tuple[str, ...]] = ("occurs", "query", "duration_days")

    @classmethod
    def get_task_seq_bounds_and_labels(cls, label_df: pl.DataFrame, schema_df: pl.DataFrame) -> pl.DataFrame:
        """Returns the event-level allowed input sequence boundaries and labels for each task sample.

        Delegates to the upstream implementation (memory-efficient `join_asof`-based) and then
        hstacks EveryQuery's task-specific annotation columns (`occurs`, `query`, `duration_days`)
        onto the result. Alignment relies on the upstream guarantee that surviving rows are
        returned in `label_df` input order; we semi-filter `label_df` to the same surviving
        subject set so row `i` of `base` corresponds to row `i` of `surviving`.

        Args:
            label_df: The DataFrame containing the task labels, in the MEDS Label DF schema.
            schema_df: A DataFrame with subject ID and a list of event timestamps for each shard.

        Returns:
            A copy of the labels DataFrame, restricted to included subjects, with the appropriate end indices
            for each task sample. Labels will be present if the `cls.LABEL_COL` is present in the input,
            along with any EveryQuery annotation columns present in `label_df`.
        """
        base = super().get_task_seq_bounds_and_labels(label_df, schema_df)
        extras = [c for c in cls.EXTRA_LABEL_COLS if c in label_df.collect_schema().names()]
        if not extras:
            return base
        sid = DataSchema.subject_id_name
        # Upstream guarantees input-order preservation for surviving rows (inner-join on subject_id).
        # hstack-align the extras by semi-filtering label_df to the same surviving row set.
        surviving = label_df.join(schema_df.lazy().select(sid).unique().collect(), on=sid, how="semi")
        return base.hstack(surviving.select(extras))

    def __init__(self, cfg: MEDSTorchDataConfig, split: str):
        super().__init__(cfg, split)

        # Derive the ``(censor, occurs)`` pair the model's loss code consumes, from
        # whichever on-disk label shape the parquet has.  Two shapes are supported:
        #
        # 1. **Collapsed (new, ``TaskQuerySchema``-shaped)** — the parquet has a single
        #    nullable ``boolean_value`` column:
        #        null  → censored (observation ended before we could see the outcome)
        #        True  → event occurred in the duration window
        #        False → no event in the window, and not censored
        #    We split it into ``boolean_value = is_null()`` (censor indicator) and
        #    ``occurs = fill_null(False)``.
        #
        # 2. **Legacy two-column** — the parquet has both ``boolean_value`` (meaning
        #    "censored") AND an explicit ``occurs`` column.  We leave both alone;
        #    super's ``self.labels`` cache is already the censor indicator.
        #
        # Detection: presence of an explicit ``occurs`` column marks the legacy shape
        # — the collapsed shape never emits it (see ``evaluate_index_df``).  This
        # preserves backward compat with any label parquets produced by pre-#96
        # pipelines still on disk.
        #
        # Done after super().__init__() has read the label parquet into self.schema_df
        # so the downstream getitem / collate path sees non-null tensors (torch's
        # bool→tensor conversion doesn't tolerate nulls).
        schema_cols = self.schema_df.collect_schema().names()
        is_legacy_two_col = "occurs" in schema_cols
        if TaskQuerySchema.boolean_value_name in schema_cols and not is_legacy_two_col:
            raw = pl.col(TaskQuerySchema.boolean_value_name)
            self.schema_df = self.schema_df.with_columns(
                raw.is_null().alias(TaskQuerySchema.boolean_value_name),
                raw.fill_null(False).alias("occurs"),
            )
            schema_cols = self.schema_df.collect_schema().names()
            # Super cached the pre-collapse ``boolean_value`` Series into ``self.labels``
            # at its ``__init__`` time; re-point it at the post-processed (non-null,
            # censor-indicator) column so ``_seeded_getitem`` and ``collate`` see the
            # correct values.
            if getattr(self, "has_task_labels", False):
                self.labels = self.schema_df[TaskQuerySchema.boolean_value_name]

        # Extra task annotations
        self.has_occurs: bool = "occurs" in schema_cols
        self.has_query: bool = TaskQuerySchema.query_name in schema_cols
        self.has_duration_days: bool = TaskQuerySchema.duration_days_name in schema_cols
        self.occurs = self.schema_df["occurs"] if self.has_occurs else None
        self.query = self.schema_df[TaskQuerySchema.query_name] if self.has_query else None
        self.duration_days = (
            self.schema_df[TaskQuerySchema.duration_days_name] if self.has_duration_days else None
        )
        # Load code vocabulary mapping (string code -> integer vocab index) for encoding queries.
        # Deliberately unguarded: an empty mapping would PAD-encode every query, and a PAD query
        # token is dropped from the attention mask while its position is still pooled as "the
        # query embedding" — so a stale/missing metadata path would silently degrade an entire
        # run into confident answers about queries the model never saw.  Fail loud instead.
        code_meta = pl.read_parquet(
            self.config.code_metadata_fp, columns=["code", "code/vocab_index"], use_pyarrow=True
        )
        codes = code_meta["code"].to_list()
        vocab_indices = code_meta["code/vocab_index"].to_list()
        self.code_to_index: dict[str, int] = {c: int(i) for c, i in zip(codes, vocab_indices, strict=False)}

    @property
    def labels_df(self) -> pl.DataFrame:
        """Returns the task labels as a DataFrame, in the MEDS Label schema, or `None` if there is no task."""
        if not self.has_task_index:
            return None

        required_cols = [LabelSchema.subject_id_name, LabelSchema.prediction_time_name]

        def read_df(fp: Path) -> pl.DataFrame:
            schema = pq.read_schema(fp)
            extras = []
            # "occurs" is not an on-disk column after the nullable-label collapse; it's
            # derived in __init__ from boolean_value.is_null().  Keep it in the list for
            # backward compatibility with any legacy label parquets that still carry it
            # explicitly — harmless if absent.
            for extra_col in [
                self.LABEL_COL,
                "occurs",
                TaskQuerySchema.query_name,
                TaskQuerySchema.duration_days_name,
            ]:
                if extra_col in schema.names:
                    extras.append(extra_col)
            label_cols = [*required_cols, *extras]
            return pl.read_parquet(fp, columns=label_cols, use_pyarrow=True)

        logger.info(f"Reading tasks from {self.config.task_labels_fps}")
        return pl.concat([read_df(fp) for fp in self.config.task_labels_fps], how="vertical")

    def encode_query(self, code_name: str) -> int:
        """Encode query using the canonical code vocabulary mapping.

        Raises:
            KeyError: if ``code_name`` is not in the cohort's code vocabulary.  Falling back to
                ``PAD_INDEX`` would be silent corruption rather than a missing feature: the query
                token is masked out of attention (``attention_mask = code != PAD_INDEX``) while
                position 0 is still pooled as the query embedding, so the model would answer a
                query it never saw with plausible-looking, near-uniform probabilities.

        Examples:
            >>> from unittest.mock import Mock
            >>> EveryQueryPytorchDataset.encode_query(Mock(code_to_index={"A//B": 3}), "A//B")
            3
            >>> EveryQueryPytorchDataset.encode_query(Mock(code_to_index={"A//B": 3}), "typo")
            Traceback (most recent call last):
                ...
            KeyError: "Query code 'typo' is not in the code vocabulary...
        """
        if code_name not in self.code_to_index:
            raise KeyError(
                f"Query code {code_name!r} is not in the code vocabulary ({len(self.code_to_index)} "
                f"codes).  Out-of-vocab queries cannot be encoded: PAD-encoding them would mask the "
                f"query token out of attention while still pooling its position, silently producing "
                f"meaningless predictions.  Check the task queries and the code metadata path."
            )
        return int(self.code_to_index[code_name])

    def _seeded_getitem(self, idx: int, seed: int | None = None) -> dict[str, torch.Tensor]:
        out = super()._seeded_getitem(idx, seed)

        dynamic_data = out["dynamic"]
        schema = dynamic_data.schema
        schema["code"] = np.int16
        query_data = QueryData(code=[self.encode_query(self.query[idx])])
        query_as_JNRT = query_data.to_JNRT(self.config.batch_mode, schema)
        out["dynamic"] = JointNestedRaggedTensorDict.concatenate([query_as_JNRT, dynamic_data])

        if getattr(self, "has_occurs", False):
            out["occurs"] = self.occurs[idx]
        if getattr(self, "has_query", False):
            out["query"] = self.query[idx]
        if getattr(self, "has_duration_days", False):
            out["duration_days"] = self.duration_days[idx]

        return out

    def collate(self, batch: list[dict]) -> EveryQueryBatch:
        out = dict(super().collate(batch).items())

        if self.has_task_labels:
            out["censor"] = out[self.LABEL_COL]
        if getattr(self, "has_occurs", False):
            out["occurs"] = torch.Tensor([item["occurs"] for item in batch]).long()
        if getattr(self, "has_query", False):
            query_ids = [self.encode_query(item["query"]) for item in batch]
            out["query"] = torch.as_tensor(query_ids).long()
        if getattr(self, "has_duration_days", False):
            out["duration_days"] = torch.as_tensor([item["duration_days"] for item in batch]).float()
        return EveryQueryBatch(**out)
