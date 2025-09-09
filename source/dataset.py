import logging
from dataclasses import dataclass
from typing import ClassVar, NamedTuple
from functools import cached_property
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch
from meds import DataSchema, LabelSchema
from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict

from meds_torchdata import MEDSPytorchDataset
from meds_torchdata.config import MEDSTorchDataConfig, StaticInclusionMode
from meds_torchdata.types import BatchMode, MEDSTorchBatch, StaticData, SubsequenceSamplingStrategy

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
    occurs: torch.BoolTensor | None = None
    query: torch.LongTensor | None = None

    # Include new annotations in label tensor names for display
    LABEL_TENSOR_NAMES: ClassVar[tuple[str]] = ("boolean_value", "censor", "occurs", "query")

    def __post_init__(self):
        # Run base validations
        super().__post_init__()

        # Validate optional per-sample annotation shapes, if provided
        if self.censor is not None:
            self._MEDSTorchBatch__check_shape("censor", (self.batch_size,))
        if self.occurs is not None:
            self._MEDSTorchBatch__check_shape("occurs", (self.batch_size,))
        if self.query is not None:
            self._MEDSTorchBatch__check_shape("query", (self.batch_size,))


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

    @classmethod
    def get_task_seq_bounds_and_labels(cls, label_df: pl.DataFrame, schema_df: pl.DataFrame) -> pl.DataFrame:
        """Returns the event-level allowed input sequence boundaries and labels for each task sample.

        This function is guaranteed to output an index of the same order and length as `label_df`. Subjects
        not present in `schema_df` will be included in the output, with null labels and indices.

        Args:
            label_df: The DataFrame containing the task labels, in the MEDS Label DF schema.
            schema_df: A DataFrame with subject ID and a list of event timestamps for each shard.

        Returns:
            A copy of the labels DataFrame, restricted to included subjects, with the appropriate end indices
            for each task sample. Labels will be present if the `cls.LABEL_COL` is present in the input.
        """

        end_idx_expr = (
            pl.col(DataSchema.time_name)
            .search_sorted(pl.col(LabelSchema.prediction_time_name), side="right")
            .last()
            .alias(cls.END_IDX)
        )

        group_cols = ["_row", DataSchema.subject_id_name, LabelSchema.prediction_time_name]
        out_cols = [DataSchema.subject_id_name, cls.END_IDX, LabelSchema.prediction_time_name]

        label_names = label_df.collect_schema().names()

        if cls.LABEL_COL in label_names:
            group_cols.append(cls.LABEL_COL)
            out_cols.append(cls.LABEL_COL)

        # Include pass-through task annotations if present
        if "occurs" in label_names:
            group_cols.append("occurs")
            out_cols.append("occurs")
        if "query" in label_names:
            group_cols.append("query")
            out_cols.append("query")

        return (
            label_df.join(schema_df, on=DataSchema.subject_id_name, how="inner", maintain_order="left")
            .with_row_index("_row")
            .explode(DataSchema.time_name)
            .group_by(group_cols, maintain_order=True)
            .agg(end_idx_expr)
            .select(out_cols)
        )

    def __init__(self, cfg: MEDSTorchDataConfig, split: str):
        super().__init__(cfg, split)

        # Extra task annotations
        self.has_occurs: bool = "occurs" in self.schema_df.collect_schema().names()
        self.has_query: bool = "query" in self.schema_df.collect_schema().names()
        self.occurs = self.schema_df["occurs"] if self.has_occurs else None
        self.query = self.schema_df["query"] if self.has_query else None
        # Load code vocabulary mapping (string code -> integer vocab index) for encoding queries
        try:
            code_meta = pl.read_parquet(self.config.code_metadata_fp, columns=["code", "code/vocab_index"], use_pyarrow=True)
            codes = code_meta["code"].to_list()
            vocab_indices = code_meta["code/vocab_index"].to_list()
            self.code_to_index: dict[str, int] = {c: int(i) for c, i in zip(codes, vocab_indices, strict=False)}
        except Exception as e:
            logger.warning(f"Failed to load code metadata for query encoding: {e}")
            self.code_to_index = {}

    @property
    def labels_df(self) -> pl.DataFrame:
        """Returns the task labels as a DataFrame, in the MEDS Label schema, or `None` if there is no task.
        """
        if not self.has_task_index:
            return None

        required_cols = [LabelSchema.subject_id_name, LabelSchema.prediction_time_name]

        def read_df(fp: Path) -> pl.DataFrame:
            schema = pq.read_schema(fp)
            extras = []
            for extra_col in [self.LABEL_COL, "occurs", "query"]:
                if extra_col in schema.names:
                    extras.append(extra_col)
            label_cols = [*required_cols, *extras]
            return pl.read_parquet(fp, columns=label_cols, use_pyarrow=True)

        logger.info(f"Reading tasks from {self.config.task_labels_fps}")
        return pl.concat([read_df(fp) for fp in self.config.task_labels_fps], how="vertical")

    def encode_query(self, code_name: str) -> int:
        """Encode query using the canonical code vocabulary mapping
        """
        try:
            return int(self.code_to_index.get(code_name, EveryQueryBatch.PAD_INDEX))
        except Exception:
            return EveryQueryBatch.PAD_INDEX

    def _seeded_getitem(self, idx: int, seed: int | None = None) -> dict[str, torch.Tensor]:
        
        out = super()._seeded_getitem(idx, seed)

        dynamic_data = out["dynamic"]
        query_data = QueryData(code=[self.encode_query(self.query[idx])])
        query_as_JNRT = query_data.to_JNRT(self.config.batch_mode, dynamic_data.schema)
        out["dynamic"] = JointNestedRaggedTensorDict.concatenate([query_as_JNRT, dynamic_data])

        if getattr(self, "has_occurs", False):
            out["occurs"] = self.occurs[idx]
        if getattr(self, "has_query", False):
            out["query"] = self.query[idx]

        return out

    def collate(self, batch: list[dict]) -> EveryQueryBatch:
        out = dict(super().collate(batch).items()) # out is MEDSTorchBatch
        if self.has_task_labels:
            out["censor"] = out[self.LABEL_COL]
        if getattr(self, "has_occurs", False):
            out["occurs"] = torch.Tensor([item["occurs"] for item in batch]).bool()
        if getattr(self, "has_query", False):
            query_ids = [self.encode_query(item["query"]) for item in batch]
            out["query"] = torch.as_tensor(query_ids).long()
        return EveryQueryBatch(**out)
