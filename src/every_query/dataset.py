import logging
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, NamedTuple

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch
from meds import DataSchema, LabelSchema, held_out_split
from meds_torchdata import MEDSPytorchDataset
from meds_torchdata.config import MEDSTorchDataConfig
from meds_torchdata.types import BatchMode, MEDSTorchBatch
from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict

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

    # Special token IDs for the uniform query prefix format:
    #   [QUANT, c1, ..., cN, SEP, DUR, SEP, patient_event_1, ..., PAD...]
    # Assigned at runtime via configure_special_tokens(backbone_vocab_size).
    NUM_SPECIAL_TOKENS: ClassVar[int] = 4
    ANY_TOKEN_ID: ClassVar[int | None] = None
    ALL_TOKEN_ID: ClassVar[int | None] = None
    SEP_TOKEN_ID: ClassVar[int | None] = None
    DUR_TOKEN_ID: ClassVar[int | None] = None

    @classmethod
    def configure_special_tokens(cls, backbone_vocab_size: int) -> int:
        """Assign special-token IDs starting at *backbone_vocab_size*.

        Must be called once before any batch is created (e.g. during model or
        dataset initialization).  Returns the total effective vocab size so
        callers can resize the backbone's embedding table if needed.
        """
        cls.ANY_TOKEN_ID = backbone_vocab_size
        cls.ALL_TOKEN_ID = backbone_vocab_size + 1
        cls.SEP_TOKEN_ID = backbone_vocab_size + 2
        cls.DUR_TOKEN_ID = backbone_vocab_size + 3
        return backbone_vocab_size + cls.NUM_SPECIAL_TOKENS

    # Extra task annotations (subject-level)
    subject_id: torch.LongTensor | None = None
    prediction_time: torch.LongTensor | None = None
    censor: torch.BoolTensor | None = None
    occurs: torch.LongTensor | None = None
    query: torch.LongTensor | None = None
    duration_days: torch.FloatTensor | None = None
    query_embed_position: torch.LongTensor | None = None

    # Include new annotations in label tensor names for display
    LABEL_TENSOR_NAMES: ClassVar[tuple[str]] = ("boolean_value", "censor", "occurs", "query", "duration_days")

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
        if self.duration_days is not None:
            self._MEDSTorchBatch__check_shape("duration_days", (self.batch_size,))
        if self.query_embed_position is not None:
            self._MEDSTorchBatch__check_shape("query_embed_position", (self.batch_size,))


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
                    "numeric_value": [[np.nan for _ in range(len(self.code))]],
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
        if "quantifier" in label_names:
            group_cols.append("quantifier")
            out_cols.append("quantifier")
        if "query_codes" in label_names:
            group_cols.append("query_codes")
            out_cols.append("query_codes")
        if "duration_days" in label_names:
            group_cols.append("duration_days")
            out_cols.append("duration_days")

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

        # convert prediction_time_name to int's instead of datetime objs
        if self.split == held_out_split:
            self.schema_df = self.schema_df.with_columns(
                pl.col(LabelSchema.prediction_time_name)
                .dt.timestamp("us")
                .alias(LabelSchema.prediction_time_name)
            )

        # Extra task annotations
        self.has_occurs: bool = "occurs" in self.schema_df.collect_schema().names()
        self.has_quantifier: bool = "quantifier" in self.schema_df.collect_schema().names()
        self.has_query_codes: bool = "query_codes" in self.schema_df.collect_schema().names()
        self.has_duration_days: bool = "duration_days" in self.schema_df.collect_schema().names()
        self.occurs = self.schema_df["occurs"] if self.has_occurs else None
        self.quantifier = self.schema_df["quantifier"] if self.has_quantifier else None
        self.query_codes = self.schema_df["query_codes"] if self.has_query_codes else None
        self.duration_days = self.schema_df["duration_days"] if self.has_duration_days else None
        # Load code vocabulary mapping (string code -> integer vocab index) for encoding queries
        try:
            code_meta = pl.read_parquet(
                self.config.code_metadata_fp, columns=["code", "code/vocab_index"], use_pyarrow=True
            )
            codes = code_meta["code"].to_list()
            vocab_indices = code_meta["code/vocab_index"].to_list()
            self.code_to_index: dict[str, int] = {
                c: int(i) for c, i in zip(codes, vocab_indices, strict=False)
            }
        except Exception as e:
            logger.warning(f"Failed to load code metadata for query encoding: {e}")
            self.code_to_index = {}

    @property
    def labels_df(self) -> pl.DataFrame:
        """Returns the task labels as a DataFrame, in the MEDS Label schema, or `None` if there is no task."""
        if not self.has_task_index:
            return None

        required_cols = [LabelSchema.subject_id_name, LabelSchema.prediction_time_name]

        def read_df(fp: Path) -> pl.DataFrame:
            schema = pq.read_schema(fp)
            extras = []
            for extra_col in [self.LABEL_COL, "occurs", "quantifier", "query_codes", "duration_days"]:
                if extra_col in schema.names:
                    extras.append(extra_col)
            label_cols = [*required_cols, *extras]
            return pl.read_parquet(fp, columns=label_cols, use_pyarrow=True)

        logger.info(f"Reading tasks from {self.config.task_labels_fps}")
        return pl.concat([read_df(fp) for fp in self.config.task_labels_fps], how="vertical")

    def encode_query(self, code_name: str) -> int:
        """Encode query using the canonical code vocabulary mapping."""
        try:
            return int(self.code_to_index.get(code_name, EveryQueryBatch.PAD_INDEX))
        except Exception:
            return EveryQueryBatch.PAD_INDEX

    def _seeded_getitem(self, idx: int, seed: int | None = None) -> dict[str, torch.Tensor]:
        out = super()._seeded_getitem(idx, seed)

        if self.split == held_out_split:
            out["subject_id"] = self.schema_df[DataSchema.subject_id_name][idx]
            out["prediction_time"] = self.schema_df[LabelSchema.prediction_time_name][idx]

        dynamic_data = out["dynamic"]
        schema = dynamic_data.schema
        schema["code"] = np.int16
        query_data = QueryData(code=[self.encode_query(c) for c in self.query_codes[idx]])
        query_as_JNRT = query_data.to_JNRT(self.config.batch_mode, schema)
        out["dynamic"] = JointNestedRaggedTensorDict.concatenate([query_as_JNRT, dynamic_data])

        if getattr(self, "has_occurs", False):
            out["occurs"] = self.occurs[idx]
        if getattr(self, "has_quantifier", False):
            out["quantifier"] = self.quantifier[idx]
        if getattr(self, "has_query_codes", False):
            out["query_codes"] = self.query_codes[idx]
        if getattr(self, "has_duration_days", False):
            out["duration_days"] = self.duration_days[idx]

        return out

    def collate(self, batch: list[dict]) -> EveryQueryBatch:
        out = dict(super().collate(batch).items())

        if self.split == held_out_split:
            out["subject_id"] = torch.as_tensor([item["subject_id"] for item in batch]).long()
            out["prediction_time"] = torch.as_tensor([item["prediction_time"] for item in batch]).long()

        if self.has_task_labels:
            out["censor"] = out[self.LABEL_COL]
        if getattr(self, "has_occurs", False):
            out["occurs"] = torch.Tensor([item["occurs"] for item in batch]).long()
        if getattr(self, "has_duration_days", False):
            out["duration_days"] = torch.as_tensor([item["duration_days"] for item in batch]).float()
        return EveryQueryBatch(**out)
