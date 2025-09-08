import logging
from functools import cached_property
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch
from meds import DataSchema, LabelSchema
from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict

from meds_torchdata.config import MEDSTorchDataConfig, StaticInclusionMode
from meds_torchdata.types import BatchMode, MEDSTorchBatch, StaticData, SubsequenceSamplingStrategy

logger = logging.getLogger(__name__)


class EveryQueryPytorchDataset(torch.utils.data.Dataset):
    """A PyTorch dataset that provides efficient PyTorch access to a MEDS dataset.

    Key design principles:
      1. The class will store an `index` variable that specifies what is the valid range of data to consider
         for any given subject in the dataset corresponding to an integer index passed to `__getitem__`.
      2. The primary input to this class in terms of data is a pre-processed set of "schema files" and "nested
         ragged tensor" data files that can be used to identify the shape of the dataset and to efficiently
         load the relevant tensor data, respectively.

    Attributes:
        config: The configuration options for the dataset.
        split: The data split to use.
        schema_dfs_by_shard: A dictionary mapping shard names to the schema DataFrames for that shard.
        subj_locations: A dictionary mapping subject IDs to their locations in the schema DataFrames.
        index: A list of tuples, where each tuple contains the subject ID and the end index for that subject.
        labels: The task labels for the dataset, if any. This will be `None` if there is no task.
    """

    LABEL_COL = LabelSchema.boolean_value_name
    END_IDX = "end_event_index"
    LAST_TIME = "window_last_observed"

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
        super().__init__()

        self.config: MEDSTorchDataConfig = cfg
        self.split: str = split

        logger.info("Reading subject schema and static data")

        self.schema_dfs_by_shard: dict[str, pl.DataFrame] = {}
        self.subj_locations: dict[int, tuple[str, int]] = {}

        for shard, schema_fp in self.config.schema_fps:
            if not shard.startswith(f"{self.split}/"):
                continue

            df = pl.read_parquet(schema_fp, use_pyarrow=True).with_columns(
                pl.col("static_code").list.eval(pl.element().fill_null(0)),
                pl.col("static_numeric_value").list.eval(pl.element().fill_null(np.nan)),
            )

            self.schema_dfs_by_shard[shard] = df
            for i, subj in enumerate(df[DataSchema.subject_id_name]):
                self.subj_locations[subj] = (shard, i)

        if not self.schema_dfs_by_shard:
            raise FileNotFoundError(
                f"No schema files found in {self.config.schema_dir}! If your data is not sharded by split, "
                "this error may occur because this codebase does not handle non-split sharded data. See "
                "Issue #79 for tracking this issue."
            )

        self.index = list(
            zip(self.schema_df[DataSchema.subject_id_name], self.schema_df[self.END_IDX], strict=False)
        )
        self.labels = self.schema_df[self.LABEL_COL] if self.has_task_labels else None
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

    @cached_property
    def schema_df(self) -> pl.DataFrame:
        """Returns the "schema" of this dataframe, cataloging each sample that will be output by row.

        This takes into account both task and non-task data, and is useful for aligning dataloader or model
        outputs to the source inputs.
        """

        base_df = self._all_schemas

        if self.has_task_index:
            df = self.get_task_seq_bounds_and_labels(self.labels_df, base_df)
        else:
            df = base_df.select(
                DataSchema.subject_id_name, pl.col(DataSchema.time_name).list.len().alias(self.END_IDX)
            )

        if (
            self.config.include_window_last_observed_in_schema
            and self.has_task_index
            and self.config.seq_sampling_strategy != SubsequenceSamplingStrategy.RANDOM
        ):
            df = (
                df.join(base_df, on=DataSchema.subject_id_name, how="left", maintain_order="left")
                .with_columns(
                    pl.from_epoch(  # This is a polars error where the timestamp was converted to ints...
                        pl.col(DataSchema.time_name).list.get(pl.col(self.END_IDX) - 1),
                        time_unit="us",
                    ).alias(self.LAST_TIME)
                )
                .drop(DataSchema.time_name)
            )

        return df

    @property
    def _all_schemas(self) -> pl.DataFrame:
        """This is a helper for easy access to the full set of schema dataframes for debugging."""

        return pl.concat(
            (
                df.select(DataSchema.subject_id_name, DataSchema.time_name)
                for df in self.schema_dfs_by_shard.values()
            ),
            how="vertical",
        )

    def __len__(self):
        """Returns the length of the dataset.
        """
        return len(self.index)

    @property
    def has_task_index(self) -> bool:
        """Returns whether the dataset has a task index specified.

        A convenience wrapper around the config property.
        """
        return self.config.task_labels_dir is not None

    @property
    def has_task_labels(self) -> bool:
        """Returns whether the dataset has a task specified with labels.
        """
        return self.has_task_index and (self.LABEL_COL in self.schema_df.collect_schema().names())

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Retrieve a single data point from the dataset.

        This method returns a dictionary corresponding to a single subject's data at the specified index. The
        data is not tensorized in this method, as that work is typically done in the collate function.

        Args:
            idx (int): The index of the data point to retrieve.

        Returns:
            A dictionary containing the static code, static numeric value, dynamic data, and task label (if
            present) for the specified subject.
        """
        return self._seeded_getitem(idx)

    def _seeded_getitem(self, idx: int, seed: int | None = None) -> dict[str, torch.Tensor]:
        """Retrieve a single data point from the dataset with a specified random seed.

        This is merely a deterministic wrapper around the `_getitem` method that allows for deterministic
        subsequence sampling.
        """

        subject_id, end_idx = self.index[idx]
        dynamic_data, static_data = self.load_subject_data(subject_id=subject_id, st=0, end=end_idx)

        match self.config.static_inclusion_mode:
            case StaticInclusionMode.OMIT:
                out = {}
                n_static_seq_els = None
            case StaticInclusionMode.INCLUDE:
                n_static_seq_els = None
                out = {
                    "static_code": static_data.code,
                    "static_numeric_value": static_data.numeric_value,
                }
            case StaticInclusionMode.PREPEND:
                n_static_seq_els = len(static_data.code) if self.config.batch_mode == BatchMode.SM else 1
                out = {"n_static_seq_els": n_static_seq_els}

        dynamic_data = self.config.process_dynamic_data(
            dynamic_data, n_static_seq_els=n_static_seq_els, rng=seed
        )

        if self.config.static_inclusion_mode == StaticInclusionMode.PREPEND:
            static_as_JNRT = static_data.to_JNRT(self.config.batch_mode, dynamic_data.schema)
            dynamic_data = JointNestedRaggedTensorDict.concatenate([static_as_JNRT, dynamic_data])

        out["dynamic"] = dynamic_data

        if self.has_task_labels:
            out[self.LABEL_COL] = self.labels[idx]
        if getattr(self, "has_occurs", False):
            out["occurs"] = self.occurs[idx]
        if getattr(self, "has_query", False):
            out["query"] = self.query[idx]

        return out

    def load_subject_data(
        self, subject_id: int, st: int, end: int
    ) -> tuple[JointNestedRaggedTensorDict, StaticData]:
        """Loads and returns the dynamic data slice for a given subject ID and permissible event range.

        Args:
            subject_id: The ID of the subject to load.
            st: The (integral) index of the first permissible event (meaning unique timestamp) that can be
                read for this subject's record. If None, no limit is applied.
            end: The (integral) index of the last permissible event (meaning unique timestamp) that can be
                 read for this subject's record. If None, no limit is applied.

        Returns:
            The subject's dynamic data and static data. The static data is returned as a StaticData named
            tuple with two fields: `code` and `numeric_value`.
        """
        shard, subject_idx = self.subj_locations[subject_id]

        dynamic_data_fp = self.config.tensorized_cohort_dir / "data" / f"{shard}.nrt"
        subject_dynamic_data = JointNestedRaggedTensorDict(tensors_fp=dynamic_data_fp)[subject_idx, st:end]

        subj_schema = self.schema_dfs_by_shard[shard][subject_idx]
        static_code = subj_schema["static_code"].item().to_list()
        static_numeric_value = subj_schema["static_numeric_value"].item().to_list()

        return subject_dynamic_data, StaticData(static_code, static_numeric_value)

    def collate(self, batch: list[dict]) -> MEDSTorchBatch:
        """Combines a batch of data points into a single, tensorized batch.

        The collated output is a fully tensorized and padded dictionary, ready for input into an
        `input_encoder`. This method uses the JointNestedRaggedTensorDict API to collate and pad the data.

        Args:
            batch (list[dict]): A list of dictionaries, each representing a single sample as
                returned by the __getitem__ method.

        Returns:
            MEDSTorchBatch: A simple, dictionary-like object containing the collated batch data. See the
            [method documentation](../types.py) for more information.
        """

        data = JointNestedRaggedTensorDict.vstack([item["dynamic"] for item in batch])
        data = data.to_dense(padding_side=self.config.padding_side)
        tensorized = {k: torch.as_tensor(v) for k, v in data.items()}

        out = {}
        out["time_delta_days"] = torch.nan_to_num(tensorized.pop("time_delta_days"), nan=0).float()
        out["code"] = tensorized.pop("code").long()
        if self.config.batch_mode == BatchMode.SEM:
            out["event_mask"] = tensorized.pop("dim1/mask")
        out["numeric_value"] = torch.nan_to_num(tensorized["numeric_value"], nan=0).float()
        out["numeric_value_mask"] = ~torch.isnan(tensorized.pop("numeric_value"))

        match self.config.static_inclusion_mode:
            case StaticInclusionMode.OMIT:
                pass
            case StaticInclusionMode.INCLUDE:
                static_data = JointNestedRaggedTensorDict(
                    {
                        "static_code": [item["static_code"] for item in batch],
                        "static_numeric_value": [item["static_numeric_value"] for item in batch],
                    }
                ).to_dense()
                static_tensorized = {k: torch.as_tensor(v) for k, v in static_data.items()}
                out["static_code"] = static_tensorized.pop("static_code").long()
                out["static_numeric_value"] = torch.nan_to_num(
                    static_tensorized["static_numeric_value"], nan=0
                ).float()
                out["static_numeric_value_mask"] = ~torch.isnan(static_tensorized["static_numeric_value"])
            case StaticInclusionMode.PREPEND:
                n_static_seq_els = [item["n_static_seq_els"] for item in batch]

                match self.config.batch_mode:
                    case BatchMode.SEM:
                        static_mask = torch.zeros_like(out["event_mask"])
                        static_mask[:, 0] = True
                    case BatchMode.SM:
                        static_mask = torch.arange(out["time_delta_days"].shape[1]).unsqueeze(
                            0
                        ) < torch.as_tensor(n_static_seq_els).unsqueeze(1)
                        static_mask = static_mask.to(
                            device=out["numeric_value_mask"].device,
                            dtype=out["numeric_value_mask"].dtype,
                        )

                out["static_mask"] = static_mask

        if self.has_task_labels:
            out[self.LABEL_COL] = torch.Tensor([item[self.LABEL_COL] for item in batch]).bool()
        if getattr(self, "has_occurs", False):
            out["occurs"] = torch.Tensor([item["occurs"] for item in batch]).bool()
        if getattr(self, "has_query", False):
            # Encode query using the canonical code vocabulary mapping
            def encode_query(q: str) -> int:
                try:
                    return int(self.code_to_index.get(q, MEDSTorchBatch.PAD_INDEX))
                except Exception:
                    return MEDSTorchBatch.PAD_INDEX
            query_ids = [encode_query(item["query"]) for item in batch]
            out["query"] = torch.as_tensor(query_ids).long()

        return MEDSTorchBatch(**out)

    def get_dataloader(self, **kwargs) -> torch.utils.data.DataLoader:
        """Constructs a PyTorch DataLoader for this dataset using the dataset's custom collate function.

        Args:
            **kwargs: Additional arguments to pass to the DataLoader constructor.

        Returns:
            torch.utils.data.DataLoader: A DataLoader object for this dataset.
        """
        return torch.utils.data.DataLoader(self, collate_fn=self.collate, **kwargs)
