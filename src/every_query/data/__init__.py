"""Data submodule: the EveryQuery dataset contract, batch type, and query schema.

The *data layer* — independent of the model architecture (``every_query.model``) and of the
Hydra-driven stage entry points (``train/``, ``generate_tasks/``).  Call through this package
so stage submodules do not need to know the internal file layout::

    from every_query.data import EveryQueryPytorchDataset, EveryQueryBatch, QueryData
"""

from every_query.data.dataset import EveryQueryBatch, EveryQueryPytorchDataset, QueryData
from every_query.data.multitask_dataset import MultitaskBoundaryBatch, MultitaskBoundaryPytorchDataset
from every_query.data.schema import MultitaskBoundarySchema, TaskQuerySchema, empty_task_query_df

__all__ = [
    "EveryQueryBatch",
    "EveryQueryPytorchDataset",
    "MultitaskBoundaryBatch",
    "MultitaskBoundaryPytorchDataset",
    "MultitaskBoundarySchema",
    "QueryData",
    "TaskQuerySchema",
    "empty_task_query_df",
]
