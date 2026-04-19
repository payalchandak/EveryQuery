# `data/`

The EveryQuery data layer: the PyTorch `Dataset` contract, the `EveryQueryBatch` named-tuple,
and the `QueryData` query-schema type. All query-specific data-shaping lives here, separate
from the model architecture.

## What lives here

- **`dataset.py`** — `EveryQueryPytorchDataset`, `EveryQueryBatch`, `QueryData`. The PyTorch
    `Dataset` implementation that maps tensorized MEDS shards + a task-labels parquet into the
    query-aware batches the model consumes. Shared between the `train/` stage (training loop)
    and the future `predict/` stage (inference).
- **`schema.py`** — `TaskQuerySchema`. Cross-stage contract for the `(subject_id, prediction_time, query, duration_days)` rows produced by `generate_tasks/` and consumed
    by `predict/` + `evaluate/`. Extends MEDS `LabelSchema` so inference and evaluation share
    the same row shape — labels live on the inherited `boolean_value` column when present.
    Initial scope (#80) is narrow: flat single query code + continuous duration. Extensions
    come later as the pipeline matures.

Call through the package so stage submodules don't need to know the file layout:

```python
from every_query.data import EveryQueryPytorchDataset, EveryQueryBatch, QueryData
```

Hydra `_target_` strings in configs use the fully-qualified path
(`every_query.data.dataset.EveryQueryPytorchDataset`) for explicitness.

## Why `data/` is separate from `model/`

The data layer evolves on a different schedule than the model architecture. The task-query
schema (#80) lives here, separate from the `nn.Module`, so schema changes diff only `data/`
and its two call-site edges (`generate_tasks/` output wiring + `train/` dataloader wiring).

## Pipeline position

```
generate_tasks/ ─► data/  ──►  model/  ─►  train/ / predict/
                  (batch      (architecture)
                   contract)
```

`data/` depends only on `meds_torchdata` for the upstream `MEDSTorchDataConfig` — it has no
dependency on `model/`, `train/`, or any stage submodule.

## Related

- Parent refactor umbrella: [#54](https://github.com/payalchandak/EveryQuery/issues/54)
- Phase 1 submodule restructure: [#79](https://github.com/payalchandak/EveryQuery/issues/79) (this PR)
- Batch-schema evolution: [#80](https://github.com/payalchandak/EveryQuery/issues/80)
