# `model/`

The EveryQuery model itself: the raw `nn.Module` architecture and the Lightning wrapper that
drives training / validation / prediction loops. Pure architecture concerns — no data-layer
shape, no Hydra entry points, no configs.

## What lives here

- **`model.py`** — `EveryQueryModel` (the ModernBERT-style encoder `nn.Module`) and
    `EveryQueryOutput` (the forward-pass output dataclass). The core architecture.
- **`lightning_module.py`** — `EveryQueryLightningModule`. Wraps `EveryQueryModel` for
    PyTorch Lightning with `training_step` / `validation_step` / `predict_step`. Shared between
    training and inference — the same LightningModule's `predict_step` is what `predict/` will
    use at inference time.
- **`task_auroc_callback.py`** — `TaskAurocTrackingCallback`. A Lightning `Callback` owning the
    dataloader over the offline-sampled pos/neg tracking pairs, plus the forward pass that scores
    them each validation epoch. Scores on rank 0 only.
- **`sampled_macro_auroc.py`** — `SampledMacroAUROC`. The `torchmetrics.Metric` the callback feeds:
    accumulates scored rows and macro-averages the per-task win/tie/loss indicator.

Call through the package so stage submodules don't need to know the file layout:

```python
from every_query.model import EveryQueryModel, EveryQueryLightningModule
```

Hydra `_target_` strings in configs use the fully-qualified module path
(`every_query.model.lightning_module.EveryQueryLightningModule`, etc.) for explicitness —
a config reader should see exactly which file the class lives in.

## Relationship to `data/`

The data-layer contract (dataset, batch, query types) lives in
[`every_query.data`](../data/). `model/` has no dependency on any stage submodule and no
dependency on the upstream `generate_tasks/` output layout — it only knows the shape of the
batch it receives, which is defined by `data/`.

This split mirrors MEICAR's `model/` (pure architecture) + MTD's dataset (shared dataset
plumbing). EQ has its own data layer because `EveryQueryBatch` carries query-specific fields
upstream MTD's batch doesn't.

## Pipeline position

```
data/   ─┐
         ├──►  model/  ─────►  predictions / loss
train/  ─┘          ▲
(or predict/)       │
                    Hydra-instantiated via train/configs/*.yaml
```

## Related

- Parent refactor umbrella: [#54](https://github.com/payalchandak/EveryQuery/issues/54)
- Phase 1 submodule restructure: [#79](https://github.com/payalchandak/EveryQuery/issues/79) (this PR)
