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
- **`answers.py`** — the architecture-independent pieces every query-answering model needs: the
    binary answer vocabulary (`ANSWER_NO` / `ANSWER_YES` / `N_ANSWER_CLASSES`),
    `validate_rope_time_pair` (keeps the model's `use_rope_time` and the batch's `time_pos_ids`
    from drifting apart) and `_init_aux_embeddings` (re-inits tables built outside the HF
    backbone to the backbone's scale).
- **`conditional_multitask_ar_model.py`** — `ConditionalMultitaskARModel`: the decoder-only
    *all-vocabulary* architecture over ordered windows `[patient, W0, C0, A0, …, W(K-1)]`. Each
    window's hidden state is projected onto the tied input-embedding table (one logit per code,
    masked BCE against packed `(B, K, V)` targets); `score_final_query` scores one code at one
    window for QuerySeq grids without building `(B, K, V)`. With `ontology_dir` the table is the
    ancestor-mixed `V_ext` one on both the input and readout sides, and leaf-only `(B, K, V)`
    targets are widened to `V_ext` inside `forward` (`derive_ancestor_targets`: an ancestor's bit
    is the OR of its descendant leaves'), so the sampler's sidecars stay leaf-only. The closure is
    checked against the cohort by *identity*, not width: `train.py` records the cohort's vocabulary
    fingerprint (`cohort_vocab_fingerprint`, the multitask manifest's `vocab_fingerprint`) as a
    model hparam, and every construction, checkpoint loads included, requires the ontology's
    observed nodes to digest to it, so a same-width foreign or renumbered ontology is refused.
- **`conditional_multitask_lightning.py`** — `ConditionalMultitaskLightningModule`: fit /
    validation on `MultitaskBoundaryBatch` (dense loss), test / predict on `MultitaskEvalBatch`
    (target-only scoring).
- **`ontology_embedding.py`** — `OntologyEmbedding` + `wrap_tok_embeddings`: ancestor-mixed
    code embeddings installed through `get_input_embeddings()`/`set_input_embeddings()`, shared
    by every architecture's patient, query-code and boundary-code lookups.

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
