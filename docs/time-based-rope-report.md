# Time-Based RoPE for EveryQuery — Integration Report

> **August 2026** · `payalchandak/EveryQuery` · branch `claude/time-based-rope-eq-eg9auc`

---

## 1. What EveryQuery Is

EveryQuery is a medical event prediction system. Given a patient's longitudinal medical record — a sequence of coded clinical events (diagnoses, labs, procedures) with timestamps — it answers:

> *"For this patient, given their history up to time t, will medical code C occur within D days?"*

The model produces two outputs per query:
- **occurs_prob**: probability the event happens (conditioned on not being censored)
- **censor_prob**: probability the patient's record ended before we could observe the outcome

This matters because medical data is right-censored — patients leave the health system, and "we never saw it" ≠ "it didn't happen."

Built on [MEDS (Medical Event Data Standard)](https://github.com/Medical-Event-Data-Standard) and `meds-torch-data`. Trains from scratch — no pretrained weights.

---

## 2. Current Architecture

**Backbone**: ModernBERT (`answerdotai/ModernBERT-base` config), initialized from config and trained from scratch. 22 layers, alternating local/global attention, Flash Attention 2 support.

**Forward pass**:
1. Query code token prepended at position 0
2. Duration embedding (MLP-encoded `duration_days`) spliced at position 1
3. Patient event history follows at positions 2+
4. ModernBERT encodes the full sequence
5. Representation at position 0 → censor MLP head + occurs MLP head
6. Loss = `w * occurs_loss + (1-w) * censor_loss`

### Key Components

| Component | File | What it does |
|-----------|------|--------------|
| `EveryQueryModel` | `model/model.py` | `nn.Module`. ModernBERT + MLP heads + duration embedder |
| `EveryQueryLightningModule` | `model/lightning_module.py` | Lightning wrapper: training/validation/predict steps |
| `EveryQueryBatch` | `data/dataset.py` | Batch dataclass with `code`, `time_delta_days`, `censor`, `occurs`, `query`, `duration_days` |
| `EveryQueryPytorchDataset` | `data/dataset.py` | Dataset class. Loads MEDS data, encodes queries, builds batches |
| Training config | `train/configs/config.yaml` | Hydra config. 22 layers, AdamW @ 1e-5, cosine LR, max_seq_len 256 |

---

## 3. How Positional Encoding Works Today

ModernBERT uses **Rotary Position Embeddings (RoPE)** internally. EveryQuery inherits this from HuggingFace — it does not implement its own.

**The critical detail**: position IDs are **sequential integers**.

```python
# From ModernBertModel.forward (HuggingFace transformers)
if position_ids is None:
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
# Result: [0, 1, 2, 3, 4, ..., seq_len-1]
```

These sequential position IDs are fed into `ModernBertRotaryEmbedding`, which computes rotation matrices. The RoPE machinery works correctly — but the positions `[0, 1, 2, ...]` encode **token order**, not **time**.

ModernBERT uses two different RoPE theta values:

| Attention type | Scope | RoPE theta |
|---------------|-------|------------|
| Local (sliding window) | Nearby tokens | `local_rope_theta` (default: 10,000) |
| Global (full attention) | Entire sequence | `global_rope_theta` (default: 160,000) |

### Where time data exists but is unused

The batch already carries `time_delta_days` — a `(batch_size, seq_len)` tensor of elapsed time between consecutive events. This data arrives in every batch, but **the model's forward pass never reads it**. The only time-related input the model consumes is `batch.duration_days` (the query horizon), which gets an MLP embedding.

---

## 4. The Gap: Why Sequential Positions Are Wrong for Medical Events

In NLP, tokens arrive at roughly uniform rate — sequential positions are a reasonable proxy for distance. Medical event sequences are fundamentally different:

> **The problem**: A patient might have a lab test on Monday, another on Tuesday (1 day apart), then nothing for 3 years, then a hospitalization. With sequential positions: `[pos=0, pos=1, pos=2]` — three equally-spaced events. But the temporal gap 1→2 (1 day) vs 2→3 (1,095 days) differs by **three orders of magnitude**.

This matters because:
- **Recency is clinical signal.** A diagnosis 2 days ago ≠ the same diagnosis 2 years ago.
- **Temporal clustering matters.** Three events in one week ≠ three events across three years.
- **The query has a temporal horizon.** "Will X happen within 30 days?" should weight recent events more — RoPE's distance-decay could provide this if "distance" meant "time."

---

## 5. What Time-Based RoPE Is

Standard RoPE rotates query/key vectors by an angle proportional to position:

```
θᵢ = base^(−2i/d)
rotation angle at position m = m · θᵢ
```

**Time-based RoPE** replaces integer position `m` with a real-valued timestamp. Instead of rotating by `m · θᵢ`, rotate by `t(m) · θᵢ`, where `t(m)` is cumulative elapsed time (in days) up to event m.

**You do NOT need to rewrite RoPE.** ModernBERT's `ModernBertRotaryEmbedding.forward()` already casts `position_ids` to float before computing `inv_freq @ position_ids`. You just need to pass it the right position IDs.

---

## 6. Integration Plan

### Step 1: Compute cumulative time position IDs

```python
# time_delta_days shape: (batch_size, seq_len)
# Contains NaN for query/duration tokens (they're synthetic)

clean_deltas = batch.time_delta_days.nan_to_num(0.0)
position_ids = clean_deltas.cumsum(dim=1)   # (B, seq_len)
```

### Step 2: Pass position IDs through to ModernBERT

Modify `_hf_inputs()` to include `position_ids` in the returned dict. ModernBERT's forward already accepts this parameter — when absent, it defaults to sequential integers.

### Step 3: Handle time scale

| Approach | Position range | Trade-off |
|----------|---------------|-----------|
| **Raw days** | 0 – ~3,650 (10yr) | Simple. ModernBERT's global theta (160,000) handles this. |
| **Log-scaled** | 0 – ~8.2 | Compresses long spans. Loses fine-grained recent resolution. |
| **Normalized (÷365)** | 0 – ~10 (years) | Human-interpretable. Recent events very close together. |
| **Capped + scaled** | 0 – max_pos_embed | Preserves relative spacing within bounds. |

**Recommendation: Start with raw days.** RoPE's geometric frequency spacing naturally handles wide position ranges. Test first, then consider alternatives.

### Step 4: Handle edge cases

- **Query token position**: Set to 0 (it's "asked at prediction_time")
- **Duration token position**: Set to 0 (same reasoning)
- **Padding**: Masked out by attention mask; set position to 0
- **NaN time deltas**: Replace with 0 before cumsum
- **Negative time deltas**: Clamp to 0

### Step 5: Make it configurable

Add `use_time_positions: bool` to `EveryQueryModel.__init__` and the Hydra config for A/B testing.

---

## 7. Files to Change

| File | Change | Scope |
|------|--------|-------|
| `model/model.py` | Add `use_time_positions` to `__init__`. Modify `_hf_inputs()` to compute and pass time-based position IDs. | **Primary** |
| `train/configs/config.yaml` | Add `use_time_positions: true` | Config |
| `train/configs/_demo_train.yaml` | Add `use_time_positions: true` | Config |
| `train/configs/fast_config.yaml` | Add `use_time_positions: true` | Config |
| `conftest.py` | Update demo model fixture | Test |
| `tests/test_model_logic.py` | Add position ID tests | Test |

**No changes needed**: `data/dataset.py` (already has `time_delta_days`), `lightning_module.py` (passes batch through), ModernBERT internals (RoPE works as-is).

---

## 8. Risks and Open Questions

### RoPE theta tuning
Default thetas (10k local / 160k global) were tuned for NLP (positions 0–8,192). A 10-year patient history puts positions at 0–3,650 — same order of magnitude, likely fine. Monitor attention patterns.

### `max_position_embeddings`
Currently `max_seq_len + 2` (258). With time-based positions, max position = max cumulative days (potentially thousands). May need to raise substantially or use dynamic rope scaling.

### Query token positioning
Position 0 = "at beginning of history." Alternative: set to max cumulative time = "asked now." Worth experimenting.

### Time delta quality
Medical timestamps can be approximate, back-dated, or out-of-order. Add `clamp(min=0)` after cumsum. Consider upstream temporal deduplication.

### Checkpoint compatibility
Old checkpoints won't have `use_time_positions`. Default to `False` in `load_from_checkpoint`.

---

## Bottom Line

**This is a small, well-scoped change.** The entire implementation lives in `model.py`'s `_hf_inputs()` method — roughly 5–10 lines of new logic. The data pipeline already provides the time deltas, and ModernBERT's RoPE already accepts float position IDs. The risk is low and the potential gain is meaningful: real temporal awareness in a domain where time is a first-class clinical variable.
