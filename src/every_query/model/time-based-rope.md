# Time-based RoPE for EveryQuery — design report

**Status:** research / design doc, no implementation yet.
**Question:** can EQ replace token-index rotary position embeddings (RoPE) with
*time-based* positions — i.e. rotate each token by its (continuous) timestamp rather
than its integer index in the sequence — and what would that take?

**Short answers up front:**

| Question | Answer |
| --- | --- |
| Does the backbone allow it? | **Yes.** ModernBERT is RoPE-only (no learned position table), and HF's `ModernBertModel.forward` accepts a `position_ids` argument whose cos/sin are computed *on the fly* by a matrix product against `position_ids.float()` — so fractional, irregular, real-valued positions work numerically with zero changes to HF code. One big caveat: this is true for the **SDPA/eager** attention path only, *not* FlashAttention 2 (details in §3.2). |
| Do I need a different HF model? | **No.** EQ builds the backbone with `ModernBertModel._from_config(...)` (`model.py:351`) — that is a **random init**; `answerdotai/ModernBERT-base` only supplies the architecture *config*, never pretrained weights. There is no pretrained-weight compatibility to preserve, so you are free to change what "position" means. |
| Is the time signal already available? | **Yes.** Every batch already carries `time_delta_days` (per-token, from `meds-torch-data`); the model currently ignores it entirely. No preprocessing or data-schema change is needed for the basic version. |
| Where does the change live? | Almost entirely in `model/model.py` (`_hf_inputs` + `__init__` + `_check_inputs`), plus a config knob in `train/configs/config.yaml`. |

---

## 1. What you're working with today

### 1.1 The stack

```
generate_tasks/  →  data/ (EveryQueryPytorchDataset → EveryQueryBatch)  →  model/ (EveryQueryModel wraps HF ModernBertModel)  →  train/ / predict/
```

- **Backbone:** `EveryQueryModel` (`model/model.py:258`) wraps HF `ModernBertModel`.
  The config template is `answerdotai/ModernBERT-base` (`AutoConfig.from_pretrained`,
  `model.py:296`), but the weights come from `_from_config` (`model.py:351`) — trained
  **from scratch**. `vocab_size` and `max_position_embeddings` are overridden from the
  data by `train.py:280-281` (`max_position_embeddings = max_seq_len + 2`, the +2
  covering the prepended query token and the inserted duration token).
- **Position encoding today:** ModernBERT has *no* absolute/learned position
  embeddings — RoPE only. EQ never passes `position_ids`, so HF defaults to
  `torch.arange(seq_len)` (`modeling_modernbert.py:856-857`): plain token-index RoPE.
  Two tokens 10 years apart and two tokens 10 minutes apart look positionally
  identical to the model if they are adjacent in the sequence.
- **Sequence layout:** `EveryQueryPytorchDataset._seeded_getitem` prepends the query
  code as token 0 (with `time_delta_days = NaN`, see `QueryData.to_JNRT`,
  `dataset.py:259-264`); `_hf_inputs` (`model.py:552`) then inserts a duration
  *embedding* at position 1 when `duration_days` is present, shifting history tokens
  right by one. So the effective sequence is `[query, (duration), history…]`, sampled
  `to_end` (history ends at the prediction time).

### 1.2 The time signal you already have

`EveryQueryBatch` inherits `time_delta_days: FloatTensor (B, S)` from `MEDSTorchBatch`.
Semantics in SM mode (the only mode `_check_inputs` accepts):

- `time_delta_days[i]` = days from token *i*'s event to the **next** event;
- zeros for measurements that are not the last measurement of their event (i.e. ties
  within an event have delta 0);
- 0/unknown at the final token; **NaN at the prepended query token**.

So the absolute time of token *i* relative to the prediction time `t` is a **reverse
cumulative sum**: `time_before_pred[i] = Σ_{j≥i} time_delta_days[j]` (last history
token ≈ 0, earliest token the most negative / largest lookback). RoPE attention scores
depend only on position *differences*, so any consistent offset/sign convention works.

The model currently consumes only `batch.code` and `batch.duration_days`;
`time_delta_days` flows through collate and is dropped on the floor at the model
boundary. Time-based RoPE is precisely the act of picking it up.

---

## 2. What "time-based RoPE" means here

Standard RoPE rotates query/key vectors at head-dim-pair *k* by angle
`θ_k · p`, with `θ_k = base^{-2k/d}` and `p` = token index. The attention logit then
depends on `θ_k · (p_i − p_j)` — a *relative* position code. Substituting
`p = time_in_days` makes the attention logit depend on **elapsed time between events**
instead of token count. Nothing about the math requires `p` to be an integer.

This is the natural fit for MEDS data: EHR event streams are irregularly sampled, and
"3 tokens ago" can mean anything from minutes to decades. Related prior art you may
want to skim: RoFormer (Su et al., 2021 — the RoPE paper; nothing in it requires
integer positions), and the EHR-transformer line that injects continuous time some
other way (CEHR-BERT's time embeddings, STraTS's continuous-value time encoding,
ETHOS's time-interval tokens — the comparison model in your experiments repo). Using
*rotary* specifically for continuous time is less common than additive time
embeddings, which makes it a nice contribution but also means you should budget for
an ablation against an additive-time-embedding baseline.

---

## 3. Does the backbone allow it?

### 3.1 Yes — the SDPA/eager path takes arbitrary real-valued positions

In the installed `transformers==4.57.6`:

- `ModernBertModel.forward` has a `position_ids` parameter
  (`modeling_modernbert.py:788`), threaded into every layer's attention.
- `ModernBertRotaryEmbedding.forward` (`modeling_modernbert.py:267-273`) computes
  `freqs = inv_freq @ position_ids.float()` — a **dynamic matrix product, not a
  cached table lookup**. The `LongTensor` type hint is cosmetic; a `FloatTensor` of
  shape `(B, S)` works, and fractional positions produce exactly the cos/sin you'd
  want.
- ModernBERT has no learned absolute position table, so `max_position_embeddings` is
  not a hard indexing limit for the non-FA2 rotary path — large "positions" (e.g.
  30,000 days) are numerically fine; whether they're *well-scaled* is a design
  question (§5.3).

### 3.2 The one real obstacle: FlashAttention 2 ignores `position_ids`

`model.py:324-326` auto-enables `attn_implementation="flash_attention_2"` whenever
`flash_attn` imports. On that path ModernBERT unpads the batch and applies
`ModernBertUnpaddedRotaryEmbedding` driven by `cu_seqlens` / `max_seqlen` only
(`modeling_modernbert.py:355-365`) — i.e. **token index within the unpadded
sequence, from a cached integer-indexed cos/sin table**. Custom `position_ids` are
silently unused. Consequences:

- **Phase 1 must force SDPA when time-RoPE is on** (one-line gate in
  `EveryQueryModel.__init__`). ModernBERT's docstring notes SDPA "requires padding
  and unpadding inputs, adding some overhead" — acceptable for experiments at
  `max_seq_len=256`, `batch=128`.
- FA2 + time-RoPE later is possible but is real work: precompute per-token cos/sin
  from continuous times and call `flash_attn`'s rotary kernels with them (or a small
  fork of the unpadded rotary class). Don't do this until time-RoPE has shown signal.

### 3.3 Second-order interactions to keep in mind

- **Local/global alternation.** Every 3rd layer is global; the others use a ±64-token
  sliding window whose mask is computed in **token-index space**
  (`_update_attention_mask`), regardless of what you pass as `position_ids`. So with
  time-RoPE, local layers become "nearest 128 tokens, scored by time-aware rotary" —
  coherent, but the *window* stays count-based. Fine to leave; note it in the paper.
- **Two rope thetas.** Local layers use `local_rope_theta=10000`, global layers
  `global_rope_theta=160000` (config defaults). Both are tuned for integer token
  positions up to a few thousand. With days as the unit, you'll likely want to
  retune/override these (§5.3) — both are already reachable via the existing
  `config_overrides` mechanism without new plumbing.
- **`max_position_embeddings = max_seq_len + 2`** (`train.py:281`) also sizes the FA2
  rotary cache and the local-attention config copy; on the SDPA path it's only a
  bookkeeping number. `_check_inputs`'s seq-length check keys off it, not off time
  values — no change needed there for time positions.

---

## 4. Where the change lives (file-by-file)

| File | Change |
| --- | --- |
| `model/model.py` — `__init__` | New hparams: `position_mode: "index" \| "time" \| "hybrid"` (+ optional `time_unit_days`, `hybrid_alpha`). Record in `self.hparams`. Force SDPA (skip the FA2 branch) when `position_mode != "index"`, with a log line saying why. |
| `model/model.py` — `_hf_inputs` | The heart of it. Build `position_ids (B, S[+1])` from `batch.time_delta_days`: nan-safe reverse cumsum → per-token time-before-prediction; assign positions to the query token (and the inserted duration row) per §5.4; return them in the kwargs dict alongside `inputs_embeds`/`attention_mask`. Padding tokens get any finite value (they're masked out). |
| `model/model.py` — `_check_inputs` | When time mode is on: assert `time_delta_days` present, finite after fill, non-negative deltas (monotone times); assert no FA2. |
| `train/configs/config.yaml` | Expose `position_mode` (default `"index"` so existing runs are byte-identical); document `config_overrides.global_rope_theta` / `local_rope_theta` as the retuning knobs. |
| `data/dataset.py` | **No change needed** for phase 1 — `time_delta_days` already reaches the batch, query-token NaN and all. |
| Tests | Unit tests for the position builder (ties → equal positions; NaN query slot handled; padding finite; `index` mode reproduces `arange`-equivalent behavior). Extend the `Mock` batches in `model.py` doctests with `time_delta_days`. A `tests/training_validity/` signal run for `position_mode=time` once wired. |

Everything else — Lightning module, predict, evaluate, task generation — is untouched:
they see the same batch and the same output contract.

---

## 5. Design decisions you have to make

### 5.1 Position definition

Recommended: `p_i = −(days before prediction time)` via reverse cumsum, so history
positions are ≤ 0 and increase toward the prediction point. RoPE only sees
differences, so this is equivalent to a forward cumsum from sequence start — but
anchoring at the prediction time makes the query/duration token placement (§5.4)
natural and makes positions comparable across subjects.

### 5.2 Pure time, or hybrid?

Pure time positions make *simultaneous* measurements (delta 0 within an event)
positionally identical — the model loses within-event ordering. That ordering is
mostly arbitrary in MEDS, so this may be a feature, but it also means RoPE
contributes zero rotation for a large fraction of adjacent-token pairs, and sub-day
structure vanishes if the unit is days. Three options, cheapest first:

1. **Pure time** (`p = t`): cleanest story; risk of degenerate ties.
2. **Hybrid scalar** (`p = idx + α·t`): one knob, keeps a within-event ordering
   signal; positions no longer purely interpretable.
3. **Split-spectrum / partial rotary** (some head-dim pairs rotate by index, others
   by time): the principled version, but requires a custom rotary module rather than
   the stock `position_ids` hook — phase 2+ material.

Recommendation: implement 1 and 2 behind `position_mode` (they share all plumbing)
and ablate; 3 only if the first results demand it.

### 5.3 Units and theta (the scaling question)

With `d_head=64` and theta=10000–160000, RoPE wavelengths span ~6.3 units up to
~10⁶ units. If the unit is *days*, the fastest components have ~week-scale
wavelengths — anything intra-day is invisible, and multi-decade lookbacks sit deep in
the slow tail. Options: express time in hours; or rescale (`t / time_unit_days`); or
retune thetas via `config_overrides`. Since your weights are from scratch, there is
no "distribution shift vs. pretraining" concern — pick units so that the *clinically
meaningful* range (hours → a few years) falls across the middle of the frequency
spectrum. Worth one small sweep. A log-compressed time (`sign preserved`) is another
option but breaks the translation-invariance interpretation of RoPE — mention as
ablation only.

### 5.4 Positions for the two synthetic tokens

- **Query token (index 0):** its `time_delta_days` is NaN by construction. Two
  choices: place it at the prediction time (`p=0`, "you are here"), or at the query
  horizon (`p = +duration_days`). The latter is genuinely interesting: the model
  would perceive the *elapsed time* between any history event and the queried future
  window directly in attention — and it would let you ablate whether the separate
  duration-embedding token (`model.py:572-580`) is still needed at all.
- **Duration token (inserted position 1):** if kept, give it the same anchor as the
  query token (`p=0` or `p=+d`). Remember `_hf_inputs` builds it at the *embedding*
  level, so the position vector must be spliced identically (length `S+1`, insert at
  1).

### 5.5 Guardrails

- Nan-safe cumsum: `nan_to_num(time_delta_days, 0.0)` before summing; assert
  finiteness afterward (extend `_check_inputs`, demo mode already runs it).
- Keep `position_mode="index"` the default until the signal test passes, so `main`
  behavior is unchanged and any regression is opt-in.
- W&B: log the resolved position mode + unit + thetas via `self.hparams` so runs are
  distinguishable.

---

## 6. Suggested phasing

1. **Phase 1 (small PR):** `position_mode` knob, SDPA gate, position builder in
   `_hf_inputs`, `_check_inputs` extensions, unit tests. Pure-time + hybrid modes.
   ~1 file of real logic.
2. **Phase 2 (experiments, in `EveryQueryExperiments`):** index vs. time vs. hybrid
   ablation; unit/theta mini-sweep; query-token-at-horizon vs. duration-token
   ablation.
3. **Phase 3 (only if adopted):** FA2-compatible unpadded time-rotary; optional
   split-spectrum rotary module.

## 7. Bottom line

You are in an unusually good position to do this: the backbone is RoPE-only, the HF
implementation computes rotary angles dynamically from whatever `position_ids` you
hand it, your weights are from scratch (no pretrained positional prior to fight), and
the continuous time signal is already sitting in every batch, unused. The entire
phase-1 change is ~50 lines in `model/model.py` plus a config knob — the real work is
the design choices in §5 and the ablations to justify them.
