# EQ-multitask — the all-vocabulary conditional query model

This is the design doc for **EQ-multitask** (`ConditionalMultitaskARModel`), the second of the two
pipelines this repository ships. The other, **EQ-single**, is the original upstream single-query
model, unchanged; the [README](../README.md) has the walkthrough for both.

EQ-multitask reworks EveryQuery from **one query per forward pass** into **every query at once**. A
single decoder-only backbone reads a patient's tokenized event stream and then a short ordered
sequence of *windows*. Each window token's hidden state is projected back onto the backbone's own
input-embedding table, so one window produces one logit for **every code in the vocabulary** — the
training objective is a masked BCE over `(batch, K, V)`, not over a sampled query. Between windows
sit a conditioning code and its teacher-forced answer, so the model learns
`P(target_v at window i | patient, W_0..W_i, (C, A)_0..(C, A)_{i-1})` rather than the marginal
`P(occurs | patient, query)`.

Two things follow from that shape and are worth stating up front, because they are what the rest of
this document is about: a "query" here is a **window**, not a horizon — it can open later than the
prediction time and close on an event — and the thing being measured is a **task cell**, not a
pooled stream of predictions.

______________________________________________________________________

## 1. Model design

`src/every_query/model/conditional_multitask_ar_model.py`.

### 1.1 One backbone, one stream

A single Hugging Face `LlamaModel` (from a configurable `LlamaConfig`, trained from scratch;
`use_cache=False`) processes the combined sequence

```
[p₁..pₘ, W₀, C₀, A₀, W₁, C₁, A₁, …, W_{K-1}]
```

— patient events, then `TOKENS_PER_WINDOW · K − 2` query tokens. `W_i` describes window `i`, `C_i`
names a code *asked about* for that window — drawn uniformly from the sampler's configured
conditioning pool, independently of the labels — and `A_i` supplies its teacher-forced answer,
looked up afterwards, which is `NO` for most codes, since most codes are rare. `C_i` is a question, not an assertion: if it named a code known to be present, `A_i`
would be constant and the conditioning would carry no information. There is no
trailing `(C, A)` pair after the last window: an answer exists to condition *later* windows, and
there are none.

Each patient row's query stream starts **immediately after its last real event** (no padding gap),
which requires right-padded patient tokens — the model checks the prefix rule rather than assuming
it, because left padding or an interior PAD would silently overwrite real tokens with query tokens.
`train.py` sizes `max_position_embeddings` to `max_seq_len + 3 · max_windows`.

**Plain causal attention.** The model passes a 2-D padding mask and lets Llama's standard
token-level causal mask do the rest. The invariant is that `W_i` sees the whole patient history,
every earlier `(W, C, A)` triple, and itself — never `A_i` (there is no `A_i` for its own window's
target) and nothing later — so all `K` windows' logits come out of one forward pass with no label
leakage. Queries attend to patient tokens directly by self-attention, and patient tokens are
themselves encoded causally.

**The readout is tied to the input table.** `logits = window_hidden @ E.T + code_bias`, where `E` is
the *effective* input embedding table — the ancestor-mixed one when an ontology is configured (§3),
the raw one otherwise. There is no separate output matrix, and `code_bias` (initialised to −3.0,
in the optimizer's no-decay group) carries the base rate that would otherwise have to be learned
into every row.

### 1.2 What a window is

A window is a pair of resolved timestamps, not a horizon:

```
start = prediction_time + start_duration        (0 => the prediction time itself)
      | first occurrence of start_event strictly after prediction_time
end   = start + duration                        (measured from the RESOLVED start)
      | first occurrence of bound_event strictly after the resolved start
target[v] = start < some occurrence of v < end                 (both endpoints open)
```

Both endpoints are open. An event boundary that never recurs runs to the end of the record; an
event start that never occurs leaves the window **empty**, so every target is false — not "the
window opens at the prediction time". The `-1.0` `EVENT_BOUND_DURATION_SENTINEL` marks an
event-bounded end or an event-defined start in the tensors, and `NO_BOUND_INDEX` (`0`) marks "no
event here".

The window token itself is built from role-distinct halves: a start spec (a duration MLP output, or
the start code's embedding plus a learned `start_marker`) plus an end spec (the same construction
with `bound_marker`), so "starts at code `X`" and "ends at code `X`" are never the same vector.
Durations are scaled by 365 days before the MLP, and the MLP's output layer is re-initialised to the
embedding scale — at the default `Linear` init its output norm is ~10× an embedding row, which let
the duration dominate the whole window token.

Event bounds are the piece most easily got silently wrong, and the test suite treats them that way:
the window is open at both ends, the boundary is the *first* occurrence strictly after the start,
and a target sharing a timestamp with the boundary does not count. That last rule is load-bearing on
MEDS data, where a discharge and everything charted with it routinely share one instant. See
[`tests/README.md`](../tests/README.md).

### 1.3 Position encoding

Four mechanisms with disjoint jobs.

- **Clinical-time RoPE.** With `use_rope_time=true` (the default in the shipped config) the rotary
    positions are the dataset's elapsed-hour `time_pos_ids` with delta tokens stripped, and *every*
    query token repeats the final real patient event's hour. That event is the prediction time and
    every window is specified at it, so no clinical time passes across `W_i, C_i, A_i` or between
    windows — earlier answers are logical conditioning, not later observations. A window that opens
    seven days out is expressed in its *token*, not in its rotary position.
- **Block-position** embeddings (learned, `max_windows` entries) carry window order, added
    identically to all three tokens of a block and never to patient tokens.
- **Token-type** embeddings carry the slot role: patient, window, condition-code, condition-answer.
- **The causal mask** — derived from physical token order, not from RoPE — enforces autoregressive
    visibility, so repeated rotary positions are safe.

### 1.4 Conditioning semantics

The logits at `W_i` estimate `P(v occurs in window i | patient, W_0..W_i, (C, A)_0..(C, A)_{i-1})`
for every `v`. Earlier answers are caller-supplied conditioning values, teacher-forced in training
and — at evaluation — taken from the grid's own true answers, even where a designed spec fixes one
(`forced_answer`, see `generate_tasks/README.md`): the grid keeps that spec only at the contexts
whose truth agrees, so the fed `A_i` is still the truth and no counterfactual is ever asked. The
label the final query is scored against is always the truth, and the final query itself can never be
forced. Feeding the model's own predictions
back in as later conditions is a separate sampling capability, deliberately not implemented.

At evaluation the model takes a second path,
`ConditionalMultitaskARModel.score_final_query`: the *same* hidden states, but only the last real
window of each row is projected, onto only that row's scored code. No `(B, K, V)` logit tensor is
built and `batch.targets` is never read, so scoring a grid costs one backbone pass plus `B` dot
products per batch.

### Censoring is expressed as a query, not a label

`TIMELINE//END` — the real MEDS end-of-timeline code, emitted once per subject at the record's last
event — is an ordinary member of the vocabulary and an ordinary conditioning code. A window in which
`TIMELINE//END` occurs is one in which the record ends; conditioning a later window on that answer
**recovers and generalizes** the original EveryQuery's implicit `P(occurs | data exist after d)`:

- `[END d]=NO,  [C d]` → `P(C | data continue past d)` (= original EveryQuery; ≈0 for terminal codes)
- `[END d]=YES, [C d]` → `P(C | record ends within d)` (the actionable form for death etc.)
- `[C d]` alone → the marginal `P(C observed)`; recoverable as a prevalence-weighted average.

This is **strictly more expressive** than the original EveryQuery, which could only ever express the
`END=NO` slice. It is also why the answers on an evaluation grid are binary and never null: an
occurrence the record is too short to observe is `False`, and censoring is carried by an explicit
`TIMELINE//END` query rather than by a missing label. `EQ_evaluate_multitask` rejects a null label
for exactly that reason.

> **v1 leak post-mortem.** An earlier design put a *same-horizon* censor query first, teacher-forced
> its answer, and 3-valued-labeled subsequent queries with censored outcomes masked from the loss.
> For terminal events this is catastrophic: death ends the record, so "data after t+30d?" is the
> logical complement of "died by 30d?", and the masking left the surviving death labels perfectly
> determined by the censor answer. The model learned to copy it — 30-day mortality AUROC came out at
> 0.991, and a classifier reading *only* the censor answer scored 0.996 on the same rows. The
> binary-occurrence + END-as-query design here removes that leak structurally.
>
> Those two figures are kept deliberately, and they are the only numbers in this document. They
> measure a defect in a design that no longer exists, not the performance of anything that ships
> here — the second exceeding the first is the whole point, and the argument is much weaker without
> them.

______________________________________________________________________

## 2. Pipeline & CLIs

| CLI                                                      | Purpose                                                                                                                     |
| -------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `EQ_build_ontology`                                      | Derive the code-ontology DAG (nodes / mix / closure parquets) that ancestor windows and mixed embeddings read.              |
| `EQ_generate_multitask_sequences`                        | Sample `K` windows per patient context and label **every** base-vocabulary code at every window, bit-packed into a sidecar. |
| `EQ_generate_evaluation_query_sequences`                 | Label the *same* `N` query specifications at every context of a cohort — the dense evaluation grid.                         |
| `EQ_train --config-name=conditional_multitask_ar_config` | Train `ConditionalMultitaskARModel` via `ConditionalMultitaskLightningModule`.                                              |
| `EQ_predict_multitask`                                   | Score each grid row's final query with the trained model: one scalar probability per row.                                   |
| `EQ_evaluate_multitask`                                  | Group those rows by task (query spec + prior answers); one AUROC and 95% row-bootstrap CI per task.                         |

The evaluation half is one chain, and the grid in the middle is a single artifact:

```
EQ_generate_evaluation_query_sequences  ->  QuerySeqSchema eval grid  ->  EQ_predict_multitask
                                                                      ->  EQ_evaluate_multitask
```

`sample_evaluation_query_sequences.py` is worth calling out, because its name misleads everyone who
meets it. It says "query sequences", but it is **the** evaluation-grid generator for the multitask
model: `EQ_predict_multitask` reads exactly its output, and it is the only generator that can emit
the explicit window starts (a start delay or a start event, with the end resolved relative to the
resolved start) that the multitask model consumes. Its window rule is deliberately identical to the
training sampler's — same open/open endpoints, same resolved-start rule, same empty-window and
infinity conventions — which is what makes a grid a valid measurement of a model trained on the
other sampler's output.

`EQ_predict_multitask` runs the Lightning predict loop (`Trainer.predict`) over a
`ConditionalMultitaskDataModule` rebuilt from the checkpoint's own cohort settings with only the
label root swapped for the grid. It is single-device and single-process **by construction** — a
multi-device trainer or a `torchrun` / `srun --ntasks>1` launch is refused — because the output is
concatenated in loader order and must stay row-aligned with the grid; the collated final-query
labels and scored codes are then re-checked row by row against the grid before anything is written.

### Key source modules

- `src/every_query/model/conditional_multitask_ar_model.py` — `ConditionalMultitaskARModel`,
    `TOKENS_PER_WINDOW = 3`, the four token types, `ConditionalMultitaskOutput`,
    `window_hidden_states` (shared by both paths) and `score_final_query`.
- `src/every_query/model/conditional_multitask_lightning.py` — the Lightning module. Each loop is
    bound to one dataset, batch type and model path; nothing branches on `self.training`. Logged
    metrics are `train/loss`, `tuning/loss` and `held_out/loss` only — full-vocabulary macro metrics
    are deliberately deferred to `EQ_evaluate_multitask` (§4).
- `src/every_query/model/answers.py` — `ANSWER_NO` / `ANSWER_YES` / `N_ANSWER_CLASSES`, plus
    `_init_aux_embeddings` and `validate_rope_time_pair`, shared with the datasets.
- `src/every_query/model/ontology_embedding.py` — `OntologyEmbedding` / `wrap_tok_embeddings`:
    substituting the *embedding module* (not the call sites) is what lets patient, start, bound and
    condition codes all inherit the ancestor mix.
- `src/every_query/data/multitask_dataset.py` — `MultitaskBoundaryPytorchDataset` /
    `MultitaskBoundaryBatch`. Memory-maps each `.labels.npy`, gathers packed rows and unpacks once per
    batch, and re-checks every stored conditioning answer against the unpacked target bit.
- `src/every_query/data/multitask_eval_dataset.py` — `QuerySeqMultitaskEvalDataset` /
    `MultitaskEvalBatch`: the adapter that maps a `QuerySeqSchema` grid row onto the model's window
    tensors (`queries[:-1]` / `answers[:-1]` become conditioning pairs; `queries[-1]` is scored).
- `src/every_query/data/query_seq_dataset.py` — `QuerySeqPytorchDataset` (the eval dataset's base),
    `EOS_CODE = "TIMELINE//END"`, `EVENT_BOUND_DURATION_SENTINEL`, `NO_BOUND_INDEX` and the
    `QuerySeqSchema` column names the samplers, predictor and evaluator all share.
- `src/every_query/data/schema.py` — `QuerySeqSchema` (`queries` / `durations` / `answers` list
    columns plus the optional `bound_events` and, for evaluation grids, `start_durations` /
    `start_events`) and `MultitaskBoundarySchema` (the training metadata rows), on top of upstream's
    `TaskQuerySchema`.
- `src/every_query/generate_tasks/sample_multitask_sequences.py` — the staged training sampler
    (prediction-time map → window draw → contexts → index → per-shard labelling), writing packed
    targets incrementally through a temporary `open_memmap` so no shard-wide target tensor is ever
    allocated.
- `src/every_query/generate_tasks/sample_evaluation_query_sequences.py` — the dense grid, plus
    `interval_table.py`, the subject-sorted interval kernel both samplers label through.
- `src/every_query/generate_tasks/query_sequence_labeling.py` — the shared labelling library the
    grid generator imports: the query universe, the sequence distribution, and the
    `label_with_event_bounds` / `label_with_explicit_starts` labellers that are the correctness oracle
    the multitask window rule is tested against.
- `src/every_query/predict/predict_multitask.py` and
    `src/every_query/evaluate/evaluate_multitask.py` — §4.
- `src/every_query/train/configs/conditional_multitask_ar_config.yaml` — the training config
    (12-layer Llama, hidden 384, 6 heads, `use_rope_time: true`, `max_windows: 5`, bf16-mixed;
    `vocab_size` and `max_position_embeddings` are sized from the data by `train.py`).

______________________________________________________________________

## 3. Ontology queries

MEDS code names are already a hierarchy, so `EQ_build_ontology` turns every `//`-prefix into a DAG
node and mints ancestor tokens above the highest leaf. The README's
[How the ontology works](../README.md#how-the-ontology-works) covers the build; what matters here is
where the ontology enters the *model*, which is two distinct places that are easy to conflate.

**Targets are always leaves.** The training sidecars are `V` wide — the cohort's observable codes,
bits aligned to the unchanged `code/vocab_index`. Under the window rule an ancestor's bit is exactly
the OR of its descendant leaves' bits, so the model derives ancestor targets per batch inside
`forward` (`derive_ancestor_targets`) instead of storing them. Nothing wider than `(B, K, V)` ever
crosses host to device, and adding an ontology changes no bit in any `.labels.npy`: `vocab_size`,
`packed_width_bytes` and `vocab_fingerprint` are identical in every mode.

**Ancestors act as events.** What the ontology *does* change in the sampler is the other half of a
window — an ancestor-valued `start_event` or `bound_event` ("until the next occurrence of any
`LAB//220645//*`") and an ancestor-valued conditioning code. That is what `ontology_mode`
(`boundaries` | `conditions` | `boundaries+conditions` | `none`) selects. The event stream is
exploded through the closure so an ancestor has ordinary intervals; the labelling table is then
rebuilt from the `code_index < V` rows, which the closure's self-pairs make identical to the
unexploded stream.

**Embeddings are mixed, and so is the readout.** The input embedding becomes `(A W)[ids]`: each
code's vector is the row-normalised weighted average of its own row and its ancestors', with an
ancestor `d` levels up contributing `decay ** d`. A rare leaf is pulled toward its better-estimated
parents, and an ancestor node — which never appears in a patient stream — still gets gradient
through its descendants. The tied readout projects onto that same **mixed** table: tying it to the
raw learned rows would score ancestors through rows the input side never sees.

**One ontology, checked by identity.** Ancestor token indices are assigned by the build, so the same
directory must be used for generation, training and evaluation. Width checks alone are not enough —
a same-width but *permuted* ontology would pair the leaf columns with the wrong closure rows and
produce a well-formed parquet of wrong answers — so the model persists the cohort's
`vocab_fingerprint` in its hyperparameters and re-verifies the ontology against *that* cohort on
every construction and every checkpoint load.

______________________________________________________________________

## 4. Evaluation methodology

**Use macro (per-task) AUROC, not pooled AUROC.** Pooled AUROC scores cross-task pairs — a positive
for one query against a negative for a *different* query — so it is dominated by base-rate
differences between queries and systematically overstates per-query skill. Since AUROC is
`P(score_pos > score_neg)` (Mann–Whitney), the quantity worth reporting is the AUROC computed
**within** one query specification, then **macro-averaged** over specifications.

That argument has a concrete home now: it is exactly why `EQ_evaluate_multitask` groups instead of
pooling.

### The grouping key is the query specification plus the conditioning answers

```python
TASK_KEY = [
    "queries",
    "durations",
    "start_durations",
    "start_events",
    "bound_events",
    "forced_answers",  # a designed spec's cohort selector; all-null when nothing is forced
    "prior_answers",
]  # prior_answers = answers[:-1], derived by the evaluator
```

`EQ_generate_evaluation_query_sequences` resolves `N` specifications once and labels **every one of
them at every context**, so the five list columns recover exactly those `N` specs, each populated by
the whole cohort (a designed spec with a `forced_answer` is the exception: it is written only where
its conditioning really happened). `prior_answers` — the teacher-forced answers the final query was conditioned on — then
splits each spec by *what the model was told*. A cell is one conditional question,
`P(A_K | patient, Q_1..Q_K, A_1..A_{K-1} = a)` for one fixed `a`, and within it the only thing
varying across rows is the patient, which is what a per-task metric needs.

Without `prior_answers` in the key a spec's AUROC pools contexts that were told different things,
and the conditioning answer can then separate the classes on its own: a `TIMELINE//END=YES` prefix
all but determines a later death label, so a model that merely echoes its conditioning scores well
while discriminating nothing within either group. That is the pooled-AUROC failure above, one level
down.

The price is sparsity. A `K`-query spec splits into up to `2^(K-1)` cells skewed hard toward
all-`False` (most codes are rare), and AUROC is undefined on a single-class cell, so expect many null
`auroc` rows at `K > 1` and read `n_rows` / `n_positive` next to every cell. A one-query spec has
`prior_answers = []` and stays one cell. The sampled grid draws `K` from `min_queries..max_queries`
(1..3 by default), so it yields more cells than `num_evaluation_sequences` — and *which* cells exist
depends on the cohort, since a conditioning nobody in the cohort has produces no row. The score is
`prob` and the class label is `label`, i.e. `answers[-1]`.

`duration_bucket` is emitted alongside each cell as a *descriptive* rollup axis, never as a grouping
key — bucketing lumps distinct horizons, and at `K > 1` it would pool rows whose conditioning
contexts differ, which is the axis most worth separating. An event-bounded query gets its own
`event-bound` bucket rather than falling through the horizon ladder, where the `-1.0` sentinel would
file it under the shortest horizon.

### One table

`<metrics_stem>.by_task.parquet` — one row per cell: the spec and `prior_answers`, `target_code`,
`n_queries`, `duration_bucket`, `n_rows` / `n_positive` / `prevalence`, `n_subjects`, `auroc` (null
when the cell is single-class), its 95% interval `auroc_ci_lo` / `auroc_ci_hi`, and
`n_degenerate_replicates`.

That is the whole output: the per-task AUROC and its interval. There is no macro and no cross-task
interval — a macro is `by_task["auroc"].mean()` if you want one, and how to weight and resample
tasks is a decision for whoever reads the table, not the evaluator.

### The bootstrap

Each task is bootstrapped on its own: draw the task's rows with replacement, recompute the AUROC,
repeat `n_resamples` times (default 1000), and read `auroc_ci_lo` / `auroc_ci_hi` off the 2.5th and
97.5th percentiles. `bootstrap_seed` (default 0) fixes the draws, and rows are put in a canonical
order first, so a given predictions parquet always yields the same intervals however it was written.

**The resampling unit is the row** — each `(subject_id, prediction_time)` is one prediction.
`prediction_times_per_subject` defaults to `1`, so rows and subjects coincide at the defaults; raise
that knob and a subject contributes several correlated rows to the same cell, which a row bootstrap
treats as independent, so the interval is somewhat narrower than a subject-level one would be.
`n_subjects` sits next to `n_rows` so that case is visible.

A resample can land single-class, where AUROC is undefined; those replicates are `nan` and the bounds
are read with `np.nanpercentile`. The count is reported per cell as `n_degenerate_replicates`, so an
interval resting on a handful of usable replicates is visible rather than silently wide.

______________________________________________________________________

## 5. Results

No numbers yet. The earlier measurements in this document's predecessor were produced by the
conditional query-sequence model that this repository no longer ships, and reporting them here would
attribute another model's behaviour to this one. Multitask numbers — per-task AUROCs over the
evaluation grid, with the intervals described in §4 — will be added once
`EQ_predict_multitask` → `EQ_evaluate_multitask` has been run on a trained checkpoint.

______________________________________________________________________

## 6. Reproducing a run

The [README](../README.md) has the seven-step walkthrough end to end, with every knob and its
default. Two notes that belong here rather than there:

- **Run the samplers and the grid with the same knobs.** The evaluation grid draws its horizons,
    event bounds and window starts from the same distributions the training sampler does, so a
    checkpoint trained with overrides needs the same overrides passed to the grid generator. Drift
    between the two does not raise: it silently puts the grid out of distribution and reads as an
    unexplained metric shift, which is the hardest kind of wrong answer to notice.
- **Check which checkout you are importing.** The venv is shared across worktrees and its editable
    install names one absolute path, so an ad-hoc script launched from a worktree can silently import
    the main checkout's code. `pyproject.toml`'s `pythonpath = ["src"]` fixes this for `pytest` and
    only for `pytest`. [`CONTRIBUTING.md`](../CONTRIBUTING.md) has the guard to copy into any
    measurement driver; this has already cost one real result.
