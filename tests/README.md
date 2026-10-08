# Tests

## Running them

```bash
uv run pytest tests/ -q
```

Heavy integration tests that train a real model are marked `slow` and skipped by default; add them
with `-m "slow or not slow"`. `pyproject.toml` also turns on `--doctest-modules`, so docstring
examples under `src/` run as part of the suite.

## Layout

- `tests/sampler/` — the query-sequence sampler, split by pipeline stage (prediction times, query
    distribution, patient contexts, index build, labelling), plus the orchestration that chains them
    and the per-shard evaluation grid.
- `tests/multitask/` — the multitask pipeline: datamodule, lightning module, orchestration,
    multi-boundary labelling, prevalence weighting, interval table, predict-side logic.
- `tests/ontology_suite/` — not tests, but the fixtures the ontology tests are built from: an
    independent `oracle.py`, a hand-computed `golden.py` truth table, and a `production.py` adapter.
    Driven by `test_ontology_golden.py` and `test_ontology_differential.py`.
- `tests/test_*.py` — everything else: CLI smoke tests, per-module logic, and the feature tests
    described below.

## Why the feature tests look the way they do

The tests covering time representation (RoPE with delta-token stripping), event-bounded queries and
ontology/hierarchical embeddings are deliberately unlike the rest of the suite. The reason is worth
keeping, because the shape of these tests is not an accident and flattening them back into ordinary
assertions would quietly remove the coverage that matters.

**"The tests pass" was not enough here.** These features landed with dozens of new test functions and
a green suite. A run against real data then found bugs anyway — and most of them produced *silently
wrong labels* rather than errors. The cause is structural: the same author wrote the code and the
tests. A green test proves the code matches the author's belief about the spec. It says nothing about
whether the belief is right. Where the belief was wrong, the test encoded the same wrong belief and
passed.

A test can also pass for a reason unrelated to what it claims to check. A bitwise
`assert not torch.equal(a, b)` is satisfied by **one ULP of float32 rounding** — about 1.19e-07 on
these tensors — so a "the feature changes the output" test written that way is green from the moment
it exists, whether or not the feature does anything. That was observed directly. Nothing here asserts
liveness with a bare inequality.

So these tests are built out of the two check kinds the author's own assertions cannot be.

### 1. Differential tests against an independent oracle

`tests/test_rope_strip_oracle.py`, `tests/test_event_bounds_oracle.py`,
`tests/ontology_suite/` (via `test_ontology_golden.py` and `test_ontology_differential.py`).

Each writes a second implementation from the *prose spec*, in plain Python loops — no polars, no
torch, no shared helper with the optimised code — and compares every output over randomised inputs.
An independent implementation cannot share an implementation bug, so a disagreement localises a real
defect in one of the two. The oracles were written from the spec transcribed into the test file
*before* the production body was read, and the transcription is kept in the file so a later reader
can check the spec rather than the code.

This is worth most where the optimised code is hardest to read, and each oracle guards a specific
silently-wrong-label failure:

- **Delta-token stripping** compacts four parallel tensors at once, re-bases each row's clock to its
    first surviving token, and *recomputes* rather than compacts the time deltas. A misalignment
    between any two of those outputs would not raise — it would hand the encoder a stream whose values
    belong to different tokens than its codes, corrupting every training sequence while every shape
    assertion still passed.
- **Event-bounded labelling** turns on window edges that are easy to get wrong and impossible to see
    wrong: the window is open at both ends, the boundary is the *first* occurrence strictly after the
    prediction time, and a target sharing a timestamp with the boundary does not count. That last rule
    is load-bearing on MEDS data, where many codes cluster on one instant — a discharge and everything
    charted with it routinely share a timestamp.
- **The ontology closure** is where the worst of these lived. A leaf code that is a strict `//`-prefix
    of another code was receiving closure rows from its descendants, so every ordinary leaf query
    naming such a code silently changed meaning from "this exact code occurred" to "this code *or any
    descendant* occurred", flipping labels False→True — no crash, no warning, a well-formed parquet of
    wrong labels. Separately, declared parent edges were followed exactly one hop and never
    transitively closed, so ancestor queries were labelled False for descendants more than one hop
    away, silently truncating the multi-level DAG the feature exists to express.

A suite that cannot fail is not evidence, so the oracles were red-proofed against the genuine
pre-fix modules and confirmed to disagree.

### 2. Liveness probes

`tests/test_feature_liveness.py`, `tests/test_feature_composition.py`, and the model-side half of
`tests/test_ontology_embedding.py`.

A feature can be plumbed through collate, reach the forward pass, and then be multiplied by zero. It
passes every shape, dtype and "runs without error" assertion while contributing nothing. Ordinary
tests cannot detect that, because they assert the model *runs* with the new tensors, not that it
*responds* to them. Three probes a dead feature cannot pass:

- **Gradient** — each new parameter receives a non-zero gradient from a batch that exercises it.
- **Sensitivity** — perturbing one new input field alone moves the output, by a margin well above
    float noise (`LIVE = 1e-6`, against ~1e-7 rounding and ~1e-4 real effects).
- **Atom invariance** — a batch using none of the new machinery is **bit-identical** with and without
    the new tensors attached, at exactly `0.0`.

Atom invariance is the one every reported number rests on. An atomic evaluation grid is entirely
time-bounded single-code queries, so if merely attaching the feature machinery perturbed them, every
score would describe a different model than the one trained.

One measurement rule, learned from a false negative: **assert at the level where the effect lives**,
not downstream of an untrained head. A randomly-initialised readout compresses an 8e-05 difference
in the backbone's representation down to ~1e-07 at the logits — the same magnitude as float noise.
The RoPE probe therefore reads the tensor the readout projects, directly: `window_hidden_states` on
`ConditionalMultitaskARModel`.

There is also a guard, `tests/test_rope_strip_guard.py`, for a configuration that is silently
acceptable rather than wrong-by-construction: stripping delta tokens without the time representation
that replaces them yields a backbone with *zero* elapsed-time information, and it trains, validates
and checkpoints with normal-looking numbers.

### Rehoming

These tests drive the features through whichever conditional model existed when they were written,
and are being moved onto the surviving multitask model as the older ones are retired. The move is
mechanical on purpose: every property above is a property of the *feature* — the time representation,
the boundary machinery, the ontology embedding — not of any one model class, so only the model
construction changes. Keep the oracle / liveness / invariance structure intact; a rewrite that
replaces a probe with a plain "it runs" assertion is a coverage loss even when the suite stays green.

### Before acting on a finding, size it

The last habit worth carrying: a confirmed bug is not yet an actionable one. Compute how much it
actually touches — how many codes, how many rows, how many scored evaluation cells — before deciding
what it invalidates. Prefix absorption above affected 2.87% of leaf codes but only 2 of 100 scored
codes, which is what established that the published comparison survived it.
