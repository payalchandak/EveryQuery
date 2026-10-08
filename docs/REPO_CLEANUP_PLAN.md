# Repo cleanup plan — trimming to two pipelines before the upstream PR

**Status:** approved plan, **not started — blocked on PR #33.** Written 2026-09-08 against `dev`
at `d8ba8f8` (PR #32 merged; PR #33 `feat/multitask-ontology-boundaries` open). The five open
questions were answered the same day and are recorded, with their consequences, in §7.

Nothing in this plan begins until #33 merges into `dev`. That is not just sequencing hygiene: #33
touches `sample_multitask_sequences.py`, the ontology plumbing and the multitask tests, which are
exactly the files §3 and §4 rearrange. Starting the trim first would put a rename-and-delete branch
against an open feature branch's diff and make #33 painful to review or rebase. Every step below
assumes `dev` already contains #33.

**Goal.** Land PR #33, then reduce `mmcdermott/EveryQuery-conditional` to exactly two pipelines and
open a PR against `payalchandak/EveryQuery`:

1. **EQ-single** — the original single-code / single-duration EveryQuery model. This is
    *verbatim upstream*: `upstream/main` contains it and nothing else.
2. **EQ-multitask** — `ConditionalMultitaskARModel`: the decoder-only, all-vocabulary,
    multi-boundary model with ontology-derived targets.

Everything else goes. The bulk of "everything else" is one coherent thing: the **conditional
query-sequence pipeline** (`ConditionalQueryEncoderDecoderModel` and its decoder-only sibling
`ConditionalQueryARModel`), plus the ~4.5k lines of one-off analysis scripts and three report PDFs
that exist only to measure it.

______________________________________________________________________

## 1. The headline number

Tracked Python in the fork today: **49,712 lines**.

| Bucket                                                                        |         Lines | Confidence            |
| ----------------------------------------------------------------------------- | ------------: | --------------------- |
| Straight deletions — src modules that only the conditional-seq pipeline uses  |           897 | high                  |
| Straight deletions — one-off analysis / report scripts                        |         4,453 | high                  |
| Straight deletions — tests that only exercise the conditional-seq pipeline    |         2,512 | high                  |
| Partial deletions inside shared modules (see §3)                              |          ~700 | medium                |
| Tests that must be **rehomed onto the multitask model**, not deleted (see §4) |         1,876 | this is the real work |
| Docs / notes                                                                  |         1,146 | high                  |
| Tracked PDFs                                                                  | 1.2 MB binary | high                  |

Roughly **8.5k lines of Python deleted outright**, ~1.9k lines of feature tests rewritten, ~1.2 MB
of binaries dropped from git.

______________________________________________________________________

## 2. What survives, in full

### 2.1 EQ-single (identical to `upstream/main`)

Do not touch these. They are what the upstream maintainer already has, and the PR diff should show
zero changes to them beyond what PRs #20–#33 legitimately added.

```
src/every_query/data/{dataset,datamodule,schema}.py, data/__init__.py, data/README.md
src/every_query/model/{model,lightning_module,task_auroc_callback}.py, model/__init__.py, README.md
src/every_query/generate_tasks/{sample_tasks,sample_evaluation_tasks,sample_task_tracking_pairs}.py
src/every_query/{train,predict,evaluate,preprocessing,utils}/  — minus the files listed in §3
src/every_query/predict/external_tasks/**
CLIs: EQ_process_data, EQ_generate_training_tasks, EQ_generate_evaluation_tasks,
      EQ_sample_task_tracking_pairs, EQ_train, EQ_predict, EQ_evaluate
```

Note `data/schema.py` has grown `QuerySeqSchema` and `MultitaskBoundarySchema` on top of upstream's
`TaskQuerySchema` — both are needed by EQ-multitask, so the file stays extended.

### 2.2 EQ-multitask

```
src/every_query/data/multitask_dataset.py                      676
src/every_query/data/multitask_eval_dataset.py                 369
src/every_query/data/conditional_multitask_datamodule.py       293
src/every_query/data/ontology.py                               832
src/every_query/data/build_ontology.py + configs/              108
src/every_query/data/rope_time.py                              224
src/every_query/model/conditional_multitask_ar_model.py        538
src/every_query/model/conditional_multitask_lightning.py       185
src/every_query/model/ontology_embedding.py                    149
src/every_query/generate_tasks/sample_multitask_sequences.py  1764
src/every_query/generate_tasks/interval_table.py               647
src/every_query/generate_tasks/sample_evaluation_query_sequences.py  1562
src/every_query/predict/predict_multitask.py                   681
configs: conditional_multitask_ar_config.yaml, _demo_train_conditional_multitask_ar.yaml,
         predict_multitask.yaml, sample_multitask_sequences_config.yaml,
         sample_evaluation_query_sequences_config.yaml, build_ontology.yaml
CLIs: EQ_build_ontology, EQ_generate_multitask_sequences,
      EQ_generate_evaluation_query_sequences, EQ_predict_multitask
```

`sample_evaluation_query_sequences.py` is worth calling out: its name says "query sequences", but
it is **the** evaluation-grid generator for the multitask model — `EQ_predict_multitask` reads
exactly its output, and only it can emit the explicit window starts the multitask model consumes.
It stays.

______________________________________________________________________

## 3. The entanglement — this is where the actual work is

The multitask pipeline was built on top of the conditional-seq pipeline, so four surviving modules
import from four dying ones. You cannot just `git rm` the conditional-seq files; each needs a small
extraction first.

### 3.1 `model/conditional_model.py` (581 lines) — split, don't delete

| Symbol                                                                  | Used by                                           | Fate                                                   |
| ----------------------------------------------------------------------- | ------------------------------------------------- | ------------------------------------------------------ |
| `ANSWER_NO`, `ANSWER_YES`, `N_ANSWER_CLASSES`                           | `seq_dataset`, multitask model, tests             | **keep**                                               |
| `_init_aux_embeddings`, `validate_rope_time_pair`                       | `conditional_multitask_ar_model`                  | **keep**                                               |
| `TOKEN_CODE/DURATION/ANSWER`, `TOKENS_PER_QUERY`                        | `conditional_ar_model`, `train.py` size-inference | **delete** (multitask has its own `TOKENS_PER_WINDOW`) |
| `build_block_causal_mask`, `masked_bce`, `ConditionalQueryOutput`       | conditional-seq only                              | **delete**                                             |
| `ConditionalQueryEncoderDecoderModel` (+ `ConditionalQueryModel` alias) | conditional-seq only                              | **delete** (~430 lines)                                |

**Action:** move the five surviving symbols into a new `src/every_query/model/answers.py` (or fold
them into `model/__init__.py`), repoint `seq_dataset` and `conditional_multitask_ar_model`, then
delete `conditional_model.py` entirely. Deleting the file rather than gutting it in place keeps the
upstream diff honest — a file called `conditional_model.py` holding five constants would confuse a
reviewer.

### 3.2 `data/seq_dataset.py` (628 lines) — keep, rename

`QuerySeqMultitaskEvalDataset` **subclasses** `ConditionalQueryPytorchDataset`, and
`multitask_dataset` / `predict_multitask` / `sample_multitask_sequences` all pull its column-name and
sentinel constants (`EVENT_BOUND_DURATION_SENTINEL`, `NO_BOUND_INDEX`, `ALL_SEQ_LABEL_COLS`, …).

Only `ConditionalQueryBatch` (the conditional-seq collate output, lines 91–181) is dead once
`conditional_lightning` goes.

**Action:** delete `ConditionalQueryBatch`, keep the rest, and rename (decision D3): module
`seq_dataset.py` → `query_seq_dataset.py`, class `ConditionalQueryPytorchDataset` →
`QuerySeqPytorchDataset`. The name is misleading once its only consumer is the multitask eval path.

The class rename is cheaper than it looks: all four configs that name it in a `data_class` field
(`conditional_config.yaml`, `conditional_ar_config.yaml`, and the two `_demo_train_conditional*`)
are themselves on the delete list, so **no surviving YAML references the class**. It is a
Python-only rename.

### 3.3 `generate_tasks/sample_query_sequences.py` (1,398 lines) — demote to a library

Two survivors import from it:

- `sample_evaluation_query_sequences` → `QuerySequenceDistribution`, `build_query_universe`,
    `label_query_sequences`, `maybe_expand_to_matching_query_nodes`, `label_with_event_bounds`,
    `label_with_explicit_starts`, and the `_ctx_id` / `_position` / bound / start column names.
- `sample_multitask_sequences` → `resolve_prediction_times`.

Dead once `EQ_generate_query_sequences` goes: `label_one_sequence_shard`, `_label_sequence_shards`,
`_validate_sequence_count`, `label_sequence_shards`, `run`, `main`, and (pending a check)
`_expand_sequences`, `_attach_queries_to_contexts`, `build_sequence_index` — roughly lines 1150–1398
plus a few above, ~300 lines.

**Action:** drop the `EQ_generate_query_sequences` entry point, delete the shard-orchestration and
Hydra `run`/`main` tail, rename the module to `query_sequence_labeling.py` (decision D3), and delete
its config `sample_query_sequences_config.yaml`.

> ⚠️ `build_sequence_index` is currently only imported by `tests/test_conditional_queries.py` (which
> is on the delete list) — but `sample_evaluation_query_sequences` has two comments asserting its
> output is *shaped exactly like* `build_sequence_index`'s. Confirm it is genuinely unreferenced
> before removing, and move those comments' invariant into the eval-grid module.

### 3.4 `train/train.py` and `utils/model_loader.py` — small edits

- `train.py:45–71` — the model-size-from-data inference branches on `ConditionalQueryARModel` and
    imports `TOKENS_PER_QUERY` from `conditional_model`. Delete that branch and its doctest; keep the
    `ConditionalMultitaskARModel` branch above it.
- `model_loader.py:43–44` — a docstring reference to `ConditionalQueryLightningModule`. One-line fix.

______________________________________________________________________

## 4. Tests — the one place this plan can lose real coverage

### 4.1 Delete outright (2,512 lines)

| File                                 | Lines | Why                                                                   |
| ------------------------------------ | ----: | --------------------------------------------------------------------- |
| `tests/test_conditional_ar_model.py` |   730 | tests `ConditionalQueryARModel`                                       |
| `tests/test_conditional_cli.py`      |   709 | end-to-end for `EQ_generate_query_sequences` → `EQ_predict_sequences` |
| `tests/test_conditional_queries.py`  | 1,073 | the conditional-seq sampler + lightning module                        |

### 4.2 Rehome onto `ConditionalMultitaskARModel` — do **not** delete (1,876 lines)

These are the feature tests for RoPE-time, event bounds and ontology embeddings. They are the
strongest tests in the repo — several were written specifically because a data run found *silently
wrong labels* that a green suite had missed (`docs/history/2026-08-21-three-features-verification.md`
is the post-mortem). They happen to drive those features through `ConditionalQueryModel` because
that was the only model when they were written.

| File                                | Lines | What to do                                                                                                                  |
| ----------------------------------- | ----: | --------------------------------------------------------------------------------------------------------------------------- |
| `tests/test_rope_time.py`           |   518 | swap the model under test for `ConditionalMultitaskARModel`; the sampler/dataset halves need no change                      |
| `tests/test_ontology_embedding.py`  |   426 | same — `OntologyEmbedding` / `wrap_tok_embeddings` are shared                                                               |
| `tests/test_event_bounded.py`       |   385 | most of it is labeller-level and survives as-is; only the model-forward assertions move                                     |
| `tests/test_ontology.py`            |   345 | mostly `data/ontology.py` — only the `ConditionalQueryModel` liveness checks move                                           |
| `tests/test_feature_composition.py` |   202 | all three features at once; must move wholesale                                                                             |
| `tests/test_feature_liveness.py`    |   151 | pure model-liveness; move wholesale, or delete if `tests/test_conditional_multitask_ar_model.py` already covers each toggle |

**Budget this honestly.** It is a day of work, not an afternoon, and it is the step where the trim
can quietly reduce the quality of the thing you are trying to upstream. Recommendation: do it as its
own PR, *before* the deletions land, so the suite is never red and so a reviewer can see coverage
move rather than vanish.

### 4.3 Keep unchanged

`tests/multitask/**`, `tests/ontology_suite/**`, `tests/test_ontology_golden.py`,
`tests/test_ontology_differential.py`, `tests/test_multitask_*.py`,
`tests/test_conditional_multitask_{ar_model,cli}.py`, `tests/test_queryseq_starts.py`,
`tests/test_window_bounds_contract.py`, `tests/test_event_bounds_oracle.py`,
`tests/test_rope_strip_{guard,oracle}.py`, `tests/test_eval_ontology_plumbing.py`,
`tests/sampler/**`, and every upstream test.

### 4.4 `conftest.py`

The root conftest's `seq_task_labels_dir` / `seq_dataset` / `seq_sample_batch` fixtures
(lines 393–469) serve the conditional-seq tests. Check whether the rehomed tests in §4.2 still want
them (they probably do, for the dataset half) before deleting.

______________________________________________________________________

## 5. Straight deletions — no untangling needed

### 5.1 src

```
src/every_query/model/conditional_ar_model.py          377
src/every_query/model/conditional_lightning.py         221
src/every_query/predict/predict_sequences.py           147   + configs/predict_sequences.yaml
src/every_query/evaluate/evaluate_sequences.py         152   + configs/evaluate_sequences.yaml
src/every_query/train/configs/conditional_config.yaml
src/every_query/train/configs/conditional_ar_config.yaml
src/every_query/train/configs/_demo_train_conditional.yaml
src/every_query/train/configs/_demo_train_conditional_ar.yaml
src/every_query/generate_tasks/configs/sample_query_sequences_config.yaml
CLIs removed: EQ_generate_query_sequences, EQ_predict_sequences, EQ_evaluate_sequences
```

> ⚠️ **`EQ_evaluate_sequences` is the multitask pipeline's only nearby evaluator, and it does not
> actually fit.** It consumes the per-query-position parquet from `EQ_predict_sequences` and groups
> by sequence position. `EQ_predict_multitask` writes a different schema — one row per grid row, with
> `target_code` / `label` / `prob`. Nothing in the repo consumes that today; the headline numbers
> came from the `scripts/eval_*.py` one-offs being deleted in §5.2.
>
> **So the trim leaves EQ-multitask with no evaluate step.** Resolved by D1: port
> `EQ_evaluate_multitask` before the upstream PR. `evaluate_sequences.py` is therefore *moved and
> adapted*, not deleted — it is the starting point for the new CLI, so do step 5 of §6 before
> deleting it, or delete it and recover the file from history. The design, including its grouping
> key, is in §6 "Step 5".

### 5.2 scripts (4,453 lines)

All of these were built to measure the conditional-seq model and feed the report PDFs:

```
scripts/build_report.py                686
scripts/build_report_v2.py             357
scripts/build_report_final.py          495
scripts/eval_v2.py                     443
scripts/eval_v3.py                     458
scripts/run_full_evaluation.py         546
scripts/eval_macro_position.py         309
scripts/eval_clinical.py               237
scripts/make_clinical_task_sequences.py 192
scripts/eval_occurs_uncensored.py      186
scripts/eval_position_effect.py        172
scripts/make_position_probe.py         166
scripts/eval_per_position.py           158
scripts/generate_mimic_sequences.py     48   (its own docstring says "Superseded")
```

**Keep** `scripts/bench_multitask_dataset.py` (110) — it benchmarks the surviving dataset.

`tests/test_cli_smoke.py::test_script_imports` import-checks every file under `scripts/`, so this
deletion shrinks that test's parametrisation automatically — no edit needed, but expect the test
count to drop.

`scripts/experiments/{00_build_ontology.sh,_common.sh}` — **delete** (decision D2). They are
machine-local (`_common.sh` hardcodes a `.venv` path and sources an ignored `env.sh`) and the
README's Hydra invocations already cover what `00_build_ontology.sh` does, so keeping them means
keeping two things in sync.

One thing in `_common.sh` is worth not losing: the venv/`PYTHONPATH` guard exists because a shared
venv's editable-install `.pth` names the **main checkout's** `src`, so a script run from a worktree
silently imports the wrong branch — this once made a blast-radius measurement compare a branch
against itself and report "0 rows changed". `pyproject.toml`'s `pythonpath = ["src"]` fixes this for
pytest and only for pytest. Carry that paragraph into `docs/MULTITASK.md` or a `CONTRIBUTING` note
before deleting the file.

### 5.3 Binaries and reports (1.2 MB)

```
EveryQuery_Conditional_Report.pdf         727 KB   (tracked at repo root)
EveryQuery_Conditional_Report_v2.pdf      142 KB   (tracked at repo root)
reports/EveryQuery_Conditional_Report_FINAL.pdf  338 KB
reports/README.md                                  (reproduce instructions for the deleted scripts)
reports/ontology_evaluation_audit.md               (13 KB; check for anything worth keeping first)
```

All three PDFs report on the conditional-seq `big_v2` run. They stay in git history; deleting them
from the tree is enough. Add `*.pdf` to `.gitignore`.

Also untracked-but-present and worth sweeping: `build_ontology.log`, `tea_debug.log`,
`node_modules/`, `outputs/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`. `.gitignore` already
covers `outputs/` and `*.log`; it does **not** cover `node_modules/` or `*.pdf` — add both.

### 5.4 docs (1,146 lines)

| File                                                         | Lines | Fate                                                                                                                                                                                                                                                                                                                                                                                             |
| ------------------------------------------------------------ | ----: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `docs/COHORT_INFERENCE_NOTES.md`                             |   376 | **delete** — self-described "reference notes, not a plan", dated 2026-07-24, about an archived checkpoint at a path that no longer matters                                                                                                                                                                                                                                                       |
| `docs/history/2026-08-18-conditional-v2-integration-plan.md` |   318 | **delete** — port plan, fully executed                                                                                                                                                                                                                                                                                                                                                           |
| `docs/history/2026-08-21-ontology-handoff.md`                |   278 | **delete** — session handoff, resolved                                                                                                                                                                                                                                                                                                                                                           |
| `docs/history/2026-08-21-three-features-verification.md`     |   174 | **keep, or salvage** — this is the "why the tests look like that" document. Fold its argument into a `tests/README.md` before deleting; do not lose it                                                                                                                                                                                                                                           |
| `docs/CONDITIONAL_QUERIES.md`                                |   280 | **rewrite** as `docs/MULTITASK.md` — the §§ on censoring-as-a-query, event bounds, ontology queries and the macro-vs-pooled AUROC argument all still describe EQ-multitask. Per D4, **drop the results sections**: they quote the conditional-seq `big_v2` run, and the rewritten doc should carry no numbers the surviving model did not produce. Add them back once PR #33's model is measured |
| `src/every_query/generate_tasks/redesign-spec.md`            |     — | **keep** — exists upstream                                                                                                                                                                                                                                                                                                                                                                       |

Git history keeps every deleted doc; the `docs/history/` files exist to brief a cold session, and
that job is done.

### 5.5 README

The README is currently *entirely* about the conditional-seq pipeline: its 7-step walkthrough is
`EQ_generate_query_sequences` → `EQ_predict_sequences` → `EQ_evaluate_sequences`, and its mermaid
diagram names all three. It needs a rewrite, not an edit — the walkthrough becomes
`EQ_build_ontology` → `EQ_generate_multitask_sequences` →
`EQ_generate_evaluation_query_sequences` → `EQ_train --config-name=conditional_multitask_ar_config`
→ `EQ_predict_multitask`, with a short section pointing at the upstream single-query CLIs rather
than the current dismissive one-liner ("still ships in the tree but is not covered here").

The badge at the top already points at `payalchandak/EveryQuery` Actions, which is right for the
upstream PR.

### 5.6 Branches

24 local branches, most merged or dead. Not part of the upstream PR, but worth pruning in the same
sweep: `git branch --merged dev` first, and keep `main`, `dev`, and #33's branch.

______________________________________________________________________

## 6. Suggested sequencing

Six PRs into `dev`, then one PR to upstream. Each step leaves the suite green.

| #   | PR                                                                                                                                                                                        | Risk                   |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- |
| 0   | Land #33                                                                                                                                                                                  | —                      |
| 1   | **Rehome the feature tests** (§4.2) onto `ConditionalMultitaskARModel`. Adds tests, deletes nothing.                                                                                      | medium — the real work |
| 2   | **Delete the analysis scripts, PDFs, reports/, stale docs** (§5.2–5.4). Pure removal, no code touched.                                                                                    | trivial                |
| 3   | **Extract the shared symbols** (§3.1–3.3): new `model/answers.py`, trim `seq_dataset`, demote `sample_query_sequences` to a library. No deletions yet — both pipelines still import fine. | low                    |
| 4   | **Delete the conditional-seq pipeline** (§5.1 + §4.1) and fix `train.py` / `model_loader.py`.                                                                                             | low, after 1 & 3       |
| 5   | **Port `EQ_evaluate_multitask`** (D1) — group by the query *spec*, macro-average AUROC over task cells. Design below.                                                                     | medium — new code      |
| 6   | **Rewrite README + `docs/MULTITASK.md`** (§5.5, §5.4), results sections omitted.                                                                                                          | low                    |
| 7   | **Stacked PR series to `payalchandak/EveryQuery`** (D5).                                                                                                                                  | —                      |

Step 0 is a hard gate, not a formality — see the status note at the top.

Doing 1 before 4 is the point of the ordering: it is the only step that can silently cost coverage.

### Step 5 — `EQ_evaluate_multitask`

Ordering constraint first: `evaluate_sequences.py` is the template, so either write this before step
4 deletes it, or accept recovering the file from history.

**Input.** The `EQ_predict_multitask` parquet, one row per grid row:

```
subject_id, prediction_time,
queries, start_durations, start_events, durations, bound_events, answers,   # list columns
target_code, label, prob
```

**Grouping key — the query specification, five list columns:**

```python
TASK_KEY = ["queries", "durations", "start_durations", "start_events", "bound_events"]
```

That *is* the task. `sample_evaluation_query_sequences` resolves `N` `SequenceSpec`s once
(`num_evaluation_sequences`, default 64) and labels **every one at every context**, so grouping on
the spec recovers exactly those `N` cells, each populated by the whole cohort — its own docstring
puts it as "for a given sequence the only thing varying across its rows is the patient, which is
what per-sequence metrics need." Pooling instead measures cross-query base-rate separation.

This supersedes the `(target_code, duration_bucket)` key floated in the first draft of this plan.
Bucketing lumps distinct horizons, and at `K > 1` it pools rows whose conditioning contexts differ,
which is the axis most worth separating.

**`answers[:-1]` is deliberately not in the key.** The conditioning answers vary per context, so
adding them would split each spec cell into up to `2^(K-1)` sub-buckets, skewed hard toward
all-False (most codes are rare) — and AUROC is undefined on a single-class cell, so most of the
partition would come back null. The score is `prob`; the class label is `label`, i.e. `answers[-1]`,
the final query's answer. Prior answers are context, not identity, and not the label.

**Output** — `<metrics_stem>.by_task.parquet`, one row per spec:

| column                               |                                                                                  |
| ------------------------------------ | -------------------------------------------------------------------------------- |
| the five `TASK_KEY` columns          | the spec itself                                                                  |
| `target_code`                        | `queries[-1]`, for readability                                                   |
| `n_queries`                          | `len(queries)`; `1` means no conditioning                                        |
| `duration_bucket`                    | bucket of the final query's horizon, descriptive only — for rollups, never a key |
| `n_rows`, `n_positive`, `prevalence` | over the cell                                                                    |
| `n_subjects`                         | distinct `subject_id` in the cell — the bootstrap's resampling unit              |
| `auroc`                              | within-cell, null when single-class (`_auroc_or_none`)                           |
| `auroc_ci_lo`, `auroc_ci_hi`         | 95% subject-cluster bootstrap, null wherever `auroc` is                          |

**Headline** = mean of the non-null `auroc` over cells, reported alongside `n_tasks_scored` and
`n_tasks_null` so a macro over 12 of 64 cells cannot pass as a macro over 64.

**Summary** — `<metrics_stem>.summary.parquet`, one row: `macro_auroc`, then three CI pairs —
`macro_auroc_ci_{lo,hi}_tasks`, `_subjects`, `_nested` — plus `n_tasks_scored`, `n_tasks_null`,
`n_resamples`, `bootstrap_seed`. All three intervals are emitted rather than selected by a flag: they
answer different questions, they come out of the same computation (below), and naming the axis in the
column means an interval read six months later carries its own definition.

#### Bootstrap 95% CIs

**Always computed, never opt-in.** Both levels — the per-cell interval on every row of
`by_task.parquet` and all three macro intervals in `summary.parquet` — are produced on every run.
There is no `--bootstrap` flag and no mode selector: a point AUROC without an interval invites being
quoted as if it were precise, and the intervals cost one pass that this design is already paying for
(see "One bootstrap pass" below). The only knob is `n_resamples` (default `1000`), which trades
runtime for interval resolution; it is not a way to switch the intervals off.

Match the convention on `upstream/task-auroc-ci`, which adds exactly this to
`task_auroc_callback.py` — `scipy.stats.bootstrap`, `n_resamples=1000`, `confidence_level=0.95`,
`method="percentile"`, `rng=np.random.default_rng(0)`, columns suffixed `_ci_lo` / `_ci_hi`, and the
task count logged next to the estimate. Adopting it verbatim means the evaluator's intervals and the
training-time callback's intervals mean the same thing and are named the same way.

> ⚠️ That branch is **not merged into `upstream/main`**. It also adds `scipy>=1.15,<2` as a direct
> dependency (1.15 is where `bootstrap` takes `rng=` rather than `random_state=`). If it lands
> before our U4, we inherit both; if it does not, U4 introduces scipy itself. Check before writing
> the import, and say so in the PR description either way.

**The resampling unit is the subject, not the row.** `prediction_times_per_subject` defaults to `1`,
so rows and subjects coincide at the defaults — but the evaluator must not assume it. Raise that knob
and a subject contributes several correlated rows to the same cell; resampling rows would then
understate the spread and quietly narrow every interval. Resample distinct `subject_id`s and take all
their rows (`eval_per_position.py:92` does the same thing one level up, resampling whole sequences so
positions stay correlated within one).

**One bootstrap pass produces every interval.** Per replicate `b`:

1. Draw a subject index `S_b` — `n_subjects` distinct `subject_id`s with replacement — **shared
    across all cells**, not redrawn per cell.
2. For each cell `c`, recompute `AUROC[c, b]` over the rows of `c` whose subject is in `S_b`.

That single `n_cells x n_resamples` grid of AUROCs yields all four numbers:

| interval                          | read off the grid as                                                                                     | resampling axis        | the question it answers                                                      |
| --------------------------------- | -------------------------------------------------------------------------------------------------------- | ---------------------- | ---------------------------------------------------------------------------- |
| `auroc_ci_{lo,hi}` (per cell `c`) | percentiles of `AUROC[c, :]`                                                                             | subjects               | would *this task's* AUROC hold on a different draw of patients?              |
| `macro_auroc_ci_*_subjects`       | percentiles of `mean_c AUROC[:, b]`                                                                      | subjects               | would the macro hold on a different cohort, holding the query specs fixed?   |
| `macro_auroc_ci_*_nested`         | within each `b`, resample cells with replacement from `AUROC[:, b]`, take the mean; percentiles over `b` | subjects **and** tasks | would the macro hold on a different cohort *and* a different draw of specs?  |
| `macro_auroc_ci_*_tasks`          | resample the *point* AUROCs with replacement, take the mean                                              | tasks                  | would the macro hold on a different draw of specs, holding the cohort fixed? |

The sharing in step 1 is what makes the macro rows valid: the grid is dense, so every cell holds the
**same** subjects and the cells are correlated through them. Redrawing per cell would destroy that
correlation and narrow the macro intervals. It costs the per-cell rows nothing — each cell's marginal
distribution `AUROC[c, :]` is still an ordinary subject bootstrap of that cell.

**Quote `_nested` as the headline uncertainty.** It is the only one of the three that resamples both
axes; `_tasks` treats each cell's AUROC as exact and sees only between-task spread, `_subjects`
conditions on the fixed `N` specs and sees only patient noise. Keep `_tasks` anyway — it is what
`upstream/task-auroc-ci`'s callback computes, so it is the number the training-time logs are directly
comparable to.

> **Correction to an earlier draft of this plan**, which called the nested interval "roughly
> `n_tasks` times the cost, which is why it is not the default". That compared it against the
> *`tasks`* bootstrap, which is a mean over 64 floats and essentially free. Against the per-cell
> bootstrap — which this design already pays for — the nested interval is **free**: step 2 is the
> same `n_cells x n_resamples` AUROC grid either way, and `_nested` is a mean over numbers already in
> it. There is no cost argument for omitting it.

**Degenerate replicates.** A resample can land single-class, where AUROC is undefined; use `np.nan`
for that cell-replicate and `np.nanpercentile` for the bounds (the pattern already in
`eval_per_position.py:132`), and record how often it happens so a cell whose interval rests on a
handful of usable replicates is visible rather than silently wide.

**Grid invariant worth asserting.** The macro rows assume every cell holds the same subject set —
true by construction for a dense grid, since each spec is labelled at every context. Check it rather
than trust it: a malformed or partially-written grid would otherwise produce a shared index that
intersects different cells to different degrees, and the macro intervals would be quietly wrong
rather than absent.

**Cost.** The whole thing is one `n_cells x n_resamples` grid of AUROC evaluations, each
`O(n log n)` — 64 cells at 1000 resamples is 64k `roc_auc_score` calls, a minute or so at 10k rows
per cell and closer to ten at 100k. Every interval above is a percentile over that one grid, so the
cost is set entirely by wanting per-cell CIs at all; the macro variants add nothing measurable. Keep
`n_resamples` a config knob. If the grid becomes the bottleneck, the fallback is DeLong's analytic
AUROC variance for the per-cell intervals (one `O(n log n)` pass, no resampling) — but note that
dropping the grid also drops `_subjects` and `_nested`, leaving only the `_tasks` interval.

#### Defensive test for the list-column `group_by`

Group directly on the list columns rather than hashing them to a `task_id` — it keeps the spec
legible in the output — but pin the behaviour, since `pyproject` allows `polars>=1.35,<2` and this
is not a heavily exercised polars path. Verified working on **polars 1.40.0** (2026-09-08): distinct
specs partition correctly, nulls inside lists do not collapse distinct specs, and the `-1.0`
sentinel separates from real horizons. The test guards the rest of the 1.x range.

1. **No collision.** Same codes with different `durations` → two cells; same codes with one position
    event-bounded (`durations` `-1.0` + a `bound_events` code) → two cells.
2. **Nulls inside lists are significant.** `bound_events` `[None, None]` and `[None, "DISCHARGE"]`
    must not group together.
3. **Sentinel floats.** `-1.0` in `durations` / `start_durations` groups by exact equality and never
    merges with a real horizon.
4. **Round-trip against the grid** — the strongest of the five, since it catches collision *and*
    fragmentation in one assertion: cell count equals the number of distinct `SequenceSpec`s in the
    input grid, and `sum(n_rows)` equals the input height (nothing dropped, nothing double-counted).
5. **Order independence.** Shuffling the input rows yields identical cells and identical metrics.

If a future polars breaks any of these, the fallback is a derived `task_id` —
`pl.struct(TASK_KEY).hash()`, or `utils/digest.py` — which reconstitutes the identifier
`predict_multitask` deliberately dropped, in the place it belongs.

### Step 7 — the stacked series

Per D5, four PRs against `payalchandak/EveryQuery:main` rather than one, each depending on the last:

| PR  | Contents                                                                                                                                                                                                                             |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| U1  | **Ontology** — `data/ontology.py`, `data/build_ontology.py`, `model/ontology_embedding.py`, `EQ_build_ontology`, `tests/ontology_suite/**`, `test_ontology*.py`. Self-contained and useful on its own; the natural first ask.        |
| U2  | **Multitask sampler** — `sample_multitask_sequences.py`, `interval_table.py`, `query_sequence_labeling.py`, `MultitaskBoundarySchema` / `QuerySeqSchema`, `EQ_generate_multitask_sequences`, the sampler half of `tests/multitask/`. |
| U3  | **Multitask model** — `conditional_multitask_ar_model.py`, `conditional_multitask_lightning.py`, the datamodule and datasets, `rope_time.py`, the train config, and the rehomed feature tests from §4.2.                             |
| U4  | **Evaluation** — `sample_evaluation_query_sequences.py`, `predict_multitask.py`, `EQ_evaluate_multitask` (+ `scipy` if `upstream/task-auroc-ci` has not landed), `docs/MULTITASK.md`, README rewrite.                                |

Confirm upstream will take a chain before splitting; if they would rather have one PR, U1–U4
collapse without rework, since the ordering is already dependency-clean.

______________________________________________________________________

## 7. Decisions — resolved 2026-09-08

|        | Question                        | Answer                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ------ | ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **D1** | The missing multitask evaluator | **Port `EQ_evaluate_multitask`.** Adapt `evaluate_sequences.py` to the one-row-per-grid-row schema; group by the query spec `(queries, durations, start_durations, start_events, bound_events)`, excluding `answers[:-1]`; macro-average AUROC over task cells; emit 95% bootstrap CIs both **per cell** and on the **macro** (three macro variants — tasks, subjects, nested — all falling out of one shared-subject-index bootstrap pass) following the `upstream/task-auroc-ci` convention; guard the list-column `group_by` with a defensive test. Shipping an inference CLI with no evaluator was judged the wrong look on an upstream PR. Full design in §6 "Step 5". |
| **D2** | `scripts/experiments/`          | **Delete entirely.** Machine-local and redundant with the README's Hydra invocations. Salvage the venv/`PYTHONPATH` warning into prose first (§5.2).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| **D3** | Vestigial names                 | **Rename both.** `seq_dataset.py` → `query_seq_dataset.py` (class → `QuerySeqPytorchDataset`), `sample_query_sequences.py` → `query_sequence_labeling.py`. Pure renames show as `R100`; no surviving YAML names the class (§3.2).                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| **D4** | `docs/CONDITIONAL_QUERIES.md`   | **Rewrite as `docs/MULTITASK.md`, drop the results sections.** They quote the conditional-seq `big_v2` run; add multitask numbers once PR #33's model is measured (§5.4).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| **D5** | Upstream PR shape               | **Stacked series** — ontology → sampler → model → eval, four PRs against `payalchandak/EveryQuery:main`. Confirm upstream accepts a chain first (§6, step 7).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |

No decisions remain open. One thing to watch that is not a decision: D1's new CLI is the only step
that adds code rather than removing it, so it is the one most likely to slip.

## 8. Verification for each step

```bash
# after every deletion PR
uv run pytest tests/ -x -q # full suite
uv run python -c "import every_query, pkgutil, importlib; \
    [importlib.import_module(m.name) for m in pkgutil.walk_packages(every_query.__path__, 'every_query.')]"
uv run pre-commit run --all-files
for cli in EQ_process_data EQ_train EQ_predict EQ_evaluate EQ_build_ontology \
	EQ_generate_multitask_sequences EQ_generate_evaluation_query_sequences \
	EQ_predict_multitask; do "$cli" --help >/dev/null || echo "BROKEN: $cli"; done
git grep -nE 'conditional_model|conditional_ar_model|conditional_lightning|predict_sequences|evaluate_sequences'
```

The last `git grep` should return nothing outside the multitask model's own `answers.py` import
after step 4. `tests/test_cli_smoke.py` enumerates the entry points and will fail loudly on a
`pyproject.toml` script that no longer resolves — that is the cheapest guard against a half-removed
CLI.
