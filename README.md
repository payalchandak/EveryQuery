# EveryQuery

[![tests](https://github.com/payalchandak/EveryQuery/actions/workflows/tests.yaml/badge.svg?branch=dev)](https://github.com/payalchandak/EveryQuery/actions/workflows/tests.yaml)
[![codecov](https://codecov.io/gh/payalchandak/EveryQuery/branch/dev/graph/badge.svg)](https://codecov.io/gh/payalchandak/EveryQuery)

A framework for training and evaluating foundation models over structured EHR data, built on
the [MEDS](https://github.com/Medical-Event-Data-Standard) ecosystem —
[`meds-torch-data`](https://github.com/mmcdermott/meds-torch-data) for tensorization,
[`MEDS-transforms`](https://github.com/mmcdermott/MEDS_transforms) for preprocessing, PyTorch
Lightning for training.

Given a tensorized MEDS cohort, EveryQuery trains a ModernBERT-style encoder to answer
"query" prediction tasks of the form: *given a subject's history up to time `t`, will code
`c` occur within `d` days?* The same trained model is then evaluated against arbitrary
`(code, duration)` combinations.

> [!NOTE]
> A substantial refactor is in progress — see [#54](https://github.com/payalchandak/EveryQuery/issues/54).
> The pipeline is being consolidated into fewer, clearer CLIs. This README reflects the
> current state on `dev`; see [Roadmap](#roadmap) for what's changing next.

## Install

**For development** (recommended):

```bash
git clone git@github.com:payalchandak/EveryQuery.git
cd EveryQuery
uv sync --group dev
cp .env.example .env # then edit paths for your machine
```

**As a dependency:**

```bash
# not yet on PyPI — installable from git for now:
pip install "git+https://github.com/payalchandak/EveryQuery.git@main"
```

## Repository layout

Every production module lives under a submodule that reflects its role:

```
src/every_query/
├── preprocessing/      → EQ_process_data        (raw MEDS → tensorized cohort)
├── generate_tasks/     → EQ_generate_tasks      (task-label parquets for PT)
├── train/              → EQ_train               (train the model)
├── predict/            → (planned: EQ_predict)  (inference; #81, draft PR #99)
│   └── external_tasks/                         (ACES + composite aggregation)
├── evaluate/           → EQ_evaluate + 3 sibling CLIs  (#83 consolidates into one; draft PR #100)
├── model/              (shared: nn.Module + LightningModule)
├── data/               (shared: PyTorch Dataset + Batch types)
├── paper_experiments/  (research-only: ID/OOD splits, ablations, figure code)
│   └── sample_codes/   (query-code sampling for paper experiments)
└── utils/              (helpers: seeds, code slugs, env-var validation)
```

Every submodule has its own `README.md` explaining what belongs there, its pipeline
position, and the tracking issues for remaining work.

## Console scripts

`pip install` exposes the CLIs below, all Hydra-configurable. Run any with `--help` or
`--cfg job` to inspect the resolved config. The **Tests** column summarises the coverage
that lands with each CLI on `dev` today — unit tests (fast, `tests/test_<name>_logic.py`
or `tests/test_<module>.py`), CLI smoke tests (`tests/test_cli_smoke.py`, `--help`-exits-0),
and end-to-end subprocess tests that run the real script against a fixture cohort.

| Script              | Stage         | Purpose                                                                     | Tests                                                                                                                                 |
| ------------------- | ------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `EQ_process_data`   | preprocessing | Orchestrate MEDS-transforms + `meds-torch-data` tensorization               | smoke; E2E via `test_process_data.py` + `test_e2e_foundation.py`                                                                      |
| `EQ_generate_tasks` | task labels   | Sample `N` tasks × `M` contexts, label via single-pass asof (PT-ready)      | smoke; unit `test_sample_tasks.py`; E2E `test_generate_tasks.py` (#107)                                                               |
| `EQ_train`          | training      | Train the ModernBERT encoder on the labeled tasks                           | smoke; unit `test_training.py`; E2E `test_train_cli.py` + `test_train.py` (#108); signal test `tests/training_validity/` (#118, slow) |
| `EQ_gen_eval_index` | eval setup    | Sample held-out prediction times into a deterministic index                 | smoke only                                                                                                                            |
| `EQ_gen_eval_tasks` | eval setup    | Slice per-duration task matrices by `(code, duration)` using the index      | smoke; unit `test_eval_suite.py`                                                                                                      |
| `EQ_evaluate`       | eval          | Run a trained checkpoint against the sliced eval tasks, write per-code AUCs | smoke; unit `test_eval.py`; full E2E pending #109 (needs #100 landed)                                                                 |
| `EQ_select_model`   | analysis      | Rank models by pairwise win rate over `(code, duration)` pairs              | smoke only                                                                                                                            |

*Planned:* `EQ_predict` (draft PR #99, closes #81) — inference entry point that consumes
`TaskQuerySchema`-conformant rows and writes a `PredictionSchema`-conformant parquet.
Once it lands, `EQ_evaluate` consolidation (draft PR #100, closes #83) collapses the
four current eval CLIs into a single metrics-only stage that reads the predictions parquet.

## Pipeline

### Current (on `dev`)

```
           MEDS cohort  ──►  EQ_process_data  ──►  tensorized cohort ($FINAL_DATA_DIR)
                                                                     │
                                                                     ▼
pre-training:                                              EQ_generate_tasks
                                                                     │  labeled task parquets
                                                                     ▼
                                                                EQ_train  ──►  best_model.ckpt
                                                                                       │
evaluation:                     EQ_gen_eval_index  ──►  EQ_gen_eval_tasks              │
                                                                │                      │
                                                                ▼                      ▼
                                                                             EQ_evaluate
                                                                                       │  per-code AUCs
                                                                                       ▼
                                                                                EQ_select_model
```

### Planned (post Phase 2 of #54)

```
           MEDS cohort  ──►  EQ_process_data  ──►  tensorized cohort
                                                           │
                                                           ▼
                                                EQ_generate_tasks
                                                           │  TaskQuerySchema parquets
                                                           ▼
                                                     EQ_train ──► best_model.ckpt
                                                                         │
                                                                         ▼
                                                                    EQ_predict  ──►  PredictionSchema
                                                                                         │
                                                                                         ▼
                                                                                    EQ_evaluate ──►  metrics
```

### 1. Preprocess

```bash
EQ_process_data \
	input_dir="$RAW" \
	intermediate_dir="$INTERMEDIATE" \
	output_dir="$FINAL_DATA_DIR"
```

Produces a tensorized MEDS cohort under `$FINAL_DATA_DIR`. `$INTERMEDIATE` is a staging
directory for the MEDS-transforms stages; `$PROCESSED` holds cross-shard metadata
(`$PROCESSED/metadata/codes.parquet` is the query-code universe the sampler draws from).

### 2. Generate pre-training task labels

```bash
EQ_generate_tasks \
	split=train \
	input_shard=0 \
	task_shard=0 \
	n_tasks=1024 \
	contexts_per_task=1
```

Sweep across shards with
`python -m every_query.generate_tasks.sample_tasks -m input_shard=0,1,2,… task_shard=range(0,K)`.
Each worker writes labeled task parquets under `$TASK_DIR/{split}/*.parquet` idempotently.
Output columns: `subject_id, prediction_time, boolean_value, occurs, query, duration_days`.
The `query` column is the MEDS code the query asks about; `duration_days` is the prediction
horizon. These two columns will constitute the `TaskQuerySchema` being defined in
[#96](https://github.com/payalchandak/EveryQuery/pull/96) (open PR, closes #80); the extra
`occurs` column (EQ's positive-class label) currently sits alongside — collapsing it into
a single nullable label is tracked in [#122] and now in-scope of #96.

### 3. Train

```bash
EQ_train \
	output_dir="$OUTPUT_DIR/outputs/\${run_id:}" \
	datamodule.config.task_labels_dir="$TASK_DIR" \
	datamodule.config.tensorized_cohort_dir="$FINAL_DATA_DIR"
```

`EQ_train` reads the long-format labels written by `EQ_generate_tasks` directly — the
inline collation step that lived in `train.py` was removed in
[#76](https://github.com/payalchandak/EveryQuery/pull/76).

Seeding: `cfg.seed` (default `140799`) is passed through `lightning.seed_everything` *before*
model + datamodule instantiation (fix landed in [#124](https://github.com/payalchandak/EveryQuery/pull/124)),
so model weight initialization is byte-reproducible across Python versions and platforms
for a given seed.

### 4. Evaluate

```bash
EQ_gen_eval_index # sample prediction times into a deterministic eval index
EQ_gen_eval_tasks # slice per-duration task matrices by (code, duration)
EQ_evaluate model_run_dirs='["'"$OUTPUT_DIR"'/outputs/YYYY-MM-DD/HH-MM-SS"]'
EQ_select_model model_run_dirs='["..."]' split=tuning
```

`model_run_dirs` is a required override (no default) in both `EQ_evaluate` and
`EQ_select_model` as of [#126](https://github.com/payalchandak/EveryQuery/pull/126);
Hydra reports "mandatory value missing" on a fresh clone instead of failing later with
`FileNotFoundError` on a stale hardcoded path.

## Configuration

All CLIs are `@hydra.main` entry points; every config knob is overridable on the command
line with `key=value` or `+new_key=value`. The config directory is resolved via
`importlib.resources.files("every_query")`, so package-shipped YAMLs work identically
whether you run from a source checkout or a `pip install`ed wheel.

### Environment variables

`ensure_env()` (in `utils/_env.py`) requires these be set before `EQ_train` and the eval
CLIs. Scope of this gate was tightened in [#127](https://github.com/payalchandak/EveryQuery/pull/127)
— `PROCESSED` and `INTERMEDIATE` were dropped because no Hydra config interpolates them
(they were only read by a dotenv fallback in the sampler, which already tolerates missing
env vars when CLI config values are supplied).

| Var              | Purpose                                                |
| ---------------- | ------------------------------------------------------ |
| `PROJECT_DIR`    | Repo root (for relative output paths in a few configs) |
| `OUTPUT_DIR`     | Where training run dirs land                           |
| `TASK_DIR`       | Where task parquets read / write                       |
| `FINAL_DATA_DIR` | Tensorized cohort (output of `EQ_process_data`)        |
| `WANDB_ENTITY`   | W&B entity for training telemetry                      |

`.env.example` is the reference — copy to `.env` and edit. Both Python (via
`python-dotenv`) and the SLURM wrappers under `scripts/` source it. Further phases of
[#117](https://github.com/payalchandak/EveryQuery/issues/117) will migrate the remaining
gated vars to `${oc.env:VAR,???}` / `${oc.env:VAR,default}` form (Hydra-native required
or optional-with-fallback) and eventually retire `ensure_env()` entirely.

### Known gotcha: code-group YAMLs

`train/configs/config.yaml`, `evaluate/conf/eval_config.yaml`, and
`evaluate/conf/gen_tasks_config.yaml` all pull a default `{train,eval}_codes/<hash>.yaml`
that is (a) generated out-of-band and (b) explicitly `.gitignore`d — so a fresh clone
can't compose them. Workaround until [#64](https://github.com/payalchandak/EveryQuery/issues/64)
lands:

- Pass `--config-dir=/path/to/your/codes_dir code_group_name=...`, or
- Generate them locally via
    `python -m every_query.paper_experiments.sample_codes.sample_train_codes` (note: currently
    has a hardcoded MIMIC path — #85 will parameterize it).

The smoke-test fixture in `tests/test_cli_smoke.py` shows the minimal shape of each file.

## Development

```bash
uv sync --group dev
uv run pytest                         # full suite, excluding slow tests (~2 min)
uv run pytest -m 'slow or not slow'   # full suite incl. slow training-validity test (~8-10 min extra)
uv run pytest tests/test_cli_smoke.py # CLI smoke tests only
uv run pre-commit run --all-files     # lint, format, codespell
```

CI runs the full `pytest -m "slow or not slow"` (both `slow`-marked and unmarked tests)
on Python 3.11 and 3.12, plus `ruff check` and `ruff format --check` on every PR; coverage
is uploaded to Codecov. Full CI session: ~10-11 min typical.

### Test layout

```
tests/
├── test_cli_smoke.py               (all 7 EQ_* CLIs; --help exits 0)
├── test_process_data.py            (E2E: EQ_process_data output shape + metadata)
├── test_generate_tasks.py          (E2E: ground-truth label recompute + reproducibility + n_tasks differential)
├── test_sample_tasks.py            (unit: sampler primitives, determinism, edge cases)
├── test_train_cli.py               (E2E: EQ_train CLI, resume flow, overwrite flag)
├── test_train.py                   (E2E: resume-actually-loads-ckpt two-stage differential)
├── test_training.py                (unit: single training step, checkpoint roundtrip, demo-mode checks)
├── test_e2e_foundation.py          (E2E: full preprocess → generate_tasks → train pipeline chains)
├── test_eval.py                    (unit: eval.py helpers)
├── test_eval_suite.py              (unit: gen_task.py, process_eval_tasks)
├── test_dataset_logic.py           (unit: EveryQueryPytorchDataset + EveryQueryBatch)
├── test_lightning_logic.py         (unit: LightningModule loss wiring, mask semantics)
├── test_model_logic.py             (unit: model heads, censored/occurs loss flip sensitivity)
├── test_run_id.py                  (unit: run_id resolver determinism)
└── training_validity/              (E2E @pytest.mark.slow: model actually learns; see its README)
    ├── __init__.py
    ├── conftest.py
    ├── README.md
    └── test_training_validity.py
```

## Roadmap

Overall refactor umbrella: [#54](https://github.com/payalchandak/EveryQuery/issues/54) —
target architecture is `preprocess → generate_tasks → train → predict → evaluate` with a
shared cross-stage task-query schema.

### Phase 2 status

| Sub-phase                      | Issue                                                       | State                                                                                |
| ------------------------------ | ----------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| 2.1: TaskQuerySchema design    | [#80](https://github.com/payalchandak/EveryQuery/issues/80) | Draft PR [#96](https://github.com/payalchandak/EveryQuery/pull/96) (open for review) |
| 2.2: EQ_predict                | [#81](https://github.com/payalchandak/EveryQuery/issues/81) | Draft PR [#99](https://github.com/payalchandak/EveryQuery/pull/99)                   |
| 2.3: eval-suite inventory      | [#82](https://github.com/payalchandak/EveryQuery/issues/82) | Open (design)                                                                        |
| 2.4: EQ_evaluate consolidation | [#83](https://github.com/payalchandak/EveryQuery/issues/83) | Draft PR [#100](https://github.com/payalchandak/EveryQuery/pull/100)                 |

### E2E testing status ([#104](https://github.com/payalchandak/EveryQuery/issues/104))

| Subprocess test                  | Issue                                                         | State                                                                                                     |
| -------------------------------- | ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `test_process_data.py`           | (pre-104)                                                     | ✅ merged                                                                                                 |
| `test_generate_tasks.py`         | [#107](https://github.com/payalchandak/EveryQuery/issues/107) | ✅ merged via [#112](https://github.com/payalchandak/EveryQuery/pull/112)                                 |
| `test_train.py`                  | [#108](https://github.com/payalchandak/EveryQuery/issues/108) | ✅ merged via [#113](https://github.com/payalchandak/EveryQuery/pull/113)                                 |
| `test_evaluate.py`               | [#109](https://github.com/payalchandak/EveryQuery/issues/109) | Blocked on #99 + #100 landing (needs `EQ_predict` + consolidated `EQ_evaluate`)                           |
| training-validity (model learns) | [#118](https://github.com/payalchandak/EveryQuery/issues/118) | ✅ merged via [#119](https://github.com/payalchandak/EveryQuery/pull/119) — runs slow, gated by `-m slow` |

### Hygiene / follow-ups

| Issue                                                         | Description                                                                                                                     |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| [#62](https://github.com/payalchandak/EveryQuery/issues/62)   | Promote `aces_to_eq` / `process_composite` to entry points — draft PR [#95](https://github.com/payalchandak/EveryQuery/pull/95) |
| [#64](https://github.com/payalchandak/EveryQuery/issues/64)   | Drop gitignored `{train,eval}_codes` defaults (design pick pending)                                                             |
| [#85](https://github.com/payalchandak/EveryQuery/issues/85)   | Rewrite `sample_codes/` dataset-agnostic — draft PR [#97](https://github.com/payalchandak/EveryQuery/pull/97)                   |
| [#117](https://github.com/payalchandak/EveryQuery/issues/117) | Env-var audit — phase 1 merged via [#127](https://github.com/payalchandak/EveryQuery/pull/127); phases 2-4 pending              |
| [#122](https://github.com/payalchandak/EveryQuery/issues/122) | Collapse EQ sampler's `boolean_value` + `occurs` into one nullable label — in scope of #96                                      |
| [#125](https://github.com/payalchandak/EveryQuery/issues/125) | Adopt hypothesis-based property tests for the sampler                                                                           |
| [#59](https://github.com/payalchandak/EveryQuery/issues/59)   | Docs: final rewrite after the refactor settles                                                                                  |

### Model / architecture research (non-blocking)

- [#101](https://github.com/payalchandak/EveryQuery/issues/101) / [#102](https://github.com/payalchandak/EveryQuery/issues/102) — RoPE for time-deltas
- [#103](https://github.com/payalchandak/EveryQuery/issues/103) — Evaluate alternatives to ModernBERT as the encoder backbone

## Acknowledgements

EveryQuery sits on top of [MEDS](https://github.com/Medical-Event-Data-Standard),
[`meds-torch-data`](https://github.com/mmcdermott/meds-torch-data),
[`MEDS-transforms`](https://github.com/mmcdermott/MEDS_transforms), and
[`MEDS_EIC_AR`](https://github.com/mmcdermott/MEDS_EIC_AR) (architectural reference). It
uses [Hydra](https://hydra.cc) for configuration, [PyTorch Lightning](https://lightning.ai)
for training, and [W&B](https://wandb.ai) for telemetry.

## License

MIT — see [LICENSE](LICENSE).

[#122]: https://github.com/payalchandak/EveryQuery/issues/122
