# EveryQuery

[![tests](https://github.com/payalchandak/EveryQuery/actions/workflows/tests.yaml/badge.svg?branch=dev)](https://github.com/payalchandak/EveryQuery/actions/workflows/tests.yaml)
[![codecov](https://codecov.io/gh/payalchandak/EveryQuery/branch/dev/graph/badge.svg)](https://codecov.io/gh/payalchandak/EveryQuery)

A framework for training and evaluating foundation models over structured EHR
data. Built on the [MEDS](https://github.com/Medical-Event-Data-Standard)
ecosystem — [`meds-torch-data`](https://github.com/mmcdermott/meds-torch-data)
for tensorization, [`MEDS-transforms`](https://github.com/mmcdermott/MEDS_transforms)
for preprocessing, PyTorch Lightning for training.

Given a tensorized MEDS cohort, EveryQuery trains a ModernBERT-style encoder to
answer "query" prediction tasks of the form: *given a subject's history up to
time `t`, will code `c` occur within `d` days?* The same trained model is
then evaluated against arbitrary (code, duration) combinations.

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
pip install EveryQuery # not yet on PyPI — installable from git for now:
pip install "git+https://github.com/payalchandak/EveryQuery.git@main"
```

> [!NOTE]
> A substantial refactor is in progress — see [#54](https://github.com/payalchandak/EveryQuery/issues/54).
> Entry-point names and the eval pipeline are likely to change. This README captures the current state on `dev`.

## Console scripts

`pip install` exposes the CLIs below, all Hydra-configurable. Run any with
`--help` or `--cfg job` to inspect the resolved config.

| Script              | Stage               | Purpose                                                                     |
| ------------------- | ------------------- | --------------------------------------------------------------------------- |
| `EQ_generate_tasks` | pre-training labels | Sample `N` tasks × `M` contexts and label them (sampling-first, fast)       |
| `EQ_train`          | training            | Train the ModernBERT encoder on labeled tasks                               |
| `EQ_gen_eval_index` | eval setup          | Sample held-out prediction times into a deterministic index                 |
| `EQ_gen_eval_tasks` | eval setup          | Slice per-duration task matrices by (code, duration) using the index        |
| `EQ_evaluate`       | eval                | Run a trained checkpoint against the sliced eval tasks, write per-code AUCs |
| `EQ_select_model`   | analysis            | Rank models by pairwise win rate over (code, duration) pairs                |

## Pipeline

```
           MEDS cohort  ──►  meds-torch-data tensorize  ──►  tensorized cohort ($FINAL_DATA_DIR)
                                                                     │
                                                                     ▼
pre-training:                                                  EQ_generate_tasks
                                                                     │ labeled task parquets
                                                                     ▼
                                                                EQ_train  ──►  best_model.ckpt
                                                                                       │
evaluation:                      EQ_gen_eval_index  ──►  EQ_gen_eval_tasks              │
                                                                │                       │
                                                                ▼                       ▼
                                                                             EQ_evaluate
                                                                                       │ per-code AUCs
                                                                                       ▼
                                                                                 EQ_select_model
```

> The eval pipeline is being consolidated — `EQ_gen_eval_tasks` currently
> expects a per-duration wide task matrix whose dedicated producer was
> removed in [#76](https://github.com/payalchandak/EveryQuery/pull/76).
> Phase 2 of the [#54](https://github.com/payalchandak/EveryQuery/issues/54)
> refactor collapses these four eval CLIs into two.

### 1. Preprocess to tensorized MEDS

EveryQuery itself does not ship a preprocessing CLI yet
([#55](https://github.com/payalchandak/EveryQuery/issues/55)). Today the
expected inputs are produced by
[`meds-torch-data`](https://github.com/mmcdermott/meds-torch-data)'s
`MTD_preprocess` or the preprocessing stages from
[MEDS-transforms](https://github.com/mmcdermott/MEDS_transforms).

Outputs expected downstream:

- `$INTERMEDIATE/data/{split}/*.parquet` — sharded event tables
- `$PROCESSED/metadata/codes.parquet` — the query-code universe
- `$FINAL_DATA_DIR/` — fully tensorized cohort (what `meds-torch-data` writes)

### 2. Generate pre-training task labels

```bash
EQ_generate_tasks \
	split=train \
	input_shard=0 \
	task_shard=0 \
	n_tasks=1024 \
	contexts_per_task=1
```

Produces labeled task parquets under `$TASK_DIR/{split}/*.parquet`. Sweep with
`python -m every_query.sample_tasks -m input_shard=0,1,2,… task_shard=range(0,K)`
to cover the task × patient product.

### 3. Train

```bash
EQ_train \
	output_dir="$OUTPUT_DIR/outputs/\${run_id:}" \
	datamodule.config.task_labels_dir="$TASK_DIR" \
	datamodule.config.tensorized_cohort_dir="$FINAL_DATA_DIR"
```

`EQ_train` reads the long-format labels written by `EQ_generate_tasks`
directly — the inline collation step that lived in `train.py` was removed
in [#76](https://github.com/payalchandak/EveryQuery/pull/76).

Run dir ends up at `$OUTPUT_DIR/outputs/YYYY-MM-DD/HH-MM-SS/` with
`best_model.ckpt`, `config.yaml`, `resolved_config.yaml`, and `checkpoints/`.

### 4. Build the evaluation index and slice task matrices

```bash
EQ_gen_eval_index # sample prediction times into a deterministic index
EQ_gen_eval_tasks # slice per-duration task matrices by (code, duration)
```

`EQ_gen_eval_tasks` expects per-duration wide task parquets at
`$TASK_DIR/{duration}/{split}/*.parquet`. Their former dedicated
producer (`EQ_generate_tasks_exhaustive`) was removed in
[#76](https://github.com/payalchandak/EveryQuery/pull/76); the Phase 2
refactor in [#54](https://github.com/payalchandak/EveryQuery/issues/54)
replaces this whole pipeline with a `FlexibleSchema`-driven `EQ_predict`
that takes a single task-specifying parquet.

### 5. Evaluate and rank

```bash
EQ_evaluate model_run_dirs='["'"$OUTPUT_DIR"'/outputs/YYYY-MM-DD/HH-MM-SS"]'
EQ_select_model model_run_dirs='["..."]' split=tuning
```

## Configuration

### Environment variables

`ensure_env()` (in `src/every_query/_env.py`) requires these be set before
`EQ_train` and the eval CLIs:

| Var              | Purpose                                                       |
| ---------------- | ------------------------------------------------------------- |
| `PROJECT_DIR`    | Repo root (for relative output paths in a few configs)        |
| `OUTPUT_DIR`     | Where training run dirs land                                  |
| `TASK_DIR`       | Where task parquets read / write                              |
| `PROCESSED`      | MEDS cohort `processed/` dir (holds `metadata/codes.parquet`) |
| `INTERMEDIATE`   | MEDS cohort `intermediate/` dir (event shards)                |
| `FINAL_DATA_DIR` | Tensorized cohort (output of `meds-torch-data`)               |
| `WANDB_ENTITY`   | W&B entity for training telemetry                             |

`.env.example` is the reference — copy to `.env` and edit. Both Python
(via `python-dotenv`) and the SLURM wrappers under `scripts/` source it.

### Hydra

Every CLI is a `@hydra.main` entry point; all config knobs are overridable
on the command line with `key=value` or `+new_key=value`. The config directory
is resolved via `importlib.resources.files("every_query")`, so package-shipped
YAMLs work identically whether you run from a source checkout or a
`pip install`ed wheel.

### Known gotcha: code-group YAMLs

`config.yaml`, `eval_config.yaml`, and `gen_tasks_config.yaml` all pull a
default `{train,eval}_codes/<hash>.yaml` that is (a) generated out-of-band
and (b) explicitly `.gitignore`d — so a fresh clone can't compose them.

Workarounds until [#64](https://github.com/payalchandak/EveryQuery/issues/64)
lands:

- Pass `--config-dir=/path/to/your/codes_dir code_group_name=...`, or
- Generate them locally via `src/every_query/sample_codes/sample_train_codes.py`
    (note: currently has a hard-coded MIMIC path — edit first).

The smoke-test fixture in `tests/test_cli_smoke.py` shows the minimal shape
of each file.

## Development

```bash
uv sync --group dev
uv run pytest                         # full suite (~90 s)
uv run pytest tests/test_cli_smoke.py # CLI smoke tests only
uv run pre-commit run --all-files     # lint, format, codespell
```

CI runs the full `pytest` plus `ruff check` and `ruff format --check` on
every PR; coverage is uploaded to Codecov.

### Roadmap / open issues

Overall refactor umbrella: [#54](https://github.com/payalchandak/EveryQuery/issues/54) — the target architecture rewrites this whole pipeline as `EQ_process_data → EQ_prepare_tasks → EQ_pretrain → EQ_predict → EQ_evaluate`, with a shared `FlexibleSchema` as the cross-stage contract.

Live child issues:

- [#55](https://github.com/payalchandak/EveryQuery/issues/55) — `EQ_process_data` preprocessing endpoint (PR [#74](https://github.com/payalchandak/EveryQuery/pull/74) open)
- [#59](https://github.com/payalchandak/EveryQuery/issues/59) — doc overhaul (this PR is the interim snapshot; final rewrite after the refactor lands)
- [#62](https://github.com/payalchandak/EveryQuery/issues/62) — promote `aces_to_eq` / `process_composite` to entry points
- [#64](https://github.com/payalchandak/EveryQuery/issues/64) — drop gitignored `{train,eval}_codes` defaults
- [#66](https://github.com/payalchandak/EveryQuery/issues/66) — unbreak `eval_config.yaml`'s hardcoded run dirs
- [#68](https://github.com/payalchandak/EveryQuery/issues/68) — wheel-install CI + staged CLI functional tests
- [#31](https://github.com/payalchandak/EveryQuery/issues/31) — `train.py` `do_overwrite=True` wipes config before re-saving
- [#32](https://github.com/payalchandak/EveryQuery/issues/32) — `eval.py` `ood_codes is None` TypeError

## Acknowledgements

EveryQuery sits on top of [MEDS](https://github.com/Medical-Event-Data-Standard),
[`meds-torch-data`](https://github.com/mmcdermott/meds-torch-data),
[`MEDS-transforms`](https://github.com/mmcdermott/MEDS_transforms), and
[`MEDS_EIC_AR`](https://github.com/mmcdermott/MEDS_EIC_AR) (architectural
reference). It uses [Hydra](https://hydra.cc) for configuration,
[PyTorch Lightning](https://lightning.ai) for training, and
[W&B](https://wandb.ai) for telemetry.

## License

MIT — see [LICENSE](LICENSE).
