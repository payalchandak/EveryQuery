# `train/`

Training stage of the EveryQuery pipeline. Home of the `EQ_train` console script.

## What lives here

- **`train.py`** — main training logic. Registered as the `EQ_train` entry point
    (`[project.scripts]` → `every_query.train.train:main`) and runnable directly as
    `python -m every_query.train.train`. Hydra-based: all knobs overridable on the CLI.
- **`configs/`** — shipped Hydra configs for the training stage.
    - `config.yaml` — default production config (ModernBERT encoder, AdamW, wandb
        logger, early-stopping on tuning/loss). Training has no query-codes knob:
        the set of queried codes is determined by whatever `EQ_generate_training_tasks`
        wrote into the task-labels parquet (`datamodule.config.task_labels_dir`, a
        required Hydra arg — typically `=$TRAINING_TASKS_DIR`). Code filtering happens
        upstream in preprocessing.
    - `fast_config.yaml` — speed-tuned override bundle that inherits `config.yaml` and
        shrinks `max_seq_len`, bumps batch size, etc. Targeted at tokenization-sweep
        runs that fit in ~5 minutes on one L40S.
    - `_demo_train.yaml` — minimal CPU-only config used by
        `tests/test_train_cli.py` to exercise the full `main` subprocess path.

## Pipeline position

```
preprocessing/     →  generate_tasks/                 →  train/      →  predict/    →  evaluate/
EQ_process_data       EQ_generate_training_tasks         EQ_train       EQ_predict      EQ_evaluate
```

`train/` consumes two artifacts from upstream:

1. The tensorized MEDS cohort at `$TENSORIZED_COHORT_DIR`, produced by
    [`preprocessing/`](../preprocessing/).
2. Long-format task-label parquets, produced by
    [`generate_tasks/`](../generate_tasks/) (no intermediate "collation" step — the
    sampler writes the dataloader's input format directly). `EQ_generate_training_tasks`
    writes these to `$TRAINING_TASKS_DIR`; pass that path as the required
    `datamodule.config.task_labels_dir` Hydra arg (typically `=$TRAINING_TASKS_DIR`).

`train/` produces a run directory at `<output_dir>/<YYYY-MM-DD>/<HH-MM-SS>/` (you supply the
required `output_dir=` base; Hydra appends the timestamp via its native `run.dir`/`sweep.dir`)
containing `best_model.ckpt`, `config.yaml` (used config), `resolved_config.yaml` (used config
with all interpolations resolved — consumed by downstream loaders), and a `checkpoints/` dir with
epoch-indexed checkpoints. Sweeps (`EQ_train -m ...`) land one `override_dirname` subdir per job
under the same timestamped folder.

## Resume behavior

`do_resume=True` reuses an existing run dir's checkpoints and `config.yaml`. Because each launch
gets a fresh timestamp, resume the *specific* run by pinning its path:
`hydra.run.dir=<output_dir>/<YYYY-MM-DD>/<HH-MM-SS> do_resume=True`. See #91 for the
structural-drift check between the resumed-from config and the new-invocation config.
