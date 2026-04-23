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
        wrote into the task-labels parquet at `$TASK_DIR`. Code filtering happens
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

1. The tensorized MEDS cohort at `$FINAL_DATA_DIR`, produced by
    [`preprocessing/`](../preprocessing/).
2. Long-format task-label parquets at `$TASK_DIR`, produced by
    [`generate_tasks/`](../generate_tasks/) (no intermediate "collation" step — the
    sampler writes the dataloader's input format directly).

`train/` produces a run directory at
`$OUTPUT_DIR/outputs/<YYYY-MM-DD>/<HH-MM-SS>/` containing `best_model.ckpt`,
`config.yaml` (used config), `resolved_config.yaml` (used config with all
interpolations resolved — consumed by downstream loaders), and a
`checkpoints/` dir with epoch-indexed checkpoints.

## Resume behavior

`do_resume=True` in the config reuses the `output_dir`'s existing checkpoints and
`config.yaml`. See #91 for the work to add a structural-drift check between the
resumed-from config and the new-invocation config.
