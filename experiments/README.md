# `experiments/`

Home for code that reproduces the specific experiments we run for the EveryQuery paper — things external users would never do in normal model usage, but we need to produce figures, ablations, and generalization claims.

> [!IMPORTANT]
> This directory is a **staging area** for the experiment/repo split tracked in [#186](https://github.com/payalchandak/EveryQuery/issues/186). The code here is **not** part of the `every_query` package — it lives outside `src/`, so it is not shipped on PyPI and is no longer importable as `every_query.paper_experiments.*`. It will be extracted into a standalone `EveryQueryExperiments` repo that depends on `EveryQuery` as an installed library (dependency direction is strictly one-way: experiments → core). The separation is about *purpose*, so the normal-usage pipeline stays obvious.

## What belongs here

Concrete examples of paper-experiments code:

- **ID-vs-OOD code-list generation.** Sampling the code universe into "in-distribution" and "out-of-distribution" splits to assess the model's generalization to never-queried codes. External users would simply train on all codes of interest — holding codes out is a research claim, not a deployment pattern.
- **Duration-ablation sweeps.** Evaluating the same model against the same tasks at many horizon lengths to characterize how probability calibration changes with `duration_days`. Not something a downstream user would run for a single deployment decision.
- **Cross-model leaderboards.** Pairwise win-rate ranking across multiple trained variants to produce comparison tables for the paper.
- **Figure/plot code.** Matplotlib / seaborn scripts and notebooks that turn the outputs of the above into paper-ready PNGs/PDFs.

## What does **not** belong here

Anything on the normal-usage path: `preprocessing/`, `generate_tasks/`, `train/`, `predict/`, `evaluate/`. Those submodules are what any external user touches when running EveryQuery on their own cohort, and they should stay obviously separate from paper-only tooling.

Cluster-specific operator tooling (SLURM submit wrappers, GCS upload scripts) also doesn't live here — that category already left the repo in [#77](https://github.com/payalchandak/EveryQuery/pull/77).

## What lives here today

- **`sample_codes/`** — scripts that sample query codes from a MEDS dataset into `train_codes`
    and `eval_codes` YAMLs, with an ID/OOD split axis. The pending dataset-agnostic rewrite (removing
    hardcoded MIMIC paths, taking `metadata_dir` as a CLI arg) is tracked as
    [#85](https://github.com/payalchandak/EveryQuery/issues/85).
- **`analysis/results.ipynb`** — the preprint analysis notebook: loads multi-duration EQ runs, compares
    against the EIC baseline, and emits the paper figures.

## Dependencies

These scripts/notebooks depend on the core `EveryQuery` library plus heavyweight plotting deps
(matplotlib, seaborn, etc.). Once this directory is extracted into `EveryQueryExperiments`, those
experiment-only deps live in that repo's dependency list rather than polluting the lean core install.

## Related

- Repo split proposal: [#186](https://github.com/payalchandak/EveryQuery/issues/186)
- Parent refactor umbrella: [#54](https://github.com/payalchandak/EveryQuery/issues/54)
- `sample_codes/` dataset-agnostic rewrite: [#85](https://github.com/payalchandak/EveryQuery/issues/85)
- Plan context for this split: [refactor plan v3](https://github.com/payalchandak/EveryQuery/issues/54#issuecomment-4271122440)
