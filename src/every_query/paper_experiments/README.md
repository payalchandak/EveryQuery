# `paper_experiments/`

Home for code that reproduces the specific experiments we run for the EveryQuery paper — things external users would never do in normal model usage, but we need to produce figures, ablations, and generalization claims.

> [!IMPORTANT]
> This is a **research-intent** split, not a packaging one. Code here is still shipped on PyPI, still tested, still pre-commit-linted, still importable as `every_query.paper_experiments.*`. The separation is about *purpose*, so the normal-usage pipeline stays obvious.

## What belongs here

Concrete examples of paper-experiments code:

- **ID-vs-OOD code-list generation.** Sampling the code universe into "in-distribution" and "out-of-distribution" splits to assess the model's generalization to never-queried codes. External users would simply train on all codes of interest — holding codes out is a research claim, not a deployment pattern.
- **Duration-ablation sweeps.** Evaluating the same model against the same tasks at many horizon lengths to characterize how probability calibration changes with `duration_days`. Not something a downstream user would run for a single deployment decision.
- **Cross-model leaderboards.** Pairwise win-rate ranking across multiple trained variants to produce comparison tables for the paper.
- **Figure/plot code.** Matplotlib / seaborn scripts that turn the outputs of the above into paper-ready PNGs/PDFs.

## What does **not** belong here

Anything on the normal-usage path: `preprocessing/`, `prepare_tasks/`, `pretrain/`, `predict/`, `evaluate/`. Those submodules are what any external user touches when running EveryQuery on their own cohort, and they should stay obviously separate from paper-only tooling.

Cluster-specific operator tooling (SLURM submit wrappers, GCS upload scripts) also doesn't live here — that category already left the repo in [#77](https://github.com/payalchandak/EveryQuery/pull/77).

## Optional dependencies

If a specific experiment needs heavyweight plotting deps (matplotlib, seaborn, plotly), those go under `[project.optional-dependencies].paper` in `pyproject.toml` rather than polluting the core install. Parked until we see the real dep footprint.

## Related

- Parent refactor umbrella: [#54](https://github.com/payalchandak/EveryQuery/issues/54)
- Planned `sample_codes/` move into here: [#85](https://github.com/payalchandak/EveryQuery/issues/85)
- Plan context for this split: [refactor plan v3](https://github.com/payalchandak/EveryQuery/issues/54#issuecomment-4271122440)
